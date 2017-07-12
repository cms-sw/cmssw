/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
 *   Leszek Grzanka
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "SimRomanPot/CTPPSOpticsParameterisation/interface/ProtonReconstructionAlgorithm.h"

ProtonReconstructionAlgorithm::ProtonReconstructionAlgorithm( const edm::ParameterSet& beam_conditions, std::unordered_map<unsigned int, std::string> objects, const std::string& optics_file, bool check_aper, bool invert_coord ) :
  beamConditions_( beam_conditions ),
  halfCrossingAngleSector45_( beamConditions_.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_( beamConditions_.getParameter<double>( "halfCrossingAngleSector56" ) ),
  yOffsetSector45_( beamConditions_.getParameter<double>( "yOffsetSector45" ) ),
  yOffsetSector56_( beamConditions_.getParameter<double>( "yOffsetSector56" ) ),
  fitter_( std::make_unique<ROOT::Fit::Fitter>() ),
  checkApertures_( check_aper ), invertBeamCoordinatesSystem_( invert_coord ),
  chiSquareCalculator_( std::make_unique<ChiSquareCalculator>( beamConditions_, checkApertures_, invertBeamCoordinatesSystem_ ) )
{
  // load optics approximation
  auto f_in_optics = std::make_unique<TFile>( optics_file.c_str() );
  if ( !f_in_optics ) throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't open file " << optics_file.c_str();

  // build optics data for each object
  for ( const auto& it : objects ) {
    const unsigned int& rpId = it.first;
    const TotemRPDetId pot_id( rpId );
    const std::string& ofName = it.second;

    TObject* of_orig = f_in_optics->Get( ofName.c_str() );
    if ( !of_orig ) throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't load object " << ofName;

    RPOpticsData rpod;
    rpod.optics = std::make_shared<LHCOpticsApproximator>( *dynamic_cast<LHCOpticsApproximator*>( of_orig ) );

    // build auxiliary optical functions
    double crossing_angle = 0.;
    double vtx0_y = 0.;

    // determine LHC sector from RP id
    if ( pot_id.arm()==0 ) {
      crossing_angle = halfCrossingAngleSector45_;
      vtx0_y = yOffsetSector45_;
    }

    if ( pot_id.arm()==1 ) {
      crossing_angle = halfCrossingAngleSector56_;
      vtx0_y = yOffsetSector56_;
    }

    auto g_xi_vs_x = std::make_unique<TGraph>(),
         g_y0_vs_xi = std::make_unique<TGraph>(),
         g_v_y_vs_xi = std::make_unique<TGraph>(),
         g_L_y_vs_xi = std::make_unique<TGraph>();

    // first start by populating the interpolation graphs
    for ( double xi=0.; xi<=0.201; xi+=0.005 ) {
      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * ( 1.-xi ), vtx0_y, 0., -xi };
      double kin_out_xi[5];
      rpod.optics->Transport( kin_in_xi, kin_out_xi, checkApertures_, invertBeamCoordinatesSystem_ );

      // input: xi and vtx_y
      const double vtx_y = beamConditions_.getParameter<double>( "vertexSize" );
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * ( 1.-xi ), vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5];
      rpod.optics->Transport( kin_in_xi_vtx_y, kin_out_xi_vtx_y, checkApertures_, invertBeamCoordinatesSystem_ );

      // input: xi and th_y
      const double th_y = beamConditions_.getParameter<double>( "beamDivergence" );
      double kin_in_xi_th_y[5] = { 0., crossing_angle * ( 1.-xi ), vtx0_y, th_y * ( 1.-xi ), -xi };
      double kin_out_xi_th_y[5];
      rpod.optics->Transport( kin_in_xi_th_y, kin_out_xi_th_y, checkApertures_, invertBeamCoordinatesSystem_ );

      // fill graphs
      int idx = g_xi_vs_x->GetN();
      g_xi_vs_x->SetPoint( idx, kin_out_xi[0], xi );
      g_y0_vs_xi->SetPoint( idx, xi, kin_out_xi[2] );
      g_v_y_vs_xi->SetPoint( idx, xi, ( kin_out_xi_vtx_y[2]-kin_out_xi[2] )/vtx_y );
      g_L_y_vs_xi->SetPoint( idx, xi, ( kin_out_xi_th_y[2]-kin_out_xi[2] )/th_y );
    }

    rpod.s_xi_vs_x = std::make_shared<TSpline3>( "", g_xi_vs_x->GetX(), g_xi_vs_x->GetY(), g_xi_vs_x->GetN() );
    rpod.s_y0_vs_xi = std::make_shared<TSpline3>( "", g_y0_vs_xi->GetX(), g_y0_vs_xi->GetY(), g_y0_vs_xi->GetN() );
    rpod.s_v_y_vs_xi = std::make_shared<TSpline3>( "", g_v_y_vs_xi->GetX(), g_v_y_vs_xi->GetY(), g_v_y_vs_xi->GetN() );
    rpod.s_L_y_vs_xi = std::make_shared<TSpline3>( "", g_L_y_vs_xi->GetX(), g_L_y_vs_xi->GetY(), g_L_y_vs_xi->GetN() );

   // insert optics data
    m_rp_optics_[pot_id] = rpod;
  }

  // initialise fitter
  double pStart[] = { 0, 0, 0, 0 };
  fitter_->SetFCN( 4, *chiSquareCalculator_, pStart, 0, true );
}

ProtonReconstructionAlgorithm::~ProtonReconstructionAlgorithm()
{}

//----------------------------------------------------------------------------------------------------

void
ProtonReconstructionAlgorithm::reconstruct( const std::vector< edm::Ptr<CTPPSLocalTrackLite> >& tracks, std::vector<CTPPSSimProtonTrack>& out ) const
{
  out.clear();

  if ( tracks.size()<2 ) return;

  // first loop to extract a rough estimate of xi (mean of all xi's)
  double sum_xi0 = 0.;
  unsigned int num_tracks = 0;

  for ( const auto& trk : tracks ) {
    // find the associated RP/parameterisation and interpolate xi from x
    auto oit = m_rp_optics_.find( TotemRPDetId( trk->getRPId() ) );
    if ( oit==m_rp_optics_.end() ) continue;
    sum_xi0 += oit->second.s_xi_vs_x->Eval( trk->getX() );
    num_tracks++;
  }
  const double xi_0 = sum_xi0 / num_tracks;

  { //FIXME replace with a loop over all pots tracks to reconstruct tracks!
    // second loop to extract a rough estimate of th_y and vtx_y
    double y[2] = { 0., 0. }, v_y[2] = { 0., 0. }, L_y[2] = { 0., 0. };
    unsigned int y_idx = 0;
    for ( const auto& trk : tracks ) {
      if ( y_idx>1 ) break;
      auto oit = m_rp_optics_.find( TotemRPDetId( trk->getRPId() ) );
      if ( oit==m_rp_optics_.end() ) continue;

      y[y_idx] = trk->getY()-oit->second.s_y0_vs_xi->Eval( xi_0 );
      v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval( xi_0 );
      L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval( xi_0 );

      y_idx++;
    }

    const double inv_det = 1./( v_y[0]*L_y[1] - L_y[0]*v_y[1] );
    const double vtx_y_0 = ( L_y[1]*y[0] - L_y[0]*y[1] ) * inv_det;
    const double th_y_0 = ( v_y[0]*y[1] - v_y[1]*y[0] ) * inv_det;

    // minimisation
    fitter_->Config().ParSettings( 0 ).Set( "xi", xi_0, 0.005 );
    fitter_->Config().ParSettings( 1 ).Set( "th_x", 0., 20.e-6 );
    fitter_->Config().ParSettings( 2 ).Set( "th_y", th_y_0, 1.e-6 );
    fitter_->Config().ParSettings( 3 ).Set( "vtx_y", vtx_y_0, 1.e-6 );

    chiSquareCalculator_->tracks = &tracks;
    chiSquareCalculator_->m_rp_optics = &m_rp_optics_;

    fitter_->FitFCN();

    // extract proton parameters
    const ROOT::Fit::FitResult& result = fitter_->Result();
    if ( !result.IsValid() ) {
      edm::LogInfo("ProtonReconstructionAlgorithm") << "Fit did not succeed! returning...";
      return;
    }
    const double* params = result.GetParams();

    edm::LogInfo("ProtonReconstructionAlgorithm")
      << "at reconstructed level: "
      << "xi=" << params[0] << ", "
      << "theta_x=" << params[1] << ", "
      << "theta_y=" << params[2] << ", "
      << "vertex_y=" << params[3] << "\n";

    out.emplace_back( Local3DPoint( 0., params[3]/*vtx_y*/, 0. ), Local3DVector( params[1]/*th_x*/, params[2]/*th_y*/, 0. ), params[0]/*xi*/ );
  }
}

//----------------------------------------------------------------------------------------------------

double
ProtonReconstructionAlgorithm::ChiSquareCalculator::operator() ( const double* parameters ) const
{
  // extract proton parameters
  const double& xi = parameters[0];
  const double& th_x = parameters[1];
  const double& th_y = parameters[2];
  const double vtx_x = 0;
  const double& vtx_y = parameters[3];

  // calculate chi^2
  double S2 = 0.;

  for ( auto& trk : *tracks ) {
    double crossing_angle = 0., vtx0_y = 0.;
    const TotemRPDetId detid( trk->getRPId() );

    // determine LHC sector from RP id
    if ( detid.arm()==0 ) {
      crossing_angle = halfCrossingAngleSector45_;
      vtx0_y = yOffsetSector45_;
    }

    if ( detid.arm()==1 ) {
      crossing_angle = halfCrossingAngleSector56_;
      vtx0_y = yOffsetSector56_;
    }

    // transport proton to the RP
    auto oit = m_rp_optics->find( detid );

    double kin_in[5] = { vtx_x,	( th_x+crossing_angle ) * ( 1.-xi ), vtx0_y+vtx_y, th_y * ( 1.-xi ), -xi };
    double kin_out[5];
    oit->second.optics->Transport( kin_in, kin_out, check_apertures, invert_beam_coord_systems );

    const double& x = kin_out[0];
    const double& y = kin_out[2];

    // calculate chi^2 contributions
    const double x_diff_norm = ( x-trk->getX() ) / trk->getXUnc();
    const double y_diff_norm = ( y-trk->getY() ) / trk->getYUnc();

    // increase chi^2
    S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
  }

  edm::LogInfo("ChiSquareCalculator")
    << "xi = " << xi << ", "
    << "th_x = " << th_x << ", "
    << "th_y = " << th_y << ", "
    << "vtx_y = " << vtx_y << " | S2 = " << S2 << "\n";

  return S2;
}
