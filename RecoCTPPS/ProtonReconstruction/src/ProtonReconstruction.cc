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

//------------------------------------------------------------------------------
// JAN'S VERSION
//------------------------------------------------------------------------------

/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstruction.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

int ProtonReconstruction::Init(const std::string &optics_file_beam1, const std::string &optics_file_beam2)
{
	// open files with optics
	TFile *f_in_optics_beam1 = TFile::Open(optics_file_beam1.c_str());
	if (f_in_optics_beam1 == NULL)
	{
		printf("ERROR in ProtonReconstruction::Init > Can't open file '%s'.\n", optics_file_beam1.c_str());
		return 1;
	}

	TFile *f_in_optics_beam2 = TFile::Open(optics_file_beam2.c_str());
	if (f_in_optics_beam2 == NULL)
	{
		printf("ERROR in ProtonReconstruction::Init > Can't open file '%s'.\n", optics_file_beam2.c_str());
		return 1;
	}

	// build RP id, optics object name association
	std::map<unsigned int, std::string> nameMap = {
		{2, "ip5_to_station_150_h_1_lhcb2"},
		{3, "ip5_to_station_150_h_2_lhcb2"},
		{102, "ip5_to_station_150_h_1_lhcb1"},
		{103, "ip5_to_station_150_h_2_lhcb1"},
	};

	// build optics data for each object
	for (const auto &it : nameMap)
	{
		const unsigned int &rpId = it.first;
		const std::string &ofName = it.second;

		// determine LHC sector from RP id
		LHCSector sector = unknownSector;
		if ((rpId / 100) == 0)
			sector = sector45;
		if ((rpId / 100) == 1)
			sector = sector56;

		// load optics approximation
		TFile *f_in_optics = NULL;
		if (sector == sector45)
			f_in_optics = f_in_optics_beam2;
		if (sector == sector56)
			f_in_optics = f_in_optics_beam1;

		LHCOpticsApproximator *of_orig = (LHCOpticsApproximator *) f_in_optics->Get(ofName.c_str());

		if (of_orig == NULL)
		{
			printf("ERROR in ProtonReconstruction::Init > Can't load object '%s'.\n", ofName.c_str());
			return 2;
		}

		RPOpticsData rpod;
		rpod.optics = new LHCOpticsApproximator(* of_orig);

		// build auxiliary optical functions
		double crossing_angle = 0.;
		double vtx0_y = 0.;

		if (sector == sector45)
		{
			crossing_angle = beamConditions.half_crossing_angle_45;
			vtx0_y = beamConditions.vtx0_y_45;
		}

		if (sector == sector56)
		{
			crossing_angle = beamConditions.half_crossing_angle_56;
			vtx0_y = beamConditions.vtx0_y_56;
		}

		const bool check_appertures = false;
		const bool invert_beam_coord_sytems = true;

		TGraph *g_xi_vs_x = new TGraph();
		TGraph *g_y0_vs_xi = new TGraph();
		TGraph *g_v_y_vs_xi = new TGraph();
		TGraph *g_L_y_vs_xi = new TGraph();

		for (double xi = 0.; xi <= 0.201; xi += 0.005)
		{
			// input: only xi
			double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
			double kin_out_xi[5];
    		rpod.optics->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_sytems);

			// input: xi and vtx_y
			const double vtx_y = 10E-6;	// m
			double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
			double kin_out_xi_vtx_y[5];
    		rpod.optics->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_sytems);

			// input: xi and th_y
			const double th_y = 20E-6;	// rad
			double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
			double kin_out_xi_th_y[5];
    		rpod.optics->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_sytems);

			// fill graphs
			int idx = g_xi_vs_x->GetN();
			g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
			g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);
			g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
			g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);
		}

		rpod.s_xi_vs_x = new TSpline3("", g_xi_vs_x->GetX(), g_xi_vs_x->GetY(), g_xi_vs_x->GetN());
		delete g_xi_vs_x;

		rpod.s_y0_vs_xi = new TSpline3("", g_y0_vs_xi->GetX(), g_y0_vs_xi->GetY(), g_y0_vs_xi->GetN());
		delete g_y0_vs_xi;

		rpod.s_v_y_vs_xi = new TSpline3("", g_v_y_vs_xi->GetX(), g_v_y_vs_xi->GetY(), g_v_y_vs_xi->GetN());
		delete g_v_y_vs_xi;

		rpod.s_L_y_vs_xi = new TSpline3("", g_L_y_vs_xi->GetX(), g_L_y_vs_xi->GetY(), g_L_y_vs_xi->GetN());
		delete g_L_y_vs_xi;

		// insert optics data
		m_rp_optics[rpId] = rpod;
	}

	// initialise fitter
	chiSquareCalculator = new ChiSquareCalculator();

	fitter = new ROOT::Fit::Fitter();
	double pStart[] = {0, 0, 0, 0};
	fitter->SetFCN(4, *chiSquareCalculator, pStart, 0, true);

	// clean up
	delete f_in_optics_beam1;
	delete f_in_optics_beam2;

	return 0;
}

//----------------------------------------------------------------------------------------------------

double ProtonReconstruction::ChiSquareCalculator::operator() (const double *parameters) const
{
	// extract proton parameters
	const double &xi = parameters[0];
	const double &th_x = parameters[1];
	const double &th_y = parameters[2];
	const double vtx_x = 0;
	const double &vtx_y = parameters[3];

	// calculate chi^2 by looping over hits
	double S2 = 0.;

	for (auto &it : *tracks)
	{
		const unsigned int &rpId = it.getRPId();

		// determine LHC sector from RP id
		LHCSector sector = unknownSector;
		if ((rpId / 100) == 0)
			sector = sector45;
		if ((rpId / 100) == 1)
			sector = sector56;

		double crossing_angle = 0.;
		double vtx0_y = 0.;

		if (sector == sector45)
		{
			crossing_angle = beamConditions.half_crossing_angle_45;
			vtx0_y = beamConditions.vtx0_y_45;
		}

		if (sector == sector56)
		{
			crossing_angle = beamConditions.half_crossing_angle_56;
			vtx0_y = beamConditions.vtx0_y_56;
		}

		// transport proton to the RP
		auto oit = m_rp_optics->find(rpId);

		const bool check_appertures = false;
		const bool invert_beam_coord_sytems = true;

		double kin_in[5] = {
			vtx_x,
			(th_x + crossing_angle) * (1. - xi),
			vtx0_y + vtx_y,
			th_y * (1. - xi),
			-xi
		};
		double kin_out[5];
    	oit->second.optics->Transport(kin_in, kin_out, check_appertures, invert_beam_coord_sytems);

		const double &x = kin_out[0];
		const double &y = kin_out[2];

		// calculate chi^2 constributions
		const double x_diff_norm = (x - it.getX()) / it.getXUnc();
		const double y_diff_norm = (y - it.getY()) / it.getYUnc();

		// increase chi^2
		S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
	}

	//printf("xi=%.3E, th_x=%.3E, th_y=%.3E, vtx_y=%.3E | S2 = %.3E\n", xi, th_x, th_y, vtx_y, S2);

	return S2;
}

//----------------------------------------------------------------------------------------------------

ProtonData ProtonReconstruction::Reconstruct(LHCSector /* sector */, const vector<CTPPSLocalTrackLite> &tracks) const
{
	// by default invalid proton
	ProtonData pd;
	pd.valid = false;

	// need at least two tracks
	if (tracks.size() < 2)
		return pd;

	// check optics is available for all tracks
	for (const auto &it : tracks)
	{
		auto oit = m_rp_optics.find(it.getRPId());
		if (oit == m_rp_optics.end())
		{
			printf("ERROR in ProtonReconstruction::Reconstruct > optics not available for RP %u.\n", it.getRPId());
			return pd;
		}
	}

	// rough estimate of xi
	double S_xi0 = 0., S_1 = 0.;
	for (const auto &it : tracks)
	{
		auto oit = m_rp_optics.find(it.getRPId());
		double xi = oit->second.s_xi_vs_x->Eval(it.getX());

		S_1 += 1.;
		S_xi0 += xi;
	}

	const double xi_0 = S_xi0 / S_1;

	//printf("    xi_0 = %.3f\n", xi_0);

	// rough estimate of th_y and vtx_y
	double y[2], v_y[2], L_y[2];
	unsigned int y_idx = 0;
	for (const auto &it : tracks)
	{
		if (y_idx >= 2)
			continue;

		auto oit = m_rp_optics.find(it.getRPId());

		y[y_idx] = it.getY() - oit->second.s_y0_vs_xi->Eval(xi_0);
		v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval(xi_0);
		L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval(xi_0);

		y_idx++;
	}

	const double det = v_y[0] * L_y[1] - L_y[0] * v_y[1];
	const double vtx_y_0 = (L_y[1] * y[0] - L_y[0] * y[1]) / det;
	const double th_y_0 = (v_y[0] * y[1] - v_y[1] * y[0]) / det;

	//printf("    vtx_y_0 = %.3f mm\n", vtx_y_0 * 1E3);
	//printf("    th_y_0 = %.1f urad\n", th_y_0 * 1E6);

	// minimisation
	fitter->Config().ParSettings(0).Set("xi", xi_0, 0.005);
	fitter->Config().ParSettings(1).Set("th_x", 0., 20E-6);
	fitter->Config().ParSettings(2).Set("th_y", th_y_0, 1E-6);
	fitter->Config().ParSettings(3).Set("vtx_y", vtx_y_0, 1E-6);

	// TODO: this breaks the const-ness ??
	chiSquareCalculator->tracks = &tracks;
	chiSquareCalculator->m_rp_optics = &m_rp_optics;

	fitter->FitFCN();

	// extract proton parameters
	const ROOT::Fit::FitResult &result = fitter->Result();
	const double *fitParameters = result.GetParams();

	pd.valid = true;
	pd.vtx_x = 0.;
	pd.vtx_y = fitParameters[3];
	pd.th_x = fitParameters[1];
	pd.th_y = fitParameters[2];
	pd.xi = fitParameters[0];

	return pd;
}

