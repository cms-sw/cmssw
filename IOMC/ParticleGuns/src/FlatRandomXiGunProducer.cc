#include "IOMC/ParticleGuns/interface/FlatRandomXiGunProducer.h"

namespace edm
{
  FlatRandomXiGunProducer::FlatRandomXiGunProducer( const edm::ParameterSet& iConfig ) :
    partGunParams_( iConfig.getParameter<edm::ParameterSet>( "PGunParameters" ) ),
    partIds_       ( partGunParams_.getParameter< std::vector<int> >( "PartID" ) ),
    sqrtS_         ( partGunParams_.getParameter<double>( "SqrtS" ) ),
    beamConditions_( partGunParams_.getParameter<edm::ParameterSet>( "BeamConditions" ) ),
    minXi_         ( partGunParams_.getParameter<double>( "MinXi" ) ),
    maxXi_         ( partGunParams_.getParameter<double>( "MaxXi" ) ),
    minPhi_        ( partGunParams_.getUntrackedParameter<double>( "MinPhi", -M_PI ) ),
    maxPhi_        ( partGunParams_.getUntrackedParameter<double>( "MaxPhi", +M_PI ) ),
    thetaPhys_     ( partGunParams_.getParameter<double>( "ScatteringAngle" ) ),
    vertexSize_    ( beamConditions_.getParameter<double>( "vertexSize" ) ),
    beamDivergence_( beamConditions_.getParameter<double>( "beamDivergence" ) ),
    // switches
    simulateVertexX_         ( partGunParams_.getParameter<bool>( "SimulateVertexX" ) ),
    simulateVertexY_         ( partGunParams_.getParameter<bool>( "SimulateVertexY" ) ),
    simulateScatteringAngleX_( partGunParams_.getParameter<bool>( "SimulateScatteringAngleX" ) ),
    simulateScatteringAngleY_( partGunParams_.getParameter<bool>( "SimulateScatteringAngleY" ) ),
    simulateBeamDivergence_  ( partGunParams_.getParameter<bool>( "SimulateBeamDivergence" ) )
  {
    produces<edm::HepMCProduct>( "unsmeared" );
    produces<GenEventInfoProduct>();
  }

  FlatRandomXiGunProducer::~FlatRandomXiGunProducer()
  {}

  // ------------ method called to produce the data  ------------
  void
  FlatRandomXiGunProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    edm::Service<edm::RandomNumberGenerator> rng;
    if ( !rng.isAvailable() ) {
      throw cms::Exception("Configuration")
        << "The module inheriting from FlatRandomXiGunProducer requires the\n"
           "RandomNumberGeneratorService, which appears to be absent. Please\n"
           "add that service to your configuration or remove the modules that"
           "require it.";
    }

    CLHEP::HepRandomEngine* rnd = &( rng->getEngine( iEvent.streamID() ) );

    std::unique_ptr<edm::HepMCProduct> pOut( new edm::HepMCProduct() );

    // generate event
    auto pEvt = new HepMC::GenEvent(); // cleanup in HepMCProduct

    // generate vertex
    auto pVertex = new HepMC::GenVertex(); // cleanup in HepMCProduct
    double vx = 0., vy = 0.;
    if ( simulateVertexX_ ) vx = CLHEP::RandGauss::shoot( rnd ) * vertexSize_;
    if ( simulateVertexY_ ) vy = CLHEP::RandGauss::shoot( rnd ) * vertexSize_;
    pVertex->set_position( HepMC::FourVector( vx, vy, 0., 0. ) );

    unsigned short barcode = 0;
    for ( const auto& part : partIds_ ) {
      auto part_data = pdgTable_->particle( HepPDT::ParticleID( abs( part ) ) );
      const double mass = part_data->mass().value();

      auto p = new HepMC::GenParticle( shoot( rnd, mass ), part, 1 ); // cleanup in HepMCProduct
      p->suggest_barcode( barcode );
      pVertex->add_particle_out( p );

      barcode++;
    }

    pEvt->add_vertex( pVertex );
    pEvt->set_event_number( iEvent.id().event() );
    pOut->addHepMCData( pEvt );

    iEvent.put( std::move( pOut ), "unsmeared" );

    std::unique_ptr<GenEventInfoProduct> pGenEventInfo( new GenEventInfoProduct( pEvt ) );
    iEvent.put( std::move( pGenEventInfo ) );
  }

  //----------------------------------------------------------------------------------------------------

  HepMC::FourVector
  FlatRandomXiGunProducer::shoot( CLHEP::HepRandomEngine* rnd, double mass )
  {
    // generate xi
    const double xi = minXi_ + CLHEP::RandFlat::shoot( rnd ) * ( maxXi_-minXi_ );
    // generate phi
    const double phi = minPhi_ + CLHEP::RandFlat::shoot( rnd ) * ( maxPhi_-minPhi_ );
  
    // generate scattering angles (physics)
    double th_x_phys = 0., th_y_phys = 0.;

    if ( simulateScatteringAngleX_ ) th_x_phys += CLHEP::RandGauss::shoot( rnd ) * thetaPhys_;
    if ( simulateScatteringAngleY_ ) th_y_phys += CLHEP::RandGauss::shoot( rnd ) * thetaPhys_;

    // generate beam divergence, calculate complete angle
    double th_x = th_x_phys, th_y = th_y_phys;

    if ( simulateBeamDivergence_ ) {
      th_x += CLHEP::RandGauss::shoot( rnd ) * beamDivergence_;
      th_y += CLHEP::RandGauss::shoot( rnd ) * beamDivergence_;
    }

    const double e_part = sqrtS_/2.*( 1.-xi ); //FIXME
    const double p = sqrt( e_part*e_part-mass*mass );

    return HepMC::FourVector( p*cos( phi )*sin( th_x ), p*cos( phi )*sin( th_y ), p*cos( th_x )*cos( th_y ), e_part ); //FIXME
  }

  // ------------ method called once each stream before processing any runs, lumis or events  ------------
  void
  FlatRandomXiGunProducer::beginRun( const edm::Run&, const edm::EventSetup& iSetup )
  {
    iSetup.getData( pdgTable_ );
  }

  // ------------ method called once each stream after processing all runs, lumis and events  ------------
  void
  FlatRandomXiGunProducer::endRun( const edm::Run&, const edm::EventSetup& )
  {}

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void
  FlatRandomXiGunProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
  {
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault( desc );
  }
}
