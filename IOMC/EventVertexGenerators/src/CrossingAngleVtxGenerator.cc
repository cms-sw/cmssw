#include "IOMC/EventVertexGenerators/interface/CrossingAngleVtxGenerator.h"

CrossingAngleVtxGenerator::CrossingAngleVtxGenerator( const edm::ParameterSet& iConfig ) :
  sourceToken_( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "src" ) ) ),
  scatteringAngle_          ( iConfig.getParameter<double>( "scatteringAngle" ) ),
  vertexSize_               ( iConfig.getParameter<double>( "vertexSize" ) ),
  beamDivergence_           ( iConfig.getParameter<double>( "beamDivergence" ) ),
  halfCrossingAngleSector45_( iConfig.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_( iConfig.getParameter<double>( "halfCrossingAngleSector56" ) ),
  simulateVertexX_          ( iConfig.getParameter<bool>( "simulateVertexX" ) ),
  simulateVertexY_          ( iConfig.getParameter<bool>( "simulateVertexY" ) ),
  simulateScatteringAngleX_ ( iConfig.getParameter<bool>( "simulateScatteringAngleX" ) ),
  simulateScatteringAngleY_ ( iConfig.getParameter<bool>( "simulateScatteringAngleY" ) ),
  simulateBeamDivergence_   ( iConfig.getParameter<bool>( "simulateBeamDivergence" ) ),
  rnd_( 0 )
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( !rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "The CrossingAngleVtxGenerator requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file. \n" 
         "You must add the service\n"
         "in the configuration file or remove the modules that require it.";
  }
  produces<edm::HepMCProduct>();
}

CrossingAngleVtxGenerator::~CrossingAngleVtxGenerator() 
{}

void
CrossingAngleVtxGenerator::produce( edm::Event& iEvent, const edm::EventSetup& )
{
  edm::Service<edm::RandomNumberGenerator> rng;
  rnd_ = &( rng->getEngine( iEvent.streamID() ) );

  edm::Handle<edm::HepMCProduct> HepUnsmearedMCEvt ;
  iEvent.getByToken( sourceToken_, HepUnsmearedMCEvt );

  // Copy the HepMC::GenEvent
  HepMC::GenEvent* genevt = new HepMC::GenEvent( *HepUnsmearedMCEvt->GetEvent() );
  std::unique_ptr<edm::HepMCProduct> HepMCEvt( new edm::HepMCProduct( genevt ) );
  HepMCEvt->applyVtxGen( vertexPosition().get() );
  for ( HepMC::GenEvent::particle_iterator part=genevt->particles_begin(); part!=genevt->particles_end(); ++part ) {
    rotateParticle( *part );
  }

  iEvent.put( std::move( HepMCEvt ) );
}

std::shared_ptr<HepMC::FourVector>
CrossingAngleVtxGenerator::vertexPosition() const
{
  double vtx_x = 0.0, vtx_y = 0.0; // express in metres
  if ( simulateVertexX_ ) vtx_x += CLHEP::RandGauss::shoot( rnd_ ) * vertexSize_;
  if ( simulateVertexY_ ) vtx_y += CLHEP::RandGauss::shoot( rnd_ ) * vertexSize_;
  return std::make_shared<HepMC::FourVector>( vtx_x, vtx_y, 0.0, 0.0 );
}

void
CrossingAngleVtxGenerator::rotateParticle( HepMC::GenParticle* part ) const
{
  const HepMC::FourVector mom = part->momentum();

  // convert physics kinematics to the LHC reference frame
  double th_x = atan2( mom.x(), mom.z() ), th_y = atan2( mom.y(), mom.z() );
  if ( mom.z()<0.0 ) { th_x = M_PI-th_x; th_y = M_PI-th_y; }

  // generate scattering angles
  if ( simulateScatteringAngleX_ ) th_x += CLHEP::RandGauss::shoot( rnd_ ) * scatteringAngle_;
  if ( simulateScatteringAngleY_ ) th_y += CLHEP::RandGauss::shoot( rnd_ ) * scatteringAngle_;

  // generate beam divergence
  if ( simulateBeamDivergence_ ) {
    th_x += CLHEP::RandGauss::shoot( rnd_ ) * beamDivergence_;
    th_y += CLHEP::RandGauss::shoot( rnd_ ) * beamDivergence_;
  }

  //FIXME LHC or CMS convention?
  double half_cr_angle = 0.0;
  if ( mom.z()>0.0 ) half_cr_angle = halfCrossingAngleSector45_;
  if ( mom.z()<0.0 ) half_cr_angle = halfCrossingAngleSector56_;
  th_x += half_cr_angle;

  const double pz = sqrt( mom.perp2()/( 1.0+tan( th_x )*tan( th_x )+tan( th_y )*tan( th_y ) ) );
  HepMC::FourVector mom_smeared( pz*tan( th_x ), pz*tan( th_y ), pz, mom.m() );
  part->set_momentum( mom_smeared );
  
}
