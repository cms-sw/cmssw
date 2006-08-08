
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"

// System include files
#include <memory>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "Alignment/CSA06AlignmentAlgorithm/interface/CSA06AlignmentAlgorithm.h"

//_____________________________________________________________________________

AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) :
  theRefitterAlgo( iConfig ),
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  theSrc( iConfig.getParameter<std::string>( "src" ) ),
  stParameterSelector( iConfig.getParameter<std::string>( "parameterSelector" ) ),
  stAlignableSelector( iConfig.getParameter<std::string>( "alignableSelector" ) )
{

  edm::LogInfo("Constructor") << "Constructing producer";

  // Tell the framework what data is being produced
  setWhatProduced(this);

  setConf( iConfig );
  setSrc( iConfig.getParameter<std::string>( "src" ) );

  // get cfg for alignment algorithm
  edm::ParameterSet csa06Config 
	= iConfig.getParameter<edm::ParameterSet>( "CSA06AlignmentAlgorithm" );

  // create alignment algorithm
  theAlignmentAlgo = new CSA06AlignmentAlgorithm(csa06Config);

}


//__________________________________________________________________________________________________
// Close files, etc.
AlignmentProducer::~AlignmentProducer()
{

}


//__________________________________________________________________________________________________
// Produce tracker geometry
AlignmentProducer::ReturnType 
AlignmentProducer::produce( const TrackerDigiGeometryRecord& iRecord )
{

  edm::LogInfo("Produce") << "At producer method";

  return theTracker;
  
}


//_____________________________________________________________________________
// Initialize algorithm
void AlignmentProducer::beginOfJob( const edm::EventSetup& iSetup )
{

  edm::LogInfo("BeginJob") << "At begin job";

  // Create the tracker geometry from ideal geometry (first time only)
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  iSetup.get<IdealGeometryRecord>().get( cpv );
  iSetup.get<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),&(*gD)) );
  
  // create alignable tracker
  theAlignableTracker = new AlignableTracker( &(*gD), &(*theTracker) );

  // create alignment parameter builder
  edm::LogInfo("BeginJob") <<"Creating AlignmentParameterBuilder";
  theAlignmentParameterBuilder = new AlignmentParameterBuilder(theAlignableTracker);

  // determine which parameters are fixed/aligned (local coordinates)
  static const unsigned int npar=6;
  std::vector<bool> sel(npar,false);
    edm::LogInfo("BeginJob") <<"ParameterSelector: >" <<stParameterSelector<<"<"; 
  if (stParameterSelector.length()!=npar) {
    edm::LogInfo("BeginJob") <<"ERROR: ParameterSelector vector has wrong size!";
    exit(1);
  }
  else {
    // shifts
    if (stParameterSelector.substr(0,1)=="1") 
      sel[RigidBodyAlignmentParameters::dx]=true;
    if (stParameterSelector.substr(1,1)=="1") 
      sel[RigidBodyAlignmentParameters::dy]=true;
    if (stParameterSelector.substr(2,1)=="1") 
      sel[RigidBodyAlignmentParameters::dz]=true;
    // rotations
    if (stParameterSelector.substr(3,1)=="1") 
      sel[RigidBodyAlignmentParameters::dalpha]=true;
    if (stParameterSelector.substr(4,1)=="1") 
      sel[RigidBodyAlignmentParameters::dbeta]=true;
    if (stParameterSelector.substr(5,1)=="1") 
      sel[RigidBodyAlignmentParameters::dgamma]=true;

    for (unsigned int i=0; i<npar; i++) {
      if (sel[i]==true) edm::LogInfo("BeginJob") <<"Parameter "<< i <<" active.";
    }
  }

  // select alignables 
  edm::LogInfo("BeginJob") <<"select alignables ...";
  theAlignmentParameterBuilder->addSelection(stAlignableSelector,sel);

  // get alignables
  Alignables theAlignables = theAlignmentParameterBuilder->alignables();
  edm::LogInfo("BeginJob") <<"got alignables: "<<theAlignables.size();

  // create AlignmentParameterStore 
  //theAlignmentParameterStore = new AlignmentParameterStore(theAlignables);
  //edm::LogInfo("BeginJob") <<"store created!";

  // initialize alignment algorithm
  theAlignmentAlgo->initialize( iSetup, theAlignableTracker );

}

//_____________________________________________________________________________
// Terminate algorithm
void AlignmentProducer::endOfJob()
{

  edm::LogInfo("EndJob") << "At end of job: terminating algorithm";
  theAlignmentAlgo->terminate();

}


//__________________________________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{

  edm::LogInfo("NewLoop") << "Starting loop number " << iLoop;

}


//__________________________________________________________________________________________________
// Called at end of loop
edm::EDLooper::Status AlignmentProducer::endOfLoop( const edm::EventSetup& iSetup, 
						    unsigned int iLoop )
{
  
  edm::LogInfo("EndLoop") << "Ending loop " << iLoop;

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;

}

//_____________________________________________________________________________
// Called at each event
edm::EDLooper::Status 
AlignmentProducer::duringLoop( const edm::Event& event, const edm::EventSetup& setup )
{

  edm::LogInfo("InLoop") << "Analyzing event";

  std::auto_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);

  // Retrieve what we need from the EventSetup
  edm::ESHandle<TrackerGeometry>  m_Geometry;
  edm::ESHandle<MagneticField>    m_MagneticField;
  edm::ESHandle<TrajectoryFitter> m_TrajectoryFitter;
  edm::ESHandle<Propagator>       m_Propagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> m_RecHitBuilder;
  getFromES( setup, m_Geometry, m_MagneticField, m_TrajectoryFitter, 
             m_Propagator, m_RecHitBuilder );

  // Retrieve track collection from the event
  edm::Handle<reco::TrackCollection> m_TrackCollection;
  event.getByLabel( theSrc, m_TrackCollection );
  //getFromEvt( event, m_TrackCollection );
    
  // Run the refitter algorithm  
  AlgoProductCollection m_algoResults;
  theRefitterAlgo.runWithTrack( m_Geometry.product(),m_MagneticField.product(),
    *m_TrackCollection, m_TrajectoryFitter.product(), m_Propagator.product(), 
    m_RecHitBuilder.product(), m_algoResults );

  // Run the alignment algorithm
  theAlignmentAlgo->run(  m_algoResults );


  return kContinue;

}

