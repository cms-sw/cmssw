/// \file AlignmentProducer.cc
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.14 $
///  last update: $Date: 2006/11/07 17:48:25 $
///  by         : $Author: flucke $

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducer.h"

// System include files
#include <memory>
#include <sstream>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Alignment/TrackerAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"


using namespace std;

//_____________________________________________________________________________
AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) :
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  stNFixAlignables(iConfig.getParameter<int>("nFixAlignables") ),
  stRandomShift(iConfig.getParameter<double>("randomShift")),
  stRandomRotation(iConfig.getParameter<double>("randomRotation")),
  applyDbAlignment_( iConfig.getUntrackedParameter<bool>("applyDbAlignment",false) ),
  doMisalignmentScenario(iConfig.getParameter<bool>("doMisalignmentScenario")),
  saveToDB(iConfig.getParameter<bool>("saveToDB"))
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::AlignmentProducer";

  theParameterSet=iConfig;

  // Tell the framework what data is being produced
  setWhatProduced(this);

  // Create the alignment algorithm
  edm::ParameterSet algoConfig = iConfig.getParameter<edm::ParameterSet>( "algoConfig" );
  std::string algoName = algoConfig.getParameter<std::string>("algoName");
  theAlignmentAlgo = AlignmentAlgorithmPluginFactory::getAlgorithm( algoName, algoConfig );

  // Check if found
  if ( !theAlignmentAlgo )
	throw cms::Exception("BadConfig") << "Couldn't find algorithm called " << algoName;

}


//_____________________________________________________________________________
// Close files, etc.

AlignmentProducer::~AlignmentProducer()
{

}


//_____________________________________________________________________________
// Produce tracker geometry

AlignmentProducer::ReturnType 
AlignmentProducer::produce( const TrackerDigiGeometryRecord& iRecord )
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::produce";

  return theTracker;
  
}


//_____________________________________________________________________________
// Initialize algorithm

void AlignmentProducer::beginOfJob( const edm::EventSetup& iSetup )
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob";

  nevent=0;

  // Create the tracker geometry from ideal geometry (first time only)
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  iSetup.get<IdealGeometryRecord>().get( cpv );
  iSetup.get<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),&(*gD)) );
  
  // Retrieve and apply alignments, if requested (requires DB setup)
  if ( applyDbAlignment_ ) {
    edm::ESHandle<Alignments> alignments;
    iSetup.get<TrackerAlignmentRcd>().get( alignments );
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    iSetup.get<TrackerAlignmentErrorRcd>().get( alignmentErrors );
    GeometryAligner aligner;
    aligner.applyAlignments<TrackerGeometry>( &(*theTracker), &(*alignments), &(*alignmentErrors) );
  }

  // create alignable tracker
  theAlignableTracker = new AlignableTracker( &(*gD), &(*theTracker) );

  // create alignment parameter builder
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "Creating AlignmentParameterBuilder";
  edm::ParameterSet aliParamBuildCfg = 
    theParameterSet.getParameter<edm::ParameterSet>("ParameterBuilder");
  theAlignmentParameterBuilder = new AlignmentParameterBuilder(theAlignableTracker,
                                                               aliParamBuildCfg);
  // fix alignables
  if (stNFixAlignables>0) theAlignmentParameterBuilder->fixAlignables(stNFixAlignables);

  // get alignables
  Alignables theAlignables = theAlignmentParameterBuilder->alignables();
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "got " << theAlignables.size() << " alignables";

  // create AlignmentParameterStore 
  theAlignmentParameterStore = new AlignmentParameterStore(theAlignables);
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "AlignmentParameterStore created!";

  // Create misalignment scenario, apply to geometry
  if (doMisalignmentScenario) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                              << "applying misalignment scenario ...";
    edm::ParameterSet scenarioConfig 
      = theParameterSet.getParameter<edm::ParameterSet>( "MisalignmentScenario" );
    MisalignmentScenarioBuilder scenarioBuilder( theAlignableTracker );
    scenarioBuilder.applyScenario( scenarioConfig );
  } else {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                              << "NOT applying misalignment scenario!";
  }

  // apply simple misalignment
  const std::string sParSel(theParameterSet.getParameter<std::string>("parameterSelectorSimple"));
  this->simpleMisalignment(theAlignables, sParSel, stRandomShift, stRandomRotation, true);
  //  edm::LogInfo("Alignment") <<"[AlignmentProducer] simple misalignment done!"; anyway 'messaged'

  // initialize alignment algorithm
  theAlignmentAlgo->initialize( iSetup, theAlignableTracker, theAlignmentParameterStore );
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "after call init algo...\n"
                            << "Now physically apply alignments to tracker geometry...";

  // actually execute all misalignments
  GeometryAligner aligner;
  std::auto_ptr<Alignments> alignments(theAlignableTracker->alignments());
  std::auto_ptr<AlignmentErrors> alignmentErrors(theAlignableTracker->alignmentErrors());
  aligner.applyAlignments<TrackerGeometry>( &(*theTracker),&(*alignments),&(*alignmentErrors));

}

//_____________________________________________________________________________
// Terminate algorithm

void AlignmentProducer::endOfJob()
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob" 
                            << "Terminating algorithm.";
  theAlignmentAlgo->terminate();

  // write alignments to database

  if (saveToDB) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob" 
                              << "Writing Alignments to DB...";
    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
    // get alignments+errors
    Alignments* alignments = theAlignableTracker->alignments();
    AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();
    // Define callback tokens for the two records
    size_t alignmentsToken = poolDbService->callbackToken("Alignments");
    size_t alignmentErrorsToken = poolDbService->callbackToken("AlignmentErrors");
    // Store
    poolDbService->newValidityForNewPayload<Alignments>(alignments, 
      poolDbService->endOfTime(), alignmentsToken);
    poolDbService->newValidityForNewPayload<AlignmentErrors>(alignmentErrors, 
      poolDbService->endOfTime(), alignmentErrorsToken);
  }


}

//_____________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Starting loop number " << iLoop;

}


//_____________________________________________________________________________
// Called at end of loop

edm::EDLooper::Status 
AlignmentProducer::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                            << "Ending loop " << iLoop;

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

//_____________________________________________________________________________
// Called at each event

edm::EDLooper::Status 
AlignmentProducer::duringLoop( const edm::Event& event, 
  const edm::EventSetup& setup )
{
  nevent++;

  if ((nevent<100 && nevent%10==0) 
      ||(nevent<1000 && nevent%100==0) 
      ||(nevent<10000 && nevent%100==0) 
      ||(nevent<100000 && nevent%1000==0) 
      ||(nevent<10000000 && nevent%1000==0)) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::duringLoop" 
                              << "Events processed: " << nevent;
  }
  // Run the refitter algorithm
  AlgoProductCollection m_algoResults = theAlignmentAlgo->refitTracks( event, setup );

  // Run the alignment algorithm
  theAlignmentAlgo->run(  setup, m_algoResults );

  // Clean-up
  for ( AlgoProductCollection::const_iterator it=m_algoResults.begin();
       it!=m_algoResults.end();it++) {
    delete (*it).first;
    delete (*it).second;
  }
  m_algoResults.clear();

  return kContinue;
}

// ----------------------------------------------------------------------------

void AlignmentProducer::simpleMisalignment(const Alignables &alivec, const std::string &selection, 
                                           float shift, float rot, bool local)
{

  std::ostringstream output; // collecting output

  if (shift>0 || rot >0) {
    output << "Adding random flat shift of max size " << shift
           << " and adding random flat rotation of max size " << rot <<" to ";

    std::vector<bool> commSel(0);
    if (selection != "-1") {
      AlignmentParameterSelector aSelector(0); // no tracker needed here...
      commSel = aSelector.decodeParamSel(selection);
      output << "parameters defined by (" << selection 
             << "), representing (x,y,z,alpha,beta,gamma).";
    } else {
      output << "the active parameters of each alignable.";
    }

    for (vector<Alignable*>::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
      Alignable* ali=(*it);
      vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);

      if (abs(shift)>0.00001) {
        AlgebraicVector s(3);
        s[0]=0; s[1]=0; s[2]=0;  
        if (mysel[RigidBodyAlignmentParameters::dx]) {
          s[0]=shift*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dy]) {
          s[1]=shift*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dz]) {
          s[2]=shift*double(random()%1000-500)/500.;
        }
        
        GlobalVector globalshift;
        if (local) {
          globalshift = ali->surface().toGlobal(Local3DVector(s[0],s[1],s[2]));
        } else {
          globalshift = Global3DVector(s[0],s[1],s[2]);
        }
        ali->move(globalshift);

      //AlignmentPositionError ape(dx,dy,dz);
      //ali->addAlignmentPositionError(ape);
      }

      if (abs(rot)>0.00001) {
        AlgebraicVector r(3);
        r[0]=0; r[1]=0; r[2]=0;
        if (mysel[RigidBodyAlignmentParameters::dalpha]) {
          r[0]=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dbeta]) {
          r[1]=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dgamma]) {
          r[2]=rot*double(random()%1000-500)/500.;
        }
        AlignmentTransformations TkAT;
        Surface::RotationType mrot = TkAT.rotationType(TkAT.rotMatrix3(r));
        if (local) ali->rotateInLocalFrame(mrot);
        else ali->rotateInGlobalFrame(mrot);
        
      //ali->addAlignmentPositionErrorFromRotation(mrot);

      }
    } // end loop on alignables
  } else {
    output << "No simple misalignment added!";
  }
  edm::LogInfo("Alignment")  << "@SUB=AlignmentProducer::simpleMisalignment" << output.str();

}
