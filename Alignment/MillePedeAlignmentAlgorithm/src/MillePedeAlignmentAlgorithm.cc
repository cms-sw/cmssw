/**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.2 $
 *  $Date: 2006/11/07 10:45:09 $
 *  (last update by $Author: flucke $)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
// in header, too
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
// end in header, too

//#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeAlignmentAlgorithm.h"
#include "Mille.h"       // 'unpublished' interface located in src
#include "PedeSteerer.h" // dito
#include "PedeReader.h" // dito
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectory.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h" // Algebraic matrices
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include <fstream>
#include <vector>

#include <TMath.h>
#include <TArrayF.h>

typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

// Constructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), 
  theConfig(cfg),
  theAlignmentParameterStore(0), theAlignables(), theAlignableNavigator(0),
  theMonitor(0), theMille(0), thePedeSteer(0), theMinNumHits(cfg.getParameter<int>("minNumHits"))
{
}

// Destructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::~MillePedeAlignmentAlgorithm()
{
  delete theAlignableNavigator;
  delete theMille;
  delete theMonitor;
  delete thePedeSteer;
}

// Call at beginning of job ---------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::initialize(const edm::EventSetup &setup, 
  AlignableTracker* tracker, AlignmentParameterStore* store)
{
  //  edm::LogWarning("Alignment") << "[MillePedeAlignmentAlgorithm] Initializing...";

  // accessor Det->AlignableDet
  theAlignableNavigator = new AlignableNavigator(tracker);

  // set alignmentParameterStore
  theAlignmentParameterStore = store;

  // get alignables
  theAlignables = theAlignmentParameterStore->alignables();

  thePedeSteer = new PedeSteerer(tracker, theAlignmentParameterStore,
                                 theConfig.getParameter<edm::ParameterSet>("pedeSteerer"));

  AlignmentIORoot aliIO;
  const std::string file(theConfig.getParameter<std::string>("treeFile"));
  // pedeOut defines whether alignment should be read in from pede output or produced from tracks
  const std::string pedeOut(theConfig.getUntrackedParameter<std::string>("pedeOut"));
  if (pedeOut.empty()) {
    theMille = new Mille(theConfig.getParameter<std::string>("binaryFile").c_str());
    const std::string monitorFile(theConfig.getUntrackedParameter<std::string>("monitorFile"));
    if (!monitorFile.empty()) {
      theMonitor = new MillePedeMonitor(monitorFile.c_str());
    }
    const int loop = 0;
    int ioerr = 0;
    aliIO.writeAlignableOriginalPositions(theAlignables, file.c_str(), loop, false, ioerr);
    if (ioerr) edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                          << "problem " << ioerr 
                                          << " in writeAlignableOriginalPositions";
    // get misalignment parameters and write to root file
    aliIO.writeAlignmentParameters(theAlignables, file.c_str(), loop, false, ioerr);
    if (ioerr) edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                          << "problem " << ioerr << " writeAlignmentParameters";


  } else {
    // FIXME: initialise everything despite of reading in from pede for pede internal iterations?
    if (this->readFromPede(pedeOut)) {
      edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                << "read successfully from " << pedeOut;
      const int loop = 1;
      int ierr = 0;
      MillePedeVariablesIORoot millePedeIO;
      millePedeIO.writeMillePedeVariables(theAlignables, file.c_str(), loop, false, ierr);
      if (ierr) edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                           << "error " << ierr << " writing MillePedeVariables";
      // get misaligned positions and write to root file
      aliIO.writeAlignableAbsolutePositions(theAlignables, file.c_str(), loop, false, ierr);
      if (ierr) edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                           << "problem " << ierr 
                                           << " in writeAlignableAbsolutePositions";
      aliIO.writeAlignmentParameters(theAlignables, file.c_str(), loop, false, ierr);
      if (ierr) edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                           << "problem " << ierr << " writeAlignmentParameters";
    } else {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                 << "problems reading from " << pedeOut;
    }
  }

}

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::terminate()
{
  // FIXME: should we delete here or in destructor?
  delete theAlignableNavigator;
  theAlignableNavigator = 0;
  delete theMille;
  theMille = 0;
  delete theMonitor;
  theMonitor = 0;
  delete thePedeSteer;
  thePedeSteer = 0;
}

// Run the algorithm on trajectories and tracks -------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::run(const edm::EventSetup &setup,
				      const TrajTrackPairCollection &tracks) 
{

  const MagneticField *magField = this->getMagneticField(setup);

  // loop over tracks  
  for (TrajTrackPairCollection::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
    Trajectory *traj = (*it).first;
    reco::Track *track = (*it).second;
    if (theMonitor) theMonitor->fillTrack(track, traj);
    ReferenceTrajectoryBase::ReferenceTrajectoryPtr refTrajPtr = 
      this->referenceTrajectory(traj->measurements().back().updatedState(),
				traj->recHits(), magField);
    if (!refTrajPtr->isValid()) continue; // currently e.g. if any invalid hit (FIXME for cosmic?)

    int nValidHits = 0;
    // Use recHits from ReferenceTrajectory (since they have the right order!):
    for (unsigned int iHit = 0; iHit < refTrajPtr->recHits().size(); ++iHit) {
      const int flag = this->addGlobalDerivatives(refTrajPtr, iHit);
      if (flag < 0) {
	nValidHits = -1;
	break;
      } else {
	nValidHits += flag;
      }
    } // end loop on hits

    if (nValidHits >= theMinNumHits) theMille->end();
    else                             theMille->kill();
  } // end of track loop
}


//____________________________________________________
ReferenceTrajectoryBase::ReferenceTrajectoryPtr MillePedeAlignmentAlgorithm::referenceTrajectory
(const TrajectoryStateOnSurface &refTsos, const ConstRecHitContainer &hitVec,
 const MagneticField *magField) const
{
  // currently assuming that all hits are valid

  ReferenceTrajectoryBase::ReferenceTrajectoryPtr refTrajPtr =   // hits are backward!
    new ReferenceTrajectory(refTsos, hitVec, true, magField, ReferenceTrajectoryBase::combined);//energyLoss);

//   edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addTrack" 
// 			    << "constructed reference trajectory";
  if (theMonitor) theMonitor->fillRefTrajectory(refTrajPtr);

  return refTrajPtr;
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::addGlobalDerivatives
(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, unsigned int iHit)
{
   // FIXME: helix tsos correct or should we use the original fitted one?

  int flagXY =
    this->globalDerivatives(refTrajPtr->recHits()[iHit], refTrajPtr->trajectoryStates()[iHit],
			    kLocalX, theFloatBuffer, theIntBuffer);
  if (flagXY > 0) {
    this->callMille(refTrajPtr, iHit, kLocalX, theFloatBuffer, theIntBuffer);

    if (this->is2D((refTrajPtr->recHits()[iHit]))) {
      const int flagY =
	this->globalDerivatives(refTrajPtr->recHits()[iHit], refTrajPtr->trajectoryStates()[iHit],
				kLocalY, theFloatBuffer, theIntBuffer);
      if (flagY != flagXY) {
	edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addGlobalDerivatives"
				     << "flagX = " << flagXY << ", flagY = " << flagY
				     << " => ignore track";
	flagXY = -1;
      } else {
	this->callMille(refTrajPtr, iHit, kLocalY, theFloatBuffer, theIntBuffer);
      }
    }
  }

  return flagXY;
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::globalDerivatives(const ConstRecHitPointer &recHitPtr,
						   const TrajectoryStateOnSurface &tsos,
						   MeasurementDirection xOrY,
						   std::vector<float> &globalDerivatives,
						   std::vector<int> &globalLabels) const
{

  globalDerivatives.clear();
  globalLabels.clear();

  // get AlignableDet for this hit, want const but ->selectedDerivatives needs non-const...
  AlignableDet *alidet = theAlignableNavigator->alignableDetFromGeomDet(recHitPtr->det());
  if (!alidet) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivatives"
				 << "AlignableDet not found in Navigator";
    return -1;
  }

  // get relevant Alignable
  Alignable *ali = theAlignmentParameterStore->alignableFromAlignableDet(alidet);
  if (!ali) { // FIXME: if not selected to be aligned? need regardAllHits, cf. below?
    // happens e.g. for pixel alignables if pixel not foreseen to be aligned
    const GlobalPoint posDet(alidet->globalPosition());
    LogDebug("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivatives"
			  << "No alignable in Store for AlignableDet at (r/z/phi) = ("
			  << posDet.perp() << "/" << posDet.z() << "/" << posDet.phi() << ").";
    return 0;
  }

  const unsigned int alignableLabel = thePedeSteer->alignableLabel(ali);
  if (0 == alignableLabel) { // FIXME: what about regardAllHits in Markus' code?
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivatives"
				 << "Label not found.";
    return -1;
  }

  // get Alignment Parameters
  const AlignmentParameters *params = ali->alignmentParameters();
  if (!params) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivatives"
				 << "No AlignableParameters for Alignable in store.";
    return -1;
  }

  //this->recursiveFillLabelHist(ali); // FIXME: not needed?

  const std::vector<bool> &selPars = params->selector();
  const AlgebraicMatrix derivs(params->derivatives(tsos, alidet));
  // cols: 2, i.e. x&y, rows: parameters, usually RigidBodyAlignmentParameters::N_PARAM
  for (unsigned int iSel = 0; iSel < selPars.size(); ++iSel) {
    if (selPars[iSel]) {
      globalDerivatives.push_back(derivs[iSel][xOrY]);
      globalLabels.push_back(thePedeSteer->parameterLabel(alignableLabel, iSel));
    }
  }
  return 1;
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::callMille
(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
 unsigned int iTrajHit, MeasurementDirection xOrY,
 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels)
{
  const unsigned int xyIndex = iTrajHit*2 + xOrY;
  const float residuum = 
    refTrajPtr->measurements()[xyIndex] - refTrajPtr->trajectoryPositions()[xyIndex];
  const float covariance = refTrajPtr->measurementErrors()[xyIndex][xyIndex];
  const float sigma = (covariance > 0. ? TMath::Sqrt(covariance) : 0.);

  const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();

  TArrayF localDerivs(locDerivMatrix.num_col());
  for (int i = 0; i < localDerivs.GetSize(); ++i) {
    localDerivs[i] = locDerivMatrix[xyIndex][i];
  }

  // FIXME: verify that &(vector[0]) is valid for all vector implementations
  theMille->mille(localDerivs.GetSize(), localDerivs.GetArray(),
		  globalDerivatives.size(), &(globalDerivatives[0]), &(globalLabels[0]),
		  residuum, sigma);
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm::is2D(const ConstRecHitPointer &recHit) const
{
  // FIXME: Check whether this is a reliable and recommended way to find out...
  return (!recHit->detUnit() || recHit->detUnit()->type().isTrackerPixel());
}

//____________________________________________________
const MagneticField*
MillePedeAlignmentAlgorithm::getMagneticField(const edm::EventSetup &setup) //const
{
  
  edm::ESHandle<TrackerGeometry>                geometryHandle;
  edm::ESHandle<MagneticField>                  magneticFieldHandle;
  edm::ESHandle<TrajectoryFitter>               trajectoryFitterHandle;
  edm::ESHandle<Propagator>                     propagatorHandle;
  edm::ESHandle<TransientTrackingRecHitBuilder> recHitBuilderHandle;
  // F. Ronga told that this access to EventSetup is optimised,
  // so I can use it here accessing several things without performance loss
  this->getFromES(setup, geometryHandle, magneticFieldHandle, trajectoryFitterHandle, 
		  propagatorHandle, recHitBuilderHandle);

  return magneticFieldHandle.product();
}


//____________________________________________________
void MillePedeAlignmentAlgorithm::recursiveFillLabelHist(Alignable *ali) const
{

  while (ali) {
    const unsigned int aliLabel = thePedeSteer->alignableLabel(ali);
    if (theMonitor) theMonitor->fillDetLabel(aliLabel);
    ali = ali->mother();
  }
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::readFromPede(const std::string &pedeOutName)
{

  PedeReader reader(pedeOutName.c_str(), *thePedeSteer);
  std::vector<Alignable*> alis;
  bool okRead = reader.read(alis);

  if (!okRead || alis.size() != theAlignables.size()) {
    edm::LogWarning("Alignment") << "@SUB=readFromPede"
                                 << "read " << alis.size() << " alignables, while " 
                                 << theAlignables.size() << " in store, read " 
                                 << (okRead ? "OK" : "not OK");
  }
  
  return okRead;
}
