/**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.4 $
 *  $Date: 2006/11/15 14:37:22 $
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
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"
#include "Mille.h"       // 'unpublished' interface located in src
#include "PedeSteerer.h" // dito
#include "PedeReader.h" // dito
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectory.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h" // Algebraic matrices
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include <fstream>
#include <vector>
#include <sstream>

#include <TMath.h>
#include <TArrayF.h>

typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

// Constructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), 
  theConfig(cfg), theMode(this->decodeMode(theConfig.getUntrackedParameter<std::string>("mode"))),
  theDir(theConfig.getUntrackedParameter<std::string>("fileDir")),
  theAlignmentParameterStore(0), theAlignables(), theAlignableNavigator(0),
  theMonitor(0), theMille(0), thePedeSteer(0), theMinNumHits(cfg.getParameter<int>("minNumHits"))
{
  if (!theDir.empty() && theDir.find_last_of('/') != theDir.size()-1) theDir += '/';// may need '/'
  edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm" << "Start in mode '"
                            << theConfig.getUntrackedParameter<std::string>("mode")
                            << "' with output directory " << theDir << ".";
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
  theAlignableNavigator = new AlignableNavigator(tracker);
  theAlignmentParameterStore = store;
  theAlignables = theAlignmentParameterStore->alignables();

  edm::ParameterSet pedeSteerCfg(theConfig.getParameter<edm::ParameterSet>("pedeSteerer"));
  pedeSteerCfg.addUntrackedParameter("fileDir", theDir);
//   thePedeSteer = new PedeSteerer(tracker, theAlignmentParameterStore, pedeSteerCfg);
  thePedeSteer = new PedeSteerer(tracker, theAlignables, pedeSteerCfg);

  // after PedeSteerer which uses the SelectionUserVariables attached to the parameters
  this->buildUserVariables(theAlignables); // for hit statistics and/or pede result

  if (theMode == kFull || theMode == kMille) {
    theMille = new Mille((theDir + theConfig.getParameter<std::string>("binaryFile")).c_str());
    const std::string moniFile(theConfig.getUntrackedParameter<std::string>("monitorFile"));
    if (moniFile.size()) theMonitor = new MillePedeMonitor((theDir + moniFile).c_str());
  }
  if (theMode == kFull || theMode == kMille || theMode == kPedeSteer || theMode == kPede) {
    this->doIO(theDir + theConfig.getParameter<std::string>("treeFile"), 0);
  }
}

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::terminate()
{
  delete theMille;// delete to close binary before running pede below (flush would be enough...)
  theMille = 0;

  bool pedeOk = true;
  if (theMode == kFull || theMode == kPede || theMode == kPedeRun) {
    pedeOk = thePedeSteer->runPede(theDir + theConfig.getParameter<std::string>("binaryFile"));
  }
  if (theMode == kFull || theMode == kPede || theMode == kPedeRun || theMode == kPedeRead) {
    if (pedeOk && this->readFromPede(thePedeSteer->pedeOutFile())) {
      edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::terminate"
                                << "Read successfully from " << thePedeSteer->pedeOutFile();
      // FIXME: problem if what is read in does not correspond to store
      theAlignmentParameterStore->applyParameters(); // FIXME ?
    } else {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::terminate"
                                 << "Problems running pede or reading from "
                                 << thePedeSteer->pedeOutFile();
    }
  }

  if (theMode == kFull || theMode == kMille) {
    this->doIO(theDir + theConfig.getParameter<std::string>("treeFile"), 1);
  } else if (theMode == kPede || theMode == kPedeRun || theMode == kPedeRead) {
    this->doIO(theDir + theConfig.getParameter<std::string>("treeFile"), 2);
  }

  // FIXME: should we delete here or in destructor?
  delete theAlignableNavigator;
  theAlignableNavigator = 0;
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
  if (theMode != kFull && theMode != kMille) return; // no theMille created...

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

    std::vector<AlignmentParameters*> parVec(refTrajPtr->recHits().size());//to add hits if all fine
    std::vector<bool> validHitVecY(refTrajPtr->recHits().size()); // collect hit statistics...
    int nValidHitsX = 0;                                // ...assuming that there are no y-only hits
    // Use recHits from ReferenceTrajectory (since they have the right order!):
    for (unsigned int iHit = 0; iHit < refTrajPtr->recHits().size(); ++iHit) {
      const int flagXY = this->addGlobalDerivatives(refTrajPtr, iHit, parVec[iHit]);
      if (flagXY < 0) { // problem
	nValidHitsX = -1;
	break;
      } else { // hit is fine, increase x/y statistics
        if (flagXY >= 1) ++nValidHitsX;
        validHitVecY[iHit] = (flagXY >= 2);
      } 
    } // end loop on hits

    if (nValidHitsX >= theMinNumHits) { // enough 'good' alignables hit: increase the hit statistics
      for (unsigned int iHit = 0; iHit < validHitVecY.size(); ++iHit) {
        if (!parVec[iHit]) continue; // in case a non-selected alignable was hit (flagXY == 0)
        MillePedeVariables *mpVar = static_cast<MillePedeVariables*>(parVec[iHit]->userVariables());
        mpVar->increaseHitsX(); // every hit has an x-measurement, cf. above...
        if (validHitVecY[iHit]) mpVar->increaseHitsY();
      }
      theMille->end();
    } else {
      theMille->kill();
    }
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

  if (theMonitor) theMonitor->fillRefTrajectory(refTrajPtr);

  return refTrajPtr;
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::addGlobalDerivatives
(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, unsigned int iHit,
 AlignmentParameters *&params)
{
   // FIXME: helix tsos correct or should we use the original fitted one?
  params = 0;
  int flagXY =
    this->globalDerivatives(refTrajPtr->recHits()[iHit], refTrajPtr->trajectoryStates()[iHit],
			    kLocalX, theFloatBuffer, theIntBuffer, params);
  if (flagXY == 1) {
    this->callMille(refTrajPtr, iHit, kLocalX, theFloatBuffer, theIntBuffer);

    if (this->is2D((refTrajPtr->recHits()[iHit]))) {
      const int flagY =
	this->globalDerivatives(refTrajPtr->recHits()[iHit], refTrajPtr->trajectoryStates()[iHit],
				kLocalY, theFloatBuffer, theIntBuffer, params);
      if (flagY != flagXY) {
	edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addGlobalDerivatives"
				     << "flagX = " << flagXY << ", flagY = " << flagY
				     << " => ignore track";
	flagXY = -1;
      } else {
	this->callMille(refTrajPtr, iHit, kLocalY, theFloatBuffer, theIntBuffer);
        flagXY = 2;
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
						   std::vector<int> &globalLabels,
                                                   AlignmentParameters *&params) const
{
  params = 0;
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
    // FIXME: In ORCA also if from stereo only the 'second' module is hit (said Markus S.)
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
  params = ali->alignmentParameters();
  if (!params) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivatives"
				 << "No AlignableParameters for Alignable in store.";
    return -1;
  }

  const std::vector<bool> &selPars = params->selector();
  const AlgebraicMatrix derivs(params->derivatives(tsos, alidet));
  // cols: 2, i.e. x&y, rows: parameters, usually RigidBodyAlignmentParameters::N_PARAM
  for (unsigned int iSel = 0; iSel < selPars.size(); ++iSel) {
    if (selPars[iSel]) {
      globalDerivatives.push_back(derivs[iSel][xOrY]/thePedeSteer->cmsToPedeFactor(iSel));
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
  // e.g. problem: What about glued detectors where only one module is hit?
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


//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::readFromPede(const std::string &pedeOutName)
{
  bool allEmpty = this->areEmptyParams(theAlignables);
  
  PedeReader reader(pedeOutName.c_str(), *thePedeSteer);
  std::vector<Alignable*> alis;
  bool okRead = reader.read(alis);

  std::stringstream out("Read ");
  out << alis.size() << " alignables";
  if (alis.size() != theAlignables.size()) {
    out << " while " << theAlignables.size() << " in store";
  }
  if (!okRead) out << ", but problems in reading"; 
  if (!allEmpty) out << ", possibly overwriting previous settings";
  out << ".";
  if (okRead && allEmpty) {
    edm::LogInfo("Alignment") << "@SUB=readFromPede" << out.str();
  } else {
    edm::LogError("Alignment") << "@SUB=readFromPede" << out.str();
  }

  // FIXME: Should we 'transfer' the alignables to the store?

  return (okRead && allEmpty);
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::areEmptyParams(const std::vector<Alignable*> &alignables) const
{
  
  for (std::vector<Alignable*>::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {
    const AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (params) {
      const AlgebraicVector &parVec(params->parameters());
      for (int i = 0; i < parVec.num_row(); ++i) {
        if (parVec[i] != 0.) {
          return false;
        }
      }
    }
  }
  
  return true;
}

//__________________________________________________________________________________________________
unsigned int MillePedeAlignmentAlgorithm::doIO(const std::string &ioFile, int loop) const
{

  AlignmentIORoot aliIO;
  int ioerr = 0;
  unsigned int result = 0;
  if (loop == 0) {
    aliIO.writeAlignableOriginalPositions(theAlignables, ioFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " in writeAlignableOriginalPositions";
      ++result;
    }
    aliIO.writeOrigRigidBodyAlignmentParameters(theAlignables, ioFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " in writeAlignmentParameters, " << loop;
      ++result;
    }
  } else {
    MillePedeVariablesIORoot millePedeIO;
    if (loop > 1) { // FIXME: How to 'save' hit statistics really in case of seperate running?
      const std::vector<AlignmentUserVariables*> mpVars = // user variables belong to millePedeIO!
        millePedeIO.readMillePedeVariables(theAlignables, ioFile.c_str(), loop - 1, ioerr);
      this->addHits(theAlignables, mpVars);
    }
    millePedeIO.writeMillePedeVariables(theAlignables, ioFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " writing MillePedeVariables";
      ++result;
    }
    aliIO.writeAlignmentParameters(theAlignables, ioFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " in writeAlignmentParameters, " << loop;
      ++result;
    }
  }
  
  aliIO.writeAlignableAbsolutePositions(theAlignables, ioFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO" << "Problem " << ioerr
                               << " in writeAlignableAbsolutePositions, " << loop;
    ++result;
  }
  aliIO.writeAlignableRelativePositions(theAlignables, ioFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO" << "Problem " << ioerr
                               << " in writeAlignableRelativePositions, " << loop;
    ++result;
  }

  return result;
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm::buildUserVariables(const std::vector<Alignable*> &alis) const
{
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(); iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (!params) {
      throw cms::Exception("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::buildUserVariables"
                                        << "No parameters for alignable";
    }
    params->setUserVariables(new MillePedeVariables(params->size()));
  }
}

//__________________________________________________________________________________________________
MillePedeAlignmentAlgorithm::EMode
MillePedeAlignmentAlgorithm::decodeMode(const std::string &mode) const
{
  if (mode == "full") return kFull;   // = mille & pede
  if (mode == "mille") return kMille; // write binary and store IO start values (includes pedeSteer)
  if (mode == "pede") return kPede;   // = pedeSteer & pedeRun
  if (mode == "pedeRun") return kPedeRun; // run pede executable (includes pedeRead)
  if (mode == "pedeSteer") return kPedeSteer; // create pede steering (fixed parameters, constraints)
  if (mode == "pedeRead") return kPedeRead;   // read in result from pede and store via IO

  throw cms::Exception("BadConfig") 
    << "Unknown mode '" << mode 
    << "', use 'full', 'mille', 'pede', 'pedeRun', 'pedeSteer' or 'pedeRead'.";

}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::addHits(const std::vector<Alignable*> &alis,
                                          const std::vector<AlignmentUserVariables*> &mpVars) const
{
  bool allOk = (mpVars.size() == alis.size());
  std::vector<AlignmentUserVariables*>::const_iterator iUser = mpVars.begin();
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(); 
       iAli != alis.end() && iUser != mpVars.end(); ++iAli, ++iUser) {
    MillePedeVariables *mpVarNew = dynamic_cast<MillePedeVariables*>(*iUser);
    AlignmentParameters *ps = (*iAli)->alignmentParameters();
    MillePedeVariables *mpVarOld = (ps ? dynamic_cast<MillePedeVariables*>(ps->userVariables()) : 0);
    if (!mpVarNew || !mpVarOld || mpVarOld->size() != mpVarNew->size()) {
      allOk = false;
      continue; // FIXME error etc.?
    }

    mpVarOld->increaseHitsX(mpVarNew->hitsX());
    mpVarOld->increaseHitsY(mpVarNew->hitsY());
  }
  
  return allOk;
}


