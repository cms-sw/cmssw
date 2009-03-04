 /**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke/Ivan Reid
 *  date       : February 2009 *  $Revision: 1.42 $
 *  $Date: 2008/11/10 14:48:42 $
 *  (last update by $Author: henderle $)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "CLHEP/Matrix/SymMatrix.h"

#include <fstream>
#include <string>
#include <set>

// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// #include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

// #include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"


// #include <vector>

// #include <TMatrixDSym.h>
// #include <TMatrixD.h>
// #include <TMatrixF.h>

// class Alignable;
// class AlignableTracker;
// class AlignableMuon;

// class AlignmentParameters;
// class AlignableNavigator;
// class AlignableDetOrUnitPtr;
// class AlignmentUserVariables;

// class AlignmentParameterStore;

// class MillePedeMonitor;
// class PedeSteerer;
// class PedeLabeler;
// class Mille;
//class TrajectoryFactoryBase;


//#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//#include "TrackingTools/PatternTools/interface/Trajectory.h"
//// in header, too
//// end in header, too
//
//#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"
//#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"
//#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"
//#include "Mille.h"       // 'unpublished' interface located in src
//#include "PedeSteerer.h" // dito
//#include "PedeReader.h" // dito
//#include "PedeLabeler.h" // dito
//
//#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
//#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"
//
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
//#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"

//#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

// includes to make known that they inherit from Alignable:
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

// #include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
// #include "DataFormats/TrackReco/interface/Track.h"


// #include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
// #include <Geometry/CommonDetUnit/interface/GeomDetType.h>

// #include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

// #include <fstream>
// #include <sstream>
// #include <algorithm>

// #include <TMath.h>
// #include <TMatrixDSymEigen.h>
// typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
// typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

class ApeSettingAlgorithm : public AlignmentAlgorithmBase
{
 public:
  /// Constructor
  ApeSettingAlgorithm(const edm::ParameterSet &cfg);

  /// Destructor
  virtual ~ApeSettingAlgorithm();

  /// Call at beginning of job
  virtual void initialize(const edm::EventSetup &setup, AlignableTracker *tracker,
			  AlignableMuon *muon, AlignmentParameterStore *store);

  /// Call at end of job
  virtual void terminate();

  /// Run the algorithm on trajectories and tracks
  virtual void run(const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks);

 private:
  edm::ParameterSet         theConfig;
//   AlignmentParameterStore  *theAlignmentParameterStore;
//   std::vector<Alignable*>   theAlignables;
  AlignableNavigator       *theAlignableNavigator;
  AlignableTracker         *theTracker;
  bool                     saveApeToAscii_,readApeFromAscii_;
 
};

//____________________________________________________
//____________________________________________________
//____________________________________________________
//____________________________________________________


// Constructor ----------------------------------------------------------------
//____________________________________________________
ApeSettingAlgorithm::ApeSettingAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), theConfig(cfg),
  // theAlignmentParameterStore(0) //, theAlignables(), 
  theAlignableNavigator(0)
{
  edm::LogInfo("Alignment") << "@SUB=ApeSettingAlgorithm" << "Start.";
  saveApeToAscii_ = theConfig.getUntrackedParameter<bool>("saveApeToASCII");
  readApeFromAscii_ = theConfig.getParameter<bool>("readApeFromASCII");
}

// Destructor ----------------------------------------------------------------
//____________________________________________________
ApeSettingAlgorithm::~ApeSettingAlgorithm()
{
  delete theAlignableNavigator;
}

// Call at beginning of job ---------------------------------------------------
//____________________________________________________
void ApeSettingAlgorithm::initialize(const edm::EventSetup &setup, 
                                             AlignableTracker *tracker, AlignableMuon *muon,
                                             AlignmentParameterStore *store)
{
  theAlignableNavigator = new AlignableNavigator(tracker, muon);
  theTracker = tracker;

  if (readApeFromAscii_)
    { std::ifstream apeReadFile(theConfig.getParameter<edm::FileInPath>("apeASCIIReadFile").fullPath().c_str()); //requires <fstream>
    if (!apeReadFile.good())
      { edm::LogInfo("Alignment") << "@SUB=initialize" <<"Problem opening APE file"
				  << theConfig.getParameter<edm::FileInPath>("apeASCIIReadFile").fullPath();
      return;
}
    std::set<int> apeList; //To avoid duplicates
    while (!apeReadFile.eof())
      { int apeId=0; double x11,x21,x22,x31,x32,x33;
      apeReadFile>>apeId>>x11>>x21>>x22>>x31>>x32>>x33>>std::ws;
      //idr What sanity checks do we need to put here?
      if (apeId != 0) //read appears valid?
	if (apeList.find(apeId) == apeList.end()) //Not previously done
	  { AlignmentPositionError ape(GlobalError(x11,x21,x22,x31,x32,x33));
	  DetId id(apeId);
	  AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(id)); //NULL if none
	  if (alidet) 
	    { alidet->setAlignmentPositionError(ape);
	    apeList.insert(apeId);
	    }
	  }
	else
	  { edm::LogInfo("Alignment") << "@SUB=initialize" << "Skipping duplicate APE for DetId "<<apeId;
	  }
      }
    apeReadFile.close();
    edm::LogInfo("Alignment") << "@SUB=initialize" << "Set "<<apeList.size()<<" APE values.";
    }
}

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void ApeSettingAlgorithm::terminate()
{
  //  edm::LogInfo("Alignment") << "@SUB=terminate" << "Could start manipulating Ape.";

  if (saveApeToAscii_)
    { AlignmentErrors* aliErr=theTracker->alignmentErrors();
    int theSize=aliErr->m_alignError.size();
    std::ofstream apeSaveFile(theConfig.getUntrackedParameter<std::string>("apeASCIISaveFile").c_str()); //requires <fstream>
    for (int i=0; i < theSize; ++i)
      { apeSaveFile<<aliErr->m_alignError[i].rawId();
      CLHEP::HepSymMatrix sm= aliErr->m_alignError[i].matrix();
      for (int j=0; j < 3; ++j)
	for (int k=0; k <= j; ++k)
	  apeSaveFile<<"  "<<sm[j][k];
      apeSaveFile<<std::endl;
      }
    delete aliErr;
    apeSaveFile.close();
    }
  // clean up at end:  // FIXME: should we delete here or in destructor?
  delete theAlignableNavigator;
  theAlignableNavigator = 0;
}

// Run the algorithm on trajectories and tracks -------------------------------
//____________________________________________________
void ApeSettingAlgorithm::run(const edm::EventSetup &setup,
				      const ConstTrajTrackPairCollection &tracks) 
{
  // nothing to do here?
}

// Plugin definition for the algorithm
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory,
		   ApeSettingAlgorithm, "ApeSettingAlgorithm");



