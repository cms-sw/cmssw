 /**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke/Ivan Reid
 *  date       : February 2009 *  $Revision: 1.3 $
 *  $Date: 2009/03/09 19:29:18 $
 *  (last update by $Author: ireid $)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignment/interface/AlignableModifier.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "CLHEP/Matrix/SymMatrix.h"

#include <fstream>
#include <string>
#include <set>

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

// includes to make known that they inherit from Alignable:
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

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
  AlignableNavigator       *theAlignableNavigator;
  AlignableTracker         *theTracker;
  bool                     saveApeToAscii_,readApeFromAscii_;
  bool                     readLocalNotGlobal_;
  bool                     setComposites_,saveComposites_;
};

//____________________________________________________
//____________________________________________________
//____________________________________________________
//____________________________________________________


// Constructor ----------------------------------------------------------------
//____________________________________________________
ApeSettingAlgorithm::ApeSettingAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), theConfig(cfg),
  theAlignableNavigator(0)
{
  edm::LogInfo("Alignment") << "@SUB=ApeSettingAlgorithm" << "Start.";
  saveApeToAscii_ = theConfig.getUntrackedParameter<bool>("saveApeToASCII");
  saveComposites_ = theConfig.getUntrackedParameter<bool>("saveComposites");
  readApeFromAscii_ = theConfig.getParameter<bool>("readApeFromASCII");
  readLocalNotGlobal_ = theConfig.getParameter<bool>("readLocalNotGlobal");
  setComposites_ = theConfig.getParameter<bool>("setComposites");
  
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
{ theAlignableNavigator = new AlignableNavigator(tracker, muon);
 theTracker = tracker;
 
 if (readApeFromAscii_)
   { std::ifstream apeReadFile(theConfig.getParameter<edm::FileInPath>("apeASCIIReadFile").fullPath().c_str()); //requires <fstream>
   if (!apeReadFile.good())
     { edm::LogInfo("Alignment") << "@SUB=initialize" <<"Problem opening APE file: skipping"
				 << theConfig.getParameter<edm::FileInPath>("apeASCIIReadFile").fullPath();
     return;
     }
   std::set<int> apeList; //To avoid duplicates
   while (!apeReadFile.eof())
     { int apeId=0; double x11,x21,x22,x31,x32,x33;
     apeReadFile>>apeId>>x11>>x21>>x22>>std::ws;
     if (!readLocalNotGlobal_) { apeReadFile>>x31>>x32>>x33>>std::ws;}
     //idr What sanity checks do we need to put here?
     if (apeId != 0) //read appears valid?
       if (apeList.find(apeId) == apeList.end()) //Not previously done
	 {  DetId id(apeId);
	 AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(id)); //NULL if none
	 if (alidet)
	   { if ((alidet->components().size()<1) || setComposites_) //the problem with glued dets...
	     { if (readLocalNotGlobal_)
	       { AlgebraicSymMatrix as(3,0); 
	       as[0][0]=x11*x11; as[1][1]=x21*x21; as[2][2]=x22*x22; //local cov.
	       align::RotationType rt=alidet->globalRotation();
	       AlgebraicMatrix am(3,3);
	       am[0][0]=rt.xx(); am[0][1]=rt.xy(); am[0][2]=rt.xz();
	       am[1][0]=rt.yx(); am[1][1]=rt.yy(); am[1][2]=rt.yz();
	       am[2][0]=rt.zx(); am[2][1]=rt.zy(); am[2][2]=rt.zz();
	       am=am.T()*as*am; //symmetric matrix
	       alidet->setAlignmentPositionError(GlobalError(am[0][0],am[1][0],am[1][1],am[2][0],am[2][1],am[2][2]));
	       }
	     else
	       { alidet->setAlignmentPositionError(GlobalError(x11,x21,x22,x31,x32,x33)); //set for global
	       }
	     apeList.insert(apeId); //Flag it's been set
	     }
	   else
	     { edm::LogInfo("Alignment") << "@SUB=initialize" << "Not Setting APE for Composite DetId "<<apeId;
	     }
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
  if (saveApeToAscii_)
    { AlignmentErrors* aliErr=theTracker->alignmentErrors();
    int theSize=aliErr->m_alignError.size();
    std::ofstream apeSaveFile(theConfig.getUntrackedParameter<std::string>("apeASCIISaveFile").c_str()); //requires <fstream>
    for (int i=0; i < theSize; ++i)
      { int id=	aliErr->m_alignError[i].rawId();
      AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(DetId(id))); //NULL if none
      if (alidet && ((alidet->components().size()<1) || saveComposites_))
	{ apeSaveFile<<id;
	CLHEP::HepSymMatrix sm= aliErr->m_alignError[i].matrix();
	for (int j=0; j < 3; ++j)
	  for (int k=0; k <= j; ++k)
	    apeSaveFile<<"  "<<sm[j][k];
	apeSaveFile<<std::endl;
	}
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



