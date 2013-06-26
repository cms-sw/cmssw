 /**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke/Ivan Reid
 *  date       : February 2009 *  $Revision: 1.12 $
 *  $Date: 2013/01/07 20:56:25 $
 *  (last update by $Author: wmtan $)
 */
/*
 *# Parameters:
 *#    saveApeToASCII -- Do we write out an APE text file?
 *#    saveComposites -- Do we write APEs for composite detectors?
 *#    saveLocalNotGlobal -- Do we write the APEs in the local or global coordinates?
 *#    apeASCIISaveFile -- The name of the save-file.
 *#    readApeFromASCII -- Do we read in APEs from a text file?
 *#    readLocalNotGlobal -- Do we read APEs in the local or the global frame?
 *#    readFullLocalMatrix -- Do we read the full local matrix or just the diagonal elements?
 *#                        -- Always write full matrix
 *# Full matrix format: DetID dxx dxy dyy dxz dyz dzz
 *# Diagonal element format: DetID sqrt(dxx) sqrt(dyy) sqrt(dzz)
 *#    setComposites -- Do we set the APEs for composite detectors or just ignore them?
 *#    apeASCIIReadFile -- Input file name.
 *# Also note:
 *#    process.AlignmentProducer.saveApeToDB -- to save as an sqlite file
 *# and associated entries in _cfg.py
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
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class ApeSettingAlgorithm : public AlignmentAlgorithmBase
{
 public:
  /// Constructor
  ApeSettingAlgorithm(const edm::ParameterSet &cfg);

  /// Destructor
  virtual ~ApeSettingAlgorithm();

  /// Call at beginning of job
  virtual void initialize(const edm::EventSetup &setup, 
			  AlignableTracker *tracker, AlignableMuon *muon, AlignableExtras *extras,
			  AlignmentParameterStore *store);

  /// Call at end of job
  virtual void terminate(const edm::EventSetup& iSetup);

  /// Run the algorithm
  virtual void run(const edm::EventSetup &setup, const EventInfo &eventInfo);

 private:
  edm::ParameterSet         theConfig;
  AlignableNavigator       *theAlignableNavigator;
  AlignableTracker         *theTracker;
  bool                     saveApeToAscii_,readApeFromAscii_,readFullLocalMatrix_;
  bool                     readLocalNotGlobal_,saveLocalNotGlobal_;
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
  saveLocalNotGlobal_ = theConfig.getUntrackedParameter<bool>("saveLocalNotGlobal");
  readApeFromAscii_ = theConfig.getParameter<bool>("readApeFromASCII");
  readLocalNotGlobal_ = theConfig.getParameter<bool>("readLocalNotGlobal");
  readFullLocalMatrix_ = theConfig.getParameter<bool>("readFullLocalMatrix");
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
				     AlignableTracker *tracker, AlignableMuon *muon, AlignableExtras *extras,
				     AlignmentParameterStore *store)
{
 theAlignableNavigator = new AlignableNavigator(tracker, muon);
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
     if (!readLocalNotGlobal_ || readFullLocalMatrix_) 
       { apeReadFile>>apeId>>x11>>x21>>x22>>x31>>x32>>x33>>std::ws;}
     else
       { apeReadFile>>apeId>>x11>>x22>>x33>>std::ws;}
     //idr What sanity checks do we need to put here?
     if (apeId != 0) //read appears valid?
       { if (apeList.find(apeId) == apeList.end()) //Not previously done
	 {  DetId id(apeId);
	 AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(id)); //NULL if none
	 if (alidet)
	   { if ((alidet->components().size()<1) || setComposites_) //the problem with glued dets...
	     { GlobalError globErr;
	     if (readLocalNotGlobal_)
	       { AlgebraicSymMatrix33 as; 
	       if (readFullLocalMatrix_)
		 { as[0][0]=x11; as[1][0]=x21; as[1][1]=x22;
		 as[2][0]=x31; as[2][1]=x32; as[2][2]=x33;
		 }
	       else
	         { as[0][0]=x11*x11; as[1][1]=x22*x22; as[2][2]=x33*x33;} //local cov.
	       align::RotationType rt=alidet->globalRotation();
	       AlgebraicMatrix33 am;
	       am[0][0]=rt.xx(); am[0][1]=rt.xy(); am[0][2]=rt.xz();
	       am[1][0]=rt.yx(); am[1][1]=rt.yy(); am[1][2]=rt.yz();
	       am[2][0]=rt.zx(); am[2][1]=rt.zy(); am[2][2]=rt.zz();
	       globErr = GlobalError(ROOT::Math::SimilarityT(am,as));
	       }
	     else
	       {
		 globErr = GlobalError(x11,x21,x22,x31,x32,x33);
	       }
	     alidet->setAlignmentPositionError(globErr, false); // do not propagate down!
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
     }
   apeReadFile.close();
   edm::LogInfo("Alignment") << "@SUB=initialize" << "Set "<<apeList.size()<<" APE values.";
   }
}
 

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void ApeSettingAlgorithm::terminate(const edm::EventSetup& iSetup)
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
	CLHEP::HepSymMatrix sm = aliErr->m_alignError[i].matrix();
	if (saveLocalNotGlobal_)
	  { align::RotationType rt=alidet->globalRotation();
	  AlgebraicMatrix am(3,3);
	  am[0][0]=rt.xx(); am[0][1]=rt.xy(); am[0][2]=rt.xz();
	  am[1][0]=rt.yx(); am[1][1]=rt.yy(); am[1][2]=rt.yz();
	  am[2][0]=rt.zx(); am[2][1]=rt.zy(); am[2][2]=rt.zz();
	  sm=sm.similarity(am); //symmetric matrix
	  } //transform to local
	for (int j=0; j < 3; ++j)
	  for (int k=0; k <= j; ++k)
	    apeSaveFile<<"  "<<sm[j][k]; //always write full matrix
	
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
void ApeSettingAlgorithm::run(const edm::EventSetup &setup, const EventInfo &eventInfo)
{
  // nothing to do here?
}

// Plugin definition for the algorithm
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory,
		   ApeSettingAlgorithm, "ApeSettingAlgorithm");



