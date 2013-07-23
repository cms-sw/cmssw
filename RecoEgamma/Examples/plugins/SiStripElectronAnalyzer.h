#ifndef RecoEgamma_Examples_SiStripElectronAnalyzer_h
#define RecoEgamma_Examples_SiStripElectronAnalyzer_h
// -*- C++ -*-
//
// Package:     RecoEgamma/Examples
// Class  :     SiStripElectronAnalyzer
//
/**\class SiStripElectronAnalyzer SiStripElectronAnalyzer.h RecoEgamma/Examples/interface/SiStripElectronAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Fri May 26 16:52:45 EDT 2006
// $Id: SiStripElectronAnalyzer.h,v 1.3 2009/03/06 12:42:16 chamont Exp $
//

// system include files
#include <memory>
#include <map>
#include <math.h>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "TFile.h"
#include "TH1F.h"

#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"

// forward declarations

#define myMaxHits 1000

//
// class decleration
//

class SiStripElectronAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiStripElectronAnalyzer(const edm::ParameterSet&);
      ~SiStripElectronAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void initNtuple ( void ) ;
      virtual void endJob( void );

   private:
      double unwrapPhi(double phi) const {
	while (phi > M_PI) { phi -= 2.*M_PI; }
	while (phi < -M_PI) { phi += 2.*M_PI; }
	return phi;
      }


      // ----------member data ---------------------------
      std::string fileName_;


      TFile* file_;
      TH1F* numCand_;
      TH1F* numElectrons_;
      TH1F* numSuperClusters_;
      TH1F* energySuperClusters_ ;
      TH1F* sizeSuperClusters_;
      TH1F* emaxSuperClusters_;
      TH1F* phiWidthSuperClusters_;

      TH1F* energySuperClustersEl_ ;
      TH1F* sizeSuperClustersEl_;
      TH1F* emaxSuperClustersEl_;
      TH1F* phiWidthSuperClustersEl_;

      TH1F* ptDiff ;
      TH1F* pDiff ;
      TH1F* pElectronFailed ;
      TH1F* ptElectronFailed ;
      TH1F* pElectronPassed ;
      TH1F* ptElectronPassed ;
      TH1F* sizeSuperClustersPassed ;
      TH1F* sizeSuperClustersFailed ;
      TH1F* energySuperClustersPassed ;
      TH1F* energySuperClustersFailed ;
      TH1F* eOverPFailed ;
      TH1F* eOverPPassed ;


      TH1F* numSiStereoHits_;
      TH1F* numSiMonoHits_;
      TH1F* numSiMatchedHits_;

      TTree* myTree_;

      int NShowers_ ;
      float EShower_[myMaxHits] ;
      float XShower_[myMaxHits] ;
      float YShower_[myMaxHits] ;
      float ZShower_[myMaxHits] ;

      int NStereoHits_ ;
      float StereoHitX_[myMaxHits] ;
      float StereoHitY_[myMaxHits] ;
      float StereoHitZ_[myMaxHits] ;

      float StereoHitR_[myMaxHits];
      float StereoHitPhi_[myMaxHits];
      float StereoHitTheta_[myMaxHits];

      // errors in local coords
      float StereoHitSigX_[myMaxHits] ;
      float StereoHitSigY_[myMaxHits] ;
      float StereoHitCorr_[myMaxHits] ;

      float StereoHitSignal_[myMaxHits];
      float StereoHitNoise_[myMaxHits];
      int StereoHitWidth_[myMaxHits];

      int StereoDetector_[myMaxHits];
      int StereoLayer_[myMaxHits];

     // mono corresponds to "rphi" only hits
      int NMonoHits_ ;
      float MonoHitX_[myMaxHits] ;
      float MonoHitY_[myMaxHits] ;
      float MonoHitZ_[myMaxHits] ;

      float MonoHitR_[myMaxHits];
      float MonoHitPhi_[myMaxHits];
      float MonoHitTheta_[myMaxHits];

      // errors in local coords
      float MonoHitSigX_[myMaxHits] ;
      float MonoHitSigY_[myMaxHits] ;
      float MonoHitCorr_[myMaxHits] ;

      float MonoHitSignal_[myMaxHits];
      float MonoHitNoise_[myMaxHits];
      int MonoHitWidth_[myMaxHits];

      int MonoDetector_[myMaxHits];
      int MonoLayer_[myMaxHits];

     // matched hits
      int NMatchedHits_ ;
      float MatchedHitX_[myMaxHits] ;
      float MatchedHitY_[myMaxHits] ;
      float MatchedHitZ_[myMaxHits] ;

      float MatchedHitR_[myMaxHits];
      float MatchedHitPhi_[myMaxHits];
      float MatchedHitTheta_[myMaxHits];

      // errors in local coords
      float MatchedHitSigX_[myMaxHits] ;
      float MatchedHitSigY_[myMaxHits] ;
      float MatchedHitCorr_[myMaxHits] ;

      float MatchedHitSignal_[myMaxHits];
      float MatchedHitNoise_[myMaxHits];
      int MatchedHitWidth_[myMaxHits];

      int MatchedDetector_[myMaxHits];
      int MatchedLayer_[myMaxHits];


      std::string mctruthProducer_;
      std::string mctruthCollection_;
      std::string superClusterProducer_;
      std::string superClusterCollection_;
      std::string basicClusterProducer_;
      std::string basicClusterCollection_;
      std::string eBRecHitProducer_;
      std::string eBRecHitCollection_;
      std::string siElectronProducer_;
      std::string siElectronCollection_;
      std::string electronProducer_;
      std::string electronCollection_;
      std::string siHitProducer_;
      std::string siRphiHitCollection_;
      std::string siStereoHitCollection_;
      std::string siMatchedHitCollection_;
};

#endif
