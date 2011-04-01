// -*- C++ -*-
//
// Package:    L25TauAnalyzer
// Class:      L25TauAnalyzer
// 
/**\class L25TauAnalyzer L25TauAnalyzer.cc HLTriggerOffline/Tau/src/L25TauAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Eduardo Luiggi
//         Created:  Fri Apr  4 16:37:44 CDT 2008
// $Id: L25TauAnalyzer.h,v 1.7 2009/12/18 20:44:55 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "HepMC/GenParticle.h"
#include "TLorentzVector.h"
#include <vector>
#include <string>
#include <TTree.h>
#include <TFile.h>
#include "TH1.h"


  typedef math::XYZTLorentzVectorD   LV;
  typedef std::vector<LV>            LVColl;




  struct MatchElementL25 {
    bool matched;
    double deltar;
    double mcEta;
    double mcEt;

  };

class L25TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L25TauAnalyzer(const edm::ParameterSet&);
      ~L25TauAnalyzer();

   private:

      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      MatchElementL25 match(const reco::Jet&,const LVColl&);
      float trackDrRMS(const reco::IsolatedTauTagInfo&,const reco::TrackRefVector&);

      edm::InputTag jetTagSrc_;
      edm::InputTag jetMCTagSrc_;
      std::string rootFile_;
      bool signal_;
      float minTrackPt_;
      float signalCone_;
      float isolationCone_;


      TFile *l25file;
      TTree *l25tree;

      int numPixTrkInJet;
      int numQPixTrkInJet;
      int numQPixTrkInSignalCone;
      int numQPixTrkInAnnulus;
      float jetEt;
      float jetEta;
      float jetMCEt;
      float jetMCEta;
      float trkDrRMS;
      float trkDrRMSA;
      float leadSignalTrackPt;
      float leadTrkJetDeltaR; 
  float emf;
      bool hasLeadTrk;
};
