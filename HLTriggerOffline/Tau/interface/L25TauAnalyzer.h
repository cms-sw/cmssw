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
// $Id$
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
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "TLorentzVector.h"
#include <vector>
#include <string>
#include <TTree.h>
#include <TFile.h>
#include "TH1D.h"
#include "TH1.h"
#include "TH1F.h"
#
//

  typedef math::XYZTLorentzVectorD   LV;
  typedef std::vector<LV>            LVColl;

// class decleration
//

class L25TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L25TauAnalyzer(const edm::ParameterSet&);
      ~L25TauAnalyzer();

   private:

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool match(const LV& recoJet, const LVColl& matchingObject);       
      //bool match(const CaloJet& caloJet, const LVColl& matchingObject);      
      
      edm::InputTag jetTagSrc_;
      edm::InputTag jetMCTagSrc_;
      edm::InputTag caloJets_;
      std::string rootFile_;	   //Output File Name
      
      int nTracksInIsolationRing_;
      float rMatch_;
      float rSig_;
      float rIso_;
      float minPtIsoRing_;
      float ptLeadTk_;
      float mcMatch_;
      bool signal_;
      
      TFile *l25file;		   //File to store the histos...
      TTree *l25tree;
      
      //TH1F *matchedJetsPt;

      int isolated;
      int ecalIsoJets;
      int numPixTrkInJet;
      int numQPixTrkInJet;
      int numCaloJets;
      float jetPt;
      float jetE;
      float jetEta;
      float jetPhi;	
      float leadSignalTrackPt;
      float leadTrkJetDeltaR; 
      bool l2match;
      bool l25match;
      bool hasLeadTrk;
};
