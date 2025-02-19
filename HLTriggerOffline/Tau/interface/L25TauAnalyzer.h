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
// $Id: L25TauAnalyzer.h,v 1.10 2011/03/01 22:54:26 eluiggi Exp $
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
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "HepMC/GenParticle.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TLorentzVector.h"
#include <vector>
#include <string>
#include <TTree.h>
#include <TFile.h>
#include "TH1.h"


class L25TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L25TauAnalyzer(const edm::ParameterSet&);
      ~L25TauAnalyzer();

   private:

      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      reco::PFTau match(const reco::Jet&, const reco::PFTauCollection&);
      reco::CaloJet matchedToPFTau(const reco::PFTau&, const reco::L2TauInfoAssociation&);
      void printInfo(const reco::PFTau& thePFTau, const reco::IsolatedTauTagInfo& theTauTagInfo);
      void clearVectors();
      void initializeVectors();
      edm::InputTag _l25JetSource;
      edm::InputTag _l2TauInfoAssoc;
      edm::InputTag _pfTauSource;
      edm::InputTag _pVtxSource;
      edm::InputTag _pfTauIsoSource;
      edm::InputTag _pfTauMuonDiscSource;
      math::XYZPoint theVertexPosition;

      bool signal_;
      float _minTrackPt;
      float _signalCone;
      float _isolationCone;
      float _l2l25MatchingCone;
      float _l25JetLeadTkMacthingCone;
      float _l25Dz;
      float _l25LeadTkPtMin;
      int _nTrkIso;

      TTree *l25tree;
      
      int numPixTrkInJet;
      int numQPixTrkInJet;
      int numQPixTrkInSignalCone;
      int numQPixTrkInAnnulus;
      int myNtrkIso;
      
      float l25JetEt;
      float l25JetEta;
      float l25JetPhi;
      
      std::vector<float> *l25TrkPt;
      std::vector<float> *l25TrkEta;
      std::vector<float> *l25TrkPhi;
      std::vector<float> *l25TrkDz;
      std::vector<float> *l25TrkDxy;
      std::vector<float> *l25TrkChi2;
      std::vector<float> *l25TrkChi2NdF;
      std::vector<float> *l25TrkNRecHits;
      std::vector<float> *l25TrkNValidPixelHits;
      
      std::vector<float> *l25SignalTrkPt;
      std::vector<float> *l25SignalTrkChi2NdF;
      std::vector<float> *l25SignalTrkChi2;
      std::vector<float> *l25SignalTrkDxy;
      std::vector<float> *l25SignalTrkDz;
      std::vector<float> *l25SignalTrkEta;
      std::vector<float> *l25SignalTrkPhi;
      std::vector<int> *l25SignalTrkNValidHits;
      std::vector<int> *l25SignalTrkNRecHits;
      std::vector<int> *l25SignalTrkNValidPixelHits;
      std::vector<int> *l25SignalTrkNLostHits;
      
      std::vector<float> *l25IsoTrkPt;
      std::vector<float> *l25IsoTrkChi2NdF;
      std::vector<float> *l25IsoTrkChi2;
      std::vector<float> *l25IsoTrkDxy;
      std::vector<float> *l25IsoTrkDz;
      std::vector<float> *l25IsoTrkEta;
      std::vector<float> *l25IsoTrkPhi;
      std::vector<int> *l25IsoTrkNValidHits;
      std::vector<int> *l25IsoTrkNRecHits;
      std::vector<int> *l25IsoTrkNValidPixelHits;
      std::vector<int> *l25IsoTrkNLostHits;

      bool hasLeadTrk;
      float leadSignalTrackPt;
      float leadTrkJetDeltaR;
      float pftauL25DeltaR;
      
      bool pfTauHasLeadTrk;
      bool pfTauInBounds;
      float pfTauIsoDisc;
      float pfTauMuonDisc;
      float pfTauElecDisc;
      float pfTauEt;
      float pfTauPt;
      float pfTauEta; 
      float pfTauPhi; 
      float pfTauLTPt;
      int pfTauNProngs;
      float pfTauTrkIso;
      float pfTauGammaIso;
      int pfTauNTrkIso;
      float pfTauIsoTrkPt;
      float leadTrkPtRes;
      float leadTrkDeltaR;
      float leadIsoTrkPtRes;
      float leadIsoTrkDeltaR;
      float pftauSignalTrkDeltaR;
      float pftauIsoTrkDeltaR;
      
      float l2JetEt;
      float l2JetEta;
      float l2JetPhi;
      
      bool l25MatchedToPFTau;
      bool L2MatchedToPFtau;
      bool L25MatchedToL2;
      
      bool l25Disc_LeadTkDir;
      bool l25Disc_JetDir;
      bool l25Disc_Trk5_IsoPtMin2_NTrk0;
};
