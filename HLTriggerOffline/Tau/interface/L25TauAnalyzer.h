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
// $Id: L25TauAnalyzer.h,v 1.6 2008/10/03 19:09:11 bachtis Exp $
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
      float l25SignalTrkPt;
      float l25SignalTrkChi2NdF;
      float l25SignalTrkChi2;
      int l25SignalTrkNValidHits;
      int l25SignalTrkNRecHits;
      int l25SignalTrkNValidPixelHits;
      int l25SignalTrkNLostHits;
      float l25SignalTrkDxy;
      float l25SignalTrkDz;
      float l25SignalTrkEta;
      float l25SignalTrkPhi;
      float l25IsoTrkPt;
      float l25IsoTrkChi2NdF;
      float l25IsoTrkChi2;
      int l25IsoTrkNValidHits;
      int l25IsoTrkNRecHits;
      int l25IsoTrkNValidPixelHits;
      int l25IsoTrkNLostHits;
      float l25IsoTrkDxy;
      float l25IsoTrkDz;
      float l25IsoTrkEta;
      float l25IsoTrkPhi;
      float l25JetEt;
      float l25JetEta;
      float l25JetPhi;
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
