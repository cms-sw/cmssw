// Class:      L25TauAnalyzer
// Original Author:  Eduardo Luiggi, modified by Sho Maruyama
//         Created:  Fri Apr  4 16:37:44 CDT 2008
// $Id: L25TauAnalyzer.h,v 1.3 2008/05/15 19:16:59 eluiggi Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TH1F.h"
#include "TH2F.h"

class L25TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L25TauAnalyzer(const edm::ParameterSet&);
      ~L25TauAnalyzer();
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
    
      edm::InputTag   tauSource;
      edm::InputTag l25JetSource;
      edm::InputTag l25PtCutSource;
      edm::InputTag l25IsoSource;
      TH1F* tauPt;
      TH1F* tauInvPt;
      TH1F* tauEt;
      TH1F* tauEta;
      TH1F* tauPhi;
      TH1F* tauTjDR;
      TH1F* tauTrkC05;
      TH1F* tauTrkIso;
      TH1F* tauTrkSig;
      TH1F* l25Et;
      TH1F* l25Phi;
      TH1F* l25Eta;
      TH1F* l25Pt;
      TH1F* l25PtCut;
      TH1F* l25InvPt;
      TH1F* l25Iso;
      TH1F* l25TjDR;
      TH1F* l25TrkQPx;
      TH1F* leadDR;
      TH2F* Pt;
      TH2F* Et;
      TH2F* Eta;
      TH2F* Phi;
      TH2F* TjDR;
      TH2F* TrkC05;
      TH2F* TrkSig;
      TH1F* effInvPt;
      TH1F* effPtCut;
      TH1F* effIso;
      TH1F* effDR;
      TH1F* matchDR;
      double matchingCone;
};
