//
// $Id: LeptonLRCalc.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_LeptonLRCalc_h
#define PhysicsTools_PatUtils_LeptonLRCalc_h

/**
  \class    LeptonLRCalc LeptonLRCalc.h "PhysicsTools/PatUtils/interface/LeptonLRCalc.h"
  \brief    Steering class for the combined lepton likelihood

   LeptonLRCalc allows to calculate and retrieve the combined lepton
   likelihood as originally defined in CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: LeptonLRCalc.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include "TF1.h"

#include <string>


namespace pat {


  class LeptonJetIsolationAngle;
  class LeptonVertexSignificance;


  class LeptonLRCalc {

    enum LeptonType { ElectronT, MuonT, TauT };

    public:

      LeptonLRCalc();
      LeptonLRCalc(const edm::EventSetup & iSetup, std::string electronLRFile = "", std::string muonLRFile = "", std::string tauLRFile = "");
      ~LeptonLRCalc();

      double getCalIsoE(const Electron & electron);
      double getCalIsoE(const Muon & muon);
      double getCalIsoE(const Tau & tau);
      double getTrIsoPt(const Electron & electron);
      double getTrIsoPt(const Muon & muon);
      double getTrIsoPt(const Tau & tau);
      double getLepId(const Electron & electron);
      double getLepId(const Muon & muon);
      double getLepId(const Tau & tau);
      double getLogPt(const Electron & electron);
      double getLogPt(const Muon & muon);
      double getLogPt(const Tau & tau);
      double getJetIsoA(const Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      double getJetIsoA(const Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      double getJetIsoA(const Tau & tau, const edm::Event & iEvent);
      double getVtxSign(const Electron & electron, const edm::Event & iEvent);
      double getVtxSign(const Muon & muon, const edm::Event & iEvent);
      double getVtxSign(const Tau & tau, const edm::Event & iEvent);
      double getCalIsoELR(double CalIsoE, LeptonType theType);
      double getTrIsoPtLR(double TrIsoPt, LeptonType theType);
      double getLepIdLR(double lepId, LeptonType theType);
      double getLogPtLR(double LogPt, LeptonType theType);
      double getJetIsoALR(double jetIsoA, LeptonType theType);
      double getVtxSignLR(double vtxSign, LeptonType theType);
      void calcLRVars(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLRVars(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLRVars(Tau & tau, const edm::Event & iEvent);
      void calcLRVals(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLRVals(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLRVals(Tau & tau, const edm::Event & iEvent);
      void calcLikelihood(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLikelihood(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      void calcLikelihood(Tau & tau, const edm::Event & iEvent);

    private:

      void readFitsElectron();
      void readFitsMuon();
      void readFitsTau();

    private:

      LeptonJetIsolationAngle  * theJetIsoACalc_;
      LeptonVertexSignificance * theVtxSignCalc_;

      std::string electronLRFile_;
      std::string muonLRFile_;
      std::string tauLRFile_;
      TF1 electronCalIsoEFit;
      TF1 electronTrIsoPtFit;
      TF1 electronLepIdFit;
      TF1 electronLogPtFit;
      TF1 electronJetIsoAFit;
      TF1 electronVtxSignFit;
      TF1 muonCalIsoEFit;
      TF1 muonTrIsoPtFit;
      TF1 muonLepIdFit;
      TF1 muonLogPtFit;
      TF1 muonJetIsoAFit;
      TF1 muonVtxSignFit;
      TF1 tauCalIsoEFit;
      TF1 tauTrIsoPtFit;
      TF1 tauLepIdFit;
      TF1 tauLogPtFit;
      TF1 tauJetIsoAFit;
      TF1 tauVtxSignFit;
      float electronFitMax_[6];
      float muonFitMax_[6];
      bool fitsElectronRead_;
      bool fitsMuonRead_;
      bool fitsTauRead_;

  };


}

#endif
