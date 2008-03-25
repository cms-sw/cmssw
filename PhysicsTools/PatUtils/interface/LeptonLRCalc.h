//
// $Id: LeptonLRCalc.h,v 1.3 2008/01/21 16:26:19 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_LeptonLRCalc_h
#define PhysicsTools_PatUtils_LeptonLRCalc_h

/**
  \class    pat::LeptonLRCalc LeptonLRCalc.h "PhysicsTools/PatUtils/interface/LeptonLRCalc.h"
  \brief    Steering class for the combined lepton likelihood

   LeptonLRCalc allows to calculate and retrieve the combined lepton
   likelihood as originally defined in CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: LeptonLRCalc.h,v 1.3 2008/01/21 16:26:19 lowette Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"

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

      double calIsoE(const Electron & electron);
      double calIsoE(const Muon & muon);
      double calIsoE(const Tau & tau);
      double trIsoPt(const Electron & electron);
      double trIsoPt(const Muon & muon);
      double trIsoPt(const Tau & tau);
      double lepId(const Electron & electron);
      double lepId(const Muon & muon);
      double lepId(const Tau & tau);
      double logPt(const Electron & electron);
      double logPt(const Muon & muon);
      double logPt(const Tau & tau);
      double jetIsoA(const Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      double jetIsoA(const Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      double jetIsoA(const Tau & tau, const edm::Event & iEvent);
      double vtxSign(const Electron & electron, const edm::Event & iEvent);
      double vtxSign(const Muon & muon, const edm::Event & iEvent);
      double vtxSign(const Tau & tau, const edm::Event & iEvent);
      double calIsoELR(double CalIsoE, const LeptonType & theType);
      double trIsoPtLR(double TrIsoPt, const LeptonType & theType);
      double lepIdLR(double lepId, const LeptonType & theType);
      double logPtLR(double LogPt, const LeptonType & theType);
      double jetIsoALR(double jetIsoA, const LeptonType & theType);
      double vtxSignLR(double vtxSign, const LeptonType & theType);
      void calcLRVars(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      void calcLRVars(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      void calcLRVars(Tau & tau, const edm::Event & iEvent);
      void calcLRVals(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      void calcLRVals(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      void calcLRVals(Tau & tau, const edm::Event & iEvent);
      void calcLikelihood(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
      void calcLikelihood(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent);
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
