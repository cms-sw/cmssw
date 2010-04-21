#include <memory>
#include "TH2F.h"
#include "TGraph.h"
#include "TRandom.h"
#include "TLorentzVector.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

//TFile Service 
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// class declaration
//

class JetCorrectorDemo : public edm::EDAnalyzer {
public:
  explicit JetCorrectorDemo(const edm::ParameterSet&);
  ~JetCorrectorDemo();
  typedef reco::Particle::LorentzVector LorentzVector;
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 
  std::string mJetCorService;
  bool mDebug;
  int mNHistoPoints,mNGraphPoints;
  double mEtaMin,mEtaMax,mPtMin,mPtMax;
  std::vector<double> mVEta,mVPt;
  double vjec_eta[100][1000],vjec_pt[100][1000],vpt[100][1000],veta[100][1000];
  edm::Service<TFileService> fs;
  TH2F *mJECvsEta, *mJECvsPt;
  TGraph *mVGraphEta[100],*mVGraphPt[100];
  TRandom *mRandom;
};


JetCorrectorDemo::JetCorrectorDemo(const edm::ParameterSet& iConfig)
{
  mJetCorService = iConfig.getParameter<std::string>          ("JetCorrectionService");
  mNHistoPoints  = iConfig.getParameter<int>                  ("NHistoPoints");
  mNGraphPoints  = iConfig.getParameter<int>                  ("NGraphPoints");
  mEtaMin        = iConfig.getParameter<double>               ("EtaMin");
  mEtaMax        = iConfig.getParameter<double>               ("EtaMax");
  mPtMin         = iConfig.getParameter<double>               ("PtMin");
  mPtMax         = iConfig.getParameter<double>               ("PtMax");
  mVEta          = iConfig.getParameter<std::vector<double> > ("VEta");
  mVPt           = iConfig.getParameter<std::vector<double> > ("VPt");
  mDebug         = iConfig.getUntrackedParameter<bool>        ("Debug",false);
}


JetCorrectorDemo::~JetCorrectorDemo()
{
  
}

void JetCorrectorDemo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const JetCorrector* corrector = JetCorrector::getJetCorrector(mJetCorService,iSetup);
  double jec,pt,eta;
  TLorentzVector P4;
  double dEta = (mEtaMax-mEtaMin)/mNGraphPoints;
  double rPt  = pow(mPtMax/mPtMin,1./mNGraphPoints);
  for(int i=0;i<mNHistoPoints;i++)
    {
      pt  = mRandom->Uniform(mPtMin,mPtMax);
      eta = mRandom->Uniform(mEtaMin,mEtaMax);
      P4.SetPtEtaPhiE(pt,eta,0,0); 
      LorentzVector newP4(P4.Px(),P4.Py(),P4.Pz(),P4.E());
      jec = corrector->correction(newP4);
      mJECvsEta->Fill(eta,jec);
      mJECvsPt->Fill(pt,jec);
    }
  //--------- Eta Graphs ------------------
  for(unsigned ieta=0;ieta<mVEta.size();ieta++)
    {
      for(int i=0;i<mNGraphPoints;i++) 
        {
          pt  = mPtMin*pow(rPt,i);
          eta = mVEta[ieta];
          vpt[ieta][i] = pt;
          P4.SetPtEtaPhiE(pt,eta,0,0); 
          LorentzVector newP4(P4.Px(),P4.Py(),P4.Pz(),P4.E());
          jec = corrector->correction(newP4);
          vjec_eta[ieta][i] = jec;
          if (mDebug)
            std::cout<<pt<<" "<<eta<<" "<<jec<<std::endl;
        }
    }
  //--------- Pt Graphs -------------------
  for(unsigned ipt=0;ipt<mVPt.size();ipt++)
    {
      for(int i=0;i<mNGraphPoints;i++)
        {
          eta = mEtaMin + i*dEta;
          pt  = mVPt[ipt];
          veta[ipt][i] = eta;
          P4.SetPtEtaPhiE(pt,eta,0,0); 
          LorentzVector newP4(P4.Px(),P4.Py(),P4.Pz(),P4.E());
          jec = corrector->correction(newP4);
          vjec_pt[ipt][i] = jec;
          if (mDebug)
            std::cout<<pt<<" "<<eta<<" "<<jec<<std::endl;
        }
    }

}

void JetCorrectorDemo::beginJob()
{
  if (mNGraphPoints > 1000)
    throw  cms::Exception("JetCorrectorDemo","Too many graph points !!! Maximum is 1000 !!!");
  if (mVEta.size() > 100)
    throw  cms::Exception("JetCorrectorDemo","Too many eta values !!! Maximum is 100 !!!");
  if (mVPt.size() > 100)
    throw  cms::Exception("JetCorrectorDemo","Too many pt values !!! Maximum is 100 !!!");
  mJECvsEta = fs->make<TH2F>("JECvsEta","JECvsEta",200,mEtaMin,mEtaMax,100,0,5);
  mJECvsPt  = fs->make<TH2F>("JECvsPt","JECvsPt",200,mPtMin,mPtMax,100,0,5);
  mRandom   = new TRandom();
  mRandom->SetSeed(0);
}

void JetCorrectorDemo::endJob() 
{
  char name[1000];
  for(unsigned ipt=0;ipt<mVPt.size();ipt++)
    {
      mVGraphPt[ipt] = fs->make<TGraph>(mNGraphPoints,veta[ipt],vjec_pt[ipt]);
      sprintf(name,"JEC_vs_Eta_pt%1.1f",mVPt[ipt]);
      mVGraphPt[ipt]->SetName(name);
    }
  for(unsigned ieta=0;ieta<mVEta.size();ieta++)
    {
      mVGraphEta[ieta] = fs->make<TGraph>(mNGraphPoints,vpt[ieta],vjec_eta[ieta]);
      sprintf(name,"JEC_vs_Pt_eta%1.1f",mVEta[ieta]);
      mVGraphEta[ieta]->SetName(name);
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetCorrectorDemo);
