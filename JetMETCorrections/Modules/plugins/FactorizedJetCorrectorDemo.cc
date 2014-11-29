#include <memory>
#include "TH2F.h"
#include "TGraph.h"
#include "TGraphErrors.h"
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
#include "DataFormats/Candidate/interface/Particle.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// class declaration
//

class FactorizedJetCorrectorDemo : public edm::EDAnalyzer {
public:
  explicit FactorizedJetCorrectorDemo(const edm::ParameterSet&);
  ~FactorizedJetCorrectorDemo();
  typedef reco::Particle::LorentzVector LorentzVector;

private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  std::string mJetCorService,mPayloadName,mUncertaintyTag,mUncertaintyFile;
  std::vector<std::string> mLevels;
  bool mDebug,mUseCondDB;
  int mNHistoPoints,mNGraphPoints;
  double mEtaMin,mEtaMax,mPtMin,mPtMax;
  std::vector<double> mVEta,mVPt;
  double vjec_eta[100][1000],vjec_pt[100][1000],vpt[100][1000],vptcor[100][1000],veta[100][1000];
  double vjecUnc_eta[100][1000],vUnc_eta[100][1000],vjecUnc_pt[100][1000],vUnc_pt[100][1000],vex_eta[100][1000],vex_pt[100][1000];
  edm::Service<TFileService> fs;
  TH2F *mJECvsEta, *mJECvsPt;
  TGraphErrors *mVGraphEta[100],*mVGraphPt[100],*mVGraphCorPt[100];
  TGraph *mUncEta[100], *mUncCorPt[100];
  TRandom *mRandom;
};
//
//----------- Class Implementation ------------------------------------------
//
//---------------------------------------------------------------------------
FactorizedJetCorrectorDemo::FactorizedJetCorrectorDemo(const edm::ParameterSet& iConfig)
{
  mLevels            = iConfig.getParameter<std::vector<std::string> > ("levels");
  mPayloadName       = iConfig.getParameter<std::string>          ("PayloadName");
  mUncertaintyTag    = iConfig.getParameter<std::string>          ("UncertaintyTag");
  mUncertaintyFile   = iConfig.getParameter<std::string>          ("UncertaintyFile");
  mNHistoPoints      = iConfig.getParameter<int>                  ("NHistoPoints");
  mNGraphPoints      = iConfig.getParameter<int>                  ("NGraphPoints");
  mEtaMin            = iConfig.getParameter<double>               ("EtaMin");
  mEtaMax            = iConfig.getParameter<double>               ("EtaMax");
  mPtMin             = iConfig.getParameter<double>               ("PtMin");
  mPtMax             = iConfig.getParameter<double>               ("PtMax");
  mVEta              = iConfig.getParameter<std::vector<double> > ("VEta");
  mVPt               = iConfig.getParameter<std::vector<double> > ("VPt");
  mDebug             = iConfig.getUntrackedParameter<bool>        ("Debug",false);
}
//---------------------------------------------------------------------------
FactorizedJetCorrectorDemo::~FactorizedJetCorrectorDemo()
{

}
//---------------------------------------------------------------------------
void FactorizedJetCorrectorDemo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if ( mDebug )
    std::cout << "Hello from FactorizedJetCorrectorDemo" << std::endl;
  // retreive parameters from the DB this still need a proper configurable
  // payloadName like: JetCorrectorParametersCollection_Spring10_AK5Calo.
  edm::ESHandle<JetCorrectorParametersCollection> parameters;
  iSetup.get<JetCorrectionsRecord>().get(mPayloadName, parameters);

  std::vector<JetCorrectorParameters> params;
  for(std::vector<std::string>::const_iterator level=mLevels.begin(); level!=mLevels.end(); ++level){
    const JetCorrectorParameters& ip = (*parameters)[*level]; //ip.printScreen();
    if ( mDebug )
      std::cout << "Adding level " << *level << std::endl;
    params.push_back(ip);
  }

  boost::shared_ptr<FactorizedJetCorrector> corrector ( new FactorizedJetCorrector(params));


  double jec,rawPt,corPt,eta;
  TLorentzVector P4;
  double dEta = (mEtaMax-mEtaMin)/mNGraphPoints;
  if ( mDebug )
    std::cout << "Making JEC vs Eta and pT" << std::endl;
  for(int i=0;i<mNHistoPoints;i++)
    {
      rawPt  = mRandom->Uniform(mPtMin,mPtMax);
      eta = mRandom->Uniform(mEtaMin,mEtaMax);
      P4.SetPtEtaPhiE(rawPt,eta,0,0);
      corrector->setJetEta( eta );
      corrector->setJetPt( rawPt );
      corrector->setJetE( P4.E() );
      jec = corrector->getCorrection();
      mJECvsEta->Fill(eta,jec);
      mJECvsPt->Fill(rawPt,jec);
    }
  if ( mDebug )
    std::cout << "Making JEC vs pT for different etas" << std::endl;
  //--------- Pt Graphs ------------------
  for(unsigned ieta=0;ieta<mVEta.size();ieta++)
    {
      double rPt  = pow((3500./TMath::CosH(mVEta[ieta]))/mPtMin,1./mNGraphPoints);
      for(int i=0;i<mNGraphPoints;i++)
        {
          rawPt  = mPtMin*pow(rPt,i);
          eta = mVEta[ieta];
          vpt[ieta][i] = rawPt;
          P4.SetPtEtaPhiE(rawPt,eta,0,0);
	  corrector->setJetEta( eta );
	  corrector->setJetPt( rawPt );
	  corrector->setJetE( P4.E() );
	  jec = corrector->getCorrection();
          vjec_eta[ieta][i] = jec;
          vptcor[ieta][i] = rawPt*jec;
          vex_eta[ieta][i] = 0.0;
          if (mDebug)
            std::cout<<rawPt<<" "<<eta<<" "<<jec<<" "<<rawPt*jec<<std::endl;
        }
    }
  if ( mDebug )
    std::cout << "Making JEC vs eta for different pTs" << std::endl;
  //--------- Eta Graphs -------------------
  for(unsigned ipt=0;ipt<mVPt.size();ipt++)
    {
      for(int i=0;i<mNGraphPoints;i++)
        {
          eta = mEtaMin + i*dEta;
          corPt  = mVPt[ipt];
          veta[ipt][i] = eta;
          //---------- find the raw pt -----------
          double e = 1.0;
          int nLoop(0);
          rawPt = corPt;
          while(e > 0.0001 && nLoop < 10)
             {
               P4.SetPtEtaPhiE(rawPt,eta,0,0);
               LorentzVector rawP4(P4.Px(),P4.Py(),P4.Pz(),P4.E());
	       corrector->setJetEta( eta );
	       corrector->setJetPt( rawPt );
	       corrector->setJetE( P4.E() );
	       jec = corrector->getCorrection();
               double tmp = rawPt * jec;
               e = fabs(tmp-corPt)/corPt;
               if (jec > 0)
                 rawPt = corPt/jec;
               nLoop++;
             }
          //--------- calculate the jec for the rawPt --------
          P4.SetPtEtaPhiE(rawPt,eta,0,0);
          LorentzVector rawP4(P4.Px(),P4.Py(),P4.Pz(),P4.E());
	  corrector->setJetEta( eta );
	  corrector->setJetPt( rawPt );
	  corrector->setJetE( P4.E() );
	  jec = corrector->getCorrection();
          vjec_pt[ipt][i] = jec;
          if (mDebug)
            std::cout<<rawPt<<" "<<eta<<" "<<jec<<" "<<rawPt*jec<<std::endl;
        }
    }
  if ( mDebug )
    std::cout << "See ya!" << std::endl;

}
//---------------------------------------------------------------------------
void FactorizedJetCorrectorDemo::beginJob()
{
  if (mNGraphPoints > 1000)
    throw  cms::Exception("FactorizedJetCorrectorDemo","Too many graph points !!! Maximum is 1000 !!!");
  if (mVEta.size() > 100)
    throw  cms::Exception("FactorizedJetCorrectorDemo","Too many eta values !!! Maximum is 100 !!!");
  if (mVPt.size() > 100)
    throw  cms::Exception("FactorizedJetCorrectorDemo","Too many pt values !!! Maximum is 100 !!!");
  mJECvsEta = fs->make<TH2F>("JECvsEta","JECvsEta",200,mEtaMin,mEtaMax,100,0,5);
  mJECvsPt  = fs->make<TH2F>("JECvsPt","JECvsPt",200,mPtMin,mPtMax,100,0,5);
  mRandom   = new TRandom();
  mRandom->SetSeed(0);
}
//---------------------------------------------------------------------------
void FactorizedJetCorrectorDemo::endJob()
{
  char name[1000];
  for(unsigned ipt=0;ipt<mVPt.size();ipt++)
    {
      mVGraphEta[ipt] = fs->make<TGraphErrors>(mNGraphPoints,veta[ipt],vjec_pt[ipt],vex_pt[ipt],vjecUnc_pt[ipt]);
      sprintf(name,"JEC_vs_Eta_CorPt%1.1f",mVPt[ipt]);
      mVGraphEta[ipt]->SetName(name);
      mUncEta[ipt] = fs->make<TGraph>(mNGraphPoints,veta[ipt],vUnc_pt[ipt]);
      sprintf(name,"UNC_vs_Eta_CorPt%1.1f",mVPt[ipt]);
      mUncEta[ipt]->SetName(name);
    }
  for(unsigned ieta=0;ieta<mVEta.size();ieta++)
    {
      mVGraphPt[ieta] = fs->make<TGraphErrors>(mNGraphPoints,vpt[ieta],vjec_eta[ieta],vex_eta[ieta],vjecUnc_eta[ieta]);
      sprintf(name,"JEC_vs_RawPt_eta%1.1f",mVEta[ieta]);
      mVGraphPt[ieta]->SetName(name);
      mVGraphCorPt[ieta] = fs->make<TGraphErrors>(mNGraphPoints,vptcor[ieta],vjec_eta[ieta],vex_eta[ieta],vjecUnc_eta[ieta]);
      sprintf(name,"JEC_vs_CorPt_eta%1.1f",mVEta[ieta]);
      mVGraphCorPt[ieta]->SetName(name);
      mUncCorPt[ieta] = fs->make<TGraph>(mNGraphPoints,vptcor[ieta],vUnc_eta[ieta]);
      sprintf(name,"UNC_vs_CorPt_eta%1.1f",mVEta[ieta]);
      mUncCorPt[ieta]->SetName(name);
    }
}
//---------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(FactorizedJetCorrectorDemo);
























