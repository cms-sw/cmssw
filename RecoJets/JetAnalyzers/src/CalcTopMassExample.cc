//
// Example to calculated top mass from genjets corrected on the fly with parton calibration 
// Top mass is calculated from: 
// 1) Uncorrected jets
// 2) Corrected jets with flavour ttbar correction assuming jet mass = quark mass
// 3) The same as 3 assuming massless jets
// 4) Corrected jets with mixed ttbar correction assuming massless jets
// 
// Author: Attilio Santocchia
// Date: 15.06.2009
// Tested on CMSSW_3_1_0
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/getRef.h"

#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/MatchedPartons.h"
#include "SimDataFormats/JetMatching/interface/JetMatchedPartons.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TLorentzVector.h"


// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>

class calcTopMass : public edm::EDAnalyzer {
  public:
    explicit calcTopMass(const edm::ParameterSet & );
    ~calcTopMass() {};
    virtual void analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup);
  private:
    edm::InputTag sourcePartons_;
    edm::InputTag sourceByRefer_;
    edm::Handle<reco::JetMatchedPartonsCollection> theJetPartonMatch;

    std::string m_qT_CorrectorName;
    std::string m_cT_CorrectorName;
    std::string m_bT_CorrectorName;
    std::string m_tT_CorrectorName;

    float bMass;
    float cMass;
    float qMass;
    
    TH1F *hMassNoCorr;
    TH1F *hMassCorFl0;
    TH1F *hMassCorFlM;
    TH1F *hMassCorMix;
};

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

calcTopMass::calcTopMass(const edm::ParameterSet& iConfig)
{

  sourceByRefer_ = iConfig.getParameter<InputTag> ("srcByReference");
  m_qT_CorrectorName = iConfig.getParameter <std::string> ("qTopCorrector");
  m_cT_CorrectorName = iConfig.getParameter <std::string> ("cTopCorrector");
  m_bT_CorrectorName = iConfig.getParameter <std::string> ("bTopCorrector");
  m_tT_CorrectorName = iConfig.getParameter <std::string> ("tTopCorrector");

  bMass = 4.5;
  cMass = 1.5;
  qMass = 0.3;

  Service<TFileService> fs;
  hMassNoCorr = fs->make<TH1F>("hMassNoCorr","",100,100,300);
  hMassCorFl0 = fs->make<TH1F>("hMassCorFl0","",100,100,300);
  hMassCorFlM = fs->make<TH1F>("hMassCorFlM","",100,100,300);
  hMassCorMix = fs->make<TH1F>("hMassCorMix","",100,100,300);

}

void calcTopMass::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[calcTopMass] analysing event " << iEvent.id() << endl;
  try {
    iEvent.getByLabel (sourceByRefer_ , theJetPartonMatch   );
  } catch(std::exception& ce) {
    cerr << "[calcTopMass] caught std::exception " << ce.what() << endl;
    return;
  }

  // get all correctors from top events
  const JetCorrector* qTopCorrector = JetCorrector::getJetCorrector (m_qT_CorrectorName, iSetup);
  const JetCorrector* cTopCorrector = JetCorrector::getJetCorrector (m_cT_CorrectorName, iSetup);
  const JetCorrector* bTopCorrector = JetCorrector::getJetCorrector (m_bT_CorrectorName, iSetup);
  const JetCorrector* tTopCorrector = JetCorrector::getJetCorrector (m_tT_CorrectorName, iSetup);

  TLorentzVector jetsNoCorr[6];
  TLorentzVector jetsCorFl0[6];
  TLorentzVector jetsCorFlM[6];
  TLorentzVector jetsCorMix[6];

  bool isQuarkFound[6] = {false};

  for ( JetMatchedPartonsCollection::const_iterator j  = theJetPartonMatch->begin();
                                     j != theJetPartonMatch->end();
                                     j ++ ) {

    const math::XYZTLorentzVector theJet = (*j).first.get()->p4();
    const MatchedPartons aMatch = (*j).second;
    const GenParticleRef thePhyDef = aMatch.physicsDefinitionParton() ;
    
    if(thePhyDef.isNonnull()) {

      int particPDG = thePhyDef.get()->pdgId();

      if(         particPDG ==  5 ) { //b from t
        double bTcorr = bTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[0].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[0].SetPtEtaPhiM( theJet.pt()*bTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[0].SetPtEtaPhiM( theJet.pt()*bTcorr , theJet.eta() , theJet.phi() , bMass         );
        jetsCorMix[0].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[0]=true;
      } else if ( particPDG == -5 ) { //bbar from tbar
        double bTcorr = bTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[3].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[3].SetPtEtaPhiM( theJet.pt()*bTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[3].SetPtEtaPhiM( theJet.pt()*bTcorr , theJet.eta() , theJet.phi() , bMass         );
        jetsCorMix[3].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[3]=true;
      } else if ( particPDG == 2 ) { //W+ from t
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[1].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[1].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[1].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[1].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[1]=true;
      } else if ( particPDG == 4 ) { //W+ from t
        double cTcorr = cTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[1].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[1].SetPtEtaPhiM( theJet.pt()*cTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[1].SetPtEtaPhiM( theJet.pt()*cTcorr , theJet.eta() , theJet.phi() , cMass         );
        jetsCorMix[1].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[1]=true;
      } else if ( particPDG == -1 ) { //W+ from t
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[2].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[2].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[2].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[2].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[2]=true;
      } else if ( particPDG == -3 ) { //W+ from t
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[2].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[2].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[2].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[2].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[2]=true;
      } else if ( particPDG == -2 ) { //W- from tbar
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[4].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[4].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[4].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[4].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[4]=true;
      } else if ( particPDG == -4 ) { //W- from tbar
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[4].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[4].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[4].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[4].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[4]=true;
      } else if ( particPDG == 1 ) { //W- from tbar
        double qTcorr = qTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[5].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[5].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[5].SetPtEtaPhiM( theJet.pt()*qTcorr , theJet.eta() , theJet.phi() , qMass         );
        jetsCorMix[5].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[5]=true;
      } else if ( particPDG == 3 ) { //W- from tbar
        double cTcorr = cTopCorrector->correction (theJet);
        double tTcorr = tTopCorrector->correction (theJet);
        jetsNoCorr[5].SetPtEtaPhiM( theJet.pt()        , theJet.eta() , theJet.phi() , theJet.mass() );
        jetsCorFl0[5].SetPtEtaPhiM( theJet.pt()*cTcorr , theJet.eta() , theJet.phi() , 0             );
        jetsCorFlM[5].SetPtEtaPhiM( theJet.pt()*cTcorr , theJet.eta() , theJet.phi() , cMass         );
        jetsCorMix[5].SetPtEtaPhiM( theJet.pt()*tTcorr , theJet.eta() , theJet.phi() , 0             );
        isQuarkFound[5]=true;
      }
    }
  }

  if( isQuarkFound[0] && isQuarkFound[1] && isQuarkFound[2] ) {
    TLorentzVector *theTopPNoCorr = new TLorentzVector( jetsNoCorr[0] + jetsNoCorr[1] + jetsNoCorr[2] );
    TLorentzVector *theTopPCorFl0 = new TLorentzVector( jetsCorFl0[0] + jetsCorFl0[1] + jetsCorFl0[2] );
    TLorentzVector *theTopPCorFlM = new TLorentzVector( jetsCorFlM[0] + jetsCorFlM[1] + jetsCorFlM[2] );
    TLorentzVector *theTopPCorMix = new TLorentzVector( jetsCorMix[0] + jetsCorMix[1] + jetsCorMix[2] );
    hMassNoCorr->Fill( theTopPNoCorr->M() );
    hMassCorFl0->Fill( theTopPCorFl0->M() );
    hMassCorFlM->Fill( theTopPCorFlM->M() );
    hMassCorMix->Fill( theTopPCorMix->M() );
  }

  if( isQuarkFound[3] && isQuarkFound[4] && isQuarkFound[5] ) {
    TLorentzVector *theTopMNoCorr = new TLorentzVector( jetsNoCorr[3] + jetsNoCorr[4] + jetsNoCorr[5] );
    TLorentzVector *theTopMCorFl0 = new TLorentzVector( jetsCorFl0[3] + jetsCorFl0[4] + jetsCorFl0[5] );
    TLorentzVector *theTopMCorFlM = new TLorentzVector( jetsCorFlM[3] + jetsCorFlM[4] + jetsCorFlM[5] );
    TLorentzVector *theTopMCorMix = new TLorentzVector( jetsCorMix[3] + jetsCorMix[4] + jetsCorMix[5] );
    hMassNoCorr->Fill( theTopMNoCorr->M() );
    hMassCorFl0->Fill( theTopMCorFl0->M() );
    hMassCorFlM->Fill( theTopMCorFlM->M() );
    hMassCorMix->Fill( theTopMCorMix->M() );
  }

}

DEFINE_FWK_MODULE( calcTopMass );
