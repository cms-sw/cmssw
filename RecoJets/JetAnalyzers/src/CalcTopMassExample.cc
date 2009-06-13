// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:
    edm::InputTag sourcePartons_;
    edm::InputTag sourceByRefer_;
    edm::Handle<reco::JetMatchedPartonsCollection> theJetPartonMatch;

    std::string m_qJ_CorrectorName;
    std::string m_cJ_CorrectorName;
    std::string m_bJ_CorrectorName;
    std::string m_jJ_CorrectorName;
    std::string m_qT_CorrectorName;
    std::string m_cT_CorrectorName;
    std::string m_bT_CorrectorName;
    std::string m_tT_CorrectorName;

    TLorentzVector *jetsNoCorr[6];
    TLorentzVector *jetsCorFl0[6];
    TLorentzVector *jetsCorFlM[6];
    TLorentzVector *jetsCorMix[6];

    float bMass;
    float cMass;
    float qMass;
    
};

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

calcTopMass::calcTopMass(const edm::ParameterSet& iConfig)
{
  sourceByRefer_ = iConfig.getParameter<InputTag> ("srcByReference");
  m_qJ_CorrectorName = iConfig.getParameter <std::string> ("qJetCorrector");
  m_cJ_CorrectorName = iConfig.getParameter <std::string> ("cJetCorrector");
  m_bJ_CorrectorName = iConfig.getParameter <std::string> ("bJetCorrector");
  m_jJ_CorrectorName = iConfig.getParameter <std::string> ("jJetCorrector");
  m_qT_CorrectorName = iConfig.getParameter <std::string> ("qTopCorrector");
  m_cT_CorrectorName = iConfig.getParameter <std::string> ("cTopCorrector");
  m_bT_CorrectorName = iConfig.getParameter <std::string> ("bTopCorrector");
  m_tT_CorrectorName = iConfig.getParameter <std::string> ("tTopCorrector");

  bMass = 4.5;
  cMass = 1.5;
  qMass = 0.3;

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

  // get all correctors from diJet events
  const JetCorrector* qJetCorrector = JetCorrector::getJetCorrector (m_qJ_CorrectorName, iSetup);
  const JetCorrector* cJetCorrector = JetCorrector::getJetCorrector (m_cJ_CorrectorName, iSetup);
  const JetCorrector* bJetCorrector = JetCorrector::getJetCorrector (m_bJ_CorrectorName, iSetup);
  const JetCorrector* jJetCorrector = JetCorrector::getJetCorrector (m_jJ_CorrectorName, iSetup);

  // get all correctors from top events
  const JetCorrector* qTopCorrector = JetCorrector::getJetCorrector (m_qT_CorrectorName, iSetup);
  const JetCorrector* cTopCorrector = JetCorrector::getJetCorrector (m_cT_CorrectorName, iSetup);
  const JetCorrector* bTopCorrector = JetCorrector::getJetCorrector (m_bT_CorrectorName, iSetup);
  const JetCorrector* tTopCorrector = JetCorrector::getJetCorrector (m_tT_CorrectorName, iSetup);

  // Jet Parton Matching with status=3 partons using the Physics Definition from bTag package
  cout << "-------------------- Jet Partons Matching --------------" << endl;
  for ( JetMatchedPartonsCollection::const_iterator j  = theJetPartonMatch->begin();
                                     j != theJetPartonMatch->end();
                                     j ++ ) {
    const Jet *aJet       = (*j).first.get();
    const math::XYZTLorentzVector bJet = aJet->p4();
    const MatchedPartons aMatch = (*j).second;
    const GenParticleRef thePhyDef = aMatch.physicsDefinitionParton() ;
    if(thePhyDef.isNonnull()) {
        const Candidate *theMother =  thePhyDef.get()->mother(0);
        int motherPDG = theMother->pdgId();
        int particPDG = thePhyDef.get()->pdgId();
        if( particPDG == 5 ) { //b
          double bJcorr = bJetCorrector->correction (bJet);
          double jJcorr = jJetCorrector->correction (bJet);
          double bTcorr = bTopCorrector->correction (bJet);
          double tTcorr = tTopCorrector->correction (bJet);
          jetsNoCorr[0]->SetPtEtaPhiM( aJet->pt()        , aJet->eta() , aJet->phi() , aJet->mass() );
          jetsCorFl0[0]->SetPtEtaPhiM( aJet->pt()*bTcorr , aJet->eta() , aJet->phi() , 0            );
          jetsCorFlM[0]->SetPtEtaPhiM( aJet->pt()*bTcorr , aJet->eta() , aJet->phi() , bMass        );
          jetsCorMix[0]->SetPtEtaPhiM( aJet->pt()*tTcorr , aJet->eta() , aJet->phi() , 0            );
        }
    }
    printf("[calcTopMass] (pt,eta,phi) jet = %7.2f %6.3f %6.3f \n",
             aJet->et(),
             aJet->eta(),
             aJet->phi()
          );
  }
}

DEFINE_FWK_MODULE( calcTopMass );
