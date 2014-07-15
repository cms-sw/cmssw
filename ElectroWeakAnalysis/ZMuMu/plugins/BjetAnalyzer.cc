#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include <iostream>
#include "TH1.h"

class BjetAnalysis : public edm::EDAnalyzer {
public:
  BjetAnalysis(const edm::ParameterSet & cfg);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  //  virtual void endJob();
private:
  edm::EDGetTokenT<reco::JetTagCollection> bTagToken_;
  edm::EDGetTokenT<reco::JetTagCollection> bTagToken2_;
  edm::EDGetTokenT<reco::JetTagCollection> bTagToken3_;
  std::vector<unsigned int> matched_, unMatched_;
  double  ptMin_, massMin_,massMax_,  etaMin_,  etaMax_,  trkIso_, chi2Cut_;
  int nHitCut_;
  TH1D *h_GlbMuNofHitsGlbMu_, *h_GlbMuChi2_, *h_TrkMuNofHitsGlbMu_, *h_GlbMuDxy_;

};

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
using namespace std;
using namespace reco;
using namespace edm;
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


BjetAnalysis::BjetAnalysis(const edm::ParameterSet & cfg) :
  bTagToken_(consumes<reco::JetTagCollection>(edm::InputTag("trackCountingHighEffBJetTags"))),
  bTagToken2_(consumes<reco::JetTagCollection>(edm::InputTag("jetProbabilityBJetTags"))),
  bTagToken3_(consumes<reco::JetTagCollection>(edm::InputTag("jetBProbabilityBJetTags"))),
  ptMin_(cfg.getUntrackedParameter<double>("ptMin")),
  massMin_(cfg.getUntrackedParameter<double>("massMin")),
  massMax_(cfg.getUntrackedParameter<double>("massMax")),
  etaMin_(cfg.getUntrackedParameter<double>("etaMin")),
  etaMax_(cfg.getUntrackedParameter<double>("etaMax")),
  trkIso_(cfg.getUntrackedParameter<double>("trkIso")),
  chi2Cut_(cfg.getUntrackedParameter<double>("chi2Cut")),
  nHitCut_(cfg.getUntrackedParameter<int>("nHitCut"))
{
  Service<TFileService> fs;
  TFileDirectory trackEffDir = fs->mkdir("QualityOfGlbMu");
  h_GlbMuNofHitsGlbMu_= trackEffDir.make<TH1D>("# of Hits of GlobalMuon", "# of Hits of GlobalMuon", 100, 0, 100);
  h_TrkMuNofHitsGlbMu_= trackEffDir.make<TH1D>("# of Hits of TrackerMuon", "# of Hits of TrackerMuon", 100, 0, 100);
  h_GlbMuChi2_= trackEffDir.make<TH1D>("chi2 of GlobalMuon", "chi2 of GlobalMuon", 100,0,10);
  h_GlbMuDxy_= trackEffDir.make<TH1D>("Dxy of GlobalMuon", "Dxy of GlobalMuon", 1000,-5.,5.);
}

void BjetAnalysis::analyze(const edm::Event& evt, const edm::EventSetup&) {


  // Get b tag information
  edm::Handle<reco::JetTagCollection> bTagHandle;
  evt.getByToken(bTagToken_, bTagHandle);
  const reco::JetTagCollection & bTags = *(bTagHandle.product());

  // Loop over jets and study b tag info.
  for (unsigned int i = 0; i != bTags.size(); ++i) {
    cout<<" Jet "<< i
	<<" has b tag discriminator (trackCountingHighEffBJetTags)= "<<bTags[i].second
	<< " and jet Pt = "<<bTags[i].first->pt()<<endl;
  }

  // Get b tag information
  edm::Handle<reco::JetTagCollection> bTagHandle2;
  evt.getByToken(bTagToken2_, bTagHandle2);
  const reco::JetTagCollection & bTags2 = *(bTagHandle2.product());

  // Loop over jets and study b tag info.
  for (unsigned int i = 0; i != bTags2.size(); ++i) {
    cout<<" Jet "<< i
	<<" has b tag discriminator (jetProbabilityBJetTags) = "<<bTags2[i].second
	<< " and jet Pt = "<<bTags2[i].first->pt()<<endl;
  }


  // Get b tag information
  edm::Handle<reco::JetTagCollection> bTagHandle3;
  evt.getByToken(bTagToken3_, bTagHandle3);
  const reco::JetTagCollection & bTags3 = *(bTagHandle3.product());

  // Loop over jets and study b tag info.
  for (unsigned int i = 0; i != bTags3.size(); ++i) {
    cout<<" Jet "<< i
	<<" has b tag discriminator (jetBProbabilityBJetTags) = "<<bTags3[i].second
	<< " and jet Pt = "<<bTags3[i].first->pt()<<endl;
  }










}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(BjetAnalysis);
