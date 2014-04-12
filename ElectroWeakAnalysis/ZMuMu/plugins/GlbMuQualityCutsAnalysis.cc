#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <iostream>
#include "TH1.h"

class GlbMuQualityCutsAnalysis : public edm::EDAnalyzer {
public:
  GlbMuQualityCutsAnalysis(const edm::ParameterSet & cfg);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  //  virtual void endJob();
private:
  edm::InputTag src_;
  edm::EDGetTokenT<reco::CandidateView> srcToken_;
  std::vector<unsigned int> matched_, unMatched_;
  double  ptMin_, massMin_,massMax_,  etaMin_,  etaMax_,  trkIso_, chi2Cut_;
  int nHitCut_;
  TH1D *h_GlbMuNofHitsGlbMu_, *h_GlbMuChi2_, *h_TrkMuNofHitsGlbMu_, *h_GlbMuDxy_;

};

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



GlbMuQualityCutsAnalysis::GlbMuQualityCutsAnalysis(const edm::ParameterSet & cfg) :
  src_(cfg.getParameter<InputTag>("src")),
  srcToken_(consumes<CandidateView>(src_)),
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

/*void GlbMuQualityCutsAnalysis::endJob() {
  cout << src_.encode() << endl ;

}
*/
void GlbMuQualityCutsAnalysis::analyze(const edm::Event& evt, const edm::EventSetup&) {
  Handle<CandidateView> src;
  evt.getByToken(srcToken_, src);
  int j=0;
  cout << ">> entries in " << src_ << ": " << src->size() << endl;
  for(CandidateView::const_iterator i = src->begin(); i != src->end(); ++i) {
    j++;
    const Candidate * dau1 = i->daughter(0);
    const Candidate * dau2 = i->daughter(1);
    if(dau1 == 0|| dau2 == 0)
      throw Exception(errors::InvalidReference) <<
	"one of the two daughter does not exist\n";
    const Candidate * c1 = dau1->masterClone().get();
    GenParticleRef mc1;
    const pat::Muon * mu1 = dynamic_cast<const pat::Muon*>(c1);
  std::cout << " dimuon mass  " << i->mass() << std::endl;
     std::cout << "dau1.pt() " << dau1->pt() << std::endl;
    std::cout << "dau2.pt() " << dau2->pt() << std::endl;
    std::cout << "dau1.isGlobalMuon() " << dau1->isGlobalMuon() << std::endl;
    std::cout << "dau2.isGlobalMuon() " << dau2->isGlobalMuon()<< std::endl;
    std::cout << "dau1.isTrackerMuon() " << dau1->isTrackerMuon() << std::endl;
    std::cout << "dau2.isTrackerlMuon() " << dau2->isTrackerMuon()<< std::endl;
    std::cout << "dau1.isStandAloneMuon() " << dau1->isStandAloneMuon() << std::endl;
    std::cout << "dau2.isStandAloneMuon() " << dau2->isStandAloneMuon()<< std::endl;
    std::cout << "dau1.charge() " << dau1->charge() << std::endl;
    std::cout << "dau2.charge() " << dau2->charge()<< std::endl;
  if(mu1 != 0) {

      //     if (mc1.isNonnull()) cout << "GlbMuQualityCutsAnalysis> genParticleRef1 " << mc1->pdgId() << endl;
      // double trackIso1=mu1->trackIso();
      // std::cout << " mu1 iso" << trackIso1 << std::endl;
    } else {
      const pat::GenericParticle * gp1 = dynamic_cast<const pat::GenericParticle*>(c1);
      if(gp1 == 0)
	throw Exception(errors::InvalidReference) <<
	  "first of two daughter is neither a pat::Muon not pat::GenericParticle\n";

    }
    const Candidate * c2 = dau2->masterClone().get();

    const pat::Muon * mu2 = dynamic_cast<const pat::Muon*>(c2);
    if(mu2 != 0) {

      // double trackIso2=mu2->trackIso();
     //std::cout << " mu2 iso" << trackIso2 << std::endl;
    } else {
      const pat::GenericParticle * gp2 = dynamic_cast<const pat::GenericParticle*>(c2);
      if(gp2 == 0)
	throw Exception(errors::InvalidReference) <<
	  "first of two daughter is neither a pat::Muon not pat::GenericParticle\n";

    }

    int nOfHit_1=0, nOfHit_tk_1=0, nOfHit_2=0, nOfHit_tk_2=0;
    if (mu1->isGlobalMuon() )   nOfHit_1= mu1->numberOfValidHits();
    std::cout << "n of hit of GlbMu1: " << nOfHit_1 << std::endl;
    if (mu1->isTrackerMuon() ) nOfHit_tk_1= mu1->innerTrack()->numberOfValidHits();
    std::cout << "n of hit of TrkMu1: " << nOfHit_tk_1 << std::endl;
    if (mu2->isGlobalMuon() ) nOfHit_2= mu2->numberOfValidHits();
    std::cout << "n of hit of GlbMu2: " << nOfHit_2 << std::endl;
    if (mu2->isTrackerMuon() ) nOfHit_tk_2= mu2->innerTrack()->numberOfValidHits();
    std::cout << "n of hit of TrkMu2: " << nOfHit_tk_2 << std::endl;
    h_GlbMuNofHitsGlbMu_->Fill(nOfHit_1);
    h_GlbMuNofHitsGlbMu_->Fill(nOfHit_2);
    h_TrkMuNofHitsGlbMu_->Fill(nOfHit_tk_1);
    h_TrkMuNofHitsGlbMu_->Fill(nOfHit_tk_2);
    double nChi2_1=0, nChi2_2=0;
    if (mu1->isGlobalMuon() ) nChi2_1= mu1->normChi2();
    std::cout << "chi2 of GlbMu1: " << nChi2_1 << std::endl;
    if (mu2->isGlobalMuon() )  nChi2_2= mu2->normChi2();
    std::cout << "chi2 of GlbMu2: " << nChi2_2 << std::endl;
    h_GlbMuChi2_->Fill(nChi2_1);
    h_GlbMuChi2_->Fill(nChi2_2);
    double dxy_1= mu1->dB();
    double dxy_2= mu2->dB();

    h_GlbMuDxy_->Fill(dxy_1);
    h_GlbMuDxy_->Fill(dxy_2);
    if (mu1->isGlobalMuon() && ( nOfHit_tk_1<nHitCut_)) {
      std::cout<<"found a GlbMuon with nOfHit " << nOfHit_tk_1 << ", it has eta: " << mu1->eta()<< std::endl;
  }
    if ( mu2->isGlobalMuon() && ( nOfHit_tk_2<nHitCut_)) {
      std::cout<<"found a GlbMuon with nOfHit " << nOfHit_tk_2 << ", it has eta: " << mu2->eta()<< std::endl;
  }
    if (mu1->isGlobalMuon() && ( nChi2_1 >chi2Cut_)) {
      std::cout<<"found a GlbMuon with chi2 " << nChi2_1 << ", it has chi2 of track: " << mu1->innerTrack()->normalizedChi2()<< ", and chi2 of Sta: "<<  mu1->outerTrack()->normalizedChi2() << ", eta: "<< mu1->eta()<< ",pt: "<< mu1->pt()<< std::endl;
    }
    if (mu2->isGlobalMuon() && ( nChi2_2 >chi2Cut_)) {
      std::cout<<"found a GlbMuon with chi2 " << nChi2_2 << ", it has chi2 of track: " << mu2->innerTrack()->normalizedChi2()<< ", and chi2 of Sta: "<<  mu2->outerTrack()->normalizedChi2() << ", eta:  "<<mu2->eta()<< ",pt: "<< mu2->pt()<< std::endl;
    }

  }
}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GlbMuQualityCutsAnalysis);
