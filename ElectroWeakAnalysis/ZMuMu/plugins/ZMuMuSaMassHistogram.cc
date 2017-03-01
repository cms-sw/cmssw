#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "TH1.h"
#include "TH2.h"
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;



class ZMuMuSaMassHistogram : public edm::EDAnalyzer {
public:
  typedef math::XYZVector Vector;
  ZMuMuSaMassHistogram(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;
  EDGetTokenT<CandidateView> srcToken_;
  int counter;
  double min, max;
  int Nbin;
  TH1F * ZMassSa;
  void histo(TH1F* hist, char* cx, char* cy) const;
};

void ZMuMuSaMassHistogram::histo(TH1F* hist,char* cx, char*cy) const{
  hist->GetXaxis()->SetTitle(cx);
  hist->GetYaxis()->SetTitle(cy);
  hist->GetXaxis()->SetTitleOffset(1);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetTitleSize(0.04);
  hist->GetYaxis()->SetTitleSize(0.04);
  hist->GetXaxis()->SetLabelSize(0.03);
  hist->GetYaxis()->SetLabelSize(0.03);
}


ZMuMuSaMassHistogram::ZMuMuSaMassHistogram(const ParameterSet& pset):
  srcToken_(consumes<CandidateView>(pset.getParameter<InputTag>("src_m"))),
  counter(0),
  min(pset.getUntrackedParameter<double>("min")),
  max(pset.getUntrackedParameter<double>("max")),
  Nbin(pset.getUntrackedParameter<int>("nbin")) {
  edm::Service<TFileService> fs;
  ZMassSa = fs->make<TH1F>("zMass","ZMass OneStandAlone (GeV/c^{2})",Nbin,min,max);

}


void ZMuMuSaMassHistogram::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateView> dimuons;
  event.getByToken(srcToken_, dimuons);
  for(unsigned int i=0; i< dimuons->size(); ++ i ) {
    const Candidate & zmm = (* dimuons)[i];
    const Candidate * dau0 = zmm.daughter(0);
    const Candidate * dau1 = zmm.daughter(1);
    TrackRef stAloneTrack;
    Candidate::PolarLorentzVector p4_0;
    double mu_mass;
    if(counter % 2 == 0) {
      stAloneTrack = dau0->get<TrackRef,reco::StandAloneMuonTag>();
      p4_0 = dau1->polarP4();
      mu_mass = dau0->mass();
    }
    else{
      stAloneTrack = dau1->get<TrackRef,reco::StandAloneMuonTag>();
      p4_0= dau0->polarP4();
      mu_mass = dau1->mass();
    }

    Vector momentum = stAloneTrack->momentum();
    Candidate::PolarLorentzVector p4_1(momentum.rho(), momentum.eta(),momentum.phi(), mu_mass);
    double mass = (p4_0+p4_1).mass();
    ZMassSa->Fill(mass);
    ++counter;

  }


}


void ZMuMuSaMassHistogram::endJob() {
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuSaMassHistogram);
