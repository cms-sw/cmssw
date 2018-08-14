#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "L1Trigger/L1TNtuples/interface/L1TPFMetNoMuProducer.h"

L1TPFMetNoMuProducer::L1TPFMetNoMuProducer(const edm::ParameterSet &ps)
    : thePFMETCollection_(consumes<reco::PFMETCollection>(
          ps.getParameter<edm::InputTag>("pfMETCollection"))),
      theMuonCollection_(consumes<reco::MuonCollection>(
          ps.getParameter<edm::InputTag>("muonCollection"))) {
  produces<reco::PFMETCollection>();
}

void L1TPFMetNoMuProducer::produce(edm::Event &event,
                                   const edm::EventSetup &eventSetup) {
  edm::Handle<reco::PFMETCollection> pfMet;
  event.getByToken(thePFMETCollection_, pfMet);

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(theMuonCollection_, muons);

  if (!pfMet.isValid()) {
    edm::LogWarning("L1TPFMetNoMuProducer")
        << "invalid collection for pfMet" << std::endl;
    return;
  }
  if (!muons.isValid()) {
    edm::LogWarning("L1TPFMetNoMuProducer")
        << "invalid collection for muons" << std::endl;
    return;
  }

  reco::PFMET thePFMetNoMu = pfMet.product()->front();
  double pfMetNoMuPx = thePFMetNoMu.px();
  double pfMetNoMuPy = thePFMetNoMu.py();

  double muPx(0.), muPy(0.);

  for (auto muon = muons->begin(); muon != muons->end(); ++muon) {
    if (muon->isPFMuon()) {
      muPx += muon->px();
      muPy += muon->py();
    }
  }

  pfMetNoMuPx += muPx;
  pfMetNoMuPy += muPy;
  math::XYZTLorentzVector pfMetNoMuP4(pfMetNoMuPx, pfMetNoMuPy, 0,
                                      hypot(pfMetNoMuPx, pfMetNoMuPy));

  thePFMetNoMu.setP4(pfMetNoMuP4);

  std::unique_ptr<reco::PFMETCollection> product(new reco::PFMETCollection);
  product->emplace_back(thePFMetNoMu.getSpecific(), thePFMetNoMu.sumEt(),
                        thePFMetNoMu.p4(), thePFMetNoMu.vertex());

  event.put(std::move(product));
}

DEFINE_FWK_MODULE(L1TPFMetNoMuProducer);
