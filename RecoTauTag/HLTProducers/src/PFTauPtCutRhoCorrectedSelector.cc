#include "RecoTauTag/HLTProducers/interface/PFTauPtCutRhoCorrectedSelector.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

PFTauPtCutRhoCorrectedSelector::PFTauPtCutRhoCorrectedSelector(const edm::ParameterSet& pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  srcRho_ = pset.getParameter<edm::InputTag>("srcRho");
  effectiveArea_ = pset.getParameter<double>("effectiveArea");
  minPt_ = pset.getParameter<double>("minPt");
  filter_ = pset.getParameter<bool>("filter");
  /*code*/
  /*code*/
  produces<reco::PFTauCollection>();
}
bool PFTauPtCutRhoCorrectedSelector::filter(edm::Event& evt, const edm::EventSetup& es) {
  std::auto_ptr<reco::PFTauCollection> output(new reco::PFTauCollection);

  edm::Handle<reco::PFTauCollection> taus;
  evt.getByLabel(src_, taus);

  edm::Handle<double> rho;
  evt.getByLabel(srcRho_, rho);

  for (size_t i = 0; i < taus->size(); ++i) {
    const reco::PFTau& theTau = taus->at(i);
    double correctedPt = theTau.pt() - (*rho)*(effectiveArea_);
    if (correctedPt > minPt_) {
      output->push_back(theTau);
    }
  }
  size_t outputSize = output->size();
  evt.put(output);
  if (filter_)
    return outputSize;
  else
    return true;
}
