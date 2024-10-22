// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//
// class declaration
//

class PileUpFilter : public edm::global::EDFilter<> {
public:
  explicit PileUpFilter(const edm::ParameterSet&);
  ~PileUpFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
private:
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puSummaryInfoToken_;
  double minPU_;
  double maxPU_;
  bool useTrueNumInteraction_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PileUpFilter::PileUpFilter(const edm::ParameterSet& iConfig)
    : puSummaryInfoToken_(
          consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("pileupInfoSummaryInputTag"))),
      minPU_(iConfig.getParameter<double>("minPU")),
      maxPU_(iConfig.getParameter<double>("maxPU")),
      useTrueNumInteraction_(iConfig.getUntrackedParameter<bool>("useTrueNumInteraction", true)) {
  // now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool PileUpFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  bool pass = false;

  edm::Handle<std::vector<PileupSummaryInfo>> puSummaryInfoHandle;
  if (iEvent.getByToken(puSummaryInfoToken_, puSummaryInfoHandle)) {
    for (PileupSummaryInfo const& pileup : *puSummaryInfoHandle) {
      // only use the in-time pileup
      if (pileup.getBunchCrossing() == 0) {
        // use the per-event in-time pileup
        double pu = (useTrueNumInteraction_ ? pileup.getTrueNumInteractions() : pileup.getPU_NumInteractions());
        if (pu >= minPU_ and pu < maxPU_)
          pass = true;
      }
    }
  }

  return pass;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PileUpFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pileupInfoSummaryInputTag", edm::InputTag("PileupSummaryInfo"));
  desc.add<double>("minPU", 0.);
  desc.add<double>("maxPU", 80.);
  desc.addUntracked<bool>("useTrueNumInteraction", true);

  descriptions.add("pileupFilter", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PileUpFilter);
