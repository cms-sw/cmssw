#ifndef MuEnrichFltr_h
#define MuEnrichFltr_h
//
// class declaration
//
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <atomic>

class MuEnrichType1Filter : public edm::global::EDFilter<> {
public:
  explicit MuEnrichType1Filter(const edm::ParameterSet&);
  ~MuEnrichType1Filter();

private:
  void beginJob() override;
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;
  edm::EDGetTokenT<edm::HepMCProduct> theGenToken;
  mutable std::atomic<int> nrejected;
  mutable std::atomic<int> naccepted;
  int type;
  // ----------member data ---------------------------
};

#endif
