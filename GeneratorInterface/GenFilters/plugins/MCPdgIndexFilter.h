#ifndef MCPdgIndexFilter_h
#define MCPdgIndexFilter_h
/*
 Description: filter events based on the particle PDG ID at a given
 index in the HepMC::GenEvent record.

 Original Author: Burt Betchart, 2013/08/09
*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class HepMCProduct;
}

class MCPdgIndexFilter : public edm::global::EDFilter<> {
public:
  explicit MCPdgIndexFilter(const edm::ParameterSet&);
  ~MCPdgIndexFilter() override{};

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  bool pass(const edm::Event&) const;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const std::vector<int> pdgID;
  const std::vector<unsigned> index;
  edm::EDPutTokenT<bool> putToken_;
  const unsigned maxIndex;
  const bool taggingMode;
};
#endif
