// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"

class L1TMuonOverlapFwVersionESProducer : public edm::ESProducer {
public:
  L1TMuonOverlapFwVersionESProducer(const edm::ParameterSet&);
  ~L1TMuonOverlapFwVersionESProducer() override;

  using ReturnType = std::unique_ptr<L1TMuonOverlapFwVersion>;

  ReturnType produceFwVersion(const L1TMuonOverlapFwVersionRcd&);

private:
  L1TMuonOverlapFwVersion params;
};
