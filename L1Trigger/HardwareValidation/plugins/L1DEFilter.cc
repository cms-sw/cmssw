#include "L1Trigger/HardwareValidation/plugins/L1DEFilter.h"
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"
using dedefs::DEnsys;

L1DEFilter::L1DEFilter(const edm::ParameterSet& iConfig) : nEvt_{0}, nAgree_{0} {
  DEsource_ = iConfig.getParameter<edm::InputTag>("DataEmulCompareSource");
  flagSys_ = iConfig.getUntrackedParameter<std::vector<unsigned int> >("FlagSystems");
}

L1DEFilter::~L1DEFilter() {}

void L1DEFilter::endJob() {
  //compute rate of d|e disagreeing events
  double rate = (nEvt_ > 0) ? (double)nAgree_ / (double)nEvt_ : 0;
  std::cout << "[L1DEFilter] Data|Emul mismatch event rate: " << rate << std::endl;
}

// ------------ method called on each new Event  ------------
bool L1DEFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  nEvt_++;

  bool pass = true;

  edm::Handle<L1DataEmulRecord> deRecord;
  iEvent.getByLabel(DEsource_, deRecord);

  bool dematch = deRecord->get_status();
  if (dematch)
    nAgree_++;

  bool deMatch[DEnsys];
  deRecord->get_status(deMatch);

  for (int i = 0; i < DEnsys; i++)
    if (flagSys_[i])
      pass &= deMatch[i];

  return pass;
}
