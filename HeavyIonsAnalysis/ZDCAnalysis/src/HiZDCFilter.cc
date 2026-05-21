// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h" // Required by making plug-in
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HiZDCFilter : public edm::one::EDFilter<> {
public:
  explicit HiZDCFilter(const edm::ParameterSet&);
  ~HiZDCFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<edm::SortedCollection<ZDCRecHit>> ZDCRecHitToken_;
  const double ltPlus_, gtPlus_;
  const double ltMinus_, gtMinus_;
  std::string algo_;

  float sumPlus, sumMinus;
  bool has_algo(std::string key) { return (algo_.find(key) != std::string::npos); }
};

HiZDCFilter::HiZDCFilter(const edm::ParameterSet& iConfig) :
  ZDCRecHitToken_(consumes<edm::SortedCollection<ZDCRecHit>>(iConfig.getParameter<edm::InputTag>("ZDCRecHitSource"))),
  ltPlus_(iConfig.getParameter<double>("threshold4ltPlus")),
  gtPlus_(iConfig.getParameter<double>("threshold4gtPlus")),
  ltMinus_(iConfig.getParameter<double>("threshold4ltMinus")),
  gtMinus_(iConfig.getParameter<double>("threshold4gtMinus")),
  algo_(iConfig.getParameter<std::string>("algorithm")) {
  std::transform(algo_.begin(), algo_.end(), algo_.begin(),
                 [](unsigned char c){ return std::tolower(c); });
}

HiZDCFilter::~HiZDCFilter() {}

bool HiZDCFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;
  sumPlus = 0;
  sumMinus = 0;

  const auto& zdcrechits = iEvent.get(ZDCRecHitToken_);

  for (auto const& rh : zdcrechits) { 
    HcalZDCDetId zdcid = rh.id();
    int zside = zdcid.zside();
    int section = zdcid.section();
    float energy = rh.energy();

    if (!(section == 1 || section == 2)) continue; // only count EM and HAD
    if (section == 1 && zdcid.channel() > 5) continue; // ignore extra EM channels

    if (zside < 0)
      sumMinus += energy;
    if (zside > 0)
      sumPlus += energy;
  }

  if (has_algo("lt") && has_algo("or")) {
    accepted = (sumPlus <= ltPlus_ || sumMinus <= ltMinus_);
  } else if (has_algo("lt") && has_algo("and")) {
    accepted = (sumPlus <= ltPlus_ && sumMinus <= ltMinus_);
  } else if (has_algo("gt") && has_algo("or")) {
    accepted = (sumPlus >= gtPlus_ || sumMinus >= gtMinus_);
  } else if (has_algo("gt") && has_algo("and")) {
    accepted = (sumPlus >= gtPlus_ && sumMinus >= gtMinus_);
  } else if (has_algo("xor")) {
    accepted = (sumPlus >= gtPlus_ && sumMinus <= ltMinus_)
      || (sumPlus <= ltPlus_ && sumMinus >= gtMinus_);
  } else {
    std::cout<<"error: bad algorithm "<<algo_<<std::endl;
  }

  return accepted;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiZDCFilter);
