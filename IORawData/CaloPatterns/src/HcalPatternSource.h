#ifndef IORAWDATA_CALOPATTERNS_HCALPATTERNSOURCE_H
#define IORAWDATA_CALOPATTERNS_HCALPATTERNSOURCE_H 1

#include <vector>
#include "IORawData/CaloPatterns/interface/HcalFiberPattern.h"
#include "FWCore/Framework/interface/EDProducer.h"

/** \class HcalPatternSource
  *  
  * \author J. Mans - Minnesota
  */
class HcalPatternSource : public edm::EDProducer {
public:
  HcalPatternSource(const edm::ParameterSet & pset);
  void produce(edm::Event& e, const edm::EventSetup& c) override;
private:  
  void loadPatterns(const std::string& patspec);
  void loadPatternFile(const std::string& filename);
  std::vector<int> bunches_;
  std::vector<HcalFiberPattern> patterns_;
  int presamples_, samples_;
};

#endif
