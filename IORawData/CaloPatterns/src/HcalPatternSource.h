#ifndef IORAWDATA_CALOPATTERNS_HCALPATTERNSOURCE_H
#define IORAWDATA_CALOPATTERNS_HCALPATTERNSOURCE_H 1

#include <vector>
#include "IORawData/CaloPatterns/interface/HcalFiberPattern.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"

class HcalElectronicsMap;

/** \class HcalPatternSource
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class HcalPatternSource : public edm::ConfigurableInputSource {
public:
  HcalPatternSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc);
protected:
  virtual void beginJob(edm::EventSetup const& es);
  virtual bool produce(edm::Event & e);
private:  
  void loadPatterns(const std::string& patspec);
  void loadPatternFile(const std::string& filename);
  std::vector<int> bunches_;
  std::vector<HcalFiberPattern> patterns_;
  const HcalElectronicsMap* elecmap_;
  int presamples_, samples_;
};

#endif
