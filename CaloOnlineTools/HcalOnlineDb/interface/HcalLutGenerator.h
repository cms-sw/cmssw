#ifndef HcalLutGenerator_h
#define HcalLutGenerator_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

class HcalLutGenerator : public edm::EDAnalyzer {
public:
  explicit HcalLutGenerator(const edm::ParameterSet&);
  ~HcalLutGenerator();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  HcalTopologyMode::Mode m_mode;
  int m_maxDepthHB;
  int m_maxDepthHE;
  std::string _tag;
  std::string _lin_file;
  uint32_t    _status_word_to_mask;
};


#endif
