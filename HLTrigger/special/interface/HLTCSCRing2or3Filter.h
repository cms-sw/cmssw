#ifndef HLTCSCRing2or3Filter_h
#define HLTCSCRing2or3Filter_h

#include <vector>
#include <map>

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TH1F.h"

class HLTCSCRing2or3Filter : public HLTFilter {

 public:
  explicit HLTCSCRing2or3Filter(const edm::ParameterSet&);
  ~HLTCSCRing2or3Filter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag m_input;
  unsigned int m_minHits;
  double m_xWindow, m_yWindow;
};

#endif 
