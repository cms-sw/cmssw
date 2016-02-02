#ifndef __L1TMUON_DTCOLLECTOR_H__
#define __L1TMUON_DTCOLLECTOR_H__
// 
// Class: L1TMuon::DTCollector
//
// Info: Processes the DT digis into L1TMuon trigger primitives. 
//       Positional information is not assigned here.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include <memory>
#include "SubsystemCollector.h"
#include "DTBunchCrossingCleaner.h"
#include "FWCore/Utilities/interface/InputTag.h"

class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

namespace L1TMuon {

  class DTCollector: public SubsystemCollector {
  public:
    DTCollector(const edm::ParameterSet&);
    ~DTCollector() {}

    virtual void extractPrimitives(const edm::Event&, const edm::EventSetup&, 
				   std::vector<TriggerPrimitive>&) const;
  private:
    TriggerPrimitive processDigis(const L1MuDTChambPhDigi&,
				  const int &segment_number) const;
    TriggerPrimitive processDigis(const L1MuDTChambThDigi&,
				  const int bti_group) const;
    TriggerPrimitive processDigis(const L1MuDTChambPhDigi&,
				  const L1MuDTChambThDigi&,
				  const int bti_group) const;    
    int findBTIGroupForThetaDigi(const L1MuDTChambThDigi&,
				 const int position) const;    
    const int bx_min, bx_max;
    std::unique_ptr<DTBunchCrossingCleaner> _bxc;
  };
}

#endif
