#ifndef PhysicsTools_SelectorUtils_IsolationCutApplicatorBase_h
#define PhysicsTools_SelectorUtils_IsolationCutApplicatorBase_h

//
//
//

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <unordered_map>

class IsolationCutApplicatorBase : public CutApplicatorBase {
 public:  
 IsolationCutApplicatorBase(const edm::ParameterSet& c) :
  CutApplicatorBase(c) {
  }
  
  IsolationCutApplicatorBase(const IsolationCutApplicatorBase&) = delete;
  IsolationCutApplicatorBase& operator=(const IsolationCutApplicatorBase&) = delete;

  virtual void setConsumes(edm::ConsumesCollector&) = 0;

  virtual void setIsolationValuesFromEvent(const edm::EventBase&) = 0;
    
  //! Destructor
  virtual ~IsolationCutApplicatorBase(){};
  
 protected:
  std::unordered_map<std::string,edm::EDGetTokenT<edm::ValueMap<float> > > isolationTypes_;
};

#endif
