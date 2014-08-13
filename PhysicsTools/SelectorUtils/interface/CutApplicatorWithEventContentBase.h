#ifndef PhysicsTools_SelectorUtils_CutApplicatorWithEventContentBase_h
#define PhysicsTools_SelectorUtils_CutApplicatorWithEventContentBase_h

//
//
//

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <unordered_map>

class CutApplicatorWithEventContentBase : public CutApplicatorBase {
 public:  
 CutApplicatorWithEventContentBase(const edm::ParameterSet& c) :
  CutApplicatorBase(c) {
  }
  
  CutApplicatorWithEventContentBase(const CutApplicatorWithEventContentBase&) = delete;
  CutApplicatorWithEventContentBase& operator=(const CutApplicatorWithEventContentBase&) = delete;

  virtual void setConsumes(edm::ConsumesCollector&) = 0;

  virtual void getEventContent(const edm::EventBase&) = 0;
    
  //! Destructor
  virtual ~CutApplicatorWithEventContentBase(){};
  
 protected:
  std::unordered_map<std::string,edm::InputTag> contentTags_;
  std::unordered_map<std::string,edm::EDGetToken> contentTokens_;
};

#endif
