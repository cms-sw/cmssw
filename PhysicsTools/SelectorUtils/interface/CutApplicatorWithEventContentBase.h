#ifndef PhysicsTools_SelectorUtils_CutApplicatorWithEventContentBase_h
#define PhysicsTools_SelectorUtils_CutApplicatorWithEventContentBase_h

//
//
//

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <unordered_map>
#else
#include <map>
#endif

class CutApplicatorWithEventContentBase : public CutApplicatorBase {
 public:  

 CutApplicatorWithEventContentBase(): CutApplicatorBase() {}

 CutApplicatorWithEventContentBase(const edm::ParameterSet& c) :
  CutApplicatorBase(c) {
  }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  CutApplicatorWithEventContentBase(const CutApplicatorWithEventContentBase&) = delete;
  CutApplicatorWithEventContentBase& operator=(const CutApplicatorWithEventContentBase&) = delete;


  virtual void setConsumes(edm::ConsumesCollector&) = 0;
#endif

  virtual void getEventContent(const edm::EventBase&) = 0;
    
  //! Destructor
  virtual ~CutApplicatorWithEventContentBase(){};
  
 protected:
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  std::unordered_map<std::string,edm::InputTag> contentTags_;
  std::unordered_map<std::string,edm::EDGetToken> contentTokens_;
#else
  std::map<std::string,edm::InputTag> contentTags_;
#endif
};

#endif
