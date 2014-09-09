#ifndef JetMETCorrections_Algorithms_JetCorrectorImplMakerBase_h
#define JetMETCorrections_Algorithms_JetCorrectorImplMakerBase_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/Algorithms
// Class  :     JetCorrectorImplMakerBase
// 
/**\class JetCorrectorImplMakerBase JetCorrectorImplMakerBase.h "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 29 Aug 2014 19:52:21 GMT
//

// system include files
#include <string>
#include <memory>
#include <functional>

// user include files
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"

// forward declarations
namespace edm {
  class ParameterSet;
  class EventSetup;
  class ParameterSetDescription;
}

class JetCorrectorImplMakerBase
{
  
 public:
  JetCorrectorImplMakerBase(edm::ParameterSet const&);
  virtual ~JetCorrectorImplMakerBase();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  static void addToDescription(edm::ParameterSetDescription& iDescription);
  
  // ---------- member functions ---------------------------

 protected:
  std::shared_ptr<FactorizedJetCorrectorCalculator const> getCalculator(edm::EventSetup const&,
									std::function<void(std::string const&)> levelCheck);
  
 private:
  JetCorrectorImplMakerBase(const JetCorrectorImplMakerBase&); // stop default
  
  const JetCorrectorImplMakerBase& operator=(const JetCorrectorImplMakerBase&); // stop default
  
  // ---------- member data --------------------------------
  std::string level_;
  std::string algo_;
  std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector_;
  unsigned long long cacheId_;

};


#endif
