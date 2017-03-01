// -*- C++ -*-
//
// Package:     JetMETCorrections/Algorithms
// Class  :     JetCorrectorImplMakerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 29 Aug 2014 19:52:26 GMT
//

// system include files
#include <vector>

// user include files
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetCorrectorImplMakerBase::JetCorrectorImplMakerBase(edm::ParameterSet const& iPSet):
  level_(iPSet.getParameter<std::string>("level")),
  algo_(iPSet.getParameter<std::string>("algorithm")),
  cacheId_(0)
{
}

// JetCorrectorImplMakerBase::JetCorrectorImplMakerBase(const JetCorrectorImplMakerBase& rhs)
// {
//    // do actual copying here;
// }

JetCorrectorImplMakerBase::~JetCorrectorImplMakerBase()
{
}

//
// member functions
//

std::shared_ptr<FactorizedJetCorrectorCalculator const> 
JetCorrectorImplMakerBase::getCalculator(edm::EventSetup const& iSetup, std::function<void(std::string const&)> iLevelCheck) {
  auto const& rec = iSetup.get<JetCorrectionsRecord>();
  if( cacheId_ != rec.cacheIdentifier()) {
    edm::ESHandle<JetCorrectorParametersCollection> JetCorParColl;
    rec.get(algo_,JetCorParColl); 
    auto const& parameters = ((*JetCorParColl)[level_]);

    iLevelCheck(parameters.definitions().level());
    std::vector<JetCorrectorParameters> vParam;
    vParam.push_back(parameters);
    corrector_ = std::make_shared<FactorizedJetCorrectorCalculator>(vParam);

    cacheId_ = rec.cacheIdentifier();
  }
  return corrector_;
}

void 
JetCorrectorImplMakerBase::addToDescription(edm::ParameterSetDescription& iDescription) {
  iDescription.add<std::string>("level");
  iDescription.add<std::string>("algorithm");
}
