// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescriptionFillerBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  1 16:48:00 EDT 2007
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"


//
// constants, enums and typedefs
//
const std::string edm::ParameterSetDescriptionFillerBase::kEmpty("");
const std::string edm::ParameterSetDescriptionFillerBase::kBaseForService("Service");
const std::string edm::ParameterSetDescriptionFillerBase::kBaseForESSource("ESSource");
const std::string edm::ParameterSetDescriptionFillerBase::kBaseForESProducer("ESProducer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForEDAnalyzer("EDAnalyzer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForEDProducer("EDProducer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForEDFilter("EDFilter");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForOutputModule("OutputModule");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForOneEDAnalyzer("one::EDAnalyzer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForOneEDProducer("one::EDProducer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForOneEDFilter("one::EDFilter");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForOneOutputModule("one::OutputModule");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForStreamEDAnalyzer("stream::EDAnalyzer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForStreamEDProducer("stream::EDProducer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForStreamEDFilter("stream::EDFilter");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForGlobalEDAnalyzer("global::EDAnalyzer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForGlobalEDProducer("global::EDProducer");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForGlobalEDFilter("global::EDFilter");
const std::string edm::ParameterSetDescriptionFillerBase::kExtendedBaseForGlobalOutputModule("global::OutputModule");

//
// static data member definitions
//

//
// constructors and destructor
//
//ParameterSetDescriptionFillerBase::ParameterSetDescriptionFillerBase()
//{
//}

// ParameterSetDescriptionFillerBase::ParameterSetDescriptionFillerBase(const ParameterSetDescriptionFillerBase& rhs)
// {
//    // do actual copying here;
// }
namespace edm {
ParameterSetDescriptionFillerBase::~ParameterSetDescriptionFillerBase()
{
}
}
//
// assignment operators
//
// const ParameterSetDescriptionFillerBase& ParameterSetDescriptionFillerBase::operator=(const ParameterSetDescriptionFillerBase& rhs)
// {
//   //An exception safe implementation is
//   ParameterSetDescriptionFillerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
