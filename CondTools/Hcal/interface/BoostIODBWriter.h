#ifndef CondTools_Hcal_BoostIODBWriter_h
#define CondTools_Hcal_BoostIODBWriter_h

// -*- C++ -*-
//
// Package:    CondTools/Hcal
// Class:      BoostIODBWriter
//
/**\class BoostIODBWriter BoostIODBWriter.h CondTools/Hcal/interface/BoostIODBWriter.h

 Description: writes a boost I/O blob from a file into a database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri Apr 25 17:58:33 CDT 2014
//
//

#include <string>
#include <fstream>
#include <memory>
#include <cassert>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// class declaration
//
template <class DataType>
class BoostIODBWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  typedef DataType data_type;

  explicit BoostIODBWriter(const edm::ParameterSet&);
  inline ~BoostIODBWriter() override {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string inputFile_;
  std::string record_;
};

template <class DataType>
BoostIODBWriter<DataType>::BoostIODBWriter(const edm::ParameterSet& ps)
    : inputFile_(ps.getParameter<std::string>("inputFile")), record_(ps.getParameter<std::string>("record")) {
  usesResource(cond::service::PoolDBOutputService::kSharedResource);
}

template <class DataType>
void BoostIODBWriter<DataType>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::ifstream is(inputFile_.c_str(), std::ios_base::binary);
  if (!is.is_open())
    throw cms::Exception("InvalidArgument") << "Failed to open file \"" << inputFile_ << '"' << std::endl;

  std::unique_ptr<DataType> datum(new DataType());
  eos::portable_iarchive ar(is);
  ar&* datum;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(*datum, poolDbService->currentTime(), record_);
  else
    throw cms::Exception("ConfigurationError") << "PoolDBOutputService is not available, "
                                               << "please configure it properly" << std::endl;
}

#endif  // CondTools_Hcal_BoostIODBWriter_h
