// -*- C++ -*-
//
// Package:    CondTools/RunInfo
// Class:      RunInfoTestESProducer
//
/**\class RunInfoTestESProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 02 Oct 2019 17:34:35 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

//
// class declaration
//

class RunInfoTestESProducer : public edm::ESProducer {
public:
  RunInfoTestESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<RunInfo>;

  ReturnType produce(const RunInfoRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  RunInfo makeRunInfo(edm::ParameterSet const& pset) const;
  // ----------member data ---------------------------
  std::vector<RunInfo> runInfos_;
};

//
// constants, enums and typedefs
//
namespace {
  bool ri_less(RunInfo const& iLHS, RunInfo const& iRHS) { return iLHS.m_run < iRHS.m_run; }
}  // namespace

//
// static data member definitions
//

//
// constructors and destructor
//
RunInfoTestESProducer::RunInfoTestESProducer(const edm::ParameterSet& iConfig) {
  std::vector<edm::ParameterSet> const& runInfos = iConfig.getParameter<std::vector<edm::ParameterSet>>("runInfos");
  runInfos_.reserve(runInfos.size());
  for (auto const& pset : runInfos) {
    runInfos_.emplace_back(makeRunInfo(pset));
  }
  std::sort(runInfos_.begin(), runInfos_.end(), ri_less);

  setWhatProduced(this);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
RunInfoTestESProducer::ReturnType RunInfoTestESProducer::produce(const RunInfoRcd& iRecord) {
  const int run = iRecord.validityInterval().first().eventID().run();
  RunInfo toFind;
  toFind.m_run = run;
  auto itFound = std::lower_bound(runInfos_.begin(), runInfos_.end(), toFind, ri_less);
  if (itFound == runInfos_.end() or itFound->m_run != run) {
    return nullptr;
  }
  return std::make_unique<RunInfo>(*itFound);
}

RunInfo RunInfoTestESProducer::makeRunInfo(edm::ParameterSet const& pset) const {
  RunInfo retValue;
  retValue.m_run = pset.getParameter<int>("run");
  retValue.m_start_time_ll = pset.getParameter<long long>("start_time");
  retValue.m_start_time_str = pset.getParameter<std::string>("start_time_str");
  retValue.m_stop_time_ll = pset.getParameter<long long>("stop_time");
  retValue.m_stop_time_str = pset.getParameter<std::string>("stop_time_str");
  retValue.m_fed_in = pset.getParameter<std::vector<int>>("fed_in");
  retValue.m_start_current = pset.getParameter<double>("start_current");
  retValue.m_stop_current = pset.getParameter<double>("stop_current");
  retValue.m_avg_current = pset.getParameter<double>("avg_current");
  retValue.m_min_current = pset.getParameter<double>("min_current");
  retValue.m_max_current = pset.getParameter<double>("max_current");
  retValue.m_run_intervall_micros = pset.getParameter<double>("run_intervall_micros");

  auto convert = [](std::vector<double> const& iIn) {
    std::vector<float> f;
    f.reserve(iIn.size());
    std::copy(iIn.begin(), iIn.end(), std::back_inserter(f));
    return f;
  };

  retValue.m_current = convert(pset.getParameter<std::vector<double>>("current"));
  retValue.m_times_of_currents = convert(pset.getParameter<std::vector<double>>("times_of_currents"));

  return retValue;
}

void RunInfoTestESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription runInfoDesc;
  runInfoDesc.add<int>("run");
  runInfoDesc.add<long long>("start_time", 0);
  runInfoDesc.add<std::string>("start_time_str", "");
  runInfoDesc.add<long long>("stop_time", 0);
  runInfoDesc.add<std::string>("stop_time_str", "");
  runInfoDesc.add<std::vector<int>>("fed_in", {});
  runInfoDesc.add<double>("start_current", 0);
  runInfoDesc.add<double>("stop_current", 0);
  runInfoDesc.add<double>("avg_current", 0);
  runInfoDesc.add<double>("min_current", 0);
  runInfoDesc.add<double>("max_current", 0);
  runInfoDesc.add<double>("run_intervall_micros", 0);
  runInfoDesc.add<std::vector<double>>("current", {});
  runInfoDesc.add<std::vector<double>>("times_of_currents", {});

  edm::ParameterSetDescription desc;
  desc.addVPSet("runInfos", runInfoDesc, {});

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RunInfoTestESProducer);
