// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      ConfigurableAPVCyclePhaseProducer
//
/**\class ConfigurableAPVCyclePhaseProducer ConfigurableAPVCyclePhaseProducer.cc DPGAnalysis/SiStripTools/plugins/ConfigurableAPVCyclePhaseProducer.cc

 Description: EDproducer for APVCyclePhaseCollection which uses the configuration file to assign a phase to the run

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Jan 12 09:05:45 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <string>

#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

//
// class decleration
//

class ConfigurableAPVCyclePhaseProducer : public edm::global::EDProducer<edm::RunCache<APVCyclePhaseCollection>> {
public:
  explicit ConfigurableAPVCyclePhaseProducer(const edm::ParameterSet&);
  ~ConfigurableAPVCyclePhaseProducer() override;

private:
  std::shared_ptr<APVCyclePhaseCollection> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

  // ----------member data ---------------------------

  const std::vector<std::string> _defpartnames;
  const std::vector<int> _defphases;

  std::map<int, std::vector<std::string>> _runpartnames;
  std::map<int, std::vector<int>> _runphases;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ConfigurableAPVCyclePhaseProducer::ConfigurableAPVCyclePhaseProducer(const edm::ParameterSet& iConfig)
    : _defpartnames(iConfig.getParameter<std::vector<std::string>>("defaultPartitionNames")),
      _defphases(iConfig.getParameter<std::vector<int>>("defaultPhases")) {
  produces<APVCyclePhaseCollection, edm::InEvent>();

  //now do what ever other initialization is needed

  if (_defphases.size() < _defpartnames.size()) {
    // throw exception
    throw cms::Exception("InvalidAPVCyclePhases")
        << " Inconsistent default phases/partitions vector sizes: " << _defphases.size() << " " << _defpartnames.size();
  }

  std::vector<edm::ParameterSet> vps(iConfig.getParameter<std::vector<edm::ParameterSet>>("runPhases"));

  for (std::vector<edm::ParameterSet>::const_iterator ps = vps.begin(); ps != vps.end(); ps++) {
    _runphases[ps->getParameter<int>("runNumber")] = ps->getUntrackedParameter<std::vector<int>>("phases", _defphases);
    _runpartnames[ps->getParameter<int>("runNumber")] =
        ps->getUntrackedParameter<std::vector<std::string>>("partitions", _defpartnames);

    if (_runphases[ps->getParameter<int>("runNumber")].size() <
        _runpartnames[ps->getParameter<int>("runNumber")].size()) {
      // throw exception
      throw cms::Exception("InvalidAPVCyclePhases")
          << " Inconsistent run " << ps->getParameter<int>("runNumber")
          << " phases/partitions vector sizes: " << _runphases[ps->getParameter<int>("runNumber")].size() << " "
          << _runpartnames[ps->getParameter<int>("runNumber")].size();
    }
  }
}

ConfigurableAPVCyclePhaseProducer::~ConfigurableAPVCyclePhaseProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::shared_ptr<APVCyclePhaseCollection> ConfigurableAPVCyclePhaseProducer::globalBeginRun(
    const edm::Run& iRun, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto currapvphases = std::make_shared<APVCyclePhaseCollection>();

  // fill phase map

  const std::map<int, std::vector<std::string>>& _crunpartnames = _runpartnames;
  const std::map<int, std::vector<int>>& _crunphases = _runphases;

  std::map<int, std::vector<int>>::const_iterator trphases = _crunphases.find(iRun.run());
  std::map<int, std::vector<std::string>>::const_iterator trpartnames = _crunpartnames.find(iRun.run());

  std::vector<int> phases = _defphases;
  std::vector<std::string> partnames = _defpartnames;

  if (trphases != _crunphases.end()) {
    phases = trphases->second;
  }
  if (trpartnames != _crunpartnames.end()) {
    partnames = trpartnames->second;
  }

  if (phases.size() < partnames.size()) {
    // throw exception
    throw cms::Exception("InvalidAPVCyclePhases")
        << " Inconsistent phases/partitions vector sizes: " << phases.size() << " " << partnames.size();
  }

  for (unsigned int ipart = 0; ipart < partnames.size(); ++ipart) {
    if (phases[ipart] >= 0) {
      currapvphases->get()[partnames[ipart]] = phases[ipart];
    }
  }

  for (std::map<std::string, int>::const_iterator it = currapvphases->get().begin(); it != currapvphases->get().end();
       it++) {
    edm::LogInfo("APVCyclePhaseProducerDebug") << "partition " << it->first << " phase " << it->second;
  }
  return currapvphases;
}

void ConfigurableAPVCyclePhaseProducer::produce(edm::StreamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& iSetup) const {
  using namespace edm;

  std::unique_ptr<APVCyclePhaseCollection> apvphases =
      std::make_unique<APVCyclePhaseCollection>(*runCache(iEvent.getRun().index()));

  iEvent.put(std::move(apvphases));
}

//define this as a plug-in
DEFINE_FWK_MODULE(ConfigurableAPVCyclePhaseProducer);
