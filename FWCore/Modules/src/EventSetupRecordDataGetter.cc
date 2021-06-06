// -*- C++ -*-
//
// Package:    EventSetupRecordDataGetter
// Class:      EventSetupRecordDataGetter
//
/**\class EventSetupRecordDataGetter EventSetupRecordDataGetter.cc src/EventSetupRecordDataGetter/src/EventSetupRecordDataGetter.cc

 Description: Can be configured to 'get' any Data in any EventSetup Record.  Primarily used for testing.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 28 11:10:24 EDT 2005
//
//

// system include files
#include <map>
#include <vector>
#include <memory>
#include <set>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetTokenGeneric.h"

//
// class decleration
//
namespace edm {
  class EventSetupRecordDataGetter
      : public edm::global::EDAnalyzer<edm::RunCache<std::nullptr_t>, edm::LuminosityBlockCache<std::nullptr_t>> {
  public:
    explicit EventSetupRecordDataGetter(ParameterSet const&);
    ~EventSetupRecordDataGetter() override;

    void analyze(edm::StreamID, Event const&, EventSetup const&) const final;
    std::shared_ptr<std::nullptr_t> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const final {}

    std::shared_ptr<std::nullptr_t> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                               edm::EventSetup const&) const final;
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final {}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void registerLateConsumes(eventsetup::ESRecordsToProxyIndices const&) final;

    using RecordToTokens = std::map<eventsetup::EventSetupRecordKey, std::vector<ESGetTokenGeneric>>;
    void doGet(EventSetup const&, RecordToTokens const&) const;
    // ----------member data ---------------------------
    const ParameterSet pSet_;

    typedef std::map<eventsetup::EventSetupRecordKey, std::vector<eventsetup::DataKey>> RecordToDataKeys;
    RecordToDataKeys recordToDataKeys_;
    RecordToTokens recordToTokensRuns_;
    RecordToTokens recordToTokensLumis_;
    mutable std::map<eventsetup::EventSetupRecordKey, std::atomic<unsigned long long>> recordToCacheIdentifier_;
    const bool verbose_;
  };

  //
  // constructors and destructor
  //
  EventSetupRecordDataGetter::EventSetupRecordDataGetter(ParameterSet const& iConfig)
      : pSet_(iConfig),
        recordToDataKeys_(),
        recordToCacheIdentifier_(),
        verbose_(iConfig.getUntrackedParameter<bool>("verbose")) {}

  EventSetupRecordDataGetter::~EventSetupRecordDataGetter() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void EventSetupRecordDataGetter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    descriptions.setComment("Retrieves specified data from the EventSetup sytem whenever that data changes.");

    ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)
        ->setComment("Print a message to the logger each time a data item is gotten.");

    ParameterSetDescription toGet;
    toGet.add<std::string>("record")->setComment(
        "The name of an EventSetup record holding the data you want obtained.");
    toGet.add<std::vector<std::string>>("data")->setComment(
        "The identifier for the data you wish to retrieve. "
        "The identifier is in two parts separated by a backslash '/'. "
        "The first part is the C++ class name of the data and the "
        "second part is the label used when getting the data (blank is acceptable). "
        "If there is no label, the backslash may be omitted.");

    std::vector<edm::ParameterSet> emptyVect;
    desc.addVPSet("toGet", toGet, emptyVect)
        ->setComment(
            "The contained PSets must have the following structure.\n"
            "A 'string' named 'record' that holds the name of an EventSetup record holding the data you want to "
            "obtain.\n"
            "a 'vstring' named 'data' that holds identifiers for the data you wish to retrieve. "
            "The identifier is in two parts separated by a backslash '/'. "
            "The first part is the C++ class name of the data and the "
            "second part is the label used when getting the data (blank is acceptable). "
            "If there is no label, the backslash may be omitted.\n"
            "If the VPSet is empty it means all data in the EventSetup should be retrieved.");
    descriptions.add("getEventSetupData", desc);
  }

  std::shared_ptr<std::nullptr_t> EventSetupRecordDataGetter::globalBeginRun(Run const&,
                                                                             EventSetup const& iSetup) const {
    doGet(iSetup, recordToTokensRuns_);
    return {};
  }

  std::shared_ptr<std::nullptr_t> EventSetupRecordDataGetter::globalBeginLuminosityBlock(
      LuminosityBlock const&, EventSetup const& iSetup) const {
    doGet(iSetup, recordToTokensLumis_);
    return {};
  }

  void EventSetupRecordDataGetter::analyze(edm::StreamID,
                                           edm::Event const& /*iEvent*/,
                                           edm::EventSetup const& iSetup) const {}

  void EventSetupRecordDataGetter::registerLateConsumes(eventsetup::ESRecordsToProxyIndices const& iInfo) {
    auto const& toGet = pSet_.getParameterSetVector("toGet");

    for (auto const& iGet : toGet) {
      std::string recordName = iGet.getParameter<std::string>("record");

      eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
      if (recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag()) {
        //record not found
        edm::LogWarning("DataGetter") << "Record \"" << recordName << "\" does not exist " << std::endl;

        continue;
      }
      auto dataNames = iGet.getParameter<std::vector<std::string>>("data");
      std::vector<eventsetup::DataKey> dataKeys;
      for (auto const& datum : dataNames) {
        std::string datumName(datum, 0, datum.find_first_of('/'));
        std::string labelName;
        if (datum.size() != datumName.size()) {
          labelName = std::string(datum, datumName.size() + 1);
        }
        eventsetup::TypeTag datumType = eventsetup::TypeTag::findType(datumName);
        if (datumType == eventsetup::TypeTag()) {
          //not found
          edm::LogWarning("DataGetter") << "data item of type \"" << datumName << "\" does not exist" << std::endl;

          continue;
        }
        eventsetup::DataKey datumKey(datumType, labelName.c_str());
        dataKeys.push_back(datumKey);
      }
      recordToDataKeys_.insert(std::make_pair(recordKey, dataKeys));
      recordToCacheIdentifier_.insert(std::make_pair(recordKey, 0));
    }
    if (toGet.empty()) {
      //This means we should get everything in the EventSetup
      std::vector<eventsetup::EventSetupRecordKey> recordKeys = iInfo.recordKeys();

      for (auto const& rKey : recordKeys) {
        auto range = iInfo.keysForRecord(rKey);
        recordToDataKeys_.insert(std::make_pair(rKey, std::vector<eventsetup::DataKey>(range.first, range.second)));
        recordToCacheIdentifier_.insert(std::make_pair(rKey, 0));
      }
    }
    for (auto const& r : recordToDataKeys_) {
      auto& runs = recordToTokensRuns_[r.first];
      auto& lumis = recordToTokensLumis_[r.first];
      runs.reserve(r.second.size());
      lumis.reserve(r.second.size());
      for (auto const& dk : r.second) {
        runs.push_back(esConsumes<edm::Transition::BeginRun>(r.first, dk));
        lumis.push_back(esConsumes<edm::Transition::BeginLuminosityBlock>(r.first, dk));
      }
    }
  }

  void EventSetupRecordDataGetter::doGet(EventSetup const& iSetup, RecordToTokens const& iRecordToTokens) const {
    using namespace edm::eventsetup;

    //For each requested Record get the requested data only if the Record is in a new IOV

    for (auto const& record : recordToDataKeys_) {
      auto pRecord = iSetup.find(record.first);
      if (not pRecord) {
        edm::LogWarning("RecordNotInIOV")
            << "The EventSetup Record '" << record.first.name() << "' is not available for this IOV.";
      }
      auto const& tokens = iRecordToTokens.find(record.first)->second;
      auto ci = recordToCacheIdentifier_[record.first].load();
      if (pRecord.has_value() && pRecord->cacheIdentifier() != ci) {
        recordToCacheIdentifier_[record.first].compare_exchange_strong(ci, pRecord->cacheIdentifier());
        auto const& keys = record.second;
        size_t i = 0;
        for (auto const& token : tokens) {
          if (!pRecord->doGet(token)) {
            auto const& key = keys[i];
            edm::LogWarning("DataGetter")
                << "No data of type \"" << key.type().name() << "\" with name \"" << key.name().value()
                << "\" in record " << record.first.type().name() << " found " << std::endl;
          } else {
            if (verbose_) {
              auto const& key = keys[i];
              edm::LogSystem("DataGetter")
                  << "got data of type \"" << key.type().name() << "\" with name \"" << key.name().value()
                  << "\" in record " << record.first.type().name() << std::endl;
            }
          }
          ++i;
        }
      }
    }
  }
}  // namespace edm
using edm::EventSetupRecordDataGetter;
DEFINE_FWK_MODULE(EventSetupRecordDataGetter);
