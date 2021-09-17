
// -*- C++ -*-
//
// Package:    PrintEventSetupContent
// Class:      PrintEventSetupContent
//
/**\class PrintEventSetupContent PrintEventSetupContent.cc GetRecordName/PrintEventSetupContent/src/PrintEventSetupContent.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Weng Yao
//         Created:  Tue Oct  2 13:49:56 EDT 2007
//
//

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <iostream>
#include <map>
#include <memory>

//
// class decleration
//

namespace edm {
  class PrintEventSetupContent : public one::EDAnalyzer<one::WatchRuns, one::WatchLuminosityBlocks> {
  public:
    explicit PrintEventSetupContent(ParameterSet const&);
    ~PrintEventSetupContent() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void beginJob() override;

    void analyze(Event const&, EventSetup const&) override;
    void endJob() override;
    void beginRun(Run const&, EventSetup const&) override;
    void endRun(Run const&, EventSetup const&) override;
    void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;

    void print(EventSetup const&);

    // ----------member data ---------------------------
    const bool printProviders_;
    const bool compact_;
    std::map<eventsetup::EventSetupRecordKey, unsigned long long> cacheIdentifiers_;
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
  PrintEventSetupContent::PrintEventSetupContent(ParameterSet const& config)
      : printProviders_(config.getUntrackedParameter<bool>("printProviders")),
        compact_(config.getUntrackedParameter<bool>("compact")) {
    //now do what ever initialization is neededEventSetupRecordDataGetter::EventSetupRecordDataGetter(ParameterSet const& iConfig):
    //  getter = new EventSetupRecordDataGetter::EventSetupRecordDataGetter(iConfig);
  }

  PrintEventSetupContent::~PrintEventSetupContent() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  //
  // member functions
  //

  // ------------ method called to for each event  ------------
  void PrintEventSetupContent::analyze(Event const&, EventSetup const& iSetup) { print(iSetup); }

  void PrintEventSetupContent::beginRun(Run const&, EventSetup const& iSetup) { print(iSetup); }

  void PrintEventSetupContent::endRun(Run const&, EventSetup const& iSetup) {}

  void PrintEventSetupContent::beginLuminosityBlock(LuminosityBlock const&, EventSetup const& iSetup) { print(iSetup); }

  void PrintEventSetupContent::endLuminosityBlock(LuminosityBlock const&, EventSetup const& iSetup) {}

  void PrintEventSetupContent::print(EventSetup const& iSetup) {
    typedef std::vector<eventsetup::EventSetupRecordKey> Records;
    typedef std::vector<eventsetup::DataKey> Data;

    Records records;
    Data data;
    iSetup.fillAvailableRecordKeys(records);
    std::unique_ptr<LogSystem> msg;

    for (Records::iterator itrecords = records.begin(), itrecordsend = records.end(); itrecords != itrecordsend;
         ++itrecords) {
      auto rec = iSetup.find(*itrecords);

      if (rec && cacheIdentifiers_[*itrecords] != rec->cacheIdentifier()) {
        cacheIdentifiers_[*itrecords] = rec->cacheIdentifier();
        rec->fillRegisteredDataKeys(data);
        if (compact_) {
          for (Data::iterator itdata = data.begin(), itdataend = data.end(); itdata != itdataend; ++itdata) {
            if (not msg)
              msg = std::make_unique<LogSystem>("ESContent");
            else
              *msg << '\n';
            *msg << "ESContent> "
                 << "record:" << itrecords->name() << " data:" << itdata->type().name() << " '"
                 << itdata->name().value() << "'";
            if (printProviders_) {
              edm::eventsetup::ComponentDescription const* cd = rec->providerDescription(*itdata);
              *msg << " provider:" << cd->type_ << " '" << cd->label_ << "'";
            }
          }
        } else {
          if (not msg) {
            msg = std::make_unique<LogSystem>("ESContent");
            *msg << "Changed Record"
                 << "\n  "
                 << "<datatype>"
                 << " "
                 << "'label' provider: 'provider label' <provider module type>";
          }
          *msg << "\n" << itrecords->name();
          *msg << "\n start: " << rec->validityInterval().first().eventID()
               << " time: " << rec->validityInterval().first().time().value();
          *msg << "\n end:   " << rec->validityInterval().last().eventID()
               << " time: " << rec->validityInterval().last().time().value();
          for (Data::iterator itdata = data.begin(), itdataend = data.end(); itdata != itdataend; ++itdata) {
            *msg << "\n  " << itdata->type().name() << " '" << itdata->name().value() << "'";
            if (printProviders_) {
              edm::eventsetup::ComponentDescription const* cd = rec->providerDescription(*itdata);
              *msg << " provider:" << cd->type_ << " '" << cd->label_ << "'";
            }
          }
        }
      }
    }
  }

  //#ifdef THIS_IS_AN_EVENT_EXAMPLE
  //   Handle<ExampleData> pIn;
  //   iEvent.getByLabel("example", pIn);
  //#endif

  //#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  //   ESHandle<SetupData> pSetup;
  //   iSetup.get<SetupRecord>().get(pSetup);
  //#endif

  // ------------ method called once each job just before starting event loop  ------------
  void PrintEventSetupContent::beginJob() {}

  // ------------ method called once each job just after ending the event loop  ------------
  void PrintEventSetupContent::endJob() {}

  // ------------ method called once each job for validation  ------------
  void PrintEventSetupContent::fillDescriptions(ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "Print what data is available in each available EventSetup Record in the job.\n"
        "As part of the data is the C++ class type, label and which module makes that data.");
    ParameterSetDescription desc;
    desc.addUntracked<bool>("compact", false)
        ->setComment("If 'true' produces a more compact view, similar to the one used by PrintEventSetupDataRetrieval");
    desc.addUntracked<bool>("printProviders", true)
        ->setComment("If 'true' also print which ES module provides the data");
    descriptions.add("printEventSetupContent", desc);
  }
}  // namespace edm

//define this as a plug-in
using edm::PrintEventSetupContent;
DEFINE_FWK_MODULE(PrintEventSetupContent);
