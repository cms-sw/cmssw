//
// Package:    LogErrorHarvester
// Class:      LogErrorHarvester

/**\class LogErrorHarvester LogErrorHarvester.cc FWCore/Modules/src/LogErrorHarvester.cc

 Description: Harvestes LogError messages and puts them into the Event

 Implementation:
   This simple implementation writes the std::vector<ErrorSummaryEntry> in the event,
   without any fancy attempt of encoding the strings or mapping them to ints
*/
//
// Original Author:  Giovanni Petrucciani
//         Created:  Thu Dec  4 16:22:40 CET 2008
//

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/EndPathStatus.h"
#include "DataFormats/Common/interface/PathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"

// system include files
#include <memory>
#include <unordered_set>
#include <string>

//
// class decleration
//

namespace edm {
  class LogErrorHarvester : public global::EDProducer<> {
  public:
    explicit LogErrorHarvester(ParameterSet const&);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void beginJob() override;
    void produce(StreamID, Event&, EventSetup const&) const override;
    void endJob() override;
    
  };

  LogErrorHarvester::LogErrorHarvester(ParameterSet const& iPSet) {
    
    produces<std::vector<ErrorSummaryEntry>>();

    const edm::TypeID endPathStatusType{typeid(edm::EndPathStatus)};
    const edm::TypeID pathStatusType{typeid(edm::PathStatus)};
    const edm::TypeID triggerResultsType{typeid(edm::TriggerResults)};

    auto const& ignore = iPSet.getUntrackedParameter<std::vector<std::string>>("excludeModules");
    const std::unordered_set<std::string> excludedModules(ignore.begin(),ignore.end());

    auto const& includeM = iPSet.getUntrackedParameter<std::vector<std::string>>("includeModules");
    const std::unordered_set<std::string> includeModules(includeM.begin(),includeM.end());

    //Need to be sure to run only after all other EDProducers have run
    callWhenNewProductsRegistered([this,
                                   endPathStatusType,pathStatusType,triggerResultsType,
                                   excludedModules, includeModules](edm::BranchDescription const& iBD) 
    {
      if((iBD.branchType() == edm::InEvent and moduleDescription().processName() == iBD.processName()) and 
         ( (includeModules.empty() or
            includeModules.end() != includeModules.find(iBD.moduleLabel())) and
           (iBD.unwrappedTypeID() != endPathStatusType and
            iBD.unwrappedTypeID() != pathStatusType and
            iBD.unwrappedTypeID() != triggerResultsType))) {
        if(excludedModules.end() == excludedModules.find(iBD.moduleLabel())) {
          consumes(edm::TypeToGet{iBD.unwrappedTypeID(),edm::PRODUCT_TYPE},
                   edm::InputTag{iBD.moduleLabel(),
                                 iBD.productInstanceName(),
                                 iBD.processName()});
        }
      }
    });
  }

  void
  LogErrorHarvester::produce(StreamID const sid, Event& iEvent, EventSetup const&) const {
    const auto index = sid.value();
    if(!FreshErrorsExist(index)) {
      iEvent.put(std::make_unique<std::vector<ErrorSummaryEntry>>());
    } else {
      iEvent.put(std::make_unique<std::vector<ErrorSummaryEntry>>(LoggedErrorsSummary(index)));
    }
  }

  // ------------ method called once each job just before starting event loop  ------------
  void
  LogErrorHarvester::beginJob() {
    EnableLoggedErrorsSummary();
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void
  LogErrorHarvester::endJob() {
    DisableLoggedErrorsSummary();
  }


  // ------------ method called once each job for validation  ------------
  void
  LogErrorHarvester::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("excludeModules",std::vector<std::string>{})->setComment("List of module labels to exclude from consumes.");
    desc.addUntracked<std::vector<std::string>>("includeModules",std::vector<std::string>{})->setComment("List of the only module labels to include in consumes. The empty list will include all.");
    descriptions.add("logErrorHarvester", desc);
  }
}

//define this as a plug-in
using edm::LogErrorHarvester;
DEFINE_FWK_MODULE(LogErrorHarvester);
