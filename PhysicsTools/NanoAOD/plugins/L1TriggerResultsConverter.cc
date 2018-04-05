// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      L1TriggerResultsConverter
// 
/**\class L1TriggerResultsConverter L1TriggerResultsConverter.cc PhysicsTools/L1TriggerResultsConverter/plugins/L1TriggerResultsConverter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Mon, 11 Aug 2017 11:20:30 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
//
// class declaration
//

class L1TriggerResultsConverter : public edm::global::EDProducer<> {
   public:
      explicit L1TriggerResultsConverter(const edm::ParameterSet&);
      ~L1TriggerResultsConverter() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      virtual void beginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&);

      // ----------member data ---------------------------
      const bool legacyL1_;
      const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tokenLegacy_;
      const edm::EDGetTokenT<GlobalAlgBlkBxCollection> token_;
      std::vector<std::string> names_;
      std::vector<unsigned int> mask_;
      std::vector<unsigned int> indices_;
};



//
// constructors and destructor
//
L1TriggerResultsConverter::L1TriggerResultsConverter(const edm::ParameterSet& params):
 legacyL1_( params.getParameter<bool>("legacyL1") ),
 tokenLegacy_(legacyL1_?consumes<L1GlobalTriggerReadoutRecord>( params.getParameter<edm::InputTag>("src") ): edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>()),
 token_(!legacyL1_?consumes<GlobalAlgBlkBxCollection>( params.getParameter<edm::InputTag>("src") ): edm::EDGetTokenT<GlobalAlgBlkBxCollection>())
{
   produces<edm::TriggerResults>();
}


L1TriggerResultsConverter::~L1TriggerResultsConverter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void L1TriggerResultsConverter::beginRun(edm::StreamID streamID, edm::Run const&, edm::EventSetup const&setup) {
    mask_.clear();
    names_.clear();
    indices_.clear();
    if(legacyL1_){
        edm::ESHandle<L1GtTriggerMenu> handleMenu;
        edm::ESHandle<L1GtTriggerMask> handleAlgoMask;
        setup.get<L1GtTriggerMenuRcd>().get(handleMenu);
        auto const & mapping = handleMenu->gtAlgorithmAliasMap();
        for (auto const & keyval: mapping) {
	   names_.push_back(keyval.first);
	   indices_.push_back(keyval.second.algoBitNumber()); 
        } 
        setup.get<L1GtTriggerMaskAlgoTrigRcd>().get(handleAlgoMask);
        mask_ = handleAlgoMask->gtTriggerMask();
    } else {
        edm::ESHandle<L1TUtmTriggerMenu> menu;
        setup.get<L1TUtmTriggerMenuRcd>().get(menu);
        auto const & mapping = menu->getAlgorithmMap();
        for (auto const & keyval: mapping) {
           names_.push_back(keyval.first);
	   indices_.push_back(keyval.second.getIndex()); 
        }

    }
}

// ------------ method called to produce the data  ------------


void
L1TriggerResultsConverter::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
    using namespace edm;
    const std::vector<bool> * wordp=nullptr;
    if (!legacyL1_){
      edm::Handle<GlobalAlgBlkBxCollection> handleResults;
       iEvent.getByToken(token_, handleResults);
       wordp= & handleResults->at(0,0).getAlgoDecisionFinal() ;
     } else {
// Legacy access
       edm::Handle<L1GlobalTriggerReadoutRecord> handleResults;
       iEvent.getByToken(tokenLegacy_, handleResults);
       wordp = & handleResults->decisionWord();
    }
    auto const &word = *wordp;
    HLTGlobalStatus l1bitsAsHLTStatus(names_.size());
//    std::cout << word.size() << " " << names_.size() << " " << mask_.size()  << std::endl;
    unsigned indices_size = indices_.size();
    for(size_t nidx=0;nidx<indices_size; nidx++) {
        unsigned int index = indices_[nidx];
        bool result =word[index];
	if(!mask_.empty()) result &=  (mask_[index] !=0);
	l1bitsAsHLTStatus[nidx]=HLTPathStatus(result?edm::hlt::Pass:edm::hlt::Fail);
    }
    //mimic HLT trigger bits for L1
    auto out = std::make_unique<edm::TriggerResults>(l1bitsAsHLTStatus,names_);
    iEvent.put(std::move(out));

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TriggerResultsConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("legacyL1")->setComment("is legacy L1");
  desc.add<edm::InputTag>("src")->setComment("L1 input (L1GlobalTriggerReadoutRecord if legacy, GlobalAlgBlkBxCollection otherwise)");
  descriptions.add("L1TriggerResultsConverter",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TriggerResultsConverter);
