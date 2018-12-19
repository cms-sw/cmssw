#ifndef ConditionDumperInEdm_H
#define ConditionDumperInEdm_H
// -*- C++ -*-
//
// Package:    ConditionDumperInEdm
// Class:      ConditionDumperInEdm
// 
/**\class ConditionDumperInEdm ConditionDumperInEdm.cc FWCore/ConditionDumperInEdm/src/ConditionDumperInEdm.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Thu Feb 11 19:46:28 CET 2010
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

//
// class declaration
//

class ConditionDumperInEdm : public edm::one::EDProducer<edm::RunCache<edm::ConditionsInRunBlock>,
                                                         edm::LuminosityBlockCache<edm::ConditionsInLumiBlock>,
                                                         edm::EndRunProducer,
                                                         edm::EndLuminosityBlockProducer> {
   public:
      explicit ConditionDumperInEdm(const edm::ParameterSet&);
      ~ConditionDumperInEdm() override;

   private:
      std::shared_ptr<edm::ConditionsInLumiBlock> 
        globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;
      void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final {}
      void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) final;
      std::shared_ptr<edm::ConditionsInRunBlock> globalBeginRun(edm::Run const& , const edm::EventSetup&) const final;
      void globalEndRun(edm::Run const& , const edm::EventSetup&) final {}
      void endRunProduce(edm::Run& , const edm::EventSetup&) final;
      void produce(edm::Event&, const edm::EventSetup&) final;

  template <typename R, typename T>
  const T * get(const edm::EventSetup & setup) {
    edm::ESHandle<T> handle;
    setup.get<R>().get(handle);
    return handle.product();
  }

  // ----------member data ---------------------------

  const  edm::InputTag gtEvmDigisLabel_;

  edm::ConditionsInEventBlock eventBlock_;

  const edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> gtEvmDigisLabelToken_;
  const edm::EDPutTokenT<edm::ConditionsInLumiBlock> lumiToken_;
  const edm::EDPutTokenT<edm::ConditionsInRunBlock> runToken_;
  const edm::EDPutTokenT<edm::ConditionsInEventBlock> eventToken_;
};

#endif
