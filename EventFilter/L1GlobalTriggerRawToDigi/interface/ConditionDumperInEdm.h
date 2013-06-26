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
// $Id: ConditionDumperInEdm.h,v 1.4 2013/05/17 21:07:10 chrjones Exp $
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

//
// class declaration
//

class ConditionDumperInEdm : public edm::one::EDProducer<edm::EndRunProducer,
                                                         edm::EndLuminosityBlockProducer> {
   public:
      explicit ConditionDumperInEdm(const edm::ParameterSet&);
      ~ConditionDumperInEdm();

   private:
      virtual void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override final;
      virtual void endRunProduce(edm::Run& , const edm::EventSetup&) override final;
      virtual void produce(edm::Event&, const edm::EventSetup&) override final;

  template <typename R, typename T>
  const T * get(const edm::EventSetup & setup) {
    edm::ESHandle<T> handle;
    setup.get<R>().get(handle);
    return handle.product();
  }

  // ----------member data ---------------------------

  edm::InputTag gtEvmDigisLabel_;

  edm::ConditionsInLumiBlock lumiBlock_;
  edm::ConditionsInRunBlock runBlock_;
  edm::ConditionsInEventBlock eventBlock_;

};

#endif
