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
// $Id: ConditionDumperInEdm.h,v 1.3 2010/03/16 23:58:25 ghete Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

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

class ConditionDumperInEdm : public edm::EDProducer {
   public:
      explicit ConditionDumperInEdm(const edm::ParameterSet&);
      ~ConditionDumperInEdm();

   private:
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void beginRun(edm::Run& , const edm::EventSetup&);
      virtual void endRun(edm::Run& , const edm::EventSetup&);
      virtual void produce(edm::Event&, const edm::EventSetup&);

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
