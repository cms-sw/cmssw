#ifndef __TrackerDTC_TRACKERDTCPRODUCER_H__
#define __TrackerDTC_TRACKERDTCPRODUCER_H__

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"

#include <vector>
#include <memory>

namespace TrackerDTC {

  class Module;

  /*! \class  TrackerDTCProducer
 *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
 *  \author Thomas Schuh
 *  \date   2020, Jan
 */
  class TrackerDTCProducer : public edm::EDProducer {
  public:
    explicit TrackerDTCProducer(const edm::ParameterSet&);

    ~TrackerDTCProducer() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;

    void produce(edm::Event&, const edm::EventSetup&) override;

    void endJob() override {}

  private:
    Settings settings_;  // helper class to store configurations
    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<Module*> > dtcModules_;
    std::vector<Module > modules_;// collection of outer tracker sensor modules

    edm::EDGetTokenT<TTStubDetSetVec> tokenTTStubDetSetVec_;
  };

}  // namespace TrackerDTC

#endif