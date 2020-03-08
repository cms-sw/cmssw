#ifndef __L1TTrackerDTC_L1TTRACKERDTCPRODUCER_H__
#define __L1TTrackerDTC_L1TTRACKERDTCPRODUCER_H__

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>


namespace L1TTrackerDTC {

  class Settings;
  class Module;

  /*! \class  L1TTrackerDTCProducer
 *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
 *  \author Thomas Schuh
 *  \date   2020, Jan
 */
  class L1TTrackerDTCProducer : public edm::EDProducer {
  public:
    explicit L1TTrackerDTCProducer(const edm::ParameterSet&);

    ~L1TTrackerDTCProducer() override;

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;

    void produce(edm::Event&, const edm::EventSetup&) override;

    void endJob() override {}

  private:
    Settings* settings_;  // helper class to store configurations

    edm::EDGetTokenT<TTStubDetSetVec> tokenTTStubDetSetVec_;

    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<Module*> > modules_;
  };

}  // namespace L1TTrackerDTC

#endif