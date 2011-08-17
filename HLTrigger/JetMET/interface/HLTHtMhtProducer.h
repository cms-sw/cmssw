#ifndef HLTHtMhtProducer_h
#define HLTHtMhtProducer_h

/** \class HLTHtMhtProducer
 *
 *  \author Steven Lowette
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


namespace edm {
   class ConfigurationDescriptions;
}


class HLTHtMhtProducer : public edm::EDProducer {

  public:

    explicit HLTHtMhtProducer(const edm::ParameterSet & iConfig);
    ~HLTHtMhtProducer();
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    bool usePt_;
    bool useTracks_;
    std::vector<int>    minNJet_;
    std::vector<double> minPtJet_;
    std::vector<double> maxEtaJet_;
    edm::InputTag jetsLabel_;
    edm::InputTag tracksLabel_;

};

#endif
