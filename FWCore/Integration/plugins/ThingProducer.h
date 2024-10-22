#ifndef Integration_ThingProducer_h
#define Integration_ThingProducer_h

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "ThingAlgorithm.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace edmtest {
  class ThingProducer : public edm::global::EDProducer<edm::BeginRunProducer,
                                                       edm::EndRunProducer,
                                                       edm::EndLuminosityBlockProducer,
                                                       edm::BeginLuminosityBlockProducer> {
  public:
    explicit ThingProducer(edm::ParameterSet const& ps);

    ~ThingProducer() override;

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    void globalBeginRunProduce(edm::Run& r, edm::EventSetup const& c) const override;

    void globalEndRunProduce(edm::Run& r, edm::EventSetup const& c) const override;

    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) const override;

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    ThingAlgorithm alg_;
    edm::EDPutTokenT<ThingCollection> evToken_;
    edm::EDPutTokenT<ThingCollection> brToken_;
    edm::EDPutTokenT<ThingCollection> erToken_;
    edm::EDPutTokenT<ThingCollection> blToken_;
    edm::EDPutTokenT<ThingCollection> elToken_;
    bool noPut_;
  };
}  // namespace edmtest
#endif
