// CMSSW include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

// local include files
#include "api.h"

template <typename T>
class MPISender : public edm::global::EDProducer<> {
  using CollectionType = T;

public:
  MPISender(edm::ParameterSet const& config)
      : mpiPrev_(consumes<MPIToken>(config.getParameter<edm::InputTag>("channel"))),
        mpiNext_(produces<MPIToken>()),
        data_(consumes<CollectionType>(config.getParameter<edm::InputTag>("data"))),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue") << "Invalid MPISender instance value, please use a value between 1 and 255";
    }
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(mpiPrev_);

    // read the data to be sent over the MPI channel
    auto data = event.get(data_);

    // send the data over MPI
    // note: currently this uses a blocking send
    token.channel()->sendSerializedProduct(instance_, data);

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(mpiNext_, token);
  }

private:
  edm::EDGetTokenT<MPIToken> const mpiPrev_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const mpiNext_;  // copy of the MPIToken that may be used to implement an ordering relation
  edm::EDGetTokenT<CollectionType> const data_;  // data to be read from the Event and sent over the channel
  int32_t const instance_;                       // instance used to identify the source-destination pair
};

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Provenance/interface/EventID.h"
using MPISenderEventID = MPISender<edm::EventID>;
DEFINE_FWK_MODULE(MPISenderEventID);

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
using MPISenderFEDRawDataCollection = MPISender<FEDRawDataCollection>;
DEFINE_FWK_MODULE(MPISenderFEDRawDataCollection);
