// C++ include files
#include <utility>

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
class MPIReceiver : public edm::global::EDProducer<> {
  using CollectionType = T;

public:
  MPIReceiver(edm::ParameterSet const& config)
      : mpiPrev_(consumes<MPIToken>(config.getParameter<edm::InputTag>("channel"))),
        mpiNext_(produces<MPIToken>()),
        data_(produces<CollectionType>()),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiver instance value, please use a value between 1 and 255";
    }
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(mpiPrev_);

    // receive the data sent over the MPI channel
    // note: currently this uses a blocking probe/recv
    CollectionType data;
    token.channel()->receiveSerializedProduct(instance_, data);

    // put the data into the Event
    event.emplace(data_, std::move(data));

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(mpiNext_, token);
  }

private:
  edm::EDGetTokenT<MPIToken> const mpiPrev_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const mpiNext_;  // copy of the MPIToken that may be used to implement an ordering relation
  edm::EDPutTokenT<CollectionType> const data_;  // data to be read over the channel and put into the Event
  int32_t const instance_;                       // instance used to identify the source-destination pair
};

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Provenance/interface/EventID.h"
using MPIReceiverEventID = MPIReceiver<edm::EventID>;
DEFINE_FWK_MODULE(MPIReceiverEventID);

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
using MPIReceiverFEDRawDataCollection = MPIReceiver<FEDRawDataCollection>;
DEFINE_FWK_MODULE(MPIReceiverFEDRawDataCollection);
