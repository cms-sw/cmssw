#ifndef HeterogeneousCore_MPICore_plugins_api_h
#define HeterogeneousCore_MPICore_plugins_api_h

// externals headers
#include <mpi.h>

// ROOT headers
#include <TClass.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"

// local headers
#include "messages.h"

class MPIChannel {
public:
  MPIChannel() = default;
  MPIChannel(MPI_Comm comm, int destination) : comm_(comm), dest_(destination) {}

  // build a new MPIChannel that uses a duplicate of the underlying communicator and the same destination
  MPIChannel duplicate() const;

  // close the underlying communicator and reset the MPIChannel to an invalid state
  void reset();

  // announce that a client has just connected
  void sendConnect() { sendEmpty_(EDM_MPI_Connect); }

  // announce that the client will disconnect
  void sendDisconnect() { sendEmpty_(EDM_MPI_Disconnect); }

  // signal the begin of stream
  void sendBeginStream() { sendEmpty_(EDM_MPI_BeginStream); }

  // signal the end of stream
  void sendEndStream() { sendEmpty_(EDM_MPI_EndStream); }

  // signal a new run, and transmit the RunAuxiliary
  void sendBeginRun(edm::RunAuxiliary const& aux) { sendRunAuxiliary_(EDM_MPI_BeginRun, aux); }

  // signal the end of run, and re-transmit the RunAuxiliary
  void sendEndRun(edm::RunAuxiliary const& aux) { sendRunAuxiliary_(EDM_MPI_EndRun, aux); }

  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  void sendBeginLuminosityBlock(edm::LuminosityBlockAuxiliary const& aux) {
    sendLuminosityBlockAuxiliary_(EDM_MPI_BeginLuminosityBlock, aux);
  }

  // signal the end of luminosity block, and re-transmit the LuminosityBlockAuxiliary
  void sendEndLuminosityBlock(edm::LuminosityBlockAuxiliary const& aux) {
    sendLuminosityBlockAuxiliary_(EDM_MPI_EndLuminosityBlock, aux);
  }

  // signal a new event, and transmit the EventAuxiliary
  void sendEvent(edm::EventAuxiliary const& aux) { sendEventAuxiliary_(aux); }

  // start processing a new event, and receive the EventAuxiliary
  MPI_Status receiveEvent(edm::EventAuxiliary& aux, int source) {
    return receiveEventAuxiliary_(aux, source, EDM_MPI_ProcessEvent);
  }

  MPI_Status receiveEvent(edm::EventAuxiliary& aux, MPI_Message& message) {
    return receiveEventAuxiliary_(aux, message);
  }

  // serialize an object of type T using its ROOT dictionary, and transmit it
  template <typename T>
  void sendSerializedProduct(int instance, T const& product) {
    static const TClass* type = TClass::GetClass<T>();
    sendSerializedProduct_(instance, type, &product);
  }

  // serialize an object of generic type using its ROOT dictionary, and transmit it
  void sendSerializedProduct(int instance, edm::ObjectWithDict const& product) {
    // the expected use case is that the product type corresponds to the actual type,
    // so we access the type with typeOf() instead of dynamicType()
    sendSerializedProduct_(instance, product.typeOf().getClass(), product.address());
  }

  // signal that an expected product will not be transmitted
  void sendSkipProduct() { sendEmpty_(EDM_MPI_SkipProduct); }

  // signal that the transmission of multiple products is complete
  void sendComplete() { sendEmpty_(EDM_MPI_SendComplete); }

  // receive and object of type T, and deserialize it using its ROOT dictionary
  template <typename T>
  void receiveSerializedProduct(int instance, T& product) {
    static const TClass* type = TClass::GetClass<T>();
    receiveSerializedProduct_(instance, type, &product);
  }

  // receive and object of generic type, and deserialize it using its ROOT dictionary
  void receiveSerializedProduct(int instance, edm::ObjectWithDict& product) {
    // the expected use case is that the product type corresponds to the actual type,
    // so we access the type with typeOf() instead of dynamicType()
    receiveSerializedProduct_(instance, product.typeOf().getClass(), product.address());
  }

private:
  // serialize an EDM object to a simplified representation that can be transmitted as an MPI message
  void edmToBuffer_(EDM_MPI_RunAuxiliary_t& buffer, edm::RunAuxiliary const& aux);
  void edmToBuffer_(EDM_MPI_LuminosityBlockAuxiliary_t& buffer, edm::LuminosityBlockAuxiliary const& aux);
  void edmToBuffer_(EDM_MPI_EventAuxiliary_t& buffer, edm::EventAuxiliary const& aux);

  // dwserialize an EDM object from a simplified representation transmitted as an MPI message
  void edmFromBuffer_(EDM_MPI_RunAuxiliary_t const& buffer, edm::RunAuxiliary& aux);
  void edmFromBuffer_(EDM_MPI_LuminosityBlockAuxiliary_t const& buffer, edm::LuminosityBlockAuxiliary& aux);
  void edmFromBuffer_(EDM_MPI_EventAuxiliary_t const& buffer, edm::EventAuxiliary& aux);

  // fill and send an EDM_MPI_Empty_t buffer
  void sendEmpty_(int tag);

  // fill and send an EDM_MPI_RunAuxiliary_t buffer
  void sendRunAuxiliary_(int tag, edm::RunAuxiliary const& aux);

  // fill and send an EDM_MPI_LuminosityBlock_t buffer
  void sendLuminosityBlockAuxiliary_(int tag, edm::LuminosityBlockAuxiliary const& aux);

  // fill and send an EDM_MPI_EventAuxiliary_t buffer
  void sendEventAuxiliary_(edm::EventAuxiliary const& aux);

  // receive an EDM_MPI_EventAuxiliary_t buffer and populate an edm::EventAuxiliary
  MPI_Status receiveEventAuxiliary_(edm::EventAuxiliary& aux, int source, int tag);
  MPI_Status receiveEventAuxiliary_(edm::EventAuxiliary& aux, MPI_Message& message);

  // serialize a generic object using its ROOT dictionary, and send the binary blob
  void sendSerializedProduct_(int instance, TClass const* type, void const* product);

  // receive a binary blob, and deserialize an object of generic type using its ROOT dictionary
  void receiveSerializedProduct_(int instance, TClass const* type, void* product);

  // MPI intercommunicator
  MPI_Comm comm_ = MPI_COMM_NULL;

  // MPI destination
  int dest_ = MPI_UNDEFINED;
};

#endif  // HeterogeneousCore_MPICore_plugins_api_h
