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

class MPISender {
public:
  MPISender() = default;
  MPISender(MPI_Comm comm, int destination) : comm_(comm), dest_(destination) {}

  // announce that a client has just connected
  void sendConnect(int stream) { sendEmpty_(EDM_MPI_Connect, stream); }

  // announce that the client will disconnect
  void sendDisconnect(int stream) { sendEmpty_(EDM_MPI_Disconnect, stream); }

  // signal the begin of stream
  void sendBeginStream(int stream) { sendEmpty_(EDM_MPI_BeginStream, stream); }

  // signal the end of stream
  void sendEndStream(int stream) { sendEmpty_(EDM_MPI_EndStream, stream); }

  // signal a new run, and transmit the RunAuxiliary
  void sendBeginRun(int stream, edm::RunAuxiliary const& aux) { sendRunAuxiliary_(EDM_MPI_BeginRun, stream, aux); }

  // signal the end of run, and re-transmit the RunAuxiliary
  void sendEndRun(int stream, edm::RunAuxiliary const& aux) { sendRunAuxiliary_(EDM_MPI_EndRun, stream, aux); }

  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  void sendBeginLuminosityBlock(int stream, edm::LuminosityBlockAuxiliary const& aux) {
    sendLuminosityBlockAuxiliary_(EDM_MPI_BeginLuminosityBlock, stream, aux);
  }

  // signal the end of luminosity block, and re-transmit the LuminosityBlockAuxiliary
  void sendEndLuminosityBlock(int stream, edm::LuminosityBlockAuxiliary const& aux) {
    sendLuminosityBlockAuxiliary_(EDM_MPI_EndLuminosityBlock, stream, aux);
  }

  // signal a new event, and transmit the EventAuxiliary
  void sendEvent(int stream, edm::EventAuxiliary const& aux) { sendEventAuxiliary_(stream, aux); }

  // start processing a new event, and receive the EventAuxiliary
  std::tuple<MPI_Status, int> receiveEvent(edm::EventAuxiliary& aux, int source) {
    return receiveEventAuxiliary_(aux, source, EDM_MPI_ProcessEvent);
  }

  std::tuple<MPI_Status, int> receiveEvent(edm::EventAuxiliary& aux, MPI_Message& message) {
    return receiveEventAuxiliary_(aux, message);
  }

  // serialize an object of type T using its ROOT dictionary, and transmit it
  template <typename T>
  void sendSerializedProduct(int stream, T const& product) {
    static const TClass* type = TClass::GetClass<T>();
    sendSerializedProduct_(stream, type, &product);
  }

  // serialize an object of generic type using its ROOT dictionary, and transmit it
  void sendSerializedProduct(int stream, edm::ObjectWithDict const& product) {
    // the expected use case is that the product type corresponds to the actual type,
    // so we access the type with typeOf() instead of dynamicType()
    sendSerializedProduct_(stream, product.typeOf().getClass(), product.address());
  }

  // signal that an expected product will not be transmitted
  void sendSkipProduct(int stream) { sendEmpty_(EDM_MPI_SkipProduct, stream); }

  // signal that tre transmission of multiple products is complete
  void sendComplete(int stream) { sendEmpty_(EDM_MPI_SendComplete, stream); }

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
  void sendEmpty_(int tag, int stream);

  // fill and send an EDM_MPI_RunAuxiliary_t buffer
  void sendRunAuxiliary_(int tag, int stream, edm::RunAuxiliary const& aux);

  // fill and send an EDM_MPI_LuminosityBlock_t buffer
  void sendLuminosityBlockAuxiliary_(int tag, int stream, edm::LuminosityBlockAuxiliary const& aux);

  // fill and send an EDM_MPI_EventAuxiliary_t buffer
  void sendEventAuxiliary_(int stream, edm::EventAuxiliary const& aux);

  // receive an EDM_MPI_EventAuxiliary_t buffer and populate an edm::EventAuxiliary
  std::tuple<MPI_Status, int> receiveEventAuxiliary_(edm::EventAuxiliary& aux, int source, int tag);
  std::tuple<MPI_Status, int> receiveEventAuxiliary_(edm::EventAuxiliary& aux, MPI_Message& message);

  // serialize a generic object using its ROOT dictionary, and send the binary blob
  void sendSerializedProduct_(int stream, TClass const* type, void const* product);

  // MPI intercommunicator
  MPI_Comm comm_ = MPI_COMM_NULL;

  // MPI destination
  int dest_ = 0;
};

#endif  // HeterogeneousCore_MPICore_plugins_api_h
