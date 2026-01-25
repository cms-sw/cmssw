#ifndef HeterogeneousCore_MPICore_interface_api_h
#define HeterogeneousCore_MPICore_interface_api_h

// C++ standard library headers
#include <bitset>
#include <iostream>
#include <type_traits>
#include <utility>

// MPI headers
#include <mpi.h>

// ROOT headers
#include <TClass.h>

// CMSSW headers
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include "HeterogeneousCore/MPICore/interface/metadata.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

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

  /*
  // start processing a new event, and receive the EventAuxiliary
  MPI_Status receiveEvent(edm::EventAuxiliary& aux, int source) {
    return receiveEventAuxiliary_(aux, source, EDM_MPI_ProcessEvent);
  }
  */

  // start processing a new event, and receive the EventAuxiliary
  MPI_Status receiveEvent(edm::EventAuxiliary& aux, MPI_Message& message) {
    return receiveEventAuxiliary_(aux, message);
  }

  void sendMetadata(int instance, std::shared_ptr<ProductMetadataBuilder> meta);
  void receiveMetadata(int instance, std::shared_ptr<ProductMetadataBuilder> meta);

  // send buffer of serialized products
  void sendBuffer(const void* buf, size_t size, int instance, EDM_MPI_MessageTag tag);

  // serialize an object of type T using its ROOT dictionary, and transmit it
  template <typename T>
  void sendProduct(int instance, T const& product) {
    if constexpr (std::is_fundamental_v<T>) {
      sendTrivialProduct_(instance, product);
    } else {
      static const TClass* type = TClass::GetClass<T>();
      if (!type) {
        throw std::runtime_error("ROOT dictionary not found for type " + std::string(typeid(T).name()));
      }
      sendSerializedProduct_(instance, type, &product);
    }
  }

  // receive an object of type T, and deserialize it using its ROOT dictionary
  template <typename T>
  void receiveProduct(int instance, T& product) {
    if constexpr (std::is_fundamental_v<T>) {
      receiveTrivialProduct_(instance, product);
    } else {
      static const TClass* type = TClass::GetClass<T>();
      if (!type) {
        throw std::runtime_error("ROOT dictionary not found for type " + std::string(typeid(T).name()));
      }
      receiveSerializedProduct_(instance, type, &product);
    }
  }

  // serialize a generic object using its ROOT dictionary, and send the binary blob
  void sendSerializedProduct_(int instance, TClass const* type, void const* product);

  // receive a binary blob, and deserialize an object of generic type using its ROOT dictionary
  void receiveSerializedProduct_(int instance, TClass const* type, void* product);

  // receive product buffer of known size
  std::unique_ptr<TBufferFile> receiveSerializedBuffer(int instance, int bufSize);

  // transfer a wrapped object using its MemoryCopyTraits
  void sendTrivialCopyProduct(int instance, const ngt::ReaderBase& reader);

  // receive into wrapped object
  void receiveInitializedTrivialCopy(int instance, ngt::WriterBase& writer);

private:
  // serialize an EDM object to a simplified representation that can be transmitted as an MPI message
  void edmToBuffer_(EDM_MPI_RunAuxiliary_t& buffer, edm::RunAuxiliary const& aux);
  void edmToBuffer_(EDM_MPI_LuminosityBlockAuxiliary_t& buffer, edm::LuminosityBlockAuxiliary const& aux);
  void edmToBuffer_(EDM_MPI_EventAuxiliary_t& buffer, edm::EventAuxiliary const& aux);

  // deserialize an EDM object from a simplified representation transmitted as an MPI message
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

  // this is what is used for sending when product is of raw fundamental type
  template <typename T>
  void sendTrivialProduct_(int instance, T const& product) {
    int tag = EDM_MPI_SendTrivialProduct | instance * EDM_MPI_MessageTagWidth_;
    MPI_Send(&product, sizeof(T), MPI_BYTE, dest_, tag, comm_);
  }

  // this is what is used when product is of raw fundamental type
  template <typename T>
  void receiveTrivialProduct_(int instance, T& product) {
    int tag = EDM_MPI_SendTrivialProduct | instance * EDM_MPI_MessageTagWidth_;
    MPI_Message message;
    MPI_Status status;
    MPI_Mprobe(dest_, tag, comm_, &message, &status);
    int size;
    MPI_Get_count(&status, MPI_BYTE, &size);
    assert(static_cast<int>(sizeof(T)) == size);
    MPI_Mrecv(&product, size, MPI_BYTE, &message, MPI_STATUS_IGNORE);
  }

  // MPI intercommunicator
  MPI_Comm comm_ = MPI_COMM_NULL;

  // MPI destination
  int dest_ = MPI_UNDEFINED;
};

#endif  // HeterogeneousCore_MPICore_interface_api_h
