// C++ standard library headers
#include <cassert>
#include <cstring>
#include <memory>
#include <string>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/conversion.h"
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

namespace {
  // copy the content of an std::string-like object to an N-sized char buffer:
  // if the string is larger than the buffer, copy only the first N bytes;
  // if the string is smaller than the buffer, fill the rest of the buffer with NUL characters.
  template <typename S, size_t N>
  void copy_and_fill(char (&dest)[N], S const& src) {
    if (std::size(src) < N) {
      memset(dest, 0x00, N);
      memcpy(dest, src.data(), std::size(src));
    } else {
      memcpy(dest, src.data(), N);
    }
  }
}  // namespace

// build a new MPIChannel that uses a duplicate of the underlying communicator and the same destination
std::unique_ptr<MPIChannel> MPIChannel::duplicate(int slot) const {
  // This is a blocking collective operation.
  MPI_Comm newcomm;
  MPI_Comm_dup(comm_, &newcomm);
  auto channel = std::make_unique<MPIChannel>(newcomm, dest_);
  channel->status_.store(kReady, std::memory_order_release);
  channel->slot_ = slot;
  LogDebug("MPI") << "channel " << slot << " transitioned to kReady";
  return channel;
}

// mark the channel as busy
void MPIChannel::acquire() {
  auto status = status_.load(std::memory_order_acquire);
  if (status != kReady) {
    throw cms::Exception("MPI") << "MPIChannel " << slot_ << " is in an invalide state";
  }
  assert(request_ == MPI_REQUEST_NULL);
  status_.store(kBusy, std::memory_order_release);
  LogDebug("MPI") << "channel " << slot_ << " transitioned to kBusy";
}

// make sure both processes have completed processing the event that was transmitted
// Note: this is a non-blocking collective operation.
void MPIChannel::sync() {
  auto status = status_.load(std::memory_order_acquire);
  if (status != kBusy) {
    throw cms::Exception("MPI") << "MPIChannel " << slot_ << " is in an invalide state";
  }
  assert(request_ == MPI_REQUEST_NULL);
  MPI_Ibarrier(comm_, &request_);
  assert(request_ != MPI_REQUEST_NULL);
  status_.store(kSync, std::memory_order_release);
  LogDebug("MPI") << "channel " << slot_ << " transitioned to kSync";
}

// check whether this channel can be used to transmit a new event
bool MPIChannel::ready() {
  auto status = status_.load(std::memory_order_acquire);
  if (status == kInvalid) {
    throw cms::Exception("MPI") << "MPIChannel " << slot_ << " is in an invalide state";
  } else if (status == kReady) {
    return true;
  } else if (status == kBusy) {
    return false;
  } else if (status == kSync) {
    int flag = 0;
    assert(request_ != MPI_REQUEST_NULL);
    // TODO check status and return value
    MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      // if the barrier was reached, MPI_Test resets the request object to MPI_REQUEST_NULL
      assert(request_ == MPI_REQUEST_NULL);
      status_.store(kReady, std::memory_order_release);
      LogDebug("MPI") << "channel " << slot_ << " transitioned to kReady";
      return true;
    } else {
      assert(request_ != MPI_REQUEST_NULL);
      LogDebug("MPI") << "channel " << slot_ << " in kSync DID NOT transition to kReady";
      return false;
    }
  }
  __builtin_unreachable();
}

// wait until this channel can be used to transmit a new event
void MPIChannel::wait() {
  auto status = status_.load(std::memory_order_acquire);
  if (status == kInvalid) {
    throw cms::Exception("MPI") << "MPIChannel " << slot_ << " is in an invalide state";
  } else if (status == kReady) {
    return;
  } else if (status == kBusy) {
    throw cms::Exception("MPI") << "MPIChannel::wait() cannot resolve a kBusy status";
    return;
  } else if (status == kSync) {
    assert(request_ != MPI_REQUEST_NULL);
    // TODO check status and return value
    MPI_Wait(&request_, MPI_STATUS_IGNORE);
    // if the barrier was reached, MPI_Test resets the request object to MPI_REQUEST_NULL
    assert(request_ == MPI_REQUEST_NULL);
    status_.store(kReady, std::memory_order_release);
    LogDebug("MPI") << "channel " << slot_ << " transitioned to kReady";
    return;
  }
  __builtin_unreachable();
}

// close the underlying communicator and reset the MPIChannel to an invalid state
void MPIChannel::reset() {
  // This is a blocking collective operation.
  MPI_Comm_disconnect(&comm_);
  dest_ = MPI_UNDEFINED;
}

// fill an edm::RunAuxiliary object from an EDM_MPI_RunAuxiliary buffer
void MPIChannel::edmFromBuffer_(EDM_MPI_RunAuxiliary_t const& buffer, edm::RunAuxiliary& aux) {
  aux = edm::RunAuxiliary(buffer.run, edm::Timestamp(buffer.beginTime), edm::Timestamp(buffer.endTime));
  aux.setProcessHistoryID(
      edm::ProcessHistoryID(std::string(buffer.processHistoryID, std::size(buffer.processHistoryID))));
}

// fill an EDM_MPI_RunAuxiliary buffer from an edm::RunAuxiliary object
void MPIChannel::edmToBuffer_(EDM_MPI_RunAuxiliary_t& buffer, edm::RunAuxiliary const& aux) {
  copy_and_fill(buffer.processHistoryID, aux.processHistoryID().compactForm());
  buffer.beginTime = aux.beginTime().value();
  buffer.endTime = aux.endTime().value();
  buffer.run = aux.id().run();
}

// fill an edm::LuminosityBlockAuxiliary object from an EDM_MPI_LuminosityBlockAuxiliary buffer
void MPIChannel::edmFromBuffer_(EDM_MPI_LuminosityBlockAuxiliary_t const& buffer, edm::LuminosityBlockAuxiliary& aux) {
  aux = edm::LuminosityBlockAuxiliary(
      buffer.run, buffer.lumi, edm::Timestamp(buffer.beginTime), edm::Timestamp(buffer.endTime));
  aux.setProcessHistoryID(
      edm::ProcessHistoryID(std::string(buffer.processHistoryID, std::size(buffer.processHistoryID))));
}

// fill an EDM_MPI_LuminosityBlockAuxiliary buffer from an edm::LuminosityBlockAuxiliary object
void MPIChannel::edmToBuffer_(EDM_MPI_LuminosityBlockAuxiliary_t& buffer, edm::LuminosityBlockAuxiliary const& aux) {
  copy_and_fill(buffer.processHistoryID, aux.processHistoryID().compactForm());
  buffer.beginTime = aux.beginTime().value();
  buffer.endTime = aux.endTime().value();
  buffer.run = aux.id().run();
  buffer.lumi = aux.id().luminosityBlock();
}

// fill an edm::EventAuxiliary object from an EDM_MPI_EventAuxiliary buffer
void MPIChannel::edmFromBuffer_(EDM_MPI_EventAuxiliary_t const& buffer, edm::EventAuxiliary& aux, unsigned int& slot) {
  aux = edm::EventAuxiliary({buffer.run, buffer.lumi, buffer.event},
                            std::string(buffer.processGuid, std::size(buffer.processGuid)),
                            edm::Timestamp(buffer.time),
                            buffer.realData,
                            static_cast<edm::EventAuxiliary::ExperimentType>(buffer.experimentType),
                            buffer.bunchCrossing,
                            buffer.storeNumber,
                            buffer.orbitNumber);
  aux.setProcessHistoryID(
      edm::ProcessHistoryID(std::string(buffer.processHistoryID, std::size(buffer.processHistoryID))));
  slot = buffer.slotId;
}

// fill an EDM_MPI_EventAuxiliary buffer from an edm::EventAuxiliary object
void MPIChannel::edmToBuffer_(EDM_MPI_EventAuxiliary_t& buffer, edm::EventAuxiliary const& aux, unsigned int slot) {
  copy_and_fill(buffer.processHistoryID, aux.processHistoryID().compactForm());
  copy_and_fill(buffer.processGuid, aux.processGUID());
  buffer.time = aux.time().value();
  buffer.realData = aux.isRealData();
  buffer.experimentType = aux.experimentType();
  buffer.bunchCrossing = aux.bunchCrossing();
  buffer.orbitNumber = aux.orbitNumber();
  buffer.storeNumber = aux.storeNumber();
  buffer.run = aux.id().run();
  buffer.lumi = aux.id().luminosityBlock();
  buffer.event = aux.id().event();
  buffer.slotId = slot;
}

// fill and send an EDM_MPI_Empty_t buffer
void MPIChannel::sendEmpty_(int tag) {
  EDM_MPI_Empty_t buffer;
  buffer.messageTag = tag;
  MPI_Send(&buffer, 1, EDM_MPI_Empty, dest_, tag, comm_);
}

// fill and send an EDM_MPI_RunAuxiliary_t buffer
void MPIChannel::sendRunAuxiliary_(int tag, edm::RunAuxiliary const& aux) {
  EDM_MPI_RunAuxiliary_t buffer;
  buffer.messageTag = tag;
  edmToBuffer_(buffer, aux);
  MPI_Send(&buffer, 1, EDM_MPI_RunAuxiliary, dest_, tag, comm_);
}

// fill and send an EDM_MPI_RunAuxiliary_t buffer
void MPIChannel::sendLuminosityBlockAuxiliary_(int tag, edm::LuminosityBlockAuxiliary const& aux) {
  EDM_MPI_LuminosityBlockAuxiliary_t buffer;
  buffer.messageTag = tag;
  edmToBuffer_(buffer, aux);
  MPI_Send(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, dest_, tag, comm_);
}

// fill and send an EDM_MPI_EventAuxiliary_t buffer
void MPIChannel::sendEventAuxiliary_(edm::EventAuxiliary const& aux, unsigned int slot) {
  EDM_MPI_EventAuxiliary_t buffer;
  buffer.messageTag = EDM_MPI_ProcessEvent;
  edmToBuffer_(buffer, aux, slot);
  MPI_Send(&buffer, 1, EDM_MPI_EventAuxiliary, dest_, EDM_MPI_ProcessEvent, comm_);
}

/*
// receive an EDM_MPI_EventAuxiliary_t buffer and populate an edm::EventAuxiliary
MPI_Status MPIChannel::receiveEventAuxiliary_(edm::EventAuxiliary& aux, unsigned int& slot, int source, int tag) {
  MPI_Status status;
  EDM_MPI_EventAuxiliary_t buffer;
  MPI_Recv(&buffer, 1, EDM_MPI_EventAuxiliary, source, tag, comm_, &status);
  edmFromBuffer_(buffer, aux, slot);
  return status;
}
*/

// receive an EDM_MPI_EventAuxiliary_t buffer and populate an edm::EventAuxiliary
MPI_Status MPIChannel::receiveEventAuxiliary_(edm::EventAuxiliary& aux, unsigned int& slot, MPI_Message& message) {
  MPI_Status status = {};
  EDM_MPI_EventAuxiliary_t buffer;
#ifdef EDM_ML_DEBUG
  memset(&buffer, 0x00, sizeof(buffer));
#endif
  status.MPI_ERROR = MPI_Mrecv(&buffer, 1, EDM_MPI_EventAuxiliary, &message, &status);
  edmFromBuffer_(buffer, aux, slot);
  return status;
}

void MPIChannel::sendMetadata(int instance, std::shared_ptr<ProductMetadataBuilder> meta) {
  int tag = EDM_MPI_SendMetadata | instance * EDM_MPI_MessageTagWidth_;
  MPI_Ssend(meta->data(), meta->size(), MPI_BYTE, dest_, tag, comm_);
}

void MPIChannel::receiveMetadata(int instance, std::shared_ptr<ProductMetadataBuilder> meta) {
  int tag = EDM_MPI_SendMetadata | instance * EDM_MPI_MessageTagWidth_;
  meta->receiveMetadata(dest_, tag, comm_);
}

void MPIChannel::sendBuffer(const void* buf, size_t size, int instance, EDM_MPI_MessageTag tag) {
  int commtag = tag | instance * EDM_MPI_MessageTagWidth_;
  MPI_Send(buf, size, MPI_BYTE, dest_, commtag, comm_);
}

void MPIChannel::sendSerializedProduct_(int instance, TClass const* type, void const* product) {
  TBufferFile buffer{TBuffer::kWrite};
  type->Streamer(const_cast<void*>(product), buffer);
  int tag = EDM_MPI_SendSerializedProduct | instance * EDM_MPI_MessageTagWidth_;
  MPI_Send(buffer.Buffer(), buffer.Length(), MPI_BYTE, dest_, tag, comm_);
}

std::unique_ptr<TBufferFile> MPIChannel::receiveSerializedBuffer(int instance, int bufSize) {
  int tag = EDM_MPI_SendSerializedProduct | instance * EDM_MPI_MessageTagWidth_;
  MPI_Status status;
  auto buffer = std::make_unique<TBufferFile>(TBuffer::kRead, bufSize);
#ifdef EDM_ML_DEBUG
  memset(buffer->Buffer(), 0xff, buffer->BufferSize());
#endif
  MPI_Recv(buffer->Buffer(), bufSize, MPI_BYTE, dest_, tag, comm_, &status);
  int receivedCount = 0;
  MPI_Get_count(&status, MPI_BYTE, &receivedCount);
  assert(receivedCount == bufSize && "received serialized buffer size mismatches the size expected from metadata");
  return buffer;
}

void MPIChannel::receiveSerializedProduct_(int instance, TClass const* type, void* product) {
  int tag = EDM_MPI_SendSerializedProduct | instance * EDM_MPI_MessageTagWidth_;
  MPI_Message message;
  MPI_Status status;
  MPI_Mprobe(dest_, tag, comm_, &message, &status);
  int size;
  MPI_Get_count(&status, MPI_BYTE, &size);
  TBufferFile buffer{TBuffer::kRead, size};
  MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
  type->Streamer(product, buffer);
}

void MPIChannel::sendTrivialCopyProduct(int instance, const ngt::ReaderBase& reader) {
  int tag = EDM_MPI_SendTrivialCopyProduct | instance * EDM_MPI_MessageTagWidth_;
  // transfer the memory regions
  auto regions = reader.regions();
  // TODO send the number of regions ?
  for (size_t i = 0; i < regions.size(); ++i) {
    assert(regions[i].data() != nullptr);
    MPI_Send(regions[i].data(), regions[i].size_bytes(), MPI_BYTE, dest_, tag, comm_);
  }
}

void MPIChannel::receiveInitializedTrivialCopy(int instance, ngt::WriterBase& writer) {
  int tag = EDM_MPI_SendTrivialCopyProduct | instance * EDM_MPI_MessageTagWidth_;
  MPI_Status status;
  // receive the memory regions
  auto regions = writer.regions();
  // TODO receive and validate the number of regions ?
  for (size_t i = 0; i < regions.size(); ++i) {
    assert(regions[i].data() != nullptr);
    MPI_Recv(regions[i].data(), regions[i].size_bytes(), MPI_BYTE, dest_, tag, comm_, &status);
  }
}
