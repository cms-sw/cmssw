#include <array>
#include <cstring>

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"

#include "conversion.h"
#include "messages.h"

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

// fill an edm::RunAuxiliary object from an EDM_MPI_RunAuxiliary buffer
void edmFromBuffer(EDM_MPI_RunAuxiliary_t const& buffer, edm::RunAuxiliary& aux) {
  aux = edm::RunAuxiliary(buffer.run, edm::Timestamp(buffer.beginTime), edm::Timestamp(buffer.endTime));
  aux.setProcessHistoryID(
      edm::ProcessHistoryID(std::string(buffer.processHistoryID, std::size(buffer.processHistoryID))));
}

// fill an EDM_MPI_RunAuxiliary buffer from an edm::RunAuxiliary object
void edmToBuffer(EDM_MPI_RunAuxiliary_t& buffer, edm::RunAuxiliary const& aux) {
  copy_and_fill(buffer.processHistoryID, aux.processHistoryID().compactForm());
  buffer.beginTime = aux.beginTime().value();
  buffer.endTime = aux.endTime().value();
  buffer.run = aux.id().run();
}

// fill an edm::LuminosityBlockAuxiliary object from an EDM_MPI_LuminosityBlockAuxiliary buffer
void edmFromBuffer(EDM_MPI_LuminosityBlockAuxiliary_t const& buffer, edm::LuminosityBlockAuxiliary& aux) {
  aux = edm::LuminosityBlockAuxiliary(
      buffer.run, buffer.lumi, edm::Timestamp(buffer.beginTime), edm::Timestamp(buffer.endTime));
  aux.setProcessHistoryID(
      edm::ProcessHistoryID(std::string(buffer.processHistoryID, std::size(buffer.processHistoryID))));
}

// fill an EDM_MPI_LuminosityBlockAuxiliary buffer from an edm::LuminosityBlockAuxiliary object
void edmToBuffer(EDM_MPI_LuminosityBlockAuxiliary_t& buffer, edm::LuminosityBlockAuxiliary const& aux) {
  copy_and_fill(buffer.processHistoryID, aux.processHistoryID().compactForm());
  buffer.beginTime = aux.beginTime().value();
  buffer.endTime = aux.endTime().value();
  buffer.run = aux.id().run();
  buffer.lumi = aux.id().luminosityBlock();
}

// fill an edm::EventAuxiliary object from an EDM_MPI_EventAuxiliary buffer
void edmFromBuffer(EDM_MPI_EventAuxiliary_t const& buffer, edm::EventAuxiliary& aux) {
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
}

// fill an EDM_MPI_EventAuxiliary buffer from an edm::EventAuxiliary object
void edmToBuffer(EDM_MPI_EventAuxiliary_t& buffer, edm::EventAuxiliary const& aux) {
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
}
