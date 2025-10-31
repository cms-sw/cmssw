#ifndef HeterogeneousCore_MPICore_plugins_conversion_h
#define HeterogeneousCore_MPICore_plugins_conversion_h

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include "messages.h"

// fill an edm::RunAuxiliary object from an EDM_MPI_RunAuxiliary_t buffer
void edmFromBuffer(EDM_MPI_RunAuxiliary_t const &, edm::RunAuxiliary &);

// fill an EDM_MPI_RunAuxiliary_t buffer from an edm::RunAuxiliary object
void edmToBuffer(EDM_MPI_RunAuxiliary_t &, edm::RunAuxiliary const &);

// fill an edm::LuminosityBlockAuxiliary object from an EDM_MPI_LuminosityBlockAuxiliary_t buffer
void edmFromBuffer(EDM_MPI_LuminosityBlockAuxiliary_t const &, edm::LuminosityBlockAuxiliary &);

// fill an EDM_MPI_LuminosityBlockAuxiliary_t buffer from an edm::LuminosityBlockAuxiliary object
void edmToBuffer(EDM_MPI_LuminosityBlockAuxiliary_t &, edm::LuminosityBlockAuxiliary const &);

// fill an edm::EventAuxiliary object from an EDM_MPI_EventAuxiliary_t buffer
void edmFromBuffer(EDM_MPI_EventAuxiliary_t const &, edm::EventAuxiliary &);

// fill an EDM_MPI_EventAuxiliary_t buffer from an edm::EventAuxiliary object
void edmToBuffer(EDM_MPI_EventAuxiliary_t &, edm::EventAuxiliary const &);

#endif  // HeterogeneousCore_MPICore_plugins_conversion_h
