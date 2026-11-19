// C++ standard library headers
#include <mutex>

// MPI headers
#include <mpi.h>

// Boost headers
#include <boost/preprocessor.hpp>

// CMSSW headers
#include "HeterogeneousCore/MPICore/interface/messages.h"

// local headers
#include "macros.h"

MPI_Datatype EDM_MPI_Empty;
MPI_Datatype EDM_MPI_RunAuxiliary;
MPI_Datatype EDM_MPI_LuminosityBlockAuxiliary;
MPI_Datatype EDM_MPI_EventAuxiliary;

void EDM_MPI_build_types_() {
  // EDM_MPI_Empty
  DECLARE_MPI_TYPE(EDM_MPI_Empty,    // MPI_Datatype
                   EDM_MPI_Empty_t,  // C++ struct
                   messageTag);      // EDM_MPI_MessageTag

  // EDM_MPI_RunAuxiliary
  DECLARE_MPI_TYPE(EDM_MPI_RunAuxiliary,    // MPI_Datatype
                   EDM_MPI_RunAuxiliary_t,  // C++ struct
                   messageTag,              // EDM_MPI_MessageTag
                   processHistoryID,        // edm::ProcessHistoryID::compactForm()
                   beginTime,               // edm::TimeValue_t
                   endTime,                 // edm::TimeValue_t
                   run);                    // edm::RunNumber_t

  // EDM_MPI_LuminosityBlockAuxiliary
  DECLARE_MPI_TYPE(EDM_MPI_LuminosityBlockAuxiliary,    // MPI_Datatype
                   EDM_MPI_LuminosityBlockAuxiliary_t,  // C++ struct
                   messageTag,                          // EDM_MPI_MessageTag
                   processHistoryID,                    // edm::ProcessHistoryID::compactForm()
                   beginTime,                           // edm::TimeValue_t
                   endTime,                             // edm::TimeValue_t
                   run,                                 // edm::RunNumber_t
                   lumi);                               // edm::LuminosityBlockNumber_t

  // EDM_MPI_EventAuxiliary
  DECLARE_MPI_TYPE(EDM_MPI_EventAuxiliary,    // MPI_Datatype
                   EDM_MPI_EventAuxiliary_t,  // C++ struct
                   messageTag,                // EDM_MPI_MessageTag
                   processHistoryID,          // edm::ProcessHistoryID::compactForm()
                   processGuid,               // process GUID
                   time,                      // edm::TimeValue_t
                   realData,                  // real data (true) vs simulation (false)
                   experimentType,            // edm::EventAuxiliary::ExperimentType
                   bunchCrossing,             // LHC bunch crossing
                   orbitNumber,               // LHC orbit number
                   storeNumber,               // LHC fill number ?
                   run,                       // edm::RunNumber_t
                   lumi,                      // edm::LuminosityBlockNumber_t
                   event,                     // edm::EventNumber_t
                   slotId);                   // MPIChannel index
}

void EDM_MPI_build_types() {
  static std::once_flag flag;
  std::call_once(flag, EDM_MPI_build_types_);
}
