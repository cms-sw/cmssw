#include <mutex>

#include <boost/preprocessor.hpp>

#include <mpi.h>

#include "macros.h"
#include "messages.h"

MPI_Datatype EDM_MPI_Empty;
MPI_Datatype EDM_MPI_RunAuxiliary;
MPI_Datatype EDM_MPI_LuminosityBlockAuxiliary;
MPI_Datatype EDM_MPI_EventAuxiliary;

MPI_Datatype EDM_MPI_MessageType[EDM_MPI_MessageTagCount_];

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
                   event);                    // edm::EventNumber_t

  EDM_MPI_MessageType[EDM_MPI_Connect] = EDM_MPI_Empty;                                  //
  EDM_MPI_MessageType[EDM_MPI_Disconnect] = EDM_MPI_Empty;                               //
  EDM_MPI_MessageType[EDM_MPI_BeginStream] = EDM_MPI_Empty;                              //
  EDM_MPI_MessageType[EDM_MPI_EndStream] = EDM_MPI_Empty;                                //
  EDM_MPI_MessageType[EDM_MPI_BeginRun] = EDM_MPI_RunAuxiliary;                          //
  EDM_MPI_MessageType[EDM_MPI_EndRun] = EDM_MPI_RunAuxiliary;                            //
  EDM_MPI_MessageType[EDM_MPI_BeginLuminosityBlock] = EDM_MPI_LuminosityBlockAuxiliary;  //
  EDM_MPI_MessageType[EDM_MPI_EndLuminosityBlock] = EDM_MPI_LuminosityBlockAuxiliary;    //
  EDM_MPI_MessageType[EDM_MPI_ProcessEvent] = EDM_MPI_EventAuxiliary;                    //
  EDM_MPI_MessageType[EDM_MPI_SendSerializedProduct] = MPI_BYTE;                         // variable-length binary blob
  EDM_MPI_MessageType[EDM_MPI_SendTrivialProduct] = MPI_BYTE;                            // variable-length binary blob
  EDM_MPI_MessageType[EDM_MPI_SkipProduct] = EDM_MPI_Empty;                              //
  EDM_MPI_MessageType[EDM_MPI_SendComplete] = EDM_MPI_Empty;                             //
}

void EDM_MPI_build_types() {
  static std::once_flag flag;
  std::call_once(flag, EDM_MPI_build_types_);
}
