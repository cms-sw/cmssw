#ifndef HeterogeneousCore_MPICore_plugins_messages_h
#define HeterogeneousCore_MPICore_plugins_messages_h

#include <cstdint>

#include <mpi.h>

/* register the MPI message types forthe EDM communication
 */
void EDM_MPI_build_types();

/* MPI message tags corresponding to EDM transitions
 */
enum EDM_MPI_MessageTag {
  EDM_MPI_Connect,
  EDM_MPI_Disconnect,
  EDM_MPI_BeginStream,
  EDM_MPI_EndStream,
  EDM_MPI_BeginRun,
  EDM_MPI_EndRun,
  EDM_MPI_BeginLuminosityBlock,
  EDM_MPI_EndLuminosityBlock,
  EDM_MPI_ProcessEvent,
  EDM_MPI_SendSerializedProduct,
  EDM_MPI_SendTrivialProduct,
  EDM_MPI_SkipProduct,
  EDM_MPI_SendComplete,
  EDM_MPI_MessageTagCount_
};

/* Ensure that the MPI message tags can fit in a single byte
 */
inline constexpr int EDM_MPI_MessageTagWidth_ = 256;
static_assert(EDM_MPI_MessageTagCount_ <= EDM_MPI_MessageTagWidth_);

extern MPI_Datatype EDM_MPI_MessageType[EDM_MPI_MessageTagCount_];

/* Common header for EDM MPI messages, containing
 *   - the message type (to allow decoding the message further)
 */
struct __attribute__((aligned(8))) EDM_MPI_Header_t {
  uint32_t messageTag;  // EDM_MPI_MessageTag
};

/* Empty EDM MPI message, used when only the header is needed
 */
struct EDM_MPI_Empty_t : public EDM_MPI_Header_t {};

// corresponding MPI type
extern MPI_Datatype EDM_MPI_Empty;

/* Run information stored in edm::RunAuxiliary,
 * augmented with the MPI message id
 *
 * See DataFormats/Provenance/interface/RunAuxiliary.h
 */
struct EDM_MPI_RunAuxiliary_t : public EDM_MPI_Header_t {
  // from DataFormats/Provenance/interface/RunAuxiliary.h
  char processHistoryID[16];  // edm::ProcessHistoryID::compactForm()
  uint64_t beginTime;         // edm::TimeValue_t
  uint64_t endTime;           // edm::TimeValue_t
  uint32_t run;               // edm::RunNumber_t
};

// corresponding MPI type
extern MPI_Datatype EDM_MPI_RunAuxiliary;

/* LuminosityBlock information stored in edm::LuminosityBlockAuxiliary,
 * augmented with the MPI message id
 *
 * See DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h
 */
struct EDM_MPI_LuminosityBlockAuxiliary_t : public EDM_MPI_Header_t {
  // from DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h
  char processHistoryID[16];  // edm::ProcessHistoryID::compactForm()
  uint64_t beginTime;         // edm::TimeValue_t
  uint64_t endTime;           // edm::TimeValue_t
  uint32_t run;               // edm::RunNumber_t
  uint32_t lumi;              // edm::LuminosityBlockNumber_t
};

// corresponding MPI type
extern MPI_Datatype EDM_MPI_LuminosityBlockAuxiliary;

/* Event information stored in edm::EventAuxiliary,
 * augmented with the MPI message id
 *
 * See DataFormats/Provenance/interface/EventAuxiliary.h
 */
struct EDM_MPI_EventAuxiliary_t : public EDM_MPI_Header_t {
  // from DataFormats/Provenance/interface/EventAuxiliary.h
  char processHistoryID[16];  // edm::ProcessHistoryID::compactForm()
  char processGuid[16];       // process GUID
  uint64_t time;              // edm::TimeValue_t
  int32_t realData;           // real data (true) vs simulation (false)
  int32_t experimentType;     // edm::EventAuxiliary::ExperimentType
  int32_t bunchCrossing;      // LHC bunch crossing
  int32_t orbitNumber;        // LHC orbit number
  int32_t storeNumber;        // LHC fill number ?
  uint32_t run;               // edm::RunNumber_t
  uint32_t lumi;              // edm::LuminosityBlockNumber_t
  uint32_t event;             // edm::EventNumber_t
};

// corresponding MPI type
extern MPI_Datatype EDM_MPI_EventAuxiliary;

// union of all possible messages
union EDM_MPI_Any_t {
  EDM_MPI_Header_t header;
  EDM_MPI_Empty_t empty;
  EDM_MPI_RunAuxiliary_t runAuxiliary;
  EDM_MPI_LuminosityBlockAuxiliary_t luminosityBlockAuxiliary;
  EDM_MPI_EventAuxiliary_t eventAuxiliary;
};

#endif  // HeterogeneousCore_MPICore_plugins_messages_h
