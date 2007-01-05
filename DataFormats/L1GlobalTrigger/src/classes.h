#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>

namespace { namespace {
    // dictionary for L1GtfeWord
    L1GtfeWord dummy20;
    edm::Wrapper<L1GtfeWord> dummy21;

    // dictionary for L1TcsWord
    L1TcsWord dummy30;
    edm::Wrapper<L1TcsWord> dummy31;

    // dictionary for L1GtFdlWord
    L1GtFdlWord dummy40;
    edm::Wrapper<L1GtFdlWord> dummy41;

    // dictionary for L1 Global Trigger Readout Setup
    L1GlobalTriggerReadoutSetup dummy50;
    edm::Wrapper<L1GlobalTriggerReadoutSetup> dummy51;

    // dictionary for L1 Global Trigger Readout Record
    L1GlobalTriggerReadoutRecord dummy10;
    edm::Wrapper<L1GlobalTriggerReadoutRecord> dummy11;

    // dictionary for L1 Global Trigger EVM Readout Record
    L1GlobalTriggerEvmReadoutRecord dummy60;
    edm::Wrapper<L1GlobalTriggerEvmReadoutRecord> dummy61;
} }

