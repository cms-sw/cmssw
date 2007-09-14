#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>
#include <map>

namespace { namespace {
    // dictionary for L1GtfeWord
    L1GtfeWord dummy20;
    edm::Wrapper<L1GtfeWord> dummy21;

    // dictionary for L1GtfeExtWord
    L1GtfeExtWord dummy25;
    edm::Wrapper<L1GtfeExtWord> dummy26;

    // dictionary for L1TcsWord
    L1TcsWord dummy30;
    edm::Wrapper<L1TcsWord> dummy31;

    // dictionary for L1GtFdlWord
    L1GtFdlWord dummy40;
    edm::Wrapper<L1GtFdlWord> dummy41;
    
    std::vector<L1GtFdlWord> dummy42;
    edm::Wrapper<std::vector<L1GtFdlWord> > dummy43;

    // dictionary for L1GtPsbWord
    L1GtPsbWord dummy80;
    edm::Wrapper<L1GtPsbWord> dummy81;
    
    std::vector<L1GtPsbWord> dummy82;
    edm::Wrapper<std::vector<L1GtPsbWord> > dummy83;

    // dictionary for L1 Global Trigger Readout Setup    
    L1GlobalTriggerReadoutSetup dummy50;
    edm::Wrapper<L1GlobalTriggerReadoutSetup> dummy51;

    // dictionary for L1 Global Trigger Readout Record
    L1GlobalTriggerReadoutRecord dummy10;
    edm::Wrapper<L1GlobalTriggerReadoutRecord> dummy11;

    // dictionary for L1 Global Trigger EVM Readout Record
    L1GlobalTriggerEvmReadoutRecord dummy60;
    edm::Wrapper<L1GlobalTriggerEvmReadoutRecord> dummy61;

    // dictionary for L1 Global Trigger Object Map
    L1GlobalTriggerObjectMap dummy70;
    edm::Wrapper<L1GlobalTriggerObjectMap> dummy71;

    std::vector<L1GlobalTriggerObjectMap> dummy78;
    edm::Wrapper<std::vector<L1GlobalTriggerObjectMap> > dummy79;

    // dictionary for L1 Global Trigger Object Map Record
    L1GlobalTriggerObjectMapRecord dummy72;
    edm::Wrapper<L1GlobalTriggerObjectMapRecord> dummy73;
    
    std::vector<L1GtObject> dummy100;
    std::vector<std::vector<L1GtObject> > dummy101; 
    
} }

