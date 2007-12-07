#ifndef HLTReco_TriggerTypeDefs_h
#define HLTReco_TriggerTypeDefs_h

/** \class trigger::TriggerTypeDefs
 *
 *  Misc. common simple typedefs
 *
 *  $Date: 2007/12/07 13:39:48 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include<vector>

namespace trigger
{

  typedef uint16_t size_type;
  typedef std::vector<size_type> Keys;

  typedef std::vector<int>       Vids;

  enum TriggerObjectType  {

    /// enum start value shifted to 81 so as to avoid clashes with PDG codes

    /// L1 - using cases as defined in enum L1GtObject, file:
    /// DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

    TriggerL1Mu        = 81,
    TriggerL1NoIsoEG   = 82,
    TriggerL1IsoEG     = 83,
    TriggerL1CenJet    = 84,
    TriggerL1ForJet    = 85,
    TriggerL1TauJet    = 86,
    TriggerL1ETM       = 87,
    TriggerL1ETT       = 88,
    TriggerL1HTT       = 89,
    TriggerL1JetCounts = 90,

    /// HLT

    TriggerPhoton      = 91,
    TriggerElectron    = 92,
    TriggerMuon        = 93,
    TriggerTau         = 94,
    TriggerJet         = 95,
    TriggerBJet        = 96,
    TriggerMET         = 97,
    TriggerHT          = 98,
    TriggerTrack       = 99,
    TriggerCluster     =100

  };

}

#endif
