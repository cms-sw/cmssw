#ifndef HLTReco_TriggerTypeDefs_h
#define HLTReco_TriggerTypeDefs_h

/** \class trigger::TriggerTypeDefs
 *
 *  Misc. common simple typedefs
 *
 *  $Date: 2007/12/05 14:24:02 $
 *  $Revision: 1.4 $
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
    TriggerPhoton   = 80,
    TriggerElectron = 81,
    TriggerMuon     = 82,
    TriggerTau      = 83,
    TriggerJet      = 84,
    TriggerBJet     = 85,
    TriggerMET      = 86,
    TriggerHT       = 87,
    TriggerTrack    = 88,
    TriggerCluster  = 89
  };

}

#endif
