#include "DataFormats/Common/interface/Wrapper.h"

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTBtiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSPhiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSThetaTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

#include <vector>
#include <map>

namespace
{
  namespace
  {
    edm::Wrapper< DTBtiTrigger >                   wBTI;
    std::vector< DTBtiTrigger >                    vBTI;
    edm::Wrapper< std::vector< DTBtiTrigger > >    wvBTI;

    edm::Wrapper< DTTSPhiTrigger >                 wTSPhi;
    std::vector< DTTSPhiTrigger >                  vTSPhi;
    edm::Wrapper< std::vector< DTTSPhiTrigger > >  wvTSPhi;

    edm::Wrapper< DTMatch >                 wDTM;
    std::vector< DTMatch >                  vDTM;
    edm::Wrapper< std::vector< DTMatch > >  wVDTM;
    edm::Ptr< DTMatch >                     ptrDTM;   

    std::string                STR;
    edm::Wrapper<std::string>  wSTR;

    edm::Wrapper< DTMatchPt >                            wDTMPt;
    std::pair< std::string, DTMatchPt >                  pDTMPt;
    edm::Wrapper< std::pair< std::string, DTMatchPt > >  wpDTMPt;
    std::map< std::string, DTMatchPt >                   mDTMPt;
    edm::Wrapper< std::map< std::string, DTMatchPt > >   wmDTMPt;
  }
}

