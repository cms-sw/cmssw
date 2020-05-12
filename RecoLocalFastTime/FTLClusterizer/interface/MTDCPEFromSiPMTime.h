#ifndef RecoLocalFastTime_FTLClusterizer_MTDCPEFromSiPMTime_H
#define RecoLocalFastTime_FTLClusterizer_MTDCPEFromSiPMTime_H 1

//-----------------------------------------------------------------------------
// \class        MTDCPEFromSiPMTime
//-----------------------------------------------------------------------------

#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

class MTDCPEFromSiPMTime : public MTDCPEBase {
public:
  MTDCPEFromSiPMTime(edm::ParameterSet const& conf, const MTDGeometry& geom);

private:
  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  LocalPoint localPosition(DetParam const& dp, ClusterParam& cp) const override;
  LocalError localError(DetParam const& dp, ClusterParam& cp) const override;
};

#endif
