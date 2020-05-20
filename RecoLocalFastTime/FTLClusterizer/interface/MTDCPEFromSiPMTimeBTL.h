#ifndef RecoLocalFastTime_FTLClusterizer_MTDCPEFromSiPMTimeBTL_H
#define RecoLocalFastTime_FTLClusterizer_MTDCPEFromSiPMTimeBTL_H 1

//-----------------------------------------------------------------------------
// \class        MTDCPEFromSiPMTimeBTL
//-----------------------------------------------------------------------------

#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

class MTDCPEFromSiPMTimeBTL : public MTDCPEBase {
public:
  MTDCPEFromSiPMTimeBTL(edm::ParameterSet const& conf, const MTDGeometry& geom);

private:
  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  LocalPoint localPosition(DetParam const& dp, ClusterParam& cp) const override;
  LocalError localError(DetParam const& dp, ClusterParam& cp) const override;
};

#endif
