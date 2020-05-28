#ifndef RecoLocalFastTime_FTLClusterizer_MTDCPESiPMTimeBTLGenericETL_H
#define RecoLocalFastTime_FTLClusterizer_MTDCPESiPMTimeBTLGenericETL_H 1

//-----------------------------------------------------------------------------
// \class        MTDCPESiPMTimeBTLGenericETL
//-----------------------------------------------------------------------------

#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

class MTDCPESiPMTimeBTLGenericETL : public MTDCPEBase {
public:
  MTDCPESiPMTimeBTLGenericETL(edm::ParameterSet const& conf, const MTDGeometry& geom);

private:
  //--------------------------------------------------------------------------
  //! This is where the action happens.
  //--------------------------------------------------------------------------
  LocalPoint localPosition(DetParam const& dp, ClusterParam& cp) const override;
  LocalError localError(DetParam const& dp, ClusterParam& cp) const override;
};

#endif
