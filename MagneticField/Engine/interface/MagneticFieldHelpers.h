#ifndef MagneticFieldHelpers_h
#define MagneticFieldHelpers_h

// #include "CondFormats/RunInfo/interface/RunInfo.h"
// #include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace magneticFieldHelpers  {

  /// Return the closer nominal field value (kGauss) to a given magnet current (A)
  int closerNominalField(float current) {
    int zeroFieldThreshold = 1000; //fixme
    float nominalCurrents[5] = {9558,14416,16819,18268,19262} ; //FIXME: replace with correct values...
    int nominalFields[5] = {20,30,35,38,40} ; //in kGauss
    if(current < zeroFieldThreshold) return 0;
    int i=0;
    for(;i<4;i++)
      {
        if(2*current < nominalCurrents[i]+nominalCurrents[i+1] )
	  return nominalFields[i];
      }
    return nominalFields[i];
  }

//   /// Return the closer nominal field value (kGauss) to the average current stored in the DB.
//   int closerNominalField(const RunInfo & runInfo) {
//     return closerNominalField(runInfo.m_avg_current);
//   }

//   /// Return the closer nominal field value (kGauss) to the average current stored in the DB.
//   int closerNominalField(const edm::EventSetup& es)
//   {
//     edm::ESHandle<RunInfo> sum;
//     es.get<RunInfoRcd>().get(sum);
//     return closerNominalField( *sum.product() );
//   }


}

#endif // MagneticFieldHelpers_h
