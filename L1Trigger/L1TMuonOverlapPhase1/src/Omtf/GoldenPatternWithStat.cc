#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "TH1.h"
#include <string>

#include "boost/multi_array/base.hpp"
#include "boost/multi_array/subarray.hpp"

////////////////////////////////////////////////////
////////////////////////////////////////////////////

GoldenPatternWithStat::GoldenPatternWithStat(const Key& aKey,
                                             unsigned int nLayers,
                                             unsigned int nRefLayers,
                                             unsigned int nPdfAddrBits)
    : GoldenPatternWithThresh(aKey, nLayers, nRefLayers, nPdfAddrBits),
      //*8 is to have the 1024 bins for the phiDist, which allows to count the largest values for the low pT muons
      statistics(boost::extents[nLayers][nRefLayers][(1 << nPdfAddrBits) * 8][STAT_BINS]){

      };

////////////////////////////////////////////////////
////////////////////////////////////////////////////
GoldenPatternWithStat::GoldenPatternWithStat(const Key& aKey, const OMTFConfiguration* omtfConfig)
    : GoldenPatternWithThresh(aKey, omtfConfig),
      statistics(boost::extents[omtfConfig->nLayers()][omtfConfig->nRefLayers()][omtfConfig->nPdfBins()][STAT_BINS]){

      };

////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPatternWithStat::updateStat(
    unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, unsigned int what, double value) {
  statistics[iLayer][iRefLayer][iBin][what] += value;
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" iBin "<<iBin<<" what "<<what<<" value "<<value<<std::endl;
}
