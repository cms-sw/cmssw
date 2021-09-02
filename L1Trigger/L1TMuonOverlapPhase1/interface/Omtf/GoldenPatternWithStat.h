#ifndef L1T_OmtfP1_GoldenPatternWithStat_H
#define L1T_OmtfP1_GoldenPatternWithStat_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"
#include <vector>
#include <ostream>

#include "TH1I.h"

class OMTFConfiguration;

//////////////////////////////////
// Golden Pattern
//////////////////////////////////

class GoldenPatternWithStat : public GoldenPatternWithThresh {
public:
  static const unsigned int STAT_BINS = 1;  //TODO change to 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  typedef boost::multi_array<float, 4> StatArrayType;
  //GoldenPatternWithStat(const Key & aKey) : GoldenPattern(aKey) {}

  GoldenPatternWithStat(const Key& aKey, unsigned int nLayers, unsigned int nRefLayers, unsigned int nPdfAddrBits);

  GoldenPatternWithStat(const Key& aKey, const OMTFConfiguration* omtfConfig);

  ~GoldenPatternWithStat() override{};

  virtual void updateStat(
      unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, unsigned int what, double value);

  //virtual void updatePdfs(double learingRate);

  friend std::ostream& operator<<(std::ostream& out, const GoldenPatternWithStat& aPattern);

  friend class PatternOptimizerBase;
  friend class PatternGenerator;

  void iniStatisitics(unsigned int pdfBinsCnt, unsigned int statBins) {
    statistics.resize(boost::extents[pdfAllRef.size()][pdfAllRef[0].size()][pdfBinsCnt][statBins]);
  }

  const StatArrayType& getStatistics() const { return statistics; }

  void setKeyPt(unsigned int pt) { theKey.thePt = pt; }

  void setKeyNumber(unsigned int number) { theKey.theNumber = number; }

private:
  StatArrayType statistics;
};
//////////////////////////////////
//////////////////////////////////
#endif
