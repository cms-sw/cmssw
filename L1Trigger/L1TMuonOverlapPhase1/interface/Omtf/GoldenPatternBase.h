#ifndef L1T_OmtfP1_GoldenPatternBase_H
#define L1T_OmtfP1_GoldenPatternBase_H

#include "boost/multi_array.hpp"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include <ostream>
#include <vector>

//////////////////////////////////
// Key
//////////////////////////////////
struct Key {
  Key(int iEta = 99, unsigned int iPt = 0, int iCharge = 0, unsigned int iNumber = 999)
      : theEtaCode(iEta), thePt(iPt), theCharge(iCharge), theNumber(iNumber) {}

  Key(int iEta, unsigned int iPt, int iCharge, unsigned int iNumber, unsigned int group, unsigned int indexInGroup)
      : theEtaCode(iEta),
        thePt(iPt),
        theCharge(iCharge),
        theNumber(iNumber),
        theGroup(group),
        theIndexInGroup(indexInGroup) {}

  inline bool operator<(const Key& o) const { return (theNumber < o.theNumber); }

  bool operator==(const Key& o) const {
    return theEtaCode == o.theEtaCode && thePt == o.thePt && theCharge == o.theCharge && theNumber == o.theNumber;
  }

  friend std::ostream& operator<<(std::ostream& out, const Key& o);

  unsigned int number() const { return theNumber; }

  int theEtaCode;

  //hardware pt, ptInGeV = (thePt-1) * 0.5GeV, where ptInGeV denotes the lover edge of the pt range cover by this pattern
  unsigned int thePt;

  int theCharge;
  unsigned int theNumber;

  //the index of the patterns group, up to 4 patterns can be grouped together, they have then the same MeanDistPhi and DistPhiBitShift
  unsigned int theGroup = 0;

  unsigned int theIndexInGroup = 1;  //starts from 1, as in xml

  void setPt(int pt) { thePt = pt; }

  void setGroup(int group) { theGroup = group; }

  void setIndexInGroup(unsigned int indexInGroup) { theIndexInGroup = indexInGroup; }

  unsigned int getHwPatternNumber() const { return theGroup * 4 + theIndexInGroup - 1; }
};
//////////////////////////////////
// Golden Pattern
//////////////////////////////////

class GoldenPatternBase {
public:
  typedef std::vector<int> vector1D;

  typedef boost::multi_array<GoldenPatternResult, 2> resultsArrayType;
  //
  // IGoldenPatterns methods
  //
  GoldenPatternBase(const Key& aKey);

  GoldenPatternBase(const Key& aKey, const OMTFConfiguration* omtfConfig);

  virtual ~GoldenPatternBase() {}

  virtual void setConfig(const OMTFConfiguration* omtfConfig);

  const OMTFConfiguration* getConfig() const { return myOmtfConfig; }

  virtual Key& key() { return theKey; }

  virtual int meanDistPhiValue(unsigned int iLayer, unsigned int iRefLayer, int refLayerPhiB = 0) const = 0;

  virtual PdfValueType pdfValue(unsigned int iLayer,
                                unsigned int iRefLayer,
                                unsigned int iBin,
                                int refLayerPhiB = 0) const = 0;

  virtual void setMeanDistPhiValue(int value,
                                   unsigned int iLayer,
                                   unsigned int iRefLayer,
                                   unsigned int paramIndex = 0) = 0;

  virtual void setPdfValue(
      PdfValueType value, unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, int refLayerPhiB = 0) = 0;

  virtual int getDistPhiBitShift(unsigned int iLayer, unsigned int iRefLayer) const = 0;

  virtual void setDistPhiBitShift(int value, unsigned int iLayer, unsigned int iRefLayer) = 0;

  ///Process single measurement layer with a single ref layer
  ///Method should be thread safe
  virtual StubResult process1Layer1RefLayer(unsigned int iRefLayer,
                                            unsigned int iLayer,
                                            MuonStubPtrs1D layerStubs,
                                            const std::vector<int>& extrapolatedPhi,
                                            const MuonStubPtr& refStub);

  ///Propagate phi from given reference layer to MB2 or ME2
  ///ME2 is used if eta of reference hit is larger than 1.1
  ///expressed in integer MicroGMT scale: 1.1/2.61*240 = 101
  virtual int propagateRefPhi(int phiRef, int etaRef, unsigned int iRefLayer) = 0;

  resultsArrayType& getResults() { return results; }

  ///last step of the event processing, before sorting and ghost busting
  virtual void finalise(unsigned int procIndx);

protected:
  ///Pattern kinematic identification (iEta,iPt,iCharge)
  Key theKey;

  const OMTFConfiguration* myOmtfConfig;

  resultsArrayType results;
};

template <class GoldenPatternType>
using GoldenPatternVec = std::vector<std::unique_ptr<GoldenPatternType> >;
//////////////////////////////////
//////////////////////////////////
#endif
