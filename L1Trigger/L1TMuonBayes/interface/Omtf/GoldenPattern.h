#ifndef OMTF_GoldenPattern_H
#define OMTF_GoldenPattern_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <vector>
#include <ostream>


class OMTFConfiguration;

//////////////////////////////////
// Golden Pattern
//////////////////////////////////

class GoldenPattern : public GoldenPatternBase {

public:
  typedef boost::multi_array<PdfValueType, 3> pdfArrayType;
  typedef boost::multi_array<short, 3> meanDistPhiArrayType;
  //
  // GoldenPatterns methods
  //
  GoldenPattern(const Key & aKey, unsigned int nLayers, unsigned int nRefLayers, unsigned int nPdfAddrBits): GoldenPatternBase(aKey),
    pdfAllRef(boost::extents[nLayers][nRefLayers][1<<nPdfAddrBits]),
    meanDistPhi(boost::extents[nLayers][nRefLayers][2]),
    distPhiBitShift(boost::extents[nLayers][nRefLayers])
  {
    reset();
  }

  GoldenPattern(const Key & aKey, const OMTFConfiguration* omtfConfig): GoldenPatternBase(aKey, omtfConfig),
      pdfAllRef(boost::extents[omtfConfig->nLayers()][omtfConfig->nRefLayers()][omtfConfig->nPdfBins()]),
      meanDistPhi(boost::extents[omtfConfig->nLayers()][omtfConfig->nRefLayers()][2]),
      distPhiBitShift(boost::extents[omtfConfig->nLayers()][omtfConfig->nRefLayers()]) {
    reset();
  }

  virtual ~GoldenPattern() {};

  virtual void setMeanDistPhi(const meanDistPhiArrayType& aMeanDistPhi) { meanDistPhi = aMeanDistPhi; }

  virtual const meanDistPhiArrayType& getMeanDistPhi() const { return meanDistPhi; }

  virtual const pdfArrayType& getPdf() const {return pdfAllRef;}

  virtual void setPdf(pdfArrayType& aPdf){  pdfAllRef = aPdf; }

  virtual int meanDistPhiValue(unsigned int iLayer, unsigned int iRefLayer, int refLayerPhiB = 0) const;

  virtual PdfValueType pdfValue(unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, int refLayerPhiB = 0) const {return pdfAllRef[iLayer][iRefLayer][iBin];}

  virtual void setMeanDistPhiValue(int value, unsigned int iLayer, unsigned int iRefLayer, unsigned int paramIndex = 0) {
    meanDistPhi[iLayer][iRefLayer][paramIndex] = value;
  }

  virtual void setPdfValue(PdfValueType value, unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, int refLayerPhiB = 0) {
    pdfAllRef[iLayer][iRefLayer][iBin] = value;
  }

/*  virtual const boost::multi_array<short, 2>& getDistPhiBitShift() const {
    return distPhiBitShift;
  }*/

  virtual int getDistPhiBitShift(unsigned int iLayer, unsigned int iRefLayer) const {
    return distPhiBitShift[iLayer][iRefLayer];
  }

  virtual void setDistPhiBitShift(int value, unsigned int iLayer, unsigned int iRefLayer)  {
    distPhiBitShift[iLayer][iRefLayer] = value;
  }

  friend std::ostream & operator << (std::ostream &out, const GoldenPattern & aPattern);

  ///Reset contents of all data vectors, keeping the vectors size
  virtual void reset();
/*  virtual void reset(unsigned int nLayers, unsigned int nRefLayers, unsigned int nPdfAddrBits);

  virtual void reset(const OMTFConfiguration* omtfConfig) {
    reset(omtfConfig->nLayers(), omtfConfig->nRefLayers(), omtfConfig->nPdfAddrBits());
  }*/

  ///Propagate phi from given reference layer to MB2 or ME2
  ///ME2 is used if eta of reference hit is larger than 1.1
  ///expressed in ingerer MicroGMT scale: 1.1/2.61*240 = 101
  virtual int propagateRefPhi(int phiRef, int etaRef, unsigned int iRefLayer);

protected:
  ///Distributions for all reference layers
  ///First index: measurement layer number
  ///Second index: refLayer number
  ///Third index: pdf bin number within layer 
  pdfArrayType pdfAllRef;

  ///Mean positions in each layer
  ///First index: measurement layer number 
  ///Second index: refLayer number
  ///Third index: index = 0 - a0, index = 1 - a1 for the linear fit meanDistPhi = a0 + a1 * phi_b
  meanDistPhiArrayType meanDistPhi;

  ///distPhi resolution can be reduced to reduce the number of bit on the LUT input
  ///distPhi = distPhi<<distPhiBitShift[layer][refLayer] (i.e. division by 2)
  ///First index: measurement layer number
  ///Second index: refLayer number
  boost::multi_array<short, 2> distPhiBitShift;
};

class GoldenPatternWithThresh : public GoldenPattern {
private:
  std::vector<PdfValueType> thresholds;
public:
  //
  // GoldenPatterns methods
  //
  GoldenPatternWithThresh(const Key & aKey, unsigned int nLayers, unsigned int nRefLayers, unsigned int nPdfAddrBits):
    GoldenPattern(aKey, nLayers, nRefLayers, nPdfAddrBits),
    thresholds(nRefLayers, 0) {

  }

  GoldenPatternWithThresh(const Key & aKey, const OMTFConfiguration* omtfConfig):
    GoldenPattern(aKey, omtfConfig),
    thresholds(myOmtfConfig->nRefLayers(), 0) {

  }

  virtual ~GoldenPatternWithThresh() {};

  PdfValueType getThreshold(unsigned int iRefLayer) const {
    return thresholds.at(iRefLayer);
  }

  void setThresholds(std::vector<PdfValueType>& tresholds) {
    this->thresholds = tresholds;
  }

  void setThreshold(unsigned int iRefLayer, PdfValueType treshold) {
    this->thresholds[iRefLayer] = treshold;
  }
};
//////////////////////////////////
//////////////////////////////////
#endif 
