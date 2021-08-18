#ifndef L1T_OmtfP1_OMTFConfiguration_H
#define L1T_OmtfP1_OMTFConfiguration_H

#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <memory>

//#undef BOOST_DISABLE_ASSERTS  //TODO remove for production version
#include "boost/multi_array.hpp"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

//typedef int omtfPdfValueType; //normal emulation is with int type
typedef float PdfValueType;  //but floats are needed for the PatternOptimizer

namespace edm {
  class ParameterSet;
}

class RefHitDef {
public:
  //FIXME: default values should be sonnected to configuration values
  RefHitDef(unsigned int aInput = 15,
            int aPhiMin = 5760,
            int aPhiMax = 5760,
            unsigned int aRegion = 99,
            unsigned int aRefLayer = 99);

public:
  bool fitsRange(int iPhi) const;

  ///Hit input number within a cone
  unsigned int iInput;

  ///Region number assigned to this referecne hit
  unsigned int iRegion;

  ///Reference layer logic number (0-7)
  unsigned int iRefLayer;

  ///Local to processor phi range.
  ///Hit has to fit into this range to be assigned to this iRegion;
  std::pair<int, int> range;

  friend std::ostream& operator<<(std::ostream& out, const RefHitDef& aRefHitDef);
};

class OMTFConfiguration : public ProcConfigurationBase {
public:
  typedef std::vector<std::pair<unsigned int, unsigned int> > vector1D_pair;
  typedef std::vector<vector1D_pair> vector2D_pair;
  typedef std::vector<vector2D_pair> vector3D_pair;

  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D> vector2D;
  typedef std::vector<vector2D> vector3D;
  typedef std::vector<vector3D> vector4D;

  OMTFConfiguration() { ; };

  virtual void configure(const L1TMuonOverlapParams* omtfParams);

  void initCounterMatrices();

  ///Find logic region number using first input number
  ///and then local phi value. The input and phi
  ///ranges are taken from DB.
  unsigned int getRegionNumberFromMap(unsigned int iInput, unsigned int iRefLayer, int iPhi) const;

  ///Check if given referecne hit is
  ///in phi range for some logic cone.
  ///Care is needed arounf +Pi and +2Pi points
  bool isInRegionRange(int iPhiStart, unsigned int coneSize, int iPhi) const;

  ///Return global phi for beggining of given processor
  ///Uses minim phi over all reference layers.
  int globalPhiStart(unsigned int iProcessor) const;

  ///Return layer number encoding subsystem, and
  ///station number in a simple formula:
  /// aLayer+100*detId.subdetId()
  ///where aLayer is a layer number counting from vertex
  uint32_t getLayerNumber(uint32_t rawId) const;

  unsigned int fwVersion() const { return (rawParams.fwVersion() >> 16) & 0xFFFF; };
  unsigned int patternsVersion() const { return rawParams.fwVersion() & 0xFFFF; };

  const L1TMuonOverlapParams* getRawParams() const { return &rawParams; };

  float minPdfVal() const { return 0.001; };
  unsigned int nLayers() const override { return rawParams.nLayers(); };
  unsigned int nHitsPerLayer() const { return rawParams.nHitsPerLayer(); };
  unsigned int nRefLayers() const { return rawParams.nRefLayers(); };
  unsigned int nPhiBits() const { return rawParams.nPhiBits(); };
  unsigned int nPdfAddrBits() const { return rawParams.nPdfAddrBits(); };
  unsigned int nPdfBins() const { return pdfBins; };
  unsigned int nPdfValBits() const { return rawParams.nPdfValBits(); };
  int pdfMaxValue() const { return pdfMaxVal; };
  unsigned int nPhiBins() const override { return rawParams.nPhiBins(); };
  unsigned int nRefHits() const { return rawParams.nRefHits(); };
  unsigned int nTestRefHits() const { return rawParams.nTestRefHits(); };
  //processors number per detector side
  unsigned int nProcessors() const { return rawParams.nProcessors(); };
  //total number of processors in the system
  unsigned int processorCnt() const { return 2 * rawParams.nProcessors(); };
  unsigned int nLogicRegions() const { return rawParams.nLogicRegions(); };
  unsigned int nInputs() const { return rawParams.nInputs(); };
  unsigned int nGoldenPatterns() const { return rawParams.nGoldenPatterns(); };

  const std::map<int, int>& getHwToLogicLayer() const { return hwToLogicLayer; }
  const std::map<int, int>& getLogicToHwLayer() const { return logicToHwLayer; }
  const std::map<int, int>& getLogicToLogic() const { return logicToLogic; }
  const std::set<int>& getBendingLayers() const { return bendingLayers; }
  const std::vector<int>& getRefToLogicNumber() const { return refToLogicNumber; }

  const std::vector<unsigned int>& getBarrelMin() const { return barrelMin; }
  const std::vector<unsigned int>& getBarrelMax() const { return barrelMax; }
  const std::vector<unsigned int>& getEndcap10DegMin() const { return endcap10DegMin; }
  const std::vector<unsigned int>& getEndcap10DegMax() const { return endcap10DegMax; }
  const std::vector<unsigned int>& getEndcap20DegMin() const { return endcap20DegMin; }
  const std::vector<unsigned int>& getEndcap20DegMax() const { return endcap20DegMax; }

  const std::vector<std::vector<int> >& getProcessorPhiVsRefLayer() const { return processorPhiVsRefLayer; }
  const std::vector<std::vector<std::vector<std::pair<int, int> > > >& getRegionPhisVsRefLayerVsInput() const {
    return regionPhisVsRefLayerVsInput;
  }
  const std::vector<std::vector<RefHitDef> >& getRefHitsDefs() const { return refHitsDefs; }

  const vector3D_pair& getConnections() const { return connections; };

  vector4D& getMeasurements4D() { return measurements4D; }
  vector4D& getMeasurements4Dref() { return measurements4Dref; }

  const vector4D& getMeasurements4D() const { return measurements4D; }
  const vector4D& getMeasurements4Dref() const { return measurements4Dref; }

  double ptUnit = 0.5;  // GeV/unit
  ///uGMT pt scale conversion
  double hwPtToGev(int hwPt) const override { return (hwPt - 1.) * ptUnit; }

  ///uGMT pt scale conversion: [0GeV, 0.5GeV) = 1 [0.5GeV, 1 Gev) = 2
  int ptGevToHw(double ptGev) const override { return (ptGev / ptUnit + 1); }

  double etaUnit = 0.010875;  //TODO from the interface note, should be defined somewhere globally
  ///center of eta bin
  virtual double hwEtaToEta(int hwEta) const { return (hwEta * etaUnit); }

  int etaToHwEta(double eta) const override { return (eta / etaUnit); }

  double phiGmtUnit = 2. * M_PI / 576;  //TODO from the interface note, should be defined somewhere globally
  //phi in radians
  virtual int phiToGlobalHwPhi(double phi) const { return std::floor(phi / phiGmtUnit); }

  //phi in radians
  virtual double hwPhiToGlobalPhi(int phi) const { return phi * phiGmtUnit; }

  ///iProcessor - 0...5
  ///phiRad [-pi,pi]
  ///return phi inside the processor
  int getProcScalePhi(unsigned int iProcessor, double phiRad) const;

  int getProcScalePhi(double phiRad, double procPhiZeroRad = 0) const override {
    return 0;  // TODO replace getProcScalePhi(unsigned int iProcessor, double phiRad) with this function
  }

  double procHwPhiToGlobalPhi(int procHwPhi, int procHwPhi0) const;

  int procPhiToGmtPhi(int procPhi) const {
    ///conversion factor from OMTF to uGMT scale is  5400/576 i.e. phiValue/=9.375;
    return floor(procPhi * 437. / (1 << 12));  // ie. use as in hw: 9.3729977
    //cannot be (procPhi * 437) >> 12, because this floor is needed
  }

  ///input phi should be in the hardware scale (nPhiBins units for 2pi), can be in range  -nPhiBins ... nPhiBins,
  //is converted to range -nPhiBins/2 +1 ... nPhiBins/2, pi = nPhiBins/2
  //virtual int foldPhi(int phi) const;

  ///Continuous processor number [0...12], to be used in as array index,
  unsigned int getProcIndx(unsigned int iProcessor, l1t::tftype mtfType) const {
    return ((mtfType - l1t::tftype::omtf_neg) * rawParams.nProcessors() + iProcessor);
  };

  //TODO implement more efficient
  bool isBendingLayer(unsigned int iLayer) const override { return getBendingLayers().count(iLayer); }

  ///pattern pt range in Gev
  struct PatternPt {
    double ptFrom = 0;
    double ptTo = 0;
    int charge = 0;
  };

  PatternPt getPatternPtRange(unsigned int patNum) const;

  //call it when the patterns are read directly from the xml, without using the LUTs
  void setPatternPtRange(const std::vector<PatternPt>& patternPts) { this->patternPts = patternPts; }

  ///charge: -1 - negative, +1 - positive
  unsigned int getPatternNum(double pt, int charge) const;

  static const unsigned int patternsInGroup = 4;

  //takes the groups from the key, it should be set during xml reading, or creating the goldenPats
  template <class GoldenPatternType>
  vector2D getPatternGroups(const std::vector<std::unique_ptr<GoldenPatternType> >& goldenPats) const {
    //unsigned int mergedCnt = 4;
    vector2D mergedPatterns;
    for (unsigned int iPat = 0; iPat < goldenPats.size(); iPat++) {
      unsigned int group = goldenPats.at(iPat)->key().theGroup;

      if (mergedPatterns.size() == group) {
        mergedPatterns.push_back(vector1D());
      }

      if (group < mergedPatterns.size()) {
        //theIndexInGroup starts from 1, as in xml
        if (mergedPatterns[group].size() == (goldenPats.at(iPat)->key().theIndexInGroup - 1))
          mergedPatterns[group].push_back(iPat);
        else
          return mergedPatterns;  //TODO should throw error
      } else
        return mergedPatterns;  //TODO should throw error
    }
    return mergedPatterns;
  }

  /**configuration from the edm::ParameterSet
   * the parameters are set (i.e. overwritten) only if their exist in the edmParameterSet
   */
  void configureFromEdmParameterSet(const edm::ParameterSet& edmParameterSet) override;

  int getGoldenPatternResultFinalizeFunction() const { return goldenPatternResultFinalizeFunction; }

  void setGoldenPatternResultFinalizeFunction(int goldenPatternResultFinalizeFunction = 0) {
    this->goldenPatternResultFinalizeFunction = goldenPatternResultFinalizeFunction;
  }

  friend std::ostream& operator<<(std::ostream& out, const OMTFConfiguration& aConfig);

  bool isNoHitValueInPdf() const { return noHitValueInPdf; }

  void setNoHitValueInPdf(bool noHitValueInPdf = false) { this->noHitValueInPdf = noHitValueInPdf; }

  int getSorterType() const { return sorterType; }

  void setSorterType(int sorterType = 0) { this->sorterType = sorterType; }

  const std::string& getGhostBusterType() const { return ghostBusterType; }

  void setGhostBusterType(const std::string& ghostBusterType = "") { this->ghostBusterType = ghostBusterType; }

private:
  L1TMuonOverlapParams rawParams;

  std::map<int, int> hwToLogicLayer;
  std::map<int, int> logicToHwLayer;
  std::map<int, int> logicToLogic;
  std::set<int> bendingLayers;
  std::vector<int> refToLogicNumber;

  ///Starting and final sectors connected to
  ///processors.
  ///Index: processor number
  std::vector<unsigned int> barrelMin;
  std::vector<unsigned int> barrelMax;
  std::vector<unsigned int> endcap10DegMin;
  std::vector<unsigned int> endcap10DegMax;
  std::vector<unsigned int> endcap20DegMin;
  std::vector<unsigned int> endcap20DegMax;

  ///Starting iPhi for each processor and each referecne layer
  ///Global phi scale is used
  ///First index: processor number
  ///Second index: referecne layer number
  std::vector<std::vector<int> > processorPhiVsRefLayer;

  ///Begin and end local phi for each logis region
  ///First index: input number
  ///Second index: reference layer number
  ///Third index: region
  ///pair.first: starting phi of region (inclusive)
  ///pair.second: ending phi of region (inclusive)
  std::vector<std::vector<std::vector<std::pair<int, int> > > > regionPhisVsRefLayerVsInput;

  ///Vector with definitions of reference hits
  ///Vector has fixed size of nRefHits
  ///Order of elements defines priority order
  ///First index: processor number (0-5)
  ///Second index: ref hit number (0-127)
  std::vector<std::vector<RefHitDef> > refHitsDefs;

  ///Map of connections
  vector3D_pair connections;

  ///4D matrices used during creation of the connections tables.
  vector4D measurements4D;
  vector4D measurements4Dref;

  std::vector<PatternPt> patternPts;

  int pdfMaxVal = 0;
  unsigned int pdfBins = 0;

  int goldenPatternResultFinalizeFunction = 0;

  bool noHitValueInPdf = false;

  int sorterType = 0;

  std::string ghostBusterType = "";
};

#endif
