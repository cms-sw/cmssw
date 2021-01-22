#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h

/** \class CSCGEMMotherboard
 *
 * Base class for TMBs for the GEM-CSC integrated local trigger. Inherits
 * from CSCUpgradeMotherboard. Provides common functionality to match
 * ALCT/CLCT to GEM pads or copads. Matching functions are templated so
 * they work both for GEMPadDigi and GEMCoPadDigi
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMCoPadProcessor.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

typedef match<GEMPadDigi> GEMPadDigiId;
typedef matches<GEMPadDigi> GEMPadDigiIds;
typedef matchesBX<GEMPadDigi> GEMPadDigiIdsBX;

typedef match<GEMCoPadDigi> GEMCoPadDigiId;
typedef matches<GEMCoPadDigi> GEMCoPadDigiIds;
typedef matchesBX<GEMCoPadDigi> GEMCoPadDigiIdsBX;

class CSCGEMMotherboard : public CSCUpgradeMotherboard {
public:
  enum Default_values { DEFAULT_MATCHING_VALUE = -99 };

  // standard constructor
  CSCGEMMotherboard(unsigned endcap,
                    unsigned station,
                    unsigned sector,
                    unsigned subsector,
                    unsigned chamber,
                    const edm::ParameterSet& conf);

  //Default constructor for testing
  CSCGEMMotherboard();

  ~CSCGEMMotherboard() override;

  // clear stored pads and copads
  void clear();

  using CSCUpgradeMotherboard::readoutLCTs;

  using CSCUpgradeMotherboard::run;

  // run TMB with GEM pad clusters as input
  virtual void run(const CSCWireDigiCollection* wiredc,
                   const CSCComparatorDigiCollection* compdc,
                   const GEMPadDigiClusterCollection* gemPads) = 0;

  /** additional processor for GEMs */
  std::unique_ptr<GEMCoPadProcessor> coPadProcessor;

  /// set CSC and GEM geometries for the matching needs
  void setGEMGeometry(const GEMGeometry* g) { gem_g = g; }

protected:
  virtual const CSCGEMMotherboardLUT* getLUT() const = 0;
  // check wether wiregroup corss strip or not. ME11 case would redefine this function
  virtual bool doesWiregroupCrossStrip(int key_wg, int key_strip) const { return true; }

  // check if a GEMDetId is valid
  bool isGEMDetId(unsigned int) const;

  // aux functions to get BX and position of a digi
  int getBX(const GEMPadDigi& p) const;
  int getBX(const GEMCoPadDigi& p) const;

  int getRoll(const GEMPadDigiId& p) const;
  int getRoll(const GEMCoPadDigiId& p) const;
  std::pair<int, int> getRolls(const CSCALCTDigi&) const;

  float getPad(const GEMPadDigi&) const;
  float getPad(const GEMCoPadDigi&) const;
  float getPad(const CSCCLCTDigi&, enum CSCPart par) const;

  // match ALCT to GEM Pad/CoPad
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  void matchingPads(const CSCALCTDigi& alct, matches<T>&) const;

  // match CLCT to GEM Pad/CoPad
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  void matchingPads(const CSCCLCTDigi& alct, matches<T>&) const;

  // find the matching pads to a pair of ALCT/CLCT
  // the first template is ALCT or CLCT
  // the second template is GEMPadDigi or GEMCoPadDigi
  template <class S, class T>
  void matchingPads(const S& d1, const S& d2, matches<T>&) const;

  // find common matches between an ALCT and CLCT
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  void matchingPads(const CSCCLCTDigi& clct1, const CSCALCTDigi& alct1, matches<T>&) const;

  // find all matching pads to a pair of ALCT and a pair of CLCT
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  void matchingPads(const CSCCLCTDigi& clct1,
                    const CSCCLCTDigi& clct2,
                    const CSCALCTDigi& alct1,
                    const CSCALCTDigi& alct2,
                    matches<T>&) const;

  // find the best matching pad to an ALCT
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  T bestMatchingPad(const CSCALCTDigi&, const matches<T>&) const;

  // find the best matching pad to an ALCT
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  T bestMatchingPad(const CSCCLCTDigi&, const matches<T>&) const;

  // find the best matching pad to an ALCT and CLCT
  // the template is GEMPadDigi or GEMCoPadDigi
  template <class T>
  T bestMatchingPad(const CSCALCTDigi&, const CSCCLCTDigi&, const matches<T>&) const;

  // correlate ALCTs/CLCTs with a set of matching GEM copads
  // use this function when the best matching copads are not clear yet
  // the template is ALCT or CLCT
  template <class T>
  void correlateLCTsGEM(const T& best,
                        const T& second,
                        const GEMCoPadDigiIds& coPads,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  // correlate ALCTs/CLCTs with their best matching GEM copads
  // the template is ALCT or CLCT
  template <class T>
  void correlateLCTsGEM(const T& best,
                        const T& second,
                        const GEMCoPadDigi&,
                        const GEMCoPadDigi&,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  // construct LCT from ALCT and GEM copad
  // fourth argument is the LCT number (1 or 2)
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const GEMCoPadDigi& gem, int i) const;

  // construct LCT from CLCT and GEM copad
  // fourth argument is the LCT number (1 or 2)
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCCLCTDigi& clct, const GEMCoPadDigi& gem, int i) const;

  // construct LCT from ALCT,CLCT and GEM copad
  // last argument is the LCT number (1 or 2)
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct,
                                        const CSCCLCTDigi& clct,
                                        const GEMCoPadDigi& gem,
                                        int i) const;

  // construct LCT from ALCT,CLCT and a single GEM pad
  // last argument is the LCT number (1 or 2)
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct,
                                        const CSCCLCTDigi& clct,
                                        const GEMPadDigi& gem,
                                        int i) const;
  /*
   * General function to construct integrated stubs from CSC and GEM information.
   * Options are:
   * 1. ALCT-CLCT-GEMPad
   * 2. ALCT-CLCT-GEMCoPad
   * 3. ALCT-GEMCoPad
   * 4. CLCT-GEMCoPad
   */
  // last argument is the LCT number (1 or 2)
  CSCCorrelatedLCTDigi constructLCTsGEM(
      const CSCALCTDigi& alct, const CSCCLCTDigi& clct, const GEMPadDigi& gem1, const GEMCoPadDigi& gem2, int i) const;

  // get the pads/copads from the digi collection and store in handy containers
  void processGEMClusters(const GEMPadDigiClusterCollection* pads);
  void processGEMPads(const GEMPadDigiCollection* pads);
  void processGEMCoPads();

  enum LCT_QualityRun3 {
    INVALID = 0,
    CLCT_2GEM = 3,
    ALCT_2GEM = 4,
    ALCTCLCT = 5,
    ALCTCLCT_1GEM = 6,
    ALCTCLCT_2GEM = 7,
  };

  // quality of the LCT when you take into account max 2 GEM layers
  CSCMotherboard::LCT_Quality findQualityGEMv1(const CSCALCTDigi&, const CSCCLCTDigi&, int gemlayer) const;
  LCT_QualityRun3 findQualityGEMv2(const CSCALCTDigi&, const CSCCLCTDigi&, int gemlayer) const;

  // print available trigger pads
  void printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart);
  void printGEMTriggerCoPads(int bx_start, int bx_stop, enum CSCPart);

  bool isPadInOverlap(int roll) const;

  /** Chamber id (trigger-type labels). */
  unsigned gemId;
  int maxPads() const;
  int maxRolls() const;

  const GEMGeometry* gem_g;
  bool gemGeometryAvailable;

  std::vector<GEMCoPadDigi> gemCoPadV;

  // map< bx , vector<gemid, pad> >
  GEMPadDigiIdsBX pads_;
  GEMCoPadDigiIdsBX coPads_;

  //  deltas used to match to GEM pads
  int maxDeltaBXPad_;
  int maxDeltaBXCoPad_;
  int maxDeltaPadL1_;
  int maxDeltaPadL2_;

  // promote ALCT-GEM pattern
  bool promoteALCTGEMpattern_;

  bool promoteALCTGEMquality_;
  bool promoteCLCTGEMquality_;

private:
  template <class T>
  const matchesBX<T>& getPads() const;

  template <class T>
  int getMaxDeltaBX() const;

  template <class T>
  int getLctTrigEnable() const;
};

template <class T>
void CSCGEMMotherboard::matchingPads(const CSCALCTDigi& alct, matches<T>& result) const {
  result.clear();
  // Invalid ALCTs have no matching pads
  if (not alct.isValid())
    return;

  // Get the corresponding roll numbers for a given ALCT
  std::pair<int, int> alctRoll = (getLUT()->get_csc_wg_to_gem_roll(theParity))[alct.getKeyWG()];

  // Get the pads in the ALCT bx
  const matchesBX<T>& lut = getPads<T>();

  // If no pads in that bx...
  if (lut.count(alct.getBX()) == 0)
    return;

  for (const auto& p : lut.at(alct.getBX())) {
    auto padRoll(getRoll(p));

    int delta;
    if (GEMDetId(p.first).station() == 2)
      delta = 1;

    // pad bx needs to match to ALCT bx
    int pad_bx = getBX(p.second) + CSCConstants::LCT_CENTRAL_BX;

    if (std::abs(alct.getBX() - pad_bx) > getMaxDeltaBX<T>())
      continue;

    // gem roll number if invalid
    if (alctRoll.first == CSCGEMMotherboard::DEFAULT_MATCHING_VALUE and
        alctRoll.second == CSCGEMMotherboard::DEFAULT_MATCHING_VALUE)
      continue;
    // ALCTs at the top of the chamber
    else if (alctRoll.first == CSCGEMMotherboard::DEFAULT_MATCHING_VALUE and padRoll > alctRoll.second)
      continue;
    // ALCTs at the bottom of the chamber
    else if (alctRoll.second == CSCGEMMotherboard::DEFAULT_MATCHING_VALUE and padRoll < alctRoll.first)
      continue;
    // ignore pads that are too far away in roll number
    else if ((alctRoll.first != CSCGEMMotherboard::DEFAULT_MATCHING_VALUE and
              alctRoll.second != CSCGEMMotherboard::DEFAULT_MATCHING_VALUE) and
             (alctRoll.first - delta > padRoll or padRoll > alctRoll.second))
      continue;
    result.push_back(p);
  }
}

template <class T>
void CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct, matches<T>& result) const {
  result.clear();
  // Invalid ALCTs have no matching pads
  if (not clct.isValid())
    return;

  auto part(getCSCPart(clct.getKeyStrip()));
  // Get the corresponding pad numbers for a given CLCT
  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(theParity, part));
  int keyStrip = clct.getKeyStrip();
  //ME1A part, convert halfstrip from 128-223 to 0-95
  if (part == CSCPart::ME1A and keyStrip > CSCConstants::MAX_HALF_STRIP_ME1B)
    keyStrip = keyStrip - CSCConstants::MAX_HALF_STRIP_ME1B - 1;
  const int lowPad(mymap[keyStrip].first);
  const int highPad(mymap[keyStrip].second);

  // Get the pads in the CLCT bx
  const matchesBX<T>& lut = getPads<T>();

  // If no pads in that bx...
  if (lut.count(clct.getBX()) == 0)
    return;

  for (const auto& p : lut.at(clct.getBX())) {
    // pad bx needs to match to CLCT bx
    int pad_bx = getBX(p.second) + CSCConstants::LCT_CENTRAL_BX;
    if (std::abs(clct.getBX() - pad_bx) > getMaxDeltaBX<T>())
      continue;

    // pad number must match
    int padNumber(getPad(p.second));
    if (std::abs(lowPad - padNumber) <= maxDeltaPadL1_ or std::abs(padNumber - highPad) <= maxDeltaPadL1_) {
      result.push_back(p);
    }
  }
}

template <class S, class T>
void CSCGEMMotherboard::matchingPads(const S& d1, const S& d2, matches<T>& result) const {
  matches<T> p1, p2;

  // pads matching to the CLCT/ALCT
  matchingPads<T>(d1, p1);

  // pads matching to the CLCT/ALCT
  matchingPads<T>(d2, p2);

  // collect *all* matching pads
  result.reserve(p1.size() + p2.size());
  result.insert(std::end(result), std::begin(p1), std::end(p1));
  result.insert(std::end(result), std::begin(p2), std::end(p2));
}

template <class T>
void CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct1, const CSCALCTDigi& alct1, matches<T>& result) const {
  matches<T> padsClct, padsAlct;

  // pads matching to the CLCT
  matchingPads<T>(clct1, padsClct);

  // pads matching to the ALCT
  matchingPads<T>(alct1, padsAlct);

  // collect all *common* pads
  intersection(padsClct, padsAlct, result);
}

template <class T>
void CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct1,
                                     const CSCCLCTDigi& clct2,
                                     const CSCALCTDigi& alct1,
                                     const CSCALCTDigi& alct2,
                                     matches<T>& result) const {
  matches<T> padsClct, padsAlct;

  // pads matching to CLCTs
  matchingPads<CSCCLCTDigi, T>(clct1, clct2, padsClct);

  // pads matching to ALCTs
  matchingPads<CSCALCTDigi, T>(alct1, alct2, padsAlct);

  // collect *all* matching pads
  result.reserve(padsClct.size() + padsAlct.size());
  result.insert(std::end(result), std::begin(padsClct), std::end(padsClct));
  result.insert(std::end(result), std::begin(padsAlct), std::end(padsAlct));
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCALCTDigi& alct1, const matches<S>& pads) const {
  S result;
  // no matching pads for invalid stub
  if (not alct1.isValid()) {
    return result;
  }
  // return the first one with the same roll number
  for (const auto& p : pads) {
    // protection against corrupted DetIds
    if (not isGEMDetId(p.first))
      continue;

    // roll number of pad and ALCT must match
    if (getRolls(alct1).first <= getRoll(p) and getRoll(p) <= getRolls(alct1).second) {
      return p.second;
    }
  }
  return result;
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCCLCTDigi& clct, const matches<S>& pads) const {
  S result;
  // no matching pads for invalid stub
  if (not clct.isValid())
    return result;

  auto part(getCSCPart(clct.getKeyStrip()));

  // return the pad with the smallest bending angle
  float averagePadNumberCSC = getPad(clct, part);
  float minDeltaPad = 999;
  for (const auto& p : pads) {
    // protection against corrupted DetIds
    if (not isGEMDetId(p.first))
      continue;

    // best pad is closest to CLCT in number of halfstrips
    float averagePadNumberGEM = getPad(p.second);
    if (std::abs(averagePadNumberCSC - averagePadNumberGEM) < minDeltaPad) {
      minDeltaPad = std::abs(averagePadNumberCSC - averagePadNumberGEM);
      result = p.second;
    }
  }
  return result;
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCALCTDigi& alct1, const CSCCLCTDigi& clct1, const matches<S>& pads) const {
  S result;
  // no matching pads for invalid stub
  if (not alct1.isValid() or not clct1.isValid()) {
    return result;
  }

  auto part(getCSCPart(clct1.getKeyStrip()));

  // return the pad with the smallest bending angle
  float averagePadNumberCSC = getPad(clct1, part);
  float minDeltaPad = 999;
  for (const auto& p : pads) {
    // protection against corrupted DetIds
    if (not isGEMDetId(p.first))
      continue;

    float averagePadNumberGEM = getPad(p.second);

    int delta;
    if (GEMDetId(p.first).station() == 2)
      delta = 1;

    // add another safety to make sure that the deltaPad is not larger than max value!!!
    if (std::abs(averagePadNumberCSC - averagePadNumberGEM) < minDeltaPad and
        getRolls(alct1).first - delta <= getRoll(p) and getRoll(p) <= getRolls(alct1).second) {
      minDeltaPad = std::abs(averagePadNumberCSC - averagePadNumberGEM);
      result = p.second;
    }
  }
  return result;
}

template <class T>
void CSCGEMMotherboard::correlateLCTsGEM(const T& bLCT,
                                         const T& sLCT,
                                         const GEMCoPadDigiIds& coPads,
                                         CSCCorrelatedLCTDigi& lct1,
                                         CSCCorrelatedLCTDigi& lct2) const {
  T bestLCT = bLCT;
  T secondLCT = sLCT;

  // Check which LCTs are valid
  bool bestValid = bestLCT.isValid();
  bool secondValid = secondLCT.isValid();

  // At this point, set both LCTs valid if they are invalid
  // Duplicate LCTs are taken into account later
  if (bestValid and !secondValid)
    secondLCT = bestLCT;
  if (!bestValid and secondValid)
    bestLCT = secondLCT;

  // get best matching copad1
  const GEMCoPadDigi& bestCoPad = bestMatchingPad<GEMCoPadDigi>(bestLCT, coPads);
  const GEMCoPadDigi& secondCoPad = bestMatchingPad<GEMCoPadDigi>(secondLCT, coPads);

  correlateLCTsGEM(bestLCT, secondLCT, bestCoPad, secondCoPad, lct1, lct2);
}

template <class T>
void CSCGEMMotherboard::correlateLCTsGEM(const T& bestLCT,
                                         const T& secondLCT,
                                         const GEMCoPadDigi& bestCoPad,
                                         const GEMCoPadDigi& secondCoPad,
                                         CSCCorrelatedLCTDigi& lct1,
                                         CSCCorrelatedLCTDigi& lct2) const {
  // construct the first LCT from ALCT(CLCT) and a GEM Copad
  if ((getLctTrigEnable<T>() and bestLCT.isValid()) or (match_trig_enable and bestLCT.isValid())) {
    lct1 = constructLCTsGEM(bestLCT, bestCoPad, 1);
  }

  // construct the second LCT from ALCT(CLCT) and a GEM Copad
  if ((getLctTrigEnable<T>() and secondLCT.isValid()) or
      (match_trig_enable and secondLCT.isValid() and secondLCT != bestLCT)) {
    lct2 = constructLCTsGEM(secondLCT, secondCoPad, 2);
  }
}

#endif
