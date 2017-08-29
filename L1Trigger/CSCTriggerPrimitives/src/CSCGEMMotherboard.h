#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h

#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/src/GEMCoPadProcessor.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

typedef match<GEMPadDigi>   GEMPadDigiId;
typedef matches<GEMPadDigi> GEMPadDigiIds;
typedef matchesBX<GEMPadDigi> GEMPadDigiIdsBX;

typedef match<GEMCoPadDigi>   GEMCoPadDigiId;
typedef matches<GEMCoPadDigi> GEMCoPadDigiIds;
typedef matchesBX<GEMCoPadDigi> GEMCoPadDigiIdsBX;

enum lctTypes{Invalid, ALCTCLCT, ALCTCLCTGEM, ALCTCLCT2GEM, ALCT2GEM, CLCT2GEM};

class CSCGEMMotherboard : public CSCUpgradeMotherboard
{
public:

  CSCGEMMotherboard(unsigned endcap, unsigned station, unsigned sector,
                    unsigned subsector, unsigned chamber,
                    const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCGEMMotherboard();

  ~CSCGEMMotherboard() override;

  void clear();

  // TMB run modes
  virtual void run(const CSCWireDigiCollection* wiredc,
		   const CSCComparatorDigiCollection* compdc,
		   const GEMPadDigiCollection* gemPads)=0;

  void run(const CSCWireDigiCollection* wiredc,
	   const CSCComparatorDigiCollection* compdc,
	   const GEMPadDigiClusterCollection* gemPads);

  /** additional processor for GEMs */
  std::unique_ptr<GEMCoPadProcessor> coPadProcessor;

  /// set CSC and GEM geometries for the matching needs
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }

protected:

  virtual const CSCGEMMotherboardLUT* getLUT() const=0;

  int getBX(const GEMPadDigi& p);
  int getBX(const GEMCoPadDigi& p);

  int getRoll(const GEMPadDigiId& p);
  int getRoll(const GEMCoPadDigiId& p);
  int getRoll(const CSCALCTDigi&);

  float getAvePad(const GEMPadDigi&);
  float getAvePad(const GEMCoPadDigi&);
  float getAvePad(const CSCCLCTDigi&, enum CSCPart part);

  // match ALCT/CLCT to Pad/CoPad
  template <class T>
  matches<T> matchingPads(const CSCALCTDigi& alct, enum CSCPart part);

  template <class T>
  matches<T> matchingPads(const CSCCLCTDigi& clct, enum CSCPart part);

  template <class T>
  T bestMatchingPad(const CSCALCTDigi&, const matches<T>&, enum CSCPart);

  template <class T>
  T bestMatchingPad(const CSCCLCTDigi&, const matches<T>&, enum CSCPart);

  template <class T>
  T bestMatchingPad(const CSCALCTDigi&, const CSCCLCTDigi&, const matches<T>&, enum CSCPart);

  template <class S, class T>
  matches<T> matchingPads(const S& d1, const S& d2, enum CSCPart part);

  template <class T>
  matches<T> matchingPads(const CSCCLCTDigi& clct1, const CSCALCTDigi& alct1, enum CSCPart part);

  template <class T>
  matches<T> matchingPads(const CSCCLCTDigi& clct1, const CSCCLCTDigi& clct2,
			  const CSCALCTDigi& alct1, const CSCALCTDigi& alct2,
			  enum CSCPart part);

  template <class T>
  void correlateLCTsGEM(T& best, T& second, const GEMCoPadDigiIds& coPads,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart);
  template <class T>
  void correlateLCTsGEM(const T& best, const T& second, const GEMCoPadDigi&, const GEMCoPadDigi&,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart);

  // specific functions
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const GEMCoPadDigi& gem, enum CSCPart, int i);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCCLCTDigi& clct, const GEMCoPadDigi& gem, enum CSCPart, int i);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct,
					const GEMCoPadDigi& gem, enum CSCPart p, int i);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct,
					const GEMPadDigi& gem, enum CSCPart p, int i);
  /*
   * General function to construct integrated stubs from CSC and GEM information.
   * Options are:
   * 1. ALCT-CLCT-GEMPad
   * 2. ALCT-CLCT-GEMCoPad
   * 3. ALCT-GEMCoPad
   * 4. CLCT-GEMCoPad
   */
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct,
					const GEMPadDigi& gem1,  const GEMCoPadDigi& gem2,
					enum CSCPart p, int i);

  void retrieveGEMPads(const GEMPadDigiCollection* pads, unsigned id);
  void retrieveGEMCoPads();

  unsigned int findQualityGEM(const CSCALCTDigi&, const CSCCLCTDigi&, int gemlayer);

  void printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart);
  void printGEMTriggerCoPads(int bx_start, int bx_stop, enum CSCPart);

  bool isPadInOverlap(int roll);

  void setupGeometry();

  /** Chamber id (trigger-type labels). */
  unsigned gemId;

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

  // send LCT old dataformat
  bool useOldLCTDataFormat_;

  // promote ALCT-GEM pattern
  bool promoteALCTGEMpattern_;

  bool promoteALCTGEMquality_;
  bool promoteCLCTGEMquality_;

  // LCT ghostbusting
  bool doLCTGhostBustingWithGEMs_;
};


template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCALCTDigi& alct1, const matches<S>& pads, enum CSCPart)
{
  S result;
  if (not alct1.isValid()) return result;

  // return the first one with the same roll number
  for (const auto& p: pads){
    if (getRoll(p) == getRoll(alct1)){
      return p.second;
    }
  }
  return result;
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCCLCTDigi& clct, const matches<S>& pads, enum CSCPart part)
{
  S result;
  if (not clct.isValid()) return result;

  // return the pad with the smallest bending angle
  float averagePadNumberCSC = getAvePad(clct, part);
  float minDeltaPad = 999;
  for (const auto& p: pads){
    float averagePadNumberGEM = getAvePad(p.second);
    if (std::abs(averagePadNumberCSC - averagePadNumberGEM) < minDeltaPad){
      minDeltaPad = std::abs(averagePadNumberCSC - averagePadNumberGEM);
      result = p.second;
    }
  }
  return result;
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCALCTDigi& alct1, const CSCCLCTDigi& clct1,
				     const matches<S>& pads, enum CSCPart part)
{
  S result;
  if (not alct1.isValid() or not clct1.isValid()) return result;

  // return the pad with the smallest bending angle
  float averagePadNumberCSC = getAvePad(clct1, part);
  float minDeltaPad = 999;
  for (const auto& p: pads){
    float averagePadNumberGEM = getAvePad(p.second);
    // add another safety to make sure that the deltaPad is not larger than max value!!!
    if (std::abs(averagePadNumberCSC - averagePadNumberGEM) < minDeltaPad and getRoll(p) == getRoll(alct1)){
      minDeltaPad = std::abs(averagePadNumberCSC - averagePadNumberGEM);
      result = p.second;
    }
  }
  return result;
}

template <class T>
void CSCGEMMotherboard::correlateLCTsGEM(T& bestLCT, T& secondLCT, const GEMCoPadDigiIds& coPads,
					 CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart p)
{
  bool bestValid     = bestLCT.isValid();
  bool secondValid   = secondLCT.isValid();

  // determine best/second
  if (bestValid and !secondValid) secondLCT = bestLCT;
  if (!bestValid and secondValid) bestLCT   = secondLCT;

  // get best matching copad1
  GEMCoPadDigi bestCoPad = bestMatchingPad<GEMCoPadDigi>(bestLCT, coPads, p);
  GEMCoPadDigi secondCoPad = bestMatchingPad<GEMCoPadDigi>(secondLCT, coPads, p);

  correlateLCTsGEM(bestLCT, secondLCT, bestCoPad, secondCoPad, lct1, lct2, p);
}


template <>
void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& bestLCT, const CSCALCTDigi& secondLCT,
					 const GEMCoPadDigi& bestCoPad, const GEMCoPadDigi& secondCoPad,
					 CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart p)
{
  if ((alct_trig_enable  and bestLCT.isValid()) or
      (match_trig_enable and bestLCT.isValid()))
    {
    lct1 = constructLCTsGEM(bestLCT, bestCoPad, p, 1);
  }

  if ((alct_trig_enable  and secondLCT.isValid()) or
      (match_trig_enable and secondLCT.isValid() and secondLCT != bestLCT))
    {
      lct2 = constructLCTsGEM(secondLCT, secondCoPad, p, 2);
    }
}


template <>
void CSCGEMMotherboard::correlateLCTsGEM(const CSCCLCTDigi& bestLCT, const CSCCLCTDigi& secondLCT,
					 const GEMCoPadDigi& bestCoPad, const GEMCoPadDigi& secondCoPad,
					 CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart p)
{
  if ((clct_trig_enable  and bestLCT.isValid()) or
      (match_trig_enable and bestLCT.isValid()))
    {
    lct1 = constructLCTsGEM(bestLCT, bestCoPad, p, 1);
  }

  if ((clct_trig_enable  and secondLCT.isValid()) or
      (match_trig_enable and secondLCT.isValid() and secondLCT != bestLCT))
    {
      lct2 = constructLCTsGEM(secondLCT, secondCoPad, p, 2);
    }
}


template <>
matches<GEMPadDigi> CSCGEMMotherboard::matchingPads(const CSCALCTDigi& alct, enum CSCPart part)
{
  matches<GEMPadDigi> result;
  if (not alct.isValid()) return result;

  std::pair<int,int> alctRoll = (getLUT()->CSCGEMMotherboardLUT::get_csc_wg_to_gem_roll(par))[alct.getKeyWG()];
  for (const auto& p: pads_[alct.getBX()]){
    auto padRoll(getRoll(p));
    // only pads in overlap are good for ME1A
    if (part==CSCPart::ME1A and !isPadInOverlap(padRoll)) continue;

    int pad_bx = getBX(p.second)+lct_central_bx;
    if (std::abs(alct.getBX()-pad_bx)>maxDeltaBXPad_) continue;

    if (alctRoll.first == -99 and alctRoll.second == -99) continue;  //invalid region
    else if (alctRoll.first == -99 and !(padRoll <= alctRoll.second)) continue; // top of the chamber
    else if (alctRoll.second == -99 and !(padRoll >= alctRoll.first)) continue; // bottom of the chamber
    else if ((alctRoll.first != -99 and alctRoll.second != -99) and // center
	     (alctRoll.first > padRoll or padRoll > alctRoll.second)) continue;
    result.push_back(p);
  }
  return result;
}

template <>
matches<GEMCoPadDigi> CSCGEMMotherboard::matchingPads(const CSCALCTDigi& alct, enum CSCPart part)
{
  matches<GEMCoPadDigi> result;
  if (not alct.isValid()) return result;

  std::pair<int,int> alctRoll = (getLUT()->CSCGEMMotherboardLUT::get_csc_wg_to_gem_roll(par))[alct.getKeyWG()];
  for (const auto& p: coPads_[alct.getBX()]){
    auto padRoll(getRoll(p));
    // only pads in overlap are good for ME1A
    if (part==CSCPart::ME1A and !isPadInOverlap(padRoll)) continue;

    int pad_bx = getBX(p.second)+lct_central_bx;
    if (std::abs(alct.getBX()-pad_bx)>maxDeltaBXCoPad_) continue;

    if (alctRoll.first == -99 and alctRoll.second == -99) continue;  //invalid region
    else if (alctRoll.first == -99 and !(padRoll <= alctRoll.second)) continue; // top of the chamber
    else if (alctRoll.second == -99 and !(padRoll >= alctRoll.first)) continue; // bottom of the chamber
    else if ((alctRoll.first != -99 and alctRoll.second != -99) and // center
	     (alctRoll.first > padRoll or padRoll > alctRoll.second)) continue;
    result.push_back(p);
  }
  return result;
}


template <>
matches<GEMPadDigi> CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct, enum CSCPart part)
{
  matches<GEMPadDigi> result;
  if (not clct.isValid()) return result;

  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(par, part));
  const int lowPad(mymap[clct.getKeyStrip()].first);
  const int highPad(mymap[clct.getKeyStrip()].second);
  for (const auto& p: pads_[clct.getBX()]){
    auto padRoll(getAvePad(p.second));
    int pad_bx = getBX(p.second)+lct_central_bx;
    if (std::abs(clct.getBX()-pad_bx)>maxDeltaBXPad_) continue;
    if (std::abs(lowPad - padRoll) <= maxDeltaPadL1_ or std::abs(padRoll - highPad) <= maxDeltaPadL1_){
      result.push_back(p);
    }
  }
  return result;
}


template <>
matches<GEMCoPadDigi> CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct, enum CSCPart part)
{
  matches<GEMCoPadDigi> result;
  if (not clct.isValid()) return result;

  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(par, part));
  const int lowPad(mymap[clct.getKeyStrip()].first);
  const int highPad(mymap[clct.getKeyStrip()].second);
  for (const auto& p: coPads_[clct.getBX()]){
    auto padRoll(getAvePad(p.second));
    int pad_bx = getBX(p.second)+lct_central_bx;
    if (std::abs(clct.getBX()-pad_bx)>maxDeltaBXCoPad_) continue;
    if (std::abs(lowPad - padRoll) <= maxDeltaPadL1_ or std::abs(padRoll - highPad) <= maxDeltaPadL1_){
      result.push_back(p);
    }
  }
  return result;
}


template <class S, class T>
matches<T> CSCGEMMotherboard::matchingPads(const S& d1, const S& d2, enum CSCPart part)
{
  matches<T> result, p1, p2;
  p1 = matchingPads<T>(d1, part);
  p2 = matchingPads<T>(d2, part);
  result.reserve(p1.size() + p2.size());
  result.insert(std::end(result), std::begin(p1), std::end(p1));
  result.insert(std::end(result), std::begin(p2), std::end(p2));
  return result;
}

template <class T>
matches<T> CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct1, const CSCALCTDigi& alct1, enum CSCPart part)
{
  matches<T> padsClct(matchingPads<T>(clct1, part));
  matches<T> padsAlct(matchingPads<T>(alct1, part));
  return intersection(padsClct, padsAlct);
}

template <class T>
matches<T> CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct1, const CSCCLCTDigi& clct2,
					   const CSCALCTDigi& alct1, const CSCALCTDigi& alct2,
					   enum CSCPart part)
{
  matches<T> padsClct(matchingPads<CSCCLCTDigi,T>(clct1, clct2, part));
  matches<T> padsAlct(matchingPads<CSCALCTDigi,T>(alct1, alct2, part));
  return intersection(padsClct, padsAlct);
}


#endif
