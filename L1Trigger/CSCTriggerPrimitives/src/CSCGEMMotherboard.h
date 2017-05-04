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


class CSCGEMMotherboard : public CSCUpgradeMotherboard
{
public:

  CSCGEMMotherboard(unsigned endcap, unsigned station, unsigned sector,
                    unsigned subsector, unsigned chamber,
                    const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCGEMMotherboard();

  virtual ~CSCGEMMotherboard();

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
  void correlateLCTsGEM(T best, T second, const GEMCoPadDigiIds& coPads, 
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, enum CSCPart);
  
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const GEMCoPadDigi& gem, enum CSCPart, int i); 
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCCLCTDigi& clct, const GEMCoPadDigi& gem, enum CSCPart, int i); 
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct, const GEMCoPadDigi& gem, enum CSCPart p, int i); 
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct, const GEMPadDigi& gem, enum CSCPart p, int i); 
  
  void retrieveGEMPads(const GEMPadDigiCollection* pads, unsigned id);
  void retrieveGEMCoPads();

  template <class T>  
  unsigned int findQualityGEM(const CSCALCTDigi&, const CSCCLCTDigi&);

  template <class T>  
  void printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart);

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
void CSCGEMMotherboard::correlateLCTsGEM(S bestLCT,
					 S secondLCT,
					 const GEMCoPadDigiIds& coPads, 
					 CSCCorrelatedLCTDigi& lct1,
					 CSCCorrelatedLCTDigi& lct2, 
					 enum CSCPart p)
{
  bool bestValid     = bestLCT.isValid();
  bool secondValid   = secondLCT.isValid();

  // get best matching copad1
  GEMCoPadDigi bestCoPad = bestMatchingPad<GEMCoPadDigi>(bestLCT, coPads, p);
  GEMCoPadDigi secondCoPad = bestMatchingPad<GEMCoPadDigi>(secondLCT, coPads, p);

  if (bestValid and !secondValid) secondLCT = bestLCT;
  if (!bestValid and secondValid) bestLCT   = secondLCT;

  bool lct_trig_enable;
  if (std::is_same<S, CSCALCTDigi>::value) lct_trig_enable = alct_trig_enable;
  if (std::is_same<S, CSCCLCTDigi>::value) lct_trig_enable = clct_trig_enable;
  
  if ((lct_trig_enable  and bestLCT.isValid()) or
      (match_trig_enable and bestLCT.isValid()))
    {
    lct1 = constructLCTsGEM(bestLCT, bestCoPad, p, 1);
  }
  
  if ((lct_trig_enable  and secondLCT.isValid()) or
      (match_trig_enable and secondLCT.isValid() and secondLCT != bestLCT))
    {
      lct2 = constructLCTsGEM(secondLCT, secondCoPad, p, 2);
    }
}

template <class S>
S CSCGEMMotherboard::bestMatchingPad(const CSCALCTDigi& alct1, const matches<S>& pads, enum CSCPart)
{
  S result;
  if (not alct1.isValid()) return result;

  // return the first one with the same roll number
  for (auto& p: pads){
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
  for (auto p: pads){
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
  for (auto& p: pads){
    float averagePadNumberGEM = getAvePad(p.second);
    // add another safety to make sure that the deltaPad is not larger than max value!!!
    if (std::abs(averagePadNumberCSC - averagePadNumberGEM) < minDeltaPad and getRoll(p) == getRoll(alct1)){
      minDeltaPad = std::abs(averagePadNumberCSC - averagePadNumberGEM);
      result = p.second;
    }
  }
  return result;
}

template <class S>
matches<S> CSCGEMMotherboard::matchingPads(const CSCALCTDigi& alct, enum CSCPart part)
{
  matches<S> result;
  if (not alct.isValid()) return result;

  // get the generic LUT  
  boost::variant<GEMPadDigiIdsBX, GEMCoPadDigiIdsBX> variant_lut;
  if (std::is_same<S, GEMPadDigi>::value) variant_lut = pads_;
  else                                    variant_lut = coPads_;
  auto actual_lut = boost::get<std::map<int, matches<S> > >(variant_lut);

  std::pair<int,int> alctRoll = (*getLUT()->CSCGEMMotherboardLUT::get_csc_wg_to_gem_roll(par))[alct.getKeyWG()];
  for (auto p: actual_lut[alct.getBX()]){
    auto padRoll(getRoll(p));
    // only pads in overlap are good for ME1A
    if (part==CSCPart::ME1A and !isPadInOverlap(padRoll)) continue;
    if (alctRoll.first == -99 and alctRoll.second == -99) continue;  //invalid region
    else if (alctRoll.first == -99 and !(padRoll <= alctRoll.second)) continue; // top of the chamber
    else if (alctRoll.second == -99 and !(padRoll >= alctRoll.first)) continue; // bottom of the chamber
    else if ((alctRoll.first != -99 and alctRoll.second != -99) and // center
	     (alctRoll.first > padRoll or padRoll > alctRoll.second)) continue;
    result.push_back(p);
  }
  return result;
}

template <class S>
matches<S> CSCGEMMotherboard::matchingPads(const CSCCLCTDigi& clct, enum CSCPart part)
{
  matches<S> result;
  if (not clct.isValid()) return result;
  
  int deltaBX;
  // get the generic LUT  
  boost::variant<GEMPadDigiIdsBX, GEMCoPadDigiIdsBX> variant_lut;
  if (std::is_same<S, GEMPadDigi>::value) {
    variant_lut = pads_;
    deltaBX = maxDeltaBXPad_;
  }
  else{
    variant_lut = coPads_;
    deltaBX = maxDeltaBXCoPad_;
  }
  auto actual_lut = boost::get<std::map<int, matches<S> > >(variant_lut);
  
  auto mymap = (*getLUT()->get_csc_hs_to_gem_pad(par, part));
  const int lowPad(mymap[clct.getKeyStrip()].first);
  const int highPad(mymap[clct.getKeyStrip()].second);
  for (auto p: actual_lut[clct.getBX()]){
    auto padRoll(getAvePad(p.second));
    int pad_bx = getBX(p.second)+lct_central_bx;
    if (std::abs(clct.getBX()-pad_bx)>deltaBX) continue;
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

template <class S>  
void CSCGEMMotherboard::printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart part)
{
  bool iscopad = false;
  // get the generic LUT  
  boost::variant<GEMPadDigiIdsBX, GEMCoPadDigiIdsBX> variant_lut;
  if (std::is_same<S, GEMPadDigi>::value) variant_lut = pads_;
  else {
    variant_lut = coPads_;
    iscopad = true;
  }
  auto actual_lut = boost::get<std::map<int, matches<S> > >(variant_lut);

  // pads or copads?
  const bool hasPads(actual_lut.size()!=0);
  
  std::cout << "------------------------------------------------------------------------" << std::endl;
  bool first = true;
  for (int bx = bx_start; bx <= bx_stop; bx++) {
    std::vector<std::pair<unsigned int, S> > in_pads = actual_lut[bx];
    if (first) {
      if (!iscopad) std::cout << "* GEM trigger pads: " << std::endl;
      else          std::cout << "* GEM trigger coincidence pads: " << std::endl;
    }
    first = false;
    if (!iscopad) std::cout << "N(pads) BX " << bx << " : " << in_pads.size() << std::endl;
    else          std::cout << "N(copads) BX " << bx << " : " << in_pads.size() << std::endl;
    if (hasPads){
      for (auto pad : in_pads){ 
	std::cout << "\tdetId " << GEMDetId(pad.first) << ", pad = " << pad.second;
	auto roll_id(GEMDetId(pad.first));
        if (part==CSCPart::ME11 and isPadInOverlap(GEMDetId(roll_id).roll())) std::cout << " (in overlap)" << std::endl;
        else std::cout << std::endl;
      }
    }
    else
      break;
  }
}

template <class S>  
unsigned int CSCGEMMotherboard::findQualityGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT)
{
  /*
    Same LCT quality definition as standard LCTs
    a4 and c4 take GEMs into account!!!
  */
  
  unsigned int quality = 0;
  bool hasPad = false, hasCoPad = false;
  if (std::is_same<S, GEMPadDigi>::value) {
    hasPad = true;
  }
  else {
    hasCoPad = true;
  }

  // 2008 definition.
  if (!(aLCT.isValid()) || !(cLCT.isValid())) {
    if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
    else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
    else quality = 0; // both absent; should never happen.
  }
  else {
    const int pattern(cLCT.getPattern());
    if (pattern == 1) quality = 3; // layer-trigger in CLCT
    else {
      // ALCT quality is the number of layers hit minus 3.
      // CLCT quality is the number of layers hit.
      int n_gem = 0;  
      if (hasPad) n_gem = 1;
      if (hasCoPad) n_gem = 2;
      bool a4;
      // GE11 
      if (theStation==1) {
	a4 = (aLCT.getQuality() >= 1);
      }
      // GE21
      else if (theStation==2) {
	a4 = (aLCT.getQuality() >= 1) or (aLCT.getQuality() >= 0 and n_gem >=1);
      }
      else {
	return -1; 
      }
      const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and n_gem>=1));
      //              quality = 4; "reserved for low-quality muons in future"
      if      (!a4 && !c4) quality = 5; // marginal anode and cathode
      else if ( a4 && !c4) quality = 6; // HQ anode, but marginal cathode
      else if (!a4 &&  c4) quality = 7; // HQ cathode, but marginal anode
      else if ( a4 &&  c4) {
	if (aLCT.getAccelerator()) quality = 8; // HQ muon, but accel ALCT
	else {
	  // quality =  9; "reserved for HQ muons with future patterns
	  // quality = 10; "reserved for HQ muons with future patterns
	  if (pattern == 2 || pattern == 3)      quality = 11;
	  else if (pattern == 4 || pattern == 5) quality = 12;
	  else if (pattern == 6 || pattern == 7) quality = 13;
	  else if (pattern == 8 || pattern == 9) quality = 14;
	  else if (pattern == 10)                quality = 15;
	  else {
	    if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
			      << "+++ findQuality: Unexpected CLCT pattern id = "
			      << pattern << "+++\n";
	  }
	}
      }
    }
  }
  return quality;
}

#endif
