/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 * $Date: 2007/03/21 15:50:43 $
 * $Revision: 1.11 $
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>

#include <iomanip>

using namespace std;

/// Constructors
CSCCLCTDigi::CSCCLCTDigi(const int valid, const int quality, const int pattern,
			 const int striptype, const int bend, const int strip,
			 const int cfeb, const int bx, const int trknmb) {
  valid_     = valid;
  quality_   = quality;
  pattern_   = pattern;
  striptype_ = striptype;
  bend_      = bend;
  strip_     = strip;
  cfeb_      = cfeb;
  bx_        = bx;
  trknmb_    = trknmb;
}

/// Default
CSCCLCTDigi::CSCCLCTDigi () {
  clear(); // set contents to zero
}

/// Clears this CLCT.
void CSCCLCTDigi::clear() {
  valid_     = 0;
  quality_   = 0;
  pattern_   = 0;
  striptype_ = 0;
  bend_      = 0;
  strip_     = 0;
  cfeb_      = 0;
  bx_        = 0;
  trknmb_    = 0;
  fullbx_   = 0;
}

bool CSCCLCTDigi::operator > (const CSCCLCTDigi& rhs) const {
  bool returnValue = false;
  /* The quality value for the cathodeLCT is different than for the
     anodeLCT.  Remember that for the anodeLCT, the more layers that were
     hit the higher the quality. For the cathodeLCT there is a pattern
     assignment given.  This is based on the range of strips hit and
     the number of layers.  The hits on the strips are divided into
     high Pt (transverse momentum) and low Pt. A half strip pattern check
     is used for high Pt; and a di-strip pattern is used for low Pt.
     The order of quality from highest to lowest is 6/6 halfstrip, 
     5/6 halfstrip, 6/6 distrip, 4/6 halfstrip, 5/6 distrip, 4/6 distrip.  
     (see CSCCathodeLCTProcessor for further details.) -JM
  */
  int quality      = getQuality();
  int rhsQuality   = rhs.getQuality();
  int stripType    = getStripType();
  int rhsStripType = rhs.getStripType();

#ifdef TB
  // Test beams' implementation.
#ifdef TBs
  // This algo below was used in 2003 and 2004 test beams, but not in MTCC.
  if (stripType == rhsStripType) { // both di-strip or half-strip
    if      (quality >  rhsQuality) {returnValue = true;}
#ifdef LATER
    else if (quality == rhsQuality) {
      // The rest SEEMS NOT TO BE USED at the moment.  Brian's comment:
      // "There is a bug in the TMB firmware in terms of swapping the lcts."
      // In the case of identical quality, select higher pattern.
      int pattern    = getPattern();
      int rhsPattern = rhs.getPattern();
      if (pattern > rhsPattern) {returnValue = true;}
      else if (pattern == rhsPattern) {
	// In the case of identical pattern, select lower key strip number.
	if (getKeyStrip() < rhs.getKeyStrip()) {returnValue = true;}
      }
    }
#endif
  }
  else if (stripType > rhsStripType) {
    // Always select half-strip pattern over di-strip pattern.
    returnValue = true;
  }
#else
  // MTCC variant.
  if (quality > rhsQuality) {returnValue = true;}
#endif

#else
  // Hack to preserve old behaviour; needs to be clarified.
  quality    -= 3;
  rhsQuality -= 3;
  if (quality < 0 || rhsQuality < 0) {
    std::cout << " +++ CSCCLCTDigi, overloaded > : undefined qualities "
	      << quality << " " << rhsQuality << " ... Do nothing +++"
	      << std::endl;
    return returnValue;
  }
  // Default ORCA option.
  if (stripType == rhsStripType) { // both di-strip or half-strip
    if (quality > rhsQuality) {returnValue = true;}
    else if (quality == rhsQuality) {
      // In the case of cathode LCTs with identical quality, the lower
      // strip number is selected.
      if (getKeyStrip() < rhs.getKeyStrip()) {returnValue = true;}
    }
  }
  else if (stripType > rhsStripType) { // halfstrip, distrip
    // 5/6, 6/6 halfstrip better than all but 6/6 distrip:
    // If halfstrip quality is 2 or 3, it beats all distrip qualities.
    // If halfstrip quality is 1, it beats everything except 
    // distrip quality or 3.
    if (quality >= rhsQuality - 1) {returnValue = true;}
  }
  else if (stripType < rhsStripType) { // distrip, halfstrip
    // If distrip quality is 3, it beats a halfstrip quality of 1.
    if (quality - 1 > rhsQuality) {returnValue = true;}
  }
#endif

  return returnValue;
}

bool CSCCLCTDigi::operator == (const CSCCLCTDigi& rhs) const {
  bool returnValue = false;

  // Exact equality.
  if (isValid()      == rhs.isValid()    && getQuality() == rhs.getQuality() &&
      getPattern()   == rhs.getPattern() && getKeyStrip()== rhs.getKeyStrip()&&
      getStripType() == rhs.getStripType() && getBend()  == getBend()        &&
      getBX()        == rhs.getBX()) {
    returnValue = true;
  }
  else {
    int stripType    = getStripType();
    int rhsStripType = rhs.getStripType();

    // Note: if both muons are either high and low pT, then there's the chance
    // that one of them is at exactly the strip of the other. Don't
    // want to chuck out muons that way!
    // The numbering is not obvious because of the 'staggering' on each of the
    // layers.  When the staggering is completely understood, this algorithm
    // should be re-checked for consistency. -JM
    if (stripType != rhsStripType) {
      if (abs(getKeyStrip() - rhs.getKeyStrip()) < 5) {
	returnValue = true;
      }
    }
  }
  return returnValue;
}

bool CSCCLCTDigi::operator != (const CSCCLCTDigi& rhs) const {
  bool returnValue = false;
  // Check exact equality.
  if (isValid()      != rhs.isValid()    || getQuality() != rhs.getQuality() ||
      getPattern()   != rhs.getPattern() || getKeyStrip()!= rhs.getKeyStrip()||
      getStripType() != rhs.getStripType() || getBend()  != getBend()        ||
      getBX()        != rhs.getBX()) {
    returnValue = true;
  }
  return returnValue;
}

/// Debug
void CSCCLCTDigi::print() const {
  if (isValid()) {
    char stripType = (getStripType() == 0) ? 'D' : 'H';
    char bend      = (getBend()      == 0) ? 'L' : 'R';

    std::cout << " CSC CLCT #"    << setw(1) << getTrknmb()
	      << ": Valid = "     << setw(1) << isValid()
	      << " Key Strip = "  << setw(3) << getKeyStrip()
	      << " Strip = "      << setw(2) << getStrip()
	      << " Quality = "    << setw(1) << getQuality()
	      << " Pattern = "    << setw(1) << getPattern()
	      << " Bend = "       << setw(1) << bend
	      << " Strip type = " << setw(1) << stripType
	      << " CFEB ID = "    << setw(1) << getCFEB()
	      << " BX = "         << setw(1) << getBX() << std::endl;
  }
  else {
    std::cout << "Not a valid Cathode LCT." << std::endl;
  }
}
