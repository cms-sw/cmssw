#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *  
 *  $Date: 2007/09/26 06:59:32 $
 *  $Revision: 1.17 $
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>

class CSCStripDigi{

public:

  // Construct from the strip number and all the other data members.
  CSCStripDigi (const int & strip, const std::vector<int> & ADCCounts, const std::vector<uint16_t> & ADCOverflow,
	        const std::vector<uint16_t> & Overlap,
		const std::vector<uint16_t> & Errorstat);

  // Construct from the strip number and the ADC readings.
  CSCStripDigi (const int & strip, const  std::vector<int> & ADCCounts);


  // Default construction.
  CSCStripDigi ();

  // Digis are equal if they are on the same strip and have same ADC readings
  bool operator==(const CSCStripDigi& digi) const;

  // Get the strip number
  int getStrip() const { return strip;}

  // Get ADC readings
  std::vector<int> getADCCounts() const { return ADCCounts; }

  /// Other getters
  std::vector<uint16_t> getADCOverflow() const {return ADCOverflow;}
  std::vector<uint16_t> getOverlappedSample() const {return OverlappedSample;}
  std::vector<uint16_t> getErrorstat() const {return Errorstat;}

  // Set the strip number
  void setStrip(int istrip) { strip = istrip; }

  // Set with a vector of ADC readings
  void setADCCounts (std::vector<int> ADCCounts);

  // Print content of digi
  void print() const;

  ///methods for calibrations
  float pedestal() const {return 0.5*(ADCCounts[0]+ADCCounts[1]);}
  float amplitude() const {return ADCCounts[4]-pedestal();}

private:
  
  uint16_t strip;
  std::vector<int> ADCCounts;
  std::vector<uint16_t> ADCOverflow;
  std::vector<uint16_t> OverlappedSample;
  std::vector<uint16_t> Errorstat;
};

#include<iostream>
// once upon a time was needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi) {
  o << " " << digi.getStrip();
  for (size_t i = 0; i<digi.getADCCounts().size(); ++i ){
    o <<" " <<(digi.getADCCounts())[i]; }
  return o;
  
}

#endif

