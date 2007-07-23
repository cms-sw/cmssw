#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *  
 *  $Date: 2007/05/03 23:27:44 $
 *  $Revision: 1.15 $
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
  int getStrip() const;

  // Get ADC readings
  std::vector<int> getADCCounts() const;

  /// Other getters
  std::vector<uint16_t> getADCOverflow() const {return ADCOverflow;}
  std::vector<uint16_t> getOverlappedSample() const {return OverlappedSample;}
  std::vector<uint16_t> getErrorstat() const {return Errorstat;}

  // Set the strip number
  void setStrip(int strip);

  // Set with a vector of ADC readings
  void setADCCounts (std::vector<int> ADCCounts);

  // Print content of digi
  void print() const;

private:
  
  uint16_t strip;
  std::vector<int> ADCCounts;
  std::vector<uint16_t> ADCOverflow;
  std::vector<uint16_t> OverlappedSample;
  std::vector<uint16_t> Errorstat;
};

#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi) {
  o << " " << digi.getStrip();
  for (size_t i = 0; i<digi.getADCCounts().size(); ++i ){
    o <<" " <<(digi.getADCCounts())[i]; }
  return o;
  
}

#endif

