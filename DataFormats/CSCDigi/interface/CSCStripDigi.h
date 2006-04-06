#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *  
 *  $Date: 2006/04/05 10:06:51 $
 *  $Revision: 1.8 $
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCStripDigi{

public:

  // Construct from the strip number and the ADC readings.
  explicit CSCStripDigi (int strip, std::vector<int> ADCCounts);

  // Default construction.
  CSCStripDigi ();

  // Digis are equal if they are on the same strip and have same ADC readings
  bool operator==(const CSCStripDigi& digi) const;

  // Get the strip number
  int getStrip() const;

  // Get ADC readings
  std::vector<int> getADCCounts() const;

  // Set the strip number
  void setStrip(int strip);

  // Set with a vector of ADC readings
  void setADCCounts (std::vector<int> ADCCounts);

  // Print content of digi
  void print() const;

private:
  friend class testCSCStripDigis;
  
  uint16_t strip;
  std::vector<int> ADCCounts;

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

