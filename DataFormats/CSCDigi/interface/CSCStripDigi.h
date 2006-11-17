#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *  
 *  $Date: 2006/05/16 15:22:57 $
 *  $Revision: 1.12 $
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCStripDigi{

public:

  // Construct from the strip number and all the other data members.
  CSCStripDigi (int strip, std::vector<int> ADCCounts, std::vector<uint16_t> ADCOverflow,
		std::vector<uint16_t> ContrData,  std::vector<uint16_t> Overlap,
		std::vector<uint16_t> Errorstat);

  // Construct from the strip number and the ADC readings.
  CSCStripDigi (int strip, std::vector<int> ADCCounts);


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
  std::vector<uint16_t> getControllerData() const {return ControllerData;}
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
  std::vector<uint16_t> ControllerData;
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

