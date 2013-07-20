#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *
 *  $Date: 2013/04/22 22:39:23 $
 *  $Revision: 1.23 $
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

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

  /// Get ADC readings
  std::vector<int> getADCCounts() const ;
  
  /// Get L1APhase from OverlappedSample (9th bit)
  std::vector<int> getL1APhase() const ; 
  
  /// Other getters
  std::vector<uint16_t> getADCOverflow() const {return ADCOverflow;}
  std::vector<uint16_t> getOverlappedSample() const {return OverlappedSample;}
  std::vector<uint16_t> getErrorstat() const {return Errorstat;}

  // Set the strip number
  void setStrip(int istrip) { strip = istrip; }

  // Set with a vector of ADC readings
  void setADCCounts (const std::vector<int>& ADCCounts);

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

std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi);

#endif

