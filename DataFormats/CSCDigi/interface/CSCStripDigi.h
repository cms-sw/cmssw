#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>
#include <iosfwd>
#include <cstdint>

class CSCStripDigi{

public:

   // Construct from the strip number and all the other data members.
  CSCStripDigi (const int & istrip, const std::vector<int> & vADCCounts, const std::vector<uint16_t> & vADCOverflow, const std::vector<uint16_t> & vOverlap, 
                const std::vector<uint16_t> & vErrorstat ) :
    strip(istrip),
    ADCCounts(vADCCounts),
    ADCOverflow(vADCOverflow),
    OverlappedSample(vOverlap),
    Errorstat(vErrorstat) {}

  // Construct from the strip number and the ADC readings.
  CSCStripDigi (const int & istrip, const std::vector<int> & vADCCounts):
    strip(istrip),
    ADCCounts(vADCCounts),
    ADCOverflow(8,0),
    OverlappedSample(8,0),
    Errorstat(8,0){}


  CSCStripDigi ():
    strip(0),
    ADCCounts(8,0),
    ADCOverflow(8,0),
    OverlappedSample(8,0),
    Errorstat(8,0){}


  // Digis are equal if they are on the same strip and have same ADC readings
  bool operator==(const CSCStripDigi& digi) const;

  // Get the strip number. counts from 1.
  int getStrip() const { return strip;}

  /// Get ADC readings
  std::vector<int> const & getADCCounts() const { return ADCCounts; }


  /// Get L1APhase from OverlappedSample (9th bit)
  std::vector<int> getL1APhase() const {
     std::vector<int> L1APhaseResult(getOverlappedSample().size());
     for (int i=0; i<(int)getOverlappedSample().size(); i++) 
          L1APhaseResult[i] = (getOverlappedSample()[i]>>8) & 0x1;
     return  L1APhaseResult;
  }
  
  int getL1APhase(int i) const {
     return (getOverlappedSample()[i]>>8) & 0x1;
  }
  
  /// Other getters
  std::vector<uint16_t> const & getADCOverflow() const {return ADCOverflow;}
  std::vector<uint16_t> const & getOverlappedSample() const {return OverlappedSample;}
  std::vector<uint16_t> const & getErrorstat() const {return Errorstat;}

  // Set the strip number
  void setStrip(int istrip) { strip = istrip; }

  // Set with a vector of ADC readings
  void setADCCounts (const std::vector<int>& ADCCounts);

  // Print content of digi
  void print() const;

  ///methods for calibrations
  float pedestal() const {return 0.5f*(ADCCounts[0]+ADCCounts[1]);}
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

