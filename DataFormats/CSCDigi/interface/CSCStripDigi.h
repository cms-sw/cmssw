#ifndef CSCStripDigi_CSCStripDigi_h
#define CSCStripDigi_CSCStripDigi_h

/** \class CSCStripDigi
 *
 * Digi for CSC Cathode Strips.
 *  
 *  $Date: 2006/03/01 09:40:30 $
 *  $Revision: 1.4 $
 *
 * \author M. Schmitt, Northwestern
 *
 */

#include <vector>

class CSCStripDigi{

public:

  //
  // The definition of the CSC strip Digi.
  // This should never be used directly, only by calling data().
  // Made public to be able to generate lcgdict, SA, 27/4/05
  // 

  //Although data is contained withing this struct it's not packed now
  //for now we do not need reinterpret casts to access data memebers.
  struct theStripDigi {
    unsigned int strip;
    std::vector<int> ADCCounts;
  };

  // Construct from the strip number and the ADC readings.
  explicit CSCStripDigi (int strip, std::vector<int> ADCCounts);

  // Construct from the struct.
  explicit CSCStripDigi (theStripDigi aStripDigi);

  // Copy constructor
  CSCStripDigi (const CSCStripDigi& digi);

  // Default construction.
  CSCStripDigi ();

  // Assignment operator
  CSCStripDigi& operator=(const CSCStripDigi& digi);

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

  // Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCStripDigis;

  // Set data words
  void set(int strip, std::vector<int> ADCCounts);
  void setData(theStripDigi p);

  // access
  theStripDigi* data();
  const theStripDigi* data() const;
  theStripDigi aStripDigi;
};

#include<iostream>
// needed by COBRA
//inline std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi) {
//  return o << " " << digi.strip()
//	   << " " << digi.ADCCounts();
//}
#endif

