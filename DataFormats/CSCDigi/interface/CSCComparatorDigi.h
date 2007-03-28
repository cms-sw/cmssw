#ifndef CSCComparatorDigi_CSCComparatorDigi_h
#define CSCComparatorDigi_CSCComparatorDigi_h

/** \class CSCComparatorDigi
 *
 * Digi for CSC Comparators.
 *  
 *  $Date: 2006/04/06 11:18:25 $
 *  $Revision: 1.5 $
 *
 * \author M. Schmitt, Northwestern
 *
 */
#include <boost/cstdint.hpp>

class CSCComparatorDigi{

public:

  // Construct from the strip number and the ADC readings.
  CSCComparatorDigi (int strip, int comparator, int timeBin);


  // Default construction.
  CSCComparatorDigi ();


  // Digis are equal if they are on the same strip and have same Comparator data
  bool operator==(const CSCComparatorDigi& digi) const;

  // sort by time first, then by strip
  bool operator<(const CSCComparatorDigi& digi) const;

  // Get the strip number
  int getStrip() const;

  // Get Comparator readings
  int getComparator() const;

  int getTimeBin() const;

  // Set the strip number
  void setStrip(int strip);

  // Set Comparator data
  void setComparator (int comparator);

  // Print content of digi
  void print() const;


private:
  friend class testCSCComparatorDigis;

  uint16_t strip;
  uint16_t comparator;
  uint16_t timeBin;

};

#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCComparatorDigi& digi) {
  return o << " " << digi.getStrip()
	   << " " << digi.getComparator()
	   << " " << digi.getTimeBin();
}  
#endif


