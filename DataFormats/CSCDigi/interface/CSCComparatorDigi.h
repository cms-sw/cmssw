#ifndef CSCComparatorDigi_CSCComparatorDigi_h
#define CSCComparatorDigi_CSCComparatorDigi_h

/** \class CSCComparatorDigi
 *
 * Digi for CSC Comparators.
 *  
 *  $Date: 2009/05/09 20:23:33 $
 *  $Revision: 1.12 $
 *
 * \author M. Schmitt, Northwestern
 *
 */
#include <iosfwd>
#include <vector>
#include <stdint.h>

class CSCComparatorDigi{

public:

  /// Construct from the strip number and the ADC readings.
  CSCComparatorDigi (int strip, int comparator, int timeBinWord);
  ///comparator here can be either 0 or 1 for left or right halfstrip of given strip

  /// Default construction.
  CSCComparatorDigi ();


  /// Digis are equal if they are on the same strip and have same Comparator data
  bool operator==(const CSCComparatorDigi& digi) const;

  /// sort by time first, then by strip
  bool operator<(const CSCComparatorDigi& digi) const;

  /// Get the strip number
    int getStrip() const { return strip_; }

  /// Get Comparator readings
  int getComparator() const { return comparator_; }

  /// Return the word with each bit corresponding to a time bin
  int getTimeBinWord() const { return timeBinWord_; }

  /// Return bin number of first time bin which is ON. Counts from 0.
  int getTimeBin() const;

  /** Return vector of the bin numbers for which time bins are ON.
   * e.g. if bits 0 and 13 fired, then this vector will contain the values 0 and 13
   */
  std::vector<int> getTimeBinsOn() const;

  /// Set the strip number
  void setStrip(int strip);

  /// Set Comparator data
  void setComparator (int comparator);

  /// Print content of digi
  void print() const;


private:

  uint16_t strip_;
  uint16_t comparator_;
  uint16_t timeBinWord_;

};

/// Output operator
std::ostream & operator<<(std::ostream & o, const CSCComparatorDigi& digi);

#endif


