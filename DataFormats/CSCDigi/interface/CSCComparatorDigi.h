#ifndef CSCComparatorDigi_CSCComparatorDigi_h
#define CSCComparatorDigi_CSCComparatorDigi_h

/** \class CSCComparatorDigi
 *
 * Digi for CSC Comparators.
 *  
 *  $Date: 2006/11/16 16:55:38 $
 *  $Revision: 1.7 $
 *
 * \author M. Schmitt, Northwestern
 *
 */
#include <boost/cstdint.hpp>
#include <iosfwd>
#include <vector>

class CSCComparatorDigi{

public:

  /// Construct from the strip number and the ADC readings.
  CSCComparatorDigi (int strip, int comparator, int timeBinWord);


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


