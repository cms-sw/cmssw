#ifndef DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
#define DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include <ostream>

/** Precision readout digi from QIE11 including TDC information

 */
class QIE11DataFrame : protected edm::DataFrame {
public:

  static const int WORDS_PER_SAMPLE = 1;
  static const int HEADER_WORDS = 1;

  QIE11DataFrame() { }
  QIE11DataFrame(const edm::DataFrameContainer& c, edm::DataFrame::size_type i) : edm::DataFrame(c,i) { }
  QIE11DataFrame(edm::DataFrame df) : edm::DataFrame(df) { }

  class Sample {
  public:
    Sample(const edm::DataFrame& frame, edm::DataFrame::size_type i) : frame_(frame),i_(i) { }
    int adc() const { return frame_[i_]&0xFF; }
    int tdc() const { return (frame_[i_]>>8)&0x3F; }
    bool soi() const { return frame_[i_]&0x4000; }
    int capid() const { return ((((frame_[0]>>8)&0x3)+i_)%4); }
  private:
    const edm::DataFrame& frame_;
    edm::DataFrame::size_type i_;
  };

  /// Get the detector id
  DetId detid() const { return DetId(id()); }
  /// total number of samples in the digi
  int samples() const { return (size()-1)/2; }
  /// get the flavor of the frame
  int flavor() const { return ((edm::DataFrame::operator[](0)>>12)&0x7); }
  /// was there a link error?
  bool linkError() const { return edm::DataFrame::operator[](0)&0x800; } 
  /// was there a capid rotation error?
  bool capidError() const { return edm::DataFrame::operator[](0)&0x400; } 
  /// was this a mark-and-pass ZS event?  
  bool wasMarkAndPass() const {return (flavor()==1); }
  /// get the sample
  inline Sample operator[](edm::DataFrame::size_type i) const { return Sample(*this,i+1); }
  void setCapid0(int cap0);
  /// set the sample contents
  void setSample(edm::DataFrame::size_type isample, int adc, int tdc, bool soi=false);

};

std::ostream& operator<<(std::ostream&, const QIE11DataFrame&);


#endif // DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
