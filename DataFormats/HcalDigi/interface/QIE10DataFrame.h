#ifndef DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H
#define DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include <ostream>

/** Precision readout digi from QIE10 including TDC information

 */
class QIE10DataFrame : protected edm::DataFrame {
public:

  static const int WORDS_PER_SAMPLE = 2;
  static const int HEADER_WORDS = 1;

  QIE10DataFrame() { }
  QIE10DataFrame(const edm::DataFrameContainer& c, edm::DataFrame::size_type i) : edm::DataFrame(c,i) { }
  QIE10DataFrame(edm::DataFrame df) : edm::DataFrame(df) { }

  class Sample {
  public:
    Sample(const edm::DataFrame& frame, edm::DataFrame::size_type i) : frame_(frame),i_(i) { }
    int adc() const { return frame_[i_]&0xFF; }
    int le_tdc() const { return frame_[i_+1]&0x3F; }
    int te_tdc() const { return (frame_[i_]>>6)&0x1F; }
    bool ok() const { return frame_[i_]&0x1000; }
    bool soi() const { return frame_[i_]&0x2000; }
    int capid() const { return (frame_[i_+1]>>12)&0x3; }
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
  /// was this a mark-and-pass ZS event?  
  bool wasMarkAndPass() const {return edm::DataFrame::operator[](0)&0x100; }
  /// get the sample
  inline Sample operator[](edm::DataFrame::size_type i) const { return Sample(*this,i*2+1); }
  /// set the sample contents
  void setSample(edm::DataFrame::size_type isample, int adc, int le_tdc, int fe_tdc, int capid, bool soi=false, bool ok=true);
  
};

std::ostream& operator<<(std::ostream&, const QIE10DataFrame&);


#endif // DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H
