#ifndef DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
#define DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#include <ostream>

/** Precision readout digi from QIE11 including TDC information

 */
class QIE11DataFrame : protected edm::DataFrame {
public:

  static const int WORDS_PER_SAMPLE = 1;
  static const int HEADER_WORDS = 1;
  static const int FLAG_WORDS = 1;

  QIE11DataFrame() { }
  QIE11DataFrame(const edm::DataFrameContainer& c, edm::DataFrame::size_type i) : edm::DataFrame(c,i) { }
  QIE11DataFrame(edm::DataFrame df) : edm::DataFrame(df) { }

  class Sample {
  public:
    Sample(const edm::DataFrame& frame, edm::DataFrame::size_type i) : frame_(frame),i_(i) { }
    static const int MASK_ADC = 0xFF;
    static const int MASK_TDC = 0x3F;
    static const int OFFSET_TDC = 8; // 8 bits
    static const int MASK_SOI = 0x4000;
    static const int MASK_CAPID = 0x3;
    static const int OFFSET_CAPID = 8;
    int adc() const { return frame_[i_]&MASK_ADC; }
    int tdc() const { return (frame_[i_]>>8)&MASK_TDC; }
    bool soi() const { return frame_[i_]&MASK_SOI; }
    int capid() const { return ((((frame_[0]>>OFFSET_CAPID)&MASK_CAPID)+i_)&MASK_CAPID); }
  private:
    const edm::DataFrame& frame_;
    edm::DataFrame::size_type i_;
  };

  /// Get the detector id
  DetId detid() const { return DetId(id()); }
  /// total number of samples in the digi
  int samples() const { return (size()-HEADER_WORDS-FLAG_WORDS)/WORDS_PER_SAMPLE; }
  /// get the flavor of the frame
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  int flavor() const { return ((edm::DataFrame::operator[](0)>>OFFSET_FLAVOR)&MASK_FLAVOR); }
  /// was there a link error?
  static const int MASK_LINKERROR = 0x800;
  bool linkError() const { return edm::DataFrame::operator[](0)&MASK_LINKERROR; } 
  /// was there a capid rotation error?
  static const int MASK_CAPIDERROR = 0x400;
  bool capidError() const { return edm::DataFrame::operator[](0)&MASK_CAPIDERROR; } 
  /// was this a mark-and-pass ZS event?
  bool wasMarkAndPass() const {return (flavor()==1); }
  /// get the sample
  inline Sample operator[](edm::DataFrame::size_type i) const { return Sample(*this,i+HEADER_WORDS); }
  void setCapid0(int cap0);
  /// set the sample contents
  void setSample(edm::DataFrame::size_type isample, int adc, int tdc, bool soi=false);
  /// get the flag word
  uint16_t flags() const { return edm::DataFrame::operator[](size()-1); }
  /// set the flag word
  void setFlags(uint16_t v);

};

std::ostream& operator<<(std::ostream&, const QIE11DataFrame&);


#endif // DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
