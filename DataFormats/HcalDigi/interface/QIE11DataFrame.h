#ifndef DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
#define DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include <ostream>

/** Precision readout digi from QIE11 including TDC information

 */
class QIE11DataFrame {
public:

  static const int WORDS_PER_SAMPLE = 1;
  static const int HEADER_WORDS = 1;
  static const int FLAG_WORDS = 1;

  constexpr QIE11DataFrame() { }
  constexpr QIE11DataFrame(edm::DataFrame const & df) : m_data(df) { }

  class Sample {
  public:
    constexpr Sample(const edm::DataFrame& frame, edm::DataFrame::size_type i) : frame_(frame),i_(i) { }
    static const int MASK_ADC = 0xFF;
    static const int MASK_TDC = 0x3F;
    static const int OFFSET_TDC = 8; // 8 bits
    static const int MASK_SOI = 0x4000;
    static const int MASK_CAPID = 0x3;
    static const int OFFSET_CAPID = 8;
    constexpr int adc() const { return frame_[i_]&MASK_ADC; }
    constexpr int tdc() const { return (frame_[i_]>>OFFSET_TDC)&MASK_TDC; }
    constexpr bool soi() const { return frame_[i_]&MASK_SOI; }
    constexpr int capid() const { return ((((frame_[0]>>OFFSET_CAPID)&MASK_CAPID)+i_-HEADER_WORDS)&MASK_CAPID); }
  private:
    const edm::DataFrame& frame_;
    edm::DataFrame::size_type i_;
  };

  constexpr void copyContent(const QIE11DataFrame& digi) {
    for (edm::DataFrame::size_type i=0; i<size() && i<digi.size();i++){
      Sample sam = digi[i];
	  setSample(i,sam.adc(),sam.tdc(),sam.soi());
    }
  }

  /// Get the detector id
  constexpr DetId detid() const { return DetId(m_data.id()); }
  constexpr edm::DataFrame::id_type id() const { return m_data.id(); }
  /// more accessors
  constexpr edm::DataFrame::size_type size() const { return m_data.size(); }
  /// iterators
  constexpr edm::DataFrame::iterator begin() { return m_data.begin(); }
  constexpr edm::DataFrame::iterator end() { return m_data.end(); }
  constexpr edm::DataFrame::const_iterator begin() const { return m_data.begin(); }
  constexpr edm::DataFrame::const_iterator end() const { return m_data.end(); }
  /// total number of samples in the digi
  constexpr int samples() const { return (size()-HEADER_WORDS-FLAG_WORDS)/WORDS_PER_SAMPLE; }
  /// for backward compatibility
  constexpr int presamples() const {
    for (int i=0; i<samples(); i++) {
      if ((*this)[i].soi()) return i;
    }
    return -1;
  }
  /// get the flavor of the frame
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  constexpr int flavor() const { return ((m_data[0]>>OFFSET_FLAVOR)&MASK_FLAVOR); }
  /// was there a link error?
  static const int MASK_LINKERROR = 0x800;
  constexpr bool linkError() const { return m_data[0]&MASK_LINKERROR; } 
  /// was there a capid rotation error?
  static const int MASK_CAPIDERROR = 0x400;
  constexpr bool capidError() const { return m_data[0]&MASK_CAPIDERROR; } 
  /// was this a mark-and-pass ZS event?
  constexpr bool zsMarkAndPass() const {return (flavor()==1); }
  /// set ZS params
  constexpr void setZSInfo(bool markAndPass) {
	if(markAndPass) m_data[0] |= (markAndPass&MASK_FLAVOR)<<OFFSET_FLAVOR;
  }
  /// get the sample
  constexpr inline Sample operator[](edm::DataFrame::size_type i) const { return Sample(m_data,i+HEADER_WORDS); }
  constexpr void setCapid0(int cap0) {
    m_data[0]&=0xFCFF; // inversion of the capid0 mask
    m_data[0]|=((cap0&Sample::MASK_CAPID)<<Sample::OFFSET_CAPID);  
  }
  /// set the sample contents
  constexpr void setSample(edm::DataFrame::size_type isample, int adc, 
                           int tdc, bool soi=false) {
    if (isample>=size()) return;
    m_data[isample+1]=(adc&Sample::MASK_ADC)|(soi?(Sample::MASK_SOI):(0))|((tdc&Sample::MASK_TDC)<<Sample::OFFSET_TDC);
  }
  /// get the flag word
  constexpr uint16_t flags() const { return m_data[size()-1]; }
  /// set the flag word
  constexpr void setFlags(uint16_t v) {
    m_data[size()-1]=v;
  }

  private:
   edm::DataFrame m_data;

};

std::ostream& operator<<(std::ostream&, const QIE11DataFrame&);

#endif // DATAFORMATS_HCALDIGI_QIE11DATAFRAME_H
