#ifndef DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H
#define DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include <ostream>

/** Precision readout digi from QIE10 including TDC information

 */
class QIE10DataFrame {
public:

  static const int WORDS_PER_SAMPLE = 2;
  static const int HEADER_WORDS = 1;
  static const int FLAG_WORDS = 1;

  QIE10DataFrame() { }
  QIE10DataFrame(edm::DataFrame const & df) : m_data(df) { }

  class Sample {
  public:
    Sample(const edm::DataFrame& frame, edm::DataFrame::size_type i) : frame_(frame),i_(i) { }
    static const int MASK_ADC = 0xFF;
    static const int MASK_LE_TDC = 0x3F;
    static const int MASK_TE_TDC = 0x1F;
    static const int OFFSET_TE_TDC = 6;
    static const int MASK_SOI = 0x2000;
    static const int MASK_OK = 0x1000;
    static const int MASK_CAPID = 0x3;
    static const int OFFSET_CAPID = 12;
    int adc() const { return frame_[i_]&MASK_ADC; }
    int le_tdc() const { return frame_[i_+1]&MASK_LE_TDC; }
    int te_tdc() const { return (frame_[i_+1]>>OFFSET_TE_TDC)&MASK_TE_TDC; }
    bool ok() const { return frame_[i_]&MASK_OK; }
    bool soi() const { return frame_[i_]&MASK_SOI; }
    int capid() const { return (frame_[i_+1]>>OFFSET_CAPID)&MASK_CAPID; }
  private:
    const edm::DataFrame& frame_;
    edm::DataFrame::size_type i_;
  };

  void copyContent(const QIE10DataFrame& src);

  /// Get the detector id
  DetId detid() const { return DetId(m_data.id()); }
  edm::DataFrame::id_type id() const { return m_data.id(); }
  /// more accessors
  edm::DataFrame::size_type size() const { return m_data.size(); }
  /// iterators
  edm::DataFrame::iterator begin() { return m_data.begin(); }
  edm::DataFrame::iterator end() { return m_data.end(); }
  edm::DataFrame::const_iterator begin() const { return m_data.begin(); }
  edm::DataFrame::const_iterator end() const { return m_data.end(); }
  /// total number of samples in the digi
  int samples() const { return (size()-HEADER_WORDS-FLAG_WORDS)/WORDS_PER_SAMPLE; }
  /// for backward compatibility
  int presamples() const;
  /// get the flavor of the frame
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  int flavor() const { return ((m_data[0]>>OFFSET_FLAVOR)&MASK_FLAVOR); }
  /// was there a link error?
  static const int MASK_LINKERROR = 0x800;
  bool linkError() const { return m_data[0]&MASK_LINKERROR; }
  /// was this a mark-and-pass ZS event?
  static const int MASK_MARKPASS = 0x100;
  bool zsMarkAndPass() const {return m_data[0]&MASK_MARKPASS; }
  /// set ZS params
  void setZSInfo(bool markAndPass);
  /// get the sample
  inline Sample operator[](edm::DataFrame::size_type i) const { return Sample(m_data,i*WORDS_PER_SAMPLE+HEADER_WORDS); }
  /// set the sample contents
  void setSample(edm::DataFrame::size_type isample, int adc, int le_tdc, int te_tdc, int capid, bool soi=false, bool ok=true);
  /// get the flag word
  uint16_t flags() const { return m_data[size()-1]; }
  /// set the flag word
  void setFlags(uint16_t v);
  
  private:
   edm::DataFrame m_data;

};

std::ostream& operator<<(std::ostream&, const QIE10DataFrame&);


#endif // DATAFORMATS_HCALDIGI_QIE10DATAFRAME_H
