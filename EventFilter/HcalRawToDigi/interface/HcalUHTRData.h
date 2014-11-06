/* -*- C++ -*- */
#ifndef HcalUHTRData_H
#define HcalUHTRData_H

#include <stdint.h>

/**  \class HcalUHTRData
 *
 *  Interpretive class for HcalUHTRData
 *  Since this class requires external specification of the length of the data, it is implemented
 *  as an interpreter, rather than a cast-able header class.
 *
 *  \author J. Mans - UMN
 */

class HcalUHTRData {
 public:
  static const int FIBERS_PER_UHTR             = 24;
  static const int CHANNELS_PER_FIBER_HF       = 4;
  static const int CHANNELS_PER_FIBER_HBHE     = 6;
  static const int CHANNELS_PER_FIBER_MAX      = 8;
  
  HcalUHTRData();
  ~HcalUHTRData() { if (m_ownData!=0) delete [] m_ownData; }
  HcalUHTRData(int version_to_create);
  HcalUHTRData(const uint64_t* data, int length_words);
  HcalUHTRData(const HcalUHTRData&);
  
  HcalUHTRData& operator=(const HcalUHTRData&);
  
  /** \brief Get the version number of this event */
  inline int getFormatVersion() const { return m_formatVersion; }
  
  /** \brief Get a pointer to the raw data */
  inline const unsigned short* getRawData16() const { return m_raw16; }
  
  /** \brief Get the length of the raw data */
  inline const int getRawLengthBytes() const { return m_rawLength64*sizeof(uint64_t); }
  
  class const_iterator {
  public:
    const_iterator(const uint16_t* ptr, const uint16_t* limit=0);
    
    bool isHeader() const { return ((*m_ptr)&0x8000)!=0; }    
    int flavor() const { return ((*m_ptr)>>12)&0x7; }    
    int errFlags() const { return ((*m_ptr)>>10)&0x3; }    
    int capid0() const { return ((*m_ptr)>>8)&0x3; }    
    int channelid() const { return ((*m_ptr))&0xFF; }    

    uint16_t value() const { return *m_ptr; }

    uint8_t adc() const;
    uint8_t le_tdc() const;
    uint8_t te_tdc() const;
    bool soi() const;
    uint8_t capid() const;
    bool ok() const;

    uint16_t operator*() const { return *m_ptr; }

    /** Increment operator is "magic" and adjusts steps to match channel requirements. */
    const_iterator& operator++();

    bool operator==(const const_iterator& i) { return m_ptr==i.m_ptr; }
    bool operator!=(const const_iterator& i) { return m_ptr!=i.m_ptr; }
    const uint16_t* raw() const { return m_ptr; }

  private:
    void determineMode();
    const uint16_t* m_ptr, *m_limit;
    const uint16_t* m_header_ptr, *m_0th_data_ptr;
    int m_microstep;
    int m_stepclass;
    int m_flavor;
  };    

  const_iterator begin() const;
  const_iterator end() const;

  class packer {
  public:
    packer(uint16_t* baseptr);
    void addHeader(int flavor, int errf, int cap0, int channelid);
    void addSample(int adc, bool soi=false, int retdc=0, int fetdc=0, int tdcstat=0);
    void addTP(int tpword, bool soi=false);
  private:
    uint16_t* m_baseptr;
    int m_ptr;
    int m_flavor;
    int m_ministep;
  };

  packer pack();


  /** \brief pack header and trailer (call _after_ pack)*/
  void packHeaderTrailer(int L1Anumber, int bcn, int submodule, int
			 orbitn, int pipeline, int ndd, int nps, int firmwareRev=0);

  /** \brief pack trailer with Mark and Pass bits */
  void packUnsuppressed(const bool* mp);
    
  /** \brief Get the HTR event number */
  inline uint32_t l1ANumber() const {  return uint32_t(m_raw64[0]>>32)&0xFFFFFF; }
  /** \brief Get the HTR bunch number */
  inline uint32_t bunchNumber() const { return uint32_t(m_raw64[0]>>20)&0xFFF; }
  /** \brief Get the HTR orbit number */
  inline uint32_t orbitNumber() const { return uint32_t(m_raw64[1]>>16)&0xFFFF; }
  /** \brief Get the raw board id */
  inline uint32_t boardId() const { return uint32_t(m_raw64[1])&0xFFFF; }
  /** \brief Get the board crate */
  inline uint32_t crateId() const { return uint32_t(m_raw64[1])&0xFF; }
  /** \brief Get the board slot */
  inline uint32_t slot() const { return uint32_t(m_raw64[1]>>8)&0xF; }

  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZS(int fiber, int fiberchan) const;
  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZSTP(int slb, int slbchan) const;
  
  /** \brief Get the HTR firmware version */
  unsigned int getFirmwareRevision() const { return uint32_t(m_raw64[1]>>48)&0xFFFF; }
  /** \brief Get the HTR firmware flavor */
  int getFirmwareFlavor() const { return uint32_t(m_raw64[1]>>32)&0xFF; }


protected:
  int m_formatVersion;
  int m_rawLength64;
  const uint64_t* m_raw64;
  const uint16_t* m_raw16;
  uint64_t* m_ownData;
};

#endif

