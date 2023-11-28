/* -*- C++ -*- */
#ifndef HcalUHTRData_H
#define HcalUHTRData_H

#include <cstdint>

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
  static const int FIBERS_PER_UHTR = 24;
  static const int CHANNELS_PER_FIBER_HF = 4;
  static const int CHANNELS_PER_FIBER_HBHE = 6;
  static const int CHANNELS_PER_FIBER_MAX = 8;

  HcalUHTRData();
  ~HcalUHTRData() {
    if (m_ownData != nullptr)
      delete[] m_ownData;
  }
  HcalUHTRData(int version_to_create);
  HcalUHTRData(const uint64_t* data, int length_words);
  HcalUHTRData(const HcalUHTRData&);

  HcalUHTRData& operator=(const HcalUHTRData&);

  /** \brief Get the version number of this event */
  inline int getFormatVersion() const { return m_formatVersion; }

  /** \brief Get a pointer to the raw data */
  inline const unsigned short* getRawData16() const { return m_raw16; }

  /** \brief Get the length of the raw data */
  inline const int getRawLengthBytes() const { return m_rawLength64 * sizeof(uint64_t); }

  class const_iterator {
  public:
    const_iterator(const uint16_t* ptr, const uint16_t* limit = nullptr);

    bool isHeader() const { return ((*m_ptr) & 0x8000) != 0; }
    int flavor() const { return ((*m_ptr) >> 12) & 0x7; }
    int errFlags() const;
    bool dataValid() const;
    int capid0() const { return ((*m_ptr) >> 8) & 0x3; }
    int channelid() const { return ((*m_ptr)) & 0xFF; }
    int technicalDataType() const;

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

    bool operator==(const const_iterator& i) { return m_ptr == i.m_ptr; }
    bool operator!=(const const_iterator& i) { return m_ptr != i.m_ptr; }
    const uint16_t* raw() const { return m_ptr; }

  private:
    void determineMode();
    const uint16_t *m_ptr, *m_limit;
    const uint16_t *m_header_ptr, *m_0th_data_ptr;
    int m_microstep;
    int m_stepclass;
    int m_flavor;
    int m_technicalDataType;
  };

  const_iterator begin() const;
  const_iterator end() const;

  /** \brief Get the HTR event number */
  inline uint32_t l1ANumber() const { return uint32_t(m_raw64[0] >> 32) & 0xFFFFFF; }
  /** \brief Get the HTR bunch number */
  inline uint32_t bunchNumber() const { return uint32_t(m_raw64[0] >> 20) & 0xFFF; }
  /** \brief Get the HTR orbit number */
  inline uint32_t orbitNumber() const { return uint32_t(m_raw64[1] >> 16) & 0xFFFF; }
  /** \brief Get the event type */
  int getEventType() const { return uint32_t(m_raw64[1] >> 40) & 0xF; }
  /** \brief Get the raw board id */
  inline uint32_t boardId() const { return uint32_t(m_raw64[1]) & 0xFFFF; }
  /** \brief Get the board crate */
  inline uint32_t crateId() const { return uint32_t(m_raw64[1]) & 0xFF; }
  /** \brief Get the board slot */
  inline uint32_t slot() const { return uint32_t(m_raw64[1] >> 8) & 0xF; }
  /** \brief Get the presamples */
  inline uint32_t presamples() const { return uint32_t(m_raw64[1] >> 12) & 0xF; }
  /** \brief Get the length from the uHTR header */
  inline uint32_t length64_uhtr() const { return uint32_t(m_raw64[0]) & 0xFFFFF; }
  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZS(int fiber, int fiberchan) const;
  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZSTP(int slb, int slbchan) const;

  /** \brief Get the HTR firmware version */
  unsigned int getFirmwareRevision() const { return uint32_t(m_raw64[1] >> 48) & 0xFFFF; }
  /** \brief Get the HTR firmware flavor */
  int getFirmwareFlavor() const { return uint32_t(m_raw64[1] >> 32) & 0xFF; }

protected:
  int m_formatVersion;
  int m_rawLength64;
  const uint64_t* m_raw64;
  const uint16_t* m_raw16;
  uint64_t* m_ownData;
};

#endif
