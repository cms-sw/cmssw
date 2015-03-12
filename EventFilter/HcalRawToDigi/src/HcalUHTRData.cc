#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include <string.h>

static const int HEADER_LENGTH_16BIT=2*sizeof(uint64_t)/sizeof(uint16_t);

HcalUHTRData::const_iterator::const_iterator(const uint16_t* ptr, const uint16_t* limit) : m_ptr(ptr), m_limit(limit) {
  if (isHeader()) determineMode();
}

HcalUHTRData::const_iterator& HcalUHTRData::const_iterator::operator++() {
  if (m_ptr==m_limit) return *this;
  if (m_stepclass==0) m_ptr++;
  else if (m_stepclass==1) {
    if (m_microstep==0) { m_ptr++; m_microstep++; }
    else { m_microstep--; }
  } 
  else if (m_stepclass==2) {
    if (isHeader()) { m_ptr++; }
    else { m_ptr+=2; }
  }

  if (isHeader()) {
    determineMode();
    m_header_ptr = m_ptr;
    m_0th_data_ptr = m_header_ptr + 1;
  }
  return *this;
}

void HcalUHTRData::const_iterator::determineMode() {
  if (!isHeader()) return;
  m_flavor=flavor();
  m_stepclass=0;
  if (m_flavor==5) { m_stepclass=1; m_microstep=0; }
  else if (m_flavor == 2) { m_stepclass=2; }
}

uint8_t HcalUHTRData::const_iterator::adc() const {
  if (m_flavor==0x5 && m_microstep==0) return ((*m_ptr)>>8)&0x7F;
  else return (*m_ptr)&0xFF;
}

uint8_t HcalUHTRData::const_iterator::le_tdc() const {
  if (m_flavor==0x5) return 0x80;
  else if (m_flavor == 2) return (m_ptr[1]&0x3F);
  else return (((*m_ptr)&0x3F00)>>8);
}

bool HcalUHTRData::const_iterator::soi() const {
  if (m_flavor==0x5) return false;
  else if (m_flavor == 2) return (m_ptr[0]&0x2000);
  else return (((*m_ptr)&0x4000));
}

uint8_t HcalUHTRData::const_iterator::te_tdc() const {
  if (m_flavor==2) return(m_ptr[1]>>6)&0x1F;
  else return 0x80;
}

uint8_t HcalUHTRData::const_iterator::capid() const {
  if (m_flavor==2) return(m_ptr[1]>>12)&0x3;
  else if (m_flavor == 1 || m_flavor == 0) {
    // For flavor 0,1 we only get the first capid in the header, and so we need
    // to count the number of data rows and figure out which cap we want,
    // knowing that they go 0->1->2->3->0
    return 0;
  }
  else { return 0; }
}

bool HcalUHTRData::const_iterator::ok() const {
  if (m_flavor == 2) { return (m_ptr[0]>>12)&0x1; }
  else if (m_flavor == 4) { return (m_ptr[0]>>13)&0x1; }
  else { return false; }
}

HcalUHTRData::const_iterator HcalUHTRData::begin() const {
  return HcalUHTRData::const_iterator(m_raw16+HEADER_LENGTH_16BIT,m_raw16+(m_rawLength64-1)*sizeof(uint64_t)/sizeof(uint16_t));
}

HcalUHTRData::const_iterator HcalUHTRData::end() const {
  return HcalUHTRData::const_iterator(m_raw16+(m_rawLength64-1)*sizeof(uint64_t)/sizeof(uint16_t),m_raw16+(m_rawLength64-1)*sizeof(uint64_t)/sizeof(uint16_t));
}

HcalUHTRData::packer::packer(uint16_t* baseptr) : m_baseptr(baseptr),m_ptr(0),m_flavor(0),m_ministep(0) {
}

void HcalUHTRData::packer::addHeader(int flavor, int errf, int cap0, int channelid) {
  m_baseptr[m_ptr]=0x8000 | 
    ((flavor&0x7)<<12) |
    ((errf&0x3)<<10) |
    ((cap0&0x3)<<8) | 
    (channelid&0xFF);
  m_flavor=flavor; 
  m_ministep=0;
  m_ptr++;
}

void HcalUHTRData::packer::addSample(int adc, bool soi, int retdc, int fetdc, int tdcstat) {
  if (m_flavor==0x5) {
    if (m_ministep==0) m_baseptr[m_ptr]=adc&0x7F;
    else m_baseptr[m_ptr]|=((adc&0x7f)<<8);
    if (m_ministep==1) m_ptr++;
    m_ministep=(m_ministep+1)%2;
  } else if ((m_flavor&0x6)==0x0) {
    m_baseptr[m_ptr]=(adc&0xFF) | ((retdc&0x3F)<<8);
    if (soi) m_baseptr[m_ptr]|=0x4000;
    m_ptr++;
  } else if ((m_flavor&0x6)==0x1) {
    m_baseptr[m_ptr]=(adc&0xFF);
    m_baseptr[m_ptr+1]=0x4000| ((tdcstat&0x3)<<10) | ((fetdc&0xF)<<6) | ((retdc)<<8);    
    if (soi) m_baseptr[m_ptr]|=0x2000;
    m_ptr+=2;
  }
}

void HcalUHTRData::packer::addTP(int tpword, bool soi) {
  if (m_flavor==0x4) {
    m_baseptr[m_ptr]=tpword&0xFFF;
    if (soi) m_baseptr[m_ptr]|=0x4000;
    m_ptr++;
  }
}

HcalUHTRData::HcalUHTRData() : m_formatVersion(-2), m_rawLength64(0), m_raw64(0), m_raw16(0), m_ownData(0) { }

HcalUHTRData::HcalUHTRData(const uint64_t* data, int length) : m_rawLength64(length),m_raw64(data),m_raw16((const uint16_t*)(data)),m_ownData(0) {
  m_formatVersion=(m_raw16[6]>>12)&0xF;
}

HcalUHTRData::HcalUHTRData(const HcalUHTRData& hd) : m_formatVersion(hd.m_formatVersion), m_rawLength64(hd.m_rawLength64), m_raw64(hd.m_raw64), m_raw16(hd.m_raw16), m_ownData(0) { }

HcalUHTRData::HcalUHTRData(int version_to_create) : m_formatVersion(version_to_create) {

  // the needed space is for the biggest possible event...
  // fibers*maxsamples/fiber
  const int needed=(0x20+FIBERS_PER_UHTR*CHANNELS_PER_FIBER_MAX*(10+1))*sizeof(uint16_t)/sizeof(uint64_t);

  m_ownData=new uint64_t[needed];
  memset(m_ownData,0,sizeof(uint64_t)*needed);
  m_rawLength64=0;
  m_raw64=m_ownData;
  m_raw16=(const uint16_t*)m_raw64;
}

HcalUHTRData& HcalUHTRData::operator=(const HcalUHTRData& hd) {
  if (m_ownData==0) {
    m_formatVersion=hd.m_formatVersion;
    m_rawLength64=hd.m_rawLength64;
    m_raw64=hd.m_raw64;
    m_raw16=hd.m_raw16;
  }
  return (*this);
}
