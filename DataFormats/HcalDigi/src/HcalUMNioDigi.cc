#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"

HcalUMNioDigi::HcalUMNioDigi() { }
HcalUMNioDigi::HcalUMNioDigi(const uint16_t* ptr, int words) {
  payload_.reserve(words);
  for (int i=0; i<words; i++) payload_.push_back(ptr[i]);
}
HcalUMNioDigi::HcalUMNioDigi(const std::vector<uint16_t>& words) : payload_(words) {
}
  
uint32_t HcalUMNioDigi::runNumber() const {
  if (invalid()) return 0;
  return payload_[10]+(uint32_t(payload_[11])<<16);
}

uint32_t HcalUMNioDigi::orbitNumber() const {
  if (invalid()) return 0;
  return payload_[5]+(uint32_t(payload_[8])<<16);
}
uint16_t HcalUMNioDigi::bunchNumber() const {
  if (invalid()) return 0;
  return (payload_[1]>>4)&0xFFF;
}
uint32_t HcalUMNioDigi::eventNumber() const {
  if (invalid()) return 0;
  return payload_[2]+(uint32_t(payload_[3]&0xFF)<<16);
}

uint8_t HcalUMNioDigi::eventType() const {
  if (invalid()) return 0;
  return (payload_[6]>>8)&0xF;
}
uint16_t HcalUMNioDigi::spillCounter() const {
  if (invalid()) return 0;
  return (payload_[9])&0x7FFF;
}
bool HcalUMNioDigi::isSpill() const {
  if (invalid()) return 0;
  return (payload_[9]&0x8000);
}

int HcalUMNioDigi::numberUserWords() const {
  if (invalid()) return 0;
  return (payload_[12]&0xFF);
}

uint16_t HcalUMNioDigi::idUserWord(int iword) const {
  if (iword>=numberUserWords() || payload_.size()<(size_t)(16+iword*3)) return 0;
  return payload_[13+3*iword];
}
uint32_t HcalUMNioDigi::valueUserWord(int iword) const {
  if (iword>=numberUserWords() || payload_.size()<size_t(16+iword*3)) return 0;
  return payload_[14+3*iword]+(uint32_t(payload_[15+3*iword])<<16);
}
bool HcalUMNioDigi::hasUserWord(int id) const {
  uint32_t dummy;
  return getUserWord(id,dummy);
}
uint32_t HcalUMNioDigi::getUserWord(int id) const {
  uint32_t dummy(0);
  getUserWord(id,dummy);
  return dummy;
}
bool HcalUMNioDigi::getUserWord(int id, uint32_t& value) const {
  int nwords=numberUserWords();
  if (size_t(16+nwords*3)>payload_.size()) return false; // invalid format...
  
  for (int i=0; i<nwords; i++) {
    if (payload_[14+3*i]==id) {
      value=payload_[15+3*i]+(uint32_t(payload_[16+3*i])<<16);
      return true;
    }
  }
  return false;
}


std::ostream& operator<<(std::ostream& s, const HcalUMNioDigi& digi) {
  s << "HcalUMNioDigi orbit/bunch " << digi.orbitNumber() << "/" << digi.bunchNumber() << std::endl;
  s << " run/l1a " << digi.runNumber() << "/" << digi.eventNumber() << std::endl;
  s << " eventType " << digi.eventType() << std::endl;
  for (int i=0; i<digi.numberUserWords(); i++)
    s << "   id=" << digi.idUserWord(i) << " value="<< digi.valueUserWord(i) << std::endl;
  return s;
}
