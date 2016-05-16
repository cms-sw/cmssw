#ifndef DATAFORMATS_HCALDIGI_HCALUMNIODIGI_H
#define DATAFORMATS_HCALDIGI_HCALUMNIODIGI_H 1

#include <vector>
#include <cstdint>
#include <ostream>

/** \class HcalUMNioDigi
  *  
  * This class contains the readout data from the uMNio uTCA card as
  * when used for orbit gap operations.
  *
  * \author J. Mans - Minnesota
  */
class HcalUMNioDigi {
public:

  HcalUMNioDigi();
  HcalUMNioDigi(const uint16_t* ptr, int words);
  HcalUMNioDigi(const std::vector<uint16_t>& words);
  
  uint32_t runNumber() const;
  uint32_t orbitNumber() const;
  uint16_t bunchNumber() const;
  uint32_t eventNumber() const;

  uint8_t eventType() const;
  uint16_t spillCounter() const;
  bool isSpill() const;

  bool invalid() const { return (payload_.size()<16) || (payload_[6]&0xF000)!=0x3000; }
  
  int numberUserWords() const;
  uint16_t idUserWord(int iword) const;
  uint32_t valueUserWord(int iword) const;
  bool hasUserWord(int id) const;
  uint32_t getUserWord(int id) const;
  bool getUserWord(int id, uint32_t& value) const;
  
private:
  std::vector<uint16_t> payload_;
};

std::ostream& operator<<(std::ostream&, const HcalUMNioDigi&);


#endif
