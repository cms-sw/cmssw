
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctFibreWord.h"

L1GctFibreWord::L1GctFibreWord() : data_(0), block_(0), index_(0), bx_(0) {}

L1GctFibreWord::L1GctFibreWord(uint32_t data, uint16_t block, uint16_t index, int16_t bx)
    : data_(data), block_(block), index_(index), bx_(bx) {}

std::ostream& operator<<(std::ostream& s, const L1GctFibreWord& fw) {
  s << "L1GctFibreWord : ";
  s << std::hex << fw.data() << ", block " << fw.block();
  s << std::dec << ", index : " << fw.index() << ", bx : " << fw.bx();
  return s;
}
