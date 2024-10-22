#ifndef L1GCTFIBREWORD_H
#define L1GCTFIBREWORD_H

#include <ostream>
#include <string>
#include <cstdint>

/*! \class L1GctFibreWord
 * \brief Global Calorimeter Trigger SC -> CC fibre data word
 *
 */

/*! \author Jim Brooke
 *  \date July 2007
 */

class L1GctFibreWord {
public:
  L1GctFibreWord();
  L1GctFibreWord(uint32_t data, uint16_t block, uint16_t index, int16_t bx);

  uint32_t data() const { return data_; }
  uint16_t block() const { return block_; }
  uint16_t index() const { return index_; }
  int16_t bx() const { return bx_; }

private:
  uint32_t data_;
  uint16_t block_;
  uint16_t index_;
  int16_t bx_;
};

std::ostream& operator<<(std::ostream& s, const L1GctFibreWord& fw);

#endif
