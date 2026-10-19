#ifndef LpGbtData_h
#define LpGbtData_h

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>

class LpGbtData {
public:
  enum { NumberOfElinks = 7 };

  LpGbtData() { setZero(); }

  void setZero() {
    std::memset(_data, 0, sizeof(uint32_t) * NumberOfElinks);
    _data[NumberOfElinks] = 0xdeadbeef;
  }

  uint32_t getData(unsigned n) const { return (n < NumberOfElinks ? _data[n] : 0xffffffff); }

  const uint32_t* getData() const { return _data; }

  void setData(unsigned n, uint32_t d) {
    if (n < NumberOfElinks)
      _data[n] = d;
  }

  uint32_t* setData() { return _data; }

  void getElinks(std::vector<uint32_t>& e, uint8_t b) const {
    assert(b < 0x80);  // Catch invalid bit

    for (unsigned i(0); i < NumberOfElinks; i++) {
      if ((b & (1 << i)) != 0)
        e.push_back(_data[i]);
    }
  }

  void setElinks(const std::vector<uint32_t>& e, uint8_t b) {
    unsigned nElink(0);

    for (unsigned i(0); i < NumberOfElinks; i++) {
      if ((b & (1 << i)) != 0) {
        if (nElink < e.size())
          _data[i] = e[nElink++];
        else
          assert(false);  // Catch more bits than values
      }
    }

    assert(nElink == e.size());  // Catch more values than bits
  }

  void print() {
    std::cout << "LpGbtData(" << this << ")::print()" << std::endl;

    for (unsigned i(0); i <= NumberOfElinks; i++) {
      std::cout << " Elink " << i << " = 0x" << std::hex << ::std::setfill('0') << std::setw(8) << _data[i] << std::dec
                << ::std::setfill(' ') << std::endl;
    }
  }

private:
  uint32_t _data[NumberOfElinks + 1];
};

#endif
