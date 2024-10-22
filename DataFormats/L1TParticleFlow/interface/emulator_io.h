#ifndef FIRMWARE_utils_emulator_io_h
#define FIRMWARE_utils_emulator_io_h

#include <fstream>
#include <vector>
#include "DataFormats/L1TParticleFlow/interface/datatypes.h"

namespace l1ct {

  template <typename T>
  inline bool writeVar(const T &src, std::fstream &to) {
    to.write(reinterpret_cast<const char *>(&src), sizeof(T));
    return to.good();
  }

  template <typename T>
  inline bool readVar(std::fstream &from, T &to) {
    from.read(reinterpret_cast<char *>(&to), sizeof(T));
    return from.good();
  }

  template <typename T>
  bool writeAP(const T &src, std::fstream &to) {
    for (unsigned int i = 0, n = T::width; i < n; i += 32) {
      ap_uint<32> word = src(std::min(i + 31, n - 1), i);
      uint32_t w32 = word.to_uint();
      if (!writeVar(w32, to))
        return false;
    }
    return true;
  }

  template <typename T>
  bool readAP(std::fstream &from, T &to) {
    uint32_t w32;
    for (unsigned int i = 0, n = T::width; i < n; i += 32) {
      if (!readVar(from, w32))
        return false;
      ap_uint<32> word = w32;
      to(std::min(i + 31, n - 1), i) = word(std::min(31u, n - i - 1), 0);
    }
    return true;
  }

  template <typename T>
  bool writeObj(const T &obj, std::fstream &to) {
    return writeAP(obj.pack(), to);
  }

  template <typename T>
  bool readObj(std::fstream &from, T &obj) {
    ap_uint<T::BITWIDTH> packed;
    if (!readAP(from, packed))
      return false;
    obj = T::unpack(packed);
    return true;
  }

  template <typename T>
  bool writeMany(const std::vector<T> &objs, std::fstream &to) {
    uint32_t number = objs.size();
    writeVar(number, to);
    for (uint32_t i = 0; i < number; ++i) {
      objs[i].write(to);
    }
    return to.good();
  }

  template <int NB>
  bool writeMany(const std::vector<ap_uint<NB>> &objs, std::fstream &to) {
    uint32_t number = objs.size();
    writeVar(number, to);
    for (uint32_t i = 0; i < number; ++i) {
      writeAP(objs[i], to);
    }
    return to.good();
  }

  template <typename T>
  bool readMany(std::fstream &from, std::vector<T> &objs) {
    uint32_t number = 0;
    readVar(from, number);
    objs.resize(number);
    for (uint32_t i = 0; i < number; ++i)
      objs[i].read(from);
    return from.good();
  }

  template <int NB>
  bool readMany(std::fstream &from, std::vector<ap_uint<NB>> &objs) {
    uint32_t number = 0;
    readVar(from, number);
    objs.resize(number);
    for (uint32_t i = 0; i < number; ++i)
      readAP(from, objs[i]);
    return from.good();
  }

}  // namespace l1ct

#endif
