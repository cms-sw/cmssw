#ifndef DataFormats_L1Scouting_OrbitFlatTable_h
#define DataFormats_L1Scouting_OrbitFlatTable_h

/**
 * A cross-breed of a FlatTable and an OrbitCollection
 */

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include <cstdint>
#include <vector>
#include <string>
#include <type_traits>

namespace l1ScoutingRun3 {

  class OrbitFlatTable : public nanoaod::FlatTable {
  public:
    static constexpr unsigned int NBX = 3564;

    OrbitFlatTable() : nanoaod::FlatTable(), bxOffsets_(orbitBufferSize_ + 1, 0) {}

    OrbitFlatTable(std::vector<unsigned> bxOffsets,
                   const std::string &name,
                   bool singleton = false,
                   bool extension = false)
        : nanoaod::FlatTable(bxOffsets.back(), name, singleton, extension), bxOffsets_(bxOffsets) {
      if (bxOffsets.size() != orbitBufferSize_ + 1) {
        throw cms::Exception("LogicError") << "Mismatch between bxOffsets.size() " << bxOffsets.size()
                                           << " and orbitBufferSize_ + 1" << (orbitBufferSize_ + 1);
      }
    }

    ~OrbitFlatTable() {}

    using FlatTable::nRows;
    using FlatTable::size;

    /// number of rows for single BX
    unsigned int nRows(unsigned bx) const {
      if (bx >= orbitBufferSize_)
        throwBadBx(bx);
      return bxOffsets_[bx + 1] - bxOffsets_[bx];
    };
    unsigned int size(unsigned bx) const { return nRows(bx); }

    /// get a column by index (const)
    template <typename T>
    auto columnData(unsigned int column) const {
      return nanoaod::FlatTable::columnData<T>(column);
    }

    /// get a column by index and bx (const)
    template <typename T>
    auto columnData(unsigned int column, unsigned bx) const {
      if (bx >= orbitBufferSize_)
        throwBadBx(bx);
      auto begin = beginData<T>(column);
      return std::span(begin + bxOffsets_[bx], begin + bxOffsets_[bx + 1]);
    }

    /// get a column by index (non-const)
    template <typename T>
    auto columnData(unsigned int column) {
      return nanoaod::FlatTable::columnData<T>(column);
    }

    /// get a column by index and bx (non-const)
    template <typename T>
    auto columnData(unsigned int column, unsigned bx) {
      if (bx >= orbitBufferSize_)
        throwBadBx(bx);
      auto begin = beginData<T>(column);
      return std::span(begin + bxOffsets_[bx], begin + bxOffsets_[bx + 1]);
    }

    /// get a column value for singleton (const)
    template <typename T>
    const auto &columValue(unsigned int column, unsigned bx) const {
      if (!singleton())
        throw cms::Exception("LogicError", "columnValue works only for singleton tables");
      if (bx >= orbitBufferSize_ || bxOffsets_[bx + 1] == bxOffsets_[bx])
        throwBadBx(bx);
      auto begin = beginData<T>(column);
      return *(begin + bxOffsets_[bx]);
    }

  private:
    std::vector<unsigned> bxOffsets_;

    // there are 3564 BX in one orbtit [1,3564], one extra
    // count added to keep first entry of the vector
    static constexpr int orbitBufferSize_ = NBX + 1;

    [[noreturn]] void throwBadBx(unsigned bx) const {
      throw cms::Exception("OrbitFlatTable") << "Trying to access bad bx " << bx;
    }
  };

}  // namespace l1ScoutingRun3

#endif
