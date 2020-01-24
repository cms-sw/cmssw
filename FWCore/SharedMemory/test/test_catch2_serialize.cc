#include "catch.hpp"
#include <utility>
#include <algorithm>
#include <iterator>

#include "FWCore/SharedMemory/interface/ROOTSerializer.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"

#include "DataFormats/TestObjects/interface/Thing.h"

namespace {
  struct ReadWriteTestBuffer {
    std::pair<char*, std::size_t> buffer() { return std::pair(&buffer_.front(), size()); }

    bool mustGetBufferAgain() { return resized_; }

    void copyToBuffer(char* iStart, std::size_t iLength) {
      buffer_.clear();
      if (iLength > buffer_.capacity()) {
        buffer_.reserve(iLength);
        resized_ = true;
      } else {
        resized_ = false;
      }
      std::copy(iStart, iStart + iLength, std::back_insert_iterator(buffer_));
    }

    std::size_t size() const { return buffer_.size(); }

    std::vector<char> buffer_;
    bool resized_ = true;
  };

  bool compare(std::vector<edmtest::Thing> const& iLHS, std::vector<edmtest::Thing> const& iRHS) {
    if (iLHS.size() != iRHS.size()) {
      return false;
    }

    for (size_t i = 0; i < iLHS.size(); ++i) {
      if (iLHS[i].a != iRHS[i].a) {
        return false;
      }
    }
    return true;
  }
}  // namespace
using namespace edm::shared_memory;
TEST_CASE("test De/ROOTSerializer", "[ROOTSerializer]") {
  SECTION("Process edmtest::Thing") {
    ReadWriteTestBuffer buffer;

    ROOTSerializer<edmtest::Thing, ReadWriteTestBuffer> serializer(buffer);
    ROOTDeserializer<edmtest::Thing, ReadWriteTestBuffer> deserializer(buffer);

    edmtest::Thing t;
    t.a = 42;

    serializer.serialize(t);
    REQUIRE(buffer.mustGetBufferAgain() == true);

    auto newT = deserializer.deserialize();

    REQUIRE(t.a == newT.a);
    SECTION("Reuse buffer") {
      t.a = 12;
      serializer.serialize(t);
      REQUIRE(buffer.mustGetBufferAgain() == false);

      auto newT = deserializer.deserialize();

      REQUIRE(t.a == newT.a);
    }
  }

  SECTION("Process std::vector<edmtest::Thing>") {
    ReadWriteTestBuffer buffer;

    ROOTSerializer<std::vector<edmtest::Thing>, ReadWriteTestBuffer> serializer(buffer);
    ROOTDeserializer<std::vector<edmtest::Thing>, ReadWriteTestBuffer> deserializer(buffer);

    std::vector<edmtest::Thing> t;
    t.reserve(4);
    for (int i = 0; i < 4; ++i) {
      edmtest::Thing temp;
      temp.a = i;
      t.push_back(temp);
    }

    serializer.serialize(t);
    REQUIRE(buffer.mustGetBufferAgain() == true);

    auto newT = deserializer.deserialize();

    REQUIRE(compare(t, newT));
    SECTION("Reuse edtest::Thing buffer") {
      for (auto& v : t) {
        v.a += 1;
      }
      serializer.serialize(t);
      REQUIRE(buffer.mustGetBufferAgain() == false);

      auto newT = deserializer.deserialize();

      REQUIRE(compare(t, newT));
    }

    SECTION("Grow") {
      t.emplace_back();
      serializer.serialize(t);
      REQUIRE(buffer.mustGetBufferAgain() == true);

      auto newT = deserializer.deserialize();

      REQUIRE(compare(t, newT));
    }

    SECTION("Shrink") {
      t.pop_back();
      t.pop_back();

      serializer.serialize(t);
      REQUIRE(buffer.mustGetBufferAgain() == false);

      auto newT = deserializer.deserialize();

      REQUIRE(compare(t, newT));
    }
  }
}
