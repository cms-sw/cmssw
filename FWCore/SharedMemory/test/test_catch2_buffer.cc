#include "catch.hpp"
#include <algorithm>
#include <cstdio>
#include <string>

#include "FWCore/SharedMemory/interface/WriteBuffer.h"
#include "FWCore/SharedMemory/interface/ReadBuffer.h"

using namespace edm::shared_memory;

TEST_CASE("test Read/WriteBuffers", "[Buffers]") {
  BufferInfo bufferInfo = {0, 0};

  std::string const uniqueName = "BufferTest" + std::to_string(getpid());

  WriteBuffer writeBuffer(uniqueName, &bufferInfo);

  ReadBuffer readBuffer(uniqueName, &bufferInfo);

  SECTION("First") {
    std::array<char, 4> dummy = {{'t', 'e', 's', 't'}};

    writeBuffer.copyToBuffer(dummy.data(), dummy.size());

    REQUIRE(readBuffer.bufferIdentifier() == 1);
    {
      auto b = readBuffer.buffer();
      REQUIRE(b.second == dummy.size());
      REQUIRE(std::equal(b.first, b.first + b.second, dummy.cbegin(), dummy.cend()));
    }

    SECTION("Smaller") {
      dummy[0] = 'm';

      writeBuffer.copyToBuffer(dummy.data(), dummy.size() - 1);

      REQUIRE(readBuffer.bufferIdentifier() == 1);
      {
        auto b = readBuffer.buffer();
        //the second argument is the buffer capacity, not the last length sent
        REQUIRE(b.second == dummy.size());
        REQUIRE(std::equal(b.first, b.first + b.second - 1, dummy.cbegin(), dummy.cbegin() + dummy.size() - 1));
      }

      SECTION("Larger") {
        std::array<char, 6> dummy = {{'l', 'a', 'r', 'g', 'e', 'r'}};
        writeBuffer.copyToBuffer(dummy.data(), dummy.size());

        REQUIRE(readBuffer.bufferIdentifier() == 2);
        {
          auto b = readBuffer.buffer();
          REQUIRE(b.second == dummy.size());
          REQUIRE(std::equal(b.first, b.first + b.second, dummy.cbegin(), dummy.cend()));
        }

        SECTION("Largest") {
          //this should go back to buffer0
          std::array<char, 7> dummy = {{'l', 'a', 'r', 'g', 'e', 's', 't'}};
          writeBuffer.copyToBuffer(dummy.data(), dummy.size());

          REQUIRE(readBuffer.bufferIdentifier() == 3);
          {
            auto b = readBuffer.buffer();
            REQUIRE(b.second == dummy.size());
            REQUIRE(std::equal(b.first, b.first + b.second, dummy.cbegin(), dummy.cend()));
          }
        }
      }
    }
  }
}
