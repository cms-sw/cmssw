
#include <algorithm>

#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <vector>

TEST_CASE("ContainerMask", "[ContainerMask]") {
  SECTION("test ContainerMask with vector") {
    const unsigned int values[] = {0, 1, 2, 3};
    const bool flags[] = {true, false, true, false};
    const unsigned int kSize = 4;
    std::vector<unsigned int> c(values, values + kSize);
    std::vector<bool> mask(flags, flags + kSize);
    REQUIRE(mask.size() == c.size());
    REQUIRE(mask.size() == kSize);

    typedef std::vector<unsigned int> ContainerType;
    edm::TestHandle<ContainerType> h(&c, edm::ProductID(1, 1));

    const edm::RefProd<ContainerType> rp(h);

    edm::ContainerMask<ContainerType> cMask(rp, mask);

    REQUIRE(cMask.size() == mask.size());

    unsigned int index = 0;
    for (std::vector<bool>::iterator it = mask.begin(), itEnd = mask.end(); it != itEnd; ++it, ++index) {
      REQUIRE(*it == cMask.mask(index));
    }

    index = 0;
    for (std::vector<unsigned int>::iterator it = c.begin(), itEnd = c.end(); it != itEnd; ++it, ++index) {
      REQUIRE(mask[index] == cMask.mask(&(*it)));
    }

    {
      std::vector<bool> alternate(4, false);
      alternate[1] = true;
      cMask.applyOrTo(alternate);
      REQUIRE(alternate[0]);
      REQUIRE(alternate[1]);
      REQUIRE(alternate[2]);
      REQUIRE(!alternate[3]);
    }
    {
      std::vector<bool> alternate(3, false);
      cMask.copyMaskTo(alternate);
      REQUIRE(alternate[0]);
      REQUIRE(!alternate[1]);
      REQUIRE(alternate[2]);
      REQUIRE(!alternate[3]);
    }
  }

  SECTION("ContainerMask with DetSetVector") {
    const bool flags[] = {true, false, true, false};
    const unsigned int kSize = 4;
    edmNew::DetSetVector<unsigned int> c;
    {
      edmNew::DetSetVector<unsigned int>::FastFiller filler(c, 0);
      filler.push_back(0);
      filler.push_back(1);
    }
    {
      edmNew::DetSetVector<unsigned int>::FastFiller filler(c, 1);
      filler.push_back(2);
      filler.push_back(3);
    }
    std::vector<bool> mask(flags, flags + kSize);
    REQUIRE(mask.size() == c.dataSize());
    REQUIRE(mask.size() == kSize);

    typedef edmNew::DetSetVector<unsigned int> ContainerType;

    edm::TestHandle<ContainerType> h(&c, edm::ProductID(1, 1));

    const edm::RefProd<ContainerType> rp(h);

    edm::ContainerMask<ContainerType> cMask(rp, mask);

    REQUIRE(cMask.size() == mask.size());

    unsigned int index = 0;
    for (std::vector<bool>::iterator it = mask.begin(), itEnd = mask.end(); it != itEnd; ++it, ++index) {
      REQUIRE(*it == cMask.mask(index));
    }

    index = 0;
    for (std::vector<unsigned int>::const_iterator it = c.data().begin(), itEnd = c.data().end(); it != itEnd;
         ++it, ++index) {
      REQUIRE(mask[index] == cMask.mask(&(*it)));
    }

    std::vector<bool> alternate(4, false);
    alternate[1] = true;
    cMask.applyOrTo(alternate);
    REQUIRE(alternate[0]);
    REQUIRE(alternate[1]);
    REQUIRE(alternate[2]);
    REQUIRE(!alternate[3]);
  }
}
