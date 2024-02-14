#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "catch.hpp"

#include <set>
#include <vector>
#include <algorithm>
#include <iostream>

TEST_CASE("SiStripHashedDetId testing", "[SiStripHashedDetId]") {
  //_____________________________________________________________
  SECTION("Check constructing SiStripHashedDetId from DetId list") {
    const auto& detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
    const auto& detIds = detInfo.getAllDetIds();
    SiStripHashedDetId hash(detIds);
    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " Successfully created hash!" << std::endl;
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check SiStripHashedDetId copy constructor") {
    const auto& detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
    const auto& dets = detInfo.getAllDetIds();

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " dets.size(): " << dets.size() << std::endl;

    SiStripHashedDetId hash(dets);

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " hash.size(): " << hash.size() << std::endl;

    // Retrieve hashed indices
    std::vector<uint32_t> hashes;
    hashes.clear();
    hashes.reserve(dets.size());
    for (const auto& idet : dets) {
      hashes.push_back(hash.hashedIndex(idet));
    }

    std::sort(hashes.begin(), hashes.end());

    SiStripHashedDetId hash2(hash);

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " Successfully copied hash map!" << std::endl;

    // Retrieve hashed indices
    std::vector<uint32_t> hashes2;

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " hashs2.size(): " << hash2.size() << std::endl;

    hashes2.clear();
    hashes2.reserve(dets.size());
    for (const auto& idet : dets) {
      hashes2.push_back(hash2.hashedIndex(idet));
    }

    std::sort(hashes2.begin(), hashes2.end());

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " Successfully sorted second hash map!" << std::endl;

    // Convert vectors to sets for easy set operations
    std::set<uint32_t> set1(hashes.begin(), hashes.end());
    std::set<uint32_t> set2(hashes2.begin(), hashes2.end());

    std::vector<uint32_t> diff1to2, diff2to1;

    // Find elements in vec1 that are not in vec2
    std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(diff1to2, diff1to2.begin()));

    // Find elements in vec2 that are not in vec1
    std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(), std::inserter(diff2to1, diff2to1.begin()));

    // Output the differences
    if (!diff1to2.empty()) {
      std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
                << " Elements in hash that are not in hash2: ";
      for (const auto& elem : diff1to2) {
        std::cout << elem << " ";
      }
      std::cout << std::endl;
    }

    if (!diff2to1.empty()) {
      std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
                << " Elements in hash2 that are not in hash: ";
      for (const auto& elem : diff2to1) {
        std::cout << elem << " ";
      }
      std::cout << std::endl;
    }

    REQUIRE(hashes == hashes2);
  }

  //_____________________________________________________________
  SECTION("Check SiStripHashedDetId assignment operator") {
    const auto& detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
    const auto& dets = detInfo.getAllDetIds();

    SiStripHashedDetId hash(dets);
    SiStripHashedDetId hash2;

    // Retrieve hashed indices
    std::vector<uint32_t> hashes;
    hashes.clear();
    hashes.reserve(dets.size());
    for (const auto& idet : dets) {
      hashes.push_back(hash.hashedIndex(idet));
    }

    std::sort(hashes.begin(), hashes.end());

    // assign hash to hash2
    hash2 = hash;

    // Retrieve hashed indices
    std::vector<uint32_t> hashes2;
    hashes2.clear();
    hashes2.reserve(dets.size());
    for (const auto& idet : dets) {
      hashes2.push_back(hash2.hashedIndex(idet));
    }

    std::sort(hashes2.begin(), hashes2.end());

    if (hashes == hashes2) {
      std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
                << " Assigned SiStripHashedDetId matches original one!" << std::endl;
    } else {
      std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
                << " Assigned SiStripHashedDetId does not match the original one!" << std::endl;
    }

    REQUIRE(hashes == hashes2);
  }

  //_____________________________________________________________
  SECTION("Check manipulating SiStripHashedDetId") {
    const auto& detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());

    const auto& unsortedDets = detInfo.getAllDetIds();

    // unfortunately SiStripDetInfo::getAllDetIds() returns a const vector
    // so in order to sory we're gonna need to copy it first

    std::vector<uint32_t> dets;
    dets.reserve(unsortedDets.size());
    std::copy(unsortedDets.begin(), unsortedDets.end(), std::back_inserter(dets));

    // sort the vector of detIds (otherwise the test won't work!)
    std::sort(dets.begin(), dets.end());

    SiStripHashedDetId hash(dets);

    // Retrieve hashed indices
    std::vector<uint32_t> hashes;
    uint32_t istart = time(NULL);
    hashes.clear();
    hashes.reserve(dets.size());
    for (const auto& idet : dets) {
      hashes.push_back(hash.hashedIndex(idet));
    }

    // Some debug
    std::stringstream ss;
    ss << "[testSiStripHashedDetId::" << __func__ << "]";
    uint16_t cntr1 = 0;
    for (const auto& ii : hashes) {
      if (ii == sistrip::invalid32_) {
        cntr1++;
        ss << std::endl << " Invalid index " << ii;
        continue;
      }
      uint32_t detid = hash.unhashIndex(ii);
      std::vector<uint32_t>::const_iterator iter = find(dets.begin(), dets.end(), detid);
      if (iter == dets.end()) {
        cntr1++;
        ss << std::endl << " Did not find value " << detid << " at index " << ii - *(hashes.begin()) << " in vector!";
      } else if (ii != static_cast<uint32_t>(iter - dets.begin())) {
        cntr1++;
        ss << std::endl
           << " Found same value " << detid << " at different indices " << ii << " and " << iter - dets.begin();
      }
    }

    if (cntr1) {
      ss << std::endl << " Found " << cntr1 << " incompatible values!";
    } else {
      ss << " Found no incompatible values!";
    }
    std::cout << ss.str() << std::endl;

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " Processed " << hashes.size() << " DetIds in " << (time(NULL) - istart) << " seconds" << std::endl;

    REQUIRE(cntr1 == 0);

    // Retrieve DetIds
    std::vector<uint32_t> detids;
    uint32_t jstart = time(NULL);
    // meaasurement!
    detids.clear();
    detids.reserve(dets.size());
    for (uint16_t idet = 0; idet < dets.size(); ++idet) {
      detids.push_back(hash.unhashIndex(idet));
    }

    // Some debug
    std::stringstream sss;
    sss << "[testSiStripHashedDetId::" << __func__ << "]";
    uint16_t cntr2 = 0;
    std::vector<uint32_t>::const_iterator iii = detids.begin();
    for (; iii != detids.end(); ++iii) {
      if (*iii != dets.at(iii - detids.begin())) {
        cntr2++;
        sss << std::endl
            << " Diff values " << *iii << " and " << dets.at(iii - detids.begin()) << " found at index "
            << iii - detids.begin() << " ";
      }
    }
    if (cntr2) {
      sss << std::endl << " Found " << cntr2 << " incompatible values!";
    } else {
      sss << " Found no incompatible values!";
    }
    std::cout << sss.str() << std::endl;

    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
              << " Processed " << detids.size() << " hashed indices in " << (time(NULL) - jstart) << " seconds"
              << std::endl;

    REQUIRE(cntr2 == 0);

    REQUIRE(true);
  }
}
