#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "catch.hpp"
#include <iostream>

TEST_CASE("SiStripHashedDetId testing", "[SiStripHashedDetId]") {
  //_____________________________________________________________
  SECTION("Check constructing SiStripHashedDetId from DetId list") {
    const auto& detInfo = SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
    const auto& detIds = detInfo.getAllDetIds();
    SiStripHashedDetId hash(detIds);
    std::cout << "Successfully created hash!" << std::endl;
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check manipulating SiStripHashedDetId") {
    const auto& detInfo = SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
    const auto& dets = detInfo.getAllDetIds();
    SiStripHashedDetId hash(dets);

    // Retrieve hashed indices
    std::vector<uint32_t> hashes;
    uint32_t istart = time(NULL);
    hashes.clear();
    hashes.reserve(dets.size());
    std::vector<uint32_t>::const_iterator idet = dets.begin();
    for (; idet != dets.end(); ++idet) {
      hashes.push_back(hash.hashedIndex(*idet));
    }
    
    // Some debug
    std::stringstream ss;
    ss << "[testSiStripHashedDetId::" << __func__ << "]";
    std::vector<uint32_t>::const_iterator ii = hashes.begin();
    uint16_t cntr1 = 0;
    for (; ii != hashes.end(); ++ii) {
      if (*ii == sistrip::invalid32_) {
	cntr1++;
	ss << std::endl << " Invalid index " << *ii;
	continue;
      }
      uint32_t detid = hash.unhashIndex(*ii);
      std::vector<uint32_t>::const_iterator iter = find(dets.begin(), dets.end(), detid);
      if (iter == dets.end()) {
	cntr1++;
	ss << std::endl << " Did not find value " << detid << " at index " << ii - hashes.begin() << " in vector!";
      } else if (*ii != static_cast<uint32_t>(iter - dets.begin())) {
	cntr1++;
	ss << std::endl
	   << " Found same value " << detid << " at different indices " << *ii << " and " << iter - dets.begin();
      }
    }
    if (cntr1) {
      ss << std::endl << " Found " << cntr1 << " incompatible values!";
    } else {
      ss << " Found no incompatible values!";
    }
    std::cout << ss.str() << std::endl;
    
    std::cout << "[testSiStripHashedDetId::" << __func__ << "]"
	      << " Processed " << hashes.size() << " DetIds in " << (time(NULL) - istart)
	      << " seconds" << std::endl;
    
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
	      << " Processed " << detids.size() << " hashed indices in " << (time(NULL) - jstart)
	      << " seconds" << std::endl;
    
    REQUIRE(true);
  }
  
  
  
}
