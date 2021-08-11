/* \file testCSCMapping.cc
 *
 * \author Tim Cox
 * Based on template from S. Argiro & N. Amapane
 */

#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include <iostream>
#include <cstdlib>

std::string releasetop(getenv("CMSSW_BASE"));
//std::string mappingFilePath= releasetop + "/src/CondFormats/CSCObjects/test/";

class testCSCMapping : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCSCMapping);

  CPPUNIT_TEST(testRead);

  CPPUNIT_TEST_SUITE_END();

public:
  testCSCMapping() : myName_("testCSCMapping"), dashedLineWidth(104), dashedLine(std::string(dashedLineWidth, '-')) {}

  void setUp() {
    char* ret = getenv("CMSSW_BASE");
    if (!ret) {
      std::cerr << "env variable SCRAMRT_LOCALRT not set, try eval `scramv1 runt -sh`" << std::endl;
      exit(1);
    } else {
      std::cout << "CMSSW_BASE set to " << ret << std::endl;
    }

    ret = getenv("CMSSW_SEARCH_PATH");
    if (!ret) {
      std::cerr << "env variable SCRAMRT_LOCALRT not set, try eval `scramv1 runt -sh`" << std::endl;
      exit(1);
    } else {
      std::cout << "CMSSW_SEARCH_PATH set to " << ret << std::endl;
    }
  }

  void tearDown() {}

  void testRead();

private:
  const std::string myName_;
  const int dashedLineWidth;
  std::string dashedLine;
};

void testCSCMapping::testRead() {
  edm::FileInPath fip("CondFormats/CSCObjects/data/csc_slice_test_map.txt");
  std::cout << "Attempt to set FileInPath to " << fip.fullPath() << std::endl;

  std::cout << myName_ << ": --- t e s t C S C M a p p i n g  ---" << std::endl;
  std::cout << "start " << dashedLine << std::endl;

  CSCReadoutMappingFromFile theMapping(fip.fullPath());

  // The following labels are irrelevant to hardware in slice test
  int tmb = -1;
  int iendcap = -1;
  int istation = -1;

  // Loop over all possible crates and dmb slots in slice test
  // TEST CSCReadoutMapping::chamber(...)

  int numberOfVmeCrates = 4;
  int numberOfDmbSlots = 10;
  int missingDmbSlot = 6;

  for (int i = 0; i < numberOfVmeCrates; ++i) {
    int vmecrate = i;
    for (int j = 1; j <= numberOfDmbSlots; ++j) {
      if (j == missingDmbSlot)
        continue;  // ***There is no slot 6***
      int dmb = j;

      std::cout << "\n"
                << myName_ << ": search for sw id for hw labels, endcap= " << iendcap << ", station=" << istation
                << ", vmecrate=" << vmecrate << ", dmb=" << dmb << ", tmb= " << tmb << std::endl;
      int id = theMapping.chamber(iendcap, istation, vmecrate, dmb, tmb);

      std::cout << myName_ << ": found chamber rawId = " << id << std::endl;

      CSCDetId cid(id);

      // We can now find real endcap & station labels
      int endcap = cid.endcap();
      int station = cid.station();

      std::cout << myName_ << ": from CSCDetId for this chamber, endcap= " << cid.endcap()
                << ", station=" << cid.station() << ", ring=" << cid.ring() << ", chamber=" << cid.chamber()
                << std::endl;

      // Now try direct mapping for specific layers & cfebs (now MUST have correct endcap & station labels!)
      // TEST CSCReadoutMapping::detId(...)
      for (int cfeb = 0; cfeb != 5; ++cfeb) {
        for (int layer = 1; layer <= 6; ++layer) {
          std::cout << myName_ << ": map layer with hw labels, endcap= " << endcap << ", station=" << station
                    << ", vmecrate=" << vmecrate << ", dmb=" << dmb << ", tmb= " << tmb << ", cfeb= " << cfeb
                    << ", layer=" << layer << std::endl;

          CSCDetId lid = theMapping.detId(endcap, station, vmecrate, dmb, tmb, cfeb, layer);

          // And check what we've actually selected...
          std::cout << myName_ << ": from CSCDetId for this layer, endcap= " << lid.endcap()
                    << ", station=" << lid.station() << ", ring=" << lid.ring() << ", chamber=" << lid.chamber()
                    << ", layer=" << lid.layer() << std::endl;
        }

        std::cout << std::endl;

      }  // end loop over cfebs
    }    // end loop over dmbs
  }      // end loop over vmes
  std::cout << dashedLine << " end" << std::endl;
}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCMapping);
