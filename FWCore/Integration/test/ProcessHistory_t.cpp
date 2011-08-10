#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <cassert>
#include <string>
#include <iostream>

bool checkRunOrLumiEntry(edm::IndexIntoFile::RunOrLumiEntry const& rl,
                         edm::IndexIntoFile::EntryNumber_t orderPHIDRun,
                         edm::IndexIntoFile::EntryNumber_t orderPHIDRunLumi,
                         edm::IndexIntoFile::EntryNumber_t entry,
                         int processHistoryIDIndex,
                         edm::RunNumber_t run,
                         edm::LuminosityBlockNumber_t lumi,
                         edm::IndexIntoFile::EntryNumber_t beginEvents,
                         edm::IndexIntoFile::EntryNumber_t endEvents) {
  if (rl.orderPHIDRun() != orderPHIDRun) return false;
  if (rl.orderPHIDRunLumi() != orderPHIDRunLumi) return false;
  if (rl.entry() != entry) return false;
  if (rl.processHistoryIDIndex() != processHistoryIDIndex) return false;
  if (rl.run() != run) return false;
  if (rl.lumi() != lumi) return false;
  if (rl.beginEvents() != beginEvents) return false;
  if (rl.endEvents() != endEvents) return false;
  return true;
}

int main()
{
  edm::ParameterSet dummyPset;
  dummyPset.registerIt();
  edm::ParameterSetID psetID = dummyPset.id();

  edm::ProcessHistory pnl1;
  assert(pnl1 == pnl1);
  edm::ProcessHistory pnl2;
  assert(pnl1 == pnl2);
  edm::ProcessConfiguration iHLT(std::string("HLT"), psetID, std::string("CMSSW_5_100_40"), edm::getPassID());
  edm::ProcessConfiguration iRECO(std::string("RECO"), psetID, std::string("5_100_42patch100"), edm::getPassID());
  pnl2.push_back(iHLT);
  assert(pnl1 != pnl2);
  edm::ProcessHistory pnl3;
  pnl3.push_back(iHLT);
  pnl3.push_back(iRECO);

  edm::ProcessHistoryID id1 = pnl1.id();
  edm::ProcessHistoryID id2 = pnl2.id();
  edm::ProcessHistoryID id3 = pnl3.id();

  assert(id1 != id2);
  assert(id2 != id3);
  assert(id3 != id1);

  edm::ProcessHistory pnl4;
  pnl4.push_back(iHLT);
  edm::ProcessHistoryID id4 = pnl4.id();
  assert(pnl4 == pnl2);
  assert (id4 == id2);

  edm::ProcessHistory pnl5;
  pnl5 = pnl3;
  assert(pnl5 == pnl3);
  assert(pnl5.id() == pnl3.id());


  edm::ProcessConfiguration pc1(std::string("HLT"), psetID, std::string(""), edm::getPassID());
  edm::ProcessConfiguration pc2(std::string("HLT"), psetID, std::string("a"), edm::getPassID());
  edm::ProcessConfiguration pc3(std::string("HLT"), psetID, std::string("1"), edm::getPassID());
  edm::ProcessConfiguration pc4(std::string("HLT"), psetID, std::string("ccc500yz"), edm::getPassID());
  edm::ProcessConfiguration pc5(std::string("HLT"), psetID, std::string("500yz872"), edm::getPassID());
  edm::ProcessConfiguration pc6(std::string("HLT"), psetID, std::string("500yz872djk999patch10"), edm::getPassID());
  edm::ProcessConfiguration pc7(std::string("HLT"), psetID, std::string("xb500yz872djk999patch10"), edm::getPassID());
  edm::ProcessConfiguration pc8(std::string("HLT"), psetID, std::string("CMSSW_4_4_0_pre5"), edm::getPassID());

  pc1.id();
  pc2.id();
  pc3.id();
  pc4.id();
  pc5.id();
  pc6.id();
  pc7.id();
  pc8.id();

  pc1.reduce();
  pc2.reduce();
  pc3.reduce();
  pc4.reduce();
  pc5.reduce();
  pc6.reduce();
  pc7.reduce();
  pc8.reduce();

  edm::ProcessConfiguration pc1expected(std::string("HLT"), psetID, std::string(""), edm::getPassID());
  edm::ProcessConfiguration pc2expected(std::string("HLT"), psetID, std::string("a"), edm::getPassID());
  edm::ProcessConfiguration pc3expected(std::string("HLT"), psetID, std::string("1"), edm::getPassID());
  edm::ProcessConfiguration pc4expected(std::string("HLT"), psetID, std::string("ccc500yz"), edm::getPassID());
  edm::ProcessConfiguration pc5expected(std::string("HLT"), psetID, std::string("500yz872"), edm::getPassID());
  edm::ProcessConfiguration pc6expected(std::string("HLT"), psetID, std::string("500yz872"), edm::getPassID());
  edm::ProcessConfiguration pc7expected(std::string("HLT"), psetID, std::string("xb500yz872"), edm::getPassID());
  edm::ProcessConfiguration pc8expected(std::string("HLT"), psetID, std::string("CMSSW_4_4"), edm::getPassID());

  assert(pc1 == pc1expected);
  assert(pc2 == pc2expected);
  assert(pc3 == pc3expected);
  assert(pc4 == pc4expected);
  assert(pc5 == pc5expected);
  assert(pc6 == pc6expected);
  assert(pc7 == pc7expected);
  assert(pc8 == pc8expected);

  assert(pc1.id() == pc1expected.id());
  assert(pc2.id() == pc2expected.id());
  assert(pc3.id() == pc3expected.id());
  assert(pc4.id() == pc4expected.id());
  assert(pc5.id() == pc5expected.id());
  assert(pc6.id() == pc6expected.id());
  assert(pc7.id() == pc7expected.id());
  assert(pc8.id() == pc8expected.id());

  assert(pc7.id() != pc8expected.id());

  edm::ProcessConfiguration iHLTreduced(std::string("HLT"), psetID, std::string("CMSSW_5_100"), edm::getPassID());
  edm::ProcessConfiguration iRECOreduced(std::string("RECO"), psetID, std::string("5_100"), edm::getPassID());
  edm::ProcessHistory phTestExpected;
  phTestExpected.push_back(iHLTreduced);
  phTestExpected.push_back(iRECOreduced);

  edm::ProcessHistory phTest = pnl3;
  phTest.id();
  phTest.reduce();
  assert(phTest == phTestExpected);
  assert(phTest.id() == phTestExpected.id());
  assert(phTest.id() != pnl3.id());

  edm::ProcessHistoryRegistry::instance()->insertMapped(pnl3);
  edm::ProcessHistoryID reducedPHID = edm::ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(pnl3.id());
  assert(reducedPHID == phTest.id());

  // Repeat a few times to test the caching optimization in FullHistoryToReducedHistoryMap
  // (You have to watch in a debugger to really verify it is working properly)
  reducedPHID = edm::ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(pnl3.id());
  assert(reducedPHID == phTest.id());

  edm::ProcessHistoryRegistry::instance()->insertMapped(pnl2);
  reducedPHID = edm::ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(pnl2.id());
  pnl2.reduce();
  assert(reducedPHID == pnl2.id());

  reducedPHID = edm::ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(pnl3.id());
  assert(reducedPHID == phTest.id());

  {
    edm::ProcessHistory ph1;
    edm::ProcessHistory ph1a;
    edm::ProcessHistory ph1b;
    edm::ProcessHistory ph2;
    edm::ProcessHistory ph2a;
    edm::ProcessHistory ph2b;
    edm::ProcessHistory ph3;
    edm::ProcessHistory ph4;

    edm::ParameterSet dummyPset;
    dummyPset.registerIt();
    edm::ParameterSetID psetID = dummyPset.id();

    edm::ProcessConfiguration pc1(std::string("HLT"), psetID, std::string("CMSSW_5_1_40"), std::string(""));
    edm::ProcessConfiguration pc1a(std::string("HLT"), psetID, std::string("CMSSW_5_1_40patch1"), std::string(""));
    edm::ProcessConfiguration pc1b(std::string("HLT"), psetID, std::string("CMSSW_5_1_40patch2"), std::string(""));
    edm::ProcessConfiguration pc2(std::string("HLT"), psetID, std::string("CMSSW_5_2_40"), std::string(""));
    edm::ProcessConfiguration pc2a(std::string("HLT"), psetID, std::string("CMSSW_5_2_40patch1"), std::string(""));
    edm::ProcessConfiguration pc2b(std::string("HLT"), psetID, std::string("CMSSW_5_2_40patch2"), std::string(""));
    edm::ProcessConfiguration pc3(std::string("HLT"), psetID, std::string("CMSSW_5_3_40"), std::string(""));
    edm::ProcessConfiguration pc4(std::string("HLT"), psetID, std::string("CMSSW_5_4_40"), std::string(""));

    ph1.push_back(pc1);
    ph1a.push_back(pc1a);
    ph1b.push_back(pc1b);
    ph2.push_back(pc2);
    ph2a.push_back(pc2a);
    ph2b.push_back(pc2b);
    ph3.push_back(pc3);
    ph4.push_back(pc4);

    edm::ProcessHistoryID phid1 = ph1.id();
    edm::ProcessHistoryID phid1a = ph1a.id();
    edm::ProcessHistoryID phid1b = ph1b.id();
    edm::ProcessHistoryID phid2 = ph2.id();
    edm::ProcessHistoryID phid2a = ph2a.id();
    edm::ProcessHistoryID phid2b = ph2b.id();
    edm::ProcessHistoryID phid3 = ph3.id();
    edm::ProcessHistoryID phid4 = ph4.id();

    edm::ProcessHistoryRegistry::instance()->insertMapped(ph1);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph1a);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph1b);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph2);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph2a);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph2b);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph3);
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph4);

    edm::IndexIntoFile indexIntoFile;
    indexIntoFile.addEntry(phid1, 1, 0, 0, 0);
    indexIntoFile.addEntry(phid2, 2, 0, 0, 1);
    indexIntoFile.addEntry(phid3, 3, 0, 0, 2);
    indexIntoFile.addEntry(phid4, 4, 0, 0, 3);

    indexIntoFile.sortVector_Run_Or_Lumi_Entries();

    indexIntoFile.reduceProcessHistoryIDs();

    edm::ProcessHistory rph1 = ph1;
    edm::ProcessHistory rph1a = ph1a;
    edm::ProcessHistory rph1b = ph1b;
    edm::ProcessHistory rph2 = ph2;
    edm::ProcessHistory rph2a = ph2a;
    edm::ProcessHistory rph2b = ph2b;
    edm::ProcessHistory rph3 = ph3;
    edm::ProcessHistory rph4 = ph4;
    rph1.reduce();
    rph1a.reduce();
    rph1b.reduce();
    rph2.reduce();
    rph2a.reduce();
    rph2b.reduce();
    rph3.reduce();
    rph4.reduce();

    std::vector<edm::ProcessHistoryID> const& v = indexIntoFile.processHistoryIDs();
    assert(v[0] == rph1.id());
    assert(v[1] == rph2.id());
    assert(v[2] == rph3.id());
    assert(v[3] == rph4.id());

    edm::IndexIntoFile indexIntoFile1;
    indexIntoFile1.addEntry(phid1,  1, 11, 0, 0);
    indexIntoFile1.addEntry(phid1,  1, 12, 0, 1);
    indexIntoFile1.addEntry(phid1,  1, 0,  0, 0);
    indexIntoFile1.addEntry(phid2,  2, 11, 0, 2);
    indexIntoFile1.addEntry(phid2,  2, 12, 0, 3);
    indexIntoFile1.addEntry(phid2,  2, 0,  0, 1);
    indexIntoFile1.addEntry(phid1a, 1, 11, 1, 0);
    indexIntoFile1.addEntry(phid1a, 1, 11, 2, 1);
    indexIntoFile1.addEntry(phid1a, 1, 11, 0, 4);
    indexIntoFile1.addEntry(phid1a, 1, 12, 1, 2);
    indexIntoFile1.addEntry(phid1a, 1, 12, 2, 3);
    indexIntoFile1.addEntry(phid1a, 1, 12, 0, 5);
    indexIntoFile1.addEntry(phid1a, 1, 0,  0, 2);
    indexIntoFile1.addEntry(phid3,  3, 0,  0, 3);
    indexIntoFile1.addEntry(phid2a, 2, 0,  0, 4);
    indexIntoFile1.addEntry(phid2b, 2, 0,  0, 5);
    indexIntoFile1.addEntry(phid4,  4, 0,  0, 6);
    indexIntoFile1.addEntry(phid1b, 1, 0,  0, 7);
    indexIntoFile1.addEntry(phid1,  5, 11, 0, 6);
    indexIntoFile1.addEntry(phid1,  5, 0,  0, 8);
    indexIntoFile1.addEntry(phid4,  1, 11, 0, 7);
    indexIntoFile1.addEntry(phid4,  1, 0,  0, 9);

    indexIntoFile1.sortVector_Run_Or_Lumi_Entries();

    std::vector<edm::ProcessHistoryID> const& v1 = indexIntoFile1.processHistoryIDs();
    assert(v1.size() == 8U);

    indexIntoFile1.reduceProcessHistoryIDs();

    std::vector<edm::ProcessHistoryID> const& rv1 = indexIntoFile1.processHistoryIDs();
    assert(rv1.size() == 4U);
    assert(rv1[0] == rph1.id());
    assert(rv1[1] == rph2.id());
    assert(rv1[2] == rph3.id());
    assert(rv1[3] == rph4.id());

    std::vector<edm::IndexIntoFile::RunOrLumiEntry>& runOrLumiEntries = indexIntoFile1.setRunOrLumiEntries();

    assert(runOrLumiEntries.size() == 18U);

    /*
    std::cout << rv1[0] << "  " << rph1 << "\n";
    std::cout << rv1[1] << "  " << rph2 << "\n";
    std::cout << rv1[2] << "  " << rph3 << "\n";
    std::cout << rv1[3] << "  " << rph4 << "\n";

    for (std::vector<edm::IndexIntoFile::RunOrLumiEntry>::const_iterator iter = runOrLumiEntries.begin(),
	   iEnd = runOrLumiEntries.end();
         iter != iEnd; ++iter) {
      std::cout << iter->orderPHIDRun() << "  "
                << iter->orderPHIDRunLumi() << "  "
                << iter->entry() << "  "
                << iter->processHistoryIDIndex() << "  "
                << iter->run() << "  "
                << iter->lumi() << "  "
                << iter->beginEvents() << "  "
                << iter->endEvents() << "\n";

    }
    */

    assert(checkRunOrLumiEntry(runOrLumiEntries.at(0),  0, -1, 0, 0, 1,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(1),  0, -1, 2, 0, 1,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(2),  0, -1, 7, 0, 1,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(3),  0,  0, 0, 0, 1, 11, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(4),  0,  0, 4, 0, 1, 11,  0,  2));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(5),  0,  1, 1, 0, 1, 12, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(6),  0,  1, 5, 0, 1, 12,  2,  4));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(7),  1, -1, 1, 1, 2,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(8),  1, -1, 4, 1, 2,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(9),  1, -1, 5, 1, 2,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(10), 1,  2, 2, 1, 2, 11, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(11), 1,  3, 3, 1, 2, 12, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(12), 3, -1, 3, 2, 3,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(13), 6, -1, 6, 3, 4,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(14), 8, -1, 8, 0, 5,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(15), 8,  6, 6, 0, 5, 11, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(16), 9, -1, 9, 3, 1,  0, -1, -1));
    assert(checkRunOrLumiEntry(runOrLumiEntries.at(17), 9,  7, 7, 3, 1, 11, -1, -1));
  }
}

