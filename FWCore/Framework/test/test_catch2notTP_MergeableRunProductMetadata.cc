#include "catch.hpp"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <memory>
#include <string>
#include <vector>

TEST_CASE("test MergeableRunProductMetadata", "[MergeableRunProductMetadata]") {

  SECTION("test nested class MetadataForProcess") {

    edm::MergeableRunProductMetadata::MetadataForProcess metaDataForProcess;
    REQUIRE(metaDataForProcess.lumis().empty());
    REQUIRE(metaDataForProcess.valid());
    REQUIRE(!metaDataForProcess.useIndexIntoFile());
    REQUIRE(!metaDataForProcess.allLumisProcessed());
    REQUIRE(metaDataForProcess.mergeDecision() == edm::MergeableRunProductMetadata::MERGE);

    std::vector<edm::LuminosityBlockNumber_t>& lumis = metaDataForProcess.lumis();
    lumis.push_back(11);
    edm::MergeableRunProductMetadata::MetadataForProcess const& acref(metaDataForProcess);
    REQUIRE(acref.lumis().at(0) == 11);

    metaDataForProcess.setMergeDecision(edm::MergeableRunProductMetadata::IGNORE);
    metaDataForProcess.setValid(false);
    metaDataForProcess.setUseIndexIntoFile(true);
    metaDataForProcess.setAllLumisProcessed(true);

    REQUIRE(!metaDataForProcess.valid());
    REQUIRE(metaDataForProcess.useIndexIntoFile());
    REQUIRE(metaDataForProcess.allLumisProcessed());
    REQUIRE(metaDataForProcess.mergeDecision() == edm::MergeableRunProductMetadata::IGNORE);

    metaDataForProcess.reset();
    REQUIRE(metaDataForProcess.lumis().empty());
    REQUIRE(metaDataForProcess.valid());
    REQUIRE(!metaDataForProcess.useIndexIntoFile());
    REQUIRE(!metaDataForProcess.allLumisProcessed());
    REQUIRE(metaDataForProcess.mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
  }

  SECTION("test constructor") {
    // edm::MergeableRunProductMetadata mergeableRunProductMetadata(nullptr);
    // REQUIRE(mergeableRunProductMetadata.metadataForProcesses().empty());

    edm::MergeableRunProductProcesses mergeableRunProductProcesses;
    edm::MergeableRunProductMetadata mergeableRunProductMetadata2(mergeableRunProductProcesses);
    REQUIRE(mergeableRunProductMetadata2.metadataForProcesses().empty());
  }

  SECTION("test main functions") {

    edm::ProductRegistry productRegistry;
    edm::ParameterSet dummyPset;
    dummyPset.registerIt();

    // not mergeable
    edm::BranchDescription prod1(edm::InRun,
                                "label",
                                "PROD",
                                "edmtest::Thing",
                                "edmtestThing",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::Thing"),
                                false);
    productRegistry.copyProduct(prod1);

    // This one should be used
    edm::BranchDescription prod2(edm::InRun,
                                "aLabel",
                                "APROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                false);
    productRegistry.copyProduct(prod2);

    //not in a Run
    edm::BranchDescription prod3(edm::InLumi,
                                "bLabel",
                                "BPROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                false);
    productRegistry.copyProduct(prod3);

    // produced
    edm::BranchDescription prod4(edm::InRun,
                                "cLabel",
                                "CPROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                true);
    productRegistry.addProduct(prod4);

    // dropped
    edm::BranchDescription prod5(edm::InRun,
                                "dLabel",
                                "DPROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                false);
    prod5.setDropped(true);
    productRegistry.copyProduct(prod5);

    // Should be used but the same process name
    edm::BranchDescription prod6(edm::InRun,
                                "eLabel",
                                "APROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                false);
    productRegistry.copyProduct(prod6);

    // Should be used
    edm::BranchDescription prod7(edm::InRun,
                                "fLabel",
                                "AAPROD",
                                "edmtest::ThingWithMerge",
                                "edmtestThingWithMerge",
                                "instance",
                                "aModule",
                                dummyPset.id(),
                                edm::TypeWithDict::byName("edmtest::ThingWithMerge"),
                                false);
    productRegistry.copyProduct(prod7);

    edm::MergeableRunProductProcesses mergeableRunProductProcesses;

    REQUIRE(mergeableRunProductProcesses.size() == 0);

    mergeableRunProductProcesses.setProcessesWithMergeableRunProducts(productRegistry);

    REQUIRE(mergeableRunProductProcesses.size() == 2);

    std::vector<std::string> expected { "AAPROD", "APROD" };
    REQUIRE(mergeableRunProductProcesses.processesWithMergeableRunProducts() == expected);

    std::vector<std::string> storedProcesses { "AAAPROD", "AAPROD", "APROD", "ZPROD" };
    edm::StoredMergeableRunProductMetadata storedMetadata(storedProcesses);
    REQUIRE(storedMetadata.processesWithMergeableRunProducts() == storedProcesses);
    REQUIRE(storedMetadata.allValidAndUseIndexIntoFile());

    unsigned long long iBeginProcess { 100 };
    unsigned long long iEndProcess { 101 };
    edm::StoredMergeableRunProductMetadata::SingleRunEntry singleRunEntry(iBeginProcess, iEndProcess);
    REQUIRE(singleRunEntry.beginProcess() == 100);
    REQUIRE(singleRunEntry.endProcess() == 101);

    unsigned long long iBeginLumi { 200 };
    unsigned long long iEndLumi { 201 };
    unsigned int iProcess { 202 };
    bool iValid { true };
    bool iUseIndexIntoFile { false };
    edm::StoredMergeableRunProductMetadata::SingleRunEntryAndProcess singleRunEntryAndProcess(
      iBeginLumi,
      iEndLumi,
      iProcess,
      iValid,
      iUseIndexIntoFile);
    REQUIRE(singleRunEntryAndProcess.beginLumi() == 200);
    REQUIRE(singleRunEntryAndProcess.endLumi() == 201);
    REQUIRE(singleRunEntryAndProcess.process() == 202);
    REQUIRE(singleRunEntryAndProcess.valid() == true);
    REQUIRE(singleRunEntryAndProcess.useIndexIntoFile() == false);

    // Fill storedMetadata with fake data
    // 9 run entries
    // Let AAPROD be not present for all run entries
    // Let APROD be there with a vector
    //              not there at all (no entries for any process)
    //              there invalid with no vector
    //              not there at all
    //              there with a vector and invalid
    // Let AAAPROD and ZPROD be there for all entries 0, 2, 3, 4
    std::vector<edm::StoredMergeableRunProductMetadata::SingleRunEntry>& singleRunEntries = storedMetadata.singleRunEntries();
    singleRunEntries.emplace_back(0,3);
    singleRunEntries.emplace_back(3,3);
    singleRunEntries.emplace_back(3,6);
    singleRunEntries.emplace_back(6,8);
    singleRunEntries.emplace_back(8,11);
    singleRunEntries.emplace_back(11,12);
    singleRunEntries.emplace_back(12,13);
    singleRunEntries.emplace_back(13,14);
    singleRunEntries.emplace_back(14,15);

    std::vector<edm::StoredMergeableRunProductMetadata::SingleRunEntryAndProcess>& singleRunEntryAndProcesses = storedMetadata.singleRunEntryAndProcesses();
    std::vector<edm::LuminosityBlockNumber_t>& lumis = storedMetadata.lumis();

    // arguments: beginLumi, endLumi, process Index, valid, useIndexIntoFile
    unsigned int iAAAPROD { 0 };
    // unsigned int iAAPROD { 1 };
    unsigned int iAPROD { 2 };
    unsigned int iZPROD { 3 };

    // 0th run entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAAAPROD, true, false);

    iBeginLumi = lumis.size();
    lumis.emplace_back(101);
    lumis.emplace_back(102);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, true, false);

    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iZPROD, true, false);

    // nothing in the 1st entry

    // 2nd run entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAAAPROD, true, false);

    iBeginLumi = lumis.size();
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, false, true);

    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iZPROD, true, false);

    // 3rd run entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAAAPROD, true, false);

    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iZPROD, true, false);

    // 4th run entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAAAPROD, true, false);

    iBeginLumi = lumis.size();
    lumis.emplace_back(111);
    lumis.emplace_back(112);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, false, false);

    iBeginLumi = lumis.size();
    lumis.emplace_back(1);
    lumis.emplace_back(2);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iZPROD, true, false);

    // 5th entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(91);
    lumis.emplace_back(92);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, true, false);

    // 6th entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(121);
    lumis.emplace_back(122);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, true, false);

    // 7th entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(95);
    lumis.emplace_back(101);
    lumis.emplace_back(105);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, true, false);

    // 8th entry
    iBeginLumi = lumis.size();
    lumis.emplace_back(1000);
    lumis.emplace_back(1002);
    lumis.emplace_back(1003);
    lumis.emplace_back(1004);
    iEndLumi = lumis.size();
    singleRunEntryAndProcesses.emplace_back(iBeginLumi, iEndLumi, iAPROD, true, false);

    storedMetadata.allValidAndUseIndexIntoFile() = false;

    bool valid;
    std::vector<edm::LuminosityBlockNumber_t>::const_iterator lumisBegin;
    std::vector<edm::LuminosityBlockNumber_t>::const_iterator lumisEnd;
    REQUIRE(storedMetadata.getLumiContent(0, std::string("APROD"), valid, lumisBegin, lumisEnd));
    {
      std::vector<edm::LuminosityBlockNumber_t> test(lumisBegin, lumisEnd);
      std::vector<edm::LuminosityBlockNumber_t> expected { 101, 102 };
      REQUIRE(test == expected);
      REQUIRE(valid);
    }
    REQUIRE(!storedMetadata.getLumiContent(1, std::string("APROD"), valid, lumisBegin, lumisEnd));
    REQUIRE(valid);
    REQUIRE(!storedMetadata.getLumiContent(2, std::string("APROD"), valid, lumisBegin, lumisEnd));
    REQUIRE(!valid);
    REQUIRE(!storedMetadata.getLumiContent(3, std::string("APROD"), valid, lumisBegin, lumisEnd));
    REQUIRE(valid);
    REQUIRE(storedMetadata.getLumiContent(4, std::string("APROD"), valid, lumisBegin, lumisEnd));
    {
      std::vector<edm::LuminosityBlockNumber_t> test(lumisBegin, lumisEnd);
      std::vector<edm::LuminosityBlockNumber_t> expected { 111, 112 };
      REQUIRE(test == expected);
      REQUIRE(!valid);
    }

    edm::IndexIntoFile indexIntoFile;

    edm::ProcessHistoryID fakePHID1;
    edm::ProcessConfiguration pc;
    auto processHistory1 = std::make_unique<edm::ProcessHistory>();
    edm::ProcessHistory& ph1 = *processHistory1;
    processHistory1->push_back(pc);
    fakePHID1 = ph1.id();

    edm::ProcessHistoryID fakePHID2;
    auto processHistory2 = std::make_unique<edm::ProcessHistory>();
    edm::ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    fakePHID2 = ph2.id();

    indexIntoFile.addEntry(fakePHID1, 11, 1001, 7, 0); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 6, 1); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 0, 0); // Lumi
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 0, 1); // Lumi
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 5, 2); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 4, 3); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1001, 0, 2); // Lumi
    indexIntoFile.addEntry(fakePHID1, 11, 1003, 5, 4); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1003, 4, 5); // Event
    indexIntoFile.addEntry(fakePHID1, 11, 1003, 0, 3); // Lumi
    indexIntoFile.addEntry(fakePHID1, 11,   0, 0, 0); // Run
    indexIntoFile.addEntry(fakePHID2, 11,   0, 0, 1); // Run
    indexIntoFile.addEntry(fakePHID2, 11, 1001, 0, 4); // Lumi
    indexIntoFile.addEntry(fakePHID2, 11, 1003, 0, 5); // Lumi
    indexIntoFile.addEntry(fakePHID2, 11, 1003, 4, 6); // Event
    indexIntoFile.addEntry(fakePHID2, 11, 1003, 0, 6); // Lumi
    indexIntoFile.addEntry(fakePHID2, 11,   0, 0, 2); // Run
    indexIntoFile.sortVector_Run_Or_Lumi_Entries();
    edm::IndexIntoFile::IndexIntoFileItr iter = indexIntoFile.begin(edm::IndexIntoFile::firstAppearanceOrder);


    edm::MergeableRunProductMetadata mergeableRunProductMetadata(mergeableRunProductProcesses);

    edm::MergeableRunProductMetadata::MetadataForProcess const* APRODMetadataForProcess =
      mergeableRunProductMetadata.metadataForOneProcess("APROD");
    REQUIRE(APRODMetadataForProcess);

    edm::MergeableRunProductMetadata::MetadataForProcess const* NullMetadataForProcess =
      mergeableRunProductMetadata.metadataForOneProcess("DOESNOTEXIST");
    REQUIRE(NullMetadataForProcess == nullptr);

    mergeableRunProductMetadata.readRun(0, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 101, 102 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::REPLACE);
      REQUIRE(mergeableRunProductMetadata.getMergeDecision("APROD") == edm::MergeableRunProductMetadata::REPLACE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!mergeableRunProductMetadata.knownImproperlyMerged("APROD"));
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    // Same one again for test purposes
    mergeableRunProductMetadata.readRun(0, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 101, 102 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::IGNORE);
      REQUIRE(mergeableRunProductMetadata.getMergeDecision("APROD") == edm::MergeableRunProductMetadata::IGNORE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    // disjoint with newly read ones coming first
    mergeableRunProductMetadata.readRun(5, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 101, 102 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(mergeableRunProductMetadata.getMergeDecision("APROD") == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    // disjoint with newly read ones coming last
    mergeableRunProductMetadata.readRun(6, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 101, 102, 121, 122 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    // impossible to merge case, shared elements and both also have at least one non-shared element
    mergeableRunProductMetadata.readRun(7, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 95, 101, 102, 105, 121, 122 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(!APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    mergeableRunProductMetadata.preReadFile();
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 95, 101, 102, 105, 121, 122 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
    }
    mergeableRunProductMetadata.readRun(3, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 95, 101, 102, 105, 121, 122 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(!APRODMetadataForProcess->valid());
      REQUIRE(APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    mergeableRunProductMetadata.preReadFile();
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 91, 92, 95, 101, 102, 105, 121, 122, 1001, 1003 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
    }

    mergeableRunProductMetadata.postWriteRun();
    {
      std::vector<edm::LuminosityBlockNumber_t> expected;
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    mergeableRunProductMetadata.readRun(0, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 101, 102 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::REPLACE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    mergeableRunProductMetadata.readRun(4, storedMetadata, edm::IndexIntoFileItrHolder(iter));
    {
      std::vector<edm::LuminosityBlockNumber_t> expected { 101, 102, 111, 112 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(!APRODMetadataForProcess->valid());
      REQUIRE(mergeableRunProductMetadata.knownImproperlyMerged("APROD"));
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }
    mergeableRunProductMetadata.preReadFile();
    mergeableRunProductMetadata.postWriteRun();
    {
      std::vector<edm::LuminosityBlockNumber_t> expected3;
      REQUIRE(expected3 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(!mergeableRunProductMetadata.gotLumisFromIndexIntoFile());

      mergeableRunProductMetadata.readRun(1, storedMetadata, edm::IndexIntoFileItrHolder(iter));

      std::vector<edm::LuminosityBlockNumber_t> expected;
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(APRODMetadataForProcess->valid());
      REQUIRE(APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());

      std::vector<edm::LuminosityBlockNumber_t> expected2 { 1001, 1003 };
      REQUIRE(expected2 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(mergeableRunProductMetadata.gotLumisFromIndexIntoFile());
    }
    mergeableRunProductMetadata.preReadFile();
    {
      std::vector<edm::LuminosityBlockNumber_t> expected3;
      REQUIRE(expected3 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(!mergeableRunProductMetadata.gotLumisFromIndexIntoFile());

      std::vector<edm::LuminosityBlockNumber_t> expected {1001, 1003 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(!APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());
    }

    mergeableRunProductMetadata.postWriteRun();
    {
      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));

      std::vector<edm::LuminosityBlockNumber_t> expected;
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      REQUIRE(APRODMetadataForProcess->mergeDecision() == edm::MergeableRunProductMetadata::MERGE);
      REQUIRE(!APRODMetadataForProcess->valid());
      REQUIRE(APRODMetadataForProcess->useIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());

      std::vector<edm::LuminosityBlockNumber_t> expected2 { 1001, 1003 };
      REQUIRE(expected2 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(mergeableRunProductMetadata.gotLumisFromIndexIntoFile());
    }
    {
      mergeableRunProductMetadata.readRun(8, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      std::vector<edm::LuminosityBlockNumber_t> expected { 1000, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      REQUIRE(APRODMetadataForProcess->lumis() == expected);
      mergeableRunProductMetadata.preReadFile();
      std::vector<edm::LuminosityBlockNumber_t> expected2 { 1000, 1001, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected2);
    }
    mergeableRunProductMetadata.postWriteRun();

    {
      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.readRun(8, storedMetadata, edm::IndexIntoFileItrHolder(iter));

      std::vector<edm::LuminosityBlockNumber_t> expected1 { 1001, 1003 };
      REQUIRE(expected1 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(mergeableRunProductMetadata.gotLumisFromIndexIntoFile());

      std::vector<edm::LuminosityBlockNumber_t> expected2 { 1000, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected2);

      mergeableRunProductMetadata.writeLumi(1004);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.preWriteRun();

      std::vector<edm::LuminosityBlockNumber_t> expected3;
      REQUIRE(expected3 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(!mergeableRunProductMetadata.gotLumisFromIndexIntoFile());
      REQUIRE(!APRODMetadataForProcess->allLumisProcessed());

      std::vector<edm::LuminosityBlockNumber_t> expected4 { 1000, 1001, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected4);

    }
    mergeableRunProductMetadata.postWriteRun();
    REQUIRE(mergeableRunProductMetadata.lumisProcessed().empty());
    {
      // ----------------------------------------------------

      std::vector<std::string> storedProcessesOutput { "AAAPROD", "AAPROD", "APROD", "ZPROD" };
      edm::StoredMergeableRunProductMetadata storedMetadataOutput(storedProcesses);

      mergeableRunProductMetadata.readRun(1, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.writeLumi(1003);
      mergeableRunProductMetadata.writeLumi(1001);
      mergeableRunProductMetadata.preWriteRun();
      mergeableRunProductMetadata.addEntryToStoredMetadata(storedMetadataOutput);
      mergeableRunProductMetadata.postWriteRun();

      REQUIRE(storedMetadataOutput.allValidAndUseIndexIntoFile());

      // ---------------------------------------------------

      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.readRun(8, storedMetadata, edm::IndexIntoFileItrHolder(iter));

      std::vector<edm::LuminosityBlockNumber_t> expected1 { 1001, 1003 };
      REQUIRE(expected1 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(mergeableRunProductMetadata.gotLumisFromIndexIntoFile());

      std::vector<edm::LuminosityBlockNumber_t> expected2 { 1000, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected2);

      mergeableRunProductMetadata.writeLumi(1004);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.writeLumi(1000);
      mergeableRunProductMetadata.writeLumi(1001);
      mergeableRunProductMetadata.writeLumi(1003);
      mergeableRunProductMetadata.preWriteRun();

      std::vector<edm::LuminosityBlockNumber_t> expected3;
      REQUIRE(expected3 == mergeableRunProductMetadata.lumisFromIndexIntoFile());
      REQUIRE(!mergeableRunProductMetadata.gotLumisFromIndexIntoFile());
      REQUIRE(APRODMetadataForProcess->allLumisProcessed());

      std::vector<edm::LuminosityBlockNumber_t> expected4 { 1000, 1001, 1002, 1003, 1004 };
      REQUIRE(APRODMetadataForProcess->lumis() == expected4);

      mergeableRunProductMetadata.addEntryToStoredMetadata(storedMetadataOutput);

      // ----------------------------------------------------

      mergeableRunProductMetadata.postWriteRun();
      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.preWriteRun();
      mergeableRunProductMetadata.addEntryToStoredMetadata(storedMetadataOutput);

      // ----------------------------------------------------

      mergeableRunProductMetadata.readRun(2, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.readRun(8, storedMetadata, edm::IndexIntoFileItrHolder(iter));
      mergeableRunProductMetadata.writeLumi(1004);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.writeLumi(1002);
      mergeableRunProductMetadata.writeLumi(1000);
      mergeableRunProductMetadata.writeLumi(1001);
      mergeableRunProductMetadata.preWriteRun();
      mergeableRunProductMetadata.addEntryToStoredMetadata(storedMetadataOutput);
      mergeableRunProductMetadata.postWriteRun();

      // ---------------------------------------------------

      std::vector<edm::StoredMergeableRunProductMetadata::SingleRunEntry>& singleRunEntries = storedMetadataOutput.singleRunEntries();
      REQUIRE(singleRunEntries.size() == 4);
      REQUIRE(singleRunEntries[0].beginProcess() == 0);
      REQUIRE(singleRunEntries[0].endProcess() == 0);
      REQUIRE(singleRunEntries[1].beginProcess() == 0);
      REQUIRE(singleRunEntries[1].endProcess() == 1);
      REQUIRE(singleRunEntries[2].beginProcess() == 1);
      REQUIRE(singleRunEntries[2].endProcess() == 3);
      REQUIRE(singleRunEntries[3].beginProcess() == 3);
      REQUIRE(singleRunEntries[3].endProcess() == 5);

      std::vector<edm::StoredMergeableRunProductMetadata::SingleRunEntryAndProcess>& singleRunEntryAndProcesses = storedMetadataOutput.singleRunEntryAndProcesses();
      REQUIRE(singleRunEntryAndProcesses.size() == 5);

      REQUIRE(singleRunEntryAndProcesses[0].beginLumi() == 0);
      REQUIRE(singleRunEntryAndProcesses[0].endLumi() == 0);
      REQUIRE(singleRunEntryAndProcesses[0].process() == 2);
      REQUIRE(!singleRunEntryAndProcesses[0].valid());
      REQUIRE(singleRunEntryAndProcesses[0].useIndexIntoFile());

      REQUIRE(singleRunEntryAndProcesses[1].beginLumi() == 0);
      REQUIRE(singleRunEntryAndProcesses[1].endLumi() == 2);
      REQUIRE(singleRunEntryAndProcesses[1].process() == 1);
      REQUIRE(singleRunEntryAndProcesses[1].valid());
      REQUIRE(!singleRunEntryAndProcesses[1].useIndexIntoFile());

      REQUIRE(singleRunEntryAndProcesses[2].beginLumi() == 0);
      REQUIRE(singleRunEntryAndProcesses[2].endLumi() == 2);
      REQUIRE(singleRunEntryAndProcesses[2].process() == 2);
      REQUIRE(!singleRunEntryAndProcesses[2].valid());
      REQUIRE(!singleRunEntryAndProcesses[2].useIndexIntoFile());

      REQUIRE(singleRunEntryAndProcesses[3].beginLumi() == 2);
      REQUIRE(singleRunEntryAndProcesses[3].endLumi() == 4);
      REQUIRE(singleRunEntryAndProcesses[3].process() == 1);
      REQUIRE(singleRunEntryAndProcesses[3].valid());
      REQUIRE(!singleRunEntryAndProcesses[3].useIndexIntoFile());

      REQUIRE(singleRunEntryAndProcesses[4].beginLumi() == 4);
      REQUIRE(singleRunEntryAndProcesses[4].endLumi() == 9);
      REQUIRE(singleRunEntryAndProcesses[4].process() == 2);
      REQUIRE(!singleRunEntryAndProcesses[4].valid());
      REQUIRE(!singleRunEntryAndProcesses[4].useIndexIntoFile());

      std::vector<edm::LuminosityBlockNumber_t>& lumis = storedMetadataOutput.lumis();
      REQUIRE(lumis.size() == 9);
      REQUIRE(lumis[0] == 1001);
      REQUIRE(lumis[1] == 1003);
      REQUIRE(lumis[2] == 1001);
      REQUIRE(lumis[3] == 1003);
      REQUIRE(lumis[4] == 1000);
      REQUIRE(lumis[5] == 1001);
      REQUIRE(lumis[6] == 1002);
      REQUIRE(lumis[7] == 1003);
      REQUIRE(lumis[8] == 1004);

      REQUIRE(!storedMetadataOutput.allValidAndUseIndexIntoFile());
    }
  }
}
