/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "EmbeddedRootSource.h"
#include "InputFile.h"
#include "RootFile.h"
#include "RootEmbeddedFileSequence.h"
#include "RootTree.h"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Random/RandFlat.h"

#include <random>

namespace edm {
  class EventPrincipal;

  RootEmbeddedFileSequence::RootEmbeddedFileSequence(ParameterSet const& pset,
                                                     EmbeddedRootSource& input,
                                                     InputFileCatalog const& catalog)
      : RootInputFileSequence(pset, catalog),
        input_(input),
        orderedProcessHistoryIDs_(),
        sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
        sameLumiBlock_(pset.getUntrackedParameter<bool>("sameLumiBlock", false)),
        fptr_(nullptr),
        eventsRemainingInFile_(0),
        // The default value provided as the second argument to the getUntrackedParameter function call
        // is not used when the ParameterSet has been validated and the parameters are not optional
        // in the description.  This is currently true when PoolSource is the primary input source.
        // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
        // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
        // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
        // and should be deleted from the code.
        initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0U)),
        treeCacheSize_(pset.getUntrackedParameter<unsigned int>("cacheSize", roottree::defaultCacheSize)),
        enablePrefetching_(false),
        enforceGUIDInFileName_(pset.getUntrackedParameter<bool>("enforceGUIDInFileName", false)) {
    if (noFiles()) {
      throw Exception(errors::NoSecondaryFiles)
          << "RootEmbeddedFileSequence no input files specified for secondary input source.\n";
    }
    //
    // The SiteLocalConfig controls the TTreeCache size and the prefetching settings.
    Service<SiteLocalConfig> pSLC;
    if (pSLC.isAvailable()) {
      if (treeCacheSize_ != 0U && pSLC->sourceTTreeCacheSize()) {
        treeCacheSize_ = *(pSLC->sourceTTreeCacheSize());
      }
      enablePrefetching_ = pSLC->enablePrefetching();
    }

    // Set the pointer to the function that reads an event.
    if (sameLumiBlock_) {
      if (sequential_) {
        fptr_ = &RootEmbeddedFileSequence::readOneSequentialWithID;
      } else {
        fptr_ = &RootEmbeddedFileSequence::readOneRandomWithID;
      }
    } else {
      if (sequential_) {
        fptr_ = &RootEmbeddedFileSequence::readOneSequential;
      } else {
        fptr_ = &RootEmbeddedFileSequence::readOneRandom;
      }
    }

    // For the secondary input source we do not stage in.
    if (sequential_) {
      // We open the first file
      if (!atFirstFile()) {
        setAtFirstFile();
        initFile(false);
      }
      assert(rootFile());
      rootFile()->setAtEventEntry(IndexIntoFile::invalidEntry);
      if (!sameLumiBlock_) {
        skipEntries(initialNumberOfEventsToSkip_);
      }
    } else {
      // We randomly choose the first file to open.
      // We cannot use the random number service yet.
      std::ifstream f("/dev/urandom");
      unsigned int seed;
      f.read(reinterpret_cast<char*>(&seed), sizeof(seed));
      std::default_random_engine dre(seed);
      size_t count = numberOfFiles();
      std::uniform_int_distribution<int> distribution(0, count - 1);
      while (!rootFile() && count != 0) {
        --count;
        int offset = distribution(dre);
        setAtFileSequenceNumber(offset);
        initFile(input_.skipBadFiles());
      }
    }
    if (rootFile()) {
      input_.productRegistryUpdate().updateFromInput(rootFile()->productRegistry()->productList());
    }
  }

  RootEmbeddedFileSequence::~RootEmbeddedFileSequence() {}

  void RootEmbeddedFileSequence::endJob() { closeFile_(); }

  void RootEmbeddedFileSequence::closeFile_() {
    // delete the RootFile object.
    if (rootFile()) {
      rootFile().reset();
    }
  }

  void RootEmbeddedFileSequence::initFile_(bool skipBadFiles) {
    initTheFile(skipBadFiles, false, nullptr, "mixingFiles", InputType::SecondarySource);
  }

  RootEmbeddedFileSequence::RootFileSharedPtr RootEmbeddedFileSequence::makeRootFile(
      std::shared_ptr<InputFile> filePtr) {
    size_t currentIndexIntoFile = sequenceNumberOfFile();
    return std::make_shared<RootFile>(fileName(),
                                      ProcessConfiguration(),
                                      logicalFileName(),
                                      filePtr,
                                      input_.nStreams(),
                                      treeCacheSize_,
                                      input_.treeMaxVirtualSize(),
                                      input_.runHelper(),
                                      input_.productSelectorRules(),
                                      InputType::SecondarySource,
                                      input_.processHistoryRegistryForUpdate(),
                                      indexesIntoFiles(),
                                      currentIndexIntoFile,
                                      orderedProcessHistoryIDs_,
                                      input_.bypassVersionCheck(),
                                      enablePrefetching_,
                                      enforceGUIDInFileName_);
  }

  void RootEmbeddedFileSequence::skipEntries(unsigned int offset) {
    // offset is decremented by the number of events actually skipped.
    bool completed = rootFile()->skipEntries(offset);
    while (!completed) {
      setAtNextFile();
      if (noMoreFiles()) {
        setAtFirstFile();
      }
      initFile(false);
      assert(rootFile());
      rootFile()->setAtEventEntry(IndexIntoFile::invalidEntry);
      completed = rootFile()->skipEntries(offset);
    }
  }

  bool RootEmbeddedFileSequence::readOneSequential(
      EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const*, bool recycleFiles) {
    assert(rootFile());
    rootFile()->nextEventEntry();
    bool found = rootFile()->readCurrentEvent(cache);
    if (!found) {
      setAtNextFile();
      if (noMoreFiles()) {
        if (recycleFiles) {
          setAtFirstFile();
        } else {
          return false;
        }
      }
      initFile(false);
      assert(rootFile());
      rootFile()->setAtEventEntry(IndexIntoFile::invalidEntry);
      return readOneSequential(cache, fileNameHash, nullptr, nullptr, recycleFiles);
    }
    fileNameHash = lfnHash();
    return true;
  }

  bool RootEmbeddedFileSequence::readOneSequentialWithID(
      EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* idp, bool recycleFiles) {
    assert(idp);
    EventID const& id = *idp;
    int offset = initialNumberOfEventsToSkip_;
    initialNumberOfEventsToSkip_ = 0;
    if (offset > 0) {
      assert(rootFile());
      while (offset > 0) {
        bool found = readOneSequentialWithID(cache, fileNameHash, nullptr, idp, recycleFiles);
        if (!found) {
          return false;
        }
        --offset;
      }
    }
    assert(rootFile());
    if (noMoreFiles() || rootFile()->indexIntoFileIter().run() != id.run() ||
        rootFile()->indexIntoFileIter().lumi() != id.luminosityBlock()) {
      bool found = skipToItem(id.run(), id.luminosityBlock(), 0, 0, false);
      if (!found) {
        return false;
      }
    }
    assert(rootFile());
    bool found = rootFile()->setEntryAtNextEventInLumi(id.run(), id.luminosityBlock());
    if (found) {
      found = rootFile()->readCurrentEvent(cache);
    }
    if (!found) {
      found = skipToItemInNewFile(id.run(), id.luminosityBlock(), 0);
      if (!found) {
        return false;
      }
      return readOneSequentialWithID(cache, fileNameHash, nullptr, idp, recycleFiles);
    }
    fileNameHash = lfnHash();
    return true;
  }

  void RootEmbeddedFileSequence::readOneSpecified(EventPrincipal& cache,
                                                  size_t& fileNameHash,
                                                  SecondaryEventIDAndFileInfo const& idx) {
    EventID const& id = idx.eventID();
    bool found = skipToItem(id.run(), id.luminosityBlock(), id.event(), idx.fileNameHash());
    if (!found) {
      throw Exception(errors::NotFound) << "RootEmbeddedFileSequence::readOneSpecified(): Secondary Input files"
                                        << " do not contain specified event:\n"
                                        << id << "\n";
    }
    assert(rootFile());
    found = rootFile()->readCurrentEvent(cache);
    assert(found);
    fileNameHash = idx.fileNameHash();
    if (fileNameHash == 0U) {
      fileNameHash = lfnHash();
    }
  }

  bool RootEmbeddedFileSequence::readOneRandom(
      EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine* engine, EventID const*, bool) {
    assert(rootFile());
    assert(engine);
    unsigned int currentSeqNumber = sequenceNumberOfFile();
    while (eventsRemainingInFile_ == 0) {
      unsigned int newSeqNumber = CLHEP::RandFlat::shootInt(engine, fileCatalogItems().size());
      setAtFileSequenceNumber(newSeqNumber);
      if (newSeqNumber != currentSeqNumber) {
        initFile(false);
        currentSeqNumber = newSeqNumber;
      }
      eventsRemainingInFile_ = rootFile()->eventTree().entries();
      if (eventsRemainingInFile_ == 0) {
        throw Exception(errors::NotFound) << "RootEmbeddedFileSequence::readOneRandom(): Secondary Input file "
                                          << fileName() << " contains no events.\n";
      }
      rootFile()->setAtEventEntry(CLHEP::RandFlat::shootInt(engine, eventsRemainingInFile_) - 1);
    }
    rootFile()->nextEventEntry();

    bool found = rootFile()->readCurrentEvent(cache);
    if (!found) {
      rootFile()->setAtEventEntry(0);
      found = rootFile()->readCurrentEvent(cache);
      assert(found);
    }
    fileNameHash = lfnHash();
    --eventsRemainingInFile_;
    return true;
  }

  bool RootEmbeddedFileSequence::readOneRandomWithID(EventPrincipal& cache,
                                                     size_t& fileNameHash,
                                                     CLHEP::HepRandomEngine* engine,
                                                     EventID const* idp,
                                                     bool recycleFiles) {
    assert(engine);
    assert(idp);
    EventID const& id = *idp;
    if (noMoreFiles() || !rootFile() || rootFile()->indexIntoFileIter().run() != id.run() ||
        rootFile()->indexIntoFileIter().lumi() != id.luminosityBlock()) {
      bool found = skipToItem(id.run(), id.luminosityBlock(), 0);
      if (!found) {
        return false;
      }
      int eventsInLumi = 0;
      assert(rootFile());
      while (rootFile()->setEntryAtNextEventInLumi(id.run(), id.luminosityBlock()))
        ++eventsInLumi;
      found = skipToItem(id.run(), id.luminosityBlock(), 0);
      assert(found);
      int eventInLumi = CLHEP::RandFlat::shootInt(engine, eventsInLumi);
      for (int i = 0; i < eventInLumi; ++i) {
        bool foundEventInLumi = rootFile()->setEntryAtNextEventInLumi(id.run(), id.luminosityBlock());
        assert(foundEventInLumi);
      }
    }
    assert(rootFile());
    bool found = rootFile()->setEntryAtNextEventInLumi(id.run(), id.luminosityBlock());
    if (found) {
      found = rootFile()->readCurrentEvent(cache);
    }
    if (!found) {
      found = rootFile()->setEntryAtItem(id.run(), id.luminosityBlock(), 0);
      if (!found) {
        return false;
      }
      return readOneRandomWithID(cache, fileNameHash, engine, idp, recycleFiles);
    }
    fileNameHash = lfnHash();
    return true;
  }

  bool RootEmbeddedFileSequence::readOneEvent(EventPrincipal& cache,
                                              size_t& fileNameHash,
                                              CLHEP::HepRandomEngine* engine,
                                              EventID const* id,
                                              bool recycleFiles) {
    assert(!sameLumiBlock_ || id != nullptr);
    assert(sequential_ || engine != nullptr);
    return (this->*fptr_)(cache, fileNameHash, engine, id, recycleFiles);
  }

  void RootEmbeddedFileSequence::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<bool>("sequential", false)
        ->setComment(
            "True: loopEvents() reads events sequentially from beginning of first file.\n"
            "False: loopEvents() first reads events beginning at random event. New files also chosen randomly");
    desc.addUntracked<bool>("sameLumiBlock", false)
        ->setComment(
            "True: loopEvents() reads events only in same lumi as the specified event.\n"
            "False: loopEvents() reads events regardless of lumi.");
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment(
            "Skip the first 'skipEvents' events. Used only if 'sequential' is True and 'sameLumiBlock' is False");
    desc.addUntracked<unsigned int>("cacheSize", roottree::defaultCacheSize)
        ->setComment("Size of ROOT TTree prefetch cache.  Affects performance.");
    desc.addUntracked<bool>("enforceGUIDInFileName", false)
        ->setComment(
            "True:  file name part is required to be equal to the GUID of the file\n"
            "False: file name can be anything");
  }
}  // namespace edm
