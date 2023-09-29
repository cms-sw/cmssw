/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "RootFile.h"
#include "RootInputFileSequence.h"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TSystem.h"

namespace edm {
  class BranchIDListHelper;
  class EventPrincipal;
  class LuminosityBlockPrincipal;
  class RunPrincipal;

  RootInputFileSequence::RootInputFileSequence(ParameterSet const& pset, InputFileCatalog const& catalog)
      : catalog_(catalog),
        lfn_("unknown"),
        lfnHash_(0U),
        usedFallback_(false),
        findFileForSpecifiedID_(nullptr),
        fileIterBegin_(fileCatalogItems().begin()),
        fileIterEnd_(fileCatalogItems().end()),
        fileIter_(fileIterEnd_),
        fileIterLastOpened_(fileIterEnd_),
        rootFile_(),
        indexesIntoFiles_(fileCatalogItems().size()) {}

  std::vector<FileCatalogItem> const& RootInputFileSequence::fileCatalogItems() const {
    return catalog_.fileCatalogItems();
  }

  std::shared_ptr<ProductRegistry const> RootInputFileSequence::fileProductRegistry() const {
    assert(rootFile());
    return rootFile()->productRegistry();
  }

  std::shared_ptr<BranchIDListHelper const> RootInputFileSequence::fileBranchIDListHelper() const {
    assert(rootFile());
    return rootFile()->branchIDListHelper();
  }

  RootInputFileSequence::~RootInputFileSequence() {}

  std::shared_ptr<RunAuxiliary> RootInputFileSequence::readRunAuxiliary_() {
    assert(rootFile());
    return rootFile()->readRunAuxiliary_();
  }

  std::shared_ptr<LuminosityBlockAuxiliary> RootInputFileSequence::readLuminosityBlockAuxiliary_() {
    assert(rootFile());
    return rootFile()->readLuminosityBlockAuxiliary_();
  }

  bool RootInputFileSequence::readRun_(RunPrincipal& runPrincipal) {
    assert(rootFile());
    return rootFile()->readRun_(runPrincipal);
  }

  void RootInputFileSequence::fillProcessBlockHelper_() {
    assert(rootFile());
    return rootFile()->fillProcessBlockHelper_();
  }

  bool RootInputFileSequence::nextProcessBlock_(ProcessBlockPrincipal& processBlockPrincipal) {
    assert(rootFile());
    return rootFile()->nextProcessBlock_(processBlockPrincipal);
  }

  void RootInputFileSequence::readProcessBlock_(ProcessBlockPrincipal& processBlockPrincipal) {
    assert(rootFile());
    rootFile()->readProcessBlock_(processBlockPrincipal);
  }

  bool RootInputFileSequence::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    assert(rootFile());
    return rootFile()->readLuminosityBlock_(lumiPrincipal);
  }

  // readEvent() is responsible for setting up the EventPrincipal.
  //
  //   1. fill an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one ProductResolver,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the caches in the EventPrincipal to know about this
  //      ProductResolver.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //

  bool RootInputFileSequence::readEvent(EventPrincipal& eventPrincipal) {
    assert(rootFile());
    return rootFile()->readEvent(eventPrincipal);
  }

  bool RootInputFileSequence::containedInCurrentFile(RunNumber_t run,
                                                     LuminosityBlockNumber_t lumi,
                                                     EventNumber_t event) const {
    if (!rootFile())
      return false;
    return rootFile()->containsItem(run, lumi, event);
  }

  bool RootInputFileSequence::skipToItemInNewFile(RunNumber_t run,
                                                  LuminosityBlockNumber_t lumi,
                                                  EventNumber_t event,
                                                  size_t fileNameHash) {
    // Look for item in files not yet opened. We have a hash of the logical file name
    assert(fileNameHash != 0U);
    // If the lookup table is not yet filled in, fill it.
    if (!findFileForSpecifiedID_) {
      // We use a multimap because there may be hash collisions (Two different LFNs could have the same hash).
      // We map the hash of the LFN to the index into the list of files.
      findFileForSpecifiedID_ =
          std::make_unique<std::unordered_multimap<size_t, size_t>>();  // propagate_const<T> has no reset() function
      auto hasher = std::hash<std::string>();
      for (auto fileIter = fileIterBegin_; fileIter != fileIterEnd_; ++fileIter) {
        findFileForSpecifiedID_->insert(std::make_pair(hasher(fileIter->logicalFileName()), fileIter - fileIterBegin_));
      }
    }
    // Look up the logical file name in the table
    auto range = findFileForSpecifiedID_->equal_range(fileNameHash);
    for (auto iter = range.first; iter != range.second; ++iter) {
      // Don't look in files previously opened, because those have already been searched.
      if (!indexesIntoFiles_[iter->second]) {
        setAtFileSequenceNumber(iter->second);
        initFile_(false);
        assert(rootFile());
        bool found = rootFile()->setEntryAtItem(run, lumi, event);
        if (found) {
          return true;
        }
      }
    }
    // Not found
    return false;
  }

  bool RootInputFileSequence::skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
    // Look for item in files not yet opened.  We do not have a valid hash of the logical file name.
    for (auto it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
      if (!*it) {
        // File not yet opened.
        setAtFileSequenceNumber(it - indexesIntoFiles_.begin());
        initFile_(false);
        assert(rootFile());
        bool found = rootFile()->setEntryAtItem(run, lumi, event);
        if (found) {
          return true;
        }
      }
    }
    // Not found
    return false;
  }

  bool RootInputFileSequence::skipToItem(
      RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash, bool currentFileFirst) {
    // Attempt to find item in currently open input file.
    bool found = currentFileFirst && rootFile() && rootFile()->setEntryAtItem(run, lumi, event);
    if (!found) {
      // If only one input file, give up now, to save time.
      if (currentFileFirst && rootFile() && indexesIntoFiles_.size() == 1) {
        return false;
      }
      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      for (auto it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if (*it && (*it)->containsItem(run, lumi, event)) {
          // We found it. Close the currently open file, and open the correct one.
          std::vector<FileCatalogItem>::const_iterator currentIter = fileIter_;
          setAtFileSequenceNumber(it - indexesIntoFiles_.begin());
          if (fileIter_ != currentIter) {
            initFile(false);
          }
          // Now get the item from the correct file.
          assert(rootFile());
          found = rootFile()->setEntryAtItem(run, lumi, event);
          assert(found);
          return true;
        }
      }
      return (fileNameHash != 0U && skipToItemInNewFile(run, lumi, event, fileNameHash)) ||
             skipToItemInNewFile(run, lumi, event);
    }
    return true;
  }

  //Initiate the file using multiple data catalogs
  void RootInputFileSequence::initTheFile(
      bool skipBadFiles, bool deleteIndexIntoFile, InputSource* input, char const* inputTypeName, InputType inputType) {
    // We are really going to close the open file.

    if (fileIterLastOpened_ != fileIterEnd_) {
      size_t currentIndexIntoFile = fileIterLastOpened_ - fileIterBegin_;
      if (deleteIndexIntoFile) {
        indexesIntoFiles_[currentIndexIntoFile].reset();
      } else {
        if (indexesIntoFiles_[currentIndexIntoFile])
          indexesIntoFiles_[currentIndexIntoFile]->inputFileClosed();
      }
      fileIterLastOpened_ = fileIterEnd_;
    }
    closeFile();

    if (noMoreFiles()) {
      // No files specified
      return;
    }

    // Check if the logical file name was found.
    if (fileNames()[0].empty()) {
      // LFN not found in catalog.
      InputFile::reportSkippedFile(fileNames()[0], logicalFileName());
      if (!skipBadFiles) {
        throw cms::Exception("LogicalFileNameNotFound", "RootFileSequenceBase::initTheFile()\n")
            << "Logical file name '" << logicalFileName() << "' was not found in the file catalog.\n"
            << "If you wanted a local file, you forgot the 'file:' prefix\n"
            << "before the file name in your configuration file.\n";
      }
      LogWarning("") << "Input logical file: " << logicalFileName()
                     << " was not found in the catalog, and will be skipped.\n";
      return;
    }

    lfn_ = logicalFileName().empty() ? fileNames()[0] : logicalFileName();
    lfnHash_ = std::hash<std::string>()(lfn_);
    usedFallback_ = false;

    std::shared_ptr<InputFile> filePtr;
    std::list<std::string> originalInfo;

    std::vector<std::string> const& fNames = fileNames();

    //this tries to open the file using multiple PFNs corresponding to different data catalogs
    {
      std::list<std::string> exInfo;
      std::list<std::string> additionalMessage;
      std::unique_ptr<InputSource::FileOpenSentry> sentry(
          input ? std::make_unique<InputSource::FileOpenSentry>(*input, lfn_) : nullptr);
      edm::Service<edm::storage::StatisticsSenderService> service;
      if (service.isAvailable()) {
        service->openingFile(lfn(), inputType, -1);
      }
      for (std::vector<std::string>::const_iterator it = fNames.begin(); it != fNames.end(); ++it) {
        try {
          usedFallback_ = (it != fNames.begin());
          std::unique_ptr<char[]> name(gSystem->ExpandPathName(it->c_str()));
          filePtr = std::make_shared<InputFile>(name.get(), "  Initiating request to open file ", inputType);
          break;
        } catch (cms::Exception const& e) {
          if (!skipBadFiles && std::next(it) == fNames.end()) {
            InputFile::reportSkippedFile((*it), logicalFileName());
            errors::ErrorCodes errorCode = usedFallback_ ? errors::FallbackFileOpenError : errors::FileOpenError;
            Exception ex(errorCode, "", e);
            ex.addContext("Calling RootInputFileSequence::initTheFile()");
            std::ostringstream out;
            out << "Input file " << (*it) << " could not be opened.";
            ex.addAdditionalInfo(out.str());
            //report previous exceptions when use other names to open file
            for (auto const& s : exInfo)
              ex.addAdditionalInfo(s);
            //report more information of the earlier file open failures in a log message
            if (not additionalMessage.empty()) {
              edm::LogWarning l("RootInputFileSequence");
              for (auto const& msg : additionalMessage) {
                l << msg << "\n";
              }
            }
            throw ex;
          } else {
            exInfo.push_back("Calling RootInputFileSequence::initTheFile(): fail to open the file with name " + (*it));
            additionalMessage.push_back(fmt::format(
                "Input file {} could not be opened, and fallback was attempted.\nAdditional information:", *it));
            char c = 'a';
            for (auto const& ai : e.additionalInfo()) {
              additionalMessage.push_back(fmt::format("  [{}] {}", c, ai));
              ++c;
            }
          }
        }
      }
    }
    if (filePtr) {
      size_t currentIndexIntoFile = fileIter_ - fileIterBegin_;
      rootFile_ = makeRootFile(filePtr);
      assert(rootFile_);
      if (input) {
        rootFile_->setSignals(&(input->preEventReadFromSourceSignal_), &(input->postEventReadFromSourceSignal_));
      }
      fileIterLastOpened_ = fileIter_;
      setIndexIntoFile(currentIndexIntoFile);
      rootFile_->reportOpened(inputTypeName);
    } else {
      std::string fName = !fNames.empty() ? fNames[0] : "";
      InputFile::reportSkippedFile(fName, logicalFileName());  //0 cause exception?
      if (!skipBadFiles) {
        throw Exception(errors::FileOpenError) << "RootFileSequenceBase::initTheFile(): Input file " << fName
                                               << " was not found or could not be opened.\n";
      }
      LogWarning("RootInputFileSequence")
          << "Input file: " << fName << " was not found or could not be opened, and will be skipped.\n";
    }
  }

  void RootInputFileSequence::closeFile() {
    edm::Service<edm::storage::StatisticsSenderService> service;
    if (rootFile() and service.isAvailable()) {
      service->closedFile(lfn(), usedFallback());
    }
    closeFile_();
  }

  void RootInputFileSequence::setIndexIntoFile(size_t index) {
    indexesIntoFiles_[index] = rootFile()->indexIntoFileSharedPtr();
  }

}  // namespace edm
