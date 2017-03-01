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

#include "TSystem.h"

namespace edm {
  class BranchIDListHelper;
  class EventPrincipal;
  class LuminosityBlockPrincipal;
  class RunPrincipal;

  RootInputFileSequence::RootInputFileSequence(
                ParameterSet const& pset,
                InputFileCatalog const& catalog) :
    catalog_(catalog),
    lfn_("unknown"),
    lfnHash_(0U),
    usedFallback_(false),
    findFileForSpecifiedID_(nullptr),
    fileIterBegin_(fileCatalogItems().begin()),
    fileIterEnd_(fileCatalogItems().end()),
    fileIter_(fileIterEnd_),
    fileIterLastOpened_(fileIterEnd_),
    rootFile_(),
    indexesIntoFiles_(fileCatalogItems().size()) {
  }

  std::vector<FileCatalogItem> const&
  RootInputFileSequence::fileCatalogItems() const {
    return catalog_.fileCatalogItems();
  }

  std::shared_ptr<ProductRegistry const>
  RootInputFileSequence::fileProductRegistry() const {
    assert(rootFile());
    return rootFile()->productRegistry();
  }

  std::shared_ptr<BranchIDListHelper const>
  RootInputFileSequence::fileBranchIDListHelper() const {
    assert(rootFile());
    return rootFile()->branchIDListHelper();
  }

  RootInputFileSequence::~RootInputFileSequence() {
  }

  std::shared_ptr<RunAuxiliary>
  RootInputFileSequence::readRunAuxiliary_() {
    assert(rootFile());
    return rootFile()->readRunAuxiliary_();
  }

  std::shared_ptr<LuminosityBlockAuxiliary>
  RootInputFileSequence::readLuminosityBlockAuxiliary_() {
    assert(rootFile());
    return rootFile()->readLuminosityBlockAuxiliary_();
  }

  void
  RootInputFileSequence::readRun_(RunPrincipal& runPrincipal) {
    assert(rootFile());
    rootFile()->readRun_(runPrincipal);
  }

  void
  RootInputFileSequence::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    assert(rootFile());
    rootFile()->readLuminosityBlock_(lumiPrincipal);
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

  void
  RootInputFileSequence::readEvent(EventPrincipal& eventPrincipal) {
    assert(rootFile());
    rootFile()->readEvent(eventPrincipal);
  }

  bool
  RootInputFileSequence::containedInCurrentFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    if(!rootFile()) return false;
    return rootFile()->containsItem(run, lumi, event);
  }

  bool
  RootInputFileSequence::skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash) {
    // Look for item in files not yet opened. We have a hash of the logical file name
    assert(fileNameHash != 0U);
    // If the lookup table is not yet filled in, fill it. 
    if(!findFileForSpecifiedID_) {
      // We use a multimap because there may be hash collisions (Two different LFNs could have the same hash).
      // We map the hash of the LFN to the index into the list of files.
      findFileForSpecifiedID_ =  std::make_unique<std::unordered_multimap<size_t, size_t>>(); // propagate_const<T> has no reset() function
      auto hasher = std::hash<std::string>();
      for(auto fileIter = fileIterBegin_; fileIter != fileIterEnd_; ++fileIter) {
        findFileForSpecifiedID_->insert(std::make_pair(hasher(fileIter->logicalFileName()), fileIter - fileIterBegin_));
      }
    }
    // Look up the logical file name in the table
    auto range = findFileForSpecifiedID_->equal_range(fileNameHash);
    for(auto iter = range.first; iter != range.second; ++iter) {
      // Don't look in files previously opened, because those have already been searched.
      if(!indexesIntoFiles_[iter->second]) {
        setAtFileSequenceNumber(iter->second);
        initFile_(false);
        assert(rootFile());
        bool found = rootFile()->setEntryAtItem(run, lumi, event);
        if(found) {
          return true;
        }
      }
    }
    // Not found
    return false;
  }

  bool
  RootInputFileSequence::skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
    // Look for item in files not yet opened.  We do not have a valid hash of the logical file name.
    for(auto it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
      if(!*it) {
        // File not yet opened.
        setAtFileSequenceNumber(it - indexesIntoFiles_.begin());
        initFile_(false);
        assert(rootFile());
        bool found = rootFile()->setEntryAtItem(run, lumi, event);
        if(found) {
          return true;
        }
      }
    }
    // Not found
    return false;
  }

  bool
  RootInputFileSequence::skipToItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash, bool currentFileFirst) {
    // Attempt to find item in currently open input file.
    bool found = currentFileFirst && rootFile() && rootFile()->setEntryAtItem(run, lumi, event);
    if(!found) {
      // If only one input file, give up now, to save time.
      if(currentFileFirst && rootFile() && indexesIntoFiles_.size() == 1) {
        return false;
      }
      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      for(auto it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if(*it && (*it)->containsItem(run, lumi, event)) {
          // We found it. Close the currently open file, and open the correct one.
          std::vector<FileCatalogItem>::const_iterator currentIter = fileIter_;
          setAtFileSequenceNumber(it - indexesIntoFiles_.begin());
          if(fileIter_ != currentIter) {
            initFile(false);
          }
          // Now get the item from the correct file.
          assert(rootFile());
          found = rootFile()->setEntryAtItem(run, lumi, event);
          assert(found);
          return true;
        }
      }
      return (fileNameHash != 0U && skipToItemInNewFile(run, lumi, event, fileNameHash)) || skipToItemInNewFile(run, lumi, event);
    }
    return true;
  }

  void
  RootInputFileSequence::initTheFile(bool skipBadFiles,
                                    bool deleteIndexIntoFile,
                                    InputSource* input, 
                                    char const* inputTypeName,
                                    InputType inputType) {
    // We are really going to close the open file.

    if(fileIterLastOpened_ != fileIterEnd_) {
      size_t currentIndexIntoFile = fileIterLastOpened_ - fileIterBegin_;
      if(deleteIndexIntoFile) {
        indexesIntoFiles_[currentIndexIntoFile].reset();
      } else {
        if(indexesIntoFiles_[currentIndexIntoFile]) indexesIntoFiles_[currentIndexIntoFile]->inputFileClosed();
      }
      fileIterLastOpened_ = fileIterEnd_;
    }
    closeFile_();

    if(noMoreFiles()) {
      // No files specified
      return;
    }

    // Check if the logical file name was found.
    if(fileName().empty()) {
      // LFN not found in catalog.
      InputFile::reportSkippedFile(fileName(), logicalFileName());
      if(!skipBadFiles) {
        throw cms::Exception("LogicalFileNameNotFound", "RootFileSequenceBase::initTheFile()\n")
          << "Logical file name '" << logicalFileName() << "' was not found in the file catalog.\n"
          << "If you wanted a local file, you forgot the 'file:' prefix\n"
          << "before the file name in your configuration file.\n";
      }
      LogWarning("") << "Input logical file: " << logicalFileName() << " was not found in the catalog, and will be skipped.\n";
      return;
    }

    lfn_ = logicalFileName().empty() ? fileName() : logicalFileName();
    lfnHash_ = std::hash<std::string>()(lfn_);
    usedFallback_ = false;

    // Determine whether we have a fallback URL specified; if so, prepare it;
    // Only valid if it is non-empty and differs from the original filename.
    bool hasFallbackUrl = !fallbackFileName().empty() && fallbackFileName() != fileName();

    std::shared_ptr<InputFile> filePtr;
    std::list<std::string> originalInfo;
    try {
      std::unique_ptr<InputSource::FileOpenSentry> sentry(input ? std::make_unique<InputSource::FileOpenSentry>(*input, lfn_, usedFallback_) : nullptr);
      std::unique_ptr<char[]> name(gSystem->ExpandPathName(fileName().c_str()));;
      filePtr = std::make_shared<InputFile>(name.get(), "  Initiating request to open file ", inputType);
    }
    catch (cms::Exception const& e) {
      if(!skipBadFiles) {
        if(hasFallbackUrl) {
          std::ostringstream out;
          out << e.explainSelf();
          
          std::unique_ptr<char[]> name(gSystem->ExpandPathName(fallbackFileName().c_str()));
          std::string pfn(name.get());
          InputFile::reportFallbackAttempt(pfn, logicalFileName(), out.str());
          originalInfo = e.additionalInfo();
        } else {
          InputFile::reportSkippedFile(fileName(), logicalFileName());
          Exception ex(errors::FileOpenError, "", e);
          ex.addContext("Calling RootFileSequenceBase::initTheFile()");
          std::ostringstream out;
          out << "Input file " << fileName() << " could not be opened.";
          ex.addAdditionalInfo(out.str());
          throw ex;
        }
      }
    }
    if(!filePtr && (hasFallbackUrl)) {
      try {
        usedFallback_ = true;
        std::unique_ptr<InputSource::FileOpenSentry> sentry(input ? std::make_unique<InputSource::FileOpenSentry>(*input, lfn_, usedFallback_) : nullptr);
        std::unique_ptr<char[]> fallbackFullName(gSystem->ExpandPathName(fallbackFileName().c_str()));
        filePtr.reset(new InputFile(fallbackFullName.get(), "  Fallback request to file ", inputType));
      }
      catch (cms::Exception const& e) {
        if(!skipBadFiles) {
          InputFile::reportSkippedFile(fileName(), logicalFileName());
          Exception ex(errors::FallbackFileOpenError, "", e);
          ex.addContext("Calling RootFileSequenceBase::initTheFile()");
          std::ostringstream out;
          out << "Input file " << fileName() << " could not be opened.\n";
          out << "Fallback Input file " << fallbackFileName() << " also could not be opened.";
          if (originalInfo.size()) {
            out << std::endl << "Original exception info is above; fallback exception info is below.";
            ex.addAdditionalInfo(out.str());
            for (auto const & s : originalInfo) {
              ex.addAdditionalInfo(s);
            }
          } else {
            ex.addAdditionalInfo(out.str());
          }
          throw ex;
        }
      }
    }
    if(filePtr) {
      size_t currentIndexIntoFile = fileIter_ - fileIterBegin_;
      rootFile_ = makeRootFile(filePtr);
      if(input) {
        rootFile_->setSignals(&(input->preEventReadFromSourceSignal_), &(input->postEventReadFromSourceSignal_));
      }
      assert(rootFile_);
      fileIterLastOpened_ = fileIter_;
      setIndexIntoFile(currentIndexIntoFile);
      rootFile_->reportOpened(inputTypeName);
    } else {
      InputFile::reportSkippedFile(fileName(), logicalFileName());
      if(!skipBadFiles) {
        throw Exception(errors::FileOpenError) <<
           "RootFileSequenceBase::initTheFile(): Input file " << fileName() << " was not found or could not be opened.\n";
      }
      LogWarning("") << "Input file: " << fileName() << " was not found or could not be opened, and will be skipped.\n";
    }
  }

  void
  RootInputFileSequence::setIndexIntoFile(size_t index) {
   indexesIntoFiles_[index] = rootFile()->indexIntoFileSharedPtr();
  }

}
