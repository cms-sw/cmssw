// Original Author:  Broen van Besien
//         Created:  Mon, 23 Mar 2015 14:56:15 GMT

#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeFileExtractor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include <zlib.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <TSystem.h>

MillePedeFileExtractor::MillePedeFileExtractor(const edm::ParameterSet& iConfig)
    : outputDir_(iConfig.getParameter<std::string>("fileDir")),
      outputFileName_(iConfig.getParameter<std::string>("outputBinaryFile")),
      maxNumberOfBinaries_(iConfig.getParameter<int>("maxNumberOfBinaries")) {
  auto fileBlobInputTag = iConfig.getParameter<edm::InputTag>("fileBlobInputTag");
  fileBlobToken_ = consumes<FileBlobCollection, edm::BranchType::InLumi>(fileBlobInputTag);
  if (hasBinaryNumberLimit()) {
    edm::LogInfo("MillePedeFileActions") << "Limiting the number of extracted binary files to " << maxNumberOfBinaries_;
  }
}

MillePedeFileExtractor::~MillePedeFileExtractor() {}

void MillePedeFileExtractor::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup&) {
  if (enoughBinaries())
    return;

  // Create output directory if not available
  if (!outputDir_.empty()) {
    std::string command = "mkdir -p " + outputDir_;
    int shellReturn = gSystem->Exec(command.c_str());
    edm::LogInfo("MillePedeFileActions") << "@SUB=MillePedeFileExtractor::endLuminosityBlock"
                                         << "Command returns " << shellReturn;
  }

  // Getting our hands on the vector of FileBlobs
  edm::Handle<FileBlobCollection> fileBlobCollection;
  iLumi.getByToken(fileBlobToken_, fileBlobCollection);
  if (fileBlobCollection.isValid()) {
    // Logging the amount of FileBlobs in the vector
    edm::LogInfo("MillePedeFileActions") << "Root file contains " << fileBlobCollection->size() << " FileBlob(s).";
    // Loop over the FileBlobs in the vector, and write them to files:
    for (const auto& blob : *fileBlobCollection) {
      if (enoughBinaries())
        break;
      // We format the filename with a number, starting from 0 to the size of
      // our vector.
      // For this to work, the outputBinaryFile config parameter must contain a
      // formatting directive for a number, like %04d.
      char theNumberedOutputFileName[200];
      sprintf(theNumberedOutputFileName, outputFileName_.c_str(), nBinaries_);

      // Log the filename to which we will write...
      edm::LogInfo("MillePedeFileActions")
          << "Writing FileBlob file to file " << outputDir_ + theNumberedOutputFileName << ".";

      // ...and perform the writing operation.
      writeGzipped(blob, outputDir_ + theNumberedOutputFileName);

      ++nBinaries_;
    }
  } else {
    edm::LogError("MillePedeFileActions") << "Error: The root file does not contain any vector of FileBlob.";
  }
}

void MillePedeFileExtractor::writeGzipped(const FileBlob& blob, const std::string& fileName) {
  // - use zlib directly to avoid boost dependencies for this simple task
  // - zlib and gzip compression differ -> get uncompressed blob first
  auto uncompressedBlob = blob.getUncompressedBlob();
  gzFile fp = gzopen(fileName.c_str(), "wb");
  if (fp == nullptr) {
    edm::LogError("MillePedeFileActions") << "Problem while opening gzipped file '" << fileName << "'.";
  }
  auto nBytes = gzwrite(fp, &uncompressedBlob->front(), uncompressedBlob->size());
  if (nBytes == 0 || nBytes != static_cast<decltype(nBytes)>(uncompressedBlob->size())) {
    edm::LogError("MillePedeFileActions") << "Problem while writing FileBlob to gzipped file '" << fileName << "'.";
  }
  auto zerr = gzclose(fp);
  if (zerr != 0) {
    edm::LogError("MillePedeFileActions") << "Problem while closing gzipped file '" << fileName << "'.";
  }
}

// Manage the parameters for the module:
// (Note that this will autogenerate the _cfi.py file.)
void MillePedeFileExtractor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("fileDir", "")
      ->setComment(
          "Keep the fileDir empty if you want to write to the current "
          "directory.");

  desc.add<std::string>("outputBinaryFile", "milleBinary%04d.dat")
      ->setComment(
          "Base filename of the files that will be created. This must "
          "contain "
          "a placeholder for an index number in the standard C formatting "
          "style, like %04d.");

  desc.add<edm::InputTag>("fileBlobInputTag", edm::InputTag("millePedeFileConverter", ""))
      ->setComment(
          "Name of the module that should have generated the blob in the "
          "root file. Make sure you overwrite this, if you have changed "
          "this is the configuration of the MillePedeFileConverter.");

  desc.add<int>("maxNumberOfBinaries", 1000)
      ->setComment(
          "Number of binaries to be extracted from the input files. "
          "Use a negative value to apply no limit.");

  descriptions.add("millePedeFileExtractor", desc);
  descriptions.setComment(
      "This is the generic cfi file for the "
      "MillePedeFileExtractor");
}
