// Original Author:  Broen van Besien
//         Created:  Mon, 23 Mar 2015 14:56:15 GMT

#include <memory>

#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeFileExtractor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/Common/interface/FileBlobCollection.h"

MillePedeFileExtractor::MillePedeFileExtractor(const edm::ParameterSet& iConfig)
    : theOutputDir(iConfig.getParameter<std::string>("fileDir")),
      theOutputFileName(iConfig.getParameter<std::string>("outputBinaryFile")),
      theFileBlobModule(iConfig.getParameter<std::string>("fileBlobModule")),
      theFileBlobLabel(iConfig.getParameter<std::string>("fileBlobLabel")) {
  // nothing else in the constructor
}

MillePedeFileExtractor::~MillePedeFileExtractor() {}

void MillePedeFileExtractor::endRun(const edm::Run& iRun,
                                    edm::EventSetup const&) {
  // Getting our hands on the vector of FileBlobs
  edm::Handle<FileBlobCollection> theFileBlobCollection;
  iRun.getByLabel(theFileBlobModule, theFileBlobLabel, theFileBlobCollection);
  if (theFileBlobCollection.isValid()) {
    // Logging the amount of FileBlobs in the vector
    int theVectorSize = theFileBlobCollection->size();
    edm::LogInfo("MillePedeFileActions") << "Root file contains "
                                         << theVectorSize << " FileBlob(s).";
    // Loop over the FileBlobs in the vector, and write them to files:
    for (std::vector<FileBlob>::const_iterator it =
             theFileBlobCollection->begin();
         it != theFileBlobCollection->end(); ++it) {
      // We format the filename with a number, starting from 0 to the size of
      // our vector.
      // For this to work, the outputBinaryFile config parameter must contain a
      // formatting directive for a number, like %04d.
      char theNumberedOutputFileName[200];
      int theNumber = it - theFileBlobCollection->begin();
      sprintf(theNumberedOutputFileName, theOutputFileName.c_str(), theNumber);
      // Log the filename to which we will write...
      edm::LogInfo("MillePedeFileActions")
          << "Writing FileBlob file to file "
          << theOutputDir + theNumberedOutputFileName << ".";
      // ...and perform the writing operation.
      it->write(theOutputDir + theNumberedOutputFileName);
      // Carefull, it seems that when writing to an impossible file, this is
      // swallowed by the FileBlob->write operation and no error is thrown.
    }
  } else {
    edm::LogError("MillePedeFileActions")
        << "Error: The root file does not contain any vector of FileBlob "
           "created by module \"" << theFileBlobModule << "\", with label \""
        << theFileBlobLabel << "\".";
  }
}

// Manage the parameters for the module:
// (Note that this will autogenerate the _cfi.py file.)
void MillePedeFileExtractor::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("fileDir", "")->setComment(
      "Keep the fileDir empty if you want to write to the current "
      "directory.");

  desc.add<std::string>("outputBinaryFile", "milleBinary%04d.dat")->setComment(
      "Base filename of the files that will be created. This must "
      "contain "
      "a placeholder for an index number in the standard C formatting "
      "style, like %04d.");

  desc.add<std::string>("fileBlobModule", "millePedeFileConverter")->setComment(
      "Name of the module that should have generated the blob in the "
      "root file. Make sure you overwrite this, if you have changed "
      "this is the configuration of the MillePedeFileConverter.");

  desc.add<std::string>("fileBlobLabel", "milleBinary.dat")->setComment(
      "It's probably a good idea to keep the label the same as the "
      "original filename(s). See configuration of "
      "MillePedeFileConverter, it should be the same there.");

  descriptions.add("millePedeFileExtractor", desc);
  descriptions.setComment(
      "This is the generic cfi file for the "
      "MillePedeFileExtractor");
}
