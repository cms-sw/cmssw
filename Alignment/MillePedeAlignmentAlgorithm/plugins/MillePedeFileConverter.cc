// Original Author:  Broen van Besien
//         Created:  Thu, 19 Mar 2015 18:12:35 GMT

#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeFileConverter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Common/interface/FileBlobCollection.h"

#include <memory>
#include <fstream>

MillePedeFileConverter::MillePedeFileConverter(const edm::ParameterSet& iConfig)
    : theInputDir(iConfig.getParameter<std::string>("fileDir")),
      theInputFileName(iConfig.getParameter<std::string>("inputBinaryFile")),
      theFileBlobLabel(iConfig.getParameter<std::string>("fileBlobLabel")) {
  // We define what this producer produces: A FileBlobCollection
  produces<FileBlobCollection, edm::InRun>(theFileBlobLabel);
}

MillePedeFileConverter::~MillePedeFileConverter() {}

void MillePedeFileConverter::endRunProduce(edm::Run& iRun,
                                           const edm::EventSetup& iSetup) {
  edm::LogInfo("MillePedeFileActions")
      << "Inserting all data from file " << theInputDir + theInputFileName
      << " as a FileBlob to the run, using label \"" << theFileBlobLabel
      << "\".";
  // Preparing the FileBlobCollection:
  std::unique_ptr<FileBlobCollection> theFileBlobCollection(
      new FileBlobCollection());
  FileBlob theFileBlob;
  try {
    // Creating the FileBlob:
    // (The FileBlob will signal problems with the file itself.)
    theFileBlob = FileBlob(theInputDir + theInputFileName, true);
  }
  catch (...) {
    // When creation of the FileBlob fails:
    edm::LogError("MillePedeFileActions")
        << "Error: No FileBlob could be created from the file \""
        << theInputDir + theInputFileName << "\".";
    throw;
  }
  if (theFileBlob.size() > 0) {
    // Adding the FileBlob to the run:
    theFileBlobCollection->addFileBlob(theFileBlob);
    iRun.put(std::move(theFileBlobCollection), theFileBlobLabel);
  }
}

// Manage the parameters for the module:
// (Note that this will autogenerate the _cfi.py file.)
void MillePedeFileConverter::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("fileDir", "")->setComment(
      "Keep the fileDir empty if you want to write to the current "
      "directory. If you use it, it should end with a slash.");

  desc.add<std::string>("inputBinaryFile", "milleBinary.dat")->setComment(
      "Filename of the file created by Mille in the AlignmentProducer");

  desc.add<std::string>("fileBlobLabel", "milleBinary.dat")->setComment(
      "It's probably a good idea to keep the label the same as the "
      "original filename(s). See configuration of "
      "MillePedeFileExtractor, it should be the same there.");

  descriptions.add("millePedeFileConverter", desc);
  descriptions.setComment(
      "This is the generic cfi file for the "
      "MillePedeFileConverter");
}
