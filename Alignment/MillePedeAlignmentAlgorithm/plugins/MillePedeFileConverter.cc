// Original Author:  Broen van Besien
//         Created:  Thu, 19 Mar 2015 18:12:35 GMT

#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeFileConverter.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <fstream>

MillePedeFileConverter::MillePedeFileConverter(const edm::ParameterSet& iConfig)
    : inputDir_(iConfig.getParameter<std::string>("fileDir")),
      inputFileName_(iConfig.getParameter<std::string>("inputBinaryFile")),
      fileBlobLabel_(iConfig.getParameter<std::string>("fileBlobLabel")),
      putToken_{produces<FileBlobCollection, edm::Transition::EndLuminosityBlock>(fileBlobLabel_)} {}

MillePedeFileConverter::~MillePedeFileConverter() {}

void MillePedeFileConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {}

void MillePedeFileConverter::globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                                             const edm::EventSetup& iSetup) const {
  edm::LogInfo("MillePedeFileActions") << "Inserting all data from file " << inputDir_ + inputFileName_
                                       << " as a FileBlob to the lumi, using label \"" << fileBlobLabel_ << "\".";
  // Preparing the FileBlobCollection:
  FileBlobCollection fileBlobCollection;

  // Creating the FileBlob:
  // (The FileBlob will signal problems with the file itself.)
  FileBlob fileBlob{inputDir_ + inputFileName_, true};

  if (fileBlob.size() > 0) {  // skip if no data or FileBlob file not found
    // Adding the FileBlob to the lumi:
    fileBlobCollection.addFileBlob(fileBlob);
  }
  iLumi.emplace(putToken_, std::move(fileBlobCollection));
}

// Manage the parameters for the module:
// (Note that this will autogenerate the _cfi.py file.)
void MillePedeFileConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("fileDir", "")
      ->setComment(
          "Keep the fileDir empty if you want to write to the current "
          "directory. If you use it, it should end with a slash.");

  desc.add<std::string>("inputBinaryFile", "milleBinary.dat")
      ->setComment("Filename of the file created by Mille in the AlignmentProducer");

  desc.add<std::string>("fileBlobLabel", "milleBinary.dat")
      ->setComment(
          "It's probably a good idea to keep the label the same as the "
          "original filename(s). See configuration of "
          "MillePedeFileExtractor, it should be the same there.");

  descriptions.add("millePedeFileConverter", desc);
  descriptions.setComment(
      "This is the generic cfi file for the "
      "MillePedeFileConverter");
}
