// Original Author:  Broen van Besien
//         Created:  Thu, 19 Mar 2015 18:12:35 GMT

/*
 * This producer will take a given (binary) file and embed it in the output
 * edm/root file.
 *
 * The file is added as a fileblob to a vector of fileblobs that exists on the
 * level of the run.
 * The operation is performed during the endRunProduce phase.
 *
 * The file can later be extracted again by the corresponding module
 * MillePedeFileExtractor.
 *
 * For more info about the possible parameters, type:
 * edmPluginHelp -p MillePedeFileConverter
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/Common/interface/FileBlobCollection.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>

class MillePedeFileConverter : public edm::global::EDProducer<edm::EndLuminosityBlockProducer> {
public:
  explicit MillePedeFileConverter(const edm::ParameterSet&);
  ~MillePedeFileConverter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const final;
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) const final;

  const std::string inputDir_;
  const std::string inputFileName_;
  const std::string fileBlobLabel_;
  const edm::EDPutTokenT<FileBlobCollection> putToken_;
};

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

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileConverter);
