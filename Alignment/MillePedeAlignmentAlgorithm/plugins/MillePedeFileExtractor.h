#ifndef MillePedeFileExtractor_h
#define MillePedeFileExtractor_h

// Original Author:  Broen van Besien
//         Created:  Mon, 23 Mar 2015 14:56:15 GMT

/*
 * This analyzer will extract all fileblobs in the input edm/root file and
 * write them as files to the file system.
 *
 * It searches for the files in the vector of fileblobs that should exist on
 * the level of the run. Unique filenames for each file are generated based
 * on a formating directive in the output filename parameter.
 *
 * The operation is performed during the endRun phase.
 *
 * The original embedding of the (binary) files in the edm/root files can be
 * done with the corresponding module MillePedeFileConverer.
 *
 * For more info about the possible parameters, type:
 * edmPluginHelp -p MillePedeFileExtractor
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/Common/interface/FileBlobCollection.h"

class MillePedeFileExtractor :
  public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
 public:
  explicit MillePedeFileExtractor(const edm::ParameterSet&);
  ~MillePedeFileExtractor();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
                                    const edm::EventSetup&) override {}
  virtual void endLuminosityBlock(const edm::LuminosityBlock&,
                                  const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}

  bool enoughBinaries() {
    return (nBinaries_ >= maxNumberOfBinaries_) && hasBinaryNumberLimit(); }
  bool hasBinaryNumberLimit() { return maxNumberOfBinaries_ > -1; }

  const std::string outputDir_;
  const std::string outputFileName_;

  edm::EDGetTokenT<FileBlobCollection> fileBlobToken_;

  const int maxNumberOfBinaries_;
  int nBinaries_{0};
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileExtractor);

#endif
