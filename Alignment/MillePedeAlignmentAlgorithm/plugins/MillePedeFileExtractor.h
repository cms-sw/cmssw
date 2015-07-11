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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MillePedeFileExtractor : public edm::EDAnalyzer {
 public:
  explicit MillePedeFileExtractor(const edm::ParameterSet&);
  ~MillePedeFileExtractor();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) {}

  std::string theOutputDir;
  std::string theOutputFileName;
  std::string theFileBlobModule;
  std::string theFileBlobLabel;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileExtractor);

#endif
