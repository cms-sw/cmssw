#ifndef MillePedeFileConverter_h
#define MillePedeFileConverter_h

// Original Author:  Broen van Besien
//         Created:  Mon, 23 Mar 2015 14:56:15 GMT

/*
 * This analyzer... blahblahblha... BVB: TODO
 *
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
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) {}

  std::string theOutputDir;
  std::string theOutputFileName;
  std::string theFileBlobModule;
  std::string theFileBlobLabel;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileExtractor);

#endif
