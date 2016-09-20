#ifndef MillePedeFileConverter_h
#define MillePedeFileConverter_h

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
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MillePedeFileConverter
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer> {
 public:
  explicit MillePedeFileConverter(const edm::ParameterSet&);
  ~MillePedeFileConverter();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override {}
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock&,
                                         const edm::EventSetup&) override final;

  const std::string inputDir_;
  const std::string inputFileName_;
  const std::string fileBlobLabel_;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileConverter);

#endif
