#ifndef MillePedeFileConverter_h
#define MillePedeFileConverter_h

// Original Author:  Broen van Besien
//         Created:  Thu, 19 Mar 2015 18:12:35 GMT

/*
 * This producer... blahblahblha... BVB: TODO
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MillePedeFileConverter
    : public edm::one::EDProducer<edm::EndRunProducer> {
 public:
  explicit MillePedeFileConverter(const edm::ParameterSet&);
  ~MillePedeFileConverter();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override {}
  virtual void endRunProduce(edm::Run& run,
                             const edm::EventSetup& iSetup) override final;

  std::string theInputDir;
  std::string theInputFileName;
  std::string theFileBlobLabel;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeFileConverter);

#endif
