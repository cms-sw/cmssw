// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
//
// class decleration
//

class AlpgenExtractor : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit AlpgenExtractor(const edm::ParameterSet&);
  ~AlpgenExtractor() override = default;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void writeHeader(const std::vector<LHERunInfoProduct::Header>::const_iterator, const std::string);
  // ----------member data ---------------------------
  const std::string unwParFile_;
  const std::string wgtFile_;
  const std::string parFile_;
  const edm::EDGetTokenT<LHERunInfoProduct> tokenLHERun_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AlpgenExtractor::AlpgenExtractor(const edm::ParameterSet& iConfig)
    : unwParFile_(iConfig.getUntrackedParameter<std::string>("unwParFile")),
      wgtFile_(iConfig.getUntrackedParameter<std::string>("wgtFile")),
      parFile_(iConfig.getUntrackedParameter<std::string>("parFile")),
      tokenLHERun_(consumes<LHERunInfoProduct, edm::InRun>(edm::InputTag("source"))) {
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each run    ------------
void AlpgenExtractor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  const edm::Handle<LHERunInfoProduct>& runInfo = iRun.getHandle(tokenLHERun_);
  std::cout << "Found " << runInfo->headers_size() << " headers." << std::endl;

  std::vector<LHERunInfoProduct::Header>::const_iterator headers = runInfo->headers_begin();
  // Skip the first header -initial comments.
  // BOTH the order and the increment of the headers variable are crucial here -
  // check the AlpgenSource code for more information.
  // Write the _unw.par file.
  headers++;
  writeHeader(headers, unwParFile_);
  // Write the .wgt file.
  headers++;
  writeHeader(headers, wgtFile_);
  // Write the .par file.
  headers++;
  writeHeader(headers, parFile_);
}

void AlpgenExtractor::writeHeader(std::vector<LHERunInfoProduct::Header>::const_iterator header,
                                  const std::string filename) {
  std::ofstream outfile;
  outfile.open(filename.c_str());
  for (LHERunInfoProduct::Header::const_iterator i = header->begin(); i != header->end(); ++i) {
    outfile << *i;
  }
  outfile.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlpgenExtractor);
