// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
//
// class decleration
//

class AlpgenExtractor : public edm::EDAnalyzer {
public:
  explicit AlpgenExtractor(const edm::ParameterSet&);
  ~AlpgenExtractor() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void writeHeader(const std::vector<LHERunInfoProduct::Header>::const_iterator, const std::string);
  // ----------member data ---------------------------
  std::string unwParFile_;
  std::string wgtFile_;
  std::string parFile_;
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
      parFile_(iConfig.getUntrackedParameter<std::string>("parFile")) {
  //now do what ever initialization is needed
}

AlpgenExtractor::~AlpgenExtractor() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each run    ------------
void AlpgenExtractor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::Handle<LHERunInfoProduct> runInfo;
  iRun.getByLabel("source", runInfo);
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

// ------------ method called to for each event  ------------
void AlpgenExtractor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just before starting event loop  ------------

// ------------ method called once each job just after ending the event loop  ------------
void AlpgenExtractor::endJob() {}

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
