// F. Cossutti
// $Date: $
// $Revision: $//

// Dump in standard ascii format the LHE file stored as string lumi product


// system include files
#include <memory>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class ExternalLHEAsciiDumper : public edm::EDAnalyzer {
public:
  explicit ExternalLHEAsciiDumper(const edm::ParameterSet&);
  ~ExternalLHEAsciiDumper();
  
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endJob();

  edm::InputTag lheProduct_;
  std::string   lheFileName_;

  // ----------member data ---------------------------
  
};

ExternalLHEAsciiDumper::ExternalLHEAsciiDumper(const edm::ParameterSet& ps):
  lheProduct_( ps.getParameter<edm::InputTag>("lheProduct") ),
  lheFileName_( ps.getParameter<std::string>("lheFileName") )
{
  
  return;
  
}

ExternalLHEAsciiDumper::~ExternalLHEAsciiDumper()
{
}

void
ExternalLHEAsciiDumper::analyze(const edm::Event&, const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------

void
ExternalLHEAsciiDumper::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {

  edm::Handle< std::string > LHEAscii;
  iLumi.getByLabel(lheProduct_,LHEAscii);

  edm::LogInfo("LumiDump") << "Dumping lumi section = " << iLumi.id();

  const char * theName(lheFileName_.c_str());
  std::ofstream outfile;
  outfile.open (theName, std::ofstream::out | std::ofstream::app);
  outfile << (*LHEAscii);
  outfile.close();

}

void ExternalLHEAsciiDumper::endJob() {
}

DEFINE_FWK_MODULE(ExternalLHEAsciiDumper);
