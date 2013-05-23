// F. Cossutti
// $Date: 2011/10/31 15:52:55 $
// $Revision: 1.1 $//

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
#include "FWCore/Framework/interface/Run.h"

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
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
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
ExternalLHEAsciiDumper::endRun(edm::Run const& iRun, edm::EventSetup const&) {

  edm::Handle< std::string > LHEAscii;
  iRun.getByLabel(lheProduct_,LHEAscii);

  const char * theName(lheFileName_.c_str());
  std::ofstream outfile;
  outfile.open (theName, std::ofstream::out | std::ofstream::app);
  outfile << (*LHEAscii);
  outfile.close();

}

void ExternalLHEAsciiDumper::endJob() {
}

DEFINE_FWK_MODULE(ExternalLHEAsciiDumper);
