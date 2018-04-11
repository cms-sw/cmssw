// F. Cossutti

// Dump in standard ascii format the LHE file stored as string lumi product


// system include files
#include <memory>
#include <string>
#include <sstream>
#include <fstream>
#include <boost/algorithm/string.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class ExternalLHEAsciiDumper : public edm::EDAnalyzer {
public:
  explicit ExternalLHEAsciiDumper(const edm::ParameterSet&);
  ~ExternalLHEAsciiDumper() override;
  
  
private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  edm::InputTag lheProduct_;
  std::string   lheFileName_;

  edm::EDGetTokenT<LHEXMLStringProduct> LHEAsciiToken_;

  // ----------member data ---------------------------
  
};

ExternalLHEAsciiDumper::ExternalLHEAsciiDumper(const edm::ParameterSet& ps):
  lheProduct_( ps.getParameter<edm::InputTag>("lheProduct") ),
  lheFileName_( ps.getParameter<std::string>("lheFileName") )
{
  
  LHEAsciiToken_ = consumes <LHEXMLStringProduct,edm::InRun> (edm::InputTag(lheProduct_));
  
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

  edm::Handle< LHEXMLStringProduct > LHEAscii;
  iRun.getByToken(LHEAsciiToken_,LHEAscii);
  
  const std::vector<std::string>& lheOutputs = LHEAscii->getStrings();

  unsigned int iout = 0;
  
  size_t lastdot = lheFileName_.find_last_of(".");
  std::string basename = lheFileName_.substr(0, lastdot);
  std::string extension = lastdot != std::string::npos ?  lheFileName_.substr(lastdot+1, std::string::npos) : "";

  for (unsigned int i = 0; i < lheOutputs.size(); ++i){
    std::ofstream outfile;
    if (iout == 0)
      outfile.open (lheFileName_.c_str(), std::ofstream::out | std::ofstream::app);
    else {
      std::stringstream fname;
      fname << basename << "_" << iout ;
      if (!extension.empty())
        fname << "." << extension;
      outfile.open (fname.str().c_str(), std::ofstream::out | std::ofstream::app);
    }
    outfile << lheOutputs[i];
    outfile.close();
    ++iout;
  }
  
  for (unsigned int i = 0; i < LHEAscii->getCompressed().size(); ++i){
    std::ofstream outfile;
    if (iout == 0)
      outfile.open (lheFileName_.c_str(), std::ofstream::out | std::ofstream::app);
    else {
      std::stringstream fname;
      fname << basename << "_" << iout ;
      if (!extension.empty())
        fname << "." << extension;
      outfile.open (fname.str().c_str(), std::ofstream::out | std::ofstream::app);
    }
    LHEAscii->writeCompressedContent(outfile,i);
    outfile.close();
    ++iout;
  }

}

DEFINE_FWK_MODULE(ExternalLHEAsciiDumper);
