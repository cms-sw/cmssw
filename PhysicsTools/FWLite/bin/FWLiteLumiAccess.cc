#include <iostream>
#include "TFile.h"
#include "TSystem.h"

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"


int main(int argc, char ** argv){
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  // initialize command line parser
  optutl::CommandLineParser parser ("Analyze FWLite Histograms");

  // parse arguments
  parser.parseArguments (argc, argv);
  std::vector<std::string> inputFiles_ = parser.stringVector("inputFiles");
  
  for(unsigned int iFile=0; iFile<inputFiles_.size(); ++iFile){
    // open input file (can be located on castor)
    TFile* inFile = TFile::Open(inputFiles_[iFile].c_str());
    if( inFile ){
      fwlite::Event ev(inFile);
      fwlite::Handle<LumiSummary> summary;
      
      std::cout << "----------- Accessing by event ----------------" << std::endl;
      
      // get run and luminosity blocks from events as well as associated 
      // products. (This works for both ChainEvent and MultiChainEvent.)
      for(ev.toBegin(); !ev.atEnd(); ++ev){
	// get the Luminosity block ID from the event
	std::cout << " Luminosity ID " << ev.getLuminosityBlock().id() << std::endl;
	// get the Run ID from the event
	std::cout <<" Run ID " << ev.getRun().id()<< std::endl;
	// get the Run ID from the luminosity block you got from the event
	std::cout << "Run via lumi " << ev.getLuminosityBlock().getRun().id() << std::endl;
	// get the integrated luminosity (or any luminosity product) from 
	// the event
	summary.getByLabel(ev.getLuminosityBlock(),"lumiProducer");
      }
      
      std::cout << "----------- Accessing by lumi block ----------------" << std::endl;
      
      double lumi_tot = 0.0;
      // loop over luminosity blocks (in analogy to looping over events)
      fwlite::LuminosityBlock ls(inFile);
      for(ls.toBegin(); !ls.atEnd(); ++ls){
	summary.getByLabel(ls,"lumiProducer");
	std::cout  << ls.id() << " Inst.  Luminosity = " << summary->avgInsRecLumi() << std::endl;
	// get the associated run from this lumi
	std::cout << "Run from lumi " << ls.getRun().id() << std::endl;
	// add up the luminosity by lumi block
	lumi_tot += summary->avgInsRecLumi();
      }
      // print the result
      std::cout << "----------------------------------------------------" << std::endl;
      std::cout << "Total luminosity from lumi sections = " << lumi_tot   << std::endl;
      std::cout << "----------------------------------------------------" << std::endl;
      
      std::cout << "----------- Accessing by run ----------------" << std::endl;
      
      // do the same for runs
      fwlite::Run r(inFile);
      for(r.toBegin(); !r.atEnd(); ++r) {
	std::cout << "Run " << r.id() << std::endl;
      }
      // close input file
      inFile->Close();
    }
  }
  return 0;
}
