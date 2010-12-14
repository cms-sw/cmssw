#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"


int main(int argc, char* argv[]) 
{
  // ----------------------------------------------------------------------
  // First Part: 
  //
  //  * enable the AutoLibraryLoader 
  //  * book the histograms of interest 
  //  * open the input file
  // ----------------------------------------------------------------------

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();

  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  // Get the python configuration
  PythonProcessDesc builder(argv[1]);
  edm::ParameterSet const& cfg = builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("MuonAnalyzer");

  // now get each parameter
  std::vector<std::string> inputFiles_( cfg.getParameter<std::vector<std::string> >("fileNames") );
  std::string outputFile_( cfg.getParameter<std::string  >("outputFile" ) );
  unsigned int reportAfter_( cfg.getParameter<unsigned int>("reportAfter") );
  int maxEvents_( cfg.getParameter<int>("maxEvents") );
  edm::InputTag muons_ ( cfg.getParameter<edm::InputTag>("muons"  ) );

  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(outputFile_.c_str());
  TFileDirectory dir = fs.mkdir("analyzeBasicPat");
  TH1F* muonPt_  = dir.make<TH1F>("muonPt", "pt",    100,  0.,300.);
  TH1F* muonEta_ = dir.make<TH1F>("muonEta","eta",   100, -3.,  3.);
  TH1F* muonPhi_ = dir.make<TH1F>("muonPhi","phi",   100, -5.,  5.);  
  
  // loop the events
  int ievt=0;  
  for(unsigned int iFile=0; iFile<inputFiles_.size(); ++iFile){
    // open input file (can be located on castor)
    TFile* inFile = TFile::Open(inputFiles_[iFile].c_str());
    if( inFile ){
      // ----------------------------------------------------------------------
      // Second Part: 
      //
      //  * loop the events in the input file 
      //  * receive the collections of interest via fwlite::Handle
      //  * fill the histograms
      //  * after the loop close the input file
      // ----------------------------------------------------------------------
      fwlite::Event ev(inFile);
      for(ev.toBegin(); !ev.atEnd(); ++ev, ++ievt){
	edm::EventBase const & event = ev;
	// break loop if maximal number of events is reached 
	if(maxEvents_>0 ? ievt+1>maxEvents_ : false) break;
	// simple event counter
	if(reportAfter_!=0 ? (ievt>0 && ievt%reportAfter_==0) : false) 
	  std::cout << "  processing event: " << ievt << std::endl;
	
	// Handle to the muon collection
	edm::Handle<std::vector<reco::Muon> > muons;
	event.getByLabel(muons_, muons);
	
	// loop muon collection and fill histograms
	for(unsigned i=0; i<muons->size(); ++i){
	  muonPt_ ->Fill( (*muons)[i].pt()  );
	  muonEta_->Fill( (*muons)[i].eta() );
	  muonPhi_->Fill( (*muons)[i].phi() );
	}
      }  
      // close input file
      inFile->Close();
    }
    // break loop if maximal number of events is reached:
    // this has to be done twice to stop the file loop as well
    if(maxEvents_>0 ? ievt+1>maxEvents_ : false) break;
  }
  return 0;
}
