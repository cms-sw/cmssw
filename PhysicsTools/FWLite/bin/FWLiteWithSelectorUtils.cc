//#include <memory>
//#include <string>
//#include <vector>
//#include <sstream>
//#include <fstream>
//#include <iostream>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "TStopwatch.h"
#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"



class WSelector : public EventSelector {

public:
  /// constructor
  WSelector(edm::ParameterSet const& params) :
    muonSrc_(params.getParameter<edm::InputTag>("muonSrc")),
    metSrc_ (params.getParameter<edm::InputTag>("metSrc")) 
  {
    double muonPtMin = params.getParameter<double>("muonPtMin");
    double metMin    = params.getParameter<double>("metMin");
    push_back("Muon Pt", muonPtMin );
    push_back("MET"    , metMin    );
    set("Muon Pt"); set("MET");
    wMuon_ = 0; met_ = 0;
    if ( params.exists("cutsToIgnore") ){
      setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );
    }
    retInternal_ = getBitTemplate();
  }
  /// destructor
  virtual ~WSelector() {}
  /// return muon candidate of W boson
  pat::Muon const& wMuon() const { return *wMuon_;}
  /// return MET of W boson
  pat::MET  const& met()   const { return *met_;  }

  /// here is where the selection occurs
  virtual bool operator()( edm::EventBase const & event, pat::strbitset & ret){
    ret.set(false);
    // Handle to the muon collection
    edm::Handle<std::vector<pat::Muon> > muons;    
    // Handle to the MET collection
    edm::Handle<std::vector<pat::MET> > met;
    // get the objects from the event
    bool gotMuons = event.getByLabel(muonSrc_, muons);
    bool gotMET   = event.getByLabel(metSrc_, met   );
    // get the MET, require to be > minimum
    if( gotMET ){
      met_ = &met->at(0);
      if( met_->pt() > cut("MET", double()) || ignoreCut("MET") ) 
	passCut(ret, "MET");
    }
    // get the highest pt muon, require to have pt > minimum
    if( gotMuons ){
      if( !ignoreCut("Muon Pt") ){
	if( muons->size() > 0 ){
	  wMuon_ = &muons->at(0);
	  if( wMuon_->pt() > cut("Muon Pt", double()) || ignoreCut("Muon Pt") ) 
	    passCut(ret, "Muon Pt");
	}
      } 
      else{
	passCut( ret, "Muon Pt");
      }
    }
    setIgnored(ret);
    return (bool)ret;
  }

protected:
  /// muon input
  edm::InputTag muonSrc_;
  /// met input
  edm::InputTag metSrc_;
  /// muon candidate from W boson
  pat::Muon const* wMuon_;
  /// MET from W boson
  pat::MET const* met_;
};


int main(int argc, char* argv[]) 
{
  // define what muon you are using; this is necessary as FWLite is not 
  // capable of reading edm::Views
  using pat::Muon;

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

  // get the python configuration
  PythonProcessDesc builder(argv[1]);
  edm::ParameterSet const& cfg = builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("WBosonAnalyzer");
  
  // now get each parameter
  int maxEvents_( cfg.getParameter<int>("maxEvents") );
  unsigned int outputEvery_( cfg.getParameter<unsigned int>("outputEvery") );
  std::string outputFile_( cfg.getParameter<std::string>("outputFile" ) );
  std::vector<std::string> inputFiles_( cfg.getParameter<std::vector<std::string> >("fileNames") );

  // initialize the W selector
  edm::ParameterSet selection = cfg.getParameter<edm::ParameterSet>("selection");
  WSelector wSelector( selection ); pat::strbitset wSelectorReturns = wSelector.getBitTemplate();
  
  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(outputFile_.c_str());
  TFileDirectory theDir = fs.mkdir("analyzeBasicPat");
  TH1F* muonPt_  = theDir.make<TH1F>("muonPt", "pt",    100,  0.,300.);
  TH1F* muonEta_ = theDir.make<TH1F>("muonEta","eta",   100, -3.,  3.);
  TH1F* muonPhi_ = theDir.make<TH1F>("muonPhi","phi",   100, -5.,  5.);  

  // start a CPU timer
  TStopwatch timer; timer.Start();

  // loop the events
  int ievt=0;  
  unsigned int nEventsAnalyzed = 0;
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
	if(outputEvery_!=0 ? (ievt>0 && ievt%outputEvery_==0) : false) 
	  std::cout << "  processing event: " << ievt << std::endl;
    
	if ( wSelector(event, wSelectorReturns ) ) {
	  pat::Muon const & wMuon = wSelector.wMuon();
	  muonPt_ ->Fill( wMuon.pt()  );
	  muonEta_->Fill( wMuon.eta() );
	  muonPhi_->Fill( wMuon.phi() );
	} 
	++nEventsAnalyzed;
      }  
      // close input file
      inFile->Close();
    }
    // break loop if maximal number of events is reached:
    // this has to be done twice to stop the file loop as well
    if(maxEvents_>0 ? ievt+1>maxEvents_ : false) break;
  }
  // stop CPU timer
  timer.Stop();

  // print selector
  wSelector.print(std::cout);

  // print some timing statistics
  double rtime = timer.RealTime();
  double ctime = timer.CpuTime ();
  // timing printouts
  printf("Analyzed events: %d \n",nEventsAnalyzed);
  printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);
  printf("%4.2f events / RealTime second .\n", (double)nEventsAnalyzed/rtime);
  printf("%4.2f events / CpuTime second .\n", (double)nEventsAnalyzed/ctime);
  return 0;
}
