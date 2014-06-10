#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include <stdlib.h>
#include <string>
#include <cstdlib>

#include "FWCore/Framework/interface/EDAnalyzer.h"

//MC-TESTER header files
#include "Generate.h"
#include "HepMCEvent.H"
#include "Setup.H"

class MCTesterCMS : public edm::EDAnalyzer{
public:
  //
  explicit MCTesterCMS( const edm::ParameterSet& ) ;
  virtual ~MCTesterCMS() {} // no need to delete ROOT stuff
  // as it'll be deleted upon closing TFile
      
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endRun( const edm::Run&, const edm::EventSetup& ) ;
  virtual void endJob() ;

private:
  edm::InputTag hepmcCollection_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
  std::string setupfile_;
}; 

using namespace edm;
using namespace std;

MCTesterCMS::MCTesterCMS( const ParameterSet& pset ):
  hepmcCollection_(pset.getParameter<edm::InputTag>("hepmcCollection"))
  ,setupfile_(pset.getUntrackedParameter<std::string>("setupfile",""))
{
  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
  if(setupfile_==""){
    setupfile_=getenv ("CMSSW_BASE");
    setupfile_+="/src/GeneratorInterface/MCTESTER/test/analyze/SETUP.C";
  }
  std::string Base= getenv ("PWD");
  std::string movecmd="if [ ! -f "+Base+"/SETUP.C ]; then  cp "+setupfile_+" "+Base+"/; fi ";
  system(movecmd.c_str());
}

void MCTesterCMS::beginJob(){
  MC_Initialize();
  return ;
}

void MCTesterCMS::analyze( const Event& e, const EventSetup& ){
  edm::Handle<HepMCProduct> EvtHandle;
  e.getByToken(hepmcCollectionToken_, EvtHandle);  
  
  const HepMC::GenEvent* Evt = EvtHandle->GetEvent();
  HepMCEvent temp_event((*(HepMC::GenEvent*)Evt),false);
  MC_Analyze(&temp_event);
}

void MCTesterCMS::endRun( const edm::Run& r, const edm::EventSetup& ){}

void MCTesterCMS::endJob(){
  MC_Finalize();
}
 
DEFINE_FWK_MODULE(MCTesterCMS);
