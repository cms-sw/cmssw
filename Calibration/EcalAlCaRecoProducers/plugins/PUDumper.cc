#include "Calibration/EcalAlCaRecoProducers/plugins/PUDumper.h"


//! ctor
PUDumper::PUDumper(const edm::ParameterSet& iConfig)
{
  //  MCPileupTag_ = iConfig.getParameter<edm::InputTag>("MCPileupTag");
	pileupSummaryToken_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter<edm::InputTag>("pileupSummary"));

  // create TTree
  edm::Service<TFileService> fs;
  PUTree_ = fs -> make<TTree>("pileup","pileup");
  
  PUTree_ -> Branch("runNumber",     &runNumber,     "runNumber/I");
  PUTree_ -> Branch("eventNumber",   &eventNumber, "eventNumber/l");
  PUTree_ -> Branch("lumiBlock",     &lumiBlock,     "lumiBlock/I");

  PUTree_ -> Branch("nBX",      &nBX,              "nBX/I");
  PUTree_ -> Branch("BX",       BX_,              "BX[nBX]/I");
  PUTree_ -> Branch("nPUtrue",  &nPUtrue_,    "nPUtrue/I");
  PUTree_ -> Branch("nPUobs",   nPUobs_,      "nPUobs[nBX]/I");
}

// ----------------------------------------------------------------



//! dtor
PUDumper::~PUDumper()
{}

// ----------------------------------------------------------------



//! loop over the reco particles and count leptons
void PUDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get the PU collection
  edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
  if( !iEvent.isRealData() ){
    iEvent.getByToken(pileupSummaryToken_, PupInfo);
  } else return;
  
  
  runNumber = iEvent.id().run();
  eventNumber = iEvent.id().event();
  if( iEvent.isRealData() ) {
    lumiBlock = iEvent.luminosityBlock();
  } else {
    lumiBlock = -1;
  }

  // loop on BX
  nBX=0;
  std::vector<PileupSummaryInfo>::const_iterator PVI;
  nPUtrue_ = PupInfo -> begin()->getTrueNumInteractions();

  for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI){
    BX_[nBX]      = PVI -> getBunchCrossing();
    nPUobs_[nBX]  = PVI -> getPU_NumInteractions();
#ifdef DEBUG    
    std::cout << "PUDumper::runNumber: " << runNumber_
              << "   BX[1]: "      << BX_[1]
              << "   nPUtrue: " << nPUtrue_
              << "   nPUobs[1]: "  << nPUobs_[1]
              << std::endl;
#endif
    nBX++;
  }    
  PUTree_ -> Fill();
}

DEFINE_FWK_MODULE(PUDumper);
