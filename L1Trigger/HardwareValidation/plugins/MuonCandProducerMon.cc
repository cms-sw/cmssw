#include <L1Trigger/HardwareValidation/plugins/MuonCandProducerMon.h>

MuonCandProducerMon::MuonCandProducerMon(const edm::ParameterSet& pset) {

  verbose_ = pset.getUntrackedParameter<int>("VerboseFlag",0);
  
  CSCinput_ = pset.getUntrackedParameter<edm::InputTag>("CSCinput",(edm::InputTag)("csctfunpacker"));
  
  // Create a dummy pset for Pt LUTs
  edm::ParameterSet ptLUTset;
  ptLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  ptLUTset.addUntrackedParameter<bool>("Binary",   false);
  ptLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  ptLUT_ = new CSCTFPtLUT(ptLUTset);

  produces<std::vector<L1MuRegionalCand> >("CSC");
}

MuonCandProducerMon::~MuonCandProducerMon(){}

void MuonCandProducerMon::beginJob(const edm::EventSetup&) {}

void MuonCandProducerMon::endJob() {
  delete ptLUT_;
}

void
MuonCandProducerMon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<L1CSCTrackCollection> tracks; 
  iEvent.getByLabel(CSCinput_, tracks);

  std::auto_ptr<std::vector<L1MuRegionalCand> > 
    cand_product(new std::vector<L1MuRegionalCand>);
  
  if(!tracks.isValid()) {
    cand_product->push_back(L1MuRegionalCand());
    iEvent.put(cand_product,"CSC");
    return;
  }
  

  typedef L1CSCTrackCollection::const_iterator ctcIt;
  
  for(ctcIt tcit=tracks->begin(); tcit!=tracks->end(); tcit++) {
    
    L1MuRegionalCand cand(tcit->first.getDataWord(), tcit->first.bx());
    
    // set pt value
    ptadd thePtAddress(tcit->first.ptLUTAddress());
    ptdat thePtData = ptLUT_->Pt(thePtAddress);
    const unsigned int rank = 
      ( thePtAddress.track_fr ? thePtData.front_rank : thePtData.rear_rank );
    unsigned int quality = 0;
    unsigned int pt = 0;
    csc::L1Track::decodeRank(rank, pt, quality);
    cand.setQualityPacked(quality & 0x3);
    cand.setPtPacked(pt & 0x1f);
    cand_product->push_back(cand);
  }    
  
  iEvent.put(cand_product,"CSC");
  
}

