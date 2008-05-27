#include <L1Trigger/HardwareValidation/plugins/MuonCandProducerMon.h>

MuonCandProducerMon::MuonCandProducerMon(const edm::ParameterSet& pset) {

  verbose_ = pset.getUntrackedParameter<int>("VerboseFlag",0);
  
  CSCinput_ = pset.getUntrackedParameter<edm::InputTag>("CSCinput",(edm::InputTag)("csctfdigis"));
  DTinput_ = pset.getUntrackedParameter<edm::InputTag>("DTinput",(edm::InputTag)("dttfdigis"));
  
  // Create a dummy pset for Csc Pt LUTs
  edm::ParameterSet ptLUTset;
  ptLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  ptLUTset.addUntrackedParameter<bool>("Binary",   false);
  ptLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  ptLUT_ = new CSCTFPtLUT(ptLUTset);

  produces<std::vector<L1MuRegionalCand> >("CSC");
  produces<std::vector<L1MuRegionalCand> >("DT");
}

MuonCandProducerMon::~MuonCandProducerMon(){}

void MuonCandProducerMon::beginJob(const edm::EventSetup&) {}

void MuonCandProducerMon::endJob() {
  delete ptLUT_;
}

void
MuonCandProducerMon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<L1CSCTrackCollection> CSCtracks; 
  iEvent.getByLabel(CSCinput_, CSCtracks);

  edm::Handle<L1MuDTTrackContainer> DTtracks; 
  iEvent.getByLabel(DTinput_, DTtracks);

  std::auto_ptr<std::vector<L1MuRegionalCand> > 
    csc_product(new std::vector<L1MuRegionalCand>);

  std::auto_ptr<std::vector<L1MuRegionalCand> > 
    dt_product(new std::vector<L1MuRegionalCand>);

  
  if(!CSCtracks.isValid()) {

    csc_product->push_back(L1MuRegionalCand());

  } else {
  
    typedef L1CSCTrackCollection::const_iterator ctcIt;
    
    for(ctcIt tcit=CSCtracks->begin(); tcit!=CSCtracks->end(); tcit++) {
      
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
      csc_product->push_back(cand);
    }    
  }

  if(!DTtracks.isValid()) {

    dt_product->push_back(L1MuRegionalCand());

  } else {
    
    typedef std::vector<L1MuDTTrackCand>::const_iterator ctcIt;

    std::vector<L1MuDTTrackCand> *dttc = DTtracks->getContainer();

    for(ctcIt it=dttc->begin(); it!=dttc->end(); it++) {
      dt_product->push_back(L1MuRegionalCand(*it)); 
    }
  }
  
  iEvent.put(csc_product,"CSC");
  iEvent.put(dt_product,"DT");
}

