#include "CSCTFanalyzer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

CSCTFanalyzer::CSCTFanalyzer(edm::ParameterSet const& pset):edm::EDAnalyzer(){
	verbose       = pset.getUntrackedParameter<unsigned  int>("verbose",0);
	trackProducer = pset.getUntrackedParameter<edm::InputTag>("trackProducer",edm::InputTag("csctfTrackDigis"));
	lctProducer   = pset.getUntrackedParameter<edm::InputTag>("cscTriggerPrimitiveDigis",edm::InputTag("cscTriggerPrimitiveDigis"));
	file = new TFile("qwe.root","RECREATE");
	tree = new TTree("dy","QWE");
	tree->Branch("nMuons", &nMuons, "nMuons/I");
	tree->Branch("phi1", &phi1, "phi1/I");
	tree->Branch("phi2", &phi2, "phi2/I");
	tree->Branch("eta1", &eta1, "eta1/I");
	tree->Branch("eta2", &eta2, "eta2/I");
	tree->Branch("pt1",  &pt1,  "pt1/I");
	tree->Branch("pt2",  &pt2,  "pt2/I");
	tree->Branch("ch1",  &ch1,  "ch1/I");
	tree->Branch("ch2",  &ch2,  "ch2/I");
	tree->Branch("bx1",  &bx1,  "bx1/I");
	tree->Branch("bx2",  &bx2,  "bx2/I");
}

void CSCTFanalyzer::endJob(void){
	tree->Write();
	file->Write();
	file->Close();
}

void CSCTFanalyzer::analyze(edm::Event const& e, edm::EventSetup const& es){
	if( lctProducer.label() != "null" ){
		edm::Handle<CSCCorrelatedLCTDigiCollection> LCTs;
		e.getByLabel(lctProducer.label(),lctProducer.instance(), LCTs);

		for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=LCTs.product()->begin(); csc!=LCTs.product()->end(); csc++){
			int lctId=0;

			CSCCorrelatedLCTDigiCollection::Range range1 = LCTs.product()->get((*csc).first);
			for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++){
				int station = (*csc).first.station()-1;
				int cscId   = (*csc).first.triggerCscId()-1;
				int sector  = (*csc).first.triggerSector()-1 + ( (*csc).first.endcap()==1 ? 0 : 6 );
				int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
				int tbin    = lct->getBX();
				int fpga    = ( subSector ? subSector-1 : station+1 );
				if((verbose&1)==1)
					std::cout<<"LCT in station="<<(station+1)<<" sector="<<(sector+1)<<" cscId="<<(cscId+1)<<" bx="<<tbin<<std::endl;
			}
		}
	}

	nMuons = 0;
	phi1=-1; eta1=-1; pt1=-1; ch1=-1, bx1=-10;
	phi2=-1; eta2=-1; pt2=-1; ch2=-1, bx2=-10;

	if( trackProducer.label() != "null" ){
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(trackProducer.label(),trackProducer.instance(),tracks);
		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++){
			if(nMuons==0){
				phi1 = trk->first.localPhi();
				eta1 = trk->first.eta_packed();
				pt1  = trk->first.pt_packed();
				ch1  = trk->first.charge_packed();
				bx1  = trk->first.BX();
			} else {
				phi2 = trk->first.localPhi();
				eta2 = trk->first.eta_packed();
				pt2  = trk->first.pt_packed();
				ch2  = trk->first.charge_packed();
				bx2  = trk->first.BX();
			}
			if( (verbose&2)==2 )
				std::cout<<"Trk in endcap="<<trk->first.endcap()<<" sector="<<trk->first.sector()<<" bx="<<trk->first.BX()
					<<" (rank="<<trk->first.rank()
					<<" localPhi="<<trk->first.localPhi()
					<<" me1D="<<trk->first.me1ID()
					<<" me2D="<<trk->first.me2ID()
					<<" me3D="<<trk->first.me3ID()
					<<" me4D="<<trk->first.me4ID()
					<<" mb1D="<<trk->first.mb1ID()
					<<")"<<std::endl;
			nMuons++;
		}
	}
	tree->Fill();

}
