#include "CSCTFanalyzer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

CSCTFanalyzer::CSCTFanalyzer(edm::ParameterSet const& pset):edm::EDAnalyzer(){
	verbose           = pset.getUntrackedParameter<unsigned  int>("verbose",0);
	dataTrackProducer = pset.getUntrackedParameter<edm::InputTag>("dataTrackProducer",edm::InputTag("csctfDigis"));
	emulTrackProducer = pset.getUntrackedParameter<edm::InputTag>("emulTrackProducer",edm::InputTag("csctfTrackDigis"));
	lctProducer       = pset.getUntrackedParameter<edm::InputTag>("lctProducer",edm::InputTag("csctfDigis"));
	file = new TFile("qwe.root","RECREATE");
	tree = new TTree("dy","QWE");
	tree->Branch("nDataMuons", &nDataMuons, "nDataMuons/I");
	tree->Branch("dphi1", &dphi1, "dphi1/I");
	tree->Branch("dphi2", &dphi2, "dphi2/I");
	tree->Branch("dphi3", &dphi3, "dphi3/I");
	tree->Branch("deta1", &deta1, "deta1/I");
	tree->Branch("deta2", &deta2, "deta2/I");
	tree->Branch("deta3", &deta3, "deta3/I");
	tree->Branch("dpt1",  &dpt1,  "dpt1/I");
	tree->Branch("dpt2",  &dpt2,  "dpt2/I");
	tree->Branch("dpt3",  &dpt3,  "dpt3/I");
	tree->Branch("dch1",  &dch1,  "dch1/I");
	tree->Branch("dch2",  &dch2,  "dch2/I");
	tree->Branch("dch3",  &dch3,  "dch3/I");
	tree->Branch("dbx1",  &dbx1,  "dbx1/I");
	tree->Branch("dbx2",  &dbx2,  "dbx2/I");
	tree->Branch("dbx3",  &dbx3,  "dbx3/I");

	tree->Branch("nEmulMuons", &nEmulMuons, "nEmulMuons/I");
	tree->Branch("ephi1", &ephi1, "ephi1/I");
	tree->Branch("ephi2", &ephi2, "ephi2/I");
	tree->Branch("ephi3", &ephi3, "ephi3/I");
	tree->Branch("eeta1", &eeta1, "eeta1/I");
	tree->Branch("eeta2", &eeta2, "eeta2/I");
	tree->Branch("eeta3", &eeta3, "eeta3/I");
	tree->Branch("ept1",  &ept1,  "ept1/I");
	tree->Branch("ept2",  &ept2,  "ept2/I");
	tree->Branch("ept3",  &ept3,  "ept3/I");
	tree->Branch("ech1",  &ech1,  "ech1/I");
	tree->Branch("ech2",  &ech2,  "ech2/I");
	tree->Branch("ech3",  &ech3,  "ech3/I");
	tree->Branch("ebx1",  &ebx1,  "ebx1/I");
	tree->Branch("ebx2",  &ebx2,  "ebx2/I");
	tree->Branch("ebx3",  &ebx3,  "ebx3/I");

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

	nDataMuons = 0; nEmulMuons = 0;
	dphi1=-1; deta1=-1; dpt1=-1; dch1=-1, dbx1=-10;
	dphi2=-1; deta2=-1; dpt2=-1; dch2=-1, dbx2=-10;
	dphi3=-1; deta3=-1; dpt3=-1; dch3=-1, dbx3=-10;
	ephi1=-1; eeta1=-1; ept1=-1; ech1=-1, ebx1=-10;
	ephi2=-1; eeta2=-1; ept2=-1; ech2=-1, ebx2=-10;
	ephi3=-1; eeta3=-1; ept3=-1; ech3=-1, ebx3=-10;

	if( dataTrackProducer.label() != "null" ){
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(dataTrackProducer.label(),dataTrackProducer.instance(),tracks);
		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++){
			switch(nDataMuons){
				case 0:
					dphi1 = trk->first.localPhi();
					deta1 = trk->first.eta_packed();
					dpt1  = trk->first.pt_packed();
					dch1  = trk->first.charge_packed();
					dbx1  = trk->first.BX();
				break;
				case 1:
					dphi2 = trk->first.localPhi();
					deta2 = trk->first.eta_packed();
					dpt2  = trk->first.pt_packed();
					dch2  = trk->first.charge_packed();
					dbx2  = trk->first.BX();
				break;
				case 2:
					dphi3 = trk->first.localPhi();
					deta3 = trk->first.eta_packed();
					dpt3  = trk->first.pt_packed();
					dch3  = trk->first.charge_packed();
					dbx3  = trk->first.BX();
				break;
				default: break;
			}
			if( (verbose&2)==2 )
				std::cout<<"Data: TRK in endcap="<<trk->first.endcap()<<" sector="<<trk->first.sector()<<" bx="<<trk->first.BX()
					<<" (rank="<<trk->first.rank()
					<<" localPhi="<<trk->first.localPhi()
					<<" me1D="<<trk->first.me1ID()
					<<" me2D="<<trk->first.me2ID()
					<<" me3D="<<trk->first.me3ID()
					<<" me4D="<<trk->first.me4ID()
					<<" mb1D="<<trk->first.mb1ID()
					<<")"<<std::endl;
			nDataMuons++;
		}
	}

	if( emulTrackProducer.label() != "null" ){
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(emulTrackProducer.label(),emulTrackProducer.instance(),tracks);
		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++){
			switch(nEmulMuons){
				case 0:
					ephi1 = trk->first.localPhi();
					eeta1 = trk->first.eta_packed();
					ept1  = trk->first.pt_packed();
					ech1  = trk->first.charge_packed();
					ebx1  = trk->first.BX();
				break;
				case 1:
					ephi2 = trk->first.localPhi();
					eeta2 = trk->first.eta_packed();
					ept2  = trk->first.pt_packed();
					ech2  = trk->first.charge_packed();
					ebx2  = trk->first.BX();
				break;
				case 2:
					ephi3 = trk->first.localPhi();
					eeta3 = trk->first.eta_packed();
					ept3  = trk->first.pt_packed();
					ech3  = trk->first.charge_packed();
					ebx3  = trk->first.BX();
				break;
				default: break;
			}
			if( (verbose&2)==2 )
				std::cout<<"Emulator: TRK in endcap="<<trk->first.endcap()<<" sector="<<trk->first.sector()<<" bx="<<trk->first.BX()
					<<" (rank="<<trk->first.rank()
					<<" localPhi="<<trk->first.localPhi()
					<<" me1D="<<trk->first.me1ID()
					<<" me2D="<<trk->first.me2ID()
					<<" me3D="<<trk->first.me3ID()
					<<" me4D="<<trk->first.me4ID()
					<<" mb1D="<<trk->first.mb1ID()
					<<")"<<std::endl;
			nEmulMuons++;
		}
	}

	tree->Fill();
}
