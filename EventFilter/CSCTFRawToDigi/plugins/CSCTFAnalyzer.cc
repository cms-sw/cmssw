#include "EventFilter/CSCTFRawToDigi/interface/CSCTFAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <strings.h>
#include <cerrno>
#include <iostream>
#include <iomanip>



CSCTFAnalyzer::CSCTFAnalyzer(const edm::ParameterSet &conf):edm::EDAnalyzer(){
	mbProducer    = conf.getUntrackedParameter<edm::InputTag>("mbProducer",edm::InputTag("csctfunpacker"));
	lctProducer   = conf.getUntrackedParameter<edm::InputTag>("lctProducer",edm::InputTag("csctfunpacker"));
	trackProducer = conf.getUntrackedParameter<edm::InputTag>("trackProducer",edm::InputTag("csctfunpacker"));
	statusProducer= conf.getUntrackedParameter<edm::InputTag>("statusProducer",edm::InputTag("csctfunpacker"));
	file = new TFile("ewq.root","RECREATE");
	tree = new TTree("dy","QWE");
	tree->Branch("dtPhi_1_plus",&dtPhi[0][0],"dtPhi_1_plus/I");
	tree->Branch("dtPhi_2_plus",&dtPhi[1][0],"dtPhi_2_plus/I");
	tree->Branch("dtPhi_3_plus",&dtPhi[2][0],"dtPhi_3_plus/I");
	tree->Branch("dtPhi_4_plus",&dtPhi[3][0],"dtPhi_4_plus/I");
	tree->Branch("dtPhi_5_plus",&dtPhi[4][0],"dtPhi_5_plus/I");
	tree->Branch("dtPhi_6_plus",&dtPhi[5][0],"dtPhi_6_plus/I");
	tree->Branch("dtPhi_7_plus",&dtPhi[6][0],"dtPhi_7_plus/I");
	tree->Branch("dtPhi_8_plus",&dtPhi[7][0],"dtPhi_8_plus/I");
	tree->Branch("dtPhi_9_plus",&dtPhi[8][0],"dtPhi_9_plus/I");
	tree->Branch("dtPhi_10_plus",&dtPhi[9][0],"dtPhi_10_plus/I");
	tree->Branch("dtPhi_11_plus",&dtPhi[10][0],"dtPhi_11_plus/I");
	tree->Branch("dtPhi_12_plus",&dtPhi[11][0],"dtPhi_12_plus/I");
	tree->Branch("dtPhi_1_minus",&dtPhi[0][1],"dtPhi_1_minus/I");
	tree->Branch("dtPhi_2_minus",&dtPhi[1][1],"dtPhi_2_minus/I");
	tree->Branch("dtPhi_3_minus",&dtPhi[2][1],"dtPhi_3_minus/I");
	tree->Branch("dtPhi_4_minus",&dtPhi[3][1],"dtPhi_4_minus/I");
	tree->Branch("dtPhi_5_minus",&dtPhi[4][1],"dtPhi_5_minus/I");
	tree->Branch("dtPhi_6_minus",&dtPhi[5][1],"dtPhi_6_minus/I");
	tree->Branch("dtPhi_7_minus",&dtPhi[6][1],"dtPhi_7_minus/I");
	tree->Branch("dtPhi_8_minus",&dtPhi[7][1],"dtPhi_8_minus/I");
	tree->Branch("dtPhi_9_minus",&dtPhi[8][1],"dtPhi_9_minus/I");
	tree->Branch("dtPhi_10_minus",&dtPhi[9][1],"dtPhi_10_minus/I");
	tree->Branch("dtPhi_11_minus",&dtPhi[10][1],"dtPhi_11_minus/I");
	tree->Branch("dtPhi_12_minus",&dtPhi[11][1],"dtPhi_12_minus/I");

	
	L1CSCS_Tok = consumes<L1CSCStatusDigiCollection>( edm::InputTag(statusProducer.label(),statusProducer.instance() ) ); 
	CSCTC_Tok = consumes<CSCTriggerContainer<csctf::TrackStub> >( edm::InputTag(mbProducer.label(),mbProducer.instance()) ); 
	CSCCDC_Tok = consumes<CSCCorrelatedLCTDigiCollection>( edm::InputTag(lctProducer.label(),lctProducer.instance()) );  
	L1CST_Tok = consumes<L1CSCTrackCollection>( edm::InputTag(trackProducer.label(),trackProducer.instance()) );  

}

void CSCTFAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c){
/*	edm::Handle<FEDRawDataCollection> rawdata;
	e.getByToken("source","",rawdata);

	const FEDRawData& fedData = rawdata->FEDData(750);
	if( fedData.size()==0 ) return;
	unsigned short *data = (unsigned short *)fedData.data();
	unsigned int    size = fedData.size()/2;
	std::cout<<"New event:"<<std::endl;
	for(unsigned i=0; i<size/4; i++)
		std::cout<<std::hex<<" "<<std::setw(6)<<data[i*4+0]<<" "<<std::setw(6)<<data[i*4+1]<<" "<<std::setw(6)<<data[i*4+2]<<" "<<std::setw(6)<<data[i*4+3]<<std::dec<<std::endl;
	std::cout<<"End of event"<<std::endl;
	return;
*/
	if( statusProducer.label() != "null" ){
		edm::Handle<L1CSCStatusDigiCollection> status;
		e.getByToken(L1CSCS_Tok ,status);
		if( status.isValid() ){
			edm::LogInfo("CSCTFAnalyzer") << "  Unpacking Errors: "<<status->first;
			for(std::vector<L1CSCSPStatusDigi>::const_iterator stat=status->second.begin();
				stat!=status->second.end(); stat++){
				//edm::LogInfo("CSCTFAnalyzer") << "   Status: SP in slot "<<stat->slot()<<"  FMM: "<<stat->FMM()<<" SE: 0x"<<std::hex<<stat->SEs()<<" VP: 0x"<<stat->VPs()<<std::dec;
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid L1CSCStatusDigiCollection products found";
	}

	if( mbProducer.label() != "null" ){
		bzero(dtPhi,sizeof(dtPhi));
		edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs;
		e.getByToken(CSCTC_Tok ,dtStubs);
		if( dtStubs.isValid() ){
			std::vector<csctf::TrackStub> vstubs = dtStubs->get();
			std::cout<<"DT size="<<vstubs.end()-vstubs.begin()<<std::endl;
			for(std::vector<csctf::TrackStub>::const_iterator stub=vstubs.begin(); stub!=vstubs.end(); stub++){
				int dtSector =(stub->sector()-1)*2 + stub->subsector()-1;
				int dtEndcap = stub->endcap()-1;
				if( dtSector>=0 && dtSector<12 && dtEndcap>=0 && dtEndcap<2 ){
					dtPhi[dtSector][dtEndcap] = stub->phiPacked();
				} else {
					edm::LogInfo("CSCTFAnalyzer: DT digi are out of range: ")<<" dtSector="<<dtSector<<" dtEndcap="<<dtEndcap;
				}
				edm::LogInfo("CSCTFAnalyzer")<<"   DT data: tbin="<<stub->BX()<<" CSC sector="<<stub->sector()<<" CSC subsector="<<stub->subsector()<<" station="<<stub->station()<<" endcap="<<stub->endcap()
						<<" phi="<<stub->phiPacked()<<" phiBend="<<stub->getBend()<<" quality="<<stub->getQuality()<<" mb_bxn="<<stub->cscid();
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid CSCTriggerContainer<csctf::TrackStub> products found";
		tree->Fill();
	}

	if( lctProducer.label() != "null" ){
		edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
		e.getByToken(CSCCDC_Tok ,corrlcts);
		if( corrlcts.isValid() ){
			for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++){
				int lctId=0;
				CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
				for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++){
					int station = (*csc).first.station()-1;
					int cscId   = (*csc).first.triggerCscId()-1;
					int sector  = (*csc).first.triggerSector()-1;// + ( (*csc).first.endcap()==1 ? 0 : 6 );
					int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
					int tbin    = lct->getBX();
					int fpga    = ( subSector ? subSector-1 : station+1 );
					// If Det Id is within range
					if( sector<0 || sector>11 || station<0 || station>3 || cscId<0 || cscId>8 || lctId<0 || lctId>1){
						edm::LogInfo("CSCTFAnalyzer: CSC digi are out of range: ");
						continue;
					}
					edm::LogInfo("CSCTFAnalyzer")<<"   Front data   endcap: "<<(*csc).first.endcap()<<"  station: "<<(station+1)<<"  sector: "<<(sector+1)<<"  subSector: "<<subSector<<"  tbin: "<<tbin<<"  cscId: "<<(cscId+1)<<"  fpga: "<<(fpga+1)<<" "<<
						"LCT(vp="<<lct->isValid()<<",qual="<<lct->getQuality()<<",wg="<<lct->getKeyWG()<<",strip="<<lct->getStrip()<<",link="<<lct->getMPCLink()<<")";
				}
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid CSCCorrelatedLCTDigiCollection products found";
	}

	if( trackProducer.label() != "null" ){
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByToken(L1CST_Tok ,tracks);
		if( tracks.isValid() ){
			int nTrk=0;
			for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++,nTrk++){
				int sector = 6*(trk->first.endcap()-1)+trk->first.sector()-1;
				int tbin   = trk->first.BX();
				edm::LogInfo("CSCTFAnalyzer") << "   Track sector: "<<(sector+1)<<"  tbin: "<<tbin<<" "<<
					"TRK(mode="<<((trk->first.ptLUTAddress()>>16)&0xF)<<",eta="<<trk->first.eta_packed()<<",phi="<<trk->first.localPhi()<<") IDs:"
					<<" me1D="<<trk->first.me1ID()<<" t1="<<trk->first.me1Tbin()
					<<" me2D="<<trk->first.me2ID()<<" t2="<<trk->first.me2Tbin()
					<<" me3D="<<trk->first.me3ID()<<" t3="<<trk->first.me3Tbin()
					<<" me4D="<<trk->first.me4ID()<<" t4="<<trk->first.me4Tbin()
					<<" mb1D="<<trk->first.mb1ID()<<" tb="<<trk->first.mb1Tbin();

			for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=trk->second.begin(); csc!=trk->second.end(); csc++){
				int lctId=0;
				CSCCorrelatedLCTDigiCollection::Range range1 = trk->second.get((*csc).first);
				for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++){
					int station = (*csc).first.station()-1;
					int cscId   = (*csc).first.triggerCscId()-1;
					int sector  = (*csc).first.triggerSector()-1;// + ( (*csc).first.endcap()==1 ? 0 : 6 );
					int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
					int tbin    = lct->getBX();
					int fpga    = ( subSector ? subSector-1 : station+1 );
					// If Det Id is within range
					if( sector<0 || sector>11 || station<0 || station>3 || cscId<0 || cscId>8 || lctId<0 || lctId>1){
						edm::LogInfo("CSCTFAnalyzer: Digi are out of range: ");
						continue;
					}
					if( lct->getQuality() < 100 )
						edm::LogInfo("CSCTFAnalyzer")<<"       Linked LCT: "<<(*csc).first.endcap()<<"  station: "<<(station+1)<<"  sector: "<<(sector+1)<<"  subSector: "<<subSector<<"  tbin: "<<tbin
							<<"  cscId: "<<(cscId+1)<<"  fpga: "<<(fpga+1)<<" LCT(vp="<<lct->isValid()<<",qual="<<lct->getQuality()<<",wg="<<lct->getKeyWG()<<",strip="<<lct->getStrip()<<")";
					else
						edm::LogInfo("CSCTFAnalyzer")<<"       Linked MB stub: "<<(*csc).first.endcap()<<"  sector: "<<(sector+1)<<"  subSector: "<<subSector<<"  tbin: "<<tbin
							<<" MB(vp="<<lct->isValid()<<",qual="<<(lct->getQuality()-100)<<",cal="<<lct->getKeyWG()<<",flag="<<lct->getStrip()<<",bc0="<<lct->getPattern()<<",phiBend="<<lct->getBend()
							<<",tbin="<<lct->getBX()<<",id="<<lct->getMPCLink()<<",bx0="<<lct->getBX0()<<",se="<<lct->getSyncErr()<<",bxn="<<lct->getCSCID()<<",phi="<<lct->getTrknmb()<<")";
				}
			}
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid L1CSCTrackCollection products found";
	}
}
