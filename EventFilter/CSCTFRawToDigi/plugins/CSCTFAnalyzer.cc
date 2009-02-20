#include "EventFilter/CSCTFRawToDigi/interface/CSCTFAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <strings.h>
#include <errno.h>
#include <iostream>
#include <iomanip>

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

CSCTFAnalyzer::CSCTFAnalyzer(const edm::ParameterSet &conf):edm::EDAnalyzer(){
	mbProducer    = conf.getUntrackedParameter<edm::InputTag>("mbProducer",edm::InputTag("csctfunpacker"));
	lctProducer   = conf.getUntrackedParameter<edm::InputTag>("lctProducer",edm::InputTag("csctfunpacker"));
	trackProducer = conf.getUntrackedParameter<edm::InputTag>("trackProducer",edm::InputTag("csctfunpacker"));
	statusProducer= conf.getUntrackedParameter<edm::InputTag>("statusProducer",edm::InputTag("csctfunpacker"));
}

void CSCTFAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c){
/*	edm::Handle<FEDRawDataCollection> rawdata;
	e.getByLabel("source","",rawdata);

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
		e.getByLabel(statusProducer.label(),statusProducer.instance(),status);
		if( status.isValid() ){
			edm::LogInfo("CSCTFAnalyzer") << "  Unpacking Errors: "<<status->first;
			for(std::vector<L1CSCSPStatusDigi>::const_iterator stat=status->second.begin();
				stat!=status->second.end(); stat++){
				edm::LogInfo("CSCTFAnalyzer") << "   Status: SP in slot "<<stat->slot()<<"  FMM: "<<stat->FMM()<<" SE: 0x"<<std::hex<<stat->SEs()<<" VP: 0x"<<stat->VPs()<<std::dec;
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid L1CSCStatusDigiCollection products found";
	}

	if( mbProducer.label() != "null" ){
		edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs;
		e.getByLabel(mbProducer.label(),mbProducer.instance(),dtStubs);
		if( dtStubs.isValid() ){
			std::vector<csctf::TrackStub> vstubs = dtStubs->get();
			for(std::vector<csctf::TrackStub>::const_iterator stub=vstubs.begin(); stub!=vstubs.end(); stub++){
				edm::LogInfo("CSCTFAnalyzer")<<"   DT data: tbin="<<stub->BX()<<" CSC sector="<<stub->sector()<<" CSC subsector"<<stub->subsector()<<" station="<<stub->station()
					<<" phi="<<stub->phiPacked()<<" phiBend="<<stub->getBend()<<" quality="<<stub->getQuality()<<" mb_bxn="<<stub->cscid();
			}
		}
		else edm::LogInfo("CSCTFAnalyzer")<<"  No valid CSCTriggerContainer<csctf::TrackStub> products found";
	}

	if( lctProducer.label() != "null" ){
		edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
		e.getByLabel(lctProducer.label(),lctProducer.instance(),corrlcts);
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
						"LCT(vp="<<lct->isValid()<<",qual="<<lct->getQuality()<<",wg="<<lct->getKeyWG()<<",strip="<<lct->getStrip()<<")";
				}
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid CSCCorrelatedLCTDigiCollection products found";
	}

	if( trackProducer.label() != "null" ){
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(trackProducer.label(),trackProducer.instance(),tracks);
		if( tracks.isValid() ){
			int nTrk=0;
			for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++,nTrk++){
				int sector = 6*(trk->first.endcap()-1)+trk->first.sector()-1;
				int tbin   = trk->first.BX();
				edm::LogInfo("CSCTFAnalyzer") << "   Track sector: "<<(sector+1)<<"  tbin: "<<tbin<<" "<<
					"TRK(mode="<<((trk->first.ptLUTAddress()>>16)&0xF)<<",eta="<<trk->first.eta_packed()<<",phi="<<trk->first.localPhi()<<")";
			}
		} else edm::LogInfo("CSCTFAnalyzer")<<"  No valid L1CSCTrackCollection products found";
	}
}
