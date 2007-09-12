#include "EventFilter/CSCTFRawToDigi/interface/CSCTFAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <strings.h>
#include <errno.h>
#include <iostream>

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

CSCTFAnalyzer::CSCTFAnalyzer(const edm::ParameterSet &conf):edm::EDAnalyzer(){
	lctProducer   = conf.getUntrackedParameter<edm::InputTag>("lctProducer",edm::InputTag("csctfunpacker","MuonCSCTFCorrelatedLCTDigi"));
	trackProducer = conf.getUntrackedParameter<edm::InputTag>("trackProducer",edm::InputTag("csctfunpacker","MuonL1CSCTrackCollection"));
}

void CSCTFAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c){
	edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
	e.getByLabel(lctProducer.label(),lctProducer.instance(),corrlcts);

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

			edm::LogInfo("CSCTFAnalyzer|print") << "  Front data   endcap: "<<(*csc).first.endcap()<<"  station: "<<(station+1)<<"  sector: "<<(sector+1)<<"  subSector: "<<subSector<<"  tbin: "<<tbin<<"  cscId: "<<(cscId+1)<<"  fpga: "<<(fpga+1)<<" "<<
				"LCT(qual="<<lct->getQuality()<<",wg="<<lct->getKeyWG()<<",strip="<<lct->getStrip()<<")";
		}
	}


	edm::Handle<L1CSCTrackCollection> tracks;
	e.getByLabel(trackProducer.label(),trackProducer.instance(),tracks);

	int nTrk=0;
	for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++,nTrk++){
		int sector = 6*(trk->first.endcap()-1)+trk->first.sector()-1;
		int tbin   = trk->first.BX();
		edm::LogInfo("CSCTFAnalyzer|print") << "   Track sector: "<<(sector+1)<<"  tbin: "<<tbin<<" "<<
			"TRK(mode="<<((trk->first.ptLUTAddress()>>16)&0xF)<<",eta="<<trk->first.eta_packed()<<",phi="<<trk->first.phi_packed()<<")";
	}
}
