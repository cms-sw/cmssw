#include "EventFilter/CSCTFRawToDigi/interface/CSCTFPacker.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"

#include <strings.h>
#include <cerrno>
#include <iostream>
#include <cstdio>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"


CSCTFPacker::CSCTFPacker(const edm::ParameterSet &conf):edm::one::EDProducer<>(){
	// "Readout" configuration
	zeroSuppression = conf.getParameter<bool>("zeroSuppression");
	nTBINs          = conf.getParameter<int> ("nTBINs");
	activeSectors   = conf.getParameter<int> ("activeSectors");

	// Configuration that controls CMSSW specific stuff
	putBufferToEvent       = conf.getParameter<bool>("putBufferToEvent");
	std::string outputFile = conf.getParameter<std::string>("outputFile");
	lctProducer            = conf.getParameter<edm::InputTag>("lctProducer");
	mbProducer             = conf.getParameter<edm::InputTag>("mbProducer");
	trackProducer          = conf.getParameter<edm::InputTag>("trackProducer");

	// Swap: if(swapME1strips && me1b && !zplus) strip = 65 - strip; // 1-64 -> 64-1 :
	swapME1strips = conf.getParameter<bool>("swapME1strips");

	file = nullptr;
	if( outputFile.length() && (file = fopen(outputFile.c_str(),"wt"))==nullptr )
		throw cms::Exception("OutputFile ")<<"CSCTFPacker: cannot open output file (errno="<<errno<<"). Try outputFile=\"\"";

	// BX window bounds in CMSSW:
	m_minBX = conf.getParameter<int>("MinBX"); //3
	m_maxBX = conf.getParameter<int>("MaxBX"); //9

	// Finds central LCT BX
	// assumes window is odd number of bins
	central_lct_bx = (m_maxBX + m_minBX)/2;
	// Find central SP BX
	// assumes window is odd number of bins
	central_sp_bx = int(nTBINs/2);

	produces<FEDRawDataCollection>("CSCTFRawData");

	CSCTC_Tok = consumes<CSCTriggerContainer<csctf::TrackStub> >( edm::InputTag(mbProducer.label(),mbProducer.instance()) ); 
	CSCCDC_Tok = consumes<CSCCorrelatedLCTDigiCollection>( edm::InputTag(lctProducer.label(),lctProducer.instance()) );  
	L1CSCTr_Tok = consumes<L1CSCTrackCollection>( edm::InputTag(trackProducer.label(),trackProducer.instance()) );  


}

CSCTFPacker::~CSCTFPacker(void){
	if( file ) fclose(file);
}

void CSCTFPacker::produce(edm::Event& e, const edm::EventSetup& c){
	edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
	e.getByToken(CSCCDC_Tok ,corrlcts);

	CSCSP_MEblock meDataRecord[12][7][5][9][2]; // LCT in sector X, tbin Y, station Z, csc W, and lct I
	bzero(&meDataRecord,sizeof(meDataRecord));
	CSCSPRecord meDataHeader[12][7]; // Data Block Header for sector X and tbin Y
	bzero(&meDataHeader,sizeof(meDataHeader));

	for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++){
		CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
		int lctId=0;
		for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++){
			int station = (*csc).first.station()-1;
			int cscId   = (*csc).first.triggerCscId()-1;
			int sector  = (*csc).first.triggerSector()-1 + ( (*csc).first.endcap()==1 ? 0 : 6 );
			int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
			int tbin = lct->getBX() - (central_lct_bx-central_sp_bx); // Shift back to hardware BX window definition
			if( tbin>6 || tbin<0 ){
				edm::LogError("CSCTFPacker|produce")<<" LCT's BX="<<tbin<<" is out of 0-6 window";
				continue;
			}
			int fpga    = ( subSector ? subSector-1 : station+1 );
///std::cout<<"Front data station: "<<station<<"  sector: "<<sector<<"  subSector: "<<subSector<<"  tbin: "<<tbin<<"  cscId: "<<cscId<<"  fpga: "<<fpga<<" LCT_qual="<<lct->getQuality()<<" LCT_strip="<<lct->getStrip()<<" LCT_wire="<<lct->getKeyWG()<<std::endl;

			// If Det Id is within range
			if( sector<0 || sector>11 || station<0 || station>3 || cscId<0 || cscId>8 || lctId<0 || lctId>1){
				edm::LogInfo("CSCTFPacker: CSC digi are out of range: ")<<"sector="<<sector<<", station="<<station<<", cscId="<<cscId<<", lctId="<<lctId;
				continue;
			}

			meDataRecord[sector][tbin][fpga][cscId][lctId].clct_pattern_number = lct->getPattern();
			meDataRecord[sector][tbin][fpga][cscId][lctId].quality_            = lct->getQuality();
			meDataRecord[sector][tbin][fpga][cscId][lctId].wire_group_id       = lct->getKeyWG();

			meDataRecord[sector][tbin][fpga][cscId][lctId].clct_pattern_id     = (swapME1strips && cscId<3 && station==0 && (*csc).first.endcap()==2 && lct->getStrip()<65 ? 65 - lct->getStrip() : lct->getStrip() );
			meDataRecord[sector][tbin][fpga][cscId][lctId].csc_id              = (*csc).first.triggerCscId();
			meDataRecord[sector][tbin][fpga][cscId][lctId].left_right          = lct->getBend();
			meDataRecord[sector][tbin][fpga][cscId][lctId].bx0_                = 0; //?;
			meDataRecord[sector][tbin][fpga][cscId][lctId].bc0_                = 0; //?;

			meDataRecord[sector][tbin][fpga][cscId][lctId].me_bxn              = 0; //?
			meDataRecord[sector][tbin][fpga][cscId][lctId].receiver_status_er1 = 0; // dummy
			meDataRecord[sector][tbin][fpga][cscId][lctId].receiver_status_dv1 = 0; // dummy
			meDataRecord[sector][tbin][fpga][cscId][lctId].aligment_fifo_full  = 0; // dummy

			meDataRecord[sector][tbin][fpga][cscId][lctId].link_id             = lct->getMPCLink();
			meDataRecord[sector][tbin][fpga][cscId][lctId].mpc_id              = 0; // Join with above?
			meDataRecord[sector][tbin][fpga][cscId][lctId].err_prop_cnt        = 0; // dummy
			meDataRecord[sector][tbin][fpga][cscId][lctId].receiver_status_er2 = 0; // dummy
			meDataRecord[sector][tbin][fpga][cscId][lctId].receiver_status_dv2 = 0; // dummy
			meDataRecord[sector][tbin][fpga][cscId][lctId].aligment_fifo_empty = 0; // dummy

			if( lct->isValid() ){
				switch( (meDataHeader[sector][tbin].vp_bits>>(fpga*3)) & 0x7 ){
					case 0x0: meDataHeader[sector][tbin].vp_bits |= (0x1 << (fpga*3)); break;
					case 0x1: meDataHeader[sector][tbin].vp_bits |= (0x3 << (fpga*3)); break;
					case 0x3: meDataHeader[sector][tbin].vp_bits |= (0x7 << (fpga*3)); break;
					default :
						edm::LogInfo("CSCTFPacker: more than 3 LCTs from a single MPC in one BX!!!");
						continue;
						break;
				}
				meDataRecord[sector][tbin][fpga][cscId][lctId].valid_pattern = 1; // for later use
			}
			meDataHeader[sector][tbin].vq_a = 0; // no digi yet?
			meDataHeader[sector][tbin].vq_b = 0; // no digi yet?
			meDataHeader[sector][tbin].se_bits = 0; // dummy
			meDataHeader[sector][tbin].sm_bits = 0; // dummy
			meDataHeader[sector][tbin].af_bits = 0; // dummy
			meDataHeader[sector][tbin].bx_bits = 0;//(lct->getBX()&??<< (fpga*3));

			meDataHeader[sector][tbin].spare_1 = 0; // for later use
		}
	}

	CSCSP_MBblock mbDataRecord[12][2][7]; // LCT in sector X, subsector Z, tbin Y
	bzero(&mbDataRecord,sizeof(mbDataRecord));
	edm::Handle< CSCTriggerContainer<csctf::TrackStub> > barrelStubs;
	if( mbProducer.label() != "null" ){
		e.getByToken(CSCTC_Tok ,barrelStubs);
		if( barrelStubs.isValid() ){
			std::vector<csctf::TrackStub> stubs = barrelStubs.product()->get();
			for(std::vector<csctf::TrackStub>::const_iterator dt=stubs.begin(); dt!=stubs.end(); dt++){
				int sector    = dt->sector()-1 + ( dt->endcap()==1 ? 0 : 6 );
				int subSector = dt->subsector()-1;
				int tbin      = dt->getBX() - (central_lct_bx-central_sp_bx); // Shift back to hardware BX window definition
				if( tbin<0 || tbin>6 || sector<0 || sector>11 || subSector<0 || subSector>11 ){
					edm::LogInfo("CSCTFPacker: CSC DT digi are out of range: ")<<" sector="<<sector<<"  subSector="<<subSector<<"  tbin="<<tbin;
					continue;
				}
				mbDataRecord[sector][subSector][tbin].quality_  = dt->getQuality();
				mbDataRecord[sector][subSector][tbin].phi_bend_ = dt->getBend();
				mbDataRecord[sector][subSector][tbin].flag_     = dt->getStrip();
				mbDataRecord[sector][subSector][tbin].cal_      = dt->getKeyWG();
				mbDataRecord[sector][subSector][tbin].phi_      = dt->phiPacked();
				mbDataRecord[sector][subSector][tbin].bxn1_     =(dt->getBX0()>>1)&0x1;
				mbDataRecord[sector][subSector][tbin].bxn0_     = dt->getBX0()&0x1;
				mbDataRecord[sector][subSector][tbin].bc0_      = dt->getPattern();
				mbDataRecord[sector][subSector][tbin].mb_bxn_   = dt->getCSCID();
				switch(subSector){
					case 0: meDataHeader[sector][tbin].vq_a = 1; break;
					case 1: meDataHeader[sector][tbin].vq_b = 1; break;
					default: edm::LogInfo("CSCTFPacker: subSector=")<<subSector; break;
				}
				mbDataRecord[sector][subSector][tbin].id_       = dt->getMPCLink(); // for later use
			}
		}
	}

	CSCSP_SPblock spDataRecord[12][7][3]; // Up to 3 tracks in sector X and tbin Y
	bzero(&spDataRecord,sizeof(spDataRecord));
	int nTrk[12][7];
	bzero(&nTrk,sizeof(nTrk));

	edm::Handle<L1CSCTrackCollection> tracks;
	if( trackProducer.label() != "null" ){
		e.getByToken(L1CSCTr_Tok ,tracks);

		for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk!=tracks->end(); trk++){
			int sector = 6*(trk->first.endcap()-1)+trk->first.sector()-1;
			int tbin   = trk->first.BX() + central_sp_bx; // Shift back to hardware BX window definition
//std::cout<<"Track["<<nTrk[sector][tbin]<<"]  sector: "<<sector<<" tbin: "<<tbin<<std::endl;
			if( tbin>6 || tbin<0 ){
				edm::LogError("CSCTFPacker|analyze")<<" Track's BX="<<tbin<<" is out of 0-6 window";
				continue;
			}
			if( sector<0 || sector>11 ){
				edm::LogError("CSCTFPacker|analyze")<<" Track's sector="<<sector<<" is out of range";
				continue;
			}
			spDataRecord[sector][tbin][nTrk[sector][tbin]].phi_       = trk->first.localPhi();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].sign_      =(trk->first.ptLUTAddress()>>20)&0x1;
			spDataRecord[sector][tbin][nTrk[sector][tbin]].front_rear = trk->first.front_rear();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].charge_    = trk->first.charge_packed(); //
			spDataRecord[sector][tbin][nTrk[sector][tbin]].eta_       = trk->first.eta_packed();

			spDataRecord[sector][tbin][nTrk[sector][tbin]].halo_      = trk->first.finehalo_packed();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].se         = 0; // dummy
			spDataRecord[sector][tbin][nTrk[sector][tbin]].deltaPhi12_= trk->first.ptLUTAddress()&0xFF;
			spDataRecord[sector][tbin][nTrk[sector][tbin]].deltaPhi23_=(trk->first.ptLUTAddress()>>8)&0xF;
			spDataRecord[sector][tbin][nTrk[sector][tbin]].bxn0_      = 0; //dummy
			spDataRecord[sector][tbin][nTrk[sector][tbin]].bc0_       = 0; //dummy

			spDataRecord[sector][tbin][nTrk[sector][tbin]].me1_id     = trk->first.me1ID();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me2_id     = trk->first.me2ID();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me3_id     = trk->first.me3ID();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me4_id     = trk->first.me4ID();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].mb_id      = trk->first.mb1ID();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].ms_id      = 0; // don't care winner()

			// Warning, digi copying was broken for <= CMSSW_3_8_x! The 5 lines of code below will give problems there:
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me1_tbin   = trk->first.me1Tbin();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me2_tbin   = trk->first.me2Tbin();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me3_tbin   = trk->first.me3Tbin();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].me4_tbin   = trk->first.me4Tbin();
			spDataRecord[sector][tbin][nTrk[sector][tbin]].mb_tbin    = trk->first.mb1Tbin();
			// As the MB stubs are not saved in simulation, we want to introduce an artificial ids
			if( trk->first.mb1ID() ){
				int subSector = (trk->first.mb1ID() - 1)%2;
				int MBtbin    = tbin - spDataRecord[sector][tbin][nTrk[sector][tbin]].mb_tbin;
				if( subSector<0 || subSector>1 || MBtbin<0 || MBtbin>7 || !mbDataRecord[sector][subSector][MBtbin].id_ )
					spDataRecord[sector][tbin][nTrk[sector][tbin]].mb_id = ( subSector ? 6 : 5 );
			}
			spDataRecord[sector][tbin][nTrk[sector][tbin]].id_ = nTrk[sector][tbin]+1; // for later use

			nTrk[sector][tbin]++;
			switch(nTrk[sector][tbin]){
				case 1: meDataHeader[sector][tbin].mode1 = (trk->first.ptLUTAddress()>>16)&0xF; break;
				case 2: meDataHeader[sector][tbin].mode2 = (trk->first.ptLUTAddress()>>16)&0xF; break;
				case 3: meDataHeader[sector][tbin].mode3 = (trk->first.ptLUTAddress()>>16)&0xF; break;
				default:
					edm::LogInfo("More than 3 tracks from one SP in the BX");
					continue;
					break;
			}
		}
	}

	CSCSPHeader  header;
	bzero(&header,sizeof(header));

	header.header_mark_1 = 0x9;
	header.header_mark_2 = 0x9;
	header.header_mark_3 = 0x9;
	header.header_mark_4 = 0x9;

	header.header_mark_5 = 0xA;
	header.header_mark_6 = 0xA;
	header.header_mark_7 = 0xA;
	header.header_mark_8 = 0xA;

	header.csr_dfc  = nTBINs;
	header.csr_dfc |= ( zeroSuppression ? 0x8 : 0x0 );
	header.csr_dfc |= 0x7F0; // All FPGAs are active
	header.skip     = 0;
	header.sp_ersv  = 2; // Format version with block of counters

	CSCSPCounters counters;
	bzero(&counters,sizeof(counters));

	CSCSPTrailer trailer;
	bzero(&trailer,sizeof(trailer));

	trailer.trailer_mark_1 = 0xF;
	trailer.trailer_mark_2 = 0xF;
	trailer.trailer_mark_3 = 0x7;
	trailer.trailer_mark_4 = 0xF;
	trailer.trailer_mark_5 = 0xF;
	trailer.trailer_mark_6 = 0xF;
	trailer.trailer_mark_7 = 0xE;
	trailer.trailer_mark_8 = 0xE;
	trailer.trailer_mark_9 = 0xE;
	trailer.trailer_mark_10= 0xE;

	unsigned short spDDUrecord[700*12], *pos=spDDUrecord; // max length
	bzero(&spDDUrecord,sizeof(spDDUrecord));
	int eventNumber = e.id().event();
	*pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0xFFFF&eventNumber; *pos++ = 0x5000|(eventNumber>>16);
	*pos++ = 0x0000; *pos++ = 0x8000; *pos++ = 0x0001; *pos++ = 0x8000;
	*pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000;

	for(int sector=0; sector<12; sector++){
		if( !(activeSectors & (1<<sector)) ) continue;
		header.sp_trigger_sector = sector+1;
		memcpy(pos,&header,16);
		pos+=8;
		memcpy(pos,&counters,8);
		pos+=4;

		for(int tbin=0; tbin<nTBINs; tbin++){
				memcpy(pos,&meDataHeader[sector][tbin],16);
				pos+=8;
				for(int fpga=0; fpga<5; fpga++){
					int nLCTs=0;
					for(int link=0; link<3; link++){
						for(int cscId=0; cscId<9; cscId++)
							for(int lctId=0; lctId<2; lctId++)
								// Only 3 LCT per BX from the same fpga are allowed (to be valid):
								if( meDataRecord[sector][tbin][fpga][cscId][lctId].valid_pattern
									&& meDataRecord[sector][tbin][fpga][cscId][lctId].link_id==link+1 ){
									memcpy(pos,&meDataRecord[sector][tbin][fpga][cscId][lctId],8);
									pos+=4;
									nLCTs++;
								}
					}
					if( !zeroSuppression ) pos += 4*(3-nLCTs);
				}
				for(int subSector=0; subSector<2; subSector++)
					if( !zeroSuppression || (subSector==0 && meDataHeader[sector][tbin].vq_a) || (subSector==1 && meDataHeader[sector][tbin].vq_b) ){
						memcpy(pos,&mbDataRecord[sector][subSector][tbin],8);
						pos+=4;
					}
				for(int trk=0; trk<3; trk++){
					if( !zeroSuppression || spDataRecord[sector][tbin][trk].id_ ){
						memcpy(pos,&spDataRecord[sector][tbin][trk],8);
						pos+=4;
					}
				}
		}
		memcpy(pos,&trailer,16);
		pos+=8;
	}

	*pos++ = 0x8000; *pos++ = 0x8000; *pos++ = 0xFFFF; *pos++ = 0x8000;
	*pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000;
	*pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000; *pos++ = 0x0000;

	if( putBufferToEvent ){
		auto data = std::make_unique<FEDRawDataCollection>();
		FEDRawData& fedRawData = data->FEDData((unsigned int)FEDNumbering::MINCSCTFFEDID);
		fedRawData.resize((pos-spDDUrecord)*sizeof(unsigned short));
		std::copy((unsigned char*)spDDUrecord,(unsigned char*)pos,fedRawData.data());
		FEDHeader  csctfFEDHeader (fedRawData.data());
		csctfFEDHeader.set(fedRawData.data(), 0, e.id().event(), 0, FEDNumbering::MINCSCTFFEDID);
		FEDTrailer csctfFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
		csctfFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8), fedRawData.size()/8, evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);
		e.put(std::move(data),"CSCTFRawData");
	}

	if(file) fwrite(spDDUrecord,2,pos-spDDUrecord,file);
}
