#include "EventFilter/CSCTFRawToDigi/interface/CSCTFPacker.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"

#include <strings.h>

#include <iostream>

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

CSCTFPacker::CSCTFPacker(const edm::ParameterSet &conf):edm::EDAnalyzer(){
	std::string mapPath = "/"+conf.getUntrackedParameter<std::string>("MappingFile","");
	TFMapping = new CSCTriggerMappingFromFile(getenv("CMSSW_BASE") + mapPath);

	zeroSuppression = conf.getUntrackedParameter<bool>("zeroSuppression");
	nTBINs          = conf.getUntrackedParameter<int> ("nTBINs");
	activeSectors   = conf.getUntrackedParameter<int> ("activeSectors");

	std::string outputFile = conf.getUntrackedParameter<std::string> ("outputFile");

	if( (file = fopen(outputFile.c_str(),"wt"))==NULL ){
		exit(0);
	}
}

CSCTFPacker::~CSCTFPacker(void){
	if( file ) fclose(file);
}

void CSCTFPacker::analyze(edm::Event const& e, edm::EventSetup const& iSetup){
	edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
	e.getByLabel("csctfunpacker","MuonCSCTFCorrelatedLCTDigi",corrlcts);

	CSCSP_MEblock meDataRecord[12][7][4][9][2]; // LCT in sector X, tbin Y, station Z, csc W, and lct I
	bzero(&meDataRecord,sizeof(meDataRecord));
	CSCSPRecord meDataHeader[12][7]; // Data Block Header for sector X and tbin Y
	bzero(&meDataHeader,sizeof(meDataHeader));

	for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++){
		int lctId=0;

		CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
		for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++){
			int station = (*csc).first.station()-1;
			int cscId   = (*csc).first.triggerCscId()-1;
			int sector  = (*csc).first.triggerSector()-1;
			int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
			int tbin    = lct->getBX();
			int fpga    = ( subSector ? subSector-1 : station+1 );

//std::cout<<"Front data: "<<station<<"  sector: "<<sector<<"  subSector: "<<subSector<<"  tbin: "<<tbin<<"  cscId: "<<cscId<<endl;

			// If Det Id is within range
			if( sector<0 || sector>5 || station<0 || station>3 || cscId<0 || cscId>8 || lctId<0 || lctId>1){
				edm::LogInfo("CSCTFPacker: CSC digi are out of range");
				continue;
			}

			meDataRecord[sector][tbin][fpga][cscId][lctId].clct_pattern_number = lct->getPattern();
			meDataRecord[sector][tbin][fpga][cscId][lctId].quality_            = lct->getQuality();
			meDataRecord[sector][tbin][fpga][cscId][lctId].wire_group_id       = lct->getKeyWG();

			meDataRecord[sector][tbin][fpga][cscId][lctId].clct_pattern_id     = lct->getStrip();
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

			meDataHeader[sector][tbin].spare_1 = 1; // for later use
		}
	}


	edm::Handle<L1CSCTrackCollection> tracks;
	e.getByLabel("csctfunpacker","MuonL1CSCTrackCollection",tracks);

	CSCSP_SPblock spDataRecord[12][7][3]; // Up to 3 tracks in sector X and tbin Y
	int nTrk=0;

	for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++){
/*		for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=trk->second.product().begin(); csc!=trk->second.product().end(); csc++){
			CSCCorrelatedLCTDigiCollection::Range range = corrlcts.product()->get(csc->first);
			if( range.size() != 1 ) KARAUL;
			else {
				range.first();
			}
			int station = csc->first.station()-1;
			int cscId   = csc->first.triggerCscId()-1;
			int sector  = csc->first.triggerSector()-1;
			int tbin    = trk->getBX();
		}
*/
		int sector = 6*trk->first.endcap()+trk->first.sector();
		int tbin   = trk->first.BX();
		spDataRecord[sector][tbin][nTrk].phi_       = trk->first.phi_packed();
		spDataRecord[sector][tbin][nTrk].sign_      =(trk->first.ptLUTAddress()>>20)&0x1;
		spDataRecord[sector][tbin][nTrk].front_rear = 0; // not necessary
		spDataRecord[sector][tbin][nTrk].charge_    = trk->first.chargeValue(); //
		spDataRecord[sector][tbin][nTrk].eta_       = trk->first.eta_packed();

		spDataRecord[sector][tbin][nTrk].halo_      = trk->first.finehalo_packed();
		spDataRecord[sector][tbin][nTrk].se         = 0; // dummy
		spDataRecord[sector][tbin][nTrk].deltaPhi12_= trk->first.ptLUTAddress()&0xFF;
		spDataRecord[sector][tbin][nTrk].deltaPhi23_=(trk->first.ptLUTAddress()>>8)&0xF;
		spDataRecord[sector][tbin][nTrk].bxn0_      = 0; //dummy
		spDataRecord[sector][tbin][nTrk].bc0_       = 0; //dummy

		spDataRecord[sector][tbin][nTrk].me1_id     = trk->first.me1ID();
		spDataRecord[sector][tbin][nTrk].me2_id     = trk->first.me2ID();
		spDataRecord[sector][tbin][nTrk].me3_id     = trk->first.me3ID();
		spDataRecord[sector][tbin][nTrk].me4_id     = trk->first.me4ID();
		spDataRecord[sector][tbin][nTrk].mb_id      = trk->first.mb1ID();
		spDataRecord[sector][tbin][nTrk].ms_id      = 0; // don't care

		spDataRecord[sector][tbin][nTrk].me1_tbin   = 0; // Unknown !
		spDataRecord[sector][tbin][nTrk].me2_tbin   = 0; // Unknown !
		spDataRecord[sector][tbin][nTrk].me3_tbin   = 0; // Unknown !
		spDataRecord[sector][tbin][nTrk].me4_tbin   = 0; // Unknown !
		spDataRecord[sector][tbin][nTrk].mb_tbin    = 0; // Unknown !

		nTrk++;

		spDataRecord[sector][tbin][nTrk].id_ = nTrk; // for later use
		switch(nTrk){
			case 1: meDataHeader[sector][tbin].mode1 = (trk->first.ptLUTAddress()>>16)&0xF; break;
			case 2: meDataHeader[sector][tbin].mode2 = (trk->first.ptLUTAddress()>>16)&0xF; break;
			case 3: meDataHeader[sector][tbin].mode3 = (trk->first.ptLUTAddress()>>16)&0xF; break;
			default:
				edm::LogInfo("More than 3 tracks from one SP in the BX");
				continue;
				break;
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
	header.csr_dfc |= 0x7F; // All FPGAs are active

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
	memcpy(pos,&header,16);
	pos+=8;

	for(int sector=0; sector<12; sector++){
		if( !(activeSectors & (1<<sector)) ) continue;
//std::cout<<"Writing sector: "<<sector<<endl;
		for(int tbin=0; tbin<nTBINs; tbin++){
//std::cout<<"Writing tbin: "<<tbin<<endl;
			//if( !zeroSuppression || meDataHeader[sector][tbin].spare_1 )
				memcpy(pos,&meDataHeader[sector][tbin],16);
				pos+=8;
				for(int fpga=0; fpga<5; fpga++){
//std::cout<<"Writing FPGA: "<<fpga<<endl;
					int nLCTs=0;
					for(int cscId=0; cscId<9; cscId++)
						for(int lctId=0; lctId<2; lctId++)
							// Only 3 LCT per BX from the same fpga are allowed (to be valid):
							if( meDataRecord[sector][tbin][fpga][cscId][lctId].valid_pattern ){
//std::cout<<"Writing CSC: "<<cscId<<"  LCT: "<<lctId<<endl;
								memcpy(pos,&meDataRecord[sector][tbin][fpga][cscId][lctId],8);
								pos+=4;
								nLCTs++;
							}
					// Here we imply that readout is set to be active (in DFC)
					if( meDataHeader[sector][tbin].vq_a ) pos += 4;
					if( meDataHeader[sector][tbin].vq_b ) pos += 4;
				}
				for(int trk=0; trk<3; trk++){
					if( spDataRecord[sector][tbin][trk].id_ ){
						memcpy(pos,&spDataRecord[sector][tbin][trk],8);
						pos+=4;
					}
				}
		}
	}

	memcpy(pos,&trailer,16);
	pos+=8;

	fwrite(spDDUrecord,2,pos-spDDUrecord,file);
}
