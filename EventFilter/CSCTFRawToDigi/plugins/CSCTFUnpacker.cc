#include "EventFilter/CSCTFRawToDigi/interface/CSCTFUnpacker.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//Digi
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1Track.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCSPStatusDigi.h"

//Digi collections
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"

//Unique key
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//Don't know what
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
//#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

//#include <iostream>
#include <sstream>

CSCTFUnpacker::CSCTFUnpacker(const edm::ParameterSet& pset):edm::EDProducer(),mapping(0),monitor(0){
	LogDebug("CSCTFUnpacker|ctor")<<"Started ...";

	// Initialize slot<->sector assignment
	slot2sector = pset.getUntrackedParameter< std::vector<int> >("slot2sector",std::vector<int>(0));
	if( slot2sector.size() != 22 ){
		if( slot2sector.size() ) edm::LogError("CSCTFUnpacker")<<"Wrong 'untracked vint32 slot2sector' size."
			<<" SectorProcessor boards reside in some of 22 slots and assigned to 12 sectors. Using defaults";
		// Use default assignment
		LogDebug("CSCTFUnpacker|ctor")<<"Creating default slot<->sector assignment";
		slot2sector.resize(22);
		slot2sector[0] = 0; slot2sector[1] = 0; slot2sector[2] = 0;
		slot2sector[3] = 0; slot2sector[4] = 0; slot2sector[5] = 0;
		slot2sector[6] = 1; slot2sector[7] = 2; slot2sector[8] = 3;
		slot2sector[9] = 4; slot2sector[10]= 5; slot2sector[11]= 6;
		slot2sector[12]= 0; slot2sector[13]= 0;
		slot2sector[14]= 0; slot2sector[15]= 0;
		slot2sector[16]= 7; slot2sector[17]= 8; slot2sector[18]= 9;
		slot2sector[19]=10; slot2sector[20]=11; slot2sector[21]=12;
	} else {
		LogDebug("CSCTFUnpacker|ctor")<<"Reassigning slot<->sector map according to 'untracked vint32 slot2sector'";
		for(int slot=0; slot<22; slot++)
			if( slot2sector[slot]<0 || slot2sector[slot]>12 )
				throw cms::Exception("Invalid configuration")<<"CSCTFUnpacker: sector index is set out of range (slot2sector["<<slot<<"]="<<slot2sector[slot]<<", should be [0-12])";
	}
	slot2sector.resize(32); // just for safety (in case of bad data)

	if( pset.getUntrackedParameter<bool>("runDQM", false) )
		monitor = edm::Service<CSCTFMonitorInterface>().operator->();

	std::string mappingFile = pset.getUntrackedParameter<std::string>("mapping",std::string(""));
	if( mappingFile.length() ){
		LogDebug("CSCTFUnpacker|ctor") << "Define ``mapping'' only if you want to screw up real geometry";
		mapping = new CSCTriggerMappingFromFile(mappingFile);
	} else {
		LogDebug("CSCTFUnpacker|ctor") << "Generating default hw<->geometry mapping";
		class M: public CSCTriggerSimpleMapping{ void fill(void){} };
		mapping = new M();
		for(int endcap=1; endcap<=2; endcap++)
			for(int station=1; station<=4; station++)
				for(int sector=1; sector<=6; sector++)
					for(int csc=1; csc<=9; csc++){
						if( station==1 ){
							mapping->addRecord(endcap,station,sector,1,csc,endcap,station,sector,1,csc);
							mapping->addRecord(endcap,station,sector,2,csc,endcap,station,sector,2,csc);
						} else
							mapping->addRecord(endcap,station,sector,0,csc,endcap,station,sector,0,csc);
					}
	}

	producer = pset.getUntrackedParameter<edm::InputTag>("producer",edm::InputTag("source"));

	produces<CSCCorrelatedLCTDigiCollection>("MuonCSCTFCorrelatedLCTDigi");
	produces<L1CSCTrackCollection>          ("MuonL1CSCTrackCollection");
	produces<L1CSCStatusDigiCollection>     ("MuonL1CSCStatusDigiCollection");

	LogDebug("CSCTFUnpacker|ctor") << "... and finished";
}

CSCTFUnpacker::~CSCTFUnpacker(){
	if( mapping ) delete mapping;
}

void CSCTFUnpacker::produce(edm::Event& e, const edm::EventSetup& c){
	// Get a handle to the FED data collection
	edm::Handle<FEDRawDataCollection> rawdata;
	e.getByLabel(producer.label(),producer.instance(),rawdata);

	// create the collection of CSC wire and strip Digis
	std::auto_ptr<CSCCorrelatedLCTDigiCollection> LCTProduct(new CSCCorrelatedLCTDigiCollection);
	std::auto_ptr<L1CSCTrackCollection>           trackProduct(new L1CSCTrackCollection);
	std::auto_ptr<L1CSCStatusDigiCollection>      statusProduct(new L1CSCStatusDigiCollection);

	for(int fedid=FEDNumbering::getCSCTFFEDIds().first; fedid<=FEDNumbering::getCSCTFFEDIds().second; fedid++){
		const FEDRawData& fedData = rawdata->FEDData(fedid);
		if( fedData.size()==0 ) continue;
		//LogDebug("CSCTFUnpacker|produce");
		//if( monitor ) monitor->process((unsigned short*)fedData.data());
		unsigned int unpacking_status = tfEvent.unpack((unsigned short*)fedData.data(),fedData.size()/2);
		if( unpacking_status==0 ){
			// There may be several SPs in event
			std::vector<CSCSPEvent> SPs = tfEvent.SPs();
			// Cycle over all of them
			for(std::vector<CSCSPEvent>::const_iterator sp=SPs.begin(); sp!=SPs.end(); sp++){

				L1CSCSPStatusDigi status; ///
				status.sp_slot    = sp->header().slot();
				status.l1a_bxn    = sp->header().BXN();
				status.fmm_status = sp->header().status();

				for(unsigned int tbin=0; tbin<sp->header().nTBINs(); tbin++){

					status.se |= sp->record(tbin).SEs();
					status.sm |= sp->record(tbin).SMs();
					status.bx |= sp->record(tbin).BXs();
					status.af |= sp->record(tbin).AFs();
					status.vp |= sp->record(tbin).VPs();

					for(unsigned int FPGA=0; FPGA<5; FPGA++)
						for(unsigned int MPClink=0; MPClink<3; ++MPClink){
							std::vector<CSCSP_MEblock> lct = sp->record(tbin).LCT(FPGA,MPClink);
							if( lct.size()==0 ) continue;

							status.link_status[lct[0].spInput()] |=
								(1<<lct[0].receiver_status_frame1())|
								(1<<lct[0].receiver_status_frame2())|
								((lct[0].aligment_fifo()?1:0)<<4);
							status.mpc_link_id |= (lct[0].link()<<2)|lct[0].mpc();

							int station = ( FPGA ? FPGA : 1 );
							int endcap=0, sector=0;
							if( slot2sector[sp->header().slot()] ){
								endcap = slot2sector[sp->header().slot()]/7 + 1;
								sector = slot2sector[sp->header().slot()]%7;
							} else {
								endcap = (sp->header().endcap()?1:2);
								sector =  sp->header().sector();
							}
							int subsector = ( FPGA>1 ? 0 : FPGA+1 );
							int cscid   = lct[0].csc() ;

							try{
								CSCDetId id = mapping->detId(endcap,station,sector,subsector,cscid,0);
								// corrlcts now have no layer associated with them
								LCTProduct->insertDigi(id,CSCCorrelatedLCTDigi(0,lct[0].vp(),lct[0].quality(),lct[0].wireGroup(),
													       lct[0].strip(),lct[0].pattern(),lct[0].l_r(),
													       lct[0].tbin(),lct[0].link() ));

// LogDebug("CSCUnpacker|produce") << "Unpacked digi: "<< aFB.frontDigiData(FPGA,MPClink);

							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event "
								      <<sp->header().L1A()<<" (endcap="<<endcap<<",station="<<station<<",sector="<<sector<<",subsector="<<subsector<<",cscid="<<cscid<<",spSlot="<<sp->header().slot()<<")";
							}

						}

					std::vector<CSCSP_SPblock> tracks = sp->record(tbin).tracks();
					for(std::vector<CSCSP_SPblock>::const_iterator iter=tracks.begin(); iter!=tracks.end(); iter++){
						L1CSCTrack track;
						if( slot2sector[sp->header().slot()] ){
							track.first.m_endcap = slot2sector[sp->header().slot()]/7;
							track.first.m_sector = slot2sector[sp->header().slot()]%7;
						} else {
							track.first.m_endcap = (sp->header().endcap()?1:2);
							track.first.m_sector = sp->header().sector();
						}

						track.first.m_lphi      = iter->phi();
						track.first.m_ptAddress = iter->ptLUTaddress();
						track.first.setStationIds(iter->ME1_id(),iter->ME2_id(),iter->ME3_id(),iter->ME4_id(),iter->MB_id());
						track.first.setBx(iter->tbin());

						track.first.setPhiPacked(iter->phi());
						track.first.setEtaPacked(iter->eta());
						track.first.setChargePacked((~iter->charge())&0x1);

						track.first.m_output_link = sp->record(tbin).ptSpyTrack();
						if( track.first.m_output_link ){
							track.first.m_rank = (iter->f_r()?sp->record(tbin).ptSpy()&0x1F:(sp->record(tbin).ptSpy()&0x1F00)>>8);
							track.first.setChargeValidPacked((iter->f_r()?(sp->record(tbin).ptSpy()&0x80)>>8:(sp->record(tbin).ptSpy()&0x8000)>>15));
						} else {
							track.first.m_rank = 0;
							track.first.setChargeValidPacked(0);
						}
						track.first.setFineHaloPacked(iter->halo());

						std::vector<CSCSP_MEblock> lcts = iter->LCTs();

						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++){
							int station   = ( lct->spInput()>6 ? (lct->spInput()-1)/3 : 1 );
							int subsector = ( lct->spInput()>6 ? 0 : (lct->spInput()-1)/3 + 1 );
							try{
								CSCDetId id = mapping->detId(track.first.m_endcap,station,track.first.m_sector,subsector,lct->csc(),0);
								track.second.insertDigi(id,CSCCorrelatedLCTDigi(0,lct->vp(),lct->quality(),lct->wireGroup(),
													     lct->strip(),lct->pattern(),lct->l_r(),
													     lct->tbin(),lct->link() ));
							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event"
								      <<sp->header().L1A()<<" (endcap="<<track.first.m_endcap<<",station="<<station<<",sector="<<track.first.m_sector<<",subsector="<<subsector<<",cscid="<<lct->csc()<<",spSlot="<<sp->header().slot()<<")";
							}
						}
						trackProduct->push_back( track );
					}
				}
				statusProduct->second.push_back( status );
			}
		} else {
			edm::LogError("CSCTFUnpacker|produce")<<" problem of unpacking TF event: 0x"<<std::hex<<unpacking_status<<std::dec<<" code";
		}

		statusProduct->first  = unpacking_status;

	} //end of fed cycle
	e.put(LCTProduct,"MuonCSCTFCorrelatedLCTDigi"); // put processed lcts into the event.
	e.put(trackProduct,"MuonL1CSCTrackCollection");
	e.put(statusProduct,"MuonL1CSCStatusDigiCollection");
}
