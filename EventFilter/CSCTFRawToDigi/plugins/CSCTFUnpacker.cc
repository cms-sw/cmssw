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
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

//Digi collections
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"

//Unique key
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//Don't know what
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
//#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

//#include <iostream>
#include <sstream>

CSCTFUnpacker::CSCTFUnpacker(const edm::ParameterSet& pset):edm::stream::EDProducer<>(),mapping(0){
	LogDebug("CSCTFUnpacker|ctor")<<"Started ...";

	// Edges of the time window, which LCTs are put into (unlike tracks, which are always centred around 0):
	m_minBX = pset.getParameter<int>("MinBX"); //3
	m_maxBX = pset.getParameter<int>("MaxBX"); //9

	// Swap: if(swapME1strips && me1b && !zplus) strip = 65 - strip; // 1-64 -> 64-1 :
	swapME1strips = pset.getParameter<bool>("swapME1strips");

	// Initialize slot<->sector assignment
	slot2sector = pset.getParameter< std::vector<int> >("slot2sector");
	LogDebug("CSCTFUnpacker|ctor")<<"Verifying slot<->sector map from 'vint32 slot2sector'";
	for(int slot=0; slot<22; slot++)
		if( slot2sector[slot]<0 || slot2sector[slot]>12 )
			throw cms::Exception("Invalid configuration")<<"CSCTFUnpacker: sector index is set out of range (slot2sector["<<slot<<"]="<<slot2sector[slot]<<", should be [0-12])";
	// Just for safety (in case of bad data):
	slot2sector.resize(32);

	// As we use standard CSC digi containers, we have to initialize mapping:
	std::string mappingFile = pset.getParameter<std::string>("mappingFile");
	if( mappingFile.length() ){
		LogDebug("CSCTFUnpacker|ctor") << "Define ``mapping'' only if you want to screw up real geometry";
		mapping = new CSCTriggerMappingFromFile(mappingFile);
	} else {
		LogDebug("CSCTFUnpacker|ctor") << "Generating default hw<->geometry mapping";
		class M: public CSCTriggerSimpleMapping{ void fill(void) override{} };
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

	producer = pset.getParameter<edm::InputTag>("producer");

	produces<CSCCorrelatedLCTDigiCollection>();
	produces<L1CSCTrackCollection>();
	produces<L1CSCStatusDigiCollection>();
	produces<CSCTriggerContainer<csctf::TrackStub> >("DT");
}

CSCTFUnpacker::~CSCTFUnpacker(){
	if( mapping ) delete mapping;
}

void CSCTFUnpacker::produce(edm::Event& e, const edm::EventSetup& c){
	// Get a handle to the FED data collection
	edm::Handle<FEDRawDataCollection> rawdata;
	e.getByLabel(producer.label(),producer.instance(),rawdata);

	// create the collection of CSC wire and strip digis as well as of DT stubs, which we receive from DTTF
	std::auto_ptr<CSCCorrelatedLCTDigiCollection> LCTProduct(new CSCCorrelatedLCTDigiCollection);
	std::auto_ptr<L1CSCTrackCollection>           trackProduct(new L1CSCTrackCollection);
	std::auto_ptr<L1CSCStatusDigiCollection>      statusProduct(new L1CSCStatusDigiCollection);
	std::auto_ptr<CSCTriggerContainer<csctf::TrackStub> > dtProduct(new CSCTriggerContainer<csctf::TrackStub>);

	for(int fedid=FEDNumbering::MINCSCTFFEDID; fedid<=FEDNumbering::MAXCSCTFFEDID; fedid++){
		const FEDRawData& fedData = rawdata->FEDData(fedid);
		if( fedData.size()==0 ) continue;
		//LogDebug("CSCTFUnpacker|produce");
		//if( monitor ) monitor->process((unsigned short*)fedData.data());
		unsigned int unpacking_status = tfEvent.unpack((unsigned short*)fedData.data(),fedData.size()/2);
		if( unpacking_status==0 ){
			// There may be several SPs in event
			std::vector<const CSCSPEvent*> SPs = tfEvent.SPs_fast();
			// Cycle over all of them
			for(std::vector<const CSCSPEvent *>::const_iterator spItr=SPs.begin(); spItr!=SPs.end(); spItr++){
				const CSCSPEvent *sp = *spItr;

				L1CSCSPStatusDigi status; ///
				status.sp_slot    = sp->header().slot();
				status.l1a_bxn    = sp->header().BXN();
				status.fmm_status = sp->header().status();
				status.track_cnt  = sp->counters().track_counter();
				status.orbit_cnt  = sp->counters().orbit_counter();

				// Finds central LCT BX
				// assumes window is odd number of bins
				int central_lct_bx = (m_maxBX + m_minBX)/2;

				// Find central SP BX
				// assumes window is odd number of bins
				int central_sp_bx = int(sp->header().nTBINs()/2);

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
								sector = slot2sector[sp->header().slot()];
								if( sector>6 ) sector -= 6;
							} else {
								endcap = (sp->header().endcap()?1:2);
								sector =  sp->header().sector();
							}
							int subsector = ( FPGA>1 ? 0 : FPGA+1 );
							int cscid   = lct[0].csc() ;

							try{
								CSCDetId id = mapping->detId(endcap,station,sector,subsector,cscid,0);
								// corrlcts now have no layer associated with them
								LCTProduct->insertDigi(id,
									CSCCorrelatedLCTDigi(
										0,lct[0].vp(),lct[0].quality(),lct[0].wireGroup(),
										(swapME1strips && cscid<=3 && station==1 && endcap==2 && lct[0].strip()<65 ? 65 - lct[0].strip() : lct[0].strip() ),
										lct[0].pattern(),lct[0].l_r(),
										(lct[0].tbin()+(central_lct_bx-central_sp_bx)),
										lct[0].link(), lct[0].BXN(), 0, cscid )
									);
							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event "
								      <<sp->header().L1A()<<" (endcap="<<endcap<<",station="<<station<<",sector="<<sector<<",subsector="<<subsector<<",cscid="<<cscid<<",spSlot="<<sp->header().slot()<<")";
							}

						}

					std::vector<CSCSP_MBblock> mbStubs = sp->record(tbin).mbStubs();
					for(std::vector<CSCSP_MBblock>::const_iterator iter=mbStubs.begin(); iter!=mbStubs.end(); iter++){
						int endcap, sector;
						if( slot2sector[sp->header().slot()] ){
							endcap = slot2sector[sp->header().slot()]/7 + 1;
							sector = slot2sector[sp->header().slot()];
							if( sector>6 ) sector -= 6;
						} else {
							endcap = (sp->header().endcap()?1:2);
							sector =  sp->header().sector();
						}
						const unsigned int csc2dt[6][2] = {{2,3},{4,5},{6,7},{8,9},{10,11},{12,1}};
						DTChamberId id((endcap==1?2:-2),1, csc2dt[sector-1][iter->id()-1]);
						CSCCorrelatedLCTDigi base(0,iter->vq(),iter->quality(),iter->cal(),iter->flag(),iter->bc0(),iter->phi_bend(),tbin+(central_lct_bx-central_sp_bx),iter->id(),iter->bxn(),iter->timingError(),iter->BXN());
						csctf::TrackStub dtStub(base,id,iter->phi(),0);
						dtProduct->push_back(dtStub);
					}

					std::vector<CSCSP_SPblock> tracks = sp->record(tbin).tracks();
					unsigned int trkNumber=0;
					for(std::vector<CSCSP_SPblock>::const_iterator iter=tracks.begin(); iter!=tracks.end(); iter++,trkNumber++){
						L1CSCTrack track;
						if( slot2sector[sp->header().slot()] ){
							track.first.m_endcap = slot2sector[sp->header().slot()]/7 + 1;
							track.first.m_sector = slot2sector[sp->header().slot()];
							if(  track.first.m_sector>6 ) track.first.m_sector -= 6;
						} else {
							track.first.m_endcap = (sp->header().endcap()?1:2);
							track.first.m_sector =  sp->header().sector();
						}

						track.first.m_lphi      = iter->phi();
						track.first.m_ptAddress = iter->ptLUTaddress();
						track.first.m_fr        = iter->f_r();
						track.first.m_ptAddress|=(iter->f_r() << 21);

						track.first.setStationIds(iter->ME1_id(),iter->ME2_id(),iter->ME3_id(),iter->ME4_id(),iter->MB_id());
						track.first.setTbins(iter->ME1_tbin(), iter->ME2_tbin(), iter->ME3_tbin(), iter->ME4_tbin(), iter->MB_tbin() );
						track.first.setBx(iter->tbin()-central_sp_bx);
						track.first.setBits(iter->syncErr(), iter->bx0(), iter->bc0());

						track.first.setLocalPhi(iter->phi());
						track.first.setEtaPacked(iter->eta());
						track.first.setChargePacked(iter->charge());

						track.first.m_output_link = iter->id();
						if( track.first.m_output_link ){
							track.first.m_rank = (iter->f_r()?sp->record(tbin).ptSpy()&0x7F:(sp->record(tbin).ptSpy()&0x7F00)>>8);
							track.first.setChargeValidPacked((iter->f_r()?(sp->record(tbin).ptSpy()&0x80)>>8:(sp->record(tbin).ptSpy()&0x8000)>>15));
						} else {
							track.first.m_rank = 0;
							track.first.setChargeValidPacked(0);
						}
						track.first.setFineHaloPacked(iter->halo());

						track.first.m_winner = iter->MS_id()&(1<<trkNumber);

						std::vector<CSCSP_MEblock> lcts = iter->LCTs();
						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++){
							int station   = ( lct->spInput()>6 ? (lct->spInput()-1)/3 : 1 );
							int subsector = ( lct->spInput()>6 ? 0 : (lct->spInput()-1)/3 + 1 );
							try{
								CSCDetId id = mapping->detId(track.first.m_endcap,station,track.first.m_sector,subsector,lct->csc(),0);
								track.second.insertDigi(id,
									CSCCorrelatedLCTDigi(
										0,lct->vp(),lct->quality(),lct->wireGroup(),
										(swapME1strips && lct->csc()<=3 && station==1 && track.first.m_endcap==2 && lct[0].strip()<65 ? 65 - lct[0].strip() : lct[0].strip() ),
										lct->pattern(),lct->l_r(),
										(lct->tbin()+(central_lct_bx-central_sp_bx)),
										lct->link(), lct->BXN(), 0, lct->csc() )
									);
							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding track digi to collection in event"
								      <<sp->header().L1A()<<" (endcap="<<track.first.m_endcap<<",station="<<station<<",sector="<<track.first.m_sector<<",subsector="<<subsector<<",cscid="<<lct->csc()<<",spSlot="<<sp->header().slot()<<")";
							}
						}

						std::vector<CSCSP_MBblock> mbStubs = iter->dtStub();
						for(std::vector<CSCSP_MBblock>::const_iterator iter=mbStubs.begin(); iter!=mbStubs.end(); iter++){
							CSCDetId id = mapping->detId(track.first.m_endcap,1,track.first.m_sector,iter->id(),1,0);
							track.second.insertDigi(id,
								CSCCorrelatedLCTDigi(iter->phi(),iter->vq(),iter->quality()+100,iter->cal(),iter->flag(),iter->bc0(),iter->phi_bend(),tbin+(central_lct_bx-central_sp_bx),iter->id(),iter->bxn(),iter->timingError(),iter->BXN())
							);
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
	e.put(dtProduct,"DT");
	e.put(LCTProduct); // put processed lcts into the event.
	e.put(trackProduct);
	e.put(statusProduct);
}
