#include "EventFilter/CSCTFRawToDigi/interface/CSCTFUnpacker.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//Digi stuff
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1Track.h"
//#include "DataFormats/CSCTFObjects/interface/CSCTFL1Track.h"

//Include LCT digis later
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
//#include "DataFormats/CSCTFObjects/interface/CSCTFL1TrackCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include <EventFilter/CSCTFRawToDigi/interface/CSCTFMonitorInterface.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

//CSC Track Finder Raw Data Formats // TB and DDU
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventData.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"
// more to come later

#include <iostream>


CSCTFUnpacker::CSCTFUnpacker(const edm::ParameterSet & pset):ptlut(pset)
{
  LogDebug("CSCTFUnpacker|ctor") << "starting CSCTFConstructor";

  instantiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  testBeam = pset.getUntrackedParameter<bool>("TestBeamData",false);
  std::string mapPath = "/" + pset.getUntrackedParameter<std::string>("MappingFile","");
//  if(testBeam)
///    {
      TBFEDid = pset.getUntrackedParameter<int>("TBFedId");
      TBendcap = pset.getUntrackedParameter<int>("TBEndcap");
      TBsector = pset.getUntrackedParameter<int>("TBSector");
//    }
//  else
//    {
//      TBFEDid = 0;
//      TBsector = 0;
//      TBendcap = 0;
//    }
  debug = pset.getUntrackedParameter<bool>("debug",false);
  TFmapping = new CSCTriggerMappingFromFile(getenv("CMSSW_BASE")+mapPath);

  if(instantiateDQM){

    monitor = edm::Service<CSCTFMonitorInterface>().operator->();

  }

  numOfEvents = 0;

  produces<CSCCorrelatedLCTDigiCollection>("MuonCSCTFCorrelatedLCTDigi");
  produces<L1CSCTrackCollection>("MuonL1CSCTrackCollection");

  LogDebug("CSCTFUnpacker|ctor") << "... and finished";
}

CSCTFUnpacker::~CSCTFUnpacker(){

  //fill destructor here
  //delete dccData;
  delete TFmapping;
}


void CSCTFUnpacker::produce(edm::Event & e, const edm::EventSetup& c)
{

  //create data pointers, to use later
  // CSCTFDDUEventData * ddu = NULL;
  CSCTFTBEventData *tbdata = NULL;

  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType( rawdata);

  // create the collection of CSC wire and strip Digis
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> LCTProduct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<L1CSCTrackCollection> trackProduct(new L1CSCTrackCollection);
  //std::auto_ptr<CSCTFL1TrackCollection> trackProduct(new CSCRPCDigiCollection);

//  for(int fedid = FEDNumbering::getCSCFEDIds().first;
//      fedid <= ((testBeam) ? (FEDNumbering::getCSCFEDIds().first) : FEDNumbering::getCSCFEDIds().second);
  for(int fedid = FEDNumbering::getCSCTFFEDIds().first;
      fedid <= ((testBeam) ? (FEDNumbering::getCSCTFFEDIds().first) : FEDNumbering::getCSCTFFEDIds().second);
      ++fedid)
    {

      const FEDRawData& fedData = rawdata->FEDData(fedid);
      if(fedData.size())
	{

	  if(testBeam){
	    tbdata = new CSCTFTBEventData((unsigned short*)fedData.data());

	  LogDebug("CSCTFUnpacker|produce") << (*tbdata);

	  ++numOfEvents;

	  if(instantiateDQM)
	    {
	      if(tbdata) monitor->process(*tbdata);
	      // else not implemented yet.
	    }

		CSCTFTBFrontBlock aFB;
		CSCTFTBSPBlock aSPB;
		CSCTFTBSPData aSPD;

		for(int BX = 1; BX<= tbdata->eventHeader().numBX() ; ++BX)
		  {
		    aFB = tbdata->frontDatum(BX);
		    for(int FPGA = 1; FPGA <= tbdata->eventHeader().numMPC() ; ++FPGA)
		      {
			for(int MPClink = 1; MPClink <= tbdata->eventHeader().numLinks() ; ++MPClink)
			  {
				int subsector = 0;
				int station = 0;

				if(FPGA == 1) subsector = 1;
				if(FPGA == 2) subsector = 2;
				station = (((FPGA - 1) == 0) ? 1 : FPGA-1);

				int cscid = aFB.frontData(FPGA,MPClink).CSCIDPacked();
				if(cscid)
				  {
				    try
				      {
					CSCDetId id = TFmapping->detId(TBendcap,station,TBsector,subsector,cscid,3);
					// corrlcts reside on the key layer which is layer 3.
					LCTProduct->insertDigi(id,aFB.frontDigiData(FPGA,MPClink));
					LogDebug("CSCUnpacker|produce") << "Unpacked digi: "<< aFB.frontDigiData(FPGA,MPClink);
				      }
				    catch(cms::Exception &e)
				      {
					edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event"
								      << tbdata->eventHeader().getLvl1num();
				      }
				    }

			  }
		      }
		  }

	} else {
		CSCTFEvent tfEvent;
		if( tfEvent.unpack((unsigned short*)fedData.data(),fedData.size()/2) )
			edm::LogError("CSCTFUnpacker|produce")<<" problem of unpacking TF event";
		else {
			++numOfEvents;
			// There may be several SPs in event
			std::vector<CSCSPEvent> SPs = tfEvent.SPs();
			// Cycle over all of them
			for(std::vector<CSCSPEvent>::const_iterator sp=SPs.begin(); sp!=SPs.end(); sp++){
				for(unsigned int tbin=0; tbin<sp->header().nTBINs(); tbin++){
					for(unsigned int FPGA=0; FPGA<5; FPGA++)
						for(unsigned int MPClink=0; MPClink<3; ++MPClink){
							std::vector<CSCSP_MEblock> lct = sp->record(tbin).LCT(FPGA,MPClink);
							if( lct.size()==0 ) continue;
							int station = ( FPGA ? FPGA : 1 );
							int endcap  = (TBendcap?TBendcap:(sp->header().endcap()?1:2)); // Make use of TBendcap for now
							int sector  = (TBsector?TBsector: sp->header().sector());      // Make use of TBsector until SP will be correctly set up
							int subsector = ( FPGA>1 ? 0 : FPGA+1 );
							int cscid   = lct[0].csc() ;

							try{
								CSCDetId id = TFmapping->detId(endcap,station,sector,subsector,cscid,0);
								// corrlcts now have no layer associated with them
								LCTProduct->insertDigi(id,CSCCorrelatedLCTDigi(0,lct[0].vp(),lct[0].quality(),lct[0].wireGroup(),
													       lct[0].strip(),lct[0].pattern(),lct[0].l_r(),
													       lct[0].tbin(),lct[0].link() ));
							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event"
												      << sp->header().L1A();
							}

						}

					std::vector<CSCSP_SPblock> tracks = sp->record(tbin).tracks();
					for(std::vector<CSCSP_SPblock>::const_iterator iter=tracks.begin(); iter!=tracks.end(); iter++){
					        L1CSCTrack track;
						track.first.m_endcap = (TBendcap?TBendcap:(sp->header().endcap()?1:2)); // Make use of TBendcap for now
						track.first.m_sector = (TBsector?TBsector: sp->header().sector());      // Make use of TBsector until SP will be correctly set up
						track.first.m_lphi      = iter->phi();
						track.first.m_ptAddress = iter->ptLUTaddress();
						track.first.setStationIds(iter->ME1_id(),iter->ME2_id(),iter->ME3_id(),iter->ME4_id(),iter->MB_id());
						track.first.setBx(iter->tbin());
						pt_data pt = ptlut.Pt(iter->deltaPhi12(),iter->deltaPhi23(),iter->eta(),iter->mode(),iter->f_r(),iter->sign());
						track.first.m_rank      = (iter->f_r()?pt.front_rank:pt.rear_rank);
						//track.f_r         = iter->f_r();
						track.first.setPhiPacked(iter->phi());
						track.first.setEtaPacked(iter->eta());
//						track.setPtPacked(pt);
						track.first.setChargePacked((~iter->charge())&0x1);
						track.first.setChargeValidPacked((iter->f_r()?pt.charge_valid_front:pt.charge_valid_rear));
						track.first.setFineHaloPacked(iter->halo());
//						track.setQualityPacked( quality );

						std::vector<CSCSP_MEblock> lcts = iter->LCTs();

						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++){
							int station   = ( lct->spInput()>6 ? (lct->spInput()-1)/3 : 1 );
							int subsector = ( lct->spInput()>6 ? 0 : (lct->spInput()-1)/3 + 1 );
							try{
								CSCDetId id = TFmapping->detId(track.first.m_endcap,station,track.first.m_sector,subsector,lct->csc(),0);
								track.second.insertDigi(id,CSCCorrelatedLCTDigi(0,lct->vp(),lct->quality(),lct->wireGroup(),
													     lct->strip(),lct->pattern(),lct->l_r(),
													     lct->tbin(),lct->link() ));
							} catch(cms::Exception &e) {
								edm::LogInfo("CSCTFUnpacker|produce") << e.what() << "Not adding digi to collection in event"
												      << sp->header().L1A();
							}
						}
						trackProduct->push_back( track );
					}
				}
			}
		}
	}
	}

	} //end of if(fedData.size())

  e.put(LCTProduct,"MuonCSCTFCorrelatedLCTDigi"); // put processed lcts into the event.
  e.put(trackProduct,"MuonL1CSCTrackCollection");

  if(tbdata) delete tbdata;
  tbdata = NULL;
}
