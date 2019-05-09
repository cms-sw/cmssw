#include "L1Trigger/DTPhase2Trigger/interface/DTTrigPhase2Prod.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"


// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TMath.h"


#include "L1Trigger/DTPhase2Trigger/src/muonpath.h"

using namespace edm;
using namespace std;

typedef vector<DTSectCollPhSegm>  SectCollPhiColl;
typedef SectCollPhiColl::const_iterator SectCollPhiColl_iterator;
typedef vector<DTSectCollThSegm>  SectCollThetaColl;
typedef SectCollThetaColl::const_iterator SectCollThetaColl_iterator;

/*
  Channels are labeled following next schema:
    ---------------------------------
    |   6   |   7   |   8   |   9   |
    ---------------------------------
        |   3   |   4   |   5   |
        -------------------------
            |   1   |   2   |
            -----------------
                |   0   |
                ---------
*/

/* Cell's combination, following previous labeling, to obtain every possible  muon's path. Others cells combinations imply non straight paths */
// const int CHANNELS_PATH_ARRANGEMENTS[8][4] = {
//     {0, 1, 3, 6}, {0, 1, 3, 7}, {0, 1, 4, 7}, {0, 1, 4, 8},
//     {0, 2, 4, 7}, {0, 2, 4, 8}, {0, 2, 5, 8}, {0, 2, 5, 9}
// };

/* For each of the previous cell's combinations, this array stores the associated cell's displacement, relative to lower layer cell, measured in semi-cell length units */

// const int CELL_HORIZONTAL_LAYOUTS[8][4] = {
//     {0, -1, -2, -3}, {0, -1, -2, -1}, {0, -1, 0, -1}, {0, -1, 0, 1},
//     {0,  1,  0, -1}, {0,  1,  0,  1}, {0,  1, 2,  1}, {0,  1, 2, 3}
// };


DTTrigPhase2Prod::DTTrigPhase2Prod(const ParameterSet& pset):
  chInDummy({DTPrimitive()}),   
  timeFromP1ToP2(0),
  currentBaseChannel(-1),
  chiSquareThreshold(50),
  bxTolerance(30),
  minQuality(LOWQGHOST)
{
    for (int lay = 0; lay < NUM_LAYERS; lay++)  {
     for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
	channelIn[lay][ch] = {chInDummy};
	channelIn[lay][ch].clear();
     }
    }
    
    produces<L1MuDTChambContainer>();
    produces<L1MuDTChambPhContainer>();
    produces<L1MuDTChambThContainer>();
    produces<L1Phase2MuDTPhContainer>();
    
    debug = pset.getUntrackedParameter<bool>("debug");
    pinta = pset.getUntrackedParameter<bool>("pinta");
    tanPhiTh = pset.getUntrackedParameter<double>("tanPhiTh");
    dT0_correlate_TP = pset.getUntrackedParameter<double>("dT0_correlate_TP");
    min_dT0_match_segment = pset.getUntrackedParameter<double>("min_dT0_match_segment");
    min_phinhits_match_segment = pset.getUntrackedParameter<int>("min_phinhits_match_segment");
    minx_match_2digis = pset.getUntrackedParameter<double>("minx_match_2digis");

    do_correlation = pset.getUntrackedParameter<bool>("do_correlation");
    p2_df = pset.getUntrackedParameter<int>("p2_df");
    filter_primos = pset.getUntrackedParameter<bool>("filter_primos");
    
    txt_ttrig_bc0 = pset.getUntrackedParameter<bool>("apply_txt_ttrig_bc0");
    
    dt4DSegmentsToken = consumes<DTRecSegment4DCollection>(pset.getParameter < edm::InputTag > ("dt4DSegments"));
    dtDigisToken = consumes< DTDigiCollection >(pset.getParameter<edm::InputTag>("digiTag"));

    rpcRecHitsLabel = consumes<RPCRecHitCollection>(pset.getUntrackedParameter < edm::InputTag > ("rpcRecHits"));
  
    
  // Choosing grouping scheme:
  grcode = pset.getUntrackedParameter<Int_t>("grouping_code");
  
  if (grcode == 0) grouping_obj = new InitialGrouping(pset);
  else {
    if (debug) cout << "DTp2::constructor: Non-valid grouping code. Choosing InitialGrouping by default." << endl;
    grouping_obj = new InitialGrouping(pset);
  }
  
  
    if(pinta){
	std::cout<<"BOOKING HISTOS"<<std::endl;

	theFileOut = new TFile("dt_phase2.root", "RECREATE");

	Nsegments = new TH1F("Nsegments","Nsegments",21,-0.5,20.5);
	NmetaPrimitives = new TH1F("NmetaPrimitives","NmetaPrimitives",201,-0.5,200.5);
	NfilteredMetaPrimitives = new TH1F("NfilteredMetaPrimitives","NfilteredMetaPrimitives",201,-0.5,200.5);
	NcorrelatedMetaPrimitives = new TH1F("NcorrelatedMetaPrimitives","NcorrelatedMetaPrimitives",201,-0.5,200.5);
	Ngroups = new TH1F("Ngroups","Ngroups",201,-0.5,200.5);
	Nquality = new TH1F("Nquality","Nquality",9,0.5,9.5);
	Nquality_matched = new TH1F("Nquality_matched","Nquality_matched",9,0.5,9.5);
	Nsegosl = new TH1F("Nsegosl","Nsegosl",100,-10,10);
	Nsegosl31 = new TH1F("Nsegosl31","Nsegosl31",100,-10,10);
	Nmd = new TH1F("Nmd","Nmd",11,-0.5,10.5);
	Nmd31 = new TH1F("Nmd31","Nmd31",11,-0.5,10.5);
	Nhits_segment_tp = new TH2F("Nhits_segment_tp","Nhits_segment_tp",10,-0.5,9.5,10,-0.5,9.5);
	
	char name [128];

	for(int wh=-2;wh<=2;wh++){
	    int iwh=wh+2;
	    auto swh = std::to_string(wh);
	    for(int st=1;st<=4;st++){
		int ist=st-1;
		auto sst = std::to_string(st);
		for(int se=1;se<=14;se++){
		    if(se>=13&&st!=4)continue;
		    int ise=se-1;
		    auto sse = std::to_string(se);
		    for(int qu=1;qu<=9;qu++){
			int iqu=qu-1;
			auto squ = std::to_string(qu);
			  
			std::string nameSL = "Wh"+swh+"_St"+sst+"_Se"+sse+"_Qu"+squ;
			  
			//TIME
			sprintf(name,"TIMEPhase2_%s",nameSL.c_str());
			TIMEPhase2[iwh][ist][ise][iqu] = new TH1F(name,name,100,-0.5,89075.5);
			      
			//T0
			sprintf(name,"TOPhase2_%s",nameSL.c_str());
			T0Phase2[iwh][ist][ise][iqu] = new TH1F(name,name,100,-0.5,89075.5);
    
			//2D
			sprintf(name,"segment_vs_jm_x_%s",nameSL.c_str());
			segment_vs_jm_x[iwh][ist][ise][iqu] = new TH2F(name,name,250,-250,250,250,-250,250);
			  
			sprintf(name,"segment_vs_jm_x_gauss_%s",nameSL.c_str());
			segment_vs_jm_x_gauss[iwh][ist][ise][iqu] = new TH1F(name,name,300,-0.04,0.04);

			sprintf(name,"segment_vs_jm_tanPhi_%s",nameSL.c_str());
			segment_vs_jm_tanPhi[iwh][ist][ise][iqu] = new TH2F(name,name,100,-1.5,1.5,100,-1.5,1.5);

			sprintf(name,"segment_vs_jm_tanPhi_gauss_%s",nameSL.c_str());
			segment_vs_jm_tanPhi_gauss[iwh][ist][ise][iqu] = new TH1F(name,name,300,-0.5,0.5); //for single ones resolution
			//segment_vs_jm_tanPhi_gauss[iwh][ist][ise][iqu] = new TH1F(name,name,300,-0.01,0.01); //for correlated
			  
			sprintf(name,"segment_vs_jm_T0_%s",nameSL.c_str());
			segment_vs_jm_T0[iwh][ist][ise][iqu] = new TH2F(name,name,100,0,90000,100,0,90000);

			sprintf(name,"segment_vs_jm_T0_gauss_%s",nameSL.c_str());
			segment_vs_jm_T0_gauss[iwh][ist][ise][iqu] = new TH1F(name,name,300,-100,100);

			sprintf(name,"segment_vs_jm_T0_gauss_all_%s",nameSL.c_str());
			segment_vs_jm_T0_gauss_all[iwh][ist][ise][iqu] = new TH1F(name,name,300,-100,100);
			
			sprintf(name,"observed_tanPsi_%s",nameSL.c_str());
			observed_tanPsi[iwh][ist][ise][iqu] = new TH1F(name,name,100,-1.5,1.5);

			sprintf(name,"all_observed_tanPsi_%s",nameSL.c_str());
			all_observed_tanPsi[iwh][ist][ise][iqu] = new TH1F(name,name,100,-1.5,1.5);
			
			sprintf(name,"observed_x_%s",nameSL.c_str());
			observed_x[iwh][ist][ise][iqu] = new TH1F(name,name,250,-250,250);

			sprintf(name,"all_observed_x_%s",nameSL.c_str());
			all_observed_x[iwh][ist][ise][iqu] = new TH1F(name,name,250,-250,250);

			sprintf(name,"observed_t0_%s",nameSL.c_str());
			observed_t0[iwh][ist][ise][iqu] = new TH1F(name,name,100,0,90000);

			sprintf(name,"all_observed_t0_%s",nameSL.c_str());
			all_observed_t0[iwh][ist][ise][iqu] = new TH1F(name,name,100,-100,100);
			
			sprintf(name,"chi2_%s",nameSL.c_str());
			chi2[iwh][ist][ise][iqu] = new TH1F(name,name,100,0.,0.02);

			sprintf(name,"TPphi_%s",nameSL.c_str());
			TPphi[iwh][ist][ise][iqu] = new TH1F(name,name,250,-1.5,1.5);

			sprintf(name,"TPphiB_%s",nameSL.c_str());
			TPphiB[iwh][ist][ise][iqu] = new TH1F(name,name,250,-1.5,1.5);

			sprintf(name,"MP_x_back_%s",nameSL.c_str());
			MP_x_back[iwh][ist][ise][iqu] = new TH2F(name,name,100,-250,250,100,-250,250);

			sprintf(name,"MP_psi_back_%s",nameSL.c_str());
			MP_psi_back[iwh][ist][ise][iqu] = new TH2F(name,name,100,-1.5,1.5,100,-1.5,1.5);

			
		    }
		    std::string nameSL = "Wh"+swh+"_St"+sst+"_Se"+sse;
		    sprintf(name,"expected_tanPsi_%s",nameSL.c_str());
		    expected_tanPsi[iwh][ist][ise] = new TH1F(name,name,100,-1.5,1.5);

		    sprintf(name,"expected_x_%s",nameSL.c_str());
		    expected_x[iwh][ist][ise] = new TH1F(name,name,250,-250,250);

		    sprintf(name,"expected_t0_%s",nameSL.c_str());
		    expected_t0[iwh][ist][ise] = new TH1F(name,name,100,0,90000);
		    
		}
	    }
	}
	
    }	  

    int rawId;
    
    //ttrig
    ttrig_filename = pset.getUntrackedParameter<std::string>("ttrig_filename");
    if(txt_ttrig_bc0){
	std::ifstream ifin(ttrig_filename.c_str());
	double ttrig;
	while (ifin.good()){
	    ifin >> rawId >> ttrig;
	    ttriginfo[rawId]=ttrig;
	}
    }
    
    //z
    z_filename = pset.getUntrackedParameter<std::string>("z_filename");
    std::ifstream ifin2(z_filename.c_str());
    double z;
    while (ifin2.good()){
        ifin2 >> rawId >> z;
        zinfo[rawId]=z;
    }

    //shift
    shift_filename = pset.getUntrackedParameter<std::string>("shift_filename");
    std::ifstream ifin3(shift_filename.c_str());
    double shift;
    while (ifin3.good()){
        ifin3 >> rawId >> shift;
        shiftinfo[rawId]=shift;
    }

    chosen_sl = pset.getUntrackedParameter<int>("trigger_with_sl");
    
    if(chosen_sl!=1 && chosen_sl!=3 && chosen_sl!=4){
	std::cout<<"chosen sl must be 1,3 or 4(both superlayers)"<<std::endl;
	assert(chosen_sl!=1 && chosen_sl!=3 && chosen_sl!=4); //4 means run using the two superlayers
    }
	
}

DTTrigPhase2Prod::~DTTrigPhase2Prod(){

    //delete inMuonPath;
    //delete outValidMuonPath;
    
    if(debug) std::cout<<"DTp2: calling destructor"<<std::endl;

    if(pinta){
	if(debug) std::cout<<"DTp2: writing histograms and files"<<std::endl;
	
	theFileOut->cd();

	for(int wh=-2;wh<=2;wh++){
	    int iwh=wh+2;
	    for(int st=1;st<=4;st++){
		int ist=st-1;
		for(int se=1;se<=14;se++){
		    int ise=se-1;
		    if(se>=13&&st!=4)continue;
		    for(int qu=1;qu<=9;qu++){
			int iqu=qu-1;

			//digi TIME
			TIMEPhase2[iwh][ist][ise][iqu]->Write();
			      
			//digiT0
			T0Phase2[iwh][ist][ise][iqu]->Write();
    
			//2D
			segment_vs_jm_x[iwh][ist][ise][iqu]->Write();
			  
			segment_vs_jm_x_gauss[iwh][ist][ise][iqu]->Write();

			segment_vs_jm_tanPhi[iwh][ist][ise][iqu]->Write();

			segment_vs_jm_tanPhi_gauss[iwh][ist][ise][iqu]->Write();
			  
			segment_vs_jm_T0[iwh][ist][ise][iqu]->Write();

			segment_vs_jm_T0_gauss[iwh][ist][ise][iqu]->Write();
			segment_vs_jm_T0_gauss_all[iwh][ist][ise][iqu]->Write();

			observed_tanPsi[iwh][ist][ise][iqu]->Write();
			
			all_observed_tanPsi[iwh][ist][ise][iqu]->Write();

			observed_x[iwh][ist][ise][iqu]->Write();
			
			all_observed_x[iwh][ist][ise][iqu]->Write();

			observed_t0[iwh][ist][ise][iqu]->Write();
			
			all_observed_t0[iwh][ist][ise][iqu]->Write();

			chi2[iwh][ist][ise][iqu]->Write();
			
			TPphi[iwh][ist][ise][iqu]->Write();
			TPphiB[iwh][ist][ise][iqu]->Write();

			MP_x_back[iwh][ist][ise][iqu]->Write();
			MP_psi_back[iwh][ist][ise][iqu]->Write();

		    }
		    expected_tanPsi[iwh][ist][ise]->Write();
		    expected_x[iwh][ist][ise]->Write();
		    expected_t0[iwh][ist][ise]->Write();
		}
	    }
	}

	Nsegments->Write();
	NmetaPrimitives->Write();
	NfilteredMetaPrimitives->Write();
	NcorrelatedMetaPrimitives->Write();
	Ngroups->Write();
	Nquality->Write();
	Nquality_matched->Write();
	Nsegosl->Write();
	Nsegosl31->Write();
	Nmd->Write();
	Nmd31->Write();
	Nhits_segment_tp->Write();
	
	theFileOut->Write();
	theFileOut->Close();

	delete Nsegments;
	delete NmetaPrimitives;
	delete NfilteredMetaPrimitives;
	delete NcorrelatedMetaPrimitives;
	delete Ngroups;
	delete Nquality;
	delete Nquality_matched;
	delete Nsegosl;
	delete Nsegosl31;
	delete Nmd;
	delete Nmd31;
	delete Nhits_segment_tp;
    
	for(int wh=-2;wh<=2;wh++){
	    int iwh=wh+2;
	    for(int st=1;st<=4;st++){
		int ist=st-1;
		for(int se=1;se<=14;se++){
		    int ise=se-1;
		    if(se>=13&&st!=4)continue;
		    for(int qu=1;qu<=9;qu++){
			int iqu=qu-1;
	    
			//digi TIME
			delete TIMEPhase2[iwh][ist][ise][iqu];
	    
			//digiT0
			delete T0Phase2[iwh][ist][ise][iqu];
	    
			//2D
			delete segment_vs_jm_x[iwh][ist][ise][iqu];    
			delete segment_vs_jm_x_gauss[iwh][ist][ise][iqu];
			delete segment_vs_jm_tanPhi[iwh][ist][ise][iqu];
			delete segment_vs_jm_tanPhi_gauss[iwh][ist][ise][iqu];
			delete segment_vs_jm_T0[iwh][ist][ise][iqu];
			delete segment_vs_jm_T0_gauss[iwh][ist][ise][iqu];
			delete segment_vs_jm_T0_gauss_all[iwh][ist][ise][iqu];
			delete observed_tanPsi[iwh][ist][ise][iqu];
			delete all_observed_tanPsi[iwh][ist][ise][iqu];
			delete observed_x[iwh][ist][ise][iqu];
			delete all_observed_x[iwh][ist][ise][iqu];
			delete observed_t0[iwh][ist][ise][iqu];
			delete all_observed_t0[iwh][ist][ise][iqu];
			
			delete chi2[iwh][ist][ise][iqu];
			delete TPphi[iwh][ist][ise][iqu];
			delete TPphiB[iwh][ist][ise][iqu];

			delete MP_x_back[iwh][ist][ise][iqu];
			delete MP_psi_back[iwh][ist][ise][iqu];
			
		    }
		    delete expected_tanPsi[iwh][ist][ise];
		    delete expected_x[iwh][ist][ise];
		    delete expected_t0[iwh][ist][ise];
		}
	    }
	}
    
	delete theFileOut; 
    }
  
  delete grouping_obj; // Grouping destructor
}


void DTTrigPhase2Prod::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if(debug) cout << "DTTrigPhase2Prod::beginRun " << iRun.id().run() << endl;
  if(debug) cout << "DTTrigPhase2Prod::beginRun: getting DT geometry" << endl;
    
  if(debug) std::cout<<"getting DT geometry"<<std::endl;
  iEventSetup.get<MuonGeometryRecord>().get(dtGeo);//1103
  
  ESHandle< DTConfigManager > dtConfig ;
  iEventSetup.get< DTConfigManagerRcd >().get( dtConfig );

  grouping_obj->initialise(iEventSetup); // Grouping object initialisation
  
    //trigGeomUtils = new DTTrigGeomUtils(dtGeo);

    //filling up zcn
    for (int ist=1; ist<=4; ++ist) {
	const DTChamberId chId(-2,ist,4);
	const DTChamber *chamb = dtGeo->chamber(chId);
	const DTSuperLayer *sl1 = chamb->superLayer(DTSuperLayerId(chId,1));
	const DTSuperLayer *sl3 = chamb->superLayer(DTSuperLayerId(chId,3));
	zcn[ist-1] = .5*(chamb->surface().toLocal(sl1->position()).z() + chamb->surface().toLocal(sl3->position()).z());
    }

    const DTChamber* chamb   = dtGeo->chamber(DTChamberId(-2,4,13));
    const DTChamber* scchamb = dtGeo->chamber(DTChamberId(-2,4,4));
    xCenter[0] = scchamb->toLocal(chamb->position()).x()*.5;
    chamb   = dtGeo->chamber(DTChamberId(-2,4,14));
    scchamb = dtGeo->chamber(DTChamberId(-2,4,10));
    xCenter[1] = scchamb->toLocal(chamb->position()).x()*.5;
}


void DTTrigPhase2Prod::produce(Event & iEvent, const EventSetup& iEventSetup){
    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByToken(dtDigisToken, dtdigis);
    
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    if(pinta){
	iEvent.getByToken(dt4DSegmentsToken, all4DSegments);
	if(debug) std::cout<<"DTp2: I got the segments"<<std::endl;
    }
    //int bx32 = iEvent.eventAuxiliary().bunchCrossing()*32;
    int bx25 = iEvent.eventAuxiliary().bunchCrossing()*25;
    timeFromP1ToP2 = iEvent.eventAuxiliary().bunchCrossing();

    if(debug) std::cout <<"\t Getting the RPC RecHits"<<std::endl;
    Handle<RPCRecHitCollection> rpcHits;
    iEvent.getByToken(rpcRecHitsLabel,rpcHits);

  
  //Santi's code
  // GROUPING BEGIN
  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtdigis->begin(); detUnitIt!=dtdigis->end(); ++detUnitIt) {
    const DTLayerId& layId               = (*detUnitIt).first;
    const DTChamberId chambId            = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = (*detUnitIt).second; 
    digiMap[chambId].put(range,layId);
  }

  // generate a list muon paths for each event!!!
  std::vector<MuonPath*> muonpaths;
    for (std::vector<const DTChamber*>::const_iterator ich = dtGeo->chambers().begin(); ich != dtGeo->chambers().end(); ich++) {
    const DTChamber* chamb  = (*ich);
    DTChamberId chid        = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);
    
//     if (dmit !=digiMap.end()) buildMuonPathCandidates((*dmit).second, &muonpaths);              // Old grouping implementation
    if (dmit !=digiMap.end()) grouping_obj->run(iEvent, iEventSetup, (*dmit).second, &muonpaths);  // New grouping implementation
  }

  digiMap.clear();
  // GROUPING ENDS

    if (debug) cout << "MUON PATHS found: " << muonpaths.size() <<" in event"<<iEvent.id().event()<<endl;
    if(pinta) Ngroups->Fill(muonpaths.size());
    
    //filtro por groupos de TDC times en las mismas celdas... corrobarar si sucede... esta implementacion no existe en software pero existe en firmware
    // loop over vector of muonpahts produced by grouping
    int iGroup=1;
    std::vector<metaPrimitive> metaPrimitives;
    for(auto muonpath = muonpaths.begin();muonpath!=muonpaths.end();++muonpath){
	DTPrimitive testPrim0((*muonpath)->getPrimitive(0));
	DTPrimitive testPrim1((*muonpath)->getPrimitive(1));
	DTPrimitive testPrim2((*muonpath)->getPrimitive(2));
	DTPrimitive testPrim3((*muonpath)->getPrimitive(3));

	if(debug){
	    std::cout<<"test_grouping:group # "<<iGroup<<std::endl;
	    std::cout<<"test_grouping:\t 0 "<<testPrim0.getChannelId()<<" "<<testPrim0.getTDCTime()<<std::endl;
	    std::cout<<"test_grouping:\t 1 "<<testPrim1.getChannelId()<<" "<<testPrim1.getTDCTime()<<std::endl;
	    std::cout<<"test_grouping:\t 2 "<<testPrim2.getChannelId()<<" "<<testPrim2.getTDCTime()<<std::endl;
	    std::cout<<"test_grouping:\t 3 "<<testPrim3.getChannelId()<<" "<<testPrim3.getTDCTime()<<std::endl;
	}

	int selected_Id=0;
	if(testPrim0.getTDCTime()!=-1) selected_Id= testPrim0.getCameraId();
	else if(testPrim1.getTDCTime()!=-1) selected_Id= testPrim1.getCameraId(); 
	else if(testPrim2.getTDCTime()!=-1) selected_Id= testPrim2.getCameraId(); 
	else if(testPrim3.getTDCTime()!=-1) selected_Id= testPrim3.getCameraId(); 
 

	DTLayerId thisLId(selected_Id);
	if(debug) std::cout<<"Building up MuonPathSLId from rawId in the Primitive"<<std::endl;
	DTSuperLayerId MuonPathSLId(thisLId.wheel(),thisLId.station(),thisLId.sector(),thisLId.superLayer());
	if(debug) std::cout<<"The MuonPathSLId is"<<MuonPathSLId<<std::endl;

	if(chosen_sl==4){
	    if(debug) std::cout<<"Generating Primitives with Both Superlayers 1 and 3"<<std::endl;
	    analyze(*muonpath,metaPrimitives,MuonPathSLId);
	}else if(thisLId.superLayer()==chosen_sl){
	    if(debug) std::cout<<"Generating Primitives only with SL "<<chosen_sl<<std::endl;
	    analyze(*muonpath,metaPrimitives,MuonPathSLId);
	}
	iGroup++;
    }
    
    if(debug) std::cout<<"filling NmetaPrimtives"<<std::endl;
    
    if(pinta) NmetaPrimitives->Fill(metaPrimitives.size());

    if(debug) std::cout<<"deleting muonpaths"<<std::endl;    
    for (unsigned int i=0; i<muonpaths.size(); i++){
      delete muonpaths[i];
    }
    muonpaths.clear();
    
    //FILTER SECTIONS:
    //filtro de duplicados puro popdr'ia ir ac'a mpredundantfilter.cpp primos?
    //filtro en |tanPhi|<~1.?
    //filtro de calidad por chi2 qualityenhancefilter.cpp mirar el metodo filter

    if(debug) std::cout<<"declaring new vector for filtered"<<std::endl;    

    std::vector<metaPrimitive> filteredMetaPrimitives;

    if(filter_primos){

	if(debug) std::cout<<"filtering: starting primos filtering"<<std::endl;    
    
	int primo_index=0;
	bool oneof4=false;
	//    for (auto metaPrimitiveIt = metaPrimitives.begin(); metaPrimitiveIt != metaPrimitives.end(); ++metaPrimitiveIt){
	if(metaPrimitives.size()==1){
	    if(debug){
		std::cout<<"filtering:";
		printmP(metaPrimitives[0]);
		std::cout<<" \t is:"<<0<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
	    }
	    if(fabs(metaPrimitives[0].tanPhi)<tanPhiTh){
		filteredMetaPrimitives.push_back(metaPrimitives[0]);
		if(debug)std::cout<<"filtering: kept1 i="<<0<<std::endl;
	    }
	}
	else for(int i=1; i<int(metaPrimitives.size()); i++){ 
		if(fabs(metaPrimitives[i].tanPhi)>tanPhiTh) continue;
		if(rango(metaPrimitives[i])==4)oneof4=true;
		if(debug){
		    std::cout<<"filtering:";
		    printmP(metaPrimitives[i]);
		    std::cout<<" \t is:"<<i<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
		}
		if(arePrimos(metaPrimitives[i],metaPrimitives[i-1])!=0  and arePrimos(metaPrimitives[i],metaPrimitives[i-primo_index-1])!=0){
		    primo_index++;
		}else{
		    if(primo_index==0){
			filteredMetaPrimitives.push_back(metaPrimitives[i]);
			if(debug)std::cout<<"filtering: kept2 i="<<i<<std::endl;
		    }else{
			if(oneof4){
			    double minchi2=99999;
			    int selected_i=0;
			    for(int j=i-1;j>=i-primo_index-1;j--){
				if(rango(metaPrimitives[j])!=4) continue;
				if(minchi2>metaPrimitives[j].chi2){
				    minchi2=metaPrimitives[j].chi2;
				    selected_i=j;
				}
			    }
			    filteredMetaPrimitives.push_back(metaPrimitives[selected_i]);
			    if(debug)std::cout<<"filtering: kept4 i="<<selected_i<<std::endl;
			}else{
			    for(int j=i-1;j>=i-primo_index-1;j--){
				filteredMetaPrimitives.push_back(metaPrimitives[j]);
				if(debug)std::cout<<"filtering: kept3 i="<<j<<std::endl;
			    }
			}
		    }
		    primo_index=0;
		    oneof4=false;
		}
	    }
    }else{
	for (size_t i=0; i<metaPrimitives.size(); i++){ 
	    if(fabs(metaPrimitives[i].tanPhi)>tanPhiTh) continue;
	    filteredMetaPrimitives.push_back(metaPrimitives[i]); 
	}
    }
    
    metaPrimitives.clear();
    metaPrimitives.erase(metaPrimitives.begin(),metaPrimitives.end());

    if(pinta) NfilteredMetaPrimitives->Fill(filteredMetaPrimitives.size());
    
    
    if(debug) std::cout<<"filteredMetaPrimitives: starting correlations"<<std::endl;    
    if(!do_correlation){
	if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()<<" we found "<<filteredMetaPrimitives.size()<<" filteredMetaPrimitives (superlayer)"<<std::endl;
	vector<L1MuDTChambPhDigi> outPhi;
	vector<L1MuDTChambDigi> outP2;
	vector<L1Phase2MuDTPhDigi> outP2Ph;

	for (auto metaPrimitiveIt = filteredMetaPrimitives.begin(); metaPrimitiveIt != filteredMetaPrimitives.end(); ++metaPrimitiveIt){
	    DTSuperLayerId slId((*metaPrimitiveIt).rawId);
	    if(debug) std::cout<<"looping in final vector: SuperLayerId"<<slId<<" x="<<(*metaPrimitiveIt).x<<" quality="<<(*metaPrimitiveIt).quality<<std::endl;
	    
	    int sectorTP=slId.sector();
	    if(sectorTP==13) sectorTP=4;
	    if(sectorTP==14) sectorTP=10;
	    sectorTP=sectorTP-1;
	  
	    if(p2_df==0){
		outPhi.push_back(L1MuDTChambPhDigi((*metaPrimitiveIt).t0,
						   slId.wheel(),
						   sectorTP,
						   slId.station(),
						   (int)round((*metaPrimitiveIt).phi*65536./0.8),
						   (int)round((*metaPrimitiveIt).phiB*2048./1.4),
						   (*metaPrimitiveIt).quality,
						   1,
						   0
						   ));
	    }else if(p2_df==1){
		if(debug)std::cout<<"pushing back phase-2 dataformat agreement with Oscar for comparison with slice test"<<std::endl;
		outP2.push_back(L1MuDTChambDigi((int)round((*metaPrimitiveIt).t0/25.),   // ubx (m_bx) //bx en la orbita
						slId.wheel(),   // uwh (m_wheel)     
						sectorTP,   // usc (m_sector)    
						slId.station(),   // ust (m_station)
						(int)round((*metaPrimitiveIt).x*1000),   // uphi (_phiAngle)
						(int)round((*metaPrimitiveIt).tanPhi*4096),   // uphib (m_phiBending)
						0,   // uz (m_zCoordinate)
						0,   // uzsl (m_zSlope)
						(*metaPrimitiveIt).quality,  // uqua (m_qualityCode)
						0,  // uind (m_segmentIndex)
						(int)round((*metaPrimitiveIt).t0),  // ut0 (m_t0Segment)
						(int)round((*metaPrimitiveIt).chi2),  // uchi2 (m_chi2Segment)
						-10    // urpc (m_rpcFlag)
						));
	    }else if(p2_df==3){
		if(debug)std::cout<<"pushing back phase-2 dataformat carlo-federica dataformat"<<std::endl;
		outP2Ph.push_back(L1Phase2MuDTPhDigi((int)round((*metaPrimitiveIt).t0/25.),   // ubx (m_bx) //bx en la orbita
						     slId.wheel(),   // uwh (m_wheel)     // FIXME: It is not clear who provides this?
						     sectorTP,   // usc (m_sector)    // FIXME: It is not clear who provides this?
						     slId.station(),   // ust (m_station)
						     (int)round((*metaPrimitiveIt).phi*65536./0.8), // uphi (_phiAngle)
						     (int)round((*metaPrimitiveIt).phiB*2048./1.4), // uphib (m_phiBending)
						     (*metaPrimitiveIt).quality,  // uqua (m_qualityCode)
						     0,  // uind (m_segmentIndex)
						     (int)round((*metaPrimitiveIt).t0),  // ut0 (m_t0Segment)
						     (int)round((*metaPrimitiveIt).chi2),  // uchi2 (m_chi2Segment)
						     -10    // urpc (m_rpcFlag)
						     ));
		
	    }  
	}		

	if(p2_df==0){
	    std::unique_ptr<L1MuDTChambPhContainer> resultPhi (new L1MuDTChambPhContainer);
	    resultPhi->setContainer(outPhi); iEvent.put(std::move(resultPhi));
	    outPhi.clear();
	    outPhi.erase(outPhi.begin(),outPhi.end());
	}else if(p2_df==1){
	    std::unique_ptr<L1MuDTChambContainer> resultP2 (new L1MuDTChambContainer);
	    resultP2->setContainer(outP2); iEvent.put(std::move(resultP2));
	    outP2.clear();
	    outP2.erase(outP2.begin(),outP2.end());
	}else if(p2_df==2){
	    std::unique_ptr<L1Phase2MuDTPhContainer> resultP2Ph (new L1Phase2MuDTPhContainer);
	    resultP2Ph->setContainer(outP2Ph); iEvent.put(std::move(resultP2Ph));
	    outP2Ph.clear();
	    outP2Ph.erase(outP2Ph.begin(),outP2Ph.end());
	}
	    
    }else{
	//Silvia's code for correlationg filteredMetaPrimitives
	
	if(debug) std::cout<<"starting correlation"<<std::endl;
	
	std::vector<metaPrimitive> correlatedMetaPrimitives;
	
	for(int wh=-2;wh<=2;wh++){
	    for(int st=1;st<=4;st++){
		for(int se=1;se<=14;se++){
		    if(se>=13&&st!=4)continue;
		    
		    DTChamberId ChId(wh,st,se);
		    DTSuperLayerId sl1Id(wh,st,se,1);
		    DTSuperLayerId sl3Id(wh,st,se,3);
	      
		    //filterSL1
		    std::vector<metaPrimitive> SL1metaPrimitives;
		    for(auto metaprimitiveIt = filteredMetaPrimitives.begin();metaprimitiveIt!=filteredMetaPrimitives.end();++metaprimitiveIt)
			if(metaprimitiveIt->rawId==sl1Id.rawId())
			    SL1metaPrimitives.push_back(*metaprimitiveIt);
	      
		    //filterSL3
		    std::vector<metaPrimitive> SL3metaPrimitives;
		    for(auto metaprimitiveIt = filteredMetaPrimitives.begin();metaprimitiveIt!=filteredMetaPrimitives.end();++metaprimitiveIt)
			if(metaprimitiveIt->rawId==sl3Id.rawId())
			    SL3metaPrimitives.push_back(*metaprimitiveIt);
		    
		    if(SL1metaPrimitives.size()==0 and SL3metaPrimitives.size()==0) continue;
		    
		    if(debug) std::cout<<"correlating "<<SL1metaPrimitives.size()<<" metaPrim in SL1 and "<<SL3metaPrimitives.size()<<" in SL3 for "<<sl3Id<<std::endl;

		    bool at_least_one_correlation=false;

		    //SL1-SL3

		    for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
			    if(fabs(SL1metaPrimitive->t0-SL3metaPrimitive->t0) < dT0_correlate_TP){//time match
				double PosSL1=SL1metaPrimitive->x;
				double PosSL3=SL3metaPrimitive->x;
				double NewSlope=(PosSL1-PosSL3)/23.5;     
				double MeanT0=(SL1metaPrimitive->t0+SL3metaPrimitive->t0)/2;
				double MeanPos=(PosSL3+PosSL1)/2;
				double newChi2=(SL1metaPrimitive->chi2+SL3metaPrimitive->chi2)*0.5;//to be recalculated
				int quality = 0;
				if(SL3metaPrimitive->quality <= 2 and SL1metaPrimitive->quality <=2) quality=6;

				if((SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality <=2)
				   or (SL1metaPrimitive->quality >= 3 && SL3metaPrimitive->quality <=2) ) quality=8;

				if(SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality >=3) quality=9;
			  
				GlobalPoint jm_x_cmssw_global = dtGeo->chamber(ChId)->toGlobal(LocalPoint(MeanPos,0.,0.));//jm_x is already extrapolated to the middle of the SL
				int thisec = ChId.sector();
				if(se==13) thisec = 4;
				if(se==14) thisec = 10;
				double phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
				double psi=TMath::ATan(NewSlope);
				double phiB=hasPosRF(ChId.wheel(),ChId.sector()) ? psi-phi :-psi-phi ;
			
				correlatedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),MeanT0,MeanPos,NewSlope,phi,phiB,newChi2,quality,
						SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,
						SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,
						SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,
						SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,
						SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,
						SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,
						SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,
						SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4
						}));
				at_least_one_correlation=true;
			    }
			}
			
			if(at_least_one_correlation==false){//no correlation was found, trying with pairs of two digis in the other SL
			    
			    int matched_digis=0;
			    double minx=minx_match_2digis;
			    int best_tdc=-1;
			    int next_tdc=-1;
			    int best_wire=-1;
			    int next_wire=-1;
			    int best_layer=-1;
			    int next_layer=-1;

			    for (auto dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
				const DTLayerId dtLId = (*dtLayerId_It).first;
				DTSuperLayerId dtSLId(dtLId);
				if(dtSLId.rawId()!=sl3Id.rawId()) continue;
				double l_shift=0;
				if(dtLId.layer()==4)l_shift=1.95;
				if(dtLId.layer()==3)l_shift=0.65;
				if(dtLId.layer()==2)l_shift=-0.65;
				if(dtLId.layer()==1)l_shift=-1.95;
				double x_inSL3=SL1metaPrimitive->x-SL1metaPrimitive->tanPhi*(23.5+l_shift);
				for (auto digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){
				    DTWireId wireId(dtLId,(*digiIt).wire());
				    int x_wire = shiftinfo[wireId.rawId()]+((*digiIt).time()-SL1metaPrimitive->t0)*0.00543; 
				    int x_wire_left = shiftinfo[wireId.rawId()]-((*digiIt).time()-SL1metaPrimitive->t0)*0.00543; 
				    if(fabs(x_inSL3-x_wire)>fabs(x_inSL3-x_wire_left)) x_wire=x_wire_left; //choose the closest laterality
				    if(pinta) Nsegosl->Fill(x_inSL3-x_wire);
				    if(fabs(x_inSL3-x_wire)<minx){
					minx=fabs(x_inSL3-x_wire);
					next_wire=best_wire;
					next_tdc=best_tdc;
					next_layer=best_layer;
					
					best_wire=(*digiIt).wire();
					best_tdc=(*digiIt).time();
					best_layer=dtLId.layer();
					matched_digis++;
				    }
				}
				
			    }
			    if(pinta) Nmd->Fill(matched_digis);    
			    if(matched_digis>=2 and best_layer!=-1 and next_layer!=-1){
				int new_quality=7;
				if(SL1metaPrimitive->quality<=2) new_quality=5;

				int wi1=-1;int tdc1=-1;
				int wi2=-1;int tdc2=-1;
				int wi3=-1;int tdc3=-1;
				int wi4=-1;int tdc4=-1;
				
				if(next_layer==1) {wi1=next_wire; tdc1=next_tdc; }
				if(next_layer==2) {wi2=next_wire; tdc2=next_tdc; }
				if(next_layer==3) {wi3=next_wire; tdc3=next_tdc; }
				if(next_layer==4) {wi4=next_wire; tdc4=next_tdc; }

				if(best_layer==1) {wi1=best_wire; tdc1=best_tdc; }
				if(best_layer==2) {wi2=best_wire; tdc2=best_tdc; }
				if(best_layer==3) {wi3=best_wire; tdc3=best_tdc; }
				if(best_layer==4) {wi4=best_wire; tdc4=best_tdc; } 
				
				

				correlatedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),SL1metaPrimitive->t0,SL1metaPrimitive->x,SL1metaPrimitive->tanPhi,SL1metaPrimitive->phi,SL1metaPrimitive->phiB,SL1metaPrimitive->chi2,
						new_quality,
						SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,
						SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,
						SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,
						SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,
						wi1,tdc1,
						wi2,tdc2,
						wi3,tdc3,
						wi4,tdc4
						}));
				at_least_one_correlation=true;
			    }
			}
		    }

		    //finish SL1-SL3

		    //SL3-SL1
		    for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
			for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			    if(fabs(SL1metaPrimitive->t0-SL3metaPrimitive->t0) < dT0_correlate_TP){//time match
				//this comb was already filled up in the previous loop now we just want to know if there was at least one match
				at_least_one_correlation=true;
			    }
			}
			
			if(at_least_one_correlation==false){//no correlation was found, trying with pairs of two digis in the other SL
				
			    int matched_digis=0;
			    double minx=minx_match_2digis;
			    int best_tdc=-1;
			    int next_tdc=-1;
			    int best_wire=-1;
			    int next_wire=-1;
			    int best_layer=-1;
			    int next_layer=-1;
				
			    for (auto dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
				const DTLayerId dtLId = (*dtLayerId_It).first;
				DTSuperLayerId dtSLId(dtLId);
				if(dtSLId.rawId()!=sl1Id.rawId()) continue;
				double l_shift=0;
				if(dtLId.layer()==4)l_shift=1.95;
				if(dtLId.layer()==3)l_shift=0.65;
				if(dtLId.layer()==2)l_shift=-0.65;
				if(dtLId.layer()==1)l_shift=-1.95;
				double x_inSL1=SL3metaPrimitive->x+SL3metaPrimitive->tanPhi*(23.5-l_shift);
				for (auto digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){
				    DTWireId wireId(dtLId,(*digiIt).wire());
				    int x_wire = shiftinfo[wireId.rawId()]+((*digiIt).time()-SL3metaPrimitive->t0)*0.00543; 
				    int x_wire_left = shiftinfo[wireId.rawId()]-((*digiIt).time()-SL3metaPrimitive->t0)*0.00543; 
				    if(fabs(x_inSL1-x_wire)>fabs(x_inSL1-x_wire_left)) x_wire=x_wire_left; //choose the closest laterality
				    if(pinta) Nsegosl31->Fill(x_inSL1-x_wire);
				    if(fabs(x_inSL1-x_wire)<minx){
					minx=fabs(x_inSL1-x_wire);
					next_wire=best_wire;
					next_tdc=best_tdc;
					next_layer=best_layer;
					    
					best_wire=(*digiIt).wire();
					best_tdc=(*digiIt).time();
					best_layer=dtLId.layer();
					matched_digis++;
				    }
				}
				    
			    }
			    if(pinta) Nmd31->Fill(matched_digis);    
			    if(matched_digis>=2 and best_layer!=-1 and next_layer!=-1){
				int new_quality=7;
				if(SL3metaPrimitive->quality<=2) new_quality=5;
				    
				int wi1=-1;int tdc1=-1;
				int wi2=-1;int tdc2=-1;
				int wi3=-1;int tdc3=-1;
				int wi4=-1;int tdc4=-1;
				    
				if(next_layer==1) {wi1=next_wire; tdc1=next_tdc; }
				if(next_layer==2) {wi2=next_wire; tdc2=next_tdc; }
				if(next_layer==3) {wi3=next_wire; tdc3=next_tdc; }
				if(next_layer==4) {wi4=next_wire; tdc4=next_tdc; }
				    
				if(best_layer==1) {wi1=best_wire; tdc1=best_tdc; }
				if(best_layer==2) {wi2=best_wire; tdc2=best_tdc; }
				if(best_layer==3) {wi3=best_wire; tdc3=best_tdc; }
				if(best_layer==4) {wi4=best_wire; tdc4=best_tdc; } 
				    
				    
				    
				correlatedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),SL3metaPrimitive->t0,SL3metaPrimitive->x,SL3metaPrimitive->tanPhi,SL3metaPrimitive->phi,SL3metaPrimitive->phiB,SL3metaPrimitive->chi2,
						new_quality,
						wi1,tdc1,
						wi2,tdc2,
						wi3,tdc3,
						wi4,tdc4,
						SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,
						SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,
						SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,
						SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4
						}));
				at_least_one_correlation=true;
			    }
			}
		    }
		
		    //finish SL3-SL1

		    if(at_least_one_correlation==false){
			if(debug) std::cout<<"correlation we found zero correlations, adding both collections as they are to the correlatedMetaPrimitives"<<std::endl;
			if(debug) std::cout<<"correlation sizes:"<<SL1metaPrimitives.size()<<" "<<SL3metaPrimitives.size()<<std::endl;
			for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			    DTSuperLayerId SLId(SL1metaPrimitive->rawId);
			    DTChamberId(SLId.wheel(),SLId.station(),SLId.sector());
			    correlatedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),SL1metaPrimitive->t0,SL1metaPrimitive->x,SL1metaPrimitive->tanPhi,SL1metaPrimitive->phi,SL1metaPrimitive->phiB,SL1metaPrimitive->chi2,SL1metaPrimitive->quality,
					    SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,
					    SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,
					    SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,
					    SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,
					    -1,-1,
					    -1,-1,
					    -1,-1,
					    -1,-1
					    }));
			}
			for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
			    DTSuperLayerId SLId(SL3metaPrimitive->rawId);
			    DTChamberId(SLId.wheel(),SLId.station(),SLId.sector());
			    correlatedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),SL3metaPrimitive->t0,SL3metaPrimitive->x,SL3metaPrimitive->tanPhi,SL3metaPrimitive->phi,SL3metaPrimitive->phiB,SL3metaPrimitive->chi2,SL3metaPrimitive->quality,
					    -1,-1,
					    -1,-1,
					    -1,-1,
					    -1,-1,
					    SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,
					    SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,
					    SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,
					    SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4
					    }));
			}
		    }

		    SL1metaPrimitives.clear();
		    SL1metaPrimitives.erase(SL1metaPrimitives.begin(),SL1metaPrimitives.end());
		    SL3metaPrimitives.clear();
		    SL3metaPrimitives.erase(SL3metaPrimitives.begin(),SL3metaPrimitives.end());
		}
	    }
	}
	if(pinta) NcorrelatedMetaPrimitives->Fill(correlatedMetaPrimitives.size());

	filteredMetaPrimitives.clear();
	filteredMetaPrimitives.erase(filteredMetaPrimitives.begin(),filteredMetaPrimitives.end());

	if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()<<" we found "<<correlatedMetaPrimitives.size()<<" correlatedMetPrimitives (chamber)"<<std::endl;
	

	//Trying confirmation by RPCs
	
	//for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){
	//}
	
	vector<L1MuDTChambPhDigi> outPhiCH;
	vector<L1MuDTChambDigi> outP2CH;
	vector<L1Phase2MuDTPhDigi> outP2PhCH;
	
	for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){
	    DTChamberId chId((*metaPrimitiveIt).rawId);
	    if(debug) std::cout<<"looping in final vector: SuperLayerId"<<chId<<" x="<<(*metaPrimitiveIt).x<<" quality="<<(*metaPrimitiveIt).quality<<std::endl;

	    int sectorTP=chId.sector();
	    if(sectorTP==13) sectorTP=4;
	    if(sectorTP==14) sectorTP=10;
	    sectorTP=sectorTP-1;
	    
	    L1MuDTChambPhDigi thisTP((*metaPrimitiveIt).t0,
				      chId.wheel(),
				      sectorTP,
				      chId.station(),
				      (int)round((*metaPrimitiveIt).phi*65536./0.8),
				      (int)round((*metaPrimitiveIt).phiB*2048./1.4),
				      (*metaPrimitiveIt).quality,
				      1,
				      0
				      );
	    
	    if(pinta){
		all_observed_tanPsi[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).tanPhi);
		all_observed_x[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).x);
		all_observed_t0[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).t0);
 		chi2[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).chi2);
 		TPphi[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).phi);
 		TPphiB[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).phiB);
		
		double x_back = trigPos((*metaPrimitiveIt));
		double psi_back = trigDir((*metaPrimitiveIt));
		
		MP_x_back[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill((*metaPrimitiveIt).x,x_back);
 		MP_psi_back[chId.wheel()+2][chId.station()-1][chId.sector()-1][(*metaPrimitiveIt).quality-1]->Fill( TMath::ATan((*metaPrimitiveIt).tanPhi) ,psi_back);

		if(debug)std::cout<<"back:(x,x_back)= "<<(*metaPrimitiveIt).x<<","<<x_back<<std::endl;
		if(debug)std::cout<<"back:(psi,psi_back)= "<<TMath::ATan((*metaPrimitiveIt).tanPhi)<<","<<psi_back<<std::endl;
	    }

	    
	    if(p2_df==0){
		outPhiCH.push_back(thisTP);
	    }else if(p2_df==1){
		if(debug)std::cout<<"pushing back slice-test dataformat"<<std::endl;
		
		outP2CH.push_back(L1MuDTChambDigi((int)round((*metaPrimitiveIt).t0/25.),
						  chId.wheel(),
						  sectorTP,
						  chId.station(),
						  (int)round((*metaPrimitiveIt).phi*65536./0.8),
						  (int)round((*metaPrimitiveIt).phiB*2048./1.4),
						  0,
						  0,
						  (*metaPrimitiveIt).quality,
						  0,
						  (int)round((*metaPrimitiveIt).t0),
						  (int)round((*metaPrimitiveIt).chi2),
						  -10
						  ));
	    }else if(p2_df==2){
		if(debug)std::cout<<"pushing back carlo-federica dataformat"<<std::endl;
		
		outP2PhCH.push_back(L1Phase2MuDTPhDigi((int)round((*metaPrimitiveIt).t0/25.),
						       chId.wheel(),
						       sectorTP,
						       chId.station(),
						       (int)round((*metaPrimitiveIt).phi*65536./0.8),
						       (int)round((*metaPrimitiveIt).phiB*2048./1.4),
						       (*metaPrimitiveIt).quality,
						       0,
						       (int)round((*metaPrimitiveIt).t0),
						       (int)round((*metaPrimitiveIt).chi2),
						       -10
						       ));
		
	    }
	}

	if(p2_df==0){ 
	    std::unique_ptr<L1MuDTChambPhContainer> resultPhiCH (new L1MuDTChambPhContainer);
	    resultPhiCH->setContainer(outPhiCH); iEvent.put(std::move(resultPhiCH));
	    outPhiCH.clear();
	    outPhiCH.erase(outPhiCH.begin(),outPhiCH.end());
	}else if(p2_df==1){
	    std::unique_ptr<L1MuDTChambContainer> resultP2CH (new L1MuDTChambContainer);
	    resultP2CH->setContainer(outP2CH); iEvent.put(std::move(resultP2CH));
	    outP2CH.clear();
	    outP2CH.erase(outP2CH.begin(),outP2CH.end());
	}else if(p2_df==2){
	    std::unique_ptr<L1Phase2MuDTPhContainer> resultP2PhCH (new L1Phase2MuDTPhContainer);
	    resultP2PhCH->setContainer(outP2PhCH); iEvent.put(std::move(resultP2PhCH));
	    outP2PhCH.clear();
	    outP2PhCH.erase(outP2PhCH.begin(),outP2PhCH.end());
	}

	if(pinta){

	    //ploting all qualities before correlation
	    for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){
		Nquality->Fill(metaPrimitiveIt->quality);
	    }

	    DTRecSegment4DCollection::const_iterator segment;

	    if(pinta) Nsegments->Fill(all4DSegments->size());
	    
	    //if(debug) std::cout<<"min_phinhits_match_segment="<<min_phinhits_match_segment<<std::endl;
	    for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
		if(!segment->hasPhi()) continue;
		if(int(segment->phiSegment()->recHits().size())<min_phinhits_match_segment) continue;
		DTChamberId chId(segment->chamberId());
		
	  
		//filter CH correlated MP
		std::vector<metaPrimitive> CHmetaPrimitives;
		for(auto metaprimitiveIt = correlatedMetaPrimitives.begin();metaprimitiveIt!=correlatedMetaPrimitives.end();++metaprimitiveIt)
		    if(metaprimitiveIt->rawId==chId.rawId())
			CHmetaPrimitives.push_back(*metaprimitiveIt);
	  
		if(debug) std::cout<<"plots: In Chamber "<<chId<<" we have a phi segment and "<<CHmetaPrimitives.size()<<" correlatedMetaPrimitives"<<std::endl;
		if(CHmetaPrimitives.size()==0)continue;
	
		//T0
		double segment_t0=segment->phiSegment()->t0();
		double segment_t0Phase2=segment_t0+bx25;
	  
		//tanPhi
		LocalVector segmentDirection=segment->localDirection();
		double dx=segmentDirection.x();          
		double dz=segmentDirection.z();          
		double segment_tanPhi=dx/dz;          
		//cassert(TMath::ATan(segment_tanPhi)==TMath::ACos(dx));

		//x
		LocalPoint segmentPosition= segment->localPosition();
		//if(debug) std::cout<<"building wireId inside sl loop wire="<<1<<std::endl;
		//DTWireId wireId(wh,st,se,sl,2,1);//sl,la,wi          
		double segment_x=segmentPosition.x();          
		
		int i=-1;
		double minT=9999;
	      
		for(auto metaprimitiveIt = CHmetaPrimitives.begin();metaprimitiveIt!=CHmetaPrimitives.end();++metaprimitiveIt){
		    double deltaT0=metaprimitiveIt->t0-segment_t0Phase2;
		    if(fabs(deltaT0)<minT){
			i=std::distance(CHmetaPrimitives.begin(),metaprimitiveIt);
			minT=fabs(deltaT0);
		    }
		}

		int iwh=chId.wheel()+2;
		int ist=chId.station()-1;
		int ise=chId.sector()-1;
		int iqu=CHmetaPrimitives[i].quality-1;
	      
		expected_tanPsi[iwh][ist][ise]->Fill(segment_tanPhi);
		expected_x[iwh][ist][ise]->Fill(segment_x);
		expected_t0[iwh][ist][ise]->Fill(segment_t0Phase2);
		
		double z1=11.75;
		double z3=-1.*z1;
		//if (chId.station == 3 or chId.station == 4){
		//z1=9.95;
		//z3=-13.55;
		//}

		if (chId.station()==3 or chId.station()==4) segment_x = segment_x-segment_tanPhi*1.8; //extrapolating segment position from chamber reference frame to chamber middle SL plane in MB3&MB4
		
		if(!(CHmetaPrimitives[i].quality == 9 or CHmetaPrimitives[i].quality == 8 or CHmetaPrimitives[i].quality == 6)){
		    if(inner(CHmetaPrimitives[i])) segment_x = segment_x+segment_tanPhi*z1;
		    if(outer(CHmetaPrimitives[i])) segment_x = segment_x+segment_tanPhi*z3;
		}
		
		if(minT<min_dT0_match_segment){//the closest segment should be within min_dT0_match_segment 
		    observed_tanPsi[iwh][ist][ise][iqu]->Fill(segment_tanPhi);
		    observed_x[iwh][ist][ise][iqu]->Fill(segment_x);
		    observed_t0[iwh][ist][ise][iqu]->Fill(segment_t0Phase2);
		    
		    if(debug) std::cout<<"seg mpm "<<chId<<" -> "
				       <<segment_x<<" "<<CHmetaPrimitives[i].x<<" "	  
				       <<segment_tanPhi<<" "<<CHmetaPrimitives[i].tanPhi<<" "	    
				       <<segment_t0Phase2<<" "<<CHmetaPrimitives[i].t0<<" "<<std::endl;	  
		    
		    //correlation and matched plots
		    segment_vs_jm_x[iwh][ist][ise][iqu]->Fill(segment_x,CHmetaPrimitives[i].x);	  
		    segment_vs_jm_tanPhi[iwh][ist][ise][iqu]->Fill(segment_tanPhi,CHmetaPrimitives[i].tanPhi);
		    segment_vs_jm_T0[iwh][ist][ise][iqu]->Fill(segment_t0Phase2,CHmetaPrimitives[i].t0);
		    
		    segment_vs_jm_x_gauss[iwh][ist][ise][iqu]->Fill(segment_x-CHmetaPrimitives[i].x);
		    segment_vs_jm_tanPhi_gauss[iwh][ist][ise][iqu]->Fill(segment_tanPhi-CHmetaPrimitives[i].tanPhi);
		    segment_vs_jm_T0_gauss[iwh][ist][ise][iqu]->Fill(segment_t0Phase2-CHmetaPrimitives[i].t0);
		    segment_vs_jm_T0_gauss_all[iwh][ist][ise][iqu]->Fill(segment_t0Phase2-CHmetaPrimitives[i].t0);

		    Nquality_matched->Fill(CHmetaPrimitives[i].quality);
		    Nhits_segment_tp->Fill(segment->phiSegment()->recHits().size(),CHmetaPrimitives[i].quality);
		}else{
		    //segment could not be matched
		    if(debug) std::cout<<segment_x<<" "<<segment_tanPhi<<" "<<segment_t0Phase2<<" "<<std::endl;
		    segment_vs_jm_T0_gauss_all[iwh][ist][ise][iqu]->Fill(segment_t0Phase2-CHmetaPrimitives[i].t0);
		    Nhits_segment_tp->Fill(segment->phiSegment()->recHits().size(),0);
		    if(segment->phiSegment()->recHits().size()==4)
			if(debug)std::cout<<chId<<" ineficient event with 4 hits segments in event"<<iEvent.id().event()<<endl;

		}
	    }
	    
	    correlatedMetaPrimitives.clear();
	    correlatedMetaPrimitives.erase(correlatedMetaPrimitives.begin(),correlatedMetaPrimitives.end());
	}
    }
}

void DTTrigPhase2Prod::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  grouping_obj->finish();
};


const int DTTrigPhase2Prod::LAYER_ARRANGEMENTS[MAX_VERT_ARRANG][3] = {
    {0, 1, 2}, {1, 2, 3},                       // Grupos consecutivos
    {0, 1, 3}, {0, 2, 3}                        // Grupos salteados
};



void DTTrigPhase2Prod::setInChannels(DTDigiCollection *digis, int sl){
    //  if (digis->isValid()) return; 
  
    // before setting channels we need to clear
    for (int lay = 0; lay < NUM_LAYERS; lay++)  {
	for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
	    //      if (debug) cout << "DTp2::setInChannels --> emptying L" << lay << " Ch" << ch  << " with content " << channelIn[lay][ch].size() << endl;      
	    channelIn[lay][ch].clear();
	}
    }

    //  if (debug) cout << "DTp2::setInChannels --> initialised with empty vectors" << endl;
    // now fill with those primitives that makes sense: 
    DTDigiCollection::DigiRangeIterator dtLayerId_It;  
    for (dtLayerId_It=digis->begin(); dtLayerId_It!=digis->end(); ++dtLayerId_It){
	const DTLayerId dtLId = (*dtLayerId_It).first;
	if (dtLId.superlayer() != sl+1) continue;  //skip digis not in SL... 
    
	for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){	  
	    int layer = dtLId.layer()-1;
	    int wire = (*digiIt).wire()-1;
	    int digiTIME = (*digiIt).time();
	    //if(txt_ttrig_bc0) digiTIME = digiTIME -ttriginfo[thisWireId.rawId()];
	    int digiTIMEPhase2 =  digiTIME;
	    //if(txt_ttrig_bc0) digiTIMEPhase2 = digiTIMEPhase2 + bx25;//correction done in previous step to be updated!

      
	    // if (debug) cout << "DTp2::setInChannels --> reading digis in L"<<layer << " Ch" << wire << endl;
      
	    DTPrimitive dtpAux = DTPrimitive();
	    dtpAux.setTDCTime(digiTIMEPhase2); 
	    dtpAux.setChannelId(wire);	    // NOT SURE IF THE WANT TO INCREASE THIS VALUE BY ONE OR NOT
	    dtpAux.setLayerId(layer);	    //  L=0,1,2,3      
	    dtpAux.setSuperLayerId(sl);	    // SL=0,1,2
	    dtpAux.setCameraId(dtLId.rawId()); //
	    channelIn[layer][wire].push_back(dtpAux);
	}
    }
}
void DTTrigPhase2Prod::selectInChannels(int baseChannel) {
      
    /*
      Channels are labeled following next schema:
      Input Muxer Indexes
      ---------------------------------
      |   6   |   7   |   8   |   9   |
      ---------------------------------
      |   3   |   4   |   5   |
      -------------------------
      |   1   |   2   |
      -----------------
      |   0   |
      ---------
    */
    //  if (debug) cout << "DTp2::selectInChannels --> for baseChannel: " << baseChannel << endl;
    /*
****** CAPA 0 ******
*/
    muxInChannels[0] = channelIn[0][baseChannel];
    /*
****** CAPA 1 ******
*/
    muxInChannels[1] = channelIn[1][baseChannel];

    if (baseChannel + 1 < NUM_CH_PER_LAYER)
	muxInChannels[2] = channelIn[1][baseChannel + 1];
    else
	muxInChannels[2] = chInDummy;
    /*
****** CAPA 2 ******
*/
    if (baseChannel - 1 >= 0)
	muxInChannels[3] = channelIn[2][baseChannel - 1];
    else
	muxInChannels[3] = chInDummy;

    muxInChannels[4] = channelIn[2][baseChannel];

    if (baseChannel + 1 < NUM_CH_PER_LAYER)
	muxInChannels[5] = channelIn[2][baseChannel + 1];
    else
	muxInChannels[5] = chInDummy;
    /*
****** CAPA 3 ******
*/
    if (baseChannel - 1 >= 0)
	muxInChannels[6] = channelIn[3][baseChannel - 1];
    else
	muxInChannels[6] = chInDummy;

    muxInChannels[7] = channelIn[3][baseChannel];

    if (baseChannel + 1 < NUM_CH_PER_LAYER)
	muxInChannels[8] = channelIn[3][baseChannel + 1];
    else
	muxInChannels[8] = chInDummy;

    if (baseChannel + 2 < NUM_CH_PER_LAYER)
	muxInChannels[9] = channelIn[3][baseChannel + 2];
    else
	muxInChannels[9] = chInDummy;

}

bool DTTrigPhase2Prod::notEnoughDataInChannels(void) {
  
    // Indicadores de "layer" empty.
    bool lEmpty[4];
  
    lEmpty[0] = muxInChannels[0].empty();
  
    lEmpty[1] = muxInChannels[1].empty() &&
	muxInChannels[2].empty(); 

    lEmpty[2] = muxInChannels[3].empty() &&
	muxInChannels[4].empty() &&
	muxInChannels[5].empty(); 

    lEmpty[3] = muxInChannels[6].empty() &&
	muxInChannels[7].empty() &&
	muxInChannels[8].empty() &&
	muxInChannels[9].empty(); 

    /* Si al menos 2 layers están vacías, no se puede construir mezcla con
     * posible traza.
     */

    if ( (lEmpty[0] && lEmpty[1]) or (lEmpty[0] && lEmpty[2]) or
	 (lEmpty[0] && lEmpty[3]) or (lEmpty[1] && lEmpty[2]) or
	 (lEmpty[1] && lEmpty[3]) or (lEmpty[2] && lEmpty[3]) ) {
	//    if (debug) cout << "DTp2::NotEnoughDataInChannels" << endl;
	return true;
    }
    else {
	//    if (debug) cout << "DTp2::NotEnoughDataInChannels, we do have enough!" << endl;
	return false;
    }

}
void DTTrigPhase2Prod::resetPrvTDCTStamp(void) {
    for (int i = 0; i <= 3; i++) prevTDCTimeStamps[i] = -1;
}

bool DTTrigPhase2Prod::isEqualComb2Previous(DTPrimitive *dtPrims[4]) {
    bool answer = true;

    for (int i = 0; i <= 3; i++)
	if (prevTDCTimeStamps[i] != dtPrims[i]->getTDCTime()) {
	    answer = false;
	    for (int j = 0; j <= 3; j++) {
		prevTDCTimeStamps[j] = dtPrims[j]->getTDCTime();
	    }
	    break;
	}

    return answer;
}


//similar approach that JM's code
void DTTrigPhase2Prod::mixChannels(int supLayer, int pathId, std::vector<MuonPath*> *outMuonPath){ 
    //  if (debug) cout << "DTp2::mixChannels("<<supLayer<<","<<pathId<<")" << endl;
  //    std::vector<DTPrimitive*> data[4];
  std::vector<DTPrimitive> data[4];
  
    int horizLayout[4];
    memcpy(horizLayout, CELL_HORIZONTAL_LAYOUTS[pathId], 4 * sizeof(int));
  
    int chIdxForPath[4];
    memcpy(chIdxForPath, CHANNELS_PATH_ARRANGEMENTS[pathId], 4 * sizeof(int));
  
    // Real amount of values extracted from each channel.
    int numPrimsPerLayer[4] = {0, 0, 0, 0};
    unsigned int canal;
    int channelEmptyCnt = 0;
    for (int layer = 0; layer <= 3; layer++) {
	canal = CHANNELS_PATH_ARRANGEMENTS[pathId][layer];
	if (muxInChannels[canal].empty()) channelEmptyCnt++;
    }
  
    if (channelEmptyCnt >= 2) return;
    //


    //if (debug) cout << "DTp2::mixChannels --> no more than two empty channels" << endl;
  
    // Extraemos tantos elementos de cada canal como exija la combinacion
    for (int layer = 0; layer <= 3; layer++) {
	canal = CHANNELS_PATH_ARRANGEMENTS[pathId][layer];
	unsigned int maxPrimsToBeRetrieved = muxInChannels[canal].size();
	//    if (debug) cout << "DTp2::mixChannels --> maxPrimsToBeRetrieved " <<maxPrimsToBeRetrieved << endl;
	/*
	  If the number of primitives is zero, in order to avoid that only one
	  empty channel avoids mixing data from the other three, we, at least,
	  consider one dummy element from this channel.
	  In other cases, where two or more channels has zero elements, the final
	  combination will be not analyzable (the condition for being analyzable is
	  that it has at least three good TDC time values, not dummy), so it will
	  be discarded and not sent to the analyzer.
	*/
	if (maxPrimsToBeRetrieved == 0) maxPrimsToBeRetrieved = 1;
    
	for (unsigned int items = 0; items < maxPrimsToBeRetrieved; items++) {
      
	    //RMPTR DTPrimitive *dtpAux = new DTPrimitive(); 
	    DTPrimitive dtpAux = DTPrimitive();
	    if (muxInChannels[canal].size()!=0) {
	      //RMPTR dtpAux = (DTPrimitive*) &(muxInChannels[canal].at(items));
	      dtpAux = DTPrimitive(&(muxInChannels[canal].at(items)));
	    }
	    //      if (debug) cout << "DTp2::mixChannels --> DTPrimitive: " << dtpAux->getTDCTime() << ", " << dtpAux->getSuperLayerId() <<endl;
	    /*
	      I won't allow a whole loop cycle. When a DTPrimitive has an invalid
	      time-stamp (TDC value = -1) it means that the buffer is empty or the
	      buffer has reached the last element within the configurable time window.
	      In this case the loop is broken, but only if there is, at least, one
	      DTPrim (even invalid) on the outgoing array. This is mandatory to cope
	      with the idea explained in the previous comment block
	    */
	    //RMPRT if (dtpAux->getTDCTime() < 0 && items > 0) break;
	    if (dtpAux.getTDCTime() < 0 && items > 0) break;
	    /*
	     * En este nuevo esquema, si el HIT se corresponde con la SL sobre la
	     * que se están haciendo las mezclas, se envía al buffer intermedio
	     * de mezclas.
	     * En caso contrario, se envía una copia en blanco "inválida" para que
	     * la mezcla se complete, como ocurría en el caso de una sola SL.
	     * 
	     * En este caso, un poco chapuza, habrá bastantes casos en los que 
	     * se hagan mezclas inválidas. Por ello, la verificación que hay más
	     * adelante, en la que se comprueba si el segmento "es analizable"
	     * antes de ser enviado a la cola de salida, ES IMPRESCINDIBLE.
	     */
	    if (dtpAux.getSuperLayerId() == supLayer)   // values are 0, 1, 2 
	      data[layer].push_back(dtpAux);
	    else 
	      data[layer].push_back(  DTPrimitive() );
	    numPrimsPerLayer[layer]++;
	}
    }
  
    DTPrimitive *ptrPrimitive[4];
    /*
      Realizamos las diferentes combinaciones y las enviamos a las fifo de
      salida
    */
    int chIdx[4];
    //  if (debug) cout << "DTp2::mixChannels --> doing combinations with "
    //		     <<  numPrimsPerLayer[0] << " , "
    //		     <<  numPrimsPerLayer[1] << " , " 
    //		     <<  numPrimsPerLayer[2] << " , " 
    //		     <<  numPrimsPerLayer[3] << " per layer " << endl;
    for (chIdx[0] = 0; chIdx[0] < numPrimsPerLayer[0]; chIdx[0]++) {
	for (chIdx[1] = 0; chIdx[1] < numPrimsPerLayer[1]; chIdx[1]++) {
	    for (chIdx[2] = 0; chIdx[2] < numPrimsPerLayer[2]; chIdx[2]++) {
		for (chIdx[3] = 0; chIdx[3] < numPrimsPerLayer[3]; chIdx[3]++) {
	  
		    /*
		      Creamos una copia del objeto para poder manipular cada copia en
		      cada hilo de proceso de forma independiente, y poder destruirlas
		      cuando sea necesario, sin depender de una única referencia a lo
		      largo de todo el código.
		    */
	  
		    for (int i = 0; i <= 3; i++) {
			ptrPrimitive[i] = new DTPrimitive( (data[i])[chIdx[i]] );
		    }
	  
		    MuonPath *ptrMuonPath = new MuonPath(ptrPrimitive);
		    ptrMuonPath->setCellHorizontalLayout(horizLayout);
	  
		    //	  if (debug) cout << "Horizlayout " << ptrMuonPath->getCellHorizontalLayout() << endl;
		    /*
		      This new version of this code is redundant with PathAnalyzer code,
		      where every MuonPath not analyzable is discarded.
		      I insert this discarding mechanism here, as well, to avoid inserting
		      not-analyzable MuonPath into the candidate FIFO.
		      Equivalent code must be removed in the future from PathAnalyzer, but
		      it the mean time, at least during the testing state, I'll preserve
		      both.
		      Code in the PathAnalyzer should be doing nothing now.
		    */
		    if (ptrMuonPath->isAnalyzable()) {
			/*
			  This is a very simple filter because, during the tests, it has been
			  detected that many consecutive MuonPaths are duplicated mainly due
			  to buffers empty (or dummy) that give a TDC time-stamp = -1
			  With this filter, I'm removing those consecutive identical
			  combinations.
	      
			  If duplicated combinations are not consecutive, they won't be
			  detected here
			*/
			if ( !isEqualComb2Previous(ptrPrimitive) ) {
			    //	      if (debug) cout << "different from previous combination" << endl;
			    ptrMuonPath->setBaseChannelId(currentBaseChannel);
			    outMuonPath->push_back( ptrMuonPath );
			    //	      if (debug) cout << " Size: " << outMuonPath->size() << endl;
			}
			else delete ptrMuonPath;
		    }
		    else {
		      delete ptrMuonPath;
		    }
		}
	    }
	}
    }
  
    for (int layer = 0; layer <= 3; layer++) {
      //uncomenting this causes a seg fault
//RMPTR      int numData = data[layer].size();
//RMPTR      for (int i = 0; i < numData; i++) {
//RMPTR	data[layer][i] = (DTPrimitive*) (NULL);
//RMPTR	delete data[layer][i];
//RMPTR      }
      data[layer].clear();
      //      data[layer].erase(data[layer].begin(),data[layer].end());
    }
}

//std::vector<MuonPath> 
void DTTrigPhase2Prod::buildMuonPathCandidates(DTDigiCollection digis, std::vector<MuonPath*> *mpaths){
      
    // This function returns the analyzable mpath collection back to the the main function
    // so it can be fitted. This is in fact doing the so-called grouping.    
  
    //  std::vector<MuonPath> mpaths;
    for (int supLayer = 0; supLayer < NUM_SUPERLAYERS; supLayer++) {  // for each SL: 
	//    if (debug) cout << "DTp2::BuilMuonPathCandidates Reading SL"<< supLayer << endl;
	setInChannels(&digis,supLayer);
  
	for(int baseCh = 0; baseCh < TOTAL_BTI; baseCh++) {
	    currentBaseChannel = baseCh;
	    selectInChannels(currentBaseChannel);  //map a number of wires for a given base channel
      
	    if ( notEnoughDataInChannels() ) continue;
      
	    //      if (debug) cout << "DTp2::buildMuonPathCandidates --> now check pathId" << endl;
	    for(int pathId=0; pathId<8; pathId++){
		resetPrvTDCTStamp();

		mixChannels(supLayer,pathId, mpaths);      
	    }
	}
    }
    //return mpaths;
}

int DTTrigPhase2Prod::arePrimos(metaPrimitive primera, metaPrimitive segunda) {
    if(primera.rawId!=segunda.rawId) return 0;
    if(primera.wi1==segunda.wi1 and primera.tdc1==segunda.tdc1 and primera.wi1!=-1 and primera.tdc1!=-1) return 1;
    if(primera.wi2==segunda.wi2 and primera.tdc2==segunda.tdc2 and primera.wi2!=-1 and primera.tdc2!=-1) return 2;
    if(primera.wi3==segunda.wi3 and primera.tdc3==segunda.tdc3 and primera.wi3!=-1 and primera.tdc3!=-1) return 3;
    if(primera.wi4==segunda.wi4 and primera.tdc4==segunda.tdc4 and primera.wi4!=-1 and primera.tdc4!=-1) return 4;
    return 0;
}


void DTTrigPhase2Prod::printmP(metaPrimitive mP){
    DTSuperLayerId slId(mP.rawId);
    std::cout<<slId<<"\t"
	     <<" "<<setw(2)<<left<<mP.wi1
	     <<" "<<setw(2)<<left<<mP.wi2
	     <<" "<<setw(2)<<left<<mP.wi3
	     <<" "<<setw(2)<<left<<mP.wi4
	     <<" "<<setw(5)<<left<<mP.tdc1
	     <<" "<<setw(5)<<left<<mP.tdc2
	     <<" "<<setw(5)<<left<<mP.tdc3
	     <<" "<<setw(5)<<left<<mP.tdc4
	     <<" "<<setw(10)<<right<<mP.x
	     <<" "<<setw(9)<<left<<mP.tanPhi
	     <<" "<<setw(5)<<left<<mP.t0
	     <<" "<<setw(13)<<left<<mP.chi2
	     <<" r:"<<rango(mP);
}

int DTTrigPhase2Prod::rango(metaPrimitive primera) {
    int rango=0;
    if(primera.wi1!=-1)rango++;
    if(primera.wi2!=-1)rango++;
    if(primera.wi3!=-1)rango++;
    if(primera.wi4!=-1)rango++;
    return rango;
}


bool DTTrigPhase2Prod::outer(metaPrimitive primera){
    if(primera.wi1==-1 and primera.wi2==-1 and primera.wi3==-1 and primera.wi4==-1)
	return true;
    return false;
}

bool DTTrigPhase2Prod::inner(metaPrimitive primera){
    return !outer(primera);
}


bool DTTrigPhase2Prod::hasPosRF(int wh,int sec){
    return  wh>0 || (wh==0 && sec%4>1);
}

double DTTrigPhase2Prod::trigDir(metaPrimitive mp){
    DTChamberId chId(mp.rawId);
    int wh   = chId.wheel();
    int sec  = chId.sector();
    double phi  = mp.phi;
    double phib  = mp.phiB;    
    //double dir = (phib/512.+phi/4096.);
    double dir = (phib+phi);
    //change sign in case of negative wheels
    if (!hasPosRF(wh,sec)) { dir = -dir; }
    return dir;
}

double DTTrigPhase2Prod::trigPos(metaPrimitive mp){
    DTChamberId chId(mp.rawId);

    if(debug) cout<<"back: chId="<<chId<<endl;
    
    int wh   = chId.wheel();
    int sec  = chId.sector();
    int st   = chId.station();
    double phi  = mp.phi;
    double phin = (sec-1)*Geom::pi()/6;
    double phicenter = 0;
    double r = 0;
    double xcenter = 0;
    
    if (sec==4 && st==4) {
	GlobalPoint gpos = phi>0 ? dtGeo->chamber(DTChamberId(wh,st,13))->position() : dtGeo->chamber(DTChamberId(wh,st,4))->position();
	xcenter = phi>0 ? xCenter[0] : -xCenter[0];
	phicenter =  gpos.phi();
	r = gpos.perp();
    } else if (sec==10 && st==4) {
	GlobalPoint gpos = phi>0 ? dtGeo->chamber(DTChamberId(wh,st,14))->position() : dtGeo->chamber(DTChamberId(wh,st,10))->position();
	xcenter = phi>0 ? xCenter[1] : -xCenter[1];
	phicenter =  gpos.phi();
	r = gpos.perp();
    } else {
	GlobalPoint gpos = dtGeo->chamber(DTChamberId(wh,st,sec))->position();
	phicenter =  gpos.phi();
	r = gpos.perp();
    }

    if(debug)cout<<"back: phicenter="<<phicenter<<" phin="<<phicenter<<endl;
   
    double deltaphi = phicenter-phin;
    if(debug)cout<<"back: deltaphi="<<deltaphi<<endl;
    //double x = (tan(phi/4096.)-tan(deltaphi))*(r*cos(deltaphi) - zcn[st-1]); //zcn is in local coordinates -> z invreases approching to vertex
    double x = (tan(phi)-tan(deltaphi))*(r*cos(deltaphi) - zcn[st-1]); //zcn is in local coordinates -> z invreases approching to vertex
    if(debug)cout<<"back: x="<<x<<endl;
    if (hasPosRF(wh,sec)){ x = -x; } // change sign in case of positive wheels
    if(debug)cout<<"back: hasPosRF="<<hasPosRF(wh,sec)<<endl;
    if(debug)cout<<xcenter<<endl;
    //x+=xcenter; this s the bug found by luigi
    return x;
    
}

void DTTrigPhase2Prod::analyze(MuonPath *mPath,std::vector<metaPrimitive>& metaPrimitives,DTSuperLayerId dtSLId) {
    if(debug) std::cout<<"DTp2:analyze \t\t\t\t starts"<<std::endl;
    
    if(debug) std::cout<<"DTp2:analyze \t\t\t\t In analyze function checking if mPath->isAnalyzable() "<<mPath->isAnalyzable()<<std::endl;

    if (mPath->isAnalyzable()) {
	if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t yes it is analyzable "<<mPath->isAnalyzable()<<std::endl;
	setCellLayout( mPath->getCellHorizontalLayout() );
	evaluatePathQuality(mPath);
    }else{
	if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t no it is NOT analyzable "<<mPath->isAnalyzable()<<std::endl;
    }
    
    // Clonamos el objeto analizado.
    MuonPath *mpAux = new MuonPath(mPath);

    int wi[8],tdc[8];
    DTPrimitive Prim0(mpAux->getPrimitive(0)); wi[0]=Prim0.getChannelId();tdc[0]=Prim0.getTDCTime();
    DTPrimitive Prim1(mpAux->getPrimitive(1)); wi[1]=Prim1.getChannelId();tdc[1]=Prim1.getTDCTime();
    DTPrimitive Prim2(mpAux->getPrimitive(2)); wi[2]=Prim2.getChannelId();tdc[2]=Prim2.getTDCTime();
    DTPrimitive Prim3(mpAux->getPrimitive(3)); wi[3]=Prim3.getChannelId();tdc[3]=Prim3.getTDCTime();
    for(int i=4;i<8;i++){wi[i]=-1;tdc[i]=-1;}
    
    DTWireId wireId(dtSLId,2,1);

    if(debug) std::cout<<"DTp2:analyze \t\t\t\t checking if it passes the min quality cut "<<mPath->getQuality()<<">"<<minQuality<<std::endl;
    if ( mPath->getQuality() >= minQuality ) {
	if(debug) std::cout<<"DTp2:analyze \t\t\t\t min quality achievedCalidad: "<<mPath->getQuality()<<std::endl;
	for (int i = 0; i <= 3; i++){
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t  Capa: "<<mPath->getPrimitive(i)->getLayerId()
			       <<" Canal: "<<mPath->getPrimitive(i)->getChannelId()<<" TDCTime: "<<mPath->getPrimitive(i)->getTDCTime()<<std::endl;
	}
	if(debug) std::cout<<"DTp2:analyze \t\t\t\t Starting lateralities loop, totalNumValLateralities: "<<totalNumValLateralities<<std::endl;
	
	double best_chi2=99999.;
	double chi2_jm_tanPhi=999;
	double chi2_jm_x=-1;
	double chi2_jm_t0=-1;
	double chi2_phi=-1;
	double chi2_phiB=-1;
	double chi2_chi2=-1;
	int chi2_quality=-1;

 	for (int i = 0; i < totalNumValLateralities; i++) {//here
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<std::endl;
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" checking quality:"<<std::endl;
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" checking mPath Quality="<<mPath->getQuality()<<std::endl;   
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" latQuality[i].val="<<latQuality[i].valid<<std::endl;   
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" before if:"<<std::endl;

	    if (latQuality[i].valid and (((mPath->getQuality()==HIGHQ or mPath->getQuality()==HIGHQGHOST) and latQuality[i].quality==HIGHQ)
					 or
					 ((mPath->getQuality() == LOWQ or mPath->getQuality()==LOWQGHOST) and latQuality[i].quality==LOWQ))){
		
		if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" inside if"<<std::endl;
		mpAux->setBxTimeValue(latQuality[i].bxValue);
		if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" settingLateralCombination"<<std::endl;
		mpAux->setLateralComb(lateralities[i]);
		if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t laterality #- "<<i<<" done settingLateralCombination"<<std::endl;
		int idxHitNotValid = latQuality[i].invalidateHitIdx;
		if (idxHitNotValid >= 0) {
		    delete mpAux->getPrimitive(idxHitNotValid);
		    mpAux->setPrimitive(std::move(new DTPrimitive()), idxHitNotValid);
		}
		
		if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  calculating parameters "<<std::endl;
		calculatePathParameters(mpAux);
		/* 
		 * Si, tras calcular los parámetros, y si se trata de un segmento
		 * con 4 hits, el chi2 resultante es superior al umbral programado,
		 * lo eliminamos y no se envía al exterior.
		 * Y pasamos al siguiente elemento.
		 */
		if ((mpAux->getQuality() == HIGHQ or mpAux->getQuality() == HIGHQGHOST) && mpAux->getChiSq() > chiSquareThreshold) {//check this if!!!
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  HIGHQ or HIGHQGHOST but min chi2 or Q test not satisfied "<<std::endl;
		}				
		else{
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  inside else, returning values: "<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  BX Time = "<<mpAux->getBxTimeValue()<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  BX Id   = "<<mpAux->getBxNumId()<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  XCoor   = "<<mpAux->getHorizPos()<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  tan(Phi)= "<<mpAux->getTanPhi()<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  chi2= "<<mpAux->getChiSq()<<std::endl;
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t  lateralities = "
				       <<" "<<mpAux->getLateralComb()[0]
				       <<" "<<mpAux->getLateralComb()[1]
				       <<" "<<mpAux->getLateralComb()[2]
				       <<" "<<mpAux->getLateralComb()[3]
				       <<std::endl;

		    DTChamberId ChId(dtSLId.wheel(),dtSLId.station(),dtSLId.sector());
		
		    double jm_tanPhi=-1.*mpAux->getTanPhi(); //testing with this line

		    double jm_x=(mpAux->getHorizPos()/10.)+shiftinfo[wireId.rawId()]; 
		    //changing to chamber frame or reference:
		    //jm_x=jm_x-(zinfo[wireId.rawId()]-0.65)*jm_tanPhi; //0.65 is half hight of a cell needed to go to the middle of the superlayer, here we are extrapolating with the angle of the primitive!
		    double jm_t0=mpAux->getBxTimeValue();		      
		    int quality= mpAux->getQuality();

		    //computing phi and phiB
		    
		    double z=0;

		    double z1=11.75;
		    double z3=-1.*z1;
		    if (ChId.station() == 3 or ChId.station() == 4){
			z1=9.95;
			z3=-13.55;
		    }

		    if(dtSLId.superLayer()==1) z=z1;
		    if(dtSLId.superLayer()==3) z=z3;
		    
		    GlobalPoint jm_x_cmssw_global = dtGeo->chamber(dtSLId)->toGlobal(LocalPoint(jm_x,0.,z));//jm_x is already extrapolated to the middle of the SL
		    int thisec = dtSLId.sector();
		    if(thisec==13) thisec = 4;
		    if(thisec==14) thisec = 10;
		    double phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
		    double psi=TMath::ATan(jm_tanPhi);
		    double phiB=hasPosRF(dtSLId.wheel(),dtSLId.sector()) ? psi-phi : -psi-phi ;
		    double chi2= mpAux->getChiSq()*0.01;//in cmssw we want cm, 1 cm^2 = 100 mm^2
	    
		    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaPrimitive at x="<<jm_x<<" tanPhi:"<<jm_tanPhi<<" t0:"<<jm_t0<<std::endl;
		    
		    if(mpAux->getQuality() == HIGHQ or mpAux->getQuality() == HIGHQGHOST){//keep only the values with the best chi2 among lateralities
			if(chi2<best_chi2){
			    chi2_jm_tanPhi=jm_tanPhi;
			    chi2_jm_x=(mpAux->getHorizPos()/10.)+shiftinfo[wireId.rawId()]; 
			    //chi2_jm_x=chi2_jm_x-(zinfo[wireId.rawId()]-0.65)*chi2_jm_tanPhi; //from SL to CH no needed for co
			    chi2_jm_t0=mpAux->getBxTimeValue();		      
			    chi2_phi=phi;
			    chi2_phiB=phiB;
			    chi2_chi2=chi2;
			    chi2_quality= mpAux->getQuality();
			}
		    }else{//write the metaprimitive in case no HIGHQ or HIGHQGHOST
			if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaprimitive no HIGHQ or HIGHQGHOST"<<std::endl;
			metaPrimitives.push_back(metaPrimitive({dtSLId.rawId(),jm_t0,jm_x,jm_tanPhi,phi,phiB,chi2,quality,
					wi[0],tdc[0],
					wi[1],tdc[1],
					wi[2],tdc[2],
					wi[3],tdc[3],
					wi[4],tdc[4],
					wi[5],tdc[5],
					wi[6],tdc[6],
					wi[7],tdc[7]
					}));
			if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t\t\t\t  done pushing back metaprimitive no HIGHQ or HIGHQGHOST"<<std::endl;
		    }				
		}
	    }else{
		if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t\t\t\t  latQuality[i].valid and (((mPath->getQuality()==HIGHQ or mPath->getQuality()==HIGHQGHOST) and latQuality[i].quality==HIGHQ) or  ((mPath->getQuality() == LOWQ or mPath->getQuality()==LOWQGHOST) and latQuality[i].quality==LOWQ)) not passed"<<std::endl;
	    }
	}
	if(chi2_jm_tanPhi!=999){//
	    if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t\t\t\t  pushing back best chi2 metaPrimitive"<<std::endl;
	    metaPrimitives.push_back(metaPrimitive({dtSLId.rawId(),chi2_jm_t0,chi2_jm_x,chi2_jm_tanPhi,chi2_phi,chi2_phiB,chi2_chi2,chi2_quality,
			    wi[0],tdc[0],
			    wi[1],tdc[1],
			    wi[2],tdc[2],
			    wi[3],tdc[3],
			    wi[4],tdc[4],
			    wi[5],tdc[5],
			    wi[6],tdc[6],
			    wi[7],tdc[7]
			    }));
	}
    }
    delete mpAux;
    if(debug) std::cout<<"DTp2:analyze \t\t\t\t finishes"<<std::endl;
}


//------------------------------------------------------------------
//--- Métodos get / set
//------------------------------------------------------------------
void DTTrigPhase2Prod::setBXTolerance(int t) { bxTolerance = t; }
int  DTTrigPhase2Prod::getBXTolerance(void)  { return bxTolerance; }

void DTTrigPhase2Prod::setChiSquareThreshold(float ch2Thr) {
    chiSquareThreshold = ch2Thr;
}

void DTTrigPhase2Prod::setMinimumQuality(MP_QUALITY q) {
    if (minQuality >= LOWQGHOST) minQuality = q;
}
MP_QUALITY DTTrigPhase2Prod::getMinimumQuality(void) { return minQuality; }


//------------------------------------------------------------------
//--- Métodos privados
//------------------------------------------------------------------
void DTTrigPhase2Prod::setCellLayout(const int layout[4]) {
    memcpy(cellLayout, layout, 4 * sizeof(int));
    //celllayout[0]=layout[0];
    //celllayout[1]=layout[1];
    //celllayout[2]=layout[2];
    //celllayout[3]=layout[3];
    
    buildLateralities();
}

/**
 * Para una combinación de 4 celdas dada (las que se incluyen en el analizador,
 * una por capa), construye de forma automática todas las posibles
 * combinaciones de lateralidad (LLLL, LLRL,...) que son compatibles con una
 * trayectoria recta. Es decir, la partícula no hace un zig-zag entre los hilos
 * de diferentes celdas, al pasar de una a otra.
 */
void DTTrigPhase2Prod::buildLateralities(void) {

    LATERAL_CASES (*validCase)[4], sideComb[4];

    totalNumValLateralities = 0;
    /* Generamos todas las posibles combinaciones de lateralidad para el grupo
       de celdas que forman parte del analizador */
    for(int lowLay = LEFT; lowLay <= RIGHT; lowLay++)
	for(int midLowLay = LEFT; midLowLay <= RIGHT; midLowLay++)
	    for(int midHigLay = LEFT; midHigLay <= RIGHT; midHigLay++)
		for(int higLay = LEFT; higLay <= RIGHT; higLay++) {

		    sideComb[0] = static_cast<LATERAL_CASES>(lowLay);
		    sideComb[1] = static_cast<LATERAL_CASES>(midLowLay);
		    sideComb[2] = static_cast<LATERAL_CASES>(midHigLay);
		    sideComb[3] = static_cast<LATERAL_CASES>(higLay);

		    /* Si una combinación de lateralidades es válida, la almacenamos */
		    if (isStraightPath(sideComb)) {
			validCase = lateralities + totalNumValLateralities;
			memcpy(validCase, sideComb, 4 * sizeof(LATERAL_CASES));

			latQuality[totalNumValLateralities].valid            = false;
			latQuality[totalNumValLateralities].bxValue          = 0;
			latQuality[totalNumValLateralities].quality          = NOPATH;
			latQuality[totalNumValLateralities].invalidateHitIdx = -1;

			totalNumValLateralities++;
		    }
		}
}

/**
 * Para automatizar la generación de trayectorias compatibles con las posibles
 * combinaciones de lateralidad, este método decide si una cierta combinación
 * de lateralidad, involucrando 4 celdas de las que conforman el DTTrigPhase2Prod,
 * forma una traza recta o no. En caso negativo, la combinación de lateralidad
 * es descartada y no se analiza.
 * En el caso de implementación en FPGA, puesto que el diseño intentará
 * paralelizar al máximo la lógica combinacional, el equivalente a este método
 * debería ser un "generate" que expanda las posibles combinaciones de
 * lateralidad de celdas compatibles con el análisis.
 *
 * El métoda da por válida una trayectoria (es recta) si algo parecido al
 * cambio en la pendiente de la trayectoria, al cambiar de par de celdas
 * consecutivas, no es mayor de 1 en unidades arbitrarias de semi-longitudes
 * de celda para la dimensión horizontal, y alturas de celda para la vertical.
 */
bool DTTrigPhase2Prod::isStraightPath(LATERAL_CASES sideComb[4]) {

    //return true; //trying with all lateralities to be confirmed

    int i, ajustedLayout[4], pairDiff[3], desfase[3];

    /* Sumamos el valor de lateralidad (LEFT = 0, RIGHT = 1) al desfase
       horizontal (respecto de la celda base) para cada celda en cuestion */
    for(i = 0; i <= 3; i++) ajustedLayout[i] = cellLayout[i] + sideComb[i];
    /* Variación del desfase por pares de celdas consecutivas */
    for(i = 0; i <= 2; i++) pairDiff[i] = ajustedLayout[i+1] - ajustedLayout[i];
    /* Variación de los desfases entre todas las combinaciones de pares */
    for(i = 0; i <= 1; i++) desfase[i] = abs(pairDiff[i+1] - pairDiff[i]);
    desfase[2] = abs(pairDiff[2] - pairDiff[0]);
    /* Si algún desfase es mayor de 2 entonces la trayectoria no es recta */
    bool resultado = (desfase[0] > 1 or desfase[1] > 1 or desfase[2] > 1);

    return ( !resultado );
}

/**
 * Recorre las calidades calculadas para todas las combinaciones de lateralidad
 * válidas, para determinar la calidad final asignada al "MuonPath" con el que
 * se está trabajando.
 */
void DTTrigPhase2Prod::evaluatePathQuality(MuonPath *mPath) {

    int totalHighQ = 0, totalLowQ = 0;
    
    if(debug) std::cout<<"DTp2:evaluatePathQuality \t\t\t\t\t En evaluatePathQuality Evaluando PathQ. Celda base: "<<mPath->getBaseChannelId()<<std::endl;
    if(debug) std::cout<<"DTp2:evaluatePathQuality \t\t\t\t\t Total lateralidades: "<<totalNumValLateralities<<std::endl;

    // Por defecto.
    mPath->setQuality(NOPATH);

    /* Ensayamos los diferentes grupos de lateralidad válidos que constituyen
       las posibles trayectorias del muón por el grupo de 4 celdas.
       Posiblemente esto se tenga que optimizar de manera que, si en cuanto se
       encuentre una traza 'HIGHQ' ya no se continue evaluando mas combinaciones
       de lateralidad, pero hay que tener en cuenta los fantasmas (rectas
       paralelas) en de alta calidad que se pueden dar en los extremos del BTI.
       Posiblemente en la FPGA, si esto se paraleliza, no sea necesaria tal
       optimización */
    for (int latIdx = 0; latIdx < totalNumValLateralities; latIdx++) {
	if(debug) std::cout<<"DTp2:evaluatePathQuality \t\t\t\t\t Analizando combinacion de lateralidad: "
			   <<lateralities[latIdx][0]<<" "
			   <<lateralities[latIdx][1]<<" "	 
			   <<lateralities[latIdx][2]<<" "
			   <<lateralities[latIdx][3]<<std::endl;
	  
	evaluateLateralQuality(latIdx, mPath, &(latQuality[latIdx]));

	if (latQuality[latIdx].quality == HIGHQ) {
	    totalHighQ++;
	    if(debug) std::cout<<"DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad HIGHQ"<<std::endl;
	}
	if (latQuality[latIdx].quality == LOWQ) {
	    totalLowQ++;
	    if(debug) std::cout<<"DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad LOWQ"<<std::endl;
	}
    }
    /*
     * Establecimiento de la calidad.
     */
    if (totalHighQ == 1) {
	mPath->setQuality(HIGHQ);
    }
    else if (totalHighQ > 1) {
	mPath->setQuality(HIGHQGHOST);
    }
    else if (totalLowQ == 1) {
	mPath->setQuality(LOWQ);
    }
    else if (totalLowQ > 1) {
	mPath->setQuality(LOWQGHOST);
    }
}

void DTTrigPhase2Prod::evaluateLateralQuality(int latIdx, MuonPath *mPath,LATQ_TYPE *latQuality){

    int layerGroup[3];
    LATERAL_CASES sideComb[3];
    PARTIAL_LATQ_TYPE latQResult[4] = {
	{false, 0}, {false, 0}, {false, 0}, {false, 0}
    };

    // Default values.
    latQuality->valid            = false;
    latQuality->bxValue          = 0;
    latQuality->quality          = NOPATH;
    latQuality->invalidateHitIdx = -1;

    /* En el caso que, para una combinación de lateralidad dada, las 2
       combinaciones consecutivas de 3 capas ({0, 1, 2}, {1, 2, 3}) fueran
       traza válida, habríamos encontrado una traza correcta de alta calidad,
       por lo que sería innecesario comprobar las otras 2 combinaciones
       restantes.
       Ahora bien, para reproducir el comportamiento paralelo de la FPGA en el
       que el análisis se va a evaluar simultáneamente en todas ellas,
       construimos un código que analiza las 4 combinaciones, junto con una
       lógica adicional para discriminar la calidad final de la traza */
    for (int i = 0; i <= 3 ; i++) {
	memcpy(layerGroup, LAYER_ARRANGEMENTS[i], 3 * sizeof(int));

	// Seleccionamos la combinación de lateralidad para cada celda.
	for (int j = 0; j < 3; j++)
	    sideComb[j] = lateralities[latIdx][ layerGroup[j] ];

	validate(sideComb, layerGroup, mPath, &(latQResult[i]));
    }
    /*
      Imponemos la condición, para una combinación de lateralidad completa, que
      todas las lateralidades parciales válidas arrojen el mismo valor de BX
      (dentro de un margen) para así dar una traza consistente.
      En caso contrario esa combinación se descarta.
    */
    if ( !sameBXValue(latQResult) ) {
	// Se guardan en los default values inciales.
	if(debug) std::cout<<"DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. Tolerancia de BX excedida"<<std::endl;
	return;
    }

    // Dos trazas complementarias válidas => Traza de muón completa.
    if ((latQResult[0].latQValid && latQResult[1].latQValid) or
	(latQResult[0].latQValid && latQResult[2].latQValid) or
	(latQResult[0].latQValid && latQResult[3].latQValid) or
	(latQResult[1].latQValid && latQResult[2].latQValid) or
	(latQResult[1].latQValid && latQResult[3].latQValid) or
	(latQResult[2].latQValid && latQResult[3].latQValid))
	{
	    latQuality->valid   = true;
	    //     latQuality->bxValue = latQResult[0].bxValue;
	    /*
	     * Se hace necesario el contador de casos "numValid", en vez de promediar
	     * los 4 valores dividiendo entre 4, puesto que los casos de combinaciones
	     * de 4 hits buenos que se ajusten a una combinación como por ejemplo:
	     * L/R/L/L, dan lugar a que en los subsegmentos 0, y 1 (consecutivos) se
	     * pueda aplicar mean-timer, mientras que en el segmento 3 (en el ejemplo
	     * capas: 0,2,3, y combinación L/L/L) no se podría aplicar, dando un
	     * valor parcial de BX = 0.
	     */
	    int sumBX = 0, numValid = 0;
	    for (int i = 0; i <= 3; i++) {
		if (latQResult[i].latQValid) {
		    sumBX += latQResult[i].bxValue;
		    numValid++;
		}
	    }

	    latQuality->bxValue = sumBX / numValid;
	    latQuality->quality = HIGHQ;

	    if(debug) std::cout<<"DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. HIGHQ."<<std::endl;
	}
    // Sólo una traza disjunta válida => Traza de muón incompleta pero válida.
    else { 
	if (latQResult[0].latQValid or latQResult[1].latQValid or
	    latQResult[2].latQValid or latQResult[3].latQValid)
	    {
		latQuality->valid   = true;
		latQuality->quality = LOWQ;
		for (int i = 0; i < 4; i++)
		    if (latQResult[i].latQValid) {
			latQuality->bxValue = latQResult[i].bxValue;
			/*
			 * En los casos que haya una combinación de 4 hits válidos pero
			 * sólo 3 de ellos formen traza (calidad 2), esto permite detectar
			 * la layer con el hit que no encaja en la recta, y así poder
			 * invalidarlo, cambiando su valor por "-1" como si de una mezcla
			 * de 3 hits pura se tratara.
			 * Esto es útil para los filtros posteriores.
			 */
			latQuality->invalidateHitIdx = getOmittedHit( i );
			break;
		    }

		if(debug) std::cout<<"DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. LOWQ."<<std::endl;
	    }
	else {
	    if(debug) std::cout<<"DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. NOPATH."<<std::endl;
	}
    }
}

/**
 * Valida, para una combinación de capas (3), celdas y lateralidad, si los
 * valores temporales cumplen el criterio de mean-timer.
 * En vez de comparar con un 0 estricto, que es el resultado aritmético de las
 * ecuaciones usadas de base, se incluye en la clase un valor de tolerancia
 * que por defecto vale cero, pero que se puede ajustar a un valor más
 * adecuado
 *
 * En esta primera versión de la clase, el código de generación de ecuaciones
 * se incluye en esta función, lo que es ineficiente porque obliga a calcular
 * un montón de constantes, fijas para cada combinación de celdas, que
 * tendrían que evaluarse una sóla vez en el constructor de la clase.
 * Esta disposición en el constructor estaría más proxima a la realización que
 * se tiene que llevar a término en la FPGA (en tiempo de síntesis).
 * De momento se deja aquí porque así se entiende la lógica mejor, al estar
 * descrita de manera lineal en un sólo método.
 */
void DTTrigPhase2Prod::validate(LATERAL_CASES sideComb[3], int layerIndex[3],MuonPath* mPath, PARTIAL_LATQ_TYPE *latq)
{
    // Valor por defecto.
    latq->bxValue   = 0;
    latq->latQValid = false;
  
    if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t In validate Iniciando validacion de MuonPath para capas: "
		       <<layerIndex[0]<<"/"
		       <<layerIndex[1]<<"/"
		       <<layerIndex[2]<<std::endl;
  
    if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Lateralidades parciales: "
		       <<sideComb[0]<<"/"
		       <<sideComb[1]<<"/"
		       <<sideComb[2]<<std::endl;
  
    /* Primero evaluamos si, para la combinación concreta de celdas en curso, el
       número de celdas con dato válido es 3. Si no es así, sobre esa
       combinación no se puede aplicar el mean-timer y devolvemos "false" */
    int validCells = 0;
    for (int j = 0; j < 3; j++)
	if (mPath->getPrimitive(layerIndex[j])->isValidTime()) validCells++;
  
    if (validCells != 3) {
	if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t No hay 3 celdas validas."<<std::endl;
	return;
    }

    if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Valores de TDC: "
		       <<mPath->getPrimitive(layerIndex[0])->getTDCTime()<<"/"
		       <<mPath->getPrimitive(layerIndex[1])->getTDCTime()<<"/"
		       <<mPath->getPrimitive(layerIndex[2])->getTDCTime()<<"."
		       <<std::endl;

    if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Valid TIMES: "
		       <<mPath->getPrimitive(layerIndex[0])->isValidTime()<<"/"
		       <<mPath->getPrimitive(layerIndex[1])->isValidTime()<<"/"
		       <<mPath->getPrimitive(layerIndex[2])->isValidTime()<<"."
		       <<std::endl;

  
    /* Distancias verticales entre capas inferior/media y media/superior */
    int dVertMI = layerIndex[1] - layerIndex[0];
    int dVertSM = layerIndex[2] - layerIndex[1];

    /* Distancias horizontales entre capas inferior/media y media/superior */
    int dHorzMI = cellLayout[layerIndex[1]] - cellLayout[layerIndex[0]];
    int dHorzSM = cellLayout[layerIndex[2]] - cellLayout[layerIndex[1]];

    /* Índices de pares de capas sobre las que se está actuando
       SM => Superior + Intermedia
       MI => Intermedia + Inferior
       Jugamos con los punteros para simplificar el código */
    int *layPairSM = &layerIndex[1];
    int *layPairMI = &layerIndex[0];

    /* Pares de combinaciones de celdas para composición de ecuación. Sigue la
       misma nomenclatura que el caso anterior */
    LATERAL_CASES smSides[2], miSides[2];

    /* Teniendo en cuenta que en el índice 0 de "sideComb" se almacena la
       lateralidad de la celda inferior, jugando con aritmética de punteros
       extraemos las combinaciones de lateralidad para los pares SM y MI */

    memcpy(smSides, &sideComb[1], 2 * sizeof(LATERAL_CASES));
  
    memcpy(miSides, &sideComb[0], 2 * sizeof(LATERAL_CASES));
  
    float bxValue = 0;
    int coefsAB[2] = {0, 0}, coefsCD[2] = {0, 0};
    /* It's neccesary to be careful with that pointer's indirection. We need to
       retrieve the lateral coeficientes (+-1) from the lower/middle and
       middle/upper cell's lateral combinations. They are needed to evaluate the
       existance of a possible BX value, following it's calculation equation */
    getLateralCoeficients(miSides, coefsAB);
    getLateralCoeficients(smSides, coefsCD);

    /* Cada para de sumas de los 'coefsCD' y 'coefsAB' dan siempre como resultado
       0, +-2.

       A su vez, y pese a que las ecuaciones se han construido de forma genérica
       para cualquier combinación de celdas de la cámara, los valores de 'dVertMI' y
       'dVertSM' toman valores 1 o 2 puesto que los pares de celdas con los que se
       opera en realidad, o bien están contiguos, o bien sólo están separadas por
       una fila de celdas intermedia. Esto es debido a cómo se han combinado los
       grupos de celdas, para aplicar el mean-timer, en 'LAYER_ARRANGEMENTS'.

       El resultado final es que 'denominator' es siempre un valor o nulo, o
       múltiplo de 2 */
    int denominator = dVertMI * (coefsCD[1] + coefsCD[0]) -
	dVertSM * (coefsAB[1] + coefsAB[0]);

    if(denominator == 0) {
	if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Imposible calcular BX. Denominador para BX = 0."<<std::endl;
	return;
    }

    /* Esta ecuación ha de ser optimizada, especialmente en su implementación
       en FPGA. El 'denominator' toma siempre valores múltiplo de 2 o nulo, por lo
       habría que evitar el cociente y reemplazarlo por desplazamientos de bits */
    bxValue = (
	       dVertMI*(dHorzSM*MAXDRIFT + eqMainBXTerm(smSides, layPairSM, mPath)) -
	       dVertSM*(dHorzMI*MAXDRIFT + eqMainBXTerm(miSides, layPairMI, mPath))
	       ) / denominator;

    if(bxValue < 0) {
	if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Combinacion no valida. BX Negativo."<<std::endl;
	return;
    }

    // Redondeo del valor del tiempo de BX al nanosegundo
    if ( (bxValue - int(bxValue)) >= 0.5 ) bxValue = float(int(bxValue + 1));
    else bxValue = float(int(bxValue));

    /* Ciertos valores del tiempo de BX, siendo positivos pero objetivamente no
       válidos, pueden dar lugar a que el discriminador de traza asociado de un
       valor aparentemente válido (menor que la tolerancia y típicamente 0). Eso es
       debido a que el valor de tiempo de BX es mayor que algunos de los tiempos
       de TDC almacenados en alguna de las respectivas 'DTPrimitives', lo que da
       lugar a que, cuando se establece el valore de BX para el 'MuonPath', se
       obtengan valores de tiempo de deriva (*NO* tiempo de TDC) en la 'DTPrimitive'
       nulos, o inconsistentes, a causa de la resta entre enteros.

       Así pues, se impone como criterio de validez adicional que el valor de tiempo
       de BX (bxValue) sea siempre superior a cualesquiera valores de tiempo de TDC
       almacenados en las 'DTPrimitives' que forman el 'MuonPath' que se está
       analizando.
       En caso contrario, se descarta como inválido */

    for (int i = 0; i < 3; i++)
	if (mPath->getPrimitive(layerIndex[i])->isValidTime()) {
	    int diffTime =
		mPath->getPrimitive(layerIndex[i])->getTDCTimeNoOffset() - bxValue;

	    if (diffTime < 0 or diffTime > MAXDRIFT) {
		if(debug) std::cout<<"DTp2:validate \t\t\t\t\t\t\t Valor de BX inválido. Al menos un tiempo de TDC sin sentido"<<std::endl;
		return;
	    }
	}

    /* Si se llega a este punto, el valor de BX y la lateralidad parcial se dan
     * por válidas.
     */
    latq->bxValue   = bxValue;
    latq->latQValid = true;
}//finish validate

/**
 * Evalúa la suma característica de cada par de celdas, según la lateralidad
 * de la trayectoria.
 * El orden de los índices de capa es crítico:
 *    layerIdx[0] -> Capa más baja,
 *    layerIdx[1] -> Capa más alta
 */
int DTTrigPhase2Prod::eqMainBXTerm(LATERAL_CASES sideComb[2], int layerIdx[2],MuonPath* mPath)
{
    int eqTerm = 0, coefs[2];
    
    getLateralCoeficients(sideComb, coefs);
    
    eqTerm = coefs[0] * mPath->getPrimitive(layerIdx[0])->getTDCTimeNoOffset() +
	coefs[1] * mPath->getPrimitive(layerIdx[1])->getTDCTimeNoOffset();
    
    if(debug) std::cout<<"DTp2:eqMainBXTerm \t\t\t\t\t In eqMainBXTerm EQTerm(BX): "<<eqTerm<<std::endl;
    
    return (eqTerm);
}

/**
 * Evalúa la suma característica de cada par de celdas, según la lateralidad
 * de la trayectoria. Semejante a la anterior, pero aplica las correcciones
 * debidas a los retardos de la electrónica, junto con la del Bunch Crossing
 *
 * El orden de los índices de capa es crítico:
 *    layerIdx[0] -> Capa más baja,
 *    layerIdx[1] -> Capa más alta
 */
int DTTrigPhase2Prod::eqMainTerm(LATERAL_CASES sideComb[2], int layerIdx[2],MuonPath* mPath, int bxValue)
{
    int eqTerm = 0, coefs[2];

    getLateralCoeficients(sideComb, coefs);
    
    eqTerm = coefs[0] * (mPath->getPrimitive(layerIdx[0])->getTDCTimeNoOffset() -
			 bxValue) +
	coefs[1] * (mPath->getPrimitive(layerIdx[1])->getTDCTimeNoOffset() -
		    bxValue);
    
    if(debug) std::cout<<"DTp2:\t\t\t\t\t EQTerm(Main): "<<eqTerm<<std::endl;
    
    return (eqTerm);
}

/**
 * Devuelve los coeficientes (+1 ó -1) de lateralidad para un par dado.
 * De momento es útil para poder codificar la nueva funcionalidad en la que se
 * calcula el BX.
 */

void DTTrigPhase2Prod::getLateralCoeficients(LATERAL_CASES sideComb[2],int *coefs)
{
    if ((sideComb[0] == LEFT) && (sideComb[1] == LEFT)) {
	*(coefs)     = +1;
	*(coefs + 1) = -1;
    }
    else if ((sideComb[0] == LEFT) && (sideComb[1] == RIGHT)){
	*(coefs)     = +1;
	*(coefs + 1) = +1;
    }
    else if ((sideComb[0] == RIGHT) && (sideComb[1] == LEFT)){
	*(coefs)     = -1;
	*(coefs + 1) = -1;
    }
    else if ((sideComb[0] == RIGHT) && (sideComb[1] == RIGHT)){
	*(coefs)     = -1;
	*(coefs + 1) = +1;
    }
}

/**
 * Determines if all valid partial lateral combinations share the same value
 * of 'bxValue'.
 */
bool DTTrigPhase2Prod::sameBXValue(PARTIAL_LATQ_TYPE* latq) {

    bool result = true;
    /*
      Para evitar los errores de precision en el cálculo, en vez de forzar un
      "igual" estricto a la hora de comparar los diferentes valores de BX, se
      obliga a que la diferencia entre pares sea menor que un cierto valor umbral.
      Para hacerlo cómodo se crean 6 booleanos que evalúan cada posible diferencia
    */
  
    if(debug) std::cout<<"Dtp2:sameBXValue bxTolerance: "<<bxTolerance<<std::endl;

    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d01:"<<abs(latq[0].bxValue - latq[1].bxValue)<<std::endl;
    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d02:"<<abs(latq[0].bxValue - latq[2].bxValue)<<std::endl;
    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d03:"<<abs(latq[0].bxValue - latq[3].bxValue)<<std::endl;
    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d12:"<<abs(latq[1].bxValue - latq[2].bxValue)<<std::endl;
    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d13:"<<abs(latq[1].bxValue - latq[3].bxValue)<<std::endl;
    if(debug) std::cout<<"Dtp2:sameBXValue \t\t\t\t\t\t d23:"<<abs(latq[2].bxValue - latq[3].bxValue)<<std::endl;

    bool d01, d02, d03, d12, d13, d23;
    d01 = (abs(latq[0].bxValue - latq[1].bxValue) <= bxTolerance) ? true : false;
    d02 = (abs(latq[0].bxValue - latq[2].bxValue) <= bxTolerance) ? true : false;
    d03 = (abs(latq[0].bxValue - latq[3].bxValue) <= bxTolerance) ? true : false;
    d12 = (abs(latq[1].bxValue - latq[2].bxValue) <= bxTolerance) ? true : false;
    d13 = (abs(latq[1].bxValue - latq[3].bxValue) <= bxTolerance) ? true : false;
    d23 = (abs(latq[2].bxValue - latq[3].bxValue) <= bxTolerance) ? true : false;

    /* Casos con 4 grupos de combinaciones parciales de lateralidad validas */
    if ((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid &&
	 latq[3].latQValid) && !(d01 && d12 && d23))
	result = false;
    else
	/* Los 4 casos posibles de 3 grupos de lateralidades parciales validas */
	if ( ((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid) &&
	      !(d01 && d12)
	      )
	     or
	     ((latq[0].latQValid && latq[1].latQValid && latq[3].latQValid) &&
	      !(d01 && d13)
	      )
	     or
	     ((latq[0].latQValid && latq[2].latQValid && latq[3].latQValid) &&
	      !(d02 && d23)
	      )
	     or
	     ((latq[1].latQValid && latq[2].latQValid && latq[3].latQValid) &&
	      !(d12 && d23)
	      )
	     )
	    result = false;
	else
	    /* Por ultimo, los 6 casos posibles de pares de lateralidades parciales validas */

	    if ( ((latq[0].latQValid && latq[1].latQValid) && !d01) or
		 ((latq[0].latQValid && latq[2].latQValid) && !d02) or
		 ((latq[0].latQValid && latq[3].latQValid) && !d03) or
		 ((latq[1].latQValid && latq[2].latQValid) && !d12) or
		 ((latq[1].latQValid && latq[3].latQValid) && !d13) or
		 ((latq[2].latQValid && latq[3].latQValid) && !d23) )
		result = false;
  
    return result;
}

/** Calcula los parámetros de la(s) trayectoria(s) detectadas.
 *
 * Asume que el origen de coordenadas está en al lado 'izquierdo' de la cámara
 * con el eje 'X' en la posición media vertical de todas las celdas.
 * El eje 'Y' se apoya sobre los hilos de las capas 1 y 3 y sobre los costados
 * de las capas 0 y 2.
 */
void DTTrigPhase2Prod::calculatePathParameters(MuonPath* mPath) {
    // El orden es importante. No cambiar sin revisar el codigo.
    if(debug) std::cout<<"DTp2:calculatePathParameters \t\t\t\t\t\t  calculating calcCellDriftAndXcoor(mPath) "<<std::endl;
    calcCellDriftAndXcoor(mPath);
    //calcTanPhiXPosChamber(mPath);
    if(debug) std::cout<<"DTp2:calculatePathParameters \t\t\t\t\t\t  checking mPath->getQuality() "<<mPath->getQuality()<<std::endl;
    if (mPath->getQuality() == HIGHQ or mPath->getQuality() == HIGHQGHOST){
	if(debug) std::cout<<"DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test passed, now calcTanPhiXPosChamber4Hits(mPath) "<<std::endl;
	calcTanPhiXPosChamber4Hits(mPath);
    }else{
	if(debug) std::cout<<"DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test NOT passed calcTanPhiXPosChamber3Hits(mPath) "<<std::endl;
	calcTanPhiXPosChamber3Hits(mPath);
    }
    if(debug) std::cout<<"DTp2:calculatePathParameters \t\t\t\t\t\t calcChiSquare(mPath) "<<std::endl;
    calcChiSquare(mPath);
}

void DTTrigPhase2Prod::calcTanPhiXPosChamber(MuonPath* mPath)
{
    /*
      La mayoría del código de este método tiene que ser optimizado puesto que
      se hacen llamadas y cálculos redundantes que ya se han evaluado en otros
      métodos previos.

      Hay que hacer una revisión de las ecuaciones para almacenar en el 'MuonPath'
      una serie de parámetro característicos (basados en sumas y productos, para
      que su implementación en FPGA sea sencilla) con los que, al final del
      proceso, se puedan calcular el ángulo y la coordenada horizontal.

      De momento se deja este código funcional extraído directamente de las
      ecuaciones de la recta.
    */
    int layerIdx[2];
    /*
      To calculate path's angle are only necessary two valid primitives.
      This method should be called only when a 'MuonPath' is determined as valid,
      so, at least, three of its primitives must have a valid time.
      With this two comparitions (which can be implemented easily as multiplexors
      in the FPGA) this method ensures to catch two of those valid primitives to
      evaluate the angle.

      The first one is below the middle line of the superlayer, while the other
      one is above this line
    */
    if (mPath->getPrimitive(0)->isValidTime()) layerIdx[0] = 0;
    else layerIdx[0] = 1;

    if (mPath->getPrimitive(3)->isValidTime()) layerIdx[1] = 3;
    else layerIdx[1] = 2;

    /* We identify along which cells' sides the muon travels */
    LATERAL_CASES sideComb[2];
    sideComb[0] = (mPath->getLateralComb())[ layerIdx[0] ];
    sideComb[1] = (mPath->getLateralComb())[ layerIdx[1] ];

    /* Horizontal gap between cells in cell's semi-length units */
    int dHoriz = (mPath->getCellHorizontalLayout())[ layerIdx[1] ] -
	(mPath->getCellHorizontalLayout())[ layerIdx[0] ];

    /* Vertical gap between cells in cell's height units */
    int dVert = layerIdx[1] -layerIdx[0];

    /*-----------------------------------------------------------------*/
    /*--------------------- Phi angle calculation ---------------------*/
    /*-----------------------------------------------------------------*/
    float num = CELL_SEMILENGTH * dHoriz +
	DRIFT_SPEED *
	eqMainTerm(sideComb, layerIdx, mPath,
		   mPath->getBxTimeValue()
		   );

    float denom = CELL_HEIGHT * dVert;
    float tanPhi = num / denom;

    mPath->setTanPhi(tanPhi);

    /*-----------------------------------------------------------------*/
    /*----------------- Horizontal coord. calculation -----------------*/
    /*-----------------------------------------------------------------*/

    /*
      Using known coordinates, relative to superlayer axis reference, (left most
      superlayer side, and middle line between 2nd and 3rd layers), calculating
      horizontal coordinate implies using a basic line equation:
      (y - y0) = (x - x0) * cotg(Phi)
      This horizontal coordinate can be obtained setting y = 0 on last equation,
      and also setting y0 and x0 with the values of a known muon's path cell
      position hit.
      It's enough to use the lower cell (layerIdx[0]) coordinates. So:
      xC = x0 - y0 * tan(Phi)
    */
    float lowerXPHorizPos = mPath->getXCoorCell( layerIdx[0] );

    float lowerXPVertPos = 0; // This is only the absolute value distance.
    if (layerIdx[0] == 0) lowerXPVertPos = CELL_HEIGHT + CELL_SEMIHEIGHT;
    else                  lowerXPVertPos = CELL_SEMIHEIGHT;

    mPath->setHorizPos( lowerXPHorizPos + lowerXPVertPos * tanPhi );
}

/**
 * Cálculos de coordenada y ángulo para un caso de 4 HITS de alta calidad.
 */
void DTTrigPhase2Prod::calcTanPhiXPosChamber4Hits(MuonPath* mPath) {
    float tanPhi = (3 * mPath->getXCoorCell(3) +
		    mPath->getXCoorCell(2) -
		    mPath->getXCoorCell(1) -
		    3 * mPath->getXCoorCell(0)) / (10 * CELL_HEIGHT);

    mPath->setTanPhi(tanPhi);

    float XPos = (mPath->getXCoorCell(0) +
		  mPath->getXCoorCell(1) +
		  mPath->getXCoorCell(2) +
		  mPath->getXCoorCell(3)) / 4;

    mPath->setHorizPos( XPos );
}

/**
 * Cálculos de coordenada y ángulo para un caso de 3 HITS.
 */
void DTTrigPhase2Prod::calcTanPhiXPosChamber3Hits(MuonPath* mPath) {
    int layerIdx[2];

    if (mPath->getPrimitive(0)->isValidTime()) layerIdx[0] = 0;
    else layerIdx[0] = 1;

    if (mPath->getPrimitive(3)->isValidTime()) layerIdx[1] = 3;
    else layerIdx[1] = 2;

    /* We identify along which cells' sides the muon travels */
    LATERAL_CASES sideComb[2];
    sideComb[0] = (mPath->getLateralComb())[ layerIdx[0] ];
    sideComb[1] = (mPath->getLateralComb())[ layerIdx[1] ];

    /* Horizontal gap between cells in cell's semi-length units */
    int dHoriz = (mPath->getCellHorizontalLayout())[ layerIdx[1] ] -
	(mPath->getCellHorizontalLayout())[ layerIdx[0] ];

    /* Vertical gap between cells in cell's height units */
    int dVert = layerIdx[1] -layerIdx[0];

    /*-----------------------------------------------------------------*/
    /*--------------------- Phi angle calculation ---------------------*/
    /*-----------------------------------------------------------------*/
    float num = CELL_SEMILENGTH * dHoriz +
	DRIFT_SPEED *
	eqMainTerm(sideComb, layerIdx, mPath,
		   mPath->getBxTimeValue()
		   );

    float denom = CELL_HEIGHT * dVert;
    float tanPhi = num / denom;

    mPath->setTanPhi(tanPhi);

    /*-----------------------------------------------------------------*/
    /*----------------- Horizontal coord. calculation -----------------*/
    /*-----------------------------------------------------------------*/
    float XPos = 0;
    if (mPath->getPrimitive(0)->isValidTime() and
	mPath->getPrimitive(3)->isValidTime())
	XPos = (mPath->getXCoorCell(0) + mPath->getXCoorCell(3)) / 2;
    else
	XPos = (mPath->getXCoorCell(1) + mPath->getXCoorCell(2)) / 2;

    mPath->setHorizPos( XPos );
}

/**
 * Calcula las distancias de deriva respecto de cada "wire" y la posición
 * horizontal del punto de interacción en cada celda respecto del sistema
 * de referencia de la cámara.
 *
 * La posición horizontal de cada hilo es calculada en el "DTPrimitive".
 */
void DTTrigPhase2Prod::calcCellDriftAndXcoor(MuonPath *mPath) {
    //Distancia de deriva en la celda respecto del wire". NO INCLUYE SIGNO.
    float driftDistance;
    float wireHorizPos; // Posicion horizontal del wire.
    float hitHorizPos;  // Posicion del muon en la celda.

    for (int i = 0; i <= 3; i++)
	if (mPath->getPrimitive(i)->isValidTime()) {
	    // Drift distance.
	    driftDistance = DRIFT_SPEED *
		( mPath->getPrimitive(i)->getTDCTimeNoOffset() -
		  mPath->getBxTimeValue()
		  );

	    wireHorizPos = mPath->getPrimitive(i)->getWireHorizPos();

	    if ( (mPath->getLateralComb())[ i ] == LEFT )
		hitHorizPos = wireHorizPos - driftDistance;
	    else
		hitHorizPos = wireHorizPos + driftDistance;

	    mPath->setXCoorCell(hitHorizPos, i);
	    mPath->setDriftDistance(driftDistance, i);
	}
}

/**
 * Calcula el estimador de calidad de la trayectoria.
 */
void DTTrigPhase2Prod::calcChiSquare(MuonPath *mPath) {

    float xi, zi, factor;

    float chi = 0;
    float mu  = mPath->getTanPhi();
    float b   = mPath->getHorizPos();

    const float baseWireYPos = -1.5 * CELL_HEIGHT;

    for (int i = 0; i <= 3; i++)
	if ( mPath->getPrimitive(i)->isValidTime() ) {
	    zi = baseWireYPos + CELL_HEIGHT * i;
	    xi = mPath->getXCoorCell(i);

	    factor = xi - mu*zi - b;
	    chi += (factor * factor);
	}
    mPath->setChiSq(chi);
}


/**
 * Este método devuelve cual layer no se está utilizando en el
 * 'LAYER_ARRANGEMENT' cuyo índice se pasa como parámetro.
 * 
 * ¡¡¡ OJO !!! Este método es completamente dependiente de esa macro.
 * Si hay cambios en ella, HAY QUE CAMBIAR EL MÉTODO.
 * 
 *  LAYER_ARRANGEMENTS[MAX_VERT_ARRANG][3] = {
 *    {0, 1, 2}, {1, 2, 3},                       // Grupos consecutivos
 *    {0, 1, 3}, {0, 2, 3}                        // Grupos salteados
 *  };
 */
int DTTrigPhase2Prod::getOmittedHit(int idx) {
  
    int ans = -1;
  
    switch (idx) {
    case 0: ans = 3; break;
    case 1: ans = 0; break;
    case 2: ans = 2; break;
    case 3: ans = 1; break;
    }

    return ans;
}


int DTTrigPhase2Prod::compute_pathId(MuonPath *mPath) {
    if(debug) std::cout<<"DTp2:\t\t\t pathId: In function compute_pathId, computing_pathId for wires: ";
    for(int i=0;i<=3;i++)
	if(debug) std::cout<<mPath->getPrimitive(i)->getChannelId()<<" ";
    if(debug) std::cout<<std::endl;


    int baseChannel = mPath->getPrimitive(0)->getChannelId();
    int this_path=-1;
    for(this_path=0;this_path<8;this_path++){
	int countValidHits=0;
	for(int i=0;i<=3;i++) {			      
	    if (!mPath->getPrimitive(i)->isValidTime()) continue; //if the primitive no valido 
	    int channel = mPath->getPrimitive(i)->getChannelId();
	    int layout = CELL_HORIZONTAL_LAYOUTS[this_path][i]; 
	    if (baseChannel==999) baseChannel=channel-1; // update baseChannel if still 999
	    int diff = i%2 == 0? 2*(channel - baseChannel) : 2*(channel - baseChannel)-1;
	    if (diff==layout) countValidHits++;
	} 
	if (countValidHits > 3  and  mPath->completeMP()) return this_path;
	if (countValidHits >= 3 and !mPath->completeMP()) return this_path;
    }    	
    if(debug) std::cout<<"DTp2:compute_pathId \t\t\t pathId: pathId not found returning -1 (this should never happen)" <<this_path<<std::endl;
    return -1;
}


