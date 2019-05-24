#include "L1Trigger/DTPhase2Trigger/interface/DTTrigPhase2Prod.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"


// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include <iostream>
#include <cmath>

#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"

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


DTTrigPhase2Prod::DTTrigPhase2Prod(const ParameterSet& pset){
    
    produces<L1Phase2MuDTPhContainer>();
    
    debug = pset.getUntrackedParameter<bool>("debug");
    dump = pset.getUntrackedParameter<bool>("dump");
    min_phinhits_match_segment = pset.getUntrackedParameter<int>("min_phinhits_match_segment");

    do_correlation = pset.getUntrackedParameter<bool>("do_correlation");
    p2_df = pset.getUntrackedParameter<int>("p2_df");
    
    txt_ttrig_bc0 = pset.getUntrackedParameter<bool>("apply_txt_ttrig_bc0");
    
    dtDigisToken = consumes< DTDigiCollection >(pset.getParameter<edm::InputTag>("digiTag"));

    rpcRecHitsLabel = consumes<RPCRecHitCollection>(pset.getUntrackedParameter < edm::InputTag > ("rpcRecHits"));
    useRPC = pset.getUntrackedParameter<bool>("useRPC");
  



    
    // Choosing grouping scheme:
    grcode = pset.getUntrackedParameter<Int_t>("grouping_code");
    
    if      (grcode == 0) grouping_obj = new InitialGrouping(pset);
    else if (grcode == 1) grouping_obj = new HoughGrouping(pset);
    else if (grcode == 2) grouping_obj = new PseudoBayesGrouping(pset.getParameter<edm::ParameterSet>("PseudoBayesPattern"));
    else {
        if (debug) cout << "DTp2::constructor: Non-valid grouping code. Choosing InitialGrouping by default." << endl;
        grouping_obj = new InitialGrouping(pset);
    }
    
    mpathanalyzer        = new MuonPathAnalyzerPerSL(pset);
    mpathqualityenhancer = new MPQualityEnhancerFilter(pset);
    mpathredundantfilter = new MPRedundantFilter(pset);
    mpathassociator      = new MuonPathAssociator(pset);
}

DTTrigPhase2Prod::~DTTrigPhase2Prod(){

  //delete inMuonPath;
  //delete outValidMuonPath;
  
  if(debug) std::cout<<"DTp2: calling destructor"<<std::endl;
    
  delete grouping_obj; // Grouping destructor
  delete mpathanalyzer; // Analyzer destructor
  delete mpathqualityenhancer; // Filter destructor
  delete mpathredundantfilter; // Filter destructor
  delete mpathassociator; // Associator destructor
}


void DTTrigPhase2Prod::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if(debug) cout << "DTTrigPhase2Prod::beginRun " << iRun.id().run() << endl;
  if(debug) cout << "DTTrigPhase2Prod::beginRun: getting DT geometry" << endl;
    
  if(debug) std::cout<<"getting DT geometry"<<std::endl;
  iEventSetup.get<MuonGeometryRecord>().get(dtGeo);//1103

  if(debug) std::cout<<"getting RPC geometry"<<std::endl;
  iEventSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  ESHandle< DTConfigManager > dtConfig ;
  iEventSetup.get< DTConfigManagerRcd >().get( dtConfig );

  grouping_obj->initialise(iEventSetup); // Grouping object initialisation
  mpathanalyzer->initialise(iEventSetup); // Analyzer object initialisation
  mpathqualityenhancer->initialise(iEventSetup); // Filter object initialisation
  mpathredundantfilter->initialise(iEventSetup); // Filter object initialisation
  mpathassociator->initialise(iEventSetup); // Associator object initialisation
  
    //trigGeomUtils = new DTTrigGeomUtils(dtGeo);

    //filling up zcn
    for (int ist=0; ist<4; ++ist) {
	const DTChamberId chId(-2,ist+1,4);
	const DTChamber *chamb = dtGeo->chamber(chId);
	const DTSuperLayer *sl1 = chamb->superLayer(DTSuperLayerId(chId,1));
	const DTSuperLayer *sl3 = chamb->superLayer(DTSuperLayerId(chId,3));
	zcn[ist] = .5*(chamb->surface().toLocal(sl1->position()).z() + chamb->surface().toLocal(sl3->position()).z());
    }

    const DTChamber* chamb   = dtGeo->chamber(DTChamberId(-2,4,13));
    const DTChamber* scchamb = dtGeo->chamber(DTChamberId(-2,4,4));
    xCenter[0] = scchamb->toLocal(chamb->position()).x()*.5;
    chamb   = dtGeo->chamber(DTChamberId(-2,4,14));
    scchamb = dtGeo->chamber(DTChamberId(-2,4,10));
    xCenter[1] = scchamb->toLocal(chamb->position()).x()*.5;
}


void DTTrigPhase2Prod::produce(Event & iEvent, const EventSetup& iEventSetup){
    if(debug) cout << "DTTrigPhase2Prod::produce " << endl;
    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByToken(dtDigisToken, dtdigis);
    
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
        
        if (dmit !=digiMap.end()) grouping_obj->run(iEvent, iEventSetup, (*dmit).second, &muonpaths);
    }
    
    digiMap.clear();
    
    
    if ((grcode != 0)) {
        if (debug) cout << "DTTrigPhase2Prod::produce - WARNING: non-standard grouping chosen. Further execution still not functioning." << endl;
        return;
    }
  
    // FILTER GROUPING
    std::vector<MuonPath*> filteredmuonpaths;
    mpathredundantfilter->run(iEvent, iEventSetup, muonpaths,filteredmuonpaths);

    if(debug) std::cout<<"deleting muonpaths"<<std::endl;
    for (unsigned int i=0; i<muonpaths.size(); i++){
      delete muonpaths[i];
    }
    muonpaths.clear();

    // GROUPING ENDS
    
    
    if (debug) cout << "MUON PATHS found: " << filteredmuonpaths.size() <<" in event"<<iEvent.id().event()<<endl;
    if(debug) std::cout<<"filling NmetaPrimtives"<<std::endl;
    std::vector<metaPrimitive> metaPrimitives;
    mpathanalyzer->run(iEvent, iEventSetup,  filteredmuonpaths, metaPrimitives);  // New grouping implementation
    
    if (dump) {
	for (unsigned int i=0; i<metaPrimitives.size(); i++){
	    cout << iEvent.id().event() << " mp " << i << ": ";
	    printmP(metaPrimitives.at(i));
	    cout<<endl;
	}
    }

    for (unsigned int i=0; i<filteredmuonpaths.size(); i++){
      delete filteredmuonpaths[i];
    }
    filteredmuonpaths.clear();
    
    
    //FILTER SECTIONS:
    
    //filtro de duplicados puro popdr'ia ir ac'a mpredundantfilter.cpp primos?
    //filtro en |tanPhi|<~1.?

    if(debug) std::cout<<"declaring new vector for filtered"<<std::endl;    

    std::vector<metaPrimitive> filteredMetaPrimitives;
    mpathqualityenhancer->run(iEvent, iEventSetup, metaPrimitives, filteredMetaPrimitives);  // New grouping implementation
    
    if (dump) {
      for (unsigned int i=0; i<filteredMetaPrimitives.size(); i++){
	  cout << iEvent.id().event() << " filtered mp " << i << ": ";
	  printmP(metaPrimitives.at(i));
	  cout<<endl;
      }
    }
    
    metaPrimitives.clear();
    metaPrimitives.erase(metaPrimitives.begin(),metaPrimitives.end());
    
    if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()<<" we found "<<filteredMetaPrimitives.size()<<" filteredMetaPrimitives (superlayer)"<<std::endl;
    if(debug) std::cout<<"filteredMetaPrimitives: starting correlations"<<std::endl;    
    
    //// CORRELATION: 
    std::vector<metaPrimitive> correlatedMetaPrimitives;
    mpathassociator->run(iEvent, iEventSetup, dtdigis, filteredMetaPrimitives, correlatedMetaPrimitives);  
    
    filteredMetaPrimitives.clear();
    filteredMetaPrimitives.erase(filteredMetaPrimitives.begin(),filteredMetaPrimitives.end());

    if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()
		       <<" we found "<<correlatedMetaPrimitives.size()
		       <<" correlatedMetPrimitives (chamber)"<<std::endl;

    if (dump) {
      for (unsigned int i=0; i<correlatedMetaPrimitives.size(); i++){
	  cout << iEvent.id().event() << " correlated mp " << i << ": ";
	  printmP(metaPrimitives.at(i));
	  cout<<endl;
      }
    }
    

    /// STORING RESULTs 

    vector<L1Phase2MuDTPhDigi> outP2Ph;
    
    // First we asociate a new index to the metaprimitive depending on quality or phiB; 
    uint32_t rawId = -1; 
    int numP = -1;

    for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){
	numP++;
	rawId = (*metaPrimitiveIt).rawId;   

	int inf = 0;
	int numP2 = -1;  
	for (auto metaPrimitiveItN = correlatedMetaPrimitives.begin(); metaPrimitiveItN != correlatedMetaPrimitives.end(); ++metaPrimitiveItN){
	    numP2++;
	    if (rawId != (*metaPrimitiveItN).rawId) continue; 
	    if (numP2 == numP) {
		(*metaPrimitiveIt).index = inf; 
		break;  
	    } else if ((*metaPrimitiveIt).quality < (*metaPrimitiveItN).quality) {
		inf++;
	    } else if ((*metaPrimitiveIt).quality > (*metaPrimitiveItN).quality) {
		(*metaPrimitiveItN).index++;
	    } else if ((*metaPrimitiveIt).quality == (*metaPrimitiveItN).quality) {
		if (fabs((*metaPrimitiveIt).phiB) >= fabs((*metaPrimitiveItN).phiB) ){
		    inf++;
		} else if (fabs((*metaPrimitiveIt).phiB) < fabs((*metaPrimitiveItN).phiB) ){
		    (*metaPrimitiveItN).index++;
		}
	    }
	}

    }

    
    for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){
      DTChamberId chId((*metaPrimitiveIt).rawId);
      if(debug) std::cout<<"looping in final vector: SuperLayerId"<<chId<<" x="<<(*metaPrimitiveIt).x<<" quality="<<(*metaPrimitiveIt).quality<<std::endl;
      
      int sectorTP=chId.sector();
      if(sectorTP==13) sectorTP=4;
      if(sectorTP==14) sectorTP=10;
      sectorTP=sectorTP-1;
      
      int sl=0;
      if((*metaPrimitiveIt).quality < 6 || (*metaPrimitiveIt).quality == 7){
	  if(inner((*metaPrimitiveIt))) sl=1;
	  else sl=3;
      }


      double shift_back=0;
      if (iEvent.eventAuxiliary().bunchCrossing() == -1) //FIX MC                                                                                                 
	  shift_back = 400;

      
      if(p2_df==2){
	  if(debug)std::cout<<"pushing back phase-2 dataformat carlo-federica dataformat"<<std::endl;
	  outP2Ph.push_back(L1Phase2MuDTPhDigi((int)round((*metaPrimitiveIt).t0/25.)-shift_back,   // ubx (m_bx) //bx en la orbita
					       chId.wheel(),   // uwh (m_wheel)     // FIXME: It is not clear who provides this?
					       sectorTP,   // usc (m_sector)    // FIXME: It is not clear who provides this?
					       chId.station(),   // ust (m_station)
					       sl,   // ust (m_station)
					     (int)round((*metaPrimitiveIt).phi*65536./0.8), // uphi (_phiAngle)
					       (int)round((*metaPrimitiveIt).phiB*2048./1.4), // uphib (m_phiBending)
					     (*metaPrimitiveIt).quality,  // uqua (m_qualityCode)
					       (*metaPrimitiveIt).index,  // uind (m_segmentIndex)
					       (int)round((*metaPrimitiveIt).t0)-shift_back*25,  // ut0 (m_t0Segment)
					       (int)round((*metaPrimitiveIt).chi2*1000000),  // uchi2 (m_chi2Segment)
					       -10    // urpc (m_rpcFlag)
					       ));
	
      }
    }
    
    // Store RPC hits
    if(useRPC){
        for (RPCRecHitCollection::const_iterator rpcIt = rpcHits->begin(); rpcIt != rpcHits->end(); rpcIt++) {
            // Retrieve RPC info and translate it to DT convention if needed
            int rpc_bx = rpcIt->BunchX(); // FIXME how to get bx w.r.t. orbit start?
            int rpc_time = int(rpcIt->time());// FIXME need to follow DT convention
            RPCDetId rpcDetId = (RPCDetId)(*rpcIt).rpcId();
            if(debug) std::cout << "Getting RPC info from : " << rpcDetId << std::endl;
            int rpc_region = rpcDetId.region();
            if(rpc_region != 0 ) continue; // Region = 0 Barrel
            int rpc_wheel = rpcDetId.ring(); // In barrel, wheel is accessed via ring() method ([-2,+2])
            int rpc_dt_sector = rpcDetId.sector()-1; // DT sector:[0,11] while RPC sector:[1,12]
            int rpc_station = rpcDetId.station();

            if(debug) std::cout << "Getting RPC global point and translating to DT local coordinates" << std::endl;
            GlobalPoint rpc_gp = getRPCGlobalPosition(rpcDetId, *rpcIt);
            int rpc_global_phi = rpc_gp.phi();
            int rpc_localDT_phi = std::numeric_limits<int>::min();
            // FIXME Adaptation of L1Trigger/L1TTwinMux/src/RPCtoDTTranslator.cc radialAngle function, should maybe be updated
            if (rpcDetId.sector() == 1) rpc_localDT_phi = int(rpc_global_phi * 1024);
            else {
                if (rpc_global_phi >= 0) rpc_localDT_phi = int((rpc_localDT_phi - rpc_dt_sector * Geom::pi() / 6.) * 1024);
                else rpc_localDT_phi = int((rpc_global_phi + (13 - rpcDetId.sector()) * Geom::pi() / 6.) * 1024);
            }
            int rpc_phiB = std::numeric_limits<int>::min(); // single hit has no phiB and 0 is legal value for DT phiB
            int rpc_quality = -1; // to be decided
            int rpc_index = 0;
            int rpc_BxCnt = 0;
            int rpc_flag = 3; // only single hit for now
            if(p2_df == 2){
                if(debug)std::cout<<"pushing back phase-2 dataformat carlo-federica dataformat"<<std::endl;
                outP2Ph.push_back(L1Phase2MuDTPhDigi(rpc_bx,
                            rpc_wheel,
                            rpc_dt_sector,
                            rpc_station,
                            0, //this would be the layer in the new dataformat
                            rpc_localDT_phi,
                            rpc_phiB,
                            rpc_quality,
                            rpc_index,
                            rpc_time,
                            -1, // signle hit --> no chi2
                            rpc_flag
                            ));
            }
        }
    }

    if(p2_df==2){
      std::unique_ptr<L1Phase2MuDTPhContainer> resultP2Ph (new L1Phase2MuDTPhContainer);
      resultP2Ph->setContainer(outP2Ph); iEvent.put(std::move(resultP2Ph));
      outP2Ph.clear();
      outP2Ph.erase(outP2Ph.begin(),outP2Ph.end());
    }
}


void DTTrigPhase2Prod::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  grouping_obj->finish();
  mpathanalyzer->finish();
  mpathqualityenhancer->finish();
  mpathredundantfilter->finish();
  mpathassociator->finish();
};


bool DTTrigPhase2Prod::outer(metaPrimitive mp){
    if(mp.wi1==-1 and mp.wi2==-1 and mp.wi3==-1 and mp.wi4==-1)
	return true;
    return false;
}

bool DTTrigPhase2Prod::inner(metaPrimitive mp){
    if(mp.wi5==-1 and mp.wi6==-1 and mp.wi7==-1 and mp.wi8==-1)
        return true;
    return false;
}

bool DTTrigPhase2Prod::hasPosRF(int wh,int sec){
    return  wh>0 || (wh==0 && sec%4>1);
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

int DTTrigPhase2Prod::rango(metaPrimitive mp) {
    if(mp.quality==1 or mp.quality==2) return 3;
    if(mp.quality==3 or mp.quality==4) return 4;
    return 0;
}

GlobalPoint DTTrigPhase2Prod::getRPCGlobalPosition(RPCDetId rpcId, const RPCRecHit& rpcIt) const{

  RPCDetId rpcid = RPCDetId(rpcId);
  const LocalPoint& rpc_lp = rpcIt.localPosition();
  const GlobalPoint& rpc_gp = rpcGeo->idToDet(rpcid)->surface().toGlobal(rpc_lp);

  return rpc_gp;

}

