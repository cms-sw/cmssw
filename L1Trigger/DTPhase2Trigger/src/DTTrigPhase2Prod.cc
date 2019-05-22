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
  
    
    // Choosing grouping scheme:
    grcode = pset.getUntrackedParameter<Int_t>("grouping_code");
    
    if (grcode == 0) grouping_obj = new InitialGrouping(pset);
    else {
	if (debug) cout << "DTp2::constructor: Non-valid grouping code. Choosing InitialGrouping by default." << endl;
	grouping_obj = new InitialGrouping(pset);
    }
    
    mpathanalyzer   = new MuonPathAnalyzerPerSL(pset);
    mpathfilter     = new MuonPathFilter(pset);
    mpathassociator = new MuonPathAssociator(pset);
      
   
    
}

DTTrigPhase2Prod::~DTTrigPhase2Prod(){

  //delete inMuonPath;
  //delete outValidMuonPath;
  
  if(debug) std::cout<<"DTp2: calling destructor"<<std::endl;
    
  delete grouping_obj; // Grouping destructor
  delete mpathanalyzer; // Analyzer destructor
  delete mpathfilter; // Filter destructor
  delete mpathassociator; // Associator destructor
}


void DTTrigPhase2Prod::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if(debug) cout << "DTTrigPhase2Prod::beginRun " << iRun.id().run() << endl;
  if(debug) cout << "DTTrigPhase2Prod::beginRun: getting DT geometry" << endl;
    
  if(debug) std::cout<<"getting DT geometry"<<std::endl;
  iEventSetup.get<MuonGeometryRecord>().get(dtGeo);//1103
  
  ESHandle< DTConfigManager > dtConfig ;
  iEventSetup.get< DTConfigManagerRcd >().get( dtConfig );

  grouping_obj->initialise(iEventSetup); // Grouping object initialisation
  mpathanalyzer->initialise(iEventSetup); // Analyzer object initialisation
  mpathfilter->initialise(iEventSetup); // Filter object initialisation
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
    
	if (dmit !=digiMap.end()) grouping_obj->run(iEvent, iEventSetup, (*dmit).second, &muonpaths);  // New grouping implementation
    }

    digiMap.clear();
    // GROUPING ENDS
    if (debug) cout << "MUON PATHS found: " << muonpaths.size() <<" in event"<<iEvent.id().event()<<endl;

    std::vector<metaPrimitive> metaPrimitives;
    mpathanalyzer->run(iEvent, iEventSetup,  muonpaths, metaPrimitives);  // New grouping implementation
    
    if (dump) {
      for (unsigned int i=0; i<metaPrimitives.size(); i++){
	cout << iEvent.id().event() << " mp " << i << ": "
	     << metaPrimitives.at(i).t0 << " "
	     << metaPrimitives.at(i).x << " "
	     << metaPrimitives.at(i).tanPhi << " "
	     << metaPrimitives.at(i).phi << " "
	     << metaPrimitives.at(i).phiB << " "
	     << metaPrimitives.at(i).quality << " "
	     << endl;
      }
    }
    if(debug) std::cout<<"filling NmetaPrimtives"<<std::endl;
    
    if(debug) std::cout<<"deleting muonpaths"<<std::endl;    
    for (unsigned int i=0; i<muonpaths.size(); i++){
      delete muonpaths[i];
    }
    muonpaths.clear();
    
    //FILTER SECTIONS:
    
    //filtro de duplicados puro popdr'ia ir ac'a mpredundantfilter.cpp primos?
    //filtro en |tanPhi|<~1.?

    if(debug) std::cout<<"declaring new vector for filtered"<<std::endl;    

    std::vector<metaPrimitive> filteredMetaPrimitives;
    mpathfilter->run(iEvent, iEventSetup, metaPrimitives, filteredMetaPrimitives);  // New grouping implementation
    
    if (dump) {
      for (unsigned int i=0; i<filteredMetaPrimitives.size(); i++){
	cout << iEvent.id().event() << " filtered mp " << i << ": "
	     << filteredMetaPrimitives.at(i).t0 << " "
	     << filteredMetaPrimitives.at(i).x << " "
	     << filteredMetaPrimitives.at(i).tanPhi << " "
	     << filteredMetaPrimitives.at(i).phi << " "
	     << filteredMetaPrimitives.at(i).phiB << " "
	     << filteredMetaPrimitives.at(i).quality << " "
	     << endl;
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
	cout << iEvent.id().event() << " correlated mp " << i << ": "
	     << correlatedMetaPrimitives.at(i).t0 << " "
	     << correlatedMetaPrimitives.at(i).x << " "
	     << correlatedMetaPrimitives.at(i).tanPhi << " "
	     << correlatedMetaPrimitives.at(i).phi << " "
	     << correlatedMetaPrimitives.at(i).phiB << " "
	     << correlatedMetaPrimitives.at(i).quality << " "
	     << endl;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    /// STORING RESULT: 
    //////////////////////////////////////////
    vector<L1Phase2MuDTPhDigi> outP2Ph;
    
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
      
      if(p2_df==2){
	if(debug)std::cout<<"pushing back phase-2 dataformat carlo-federica dataformat"<<std::endl;
	outP2Ph.push_back(L1Phase2MuDTPhDigi((int)round((*metaPrimitiveIt).t0/25.),   // ubx (m_bx) //bx en la orbita
					     chId.wheel(),   // uwh (m_wheel)     // FIXME: It is not clear who provides this?
					     sectorTP,   // usc (m_sector)    // FIXME: It is not clear who provides this?
					     chId.station(),   // ust (m_station)
					     sl,   // ust (m_station)
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
  mpathfilter->finish();
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



