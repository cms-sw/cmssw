#include "L1Trigger/DTPhase2Trigger/interface/DTTrigPhase2Prod.h"

using namespace edm;
using namespace std;

typedef vector<DTSectCollPhSegm>          SectCollPhiColl;
typedef SectCollPhiColl::const_iterator   SectCollPhiColl_iterator;
typedef vector<DTSectCollThSegm>          SectCollThetaColl;
typedef SectCollThetaColl::const_iterator SectCollThetaColl_iterator;

struct {
  Bool_t operator()(std::pair<DTLayerId*, DTDigi*> a, std::pair<DTLayerId*, DTDigi*> b) const {
    return (a.second->time() < b.second->time());
  }
} DigiTimeOrdering;

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


DTTrigPhase2Prod::DTTrigPhase2Prod(const ParameterSet& pset) {
  produces<L1Phase2MuDTPhContainer>();

  debug = pset.getUntrackedParameter<bool>("debug");
  dump = pset.getUntrackedParameter<bool>("dump");
  min_phinhits_match_segment = pset.getUntrackedParameter<int>("min_phinhits_match_segment");

  do_correlation = pset.getUntrackedParameter<bool>("do_correlation");
  p2_df = pset.getUntrackedParameter<int>("p2_df");
    
  scenario = pset.getUntrackedParameter<int>("scenario");
  printPython = pset.getUntrackedParameter<bool>("printPython");
  printHits = pset.getUntrackedParameter<bool>("printHits");
       
 
  txt_ttrig_bc0 = pset.getUntrackedParameter<bool>("apply_txt_ttrig_bc0");
    
  dtDigisToken = consumes< DTDigiCollection >(pset.getParameter<edm::InputTag>("digiTag"));

  rpcRecHitsLabel = consumes<RPCRecHitCollection>(pset.getUntrackedParameter < edm::InputTag > ("rpcRecHits"));
  useRPC = pset.getUntrackedParameter<bool>("useRPC");
  
  uint32_t rawId;
  shift_filename = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
      << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_filename.fullPath();
  }
  while (ifin3.good()){
    ifin3 >> rawId >> shift;
    shiftinfo[rawId]=shift;
  }



  // Choosing grouping scheme:
  grcode = pset.getUntrackedParameter<int>("grouping_code");

  if      (grcode == 0) grouping_obj = new InitialGrouping(pset);
  else if (grcode == 1) grouping_obj = new HoughGrouping(pset.getParameter<edm::ParameterSet>("HoughGrouping"));
  else if (grcode == 2) grouping_obj = new PseudoBayesGrouping(pset.getParameter<edm::ParameterSet>("PseudoBayesPattern"));
  else {
    if (debug) cout << "DTp2::constructor: Non-valid grouping code. Choosing InitialGrouping by default." << endl;
    grouping_obj = new InitialGrouping(pset);
  }

  if (grcode==0) {
    if (debug) cout << "DTp2:constructor: JM analyzer" << endl;
    mpathanalyzer        = new MuonPathAnalyzerPerSL(pset);
  } else {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" <<endl;
    cout << "               WARNING!!!!!                   " <<endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" <<endl;
    cout << " This grouping option is not fully supported  " <<endl;
    cout << " yet.                                         " <<endl;
    cout << " USE IT AT YOUR OWN RISK!                     " <<endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" <<endl;
    if (debug) cout << "DTp2:constructor: Full chamber analyzer" << endl;
    mpathanalyzer        = new MuonPathAnalyzerInChamber(pset);
  }

  // Getting buffer option
  activateBuffer          = pset.getUntrackedParameter<Bool_t>("activateBuffer");
  superCellhalfspacewidth = pset.getUntrackedParameter<Int_t>("superCellspacewidth")/2;
  superCelltimewidth      = pset.getUntrackedParameter<Double_t>("superCelltimewidth");

  mpathqualityenhancer = new MPQualityEnhancerFilter(pset);
  mpathredundantfilter = new MPRedundantFilter(pset);
  mpathassociator      = new MuonPathAssociator(pset);
  rpc_integrator       = new RPCIntegrator(pset);
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
  delete rpc_integrator;
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
  if(debug) cout << "DTTrigPhase2Prod::produce" << endl;
  edm::Handle<DTDigiCollection> dtdigis;
  iEvent.getByToken(dtDigisToken, dtdigis);

  if(debug) std::cout <<"\t Getting the RPC RecHits"<<std::endl;
  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByToken(rpcRecHitsLabel,rpcRecHits);

  ////////////////////////////////
  // GROUPING CODE:
  ////////////////////////////////
  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtdigis->begin(); detUnitIt!=dtdigis->end(); ++detUnitIt) {
    const DTLayerId& layId               = (*detUnitIt).first;
    const DTChamberId chambId            = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    digiMap[chambId].put(range,layId);
  }

  // generate a list muon paths for each event!!!
  if (debug && activateBuffer) cout << "DTTrigPhase2Prod::produce - Getting and grouping digis per chamber using a buffer and super cells." << endl;
  else if (debug)              cout << "DTTrigPhase2Prod::produce - Getting and grouping digis per chamber." << endl;
  std::vector<MuonPath*> muonpaths;
  for (std::vector<const DTChamber*>::const_iterator ich = dtGeo->chambers().begin(); ich != dtGeo->chambers().end(); ich++) {
    // The code inside this for loop would ideally later fit inside a trigger unit (in principle, a DT station) of the future Phase 2 DT Trigger.
    const DTChamber* chamb  = (*ich);
    DTChamberId chid        = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);

    if (dmit == digiMap.end()) continue;

    if (activateBuffer) { // Use buffering (per chamber) or not
      // Import digis from the station
      std::vector<std::pair<DTLayerId*, DTDigi*>> tmpvec; tmpvec.clear();

      for (DTDigiCollection::DigiRangeIterator dtLayerIdIt = (*dmit).second.begin();   dtLayerIdIt != (*dmit).second.end();      dtLayerIdIt++) {
        for (DTDigiCollection::const_iterator  digiIt = ((*dtLayerIdIt).second).first; digiIt != ((*dtLayerIdIt).second).second; digiIt++) {
          DTLayerId* tmplayer = new DTLayerId((*dtLayerIdIt).first);
          DTDigi*    tmpdigi  = new DTDigi((*digiIt));
          tmpvec.push_back({tmplayer, tmpdigi});
        }
      }

      // Check to enhance CPU time usage
      if (tmpvec.size() == 0) continue;

      // Order digis depending on TDC time and insert them into a queue (FIFO buffer). TODO: adapt for MC simulations.
      std::sort(tmpvec.begin(), tmpvec.end(), DigiTimeOrdering);
      std::queue<std::pair<DTLayerId*, DTDigi*>> timequeue;

      for (auto &elem : tmpvec) timequeue.push(std::move(elem));
      tmpvec.clear();

      // Distribute the digis from the queue into supercells
      std::vector<DTDigiCollection*> superCells;
      superCells = distribDigis(timequeue);

      // Process each supercell & collect the resulting muonpaths (as the muonpaths std::vector is only enlarged each time
      // the groupings access it, it's not needed to "collect" the final products).
      while (!superCells.empty()) {
        grouping_obj->run(iEvent, iEventSetup, *(superCells.back()), &muonpaths);
        superCells.pop_back();
      }
    }
    else {
      grouping_obj->run(iEvent, iEventSetup, (*dmit).second, &muonpaths);
    }
  }
  digiMap.clear();


  if (dump) {
    for (unsigned int i=0; i<muonpaths.size(); i++){
      cout << iEvent.id().event() << "      mpath " << i << ": ";
      for (int lay=0; lay<muonpaths.at(i)->getNPrimitives(); lay++)
        cout << muonpaths.at(i)->getPrimitive(lay)->getChannelId() << " ";
      for (int lay=0; lay<muonpaths.at(i)->getNPrimitives(); lay++)
        cout << muonpaths.at(i)->getPrimitive(lay)->getTDCTime() << " ";
      for (int lay=0; lay<muonpaths.at(i)->getNPrimitives(); lay++)
        cout << muonpaths.at(i)->getPrimitive(lay)->getLaterality() << " ";
      cout << endl;
    }
    cout << endl;
  }

  // FILTER GROUPING
  std::vector<MuonPath*> filteredmuonpaths;
  if (grcode==0) {
    mpathredundantfilter->run(iEvent, iEventSetup, muonpaths,filteredmuonpaths);
  }
    
  if (dump) {
    for (unsigned int i=0; i<filteredmuonpaths.size(); i++){
      cout << iEvent.id().event() << " filt. mpath " << i << ": ";
      for (int lay=0; lay<filteredmuonpaths.at(i)->getNPrimitives(); lay++)
        cout << filteredmuonpaths.at(i)->getPrimitive(lay)->getChannelId() << " ";
      for (int lay=0; lay<filteredmuonpaths.at(i)->getNPrimitives(); lay++)
        cout << filteredmuonpaths.at(i)->getPrimitive(lay)->getTDCTime() << " ";
      cout << endl;
    }
    cout << endl;
  }

  ///////////////////////////////////////////
  /// FITTING SECTION;
  ///////////////////////////////////////////
  if(debug) cout << "MUON PATHS found: " << muonpaths.size() << " ("<< filteredmuonpaths.size() <<") in event "<< iEvent.id().event()<<endl;
  if(debug) std::cout<<"filling NmetaPrimtives"<<std::endl;
  std::vector<metaPrimitive> metaPrimitives;
  std::vector<MuonPath*> outmpaths;
  if (grcode==0) {
    if (debug) cout << "Fitting 1SL " << endl;
    mpathanalyzer->run(iEvent, iEventSetup,  filteredmuonpaths, metaPrimitives);
  }
  else   {
    // implementation for advanced (2SL) grouping, no filter required..
    if (debug) cout << "Fitting 2SL at once " << endl;
    mpathanalyzer->run(iEvent, iEventSetup,  muonpaths, outmpaths);
  }

  if (dump) {
    for (unsigned int i=0; i<outmpaths.size(); i++){
      cout << iEvent.id().event() << " mp " << i << ": "
      << outmpaths.at(i)->getBxTimeValue() << " "
      << outmpaths.at(i)->getHorizPos() << " "
      << outmpaths.at(i)->getTanPhi() << " "
      << outmpaths.at(i)->getPhi() << " "
      << outmpaths.at(i)->getPhiB() << " "
      << outmpaths.at(i)->getQuality() << " "
      << outmpaths.at(i)->getChiSq() << " "
      << endl;
      }
    for (unsigned int i=0; i<metaPrimitives.size(); i++){
      cout << iEvent.id().event() << " mp " << i << ": ";
      printmP(metaPrimitives.at(i));
      cout<<endl;
    }
  }


  if(debug) std::cout<<"deleting muonpaths"<<std::endl;
    for (unsigned int i=0; i<muonpaths.size(); i++){
      delete muonpaths[i];
    }
    muonpaths.clear();

    for (unsigned int i=0; i<filteredmuonpaths.size(); i++){
      delete filteredmuonpaths[i];
    }
    filteredmuonpaths.clear();
    
    
    /////////////////////////////////////
    //  FILTER SECTIONS:
    ////////////////////////////////////
    //filtro de duplicados puro popdr'ia ir ac'a mpredundantfilter.cpp primos?
    //filtro en |tanPhi|<~1.?

    if(debug) std::cout<<"declaring new vector for filtered"<<std::endl;    

    std::vector<metaPrimitive> filteredMetaPrimitives;
    if (grcode==0) mpathqualityenhancer->run(iEvent, iEventSetup, metaPrimitives, filteredMetaPrimitives);  
    
    if (dump) {
      for (unsigned int i=0; i<filteredMetaPrimitives.size(); i++){
        cout << iEvent.id().event() << " filtered mp " << i << ": ";
        printmP(filteredMetaPrimitives.at(i));
        cout<<endl;
      }
    }
    
    metaPrimitives.clear();
    metaPrimitives.erase(metaPrimitives.begin(),metaPrimitives.end());
    
    if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()<<" we found "<<filteredMetaPrimitives.size()<<" filteredMetaPrimitives (superlayer)"<<std::endl;
    if(debug) std::cout<<"filteredMetaPrimitives: starting correlations"<<std::endl;    
  
    
    ////// PRINT FILTERED METAPRIM
    if (printPython) { 
    //if (true == false) { 
    for (auto metaPrimitiveIt = filteredMetaPrimitives.begin(); metaPrimitiveIt != filteredMetaPrimitives.end(); ++metaPrimitiveIt){

      //if (metaPrimitiveIt->quality != 3 && metaPrimitiveIt->quality != 4) continue;
      DTSuperLayerId slId((metaPrimitiveIt)->rawId);
      
      DTChamberId chId((metaPrimitiveIt)->rawId);
      DTWireId wireId1(chId,1,2,1);
      DTWireId wireId3(chId,3,2,1);
      int sl = -1; 
      if(slId.superlayer() == 1) sl = 0;
      else if (slId.superlayer() == 3) sl=2;
      else continue; 

      if (sl!=0) continue;

      //int nhits = (metaPrimitiveIt->tdc1 != -1) + (metaPrimitiveIt->tdc2 != -1) + (metaPrimitiveIt->tdc3 != -1) + (metaPrimitiveIt->tdc4 != -1);
      //if (nhits == 4 && metaPrimitiveIt->quality<3) continue; 


      float shift = shiftinfo[wireId3.rawId()] - shiftinfo[wireId1.rawId()]; 
      if (printHits && metaPrimitiveIt->wi1!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << sl << " " << 0 << " " << metaPrimitiveIt->wi1 << " " << metaPrimitiveIt->tdc1 << endl;   
      if (printHits && metaPrimitiveIt->wi2!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << sl << " " << 1 << " " << metaPrimitiveIt->wi2 << " " << metaPrimitiveIt->tdc2 << endl;   
      if (printHits && metaPrimitiveIt->wi3!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << sl << " " << 2 << " " << metaPrimitiveIt->wi3 << " " << metaPrimitiveIt->tdc3 << endl;   
      if (printHits && metaPrimitiveIt->wi4!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << sl << " " << 3 << " " << metaPrimitiveIt->wi4 << " " << metaPrimitiveIt->tdc4 << endl;   


   //   if (printHits) cout << (*metaPrimitiveIt).wi1 << "," <<(*metaPrimitiveIt).wi2 << "," <<(*metaPrimitiveIt).wi3 << "," <<(*metaPrimitiveIt).wi4 << "," <<(*metaPrimitiveIt).wi5 << "," <<(*metaPrimitiveIt).wi6 << "," <<(*metaPrimitiveIt).wi7 << "," <<(*metaPrimitiveIt).wi8 << "," << (*metaPrimitiveIt).tdc1 << "," <<(*metaPrimitiveIt).tdc2 << "," <<(*metaPrimitiveIt).tdc3 << "," <<(*metaPrimitiveIt).tdc4 << "," <<(*metaPrimitiveIt).tdc5 << "," <<(*metaPrimitiveIt).tdc6 << "," <<(*metaPrimitiveIt).tdc7 << "," <<(*metaPrimitiveIt).tdc8 << "," << shiftinfo[wireId3.rawId()] - shiftinfo[wireId1.rawId()] << "," << wireId1.wheel() << "," << wireId1.sector() << "," << wireId1.station() << endl;
      if (!printHits && sl==0)   cout << (*metaPrimitiveIt).quality << " " << (*metaPrimitiveIt).x << " " << (*metaPrimitiveIt).tanPhi << " " << (int) (*metaPrimitiveIt).t0 << " " << (*metaPrimitiveIt).chi2  << " " << shiftinfo[wireId1.rawId()]<<" " <<  wireId1.wheel() << " " << wireId1.sector() << " " << wireId1.station() << " " << (*metaPrimitiveIt).wi1<< " " << (*metaPrimitiveIt).wi2<< " " << (*metaPrimitiveIt).wi3<< " " << (*metaPrimitiveIt).wi4<< " " << -1<< " " <<-1<< " " << -1<< " " << -1<< " " << (*metaPrimitiveIt).tdc1<< " " << (*metaPrimitiveIt).tdc2<< " " << (*metaPrimitiveIt).tdc3<< " " << (*metaPrimitiveIt).tdc4<< " " << -1<< " " << -1 << " " << -1<< " " << -1 << " " << (*metaPrimitiveIt).lat1<< " " << (*metaPrimitiveIt).lat2<< " " << (*metaPrimitiveIt).lat3<< " " << (*metaPrimitiveIt).lat4<< " " << -1 << " " << -1 << " " << -1 << " " << -1  << " " << eventBX<< endl;
      if (!printHits && sl==2)   cout << (*metaPrimitiveIt).quality << " " << (*metaPrimitiveIt).x << " " << (*metaPrimitiveIt).tanPhi << " " << (int) (*metaPrimitiveIt).t0 << " " << (*metaPrimitiveIt).chi2 << " " << shiftinfo[wireId1.rawId()]<<" " <<  wireId1.wheel() << " " << wireId1.sector() << " " << wireId1.station() << " " <<-1 << " " << -1<< " " << -1<< " " << -1<< " " << (*metaPrimitiveIt).wi1<< " " << (*metaPrimitiveIt).wi2<< " " << (*metaPrimitiveIt).wi3<< " " << (*metaPrimitiveIt).wi4<< " " << -1 << " " << -1 << " " << -1 << " " << -1<< " " << (*metaPrimitiveIt).tdc1<< " " << (*metaPrimitiveIt).tdc2<< " " << (*metaPrimitiveIt).tdc3<< " " << (*metaPrimitiveIt).tdc4<< " " << -1<< " " <<-1<< " " << -1 << " " << -1 << " " << (*metaPrimitiveIt).lat1<< " " << (*metaPrimitiveIt).lat2<< " " << (*metaPrimitiveIt).lat3<< " " << (*metaPrimitiveIt).lat4  << " " << eventBX <<endl;
    }
    //if (printHits) cout << "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" << endl;
    if (printHits) cout << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
    if (!printHits) cout << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1<< " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
    }

 
    /////////////////////////////////////
    //// CORRELATION: 
    /////////////////////////////////////
    std::vector<metaPrimitive> correlatedMetaPrimitives;
    if (grcode==0) mpathassociator->run(iEvent, iEventSetup, dtdigis, filteredMetaPrimitives, correlatedMetaPrimitives);
    else {
      //for(auto muonpath = muonpaths.begin();muonpath!=muonpaths.end();++muonpath) {
      for(auto muonpath = outmpaths.begin();muonpath!=outmpaths.end();++muonpath) {
        correlatedMetaPrimitives.push_back(metaPrimitive({(*muonpath)->getRawId(),(double)(*muonpath)->getBxTimeValue(),
          (*muonpath)->getHorizPos(), (*muonpath)->getTanPhi(),
          (*muonpath)->getPhi(), 	    (*muonpath)->getPhiB(),
          (*muonpath)->getChiSq(),    (int)(*muonpath)->getQuality(),
          (*muonpath)->getPrimitive(0)->getChannelId(), (*muonpath)->getPrimitive(0)->getTDCTime(), (*muonpath)->getPrimitive(0)->getLaterality(),
          (*muonpath)->getPrimitive(1)->getChannelId(), (*muonpath)->getPrimitive(1)->getTDCTime(), (*muonpath)->getPrimitive(1)->getLaterality(),
          (*muonpath)->getPrimitive(2)->getChannelId(), (*muonpath)->getPrimitive(2)->getTDCTime(), (*muonpath)->getPrimitive(2)->getLaterality(),
          (*muonpath)->getPrimitive(3)->getChannelId(), (*muonpath)->getPrimitive(3)->getTDCTime(), (*muonpath)->getPrimitive(3)->getLaterality(),
          (*muonpath)->getPrimitive(4)->getChannelId(), (*muonpath)->getPrimitive(4)->getTDCTime(), (*muonpath)->getPrimitive(4)->getLaterality(),
          (*muonpath)->getPrimitive(5)->getChannelId(), (*muonpath)->getPrimitive(5)->getTDCTime(), (*muonpath)->getPrimitive(5)->getLaterality(),
          (*muonpath)->getPrimitive(6)->getChannelId(), (*muonpath)->getPrimitive(6)->getTDCTime(), (*muonpath)->getPrimitive(6)->getLaterality(),
          (*muonpath)->getPrimitive(7)->getChannelId(), (*muonpath)->getPrimitive(7)->getTDCTime(), (*muonpath)->getPrimitive(7)->getLaterality(),
        }));
      }
    }
    filteredMetaPrimitives.clear();
    filteredMetaPrimitives.erase(filteredMetaPrimitives.begin(),filteredMetaPrimitives.end());

    if(debug) std::cout<<"DTp2 in event:"<<iEvent.id().event()
          <<" we found "<<correlatedMetaPrimitives.size()
          <<" correlatedMetPrimitives (chamber)"<<std::endl;
    
    if (dump) {
      for (unsigned int i=0; i<correlatedMetaPrimitives.size(); i++){
        cout << iEvent.id().event() << " correlated mp " << i << ": ";
        printmPC(correlatedMetaPrimitives.at(i));
        cout<<endl;
      }
    }

    double shift_back=0;

    //if (iEvent.eventAuxiliary().run() == 1) //FIX MC
    if(scenario == 0)//scope for MC
        shift_back = 400;

    if(scenario == 1)//scope for data
        shift_back=0;

    if(scenario == 2)//scope for slice test
        shift_back=0;
    
    // RPC integration
    if(useRPC) {
      if (debug) std::cout << "Start integrating RPC" << std::endl;
      rpc_integrator->initialise(iEventSetup, shift_back);
      rpc_integrator->prepareMetaPrimitives(rpcRecHits);
      rpc_integrator->matchWithDTAndUseRPCTime(correlatedMetaPrimitives);
      rpc_integrator->makeRPCOnlySegments();
      rpc_integrator->storeRPCSingleHits();
      rpc_integrator->removeRPCHitsUsed();
    }

    /// STORING RESULTs 

    vector<L1Phase2MuDTPhDigi> outP2Ph;
    
    // Assigning index value

    assignIndex(correlatedMetaPrimitives);

   // if (true == false) { 
    if (printPython && printHits)  { 
      for (DTDigiCollection::DigiRangeIterator dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){ 
        const DTLayerId& thisLayerId = (*dtLayerId_It).first;
        const DTChamberId chId = thisLayerId.chamberId();
	DTWireId wireId1(chId,1,2,1);
        DTWireId wireId3(chId,3,2,1);
        Int_t superLayer = thisLayerId.superlayerId().superLayer();
	if (superLayer == 2) continue;  
        Int_t layer = thisLayerId.layer();
        float shift = shiftinfo[wireId3.rawId()] - shiftinfo[wireId1.rawId()]; 

        for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){ 
           Int_t wire     = (*digiIt).wire() - 1; 
           Int_t digiTIME = (*digiIt).time(); 
           if (abs (digiTIME) < 200000) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << superLayer-1 << " " << layer-1 << " " << wire << " " << digiTIME << endl;   
        } 
      } 
      if (printHits)  cout << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
    } 

  

    if (true == false) { 
    //if (printPython) { 
    for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){

      //if (metaPrimitiveIt->quality != 9 && metaPrimitiveIt->quality != 9) continue;
      DTChamberId chId((metaPrimitiveIt)->rawId);
      DTWireId wireId1(chId,1,2,1);
      DTWireId wireId3(chId,3,2,1);


      //float shift = shiftinfo[wireId3.rawId()] - shiftinfo[wireId1.rawId()]; 
      /*if (printHits && metaPrimitiveIt->wi1!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 0 << " " << 0 << " " << metaPrimitiveIt->wi1 << " " << metaPrimitiveIt->tdc1 << endl;   
      if (printHits && metaPrimitiveIt->wi2!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 0 << " " << 1 << " " << metaPrimitiveIt->wi2 << " " << metaPrimitiveIt->tdc2 << endl;   
      if (printHits && metaPrimitiveIt->wi3!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 0 << " " << 2 << " " << metaPrimitiveIt->wi3 << " " << metaPrimitiveIt->tdc3 << endl;   
      if (printHits && metaPrimitiveIt->wi4!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 0 << " " << 3 << " " << metaPrimitiveIt->wi4 << " " << metaPrimitiveIt->tdc4 << endl;   
      if (printHits && metaPrimitiveIt->wi5!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 2 << " " << 0 << " " << metaPrimitiveIt->wi5 << " " << metaPrimitiveIt->tdc5 << endl;   
      if (printHits && metaPrimitiveIt->wi6!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 2 << " " << 1 << " " << metaPrimitiveIt->wi6 << " " << metaPrimitiveIt->tdc6 << endl;   
      if (printHits && metaPrimitiveIt->wi7!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 2 << " " << 2 << " " << metaPrimitiveIt->wi7 << " " << metaPrimitiveIt->tdc7 << endl;   
      if (printHits && metaPrimitiveIt->wi8!=-1) cout << chId.wheel() << " " << chId.sector() << " " << chId.station() << " " << shift << " " << 2 << " " << 3 << " " << metaPrimitiveIt->wi8 << " " << metaPrimitiveIt->tdc8 << endl;   
*/


   //   if (printHits) cout << (*metaPrimitiveIt).wi1 << "," <<(*metaPrimitiveIt).wi2 << "," <<(*metaPrimitiveIt).wi3 << "," <<(*metaPrimitiveIt).wi4 << "," <<(*metaPrimitiveIt).wi5 << "," <<(*metaPrimitiveIt).wi6 << "," <<(*metaPrimitiveIt).wi7 << "," <<(*metaPrimitiveIt).wi8 << "," << (*metaPrimitiveIt).tdc1 << "," <<(*metaPrimitiveIt).tdc2 << "," <<(*metaPrimitiveIt).tdc3 << "," <<(*metaPrimitiveIt).tdc4 << "," <<(*metaPrimitiveIt).tdc5 << "," <<(*metaPrimitiveIt).tdc6 << "," <<(*metaPrimitiveIt).tdc7 << "," <<(*metaPrimitiveIt).tdc8 << "," << shiftinfo[wireId3.rawId()] - shiftinfo[wireId1.rawId()] << "," << wireId1.wheel() << "," << wireId1.sector() << "," << wireId1.station() << endl;
      if (!printHits)   cout << (*metaPrimitiveIt).quality << " " << (*metaPrimitiveIt).x << " " << (*metaPrimitiveIt).tanPhi << " " << (int) (*metaPrimitiveIt).t0  << " "<< (*metaPrimitiveIt).chi2  << " "  << shiftinfo[wireId1.rawId()]<<" " <<  wireId1.wheel() << " " << wireId1.sector() << " " << wireId1.station() << " " << (*metaPrimitiveIt).wi1<< " " << (*metaPrimitiveIt).wi2<< " " << (*metaPrimitiveIt).wi3<< " " << (*metaPrimitiveIt).wi4<< " " << (*metaPrimitiveIt).wi5<< " " << (*metaPrimitiveIt).wi6<< " " << (*metaPrimitiveIt).wi7<< " " << (*metaPrimitiveIt).wi8<< " " << (*metaPrimitiveIt).tdc1<< " " << (*metaPrimitiveIt).tdc2<< " " << (*metaPrimitiveIt).tdc3<< " " << (*metaPrimitiveIt).tdc4<< " " << (*metaPrimitiveIt).tdc5<< " " << (*metaPrimitiveIt).tdc6<< " " << (*metaPrimitiveIt).tdc7<< " " << (*metaPrimitiveIt).tdc8<< " " << (*metaPrimitiveIt).lat1<< " " << (*metaPrimitiveIt).lat2<< " " << (*metaPrimitiveIt).lat3<< " " << (*metaPrimitiveIt).lat4<< " " << (*metaPrimitiveIt).lat5<< " " << (*metaPrimitiveIt).lat6<< " " << (*metaPrimitiveIt).lat7<< " " << (*metaPrimitiveIt).lat8  << " " << eventBX << endl;
    }
    //if (printHits) cout << "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" << endl;
    //if (printHits) cout << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
    if (!printHits) cout << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1<< " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
  }

    for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end(); ++metaPrimitiveIt){

      DTChamberId chId((*metaPrimitiveIt).rawId);
      if(debug) std::cout<<"looping in final vector: SuperLayerId"<<chId<<" x="<<(*metaPrimitiveIt).x<<" quality="<<(*metaPrimitiveIt).quality << " BX="<< round((*metaPrimitiveIt).t0 / 25.) << " index=" << (*metaPrimitiveIt).index <<std::endl;
      
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
                              (*metaPrimitiveIt).rpcFlag    // urpc (m_rpcFlag)
                              ));
      }
    }

    // Storing RPC hits that were not used elsewhere
    if(p2_df == 2 && useRPC) {
      for (auto rpc_dt_digi = rpc_integrator->rpcRecHits_translated.begin(); rpc_dt_digi != rpc_integrator->rpcRecHits_translated.end(); rpc_dt_digi++) {
        outP2Ph.push_back(*rpc_dt_digi);
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
  rpc_integrator->finish();
};


bool DTTrigPhase2Prod::outer(metaPrimitive mp){
  int counter = (mp.wi5!=-1)+(mp.wi6!=-1)+(mp.wi7!=-1)+(mp.wi8!=-1);
  if (counter > 2) return true;
  else return false;
}


bool DTTrigPhase2Prod::inner(metaPrimitive mp){
  int counter = (mp.wi1!=-1)+(mp.wi2!=-1)+(mp.wi3!=-1)+(mp.wi4!=-1);
  if (counter > 2) return true;
  else return false;
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

void DTTrigPhase2Prod::printmPC(metaPrimitive mP){
  DTChamberId ChId(mP.rawId);
  std::cout<<ChId<<"\t"
            <<" "<<setw(2)<<left<<mP.wi1
            <<" "<<setw(2)<<left<<mP.wi2
            <<" "<<setw(2)<<left<<mP.wi3
            <<" "<<setw(2)<<left<<mP.wi4
            <<" "<<setw(2)<<left<<mP.wi5
            <<" "<<setw(2)<<left<<mP.wi6
            <<" "<<setw(2)<<left<<mP.wi7
            <<" "<<setw(2)<<left<<mP.wi8
            <<" "<<setw(5)<<left<<mP.tdc1
            <<" "<<setw(5)<<left<<mP.tdc2
            <<" "<<setw(5)<<left<<mP.tdc3
            <<" "<<setw(5)<<left<<mP.tdc4
            <<" "<<setw(5)<<left<<mP.tdc5
            <<" "<<setw(5)<<left<<mP.tdc6
            <<" "<<setw(5)<<left<<mP.tdc7
            <<" "<<setw(5)<<left<<mP.tdc8
            <<" "<<setw(2)<<left<<mP.lat1
            <<" "<<setw(2)<<left<<mP.lat2
            <<" "<<setw(2)<<left<<mP.lat3
            <<" "<<setw(2)<<left<<mP.lat4
            <<" "<<setw(2)<<left<<mP.lat5
            <<" "<<setw(2)<<left<<mP.lat6
            <<" "<<setw(2)<<left<<mP.lat7
            <<" "<<setw(2)<<left<<mP.lat8
            <<" "<<setw(10)<<right<<mP.x
            <<" "<<setw(9)<<left<<mP.tanPhi
            <<" "<<setw(5)<<left<<mP.t0
            <<" "<<setw(13)<<left<<mP.chi2
            <<" r:"<<rango(mP);
}


int DTTrigPhase2Prod::rango(metaPrimitive mp) {
  if(mp.quality==1 or mp.quality==2) return 3;
  if(mp.quality==3 or mp.quality==4) return 4;
  return mp.quality;
}


void  DTTrigPhase2Prod::assignIndex(std::vector<metaPrimitive> &inMPaths)
{
  std::map <int, std::vector<metaPrimitive> >  primsPerBX;
  for (auto & metaPrimitive : inMPaths){
    int BX = round(metaPrimitive.t0 / 25.);
    primsPerBX[BX].push_back(metaPrimitive);
  }
  inMPaths.clear();
  for (auto & prims : primsPerBX) {
    assignIndexPerBX (prims.second);
    for (auto & primitive : prims.second ) inMPaths.push_back(primitive);
  }
}


void  DTTrigPhase2Prod::assignIndexPerBX(std::vector<metaPrimitive> &inMPaths)
{
  // First we asociate a new index to the metaprimitive depending on quality or phiB;
  uint32_t rawId = -1;
  int numP = -1;
  for (auto metaPrimitiveIt = inMPaths.begin(); metaPrimitiveIt != inMPaths.end(); ++metaPrimitiveIt){
    numP++;
    rawId = (*metaPrimitiveIt).rawId;
    int iOrder = assignQualityOrder((*metaPrimitiveIt));
    int inf = 0;
    int numP2 = -1;
    for (auto metaPrimitiveItN = inMPaths.begin(); metaPrimitiveItN != inMPaths.end(); ++metaPrimitiveItN){
      int nOrder = assignQualityOrder((*metaPrimitiveItN));
      numP2++;
      if (rawId != (*metaPrimitiveItN).rawId) continue;
      if (numP2 == numP) {
        (*metaPrimitiveIt).index = inf;
        break;
      } else if (iOrder < nOrder) {
        inf++;
      } else if (iOrder > nOrder) {
        (*metaPrimitiveItN).index++;
      } else if (iOrder == nOrder) {
        if (fabs((*metaPrimitiveIt).phiB) >= fabs((*metaPrimitiveItN).phiB) ){
          inf++;
        } else if (fabs((*metaPrimitiveIt).phiB) < fabs((*metaPrimitiveItN).phiB) ){
          (*metaPrimitiveItN).index++;
        }
      }
    } // ending second for
  } // ending first for
}


int DTTrigPhase2Prod::assignQualityOrder(metaPrimitive mP) {
  if (mP.quality == 9) return 9;
  if (mP.quality == 8) return 8;
  if (mP.quality == 7) return 6;
  if (mP.quality == 6) return 7;
  if (mP.quality == 5) return 3;
  if (mP.quality == 4) return 5;
  if (mP.quality == 3) return 4;
  if (mP.quality == 2) return 2;
  if (mP.quality == 1) return 1;
  return -1;
}


std::vector<DTDigiCollection*> DTTrigPhase2Prod::distribDigis(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ) {
//   cout << "Declarando..." << endl;
  std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*> tmpVector; tmpVector.clear();
  std::vector<DTDigiCollection*> collVector; collVector.clear();
//   cout << "Empezando while..." << endl;
  while (!inQ.empty()) {
    // Possible enhancement: build a supercell class that automatically encloses this code, i.e. that comprises
    // a entire supercell within it.
//     cout << "Llamando a processDigi..." << endl;
    processDigi(inQ, tmpVector);
  }
//   cout << "Terminado while" << endl;

//   cout << "Comenzando for del vector..." << endl;
  for (auto & sQ : tmpVector) {
    DTDigiCollection* tmpColl = new DTDigiCollection();
    while (!sQ->empty()) {
      tmpColl->insertDigi(*(sQ->front().first), *(sQ->front().second));
      sQ->pop();
    }
    collVector.push_back(std::move(tmpColl));
  }
  return collVector;
}


void DTTrigPhase2Prod::processDigi(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ, std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*>& vec) {
  Bool_t classified = false;
  if (vec.size() != 0) {
    for (auto & sC : vec) { // Conditions for entering a super cell.
      if ((sC->front().second->time() + superCelltimewidth) > inQ.front().second->time()) { // Time requirement
        if (TMath::Abs(sC->front().second->wire() - inQ.front().second->wire()) <= superCellhalfspacewidth) { // Spatial requirement
          sC->push(std::move(inQ.front()));
          classified = true;
        }
      }
    }
  }
  if (classified) {
    inQ.pop();
    return;
  }
//   cout << "El tamaño del vector es nulo, o no hemos podido meter al digi en ninguna cola. Declarando nueva cola..." << endl;
  std::queue<std::pair<DTLayerId*, DTDigi*>>* newQueue = new std::queue<std::pair<DTLayerId*, DTDigi*>>;
//   cout << "Introduciendo digi..." << endl;
  std::pair<DTLayerId*, DTDigi*>* tmpPair = new std::pair<DTLayerId*, DTDigi*>;
  tmpPair =  std::move(&inQ.front());
  newQueue->push(*tmpPair);
  inQ.pop();
//   cout << "Introduciendo cola nel vector..." << endl;
  vec.push_back(std::move(newQueue));
  return;
}
