#include "L1Trigger/DTTriggerPhase2/interface/DTTrigPhase2Prod.h"

using namespace edm;
using namespace std;

typedef vector<DTSectCollPhSegm> SectCollPhiColl;
typedef SectCollPhiColl::const_iterator SectCollPhiColl_iterator;
typedef vector<DTSectCollThSegm> SectCollThetaColl;
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

  debug_ = pset.getUntrackedParameter<bool>("debug");
  dump_ = pset.getUntrackedParameter<bool>("dump");

  do_correlation_ = pset.getUntrackedParameter<bool>("do_correlation");

  scenario_ = pset.getUntrackedParameter<int>("scenario");
  dtDigisToken_ = consumes<DTDigiCollection>(pset.getParameter<edm::InputTag>("digiTag"));

  rpcRecHitsLabel_ = consumes<RPCRecHitCollection>(pset.getUntrackedParameter<edm::InputTag>("rpcRecHits"));
  useRPC_ = pset.getUntrackedParameter<bool>("useRPC");

  // Choosing grouping scheme:
  grcode_ = pset.getUntrackedParameter<int>("grouping_code");

  if (grcode_ == 0)
    grouping_obj_ = new InitialGrouping(pset);
  else if (grcode_ == 1)
    grouping_obj_ = new HoughGrouping(pset.getParameter<edm::ParameterSet>("HoughGrouping"));
  else if (grcode_ == 2)
    grouping_obj_ = new PseudoBayesGrouping(pset.getParameter<edm::ParameterSet>("PseudoBayesPattern"));
  else {
    if (debug_)
      cout << "DTp2::constructor: Non-valid grouping code. Choosing InitialGrouping by default." << endl;
    grouping_obj_ = new InitialGrouping(pset);
  }

  if (grcode_ == 0) {
    if (debug_)
      cout << "DTp2:constructor: JM analyzer" << endl;
    mpathanalyzer_ = new MuonPathAnalyzerPerSL(pset);
  } else {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "               WARNING!!!!!                   " << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << " This grouping option is not fully supported  " << endl;
    cout << " yet.                                         " << endl;
    cout << " USE IT AT YOUR OWN RISK!                     " << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    if (debug_)
      cout << "DTp2:constructor: Full chamber analyzer" << endl;
    mpathanalyzer_ = new MuonPathAnalyzerInChamber(pset);
  }

  // Getting buffer option
  activateBuffer_ = pset.getUntrackedParameter<Bool_t>("activateBuffer");
  superCellhalfspacewidth_ = pset.getUntrackedParameter<Int_t>("superCellspacewidth") / 2;
  superCelltimewidth_ = pset.getUntrackedParameter<Double_t>("superCelltimewidth");

  mpathqualityenhancer_ = new MPQualityEnhancerFilter(pset);
  mpathredundantfilter_ = new MPRedundantFilter(pset);
  mpathassociator_ = new MuonPathAssociator(pset);
  rpc_integrator_ = new RPCIntegrator(pset);
}

DTTrigPhase2Prod::~DTTrigPhase2Prod() {
  //delete inMuonPath;
  //delete outValidMuonPath;

  if (debug_)
    std::cout << "DTp2: calling destructor" << std::endl;

  delete grouping_obj_;          // Grouping destructor
  delete mpathanalyzer_;         // Analyzer destructor
  delete mpathqualityenhancer_;  // Filter destructor
  delete mpathredundantfilter_;  // Filter destructor
  delete mpathassociator_;       // Associator destructor
  delete rpc_integrator_;
}

void DTTrigPhase2Prod::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (debug_)
    cout << "DTTrigPhase2Prod::beginRun " << iRun.id().run() << endl;
  if (debug_)
    cout << "DTTrigPhase2Prod::beginRun: getting DT geometry" << endl;

  if (debug_)
    std::cout << "getting DT geometry" << std::endl;
  iEventSetup.get<MuonGeometryRecord>().get(dtGeo_);  //1103

  ESHandle<DTConfigManager> dtConfig;
  iEventSetup.get<DTConfigManagerRcd>().get(dtConfig);

  grouping_obj_->initialise(iEventSetup);          // Grouping object initialisation
  mpathanalyzer_->initialise(iEventSetup);         // Analyzer object initialisation
  mpathqualityenhancer_->initialise(iEventSetup);  // Filter object initialisation
  mpathredundantfilter_->initialise(iEventSetup);  // Filter object initialisation
  mpathassociator_->initialise(iEventSetup);       // Associator object initialisation
}

void DTTrigPhase2Prod::produce(Event& iEvent, const EventSetup& iEventSetup) {
  if (debug_)
    cout << "DTTrigPhase2Prod::produce" << endl;
  edm::Handle<DTDigiCollection> dtdigis;
  iEvent.getByToken(dtDigisToken_, dtdigis);

  if (debug_)
    std::cout << "\t Getting the RPC RecHits" << std::endl;
  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByToken(rpcRecHitsLabel_, rpcRecHits);

  ////////////////////////////////
  // GROUPING CODE:
  ////////////////////////////////
  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = dtdigis->begin(); detUnitIt != dtdigis->end(); ++detUnitIt) {
    const DTLayerId& layId = (*detUnitIt).first;
    const DTChamberId chambId = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    digiMap[chambId].put(range, layId);
  }

  // generate a list muon paths for each event!!!
  if (debug_ && activateBuffer_)
    cout << "DTTrigPhase2Prod::produce - Getting and grouping digis per chamber using a buffer and super cells."
         << endl;
  else if (debug_)
    cout << "DTTrigPhase2Prod::produce - Getting and grouping digis per chamber." << endl;
  std::vector<MuonPath*> muonpaths;
  for (std::vector<const DTChamber*>::const_iterator ich = dtGeo_->chambers().begin(); ich != dtGeo_->chambers().end();
       ich++) {
    // The code inside this for loop would ideally later fit inside a trigger unit (in principle, a DT station) of the future Phase 2 DT Trigger.
    const DTChamber* chamb = (*ich);
    DTChamberId chid = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);

    if (dmit == digiMap.end())
      continue;

    if (activateBuffer_) {  // Use buffering (per chamber) or not
      // Import digis from the station
      std::vector<std::pair<DTLayerId*, DTDigi*>> tmpvec;
      tmpvec.clear();

      for (DTDigiCollection::DigiRangeIterator dtLayerIdIt = (*dmit).second.begin();
           dtLayerIdIt != (*dmit).second.end();
           dtLayerIdIt++) {
        for (DTDigiCollection::const_iterator digiIt = ((*dtLayerIdIt).second).first;
             digiIt != ((*dtLayerIdIt).second).second;
             digiIt++) {
          DTLayerId* tmplayer = new DTLayerId((*dtLayerIdIt).first);
          DTDigi* tmpdigi = new DTDigi((*digiIt));
          tmpvec.push_back({tmplayer, tmpdigi});
        }
      }

      // Check to enhance CPU time usage
      if (tmpvec.size() == 0)
        continue;

      // Order digis depending on TDC time and insert them into a queue (FIFO buffer). TODO: adapt for MC simulations.
      std::sort(tmpvec.begin(), tmpvec.end(), DigiTimeOrdering);
      std::queue<std::pair<DTLayerId*, DTDigi*>> timequeue;

      for (auto& elem : tmpvec)
        timequeue.push(std::move(elem));
      tmpvec.clear();

      // Distribute the digis from the queue into supercells
      std::vector<DTDigiCollection*> superCells;
      superCells = distribDigis(timequeue);

      // Process each supercell & collect the resulting muonpaths (as the muonpaths std::vector is only enlarged each time
      // the groupings access it, it's not needed to "collect" the final products).
      while (!superCells.empty()) {
        grouping_obj_->run(iEvent, iEventSetup, *(superCells.back()), &muonpaths);
        superCells.pop_back();
      }
    } else {
      grouping_obj_->run(iEvent, iEventSetup, (*dmit).second, &muonpaths);
    }
  }
  digiMap.clear();

  if (dump_) {
    for (unsigned int i = 0; i < muonpaths.size(); i++) {
      cout << iEvent.id().event() << "      mpath " << i << ": ";
      for (int lay = 0; lay < muonpaths.at(i)->nprimitives(); lay++)
        cout << muonpaths.at(i)->primitive(lay)->channelId() << " ";
      for (int lay = 0; lay < muonpaths.at(i)->nprimitives(); lay++)
        cout << muonpaths.at(i)->primitive(lay)->tdcTimeStamp() << " ";
      for (int lay = 0; lay < muonpaths.at(i)->nprimitives(); lay++)
        cout << muonpaths.at(i)->primitive(lay)->laterality() << " ";
      cout << endl;
    }
    cout << endl;
  }

  // FILTER GROUPING
  std::vector<MuonPath*> filteredmuonpaths;
  if (grcode_ == 0) {
    mpathredundantfilter_->run(iEvent, iEventSetup, muonpaths, filteredmuonpaths);
  }

  if (dump_) {
    for (unsigned int i = 0; i < filteredmuonpaths.size(); i++) {
      cout << iEvent.id().event() << " filt. mpath " << i << ": ";
      for (int lay = 0; lay < filteredmuonpaths.at(i)->nprimitives(); lay++)
        cout << filteredmuonpaths.at(i)->primitive(lay)->channelId() << " ";
      for (int lay = 0; lay < filteredmuonpaths.at(i)->nprimitives(); lay++)
        cout << filteredmuonpaths.at(i)->primitive(lay)->tdcTimeStamp() << " ";
      cout << endl;
    }
    cout << endl;
  }

  ///////////////////////////////////////////
  /// FITTING SECTION;
  ///////////////////////////////////////////
  if (debug_)
    cout << "MUON PATHS found: " << muonpaths.size() << " (" << filteredmuonpaths.size() << ") in event "
         << iEvent.id().event() << endl;
  if (debug_)
    std::cout << "filling NmetaPrimtives" << std::endl;
  std::vector<metaPrimitive> metaPrimitives;
  std::vector<MuonPath*> outmpaths;
  if (grcode_ == 0) {
    if (debug_)
      cout << "Fitting 1SL " << endl;
    mpathanalyzer_->run(iEvent, iEventSetup, filteredmuonpaths, metaPrimitives);
  } else {
    // implementation for advanced (2SL) grouping, no filter required..
    if (debug_)
      cout << "Fitting 2SL at once " << endl;
    mpathanalyzer_->run(iEvent, iEventSetup, muonpaths, outmpaths);
  }

  if (dump_) {
    for (unsigned int i = 0; i < outmpaths.size(); i++) {
      cout << iEvent.id().event() << " mp " << i << ": " << outmpaths.at(i)->bxTimeValue() << " "
           << outmpaths.at(i)->horizPos() << " " << outmpaths.at(i)->tanPhi() << " " << outmpaths.at(i)->phi() << " "
           << outmpaths.at(i)->phiB() << " " << outmpaths.at(i)->quality() << " " << outmpaths.at(i)->chiSquare() << " "
           << endl;
    }
    for (unsigned int i = 0; i < metaPrimitives.size(); i++) {
      cout << iEvent.id().event() << " mp " << i << ": ";
      printmP(metaPrimitives.at(i));
      cout << endl;
    }
  }

  if (debug_)
    std::cout << "deleting muonpaths" << std::endl;
  for (unsigned int i = 0; i < muonpaths.size(); i++) {
    delete muonpaths[i];
  }
  muonpaths.clear();

  for (unsigned int i = 0; i < filteredmuonpaths.size(); i++) {
    delete filteredmuonpaths[i];
  }
  filteredmuonpaths.clear();

  /////////////////////////////////////
  //  FILTER SECTIONS:
  ////////////////////////////////////
  //filtro de duplicados puro popdr'ia ir ac'a mpredundantfilter.cpp primos?
  //filtro en |tanPhi|<~1.?

  if (debug_)
    std::cout << "declaring new vector for filtered" << std::endl;

  std::vector<metaPrimitive> filteredMetaPrimitives;
  if (grcode_ == 0)
    mpathqualityenhancer_->run(iEvent, iEventSetup, metaPrimitives, filteredMetaPrimitives);

  if (dump_) {
    for (unsigned int i = 0; i < filteredMetaPrimitives.size(); i++) {
      cout << iEvent.id().event() << " filtered mp " << i << ": ";
      printmP(filteredMetaPrimitives.at(i));
      cout << endl;
    }
  }

  metaPrimitives.clear();
  metaPrimitives.erase(metaPrimitives.begin(), metaPrimitives.end());

  if (debug_)
    std::cout << "DTp2 in event:" << iEvent.id().event() << " we found " << filteredMetaPrimitives.size()
              << " filteredMetaPrimitives (superlayer)" << std::endl;
  if (debug_)
    std::cout << "filteredMetaPrimitives: starting correlations" << std::endl;

  /////////////////////////////////////
  //// CORRELATION:
  /////////////////////////////////////
  std::vector<metaPrimitive> correlatedMetaPrimitives;
  if (grcode_ == 0)
    mpathassociator_->run(iEvent, iEventSetup, dtdigis, filteredMetaPrimitives, correlatedMetaPrimitives);
  else {
    //for(auto muonpath = muonpaths.begin();muonpath!=muonpaths.end();++muonpath) {
    for (auto muonpath = outmpaths.begin(); muonpath != outmpaths.end(); ++muonpath) {
      correlatedMetaPrimitives.push_back(metaPrimitive({
          (*muonpath)->rawId(),
          (double)(*muonpath)->bxTimeValue(),
          (*muonpath)->horizPos(),
          (*muonpath)->tanPhi(),
          (*muonpath)->phi(),
          (*muonpath)->phiB(),
          (*muonpath)->chiSquare(),
          (int)(*muonpath)->quality(),
          (*muonpath)->primitive(0)->channelId(),
          (*muonpath)->primitive(0)->tdcTimeStamp(),
          (*muonpath)->primitive(0)->laterality(),
          (*muonpath)->primitive(1)->channelId(),
          (*muonpath)->primitive(1)->tdcTimeStamp(),
          (*muonpath)->primitive(1)->laterality(),
          (*muonpath)->primitive(2)->channelId(),
          (*muonpath)->primitive(2)->tdcTimeStamp(),
          (*muonpath)->primitive(2)->laterality(),
          (*muonpath)->primitive(3)->channelId(),
          (*muonpath)->primitive(3)->tdcTimeStamp(),
          (*muonpath)->primitive(3)->laterality(),
          (*muonpath)->primitive(4)->channelId(),
          (*muonpath)->primitive(4)->tdcTimeStamp(),
          (*muonpath)->primitive(4)->laterality(),
          (*muonpath)->primitive(5)->channelId(),
          (*muonpath)->primitive(5)->tdcTimeStamp(),
          (*muonpath)->primitive(5)->laterality(),
          (*muonpath)->primitive(6)->channelId(),
          (*muonpath)->primitive(6)->tdcTimeStamp(),
          (*muonpath)->primitive(6)->laterality(),
          (*muonpath)->primitive(7)->channelId(),
          (*muonpath)->primitive(7)->tdcTimeStamp(),
          (*muonpath)->primitive(7)->laterality(),
      }));
    }
  }
  filteredMetaPrimitives.clear();
  filteredMetaPrimitives.erase(filteredMetaPrimitives.begin(), filteredMetaPrimitives.end());

  if (debug_)
    std::cout << "DTp2 in event:" << iEvent.id().event() << " we found " << correlatedMetaPrimitives.size()
              << " correlatedMetPrimitives (chamber)" << std::endl;

  if (dump_) {
    for (unsigned int i = 0; i < correlatedMetaPrimitives.size(); i++) {
      cout << iEvent.id().event() << " correlated mp " << i << ": ";
      printmPC(correlatedMetaPrimitives.at(i));
      cout << endl;
    }
  }

  double shift_back = 0;

  //if (iEvent.eventAuxiliary().run() == 1) //FIX MC
  if (scenario_ == 0)  //scope for MC
    shift_back = 400;

  if (scenario_ == 1)  //scope for data
    shift_back = 0;

  if (scenario_ == 2)  //scope for slice test
    shift_back = 0;

  // RPC integration
  if (useRPC_) {
    if (debug_)
      std::cout << "Start integrating RPC" << std::endl;
    rpc_integrator_->initialise(iEventSetup, shift_back);
    rpc_integrator_->prepareMetaPrimitives(rpcRecHits);
    rpc_integrator_->matchWithDTAndUseRPCTime(correlatedMetaPrimitives);
    rpc_integrator_->makeRPCOnlySegments();
    rpc_integrator_->storeRPCSingleHits();
    rpc_integrator_->removeRPCHitsUsed();
  }

  /// STORING RESULTs

  vector<L1Phase2MuDTPhDigi> outP2Ph;

  // Assigning index value
  assignIndex(correlatedMetaPrimitives);
  for (auto metaPrimitiveIt = correlatedMetaPrimitives.begin(); metaPrimitiveIt != correlatedMetaPrimitives.end();
       ++metaPrimitiveIt) {
    DTChamberId chId((*metaPrimitiveIt).rawId);
    if (debug_)
      std::cout << "looping in final vector: SuperLayerId" << chId << " x=" << (*metaPrimitiveIt).x
                << " quality=" << (*metaPrimitiveIt).quality << " BX=" << round((*metaPrimitiveIt).t0 / 25.)
                << " index=" << (*metaPrimitiveIt).index << std::endl;

    int sectorTP = chId.sector();
    if (sectorTP == 13)
      sectorTP = 4;
    if (sectorTP == 14)
      sectorTP = 10;
    sectorTP = sectorTP - 1;
    int sl = 0;
    if ((*metaPrimitiveIt).quality < 6 || (*metaPrimitiveIt).quality == 7) {
      if (inner((*metaPrimitiveIt)))
        sl = 1;
      else
        sl = 3;
    }

    if (debug_)
      std::cout << "pushing back phase-2 dataformat carlo-federica dataformat" << std::endl;
    outP2Ph.push_back(L1Phase2MuDTPhDigi(
        (int)round((*metaPrimitiveIt).t0 / 25.) - shift_back,  // ubx (m_bx) //bx en la orbita
        chId.wheel(),    // uwh (m_wheel)     // FIXME: It is not clear who provides this?
        sectorTP,        // usc (m_sector)    // FIXME: It is not clear who provides this?
        chId.station(),  // ust (m_station)
        sl,              // ust (m_station)
        (int)round((*metaPrimitiveIt).phi * 65536. / 0.8),    // uphi (_phiAngle)
        (int)round((*metaPrimitiveIt).phiB * 2048. / 1.4),    // uphib (m_phiBending)
        (*metaPrimitiveIt).quality,                           // uqua (m_qualityCode)
        (*metaPrimitiveIt).index,                             // uind (m_segmentIndex)
        (int)round((*metaPrimitiveIt).t0) - shift_back * 25,  // ut0 (m_t0Segment)
        (int)round((*metaPrimitiveIt).chi2 * 1000000),        // uchi2 (m_chi2Segment)
        (*metaPrimitiveIt).rpcFlag                            // urpc (m_rpcFlag)
        ));
  }

  // Storing RPC hits that were not used elsewhere
  if (useRPC_) {
    for (auto rpc_dt_digi = rpc_integrator_->rpcRecHits_translated_.begin();
         rpc_dt_digi != rpc_integrator_->rpcRecHits_translated_.end();
         rpc_dt_digi++) {
      outP2Ph.push_back(*rpc_dt_digi);
    }
  }

  std::unique_ptr<L1Phase2MuDTPhContainer> resultP2Ph(new L1Phase2MuDTPhContainer);
  resultP2Ph->setContainer(outP2Ph);
  iEvent.put(std::move(resultP2Ph));
  outP2Ph.clear();
  outP2Ph.erase(outP2Ph.begin(), outP2Ph.end());
}

void DTTrigPhase2Prod::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  grouping_obj_->finish();
  mpathanalyzer_->finish();
  mpathqualityenhancer_->finish();
  mpathredundantfilter_->finish();
  mpathassociator_->finish();
  rpc_integrator_->finish();
};

bool DTTrigPhase2Prod::outer(metaPrimitive mp) {
  int counter = (mp.wi5 != -1) + (mp.wi6 != -1) + (mp.wi7 != -1) + (mp.wi8 != -1);
  if (counter > 2)
    return true;
  else
    return false;
}

bool DTTrigPhase2Prod::inner(metaPrimitive mp) {
  int counter = (mp.wi1 != -1) + (mp.wi2 != -1) + (mp.wi3 != -1) + (mp.wi4 != -1);
  if (counter > 2)
    return true;
  else
    return false;
}

bool DTTrigPhase2Prod::hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); }

void DTTrigPhase2Prod::printmP(metaPrimitive mP) {
  DTSuperLayerId slId(mP.rawId);
  std::cout << slId << "\t"
            << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2) << left << mP.wi3
            << " " << setw(2) << left << mP.wi4 << " " << setw(5) << left << mP.tdc1 << " " << setw(5) << left
            << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " " << setw(10)
            << right << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " "
            << setw(13) << left << mP.chi2 << " r:" << rango(mP);
}

void DTTrigPhase2Prod::printmPC(metaPrimitive mP) {
  DTChamberId ChId(mP.rawId);
  std::cout << ChId << "\t"
            << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2) << left << mP.wi3
            << " " << setw(2) << left << mP.wi4 << " " << setw(2) << left << mP.wi5 << " " << setw(2) << left << mP.wi6
            << " " << setw(2) << left << mP.wi7 << " " << setw(2) << left << mP.wi8 << " " << setw(5) << left << mP.tdc1
            << " " << setw(5) << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5) << left
            << mP.tdc4 << " " << setw(5) << left << mP.tdc5 << " " << setw(5) << left << mP.tdc6 << " " << setw(5)
            << left << mP.tdc7 << " " << setw(5) << left << mP.tdc8 << " " << setw(2) << left << mP.lat1 << " "
            << setw(2) << left << mP.lat2 << " " << setw(2) << left << mP.lat3 << " " << setw(2) << left << mP.lat4
            << " " << setw(2) << left << mP.lat5 << " " << setw(2) << left << mP.lat6 << " " << setw(2) << left
            << mP.lat7 << " " << setw(2) << left << mP.lat8 << " " << setw(10) << right << mP.x << " " << setw(9)
            << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " " << setw(13) << left << mP.chi2
            << " r:" << rango(mP);
}

int DTTrigPhase2Prod::rango(metaPrimitive mp) {
  if (mp.quality == 1 or mp.quality == 2)
    return 3;
  if (mp.quality == 3 or mp.quality == 4)
    return 4;
  return mp.quality;
}

void DTTrigPhase2Prod::assignIndex(std::vector<metaPrimitive>& inMPaths) {
  std::map<int, std::vector<metaPrimitive>> primsPerBX;
  for (auto& metaPrimitive : inMPaths) {
    int BX = round(metaPrimitive.t0 / 25.);
    primsPerBX[BX].push_back(metaPrimitive);
  }
  inMPaths.clear();
  for (auto& prims : primsPerBX) {
    assignIndexPerBX(prims.second);
    for (auto& primitive : prims.second)
      inMPaths.push_back(primitive);
  }
}

void DTTrigPhase2Prod::assignIndexPerBX(std::vector<metaPrimitive>& inMPaths) {
  // First we asociate a new index to the metaprimitive depending on quality or phiB;
  uint32_t rawId = -1;
  int numP = -1;
  for (auto metaPrimitiveIt = inMPaths.begin(); metaPrimitiveIt != inMPaths.end(); ++metaPrimitiveIt) {
    numP++;
    rawId = (*metaPrimitiveIt).rawId;
    int iOrder = assignQualityOrder((*metaPrimitiveIt));
    int inf = 0;
    int numP2 = -1;
    for (auto metaPrimitiveItN = inMPaths.begin(); metaPrimitiveItN != inMPaths.end(); ++metaPrimitiveItN) {
      int nOrder = assignQualityOrder((*metaPrimitiveItN));
      numP2++;
      if (rawId != (*metaPrimitiveItN).rawId)
        continue;
      if (numP2 == numP) {
        (*metaPrimitiveIt).index = inf;
        break;
      } else if (iOrder < nOrder) {
        inf++;
      } else if (iOrder > nOrder) {
        (*metaPrimitiveItN).index++;
      } else if (iOrder == nOrder) {
        if (fabs((*metaPrimitiveIt).phiB) >= fabs((*metaPrimitiveItN).phiB)) {
          inf++;
        } else if (fabs((*metaPrimitiveIt).phiB) < fabs((*metaPrimitiveItN).phiB)) {
          (*metaPrimitiveItN).index++;
        }
      }
    }  // ending second for
  }    // ending first for
}

int DTTrigPhase2Prod::assignQualityOrder(metaPrimitive mP) {
  if (mP.quality == 9)
    return 9;
  if (mP.quality == 8)
    return 8;
  if (mP.quality == 7)
    return 6;
  if (mP.quality == 6)
    return 7;
  if (mP.quality == 5)
    return 3;
  if (mP.quality == 4)
    return 5;
  if (mP.quality == 3)
    return 4;
  if (mP.quality == 2)
    return 2;
  if (mP.quality == 1)
    return 1;
  return -1;
}

std::vector<DTDigiCollection*> DTTrigPhase2Prod::distribDigis(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ) {
  //   cout << "Declarando..." << endl;
  std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*> tmpVector;
  tmpVector.clear();
  std::vector<DTDigiCollection*> collVector;
  collVector.clear();
  //   cout << "Empezando while..." << endl;
  while (!inQ.empty()) {
    // Possible enhancement: build a supercell class that automatically encloses this code, i.e. that comprises
    // a entire supercell within it.
    //     cout << "Llamando a processDigi..." << endl;
    processDigi(inQ, tmpVector);
  }
  //   cout << "Terminado while" << endl;

  //   cout << "Comenzando for del vector..." << endl;
  for (auto& sQ : tmpVector) {
    DTDigiCollection* tmpColl = new DTDigiCollection();
    while (!sQ->empty()) {
      tmpColl->insertDigi(*(sQ->front().first), *(sQ->front().second));
      sQ->pop();
    }
    collVector.push_back(std::move(tmpColl));
  }
  return collVector;
}

void DTTrigPhase2Prod::processDigi(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ,
                                   std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*>& vec) {
  Bool_t classified = false;
  if (vec.size() != 0) {
    for (auto& sC : vec) {  // Conditions for entering a super cell.
      if ((sC->front().second->time() + superCelltimewidth_) > inQ.front().second->time()) {  // Time requirement
        if (TMath::Abs(sC->front().second->wire() - inQ.front().second->wire()) <=
            superCellhalfspacewidth_) {  // Spatial requirement
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
  tmpPair = std::move(&inQ.front());
  newQueue->push(*tmpPair);
  inQ.pop();
  //   cout << "Introduciendo cola nel vector..." << endl;
  vec.push_back(std::move(newQueue));
  return;
}
