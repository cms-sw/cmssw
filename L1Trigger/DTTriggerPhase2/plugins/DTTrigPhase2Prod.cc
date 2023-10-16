#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/TrapezoidalGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/HoughGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/PseudoBayesGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/LateralityProvider.h"
#include "L1Trigger/DTTriggerPhase2/interface/LateralityBasicProvider.h"
#include "L1Trigger/DTTriggerPhase2/interface/LateralityCoarsedProvider.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathSLFitter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathCorFitter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyticAnalyzer.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerInChamber.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAssociator.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathConfirmator.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPSLFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPCorFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPRedundantFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPCleanHitsFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilterBayes.h"
#include "L1Trigger/DTTriggerPhase2/interface/GlobalCoordsObtainer.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThDigi.h"

// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

//RPC TP
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/DTTriggerPhase2/interface/RPCIntegrator.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <cmath>

using namespace edm;
using namespace std;
using namespace cmsdt;

class DTTrigPhase2Prod : public edm::stream::EDProducer<> {
  typedef std::map<DTChamberId, DTDigiCollection, std::less<DTChamberId>> DTDigiMap;
  typedef DTDigiMap::iterator DTDigiMap_iterator;
  typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

public:
  //! Constructor
  DTTrigPhase2Prod(const edm::ParameterSet& pset);

  //! Destructor
  ~DTTrigPhase2Prod() override;

  //! Create Trigger Units before starting event processing
  void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  //! Producer: process every event and generates trigger data
  void produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) override;

  //! endRun: finish things
  void endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  // Methods
  int rango(const metaPrimitive& mp) const;
  bool outer(const metaPrimitive& mp) const;
  bool inner(const metaPrimitive& mp) const;
  void printmP(const std::string& ss, const metaPrimitive& mP) const;
  void printmP(const metaPrimitive& mP) const;
  void printmPC(const std::string& ss, const metaPrimitive& mP) const;
  void printmPC(const metaPrimitive& mP) const;
  bool hasPosRF(int wh, int sec) const;

  // Getter-methods
  MP_QUALITY getMinimumQuality(void);

  // Setter-methods
  void setChiSquareThreshold(float ch2Thr);
  void setMinimumQuality(MP_QUALITY q);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // data-members
  const DTGeometry* dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;
  std::vector<std::pair<int, MuonPath>> primitives_;

private:
  // Trigger Configuration Manager CCB validity flag
  bool my_CCBValid_;

  // BX offset used to correct DTTPG output
  int my_BXoffset_;

  // Debug Flag
  bool debug_;
  bool dump_;
  double dT0_correlate_TP_;
  int scenario_;
  int df_extended_;
  int max_index_;

  bool output_mixer_;
  bool output_latpredictor_;
  bool output_slfitter_;
  bool output_slfilter_;
  bool output_confirmed_;
  bool output_matcher_;
  bool skip_processing_;
  bool allow_confirmation_;

  // ParameterSet
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken_;
  edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsLabel_;

  // Grouping attributes and methods
  int algo_;  // Grouping code
  std::unique_ptr<MotherGrouping> grouping_obj_;
  std::unique_ptr<MuonPathAnalyzer> mpathanalyzer_;
  std::unique_ptr<LateralityProvider> latprovider_;
  std::unique_ptr<MPFilter> mpathqualityenhancer_;
  std::unique_ptr<MPFilter> mpathqualityenhancerbayes_;
  std::unique_ptr<MPFilter> mpathredundantfilter_;
  std::unique_ptr<MPFilter> mpathhitsfilter_;
  std::unique_ptr<MuonPathAnalyzer> mpathassociator_;
  std::unique_ptr<MuonPathConfirmator> mpathconfirmator_;
  std::unique_ptr<MPFilter> mpathcorfilter_;
  std::shared_ptr<GlobalCoordsObtainer> globalcoordsobtainer_;

  // Buffering
  bool activateBuffer_;
  int superCellhalfspacewidth_;
  float superCelltimewidth_;
  std::vector<DTDigiCollection*> distribDigis(std::queue<std::pair<DTLayerId, DTDigi>>& inQ);
  void processDigi(std::queue<std::pair<DTLayerId, DTDigi>>& inQ,
                   std::vector<std::queue<std::pair<DTLayerId, DTDigi>>*>& vec);

  // RPC
  std::unique_ptr<RPCIntegrator> rpc_integrator_;
  bool useRPC_;

  void assignIndex(std::vector<metaPrimitive>& inMPaths);
  void assignIndexPerBX(std::vector<metaPrimitive>& inMPaths);
  int assignQualityOrder(const metaPrimitive& mP) const;

  const std::unordered_map<int, int> qmap_;
};

namespace {
  struct {
    bool operator()(std::pair<DTLayerId, DTDigi> a, std::pair<DTLayerId, DTDigi> b) const {
      return (a.second.time() < b.second.time());
    }
  } const DigiTimeOrdering;
}  // namespace

DTTrigPhase2Prod::DTTrigPhase2Prod(const ParameterSet& pset)
    : qmap_({{8, 8}, {7, 7}, {6, 6}, {4, 4}, {3, 3}, {2, 2}, {1, 1}}) {
  produces<L1Phase2MuDTPhContainer>();
  produces<L1Phase2MuDTThContainer>();
  produces<L1Phase2MuDTExtPhContainer>();
  produces<L1Phase2MuDTExtThContainer>();

  debug_ = pset.getUntrackedParameter<bool>("debug");
  dump_ = pset.getUntrackedParameter<bool>("dump");

  scenario_ = pset.getParameter<int>("scenario");

  df_extended_ = pset.getParameter<int>("df_extended");
  max_index_ = pset.getParameter<int>("max_primitives") - 1;

  dtDigisToken_ = consumes<DTDigiCollection>(pset.getParameter<edm::InputTag>("digiTag"));

  rpcRecHitsLabel_ = consumes<RPCRecHitCollection>(pset.getParameter<edm::InputTag>("rpcRecHits"));
  useRPC_ = pset.getParameter<bool>("useRPC");

  // Choosing grouping scheme:
  algo_ = pset.getParameter<int>("algo");

  // shortcuts

  output_mixer_ = pset.getParameter<bool>("output_mixer");
  output_latpredictor_ = pset.getParameter<bool>("output_latpredictor");
  output_slfitter_ = pset.getParameter<bool>("output_slfitter");
  output_slfilter_ = pset.getParameter<bool>("output_slfilter");
  output_confirmed_ = pset.getParameter<bool>("output_confirmed");
  output_matcher_ = pset.getParameter<bool>("output_matcher");
  allow_confirmation_ = pset.getParameter<bool>("allow_confirmation");

  edm::ConsumesCollector consumesColl(consumesCollector());
  globalcoordsobtainer_ = std::make_shared<GlobalCoordsObtainer>(pset);
  globalcoordsobtainer_->generate_luts();

  if (algo_ == PseudoBayes) {
    grouping_obj_ =
        std::make_unique<PseudoBayesGrouping>(pset.getParameter<edm::ParameterSet>("PseudoBayesPattern"), consumesColl);
  } else if (algo_ == HoughTrans) {
    grouping_obj_ =
        std::make_unique<HoughGrouping>(pset.getParameter<edm::ParameterSet>("HoughGrouping"), consumesColl);
  } else {
    grouping_obj_ = std::make_unique<TrapezoidalGrouping>(pset, consumesColl);
  }

  if (algo_ == Standard) {
    if (debug_)
      LogDebug("DTTrigPhase2Prod") << "DTp2:constructor: JM analyzer";
    mpathanalyzer_ = std::make_unique<MuonPathSLFitter>(pset, consumesColl, globalcoordsobtainer_);
    latprovider_ = std::make_unique<LateralityCoarsedProvider>(pset, consumesColl);
  } else {
    if (debug_)
      LogDebug("DTTrigPhase2Prod") << "DTp2:constructor: Full chamber analyzer";
    mpathanalyzer_ = std::make_unique<MuonPathAnalyzerInChamber>(pset, consumesColl, globalcoordsobtainer_);
  }

  // Getting buffer option
  activateBuffer_ = pset.getParameter<bool>("activateBuffer");
  superCellhalfspacewidth_ = pset.getParameter<int>("superCellspacewidth") / 2;
  superCelltimewidth_ = pset.getParameter<double>("superCelltimewidth");

  mpathqualityenhancer_ = std::make_unique<MPSLFilter>(pset);
  mpathqualityenhancerbayes_ = std::make_unique<MPQualityEnhancerFilterBayes>(pset);
  mpathredundantfilter_ = std::make_unique<MPRedundantFilter>(pset);
  mpathhitsfilter_ = std::make_unique<MPCleanHitsFilter>(pset);
  mpathconfirmator_ = std::make_unique<MuonPathConfirmator>(pset, consumesColl);
  mpathassociator_ = std::make_unique<MuonPathCorFitter>(pset, consumesColl, globalcoordsobtainer_);
  mpathcorfilter_ = std::make_unique<MPCorFilter>(pset);
  rpc_integrator_ = std::make_unique<RPCIntegrator>(pset, consumesColl);

  dtGeomH = esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

DTTrigPhase2Prod::~DTTrigPhase2Prod() {
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "DTp2: calling destructor" << std::endl;
}

void DTTrigPhase2Prod::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "beginRun " << iRun.id().run();
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "beginRun: getting DT geometry";

  grouping_obj_->initialise(iEventSetup);               // Grouping object initialisation
  mpathanalyzer_->initialise(iEventSetup);              // Analyzer object initialisation
  mpathqualityenhancer_->initialise(iEventSetup);       // Filter object initialisation
  mpathredundantfilter_->initialise(iEventSetup);       // Filter object initialisation
  mpathqualityenhancerbayes_->initialise(iEventSetup);  // Filter object initialisation
  mpathhitsfilter_->initialise(iEventSetup);
  mpathassociator_->initialise(iEventSetup);  // Associator object initialisation
  mpathcorfilter_->initialise(iEventSetup);

  if (auto geom = iEventSetup.getHandle(dtGeomH)) {
    dtGeo_ = &(*geom);
  }
}

void DTTrigPhase2Prod::produce(Event& iEvent, const EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "produce";
  edm::Handle<DTDigiCollection> dtdigis;
  iEvent.getByToken(dtDigisToken_, dtdigis);

  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "\t Getting the RPC RecHits" << std::endl;
  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByToken(rpcRecHitsLabel_, rpcRecHits);

  ////////////////////////////////
  // GROUPING CODE:
  ////////////////////////////////

  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (const auto& detUnitIt : *dtdigis) {
    const DTLayerId& layId = detUnitIt.first;
    const DTChamberId chambId = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = detUnitIt.second;
    digiMap[chambId].put(range, layId);
  }

  // generate a list muon paths for each event!!!
  if (debug_ && activateBuffer_)
    LogDebug("DTTrigPhase2Prod") << "produce - Getting and grouping digis per chamber using a buffer and super cells.";
  else if (debug_)
    LogDebug("DTTrigPhase2Prod") << "produce - Getting and grouping digis per chamber.";

  std::map<int, MuonPathPtrs> muonpaths;
  for (const auto& ich : dtGeo_->chambers()) {
    // The code inside this for loop would ideally later fit inside a trigger unit (in principle, a DT station) of the future Phase 2 DT Trigger.
    const DTChamber* chamb = ich;
    DTChamberId chid = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);

    if (dmit == digiMap.end())
      continue;

    if (activateBuffer_) {  // Use buffering (per chamber) or not
      // Import digis from the station
      std::vector<std::pair<DTLayerId, DTDigi>> tmpvec;
      tmpvec.clear();

      for (const auto& dtLayerIdIt : (*dmit).second) {
        for (DTDigiCollection::const_iterator digiIt = (dtLayerIdIt.second).first;
             digiIt != (dtLayerIdIt.second).second;
             digiIt++) {
          tmpvec.emplace_back(dtLayerIdIt.first, *digiIt);
        }
      }

      // Check to enhance CPU time usage
      if (tmpvec.empty())
        continue;

      // Order digis depending on TDC time and insert them into a queue (FIFO buffer). TODO: adapt for MC simulations.
      std::sort(tmpvec.begin(), tmpvec.end(), DigiTimeOrdering);
      std::queue<std::pair<DTLayerId, DTDigi>> timequeue;

      for (const auto& elem : tmpvec)
        timequeue.emplace(elem);
      tmpvec.clear();

      // Distribute the digis from the queue into supercells
      std::vector<DTDigiCollection*> superCells;
      superCells = distribDigis(timequeue);

      // Process each supercell & collect the resulting muonpaths (as the muonpaths std::vector is only enlarged each time
      // the groupings access it, it's not needed to "collect" the final products).

      while (!superCells.empty()) {
        grouping_obj_->run(iEvent, iEventSetup, *(superCells.back()), muonpaths[chid.rawId()]);
        superCells.pop_back();
      }
    } else {
      grouping_obj_->run(iEvent, iEventSetup, (*dmit).second, muonpaths[chid.rawId()]);
    }
  }
  digiMap.clear();

  if (dump_) {
    for (auto& ch_muonpaths : muonpaths) {
      for (unsigned int i = 0; i < ch_muonpaths.second.size(); i++) {
        stringstream ss;
        ss << iEvent.id().event() << "      mpath " << i << ": ";
        for (int lay = 0; lay < ch_muonpaths.second.at(i)->nprimitives(); lay++)
          ss << ch_muonpaths.second.at(i)->primitive(lay)->channelId() << " ";
        for (int lay = 0; lay < ch_muonpaths.second.at(i)->nprimitives(); lay++)
          ss << ch_muonpaths.second.at(i)->primitive(lay)->tdcTimeStamp() << " ";
        for (int lay = 0; lay < ch_muonpaths.second.at(i)->nprimitives(); lay++)
          ss << ch_muonpaths.second.at(i)->primitive(lay)->laterality() << " ";
        LogInfo("DTTrigPhase2Prod") << ss.str();
      }
    }
  }

  std::map<int, std::vector<lat_vector>> lateralities;
  if (!output_mixer_) {
    for (auto& ch_muonpaths : muonpaths) {
      if (algo_ == Standard) {
        latprovider_->run(iEvent, iEventSetup, ch_muonpaths.second, lateralities[ch_muonpaths.first]);
      }
    }
  }

  // FILTER GROUPING
  std::map<int, MuonPathPtrs> filteredmuonpaths;
  for (auto& ch_muonpaths : muonpaths) {
    if (algo_ == Standard) {
      mpathredundantfilter_->run(iEvent, iEventSetup, ch_muonpaths.second, filteredmuonpaths[ch_muonpaths.first]);
    } else {
      mpathhitsfilter_->run(iEvent, iEventSetup, ch_muonpaths.second, filteredmuonpaths[ch_muonpaths.first]);
    }
  }

  if (dump_) {
    for (auto& ch_filteredmuonpaths : filteredmuonpaths) {
      for (unsigned int i = 0; i < ch_filteredmuonpaths.second.size(); i++) {
        stringstream ss;
        ss << iEvent.id().event() << " filt. mpath " << i << ": ";
        for (int lay = 0; lay < ch_filteredmuonpaths.second.at(i)->nprimitives(); lay++)
          ss << ch_filteredmuonpaths.second.at(i)->primitive(lay)->channelId() << " ";
        for (int lay = 0; lay < ch_filteredmuonpaths.second.at(i)->nprimitives(); lay++)
          ss << ch_filteredmuonpaths.second.at(i)->primitive(lay)->tdcTimeStamp() << " ";
        LogInfo("DTTrigPhase2Prod") << ss.str();
      }
    }
  }

  skip_processing_ = output_mixer_ || output_latpredictor_;

  ///////////////////////////////////////////
  /// Fitting SECTION;
  ///////////////////////////////////////////

  if (debug_) {
    for (auto& ch_muonpaths : muonpaths) {
      LogDebug("DTTrigPhase2Prod") << "MUON PATHS found: " << ch_muonpaths.second.size() << " ("
                                   << filteredmuonpaths[ch_muonpaths.first].size() << ") in event "
                                   << iEvent.id().event();
    }
  }
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "filling NmetaPrimtives" << std::endl;
  std::map<int, std::vector<metaPrimitive>> metaPrimitives;
  std::map<int, MuonPathPtrs> outmpaths;
  if (algo_ == Standard) {
    if (debug_)
      LogDebug("DTTrigPhase2Prod") << "Fitting 1SL ";
    for (auto& ch_muonpaths : muonpaths) {  // FIXME, do we need filtered muonpaths?
      if (!output_mixer_ && !output_latpredictor_)
        mpathanalyzer_->run(iEvent,
                            iEventSetup,
                            ch_muonpaths.second,
                            lateralities[ch_muonpaths.first],
                            metaPrimitives[ch_muonpaths.first]);
      else if (output_mixer_) {
        for (auto& inMPath : ch_muonpaths.second) {
          auto sl = inMPath->primitive(0)->superLayerId();  // 0, 1, 2
          int selected_lay = 1;
          if (inMPath->primitive(0)->tdcTimeStamp() != -1)
            selected_lay = 0;
          int dumLayId = inMPath->primitive(selected_lay)->cameraId();
          auto dtDumlayerId = DTLayerId(dumLayId);
          DTSuperLayerId MuonPathSLId(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), sl + 1);
          if (sl == 0)
            metaPrimitives[ch_muonpaths.first].emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           inMPath->primitive(0)->channelId(),
                                                                           inMPath->primitive(0)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(1)->channelId(),
                                                                           inMPath->primitive(1)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(2)->channelId(),
                                                                           inMPath->primitive(2)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(3)->channelId(),
                                                                           inMPath->primitive(3)->tdcTimeStamp(),
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1}));
          else
            metaPrimitives[ch_muonpaths.first].emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           inMPath->primitive(0)->channelId(),
                                                                           inMPath->primitive(0)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(1)->channelId(),
                                                                           inMPath->primitive(1)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(2)->channelId(),
                                                                           inMPath->primitive(2)->tdcTimeStamp(),
                                                                           -1,
                                                                           inMPath->primitive(3)->channelId(),
                                                                           inMPath->primitive(3)->tdcTimeStamp(),
                                                                           -1,
                                                                           -1}));
        }
      } else if (output_latpredictor_) {
        int imp = -1;
        for (auto& inMPath : ch_muonpaths.second) {
          imp++;
          auto sl = inMPath->primitive(0)->superLayerId();  // 0, 1, 2
          int selected_lay = 1;
          if (inMPath->primitive(0)->tdcTimeStamp() != -1)
            selected_lay = 0;
          int dumLayId = inMPath->primitive(selected_lay)->cameraId();
          auto dtDumlayerId = DTLayerId(dumLayId);
          DTSuperLayerId MuonPathSLId(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), sl + 1);
          for (auto& latcomb : lateralities[ch_muonpaths.first][imp]) {
            if (sl == 0)
              metaPrimitives[ch_muonpaths.first].emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             inMPath->primitive(0)->channelId(),
                                                                             inMPath->primitive(0)->tdcTimeStamp(),
                                                                             latcomb[0],
                                                                             inMPath->primitive(1)->channelId(),
                                                                             inMPath->primitive(1)->tdcTimeStamp(),
                                                                             latcomb[1],
                                                                             inMPath->primitive(2)->channelId(),
                                                                             inMPath->primitive(2)->tdcTimeStamp(),
                                                                             latcomb[2],
                                                                             inMPath->primitive(3)->channelId(),
                                                                             inMPath->primitive(3)->tdcTimeStamp(),
                                                                             latcomb[3],
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1}));
            else
              metaPrimitives[ch_muonpaths.first].emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             -1,
                                                                             inMPath->primitive(0)->channelId(),
                                                                             inMPath->primitive(0)->tdcTimeStamp(),
                                                                             latcomb[0],
                                                                             inMPath->primitive(1)->channelId(),
                                                                             inMPath->primitive(1)->tdcTimeStamp(),
                                                                             latcomb[1],
                                                                             inMPath->primitive(2)->channelId(),
                                                                             inMPath->primitive(2)->tdcTimeStamp(),
                                                                             latcomb[2],
                                                                             inMPath->primitive(3)->channelId(),
                                                                             inMPath->primitive(3)->tdcTimeStamp(),
                                                                             latcomb[3],
                                                                             -1}));
          }
        }
      }
    }
  } else {
    // implementation for advanced (2SL) grouping, no filter required..
    if (debug_)
      LogDebug("DTTrigPhase2Prod") << "Fitting 2SL at once ";
    for (auto& ch_muonpaths : muonpaths) {
      mpathanalyzer_->run(iEvent, iEventSetup, ch_muonpaths.second, outmpaths[ch_muonpaths.first]);
    }
  }

  skip_processing_ = skip_processing_ || output_slfitter_;

  if (dump_) {
    for (auto& ch_outmpaths : outmpaths) {
      for (unsigned int i = 0; i < ch_outmpaths.second.size(); i++) {
        LogInfo("DTTrigPhase2Prod") << iEvent.id().event() << " mp " << i << ": "
                                    << ch_outmpaths.second.at(i)->bxTimeValue() << " "
                                    << ch_outmpaths.second.at(i)->horizPos() << " "
                                    << ch_outmpaths.second.at(i)->tanPhi() << " " << ch_outmpaths.second.at(i)->phi()
                                    << " " << ch_outmpaths.second.at(i)->phiB() << " "
                                    << ch_outmpaths.second.at(i)->quality() << " "
                                    << ch_outmpaths.second.at(i)->chiSquare();
      }
    }
    for (auto& ch_metaPrimitives : metaPrimitives) {
      for (unsigned int i = 0; i < ch_metaPrimitives.second.size(); i++) {
        stringstream ss;
        ss << iEvent.id().event() << " mp " << i << ": ";
        printmP(ss.str(), ch_metaPrimitives.second.at(i));
      }
    }
  }

  muonpaths.clear();
  filteredmuonpaths.clear();

  /////////////////////////////////////
  //// CONFIRMATION:
  /////////////////////////////////////

  std::map<int, std::vector<metaPrimitive>> confirmedMetaPrimitives;
  for (auto& ch_metaPrimitives : metaPrimitives) {
    if (!skip_processing_ && allow_confirmation_)
      mpathconfirmator_->run(
          iEvent, iEventSetup, ch_metaPrimitives.second, dtdigis, confirmedMetaPrimitives[ch_metaPrimitives.first]);
    else
      for (auto& mp : ch_metaPrimitives.second) {
        confirmedMetaPrimitives[ch_metaPrimitives.first].push_back(mp);
      }
  }

  metaPrimitives.clear();
  skip_processing_ = skip_processing_ || output_confirmed_;

  /////////////////////////////////////
  //  FILTER SECTIONS:
  ////////////////////////////////////

  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "declaring new vector for filtered" << std::endl;

  std::map<int, std::vector<metaPrimitive>> filteredMetaPrimitives;
  if (algo_ == Standard)
    for (auto& ch_confirmedMetaPrimitives : confirmedMetaPrimitives) {
      if (!skip_processing_)
        mpathqualityenhancer_->run(iEvent,
                                   iEventSetup,
                                   ch_confirmedMetaPrimitives.second,
                                   filteredMetaPrimitives[ch_confirmedMetaPrimitives.first]);
      else
        for (auto& mp : ch_confirmedMetaPrimitives.second) {
          filteredMetaPrimitives[ch_confirmedMetaPrimitives.first].push_back(mp);
        }
    }
  if (dump_) {
    for (auto& ch_filteredMetaPrimitives : filteredMetaPrimitives) {
      for (unsigned int i = 0; i < ch_filteredMetaPrimitives.second.size(); i++) {
        stringstream ss;
        ss << iEvent.id().event() << " filtered mp " << i << ": ";
        printmP(ss.str(), ch_filteredMetaPrimitives.second.at(i));
      }
    }
  }

  skip_processing_ = skip_processing_ || output_slfilter_;
  confirmedMetaPrimitives.clear();

  if (debug_)
    for (auto& ch_filteredMetaPrimitives : filteredMetaPrimitives) {
      LogDebug("DTTrigPhase2Prod") << "DTp2 in event:" << iEvent.id().event() << " we found "
                                   << ch_filteredMetaPrimitives.second.size() << " filteredMetaPrimitives (superlayer)"
                                   << std::endl;
    }
  if (debug_)
    LogDebug("DTTrigPhase2Prod") << "filteredMetaPrimitives: starting correlations" << std::endl;

  /////////////////////////////////////
  //// CORRELATION:
  /////////////////////////////////////

  std::map<int, std::vector<metaPrimitive>> correlatedMetaPrimitives;
  if (algo_ == Standard) {
    for (auto& ch_filteredMetaPrimitives : filteredMetaPrimitives) {
      if (!skip_processing_)
        mpathassociator_->run(iEvent,
                              iEventSetup,
                              ch_filteredMetaPrimitives.second,
                              correlatedMetaPrimitives[ch_filteredMetaPrimitives.first]);
      else
        for (auto& mp : ch_filteredMetaPrimitives.second) {
          correlatedMetaPrimitives[ch_filteredMetaPrimitives.first].push_back(mp);
        }
    }
  } else {
    for (auto& ch_outmpaths : outmpaths) {
      for (const auto& muonpath : ch_outmpaths.second) {
        correlatedMetaPrimitives[ch_outmpaths.first].emplace_back(muonpath->rawId(),
                                                                  (double)muonpath->bxTimeValue(),
                                                                  muonpath->horizPos(),
                                                                  muonpath->tanPhi(),
                                                                  muonpath->phi(),
                                                                  muonpath->phiB(),
                                                                  muonpath->phi_cmssw(),
                                                                  muonpath->phiB_cmssw(),
                                                                  muonpath->chiSquare(),
                                                                  (int)muonpath->quality(),
                                                                  muonpath->primitive(0)->channelId(),
                                                                  muonpath->primitive(0)->tdcTimeStamp(),
                                                                  muonpath->primitive(0)->laterality(),
                                                                  muonpath->primitive(1)->channelId(),
                                                                  muonpath->primitive(1)->tdcTimeStamp(),
                                                                  muonpath->primitive(1)->laterality(),
                                                                  muonpath->primitive(2)->channelId(),
                                                                  muonpath->primitive(2)->tdcTimeStamp(),
                                                                  muonpath->primitive(2)->laterality(),
                                                                  muonpath->primitive(3)->channelId(),
                                                                  muonpath->primitive(3)->tdcTimeStamp(),
                                                                  muonpath->primitive(3)->laterality(),
                                                                  muonpath->primitive(4)->channelId(),
                                                                  muonpath->primitive(4)->tdcTimeStamp(),
                                                                  muonpath->primitive(4)->laterality(),
                                                                  muonpath->primitive(5)->channelId(),
                                                                  muonpath->primitive(5)->tdcTimeStamp(),
                                                                  muonpath->primitive(5)->laterality(),
                                                                  muonpath->primitive(6)->channelId(),
                                                                  muonpath->primitive(6)->tdcTimeStamp(),
                                                                  muonpath->primitive(6)->laterality(),
                                                                  muonpath->primitive(7)->channelId(),
                                                                  muonpath->primitive(7)->tdcTimeStamp(),
                                                                  muonpath->primitive(7)->laterality());
      }
    }
  }

  skip_processing_ = skip_processing_ || output_matcher_;

  if (debug_)
    for (auto& ch_correlatedMetaPrimitives : correlatedMetaPrimitives) {
      LogDebug("DTTrigPhase2Prod") << "DTp2 in event:" << iEvent.id().event() << " we found "
                                   << ch_correlatedMetaPrimitives.second.size() << " correlatedMetPrimitives (chamber)";
    }
  if (dump_) {
    for (auto& ch_correlatedMetaPrimitives : correlatedMetaPrimitives) {
      LogDebug("DTTrigPhase2Prod") << "DTp2 in event:" << iEvent.id().event() << " we found "
                                   << ch_correlatedMetaPrimitives.second.size() << " correlatedMetPrimitives (chamber)";
    }
    for (auto& ch_correlatedMetaPrimitives : correlatedMetaPrimitives) {
      for (unsigned int i = 0; i < ch_correlatedMetaPrimitives.second.size(); i++) {
        stringstream ss;
        ss << iEvent.id().event() << " correlated mp " << i << ": ";
        printmPC(ss.str(), ch_correlatedMetaPrimitives.second.at(i));
      }
    }
  }

  // Correlated Filtering
  std::map<int, std::vector<metaPrimitive>> filtCorrelatedMetaPrimitives;
  if (algo_ == Standard) {
    for (auto& ch_filteredMetaPrimitives : filteredMetaPrimitives) {
      if (!skip_processing_)
        mpathcorfilter_->run(iEvent,
                             iEventSetup,
                             ch_filteredMetaPrimitives.second,
                             correlatedMetaPrimitives[ch_filteredMetaPrimitives.first],
                             filtCorrelatedMetaPrimitives[ch_filteredMetaPrimitives.first]);
      else {
        for (auto& mp : ch_filteredMetaPrimitives.second) {
          filtCorrelatedMetaPrimitives[ch_filteredMetaPrimitives.first].push_back(mp);
        }
        if (output_matcher_)
          for (auto& mp : correlatedMetaPrimitives[ch_filteredMetaPrimitives.first]) {
            filtCorrelatedMetaPrimitives[ch_filteredMetaPrimitives.first].push_back(mp);
          }
      }
    }
  }

  correlatedMetaPrimitives.clear();
  filteredMetaPrimitives.clear();

  double shift_back = 0;
  if (scenario_ == MC)  //scope for MC
    shift_back = 400;
  else if (scenario_ == DATA)  //scope for data
    shift_back = 0;
  else if (scenario_ == SLICE_TEST)  //scope for slice test
    shift_back = 400;

  // RPC integration
  if (useRPC_) {
    rpc_integrator_->initialise(iEventSetup, shift_back);
    rpc_integrator_->prepareMetaPrimitives(rpcRecHits);
    for (auto& ch_correlatedMetaPrimitives : filtCorrelatedMetaPrimitives) {
      rpc_integrator_->matchWithDTAndUseRPCTime(ch_correlatedMetaPrimitives.second);  // Probably this is a FIXME
    }
    rpc_integrator_->makeRPCOnlySegments();
    rpc_integrator_->storeRPCSingleHits();
    rpc_integrator_->removeRPCHitsUsed();
  }

  /// STORING RESULTs
  vector<L1Phase2MuDTPhDigi> outP2Ph;
  vector<L1Phase2MuDTExtPhDigi> outExtP2Ph;
  vector<L1Phase2MuDTThDigi> outP2Th;
  vector<L1Phase2MuDTExtThDigi> outExtP2Th;

  // Assigning index value
  if (!skip_processing_)
    for (auto& ch_correlatedMetaPrimitives : filtCorrelatedMetaPrimitives) {
      assignIndex(ch_correlatedMetaPrimitives.second);
    }

  for (auto& ch_correlatedMetaPrimitives : filtCorrelatedMetaPrimitives) {
    for (const auto& metaPrimitiveIt : ch_correlatedMetaPrimitives.second) {
      DTChamberId chId(metaPrimitiveIt.rawId);
      DTSuperLayerId slId(metaPrimitiveIt.rawId);
      if (debug_)
        LogDebug("DTTrigPhase2Prod") << "looping in final vector: SuperLayerId" << chId << " x=" << metaPrimitiveIt.x
                                     << " quality=" << metaPrimitiveIt.quality
                                     << " BX=" << round(metaPrimitiveIt.t0 / 25.) << " index=" << metaPrimitiveIt.index;

      int sectorTP = chId.sector();
      //sectors 13 and 14 exist only for the outermost stations for sectors 4 and 10 respectively
      //due to the larger MB4 that are divided into two.
      if (sectorTP == 13)
        sectorTP = 4;
      if (sectorTP == 14)
        sectorTP = 10;
      sectorTP = sectorTP - 1;
      int sl = 0;
      if (metaPrimitiveIt.quality < LOWLOWQ || metaPrimitiveIt.quality == CHIGHQ) {
        if (inner(metaPrimitiveIt))
          sl = 1;
        else
          sl = 3;
      }

      float tp_t0 =
          (metaPrimitiveIt.t0 - shift_back * LHC_CLK_FREQ) * ((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ);

      if (debug_)
        LogDebug("DTTrigPhase2Prod") << "pushing back phase-2 dataformat carlo-federica dataformat";

      if (slId.superLayer() != 2) {
        if (df_extended_ == 1 || df_extended_ == 2) {
          int pathWireId[8] = {metaPrimitiveIt.wi1,
                               metaPrimitiveIt.wi2,
                               metaPrimitiveIt.wi3,
                               metaPrimitiveIt.wi4,
                               metaPrimitiveIt.wi5,
                               metaPrimitiveIt.wi6,
                               metaPrimitiveIt.wi7,
                               metaPrimitiveIt.wi8};

          int pathTDC[8] = {max((int)round(metaPrimitiveIt.tdc1 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc2 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc3 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc4 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc5 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc6 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc7 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc8 - shift_back * LHC_CLK_FREQ), -1)};

          int pathLat[8] = {metaPrimitiveIt.lat1,
                            metaPrimitiveIt.lat2,
                            metaPrimitiveIt.lat3,
                            metaPrimitiveIt.lat4,
                            metaPrimitiveIt.lat5,
                            metaPrimitiveIt.lat6,
                            metaPrimitiveIt.lat7,
                            metaPrimitiveIt.lat8};

          // phiTP (extended DF)
          outExtP2Ph.emplace_back(
              L1Phase2MuDTExtPhDigi((int)round(metaPrimitiveIt.t0 / (float)LHC_CLK_FREQ) - shift_back,
                                    chId.wheel(),                                           // uwh   (m_wheel)
                                    sectorTP,                                               // usc   (m_sector)
                                    chId.station(),                                         // ust   (m_station)
                                    sl,                                                     // ust   (m_station)
                                    (int)round(metaPrimitiveIt.phi * PHIRES_CONV),          // uphi  (m_phiAngle)
                                    (int)round(metaPrimitiveIt.phiB * PHIBRES_CONV),        // uphib (m_phiBending)
                                    metaPrimitiveIt.quality,                                // uqua  (m_qualityCode)
                                    metaPrimitiveIt.index,                                  // uind  (m_segmentIndex)
                                    tp_t0,                                                  // ut0   (m_t0Segment)
                                    (int)round(metaPrimitiveIt.chi2 * CHI2RES_CONV),        // uchi2 (m_chi2Segment)
                                    (int)round(metaPrimitiveIt.x * 1000),                   // ux    (m_xLocal)
                                    (int)round(metaPrimitiveIt.tanPhi * 1000),              // utan  (m_tanPsi)
                                    (int)round(metaPrimitiveIt.phi_cmssw * PHIRES_CONV),    // uphi  (m_phiAngleCMSSW)
                                    (int)round(metaPrimitiveIt.phiB_cmssw * PHIBRES_CONV),  // uphib (m_phiBendingCMSSW)
                                    metaPrimitiveIt.rpcFlag,                                // urpc  (m_rpcFlag)
                                    pathWireId,
                                    pathTDC,
                                    pathLat));
        }
        if (df_extended_ == 0 || df_extended_ == 2) {
          // phiTP (standard DF)
          outP2Ph.push_back(L1Phase2MuDTPhDigi(
              (int)round(metaPrimitiveIt.t0 / (float)LHC_CLK_FREQ) - shift_back,
              chId.wheel(),                                     // uwh (m_wheel)
              sectorTP,                                         // usc (m_sector)
              chId.station(),                                   // ust (m_station)
              sl,                                               // ust (m_station)
              (int)round(metaPrimitiveIt.phi * PHIRES_CONV),    // uphi (_phiAngle)
              (int)round(metaPrimitiveIt.phiB * PHIBRES_CONV),  // uphib (m_phiBending)
              metaPrimitiveIt.quality,                          // uqua (m_qualityCode)
              metaPrimitiveIt.index,                            // uind (m_segmentIndex)
              tp_t0,                                            // ut0 (m_t0Segment)
              (int)round(metaPrimitiveIt.chi2 * CHI2RES_CONV),  // uchi2 (m_chi2Segment)
              metaPrimitiveIt.rpcFlag                           // urpc (m_rpcFlag)
              ));
        }
      } else {
        if (df_extended_ == 1 || df_extended_ == 2) {
          int pathWireId[4] = {metaPrimitiveIt.wi1, metaPrimitiveIt.wi2, metaPrimitiveIt.wi3, metaPrimitiveIt.wi4};

          int pathTDC[4] = {max((int)round(metaPrimitiveIt.tdc1 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc2 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc3 - shift_back * LHC_CLK_FREQ), -1),
                            max((int)round(metaPrimitiveIt.tdc4 - shift_back * LHC_CLK_FREQ), -1)};

          int pathLat[4] = {metaPrimitiveIt.lat1, metaPrimitiveIt.lat2, metaPrimitiveIt.lat3, metaPrimitiveIt.lat4};

          // thTP (extended DF)
          outExtP2Th.emplace_back(
              L1Phase2MuDTExtThDigi((int)round(metaPrimitiveIt.t0 / (float)LHC_CLK_FREQ) - shift_back,
                                    chId.wheel(),                                           // uwh   (m_wheel)
                                    sectorTP,                                               // usc   (m_sector)
                                    chId.station(),                                         // ust   (m_station)
                                    (int)round(metaPrimitiveIt.phi * ZRES_CONV),            // uz    (m_zGlobal)
                                    (int)round(metaPrimitiveIt.phiB * KRES_CONV),           // uk    (m_kSlope)
                                    metaPrimitiveIt.quality,                                // uqua  (m_qualityCode)
                                    metaPrimitiveIt.index,                                  // uind  (m_segmentIndex)
                                    tp_t0,                                                  // ut0   (m_t0Segment)
                                    (int)round(metaPrimitiveIt.chi2 * CHI2RES_CONV),        // uchi2 (m_chi2Segment)
                                    (int)round(metaPrimitiveIt.x * 1000),                   // ux    (m_yLocal)
                                    (int)round(metaPrimitiveIt.phi_cmssw * PHIRES_CONV),    // uphi  (m_zCMSSW)
                                    (int)round(metaPrimitiveIt.phiB_cmssw * PHIBRES_CONV),  // uphib (m_kCMSSW)
                                    metaPrimitiveIt.rpcFlag,                                // urpc  (m_rpcFlag)
                                    pathWireId,
                                    pathTDC,
                                    pathLat));
        }
        if (df_extended_ == 0 || df_extended_ == 2) {
          // thTP (standard DF)
          outP2Th.push_back(L1Phase2MuDTThDigi(
              (int)round(metaPrimitiveIt.t0 / (float)LHC_CLK_FREQ) - shift_back,
              chId.wheel(),                                     // uwh (m_wheel)
              sectorTP,                                         // usc (m_sector)
              chId.station(),                                   // ust (m_station)
              (int)round(metaPrimitiveIt.phi * ZRES_CONV),      // uz (m_zGlobal)
              (int)round(metaPrimitiveIt.phiB * KRES_CONV),     // uk (m_kSlope)
              metaPrimitiveIt.quality,                          // uqua (m_qualityCode)
              metaPrimitiveIt.index,                            // uind (m_segmentIndex)
              tp_t0,                                            // ut0 (m_t0Segment)
              (int)round(metaPrimitiveIt.chi2 * CHI2RES_CONV),  // uchi2 (m_chi2Segment)
              metaPrimitiveIt.rpcFlag                           // urpc (m_rpcFlag)
              ));
        }
      }
    }
  }

  // Storing RPC hits that were not used elsewhere
  if (useRPC_) {
    for (auto rpc_dt_digi = rpc_integrator_->rpcRecHits_translated_.begin();
         rpc_dt_digi != rpc_integrator_->rpcRecHits_translated_.end();
         rpc_dt_digi++) {
      outP2Ph.push_back(*rpc_dt_digi);
    }
  }

  // Storing Phi results
  if (df_extended_ == 1 || df_extended_ == 2) {
    std::unique_ptr<L1Phase2MuDTExtPhContainer> resultExtP2Ph(new L1Phase2MuDTExtPhContainer);
    resultExtP2Ph->setContainer(outExtP2Ph);
    iEvent.put(std::move(resultExtP2Ph));
  }
  if (df_extended_ == 0 || df_extended_ == 2) {
    std::unique_ptr<L1Phase2MuDTPhContainer> resultP2Ph(new L1Phase2MuDTPhContainer);
    resultP2Ph->setContainer(outP2Ph);
    iEvent.put(std::move(resultP2Ph));
  }
  outExtP2Ph.clear();
  outExtP2Ph.erase(outExtP2Ph.begin(), outExtP2Ph.end());
  outP2Ph.clear();
  outP2Ph.erase(outP2Ph.begin(), outP2Ph.end());

  // Storing Theta results
  if (df_extended_ == 1 || df_extended_ == 2) {
    std::unique_ptr<L1Phase2MuDTExtThContainer> resultExtP2Th(new L1Phase2MuDTExtThContainer);
    resultExtP2Th->setContainer(outExtP2Th);
    iEvent.put(std::move(resultExtP2Th));
  }
  if (df_extended_ == 0 || df_extended_ == 2) {
    std::unique_ptr<L1Phase2MuDTThContainer> resultP2Th(new L1Phase2MuDTThContainer);
    resultP2Th->setContainer(outP2Th);
    iEvent.put(std::move(resultP2Th));
  }
  outExtP2Th.clear();
  outExtP2Th.erase(outExtP2Th.begin(), outExtP2Th.end());
  outP2Th.clear();
  outP2Th.erase(outP2Th.begin(), outP2Th.end());
}

void DTTrigPhase2Prod::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  grouping_obj_->finish();
  mpathanalyzer_->finish();
  mpathqualityenhancer_->finish();
  mpathqualityenhancerbayes_->finish();
  mpathredundantfilter_->finish();
  mpathhitsfilter_->finish();
  mpathassociator_->finish();
  rpc_integrator_->finish();
};

bool DTTrigPhase2Prod::outer(const metaPrimitive& mp) const {
  int counter = (mp.wi5 != -1) + (mp.wi6 != -1) + (mp.wi7 != -1) + (mp.wi8 != -1);
  return (counter > 2);
}

bool DTTrigPhase2Prod::inner(const metaPrimitive& mp) const {
  int counter = (mp.wi1 != -1) + (mp.wi2 != -1) + (mp.wi3 != -1) + (mp.wi4 != -1);
  return (counter > 2);
}

bool DTTrigPhase2Prod::hasPosRF(int wh, int sec) const { return wh > 0 || (wh == 0 && sec % 4 > 1); }

void DTTrigPhase2Prod::printmP(const string& ss, const metaPrimitive& mP) const {
  DTSuperLayerId slId(mP.rawId);
  LogInfo("DTTrigPhase2Prod") << ss << (int)slId << "\t " << setw(2) << left << mP.wi1 << " " << setw(2) << left
                              << mP.wi2 << " " << setw(2) << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " "
                              << setw(5) << left << mP.tdc1 << " " << setw(5) << left << mP.tdc2 << " " << setw(5)
                              << left << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " " << setw(10) << right
                              << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " "
                              << setw(13) << left << mP.chi2 << " r:" << rango(mP);
}

void DTTrigPhase2Prod::printmP(const metaPrimitive& mP) const {
  DTSuperLayerId slId(mP.rawId);
  std::cout << (int)slId << "\t " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2)
            << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " " << setw(5) << left << mP.tdc1 << " " << setw(5)
            << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " "
            << setw(10) << right << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0
            << " " << setw(13) << left << mP.chi2 << " r:" << rango(mP) << std::endl;
}

void DTTrigPhase2Prod::printmPC(const string& ss, const metaPrimitive& mP) const {
  DTChamberId ChId(mP.rawId);
  LogInfo("DTTrigPhase2Prod") << ss << (int)ChId << "\t  " << setw(2) << left << mP.wi1 << " " << setw(2) << left
                              << mP.wi2 << " " << setw(2) << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " "
                              << setw(2) << left << mP.wi5 << " " << setw(2) << left << mP.wi6 << " " << setw(2) << left
                              << mP.wi7 << " " << setw(2) << left << mP.wi8 << " " << setw(5) << left << mP.tdc1 << " "
                              << setw(5) << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5)
                              << left << mP.tdc4 << " " << setw(5) << left << mP.tdc5 << " " << setw(5) << left
                              << mP.tdc6 << " " << setw(5) << left << mP.tdc7 << " " << setw(5) << left << mP.tdc8
                              << " " << setw(2) << left << mP.lat1 << " " << setw(2) << left << mP.lat2 << " "
                              << setw(2) << left << mP.lat3 << " " << setw(2) << left << mP.lat4 << " " << setw(2)
                              << left << mP.lat5 << " " << setw(2) << left << mP.lat6 << " " << setw(2) << left
                              << mP.lat7 << " " << setw(2) << left << mP.lat8 << " " << setw(10) << right << mP.x << " "
                              << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " " << setw(13)
                              << left << mP.chi2 << " r:" << rango(mP);
}

void DTTrigPhase2Prod::printmPC(const metaPrimitive& mP) const {
  DTChamberId ChId(mP.rawId);
  std::cout << (int)ChId << "\t  " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2)
            << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " " << setw(2) << left << mP.wi5 << " " << setw(2)
            << left << mP.wi6 << " " << setw(2) << left << mP.wi7 << " " << setw(2) << left << mP.wi8 << " " << setw(5)
            << left << mP.tdc1 << " " << setw(5) << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " "
            << setw(5) << left << mP.tdc4 << " " << setw(5) << left << mP.tdc5 << " " << setw(5) << left << mP.tdc6
            << " " << setw(5) << left << mP.tdc7 << " " << setw(5) << left << mP.tdc8 << " " << setw(2) << left
            << mP.lat1 << " " << setw(2) << left << mP.lat2 << " " << setw(2) << left << mP.lat3 << " " << setw(2)
            << left << mP.lat4 << " " << setw(2) << left << mP.lat5 << " " << setw(2) << left << mP.lat6 << " "
            << setw(2) << left << mP.lat7 << " " << setw(2) << left << mP.lat8 << " " << setw(10) << right << mP.x
            << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " " << setw(13) << left
            << mP.chi2 << " r:" << rango(mP) << std::endl;
}

int DTTrigPhase2Prod::rango(const metaPrimitive& mp) const {
  if (mp.quality == 1 or mp.quality == 2)
    return 3;
  if (mp.quality == 3 or mp.quality == 4)
    return 4;
  return mp.quality;
}

void DTTrigPhase2Prod::assignIndex(std::vector<metaPrimitive>& inMPaths) {
  std::map<int, std::vector<metaPrimitive>> primsPerBX;
  for (const auto& metaPrimitive : inMPaths) {
    int BX = round(metaPrimitive.t0 / 25.);
    primsPerBX[BX].push_back(metaPrimitive);
  }
  inMPaths.clear();
  for (auto& prims : primsPerBX) {
    assignIndexPerBX(prims.second);
    for (const auto& primitive : prims.second)
      if (primitive.index <= max_index_)
        inMPaths.push_back(primitive);
  }
}

void DTTrigPhase2Prod::assignIndexPerBX(std::vector<metaPrimitive>& inMPaths) {
  // First we asociate a new index to the metaprimitive depending on quality or phiB;
  uint32_t rawId = -1;
  int numP = -1;
  for (auto& metaPrimitiveIt : inMPaths) {
    numP++;
    rawId = metaPrimitiveIt.rawId;
    int iOrder = assignQualityOrder(metaPrimitiveIt);
    int inf = 0;
    int numP2 = -1;
    for (auto& metaPrimitiveItN : inMPaths) {
      int nOrder = assignQualityOrder(metaPrimitiveItN);
      numP2++;
      if (rawId != metaPrimitiveItN.rawId)
        continue;
      if (numP2 == numP) {
        metaPrimitiveIt.index = inf;
        break;
      } else if (iOrder < nOrder) {
        inf++;
      } else if (iOrder > nOrder) {
        metaPrimitiveItN.index++;
      } else if (iOrder == nOrder) {
        if (std::abs(metaPrimitiveIt.phiB) >= std::abs(metaPrimitiveItN.phiB)) {
          inf++;
        } else if (std::abs(metaPrimitiveIt.phiB) < std::abs(metaPrimitiveItN.phiB)) {
          metaPrimitiveItN.index++;
        }
      }
    }  // ending second for
  }    // ending first for
}

int DTTrigPhase2Prod::assignQualityOrder(const metaPrimitive& mP) const {
  if (mP.quality > 8 || mP.quality < 1)
    return -1;

  return qmap_.find(mP.quality)->second;
}

std::vector<DTDigiCollection*> DTTrigPhase2Prod::distribDigis(std::queue<std::pair<DTLayerId, DTDigi>>& inQ) {
  std::vector<std::queue<std::pair<DTLayerId, DTDigi>>*> tmpVector;
  tmpVector.clear();
  std::vector<DTDigiCollection*> collVector;
  collVector.clear();
  while (!inQ.empty()) {
    processDigi(inQ, tmpVector);
  }
  for (auto& sQ : tmpVector) {
    DTDigiCollection tmpColl;
    while (!sQ->empty()) {
      tmpColl.insertDigi((sQ->front().first), (sQ->front().second));
      sQ->pop();
    }
    collVector.push_back(&tmpColl);
  }
  return collVector;
}

void DTTrigPhase2Prod::processDigi(std::queue<std::pair<DTLayerId, DTDigi>>& inQ,
                                   std::vector<std::queue<std::pair<DTLayerId, DTDigi>>*>& vec) {
  bool classified = false;
  if (!vec.empty()) {
    for (auto& sC : vec) {  // Conditions for entering a super cell.
      if ((sC->front().second.time() + superCelltimewidth_) > inQ.front().second.time()) {
        // Time requirement
        if (TMath::Abs(sC->front().second.wire() - inQ.front().second.wire()) <= superCellhalfspacewidth_) {
          // Spatial requirement
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

  std::queue<std::pair<DTLayerId, DTDigi>> newQueue;

  std::pair<DTLayerId, DTDigi> tmpPair;
  tmpPair = std::move(inQ.front());
  newQueue.push(tmpPair);
  inQ.pop();

  vec.push_back(&newQueue);
}

void DTTrigPhase2Prod::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // dtTriggerPhase2PrimitiveDigis
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("CalibratedDigis"));
  desc.add<int>("timeTolerance", 999999);
  desc.add<double>("tanPhiTh", 1.0);
  desc.add<double>("tanPhiThw2max", 1.3);
  desc.add<double>("tanPhiThw2min", 0.5);
  desc.add<double>("tanPhiThw1max", 0.9);
  desc.add<double>("tanPhiThw1min", 0.2);
  desc.add<double>("tanPhiThw0", 0.5);
  desc.add<double>("chi2Th", 0.01);
  desc.add<double>("chi2corTh", 0.1);
  desc.add<bool>("useBX_correlation", false);
  desc.add<double>("dT0_correlate_TP", 25.0);
  desc.add<int>("dBX_correlate_TP", 0);
  desc.add<double>("dTanPsi_correlate_TP", 99999.0);
  desc.add<bool>("clean_chi2_correlation", true);
  desc.add<bool>("allow_confirmation", true);
  desc.add<double>("minx_match_2digis", 1.0);
  desc.add<int>("scenario", 0);
  desc.add<int>("df_extended", 0);
  desc.add<int>("max_primitives", 999);
  desc.add<bool>("output_mixer", false);
  desc.add<bool>("output_latpredictor", false);
  desc.add<bool>("output_slfitter", false);
  desc.add<bool>("output_slfilter", false);
  desc.add<bool>("output_confirmed", false);
  desc.add<bool>("output_matcher", false);
  desc.add<edm::FileInPath>("ttrig_filename", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/wire_rawId_ttrig.txt"));
  desc.add<edm::FileInPath>("z_filename", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/wire_rawId_z.txt"));
  desc.add<edm::FileInPath>("lut_sl1", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/fitterlut_sl1.dat"));
  desc.add<edm::FileInPath>("lut_sl2", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/fitterlut_slx.dat"));
  desc.add<edm::FileInPath>("lut_sl3", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/fitterlut_sl3.dat"));
  desc.add<edm::FileInPath>("lut_2sl", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/fitterlut_2sl.dat"));
  desc.add<edm::FileInPath>("shift_filename", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/wire_rawId_x.txt"));
  desc.add<edm::FileInPath>("maxdrift_filename",
                            edm::FileInPath("L1Trigger/DTTriggerPhase2/data/drift_time_per_chamber.txt"));
  desc.add<edm::FileInPath>("shift_theta_filename", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/theta_shift.txt"));
  desc.add<edm::FileInPath>("global_coords_filename",
                            edm::FileInPath("L1Trigger/DTTriggerPhase2/data/global_coord_perp_x_phi0.txt"));
  desc.add<int>("algo", 0);
  desc.add<int>("minHits4Fit", 3);
  desc.add<bool>("splitPathPerSL", true);
  desc.addUntracked<bool>("debug", false);
  desc.addUntracked<bool>("dump", false);
  desc.add<edm::InputTag>("rpcRecHits", edm::InputTag("rpcRecHits"));
  desc.add<bool>("useRPC", false);
  desc.add<int>("bx_window", 1);
  desc.add<double>("phi_window", 50.0);
  desc.add<int>("max_quality_to_overwrite_t0", 9);
  desc.add<bool>("storeAllRPCHits", false);
  desc.add<bool>("activateBuffer", false);
  desc.add<double>("superCelltimewidth", 400);
  desc.add<int>("superCellspacewidth", 20);
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<bool>("debug", false);
    psd0.add<double>("angletan", 0.3);
    psd0.add<double>("anglebinwidth", 1.0);
    psd0.add<double>("posbinwidth", 2.1);
    psd0.add<double>("maxdeltaAngDeg", 10);
    psd0.add<double>("maxdeltaPos", 10);
    psd0.add<int>("UpperNumber", 6);
    psd0.add<int>("LowerNumber", 4);
    psd0.add<double>("MaxDistanceToWire", 0.03);
    psd0.add<int>("minNLayerHits", 6);
    psd0.add<int>("minSingleSLHitsMax", 3);
    psd0.add<int>("minSingleSLHitsMin", 3);
    psd0.add<bool>("allowUncorrelatedPatterns", true);
    psd0.add<int>("minUncorrelatedHits", 3);
    desc.add<edm::ParameterSetDescription>("HoughGrouping", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<edm::FileInPath>(
        "pattern_filename", edm::FileInPath("L1Trigger/DTTriggerPhase2/data/PseudoBayesPatterns_uncorrelated_v0.root"));
    psd0.addUntracked<bool>("debug", false);
    psd0.add<int>("minNLayerHits", 3);
    psd0.add<int>("minSingleSLHitsMax", 3);
    psd0.add<int>("minSingleSLHitsMin", 0);
    psd0.add<int>("allowedVariance", 1);
    psd0.add<bool>("allowDuplicates", false);
    psd0.add<bool>("setLateralities", true);
    psd0.add<bool>("allowUncorrelatedPatterns", true);
    psd0.add<int>("minUncorrelatedHits", 3);
    psd0.add<bool>("saveOnPlace", true);
    psd0.add<int>("maxPathsPerMatch", 256);
    desc.add<edm::ParameterSetDescription>("PseudoBayesPattern", psd0);
  }
  descriptions.add("dtTriggerPhase2PrimitiveDigis", desc);
}

DEFINE_FWK_MODULE(DTTrigPhase2Prod);
