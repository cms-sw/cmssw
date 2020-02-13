#ifndef Phase2L1Trigger_DTTrigger_DTTrigPhase2Prod_cc
#define Phase2L1Trigger_DTTrigger_DTTrigPhase2Prod_cc
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"

#include "L1Trigger/DTPhase2Trigger/interface/MotherGrouping.h"
#include "L1Trigger/DTPhase2Trigger/interface/InitialGrouping.h"
#include "L1Trigger/DTPhase2Trigger/interface/HoughGrouping.h"
#include "L1Trigger/DTPhase2Trigger/interface/PseudoBayesGrouping.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzer.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzerPerSL.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzerInChamber.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAssociator.h"
#include "L1Trigger/DTPhase2Trigger/interface/MPFilter.h"
#include "L1Trigger/DTPhase2Trigger/interface/MPQualityEnhancerFilter.h"
#include "L1Trigger/DTPhase2Trigger/interface/MPRedundantFilter.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

//RPC TP
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/DTPhase2Trigger/interface/RPCIntegrator.h"


#include <fstream>
#include <iostream>
#include <queue>
#include <cmath>


class DTTrigPhase2Prod: public edm::EDProducer {
  typedef std::map< DTChamberId,DTDigiCollection,std::less<DTChamberId> > DTDigiMap;
  typedef DTDigiMap::iterator DTDigiMap_iterator;
  typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

  public:
    //! Constructor
    DTTrigPhase2Prod(const edm::ParameterSet& pset);

    //! Destructor
    ~DTTrigPhase2Prod() override;
    
    //! Create Trigger Units before starting event processing
    //void beginJob(const edm::EventSetup & iEventSetup);
    void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;
  
    //! Producer: process every event and generates trigger data
    void produce(edm::Event & iEvent, const edm::EventSetup& iEventSetup) override;
    
    //! endRun: finish things
    void endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;
    
    edm::ESHandle<DTGeometry> dtGeo;

    std::vector<std::pair<int,MuonPath>> primitives;

    int rango(metaPrimitive mp);
    bool outer(metaPrimitive mp);
    bool inner(metaPrimitive mp);
    void printmP(metaPrimitive mP);
    void printmPC(metaPrimitive mP);

    double trigPos(metaPrimitive mP);
    double trigDir(metaPrimitive mp);

    bool hasPosRF(int wh,int sec);

    double zcn[4];
    double xCenter[2];
    
    void setBXTolerance(int t);
    int getBXTolerance(void);

    void setChiSquareThreshold(float ch2Thr);
    
    void setMinimumQuality(MP_QUALITY q);
    MP_QUALITY getMinimumQuality(void);

    DTTrigGeomUtils *trigGeomUtils;

  private:
    // Trigger Configuration Manager CCB validity flag
    bool my_CCBValid;

    // BX offset used to correct DTTPG output
    int my_BXoffset;

    // Debug Flag
    bool debug;
    bool dump;
    double dT0_correlate_TP;
    int min_phinhits_match_segment;
    bool do_correlation;
    int p2_df;
    int scenario;
    bool printHits;
    bool printPython;
    int eventBX;


    // txt ttrig flag
    bool txt_ttrig_bc0;
    // shift
    edm::FileInPath shift_filename;
    std::map<int,float> shiftinfo;
            
    // ParameterSet
    edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegmentsToken;
    edm::EDGetTokenT<DTDigiCollection> dtDigisToken;
    edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsLabel;

    // Grouping attributes and methods
    Int_t grcode; // Grouping code
    MotherGrouping* grouping_obj;
    MuonPathAnalyzer* mpathanalyzer;
    MPFilter* mpathqualityenhancer;
    MPFilter* mpathredundantfilter;
    MuonPathAssociator* mpathassociator;

    // Buffering
    Bool_t  activateBuffer;
    Int_t   superCellhalfspacewidth;
    Float_t superCelltimewidth;
    std::vector<DTDigiCollection*> distribDigis(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ);
    void processDigi(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ, std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*>& vec);

    // RPC
    RPCIntegrator* rpc_integrator;
    bool useRPC;

    void assignIndex(std::vector<metaPrimitive> &inMPaths);
    void assignIndexPerBX(std::vector<metaPrimitive> &inMPaths);
    int assignQualityOrder(metaPrimitive mP);
};


#endif
