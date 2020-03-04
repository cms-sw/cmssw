#ifndef TrackAnalyzer_H
#define TrackAnalyzer_H
//
/**\class TrackingAnalyzer TrackingAnalyzer.cc 
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008

#include <memory>
#include <fstream>
#include <unordered_map>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class BeamSpot;
class TrackAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  TrackAnalyzer(const edm::ParameterSet&);
  TrackAnalyzer(const edm::ParameterSet&, edm::ConsumesCollector& iC);
  ~TrackAnalyzer();
  void initHisto(DQMStore::IBooker& ibooker, const edm::EventSetup&, const edm::ParameterSet&);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track);

  void doReset();
  // Compute and locally store the number of Good vertices found
  // in the event. This information is used as X-axis value in
  // the hit-efficiency plots derived from the hit patter. This
  // ugly design to avoid comuting this very same quantity for
  // each and every track while in the analyze method. A
  // redesign of the class is needed in the future.
  void setNumberOfGoodVertices(const edm::Event&);
  void setBX(const edm::Event&);
  void setLumi(const edm::Event&, const edm::EventSetup& iSetup);

private:
  void initHistos();
  void fillHistosForState(const edm::EventSetup& iSetup, const reco::Track& track, std::string sname);
  void bookHistosForState(std::string sname, DQMStore::IBooker& ibooker);
  void bookHistosForHitProperties(DQMStore::IBooker& ibooker);
  void bookHistosForLScertification(DQMStore::IBooker& ibooker);
  void bookHistosForBeamSpot(DQMStore::IBooker& ibooker);
  void bookHistosForTrackerSpecific(DQMStore::IBooker& ibooker);
  void bookHistosForEfficiencyFromHitPatter(DQMStore::IBooker& ibooker,
                                            const edm::EventSetup& iSetup,
                                            const std::string suffix,
                                            bool useInac);
  void fillHistosForHitProperties(const edm::EventSetup& iSetup, const reco::Track& track, std::string sname);
  void fillHistosForLScertification(const edm::EventSetup& iSetup, const reco::Track& track, std::string sname);
  void fillHistosForTrackerSpecific(const reco::Track& track);
  void fillHistosForEfficiencyFromHitPatter(const reco::Track& track,
                                            const std::string suffix,
                                            const float monitoring,
                                            bool useInac);

  // ----------member data ---------------------------
  std::string TopFolder_;

  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiscalersToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;

  float lumi_factor_per_bx_;

  edm::ParameterSet const* conf_;

  std::string stateName_;

  bool doTrackerSpecific_;
  bool doAllPlots_;
  bool doBSPlots_;
  bool doPVPlots_;
  bool doDCAPlots_;
  bool doGeneralPropertiesPlots_;
  bool doMeasurementStatePlots_;
  bool doHitPropertiesPlots_;
  bool doRecHitVsPhiVsEtaPerTrack_;
  bool doRecHitVsPtVsEtaPerTrack_;
  // ADD by Mia
  bool doLayersVsPhiVsEtaPerTrack_;
  bool doTrackRecHitVsPhiVsEtaPerTrack_;
  bool doTrackRecHitVsPtVsEtaPerTrack_;
  bool doTrackLayersVsPhiVsEtaPerTrack_;
  bool doTrack2DChi2Plots_;
  bool doRecHitsPerTrackProfile_;
  // ADD by Mia in order to clean the tracking MEs
  // do not plot *Theta* and TrackPx* and TrackPy*
  bool doThetaPlots_;
  bool doTrackPxPyPlots_;
  // ADD by Mia in order to not plot DistanceOfClosestApproach w.r.t. (0,0,0)
  // the DistanceOfClosestApproach w.r.t. the beam-spot is already shown in DistanceOfClosestApproachToBS
  bool doDCAwrtPVPlots_;
  bool doDCAwrt000Plots_;

  bool doLumiAnalysis_;

  // ADD by Mia in order to turnON test MEs
  bool doTestPlots_;

  //For HI Plots
  bool doHIPlots_;

  // IP significance plots
  bool doSIPPlots_;

  // Compute the hit-finding efficiency using the HitPattern of
  // the reconstructed tracks
  bool doEffFromHitPatternVsPU_;
  bool doEffFromHitPatternVsBX_;
  bool doEffFromHitPatternVsLUMI_;
  int pvNDOF_;
  bool useBPixLayer1_;
  int minNumberOfPixelsPerCluster_;
  float minPixelClusterCharge_;
  std::string qualityString_;

  struct TkParameterMEs {
    TkParameterMEs()
        : TrackP(nullptr),
          TrackPx(nullptr),
          TrackPy(nullptr),
          TrackPz(nullptr),
          TrackPt(nullptr)

          ,
          TrackPxErr(nullptr),
          TrackPyErr(nullptr),
          TrackPzErr(nullptr),
          TrackPtErr(nullptr),
          TrackPErr(nullptr)

          ,
          TrackPtErrVsEta(nullptr)

          ,
          TrackQ(nullptr)

          ,
          TrackPhi(nullptr),
          TrackEta(nullptr),
          TrackTheta(nullptr)

          ,
          TrackPhiErr(nullptr),
          TrackEtaErr(nullptr),
          TrackThetaErr(nullptr)

          ,
          NumberOfRecHitsPerTrackVsPhi(nullptr),
          NumberOfRecHitsPerTrackVsTheta(nullptr),
          NumberOfRecHitsPerTrackVsEta(nullptr),
          NumberOfRecHitVsPhiVsEtaPerTrack(nullptr)

          ,
          NumberOfValidRecHitsPerTrackVsPhi(nullptr),
          NumberOfValidRecHitsPerTrackVsTheta(nullptr),
          NumberOfValidRecHitsPerTrackVsEta(nullptr),
          NumberOfValidRecHitsPerTrackVsPt(nullptr),
          NumberOfValidRecHitVsPhiVsEtaPerTrack(nullptr),
          NumberOfValidRecHitVsPtVsEtaPerTrack(nullptr)

          ,
          NumberOfLostRecHitsPerTrackVsPhi(nullptr),
          NumberOfLostRecHitsPerTrackVsTheta(nullptr),
          NumberOfLostRecHitsPerTrackVsEta(nullptr),
          NumberOfLostRecHitsPerTrackVsPt(nullptr),
          NumberOfLostRecHitVsPhiVsEtaPerTrack(nullptr),
          NumberOfLostRecHitVsPtVsEtaPerTrack(nullptr)

          ,
          NumberOfMIRecHitsPerTrackVsPhi(nullptr),
          NumberOfMIRecHitsPerTrackVsTheta(nullptr),
          NumberOfMIRecHitsPerTrackVsEta(nullptr),
          NumberOfMIRecHitsPerTrackVsPt(nullptr),
          NumberOfMIRecHitVsPhiVsEtaPerTrack(nullptr),
          NumberOfMIRecHitVsPtVsEtaPerTrack(nullptr)

          ,
          NumberOfMORecHitsPerTrackVsPhi(nullptr),
          NumberOfMORecHitsPerTrackVsTheta(nullptr),
          NumberOfMORecHitsPerTrackVsEta(nullptr),
          NumberOfMORecHitsPerTrackVsPt(nullptr),
          NumberOfMORecHitVsPhiVsEtaPerTrack(nullptr),
          NumberOfMORecHitVsPtVsEtaPerTrack(nullptr)

          ,
          NumberOfLayersPerTrackVsPhi(nullptr),
          NumberOfLayersPerTrackVsTheta(nullptr),
          NumberOfLayersPerTrackVsEta(nullptr)

          ,
          Chi2oNDFVsNHits(nullptr),
          Chi2oNDFVsPt(nullptr),
          Chi2oNDFVsEta(nullptr),
          Chi2oNDFVsPhi(nullptr),
          Chi2oNDFVsTheta(nullptr)

          ,
          Chi2ProbVsEta(nullptr),
          Chi2ProbVsPhi(nullptr),
          Chi2ProbVsTheta(nullptr) {}

    MonitorElement* TrackP;
    MonitorElement* TrackPx;
    MonitorElement* TrackPy;
    MonitorElement* TrackPz;
    MonitorElement* TrackPt;
    MonitorElement* TrackPt_NegEta_Phi_btw_neg16_neg32;
    MonitorElement* TrackPt_NegEta_Phi_btw_0_neg16;
    MonitorElement* TrackPt_NegEta_Phi_btw_16_0;
    MonitorElement* TrackPt_NegEta_Phi_btw_32_16;
    MonitorElement* TrackPt_PosEta_Phi_btw_neg16_neg32;
    MonitorElement* TrackPt_PosEta_Phi_btw_0_neg16;
    MonitorElement* TrackPt_PosEta_Phi_btw_16_0;
    MonitorElement* TrackPt_PosEta_Phi_btw_32_16;
    MonitorElement* Ratio_byFolding;
    MonitorElement* Ratio_byFolding2;
    MonitorElement* TrackPtHighPurity;
    MonitorElement* TrackPtTight;
    MonitorElement* TrackPtLoose;
    MonitorElement* Quality;

    MonitorElement* TrackPxErr;
    MonitorElement* TrackPyErr;
    MonitorElement* TrackPzErr;
    MonitorElement* TrackPtErr;
    MonitorElement* TrackPErr;

    MonitorElement* TrackPtErrVsEta;

    MonitorElement* TrackQ;
    MonitorElement* TrackQoverP;

    MonitorElement* TrackPhi;
    MonitorElement* TrackEta;
    MonitorElement* TrackEtaHighPurity;
    MonitorElement* TrackEtaTight;
    MonitorElement* TrackEtaLoose;
    MonitorElement* TrackEtaPhi = nullptr;
    MonitorElement* TrackEtaPhiInverted = nullptr;
    MonitorElement* TrackEtaPhiInvertedoutofphase = nullptr;
    MonitorElement* TkEtaPhi_Ratio_byFoldingmap = nullptr;
    MonitorElement* TkEtaPhi_Ratio_byFoldingmap_op = nullptr;
    MonitorElement* TkEtaPhi_RelativeDifference_byFoldingmap = nullptr;
    MonitorElement* TkEtaPhi_RelativeDifference_byFoldingmap_op = nullptr;
    MonitorElement* TrackEtaPhiInner = nullptr;
    MonitorElement* TrackEtaPhiOuter = nullptr;

    MonitorElement* TrackTheta;

    MonitorElement* TrackPhiErr;
    MonitorElement* TrackEtaErr;
    MonitorElement* TrackThetaErr;

    MonitorElement* NumberOfRecHitsPerTrackVsPhi;
    MonitorElement* NumberOfRecHitsPerTrackVsTheta;
    MonitorElement* NumberOfRecHitsPerTrackVsEta;
    MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack;

    MonitorElement* NumberOfValidRecHitsPerTrackVsPhi;
    MonitorElement* NumberOfValidRecHitsPerTrackVsTheta;
    MonitorElement* NumberOfValidRecHitsPerTrackVsEta;
    MonitorElement* NumberOfValidRecHitsPerTrackVsPt;
    MonitorElement* NumberOfValidRecHitVsPhiVsEtaPerTrack;
    MonitorElement* NumberOfValidRecHitVsPtVsEtaPerTrack;

    MonitorElement* NumberOfLostRecHitsPerTrackVsPhi;
    MonitorElement* NumberOfLostRecHitsPerTrackVsTheta;
    MonitorElement* NumberOfLostRecHitsPerTrackVsEta;
    MonitorElement* NumberOfLostRecHitsPerTrackVsPt;
    MonitorElement* NumberOfLostRecHitVsPhiVsEtaPerTrack;
    MonitorElement* NumberOfLostRecHitVsPtVsEtaPerTrack;

    MonitorElement* NumberOfMIRecHitsPerTrackVsPhi;
    MonitorElement* NumberOfMIRecHitsPerTrackVsTheta;
    MonitorElement* NumberOfMIRecHitsPerTrackVsEta;
    MonitorElement* NumberOfMIRecHitsPerTrackVsPt;
    MonitorElement* NumberOfMIRecHitVsPhiVsEtaPerTrack;
    MonitorElement* NumberOfMIRecHitVsPtVsEtaPerTrack;

    MonitorElement* NumberOfMORecHitsPerTrackVsPhi;
    MonitorElement* NumberOfMORecHitsPerTrackVsTheta;
    MonitorElement* NumberOfMORecHitsPerTrackVsEta;
    MonitorElement* NumberOfMORecHitsPerTrackVsPt;
    MonitorElement* NumberOfMORecHitVsPhiVsEtaPerTrack;
    MonitorElement* NumberOfMORecHitVsPtVsEtaPerTrack;

    MonitorElement* NumberOfLayersPerTrackVsPhi;
    MonitorElement* NumberOfLayersPerTrackVsTheta;
    MonitorElement* NumberOfLayersPerTrackVsEta;

    MonitorElement* Chi2oNDFVsNHits;
    MonitorElement* Chi2oNDFVsPt;
    MonitorElement* Chi2oNDFVsEta;
    MonitorElement* Chi2oNDFVsPhi;
    MonitorElement* Chi2oNDFVsTheta;

    MonitorElement* Chi2ProbVsEta;
    MonitorElement* Chi2ProbVsPhi;
    MonitorElement* Chi2ProbVsTheta;
  };
  std::map<std::string, TkParameterMEs> TkParameterMEMap;

  MonitorElement* NumberOfRecHitsPerTrack;
  MonitorElement* NumberOfValidRecHitsPerTrack;
  MonitorElement* NumberOfLostRecHitsPerTrack;
  MonitorElement* NumberOfMIRecHitsPerTrack = nullptr;
  MonitorElement* NumberOfMORecHitsPerTrack = nullptr;

  MonitorElement* NumberOfRecHitsPerTrackVsPhi = nullptr;
  MonitorElement* NumberOfRecHitsPerTrackVsTheta = nullptr;
  MonitorElement* NumberOfRecHitsPerTrackVsEta = nullptr;
  MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack = nullptr;

  MonitorElement* NumberOfValidRecHitsPerTrackVsPhi = nullptr;
  MonitorElement* NumberOfValidRecHitsPerTrackVsTheta = nullptr;
  MonitorElement* NumberOfValidRecHitsPerTrackVsEta = nullptr;
  MonitorElement* NumberOfValidRecHitsPerTrackVsPt = nullptr;
  MonitorElement* NumberOfValidRecHitVsPhiVsEtaPerTrack = nullptr;
  MonitorElement* NumberOfValidRecHitVsPtVsEtaPerTrack = nullptr;

  MonitorElement* NumberOfLostRecHitsPerTrackVsPhi = nullptr;
  MonitorElement* NumberOfLostRecHitsPerTrackVsTheta = nullptr;
  MonitorElement* NumberOfLostRecHitsPerTrackVsEta = nullptr;
  MonitorElement* NumberOfLostRecHitsPerTrackVsPt = nullptr;
  MonitorElement* NumberOfLostRecHitVsPhiVsEtaPerTrack = nullptr;
  MonitorElement* NumberOfLostRecHitVsPtVsEtaPerTrack = nullptr;

  MonitorElement* NumberOfMIRecHitsPerTrackVsPhi = nullptr;
  MonitorElement* NumberOfMIRecHitsPerTrackVsTheta = nullptr;
  MonitorElement* NumberOfMIRecHitsPerTrackVsEta = nullptr;
  MonitorElement* NumberOfMIRecHitsPerTrackVsPt = nullptr;
  MonitorElement* NumberOfMIRecHitVsPhiVsEtaPerTrack = nullptr;
  MonitorElement* NumberOfMIRecHitVsPtVsEtaPerTrack = nullptr;

  MonitorElement* NumberOfMORecHitsPerTrackVsPhi = nullptr;
  MonitorElement* NumberOfMORecHitsPerTrackVsTheta = nullptr;
  MonitorElement* NumberOfMORecHitsPerTrackVsEta = nullptr;
  MonitorElement* NumberOfMORecHitsPerTrackVsPt = nullptr;
  MonitorElement* NumberOfMORecHitVsPhiVsEtaPerTrack = nullptr;
  MonitorElement* NumberOfMORecHitVsPtVsEtaPerTrack = nullptr;

  MonitorElement* ValidFractionPerTrack = nullptr;
  MonitorElement* ValidFractionVsPhiVsEtaPerTrack = nullptr;

  MonitorElement* NumberOfLayersPerTrack[4] = {nullptr, nullptr, nullptr, nullptr};

  MonitorElement* NumberOfLayersPerTrackVsPhi;
  MonitorElement* NumberOfLayersPerTrackVsTheta;
  MonitorElement* NumberOfLayersPerTrackVsEta;

  MonitorElement* NumberOfLayersVsPhiVsEtaPerTrack[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

  MonitorElement* Chi2;
  MonitorElement* Chi2Prob;
  MonitorElement* Chi2oNDF;

  MonitorElement* Chi2oNDFVsNHits = nullptr;
  MonitorElement* Chi2oNDFVsPt = nullptr;
  MonitorElement* Chi2oNDFVsEta = nullptr;
  MonitorElement* Chi2oNDFVsPhi;
  MonitorElement* Chi2oNDFVsTheta;

  MonitorElement* Chi2ProbVsEta;
  MonitorElement* Chi2ProbVsPhi;
  MonitorElement* Chi2ProbVsTheta;

  MonitorElement* DistanceOfClosestApproach;
  MonitorElement* DistanceOfClosestApproachError;
  MonitorElement* DistanceOfClosestApproachErrorVsPt;
  MonitorElement* DistanceOfClosestApproachErrorVsEta;
  MonitorElement* DistanceOfClosestApproachErrorVsPhi;
  MonitorElement* DistanceOfClosestApproachErrorVsDxy;
  MonitorElement* DistanceOfClosestApproachToBS;
  MonitorElement* DistanceOfClosestApproachToBSdz;
  MonitorElement* AbsDistanceOfClosestApproachToBS;
  MonitorElement* DistanceOfClosestApproachToPV;
  MonitorElement* DistanceOfClosestApproachToPVZoom;
  MonitorElement* DeltaZToPV;
  MonitorElement* DeltaZToPVZoom;
  MonitorElement* DistanceOfClosestApproachVsTheta;
  MonitorElement* DistanceOfClosestApproachVsPhi;
  MonitorElement* DistanceOfClosestApproachToBSVsPhi;
  MonitorElement* DistanceOfClosestApproachToBSVsEta;
  MonitorElement* DistanceOfClosestApproachToPVVsPhi;
  MonitorElement* DistanceOfClosestApproachVsEta;
  MonitorElement* xPointOfClosestApproach;
  MonitorElement* xPointOfClosestApproachToPV;
  MonitorElement* xPointOfClosestApproachVsZ0wrt000;
  MonitorElement* xPointOfClosestApproachVsZ0wrtBS;
  MonitorElement* xPointOfClosestApproachVsZ0wrtPV;
  MonitorElement* yPointOfClosestApproach;
  MonitorElement* yPointOfClosestApproachToPV;
  MonitorElement* yPointOfClosestApproachVsZ0wrt000;
  MonitorElement* yPointOfClosestApproachVsZ0wrtBS;
  MonitorElement* yPointOfClosestApproachVsZ0wrtPV;
  MonitorElement* zPointOfClosestApproach;
  MonitorElement* zPointOfClosestApproachToPV;
  MonitorElement* zPointOfClosestApproachVsPhi;
  MonitorElement *algorithm, *oriAlgo;
  MonitorElement* stoppingSource;
  MonitorElement* stoppingSourceVSeta;
  MonitorElement* stoppingSourceVSphi;
  // TESTING MEs
  MonitorElement* TESTDistanceOfClosestApproachToBS;
  MonitorElement* TESTDistanceOfClosestApproachToBSVsPhi;

  // add by Mia in order to deal w/ LS transitions
  MonitorElement* Chi2oNDF_lumiFlag;
  MonitorElement* NumberOfRecHitsPerTrack_lumiFlag;

  //new plots for Heavy Ion DQM
  MonitorElement* LongDCASig;
  MonitorElement* TransDCASig;
  MonitorElement* dNdPhi_HighPurity;
  MonitorElement* dNdEta_HighPurity;
  MonitorElement* dNdPt_HighPurity;
  MonitorElement* NhitVsEta_HighPurity;
  MonitorElement* NhitVsPhi_HighPurity;
  MonitorElement* Ptdist_HighPurity;
  MonitorElement* dNhitdPt_HighPurity;

  // IP significance plots
  MonitorElement* sipDxyToBS;
  MonitorElement* sipDzToBS;
  MonitorElement* sip3dToPV;
  MonitorElement* sip2dToPV;
  MonitorElement* sipDxyToPV;
  MonitorElement* sipDzToPV;

  struct TkRecHitsPerSubDetMEs {
    MonitorElement* NumberOfRecHitsPerTrack;
    MonitorElement* NumberOfRecHitsPerTrackVsPhi;
    MonitorElement* NumberOfRecHitsPerTrackVsEta;
    MonitorElement* NumberOfRecHitsPerTrackVsPt;
    MonitorElement* NumberOfLayersPerTrack;
    MonitorElement* NumberOfLayersPerTrackVsPhi;
    MonitorElement* NumberOfLayersPerTrackVsEta;
    MonitorElement* NumberOfLayersPerTrackVsPt;
    MonitorElement* RecHitChi2PerTrack;

    int detectorId;
    std::string detectorTag;
  };
  std::map<std::string, TkRecHitsPerSubDetMEs> TkRecHitsPerSubDetMEMap;

  struct Key {
    int det;
    int subdet;
    int monitoring;
    explicit Key(int det, int subdet, int monitoring) : det(det), subdet(subdet), monitoring(monitoring){};
    bool operator==(const Key& other) const {
      return (det == other.det && subdet == other.subdet && monitoring == other.monitoring);
    }
  };

  struct KeyHasher {
    std::size_t operator()(const Key& k) const {
      // 3 bits (0x7) for kind of monitoring (7 kinds at most)
      // next 8 bits to the subdetector (255 subdetectors at most)
      // next 8 bits to the detector (255 detectors at most)
      return (size_t)((k.monitoring & (0x7)) | ((k.subdet & (0xff)) << 3) | ((k.det & (0xff)) << 11));
    }
  };

  std::unordered_map<Key, MonitorElement*, KeyHasher> hits_valid_;
  std::unordered_map<Key, MonitorElement*, KeyHasher> hits_missing_;
  std::unordered_map<Key, MonitorElement*, KeyHasher> hits_inactive_;
  std::unordered_map<Key, MonitorElement*, KeyHasher> hits_bad_;
  std::unordered_map<Key, MonitorElement*, KeyHasher> hits_total_;
  unsigned int good_vertices_;
  unsigned int bx_;
  float pixel_lumi_;
  float scal_lumi_;
  enum monQuantity { VsPU, VsBX, VsPIXELLUMI, VsSCALLUMI, END };
  std::string monName[monQuantity::END] = {"", "VsBX", "VsPIXELLUMI", "VsSCALLUMI"};

  std::string histname;  //for naming the histograms according to algorithm used
};
#endif
