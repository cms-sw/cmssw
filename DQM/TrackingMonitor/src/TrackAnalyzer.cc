/*
 *  See header file for a description of this class.
 *
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include <string>
#include "TMath.h"
#include "DQM/TrackingMonitor/interface/GetLumi.h"

using namespace dqm;

namespace {
  template <typename T, size_t N>
  std::array<T, N+1> makeLogBins(const double min, const double max) {
    const double minLog10 = std::log10(min);
    const double maxLog10 = std::log10(max);
    const double width = (maxLog10-minLog10)/N;
    std::array<T, N+1> ret;
    ret[0] = std::pow(10,minLog10);
    const double mult = std::pow(10, width);
    for(size_t i=1; i<= N; ++i) {
      ret[i] = ret[i-1]*mult;
    }
    return ret;
  }
}

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig) 
    : conf_( nullptr )
    , stateName_                       (iConfig.getParameter<std::string>("MeasurementState") )
    , doTrackerSpecific_               ( iConfig.getParameter<bool>("doTrackerSpecific") )
    , doAllPlots_                      ( iConfig.getParameter<bool>("doAllPlots") )
    , doBSPlots_                       ( iConfig.getParameter<bool>("doBeamSpotPlots") )
    , doPVPlots_                       ( iConfig.getParameter<bool>("doPrimaryVertexPlots") )
    , doDCAPlots_                      ( iConfig.getParameter<bool>("doDCAPlots") )
    , doGeneralPropertiesPlots_        ( iConfig.getParameter<bool>("doGeneralPropertiesPlots") )
    , doMeasurementStatePlots_         ( iConfig.getParameter<bool>("doMeasurementStatePlots") )
    , doHitPropertiesPlots_            ( iConfig.getParameter<bool>("doHitPropertiesPlots") )
    , doRecHitVsPhiVsEtaPerTrack_      ( iConfig.getParameter<bool>("doRecHitVsPhiVsEtaPerTrack") )
    , doRecHitVsPtVsEtaPerTrack_       ( iConfig.getParameter<bool>("doRecHitVsPtVsEtaPerTrack") )
    , doLayersVsPhiVsEtaPerTrack_      ( iConfig.getParameter<bool>("doLayersVsPhiVsEtaPerTrack") )
    , doRecHitsPerTrackProfile_        ( iConfig.getParameter<bool>("doRecHitsPerTrackProfile") )
    , doThetaPlots_                    ( iConfig.getParameter<bool>("doThetaPlots") )
    , doTrackPxPyPlots_                ( iConfig.getParameter<bool>("doTrackPxPyPlots") )
    , doDCAwrtPVPlots_                 ( iConfig.getParameter<bool>("doDCAwrtPVPlots") )
    , doDCAwrt000Plots_                ( iConfig.getParameter<bool>("doDCAwrt000Plots") )
    , doLumiAnalysis_                  ( iConfig.getParameter<bool>("doLumiAnalysis") )
    , doTestPlots_                     ( iConfig.getParameter<bool>("doTestPlots") )
    , doHIPlots_                       ( iConfig.getParameter<bool>("doHIPlots")  )
    , doSIPPlots_                      ( iConfig.getParameter<bool>("doSIPPlots") )
    , doEffFromHitPatternVsPU_         ( iConfig.getParameter<bool>("doEffFromHitPatternVsPU")   )
    , doEffFromHitPatternVsBX_         ( iConfig.getParameter<bool>("doEffFromHitPatternVsBX")   )
    , doEffFromHitPatternVsLUMI_       ( iConfig.getParameter<bool>("doEffFromHitPatternVsLUMI") )
    , pvNDOF_                          ( iConfig.getParameter<int> ("pvNDOF") )
    , useBPixLayer1_                   ( iConfig.getParameter<bool>("useBPixLayer1") )
    , minNumberOfPixelsPerCluster_     ( iConfig.getParameter<int>("minNumberOfPixelsPerCluster") )
    , minPixelClusterCharge_           ( iConfig.getParameter<double>("minPixelClusterCharge") )
    , qualityString_                   ( iConfig.getParameter<std::string>("qualityString"))
    , good_vertices_(0)
    , bx_(0)
    , pixel_lumi_(0.)
    , scal_lumi_(0.)
{
  initHistos();
  TopFolder_ = iConfig.getParameter<std::string>("FolderName"); 
}

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC) 
  : TrackAnalyzer(iConfig)
{
  edm::InputTag bsSrc                 = iConfig.getParameter<edm::InputTag>("beamSpot");
  edm::InputTag primaryVertexInputTag = iConfig.getParameter<edm::InputTag>("primaryVertex");
  edm::InputTag pixelClusterInputTag  = iConfig.getParameter<edm::InputTag>("pixelCluster4lumi");
  edm::InputTag scalInputTag          = iConfig.getParameter<edm::InputTag>("scal");
  beamSpotToken_      = iC.consumes<reco::BeamSpot>(bsSrc);
  pvToken_            = iC.consumes<reco::VertexCollection>(primaryVertexInputTag);
  pixelClustersToken_ = iC.mayConsume<edmNew::DetSetVector<SiPixelCluster> >(pixelClusterInputTag);
  lumiscalersToken_   = iC.mayConsume<LumiScalersCollection>(scalInputTag);
  
  if(useBPixLayer1_) 
    lumi_factor_per_bx_ = GetLumi::FREQ_ORBIT * GetLumi::SECONDS_PER_LS / GetLumi::XSEC_PIXEL_CLUSTER  ;
  else
    lumi_factor_per_bx_ = GetLumi::FREQ_ORBIT * GetLumi::SECONDS_PER_LS / GetLumi::rXSEC_PIXEL_CLUSTER  ;


}

void TrackAnalyzer::initHistos()
{
  Chi2 = nullptr;
  Chi2Prob = nullptr;
  Chi2ProbVsPhi = nullptr;
  Chi2ProbVsEta = nullptr;
  Chi2oNDF = nullptr;
  Chi2oNDFVsEta = nullptr;
  Chi2oNDFVsPhi = nullptr;
  Chi2oNDFVsTheta = nullptr;
  	    
  NumberOfRecHitsPerTrack = nullptr;
  NumberOfValidRecHitsPerTrack = nullptr;
  NumberOfLostRecHitsPerTrack = nullptr;

  NumberOfRecHitsPerTrackVsPhi = nullptr;
  NumberOfRecHitsPerTrackVsTheta = nullptr;
  NumberOfRecHitsPerTrackVsEta = nullptr;

  NumberOfRecHitVsPhiVsEtaPerTrack = nullptr;

  NumberOfValidRecHitsPerTrackVsPhi = nullptr;
  NumberOfValidRecHitsPerTrackVsEta = nullptr;

  DistanceOfClosestApproach = nullptr;
  DistanceOfClosestApproachError = nullptr;
  DistanceOfClosestApproachErrorVsPt = nullptr;
  DistanceOfClosestApproachErrorVsEta = nullptr;
  DistanceOfClosestApproachErrorVsPhi = nullptr;
  DistanceOfClosestApproachErrorVsDxy = nullptr;
  DistanceOfClosestApproachToBS = nullptr;
  AbsDistanceOfClosestApproachToBS = nullptr;
  DistanceOfClosestApproachToPV = nullptr;
  DistanceOfClosestApproachToPVZoom = nullptr;
  DeltaZToPV = nullptr;
  DeltaZToPVZoom = nullptr;
  DistanceOfClosestApproachVsTheta = nullptr;
  DistanceOfClosestApproachVsPhi = nullptr;  
  DistanceOfClosestApproachToBSVsPhi = nullptr;
  DistanceOfClosestApproachToPVVsPhi = nullptr;
  DistanceOfClosestApproachVsEta = nullptr;
  xPointOfClosestApproach = nullptr;
  xPointOfClosestApproachVsZ0wrt000 = nullptr;
  xPointOfClosestApproachVsZ0wrtBS = nullptr;
  yPointOfClosestApproach = nullptr;
  yPointOfClosestApproachVsZ0wrt000 = nullptr;
  yPointOfClosestApproachVsZ0wrtBS = nullptr;
  zPointOfClosestApproach = nullptr;
  zPointOfClosestApproachVsPhi = nullptr;
  algorithm = nullptr;
  oriAlgo = nullptr;
  stoppingSource = nullptr;
  stoppingSourceVSeta = nullptr;
  stoppingSourceVSphi = nullptr;
    // TESTING
  TESTDistanceOfClosestApproachToBS = nullptr;
  TESTDistanceOfClosestApproachToBSVsPhi = nullptr;

// by Mia in order to deal w/ LS transitions
  Chi2oNDF_lumiFlag = nullptr;
  NumberOfRecHitsPerTrack_lumiFlag = nullptr;

  ////////////////////////////////////////////////////////////                                                                                                                                             
  //special Plots for HI DQM  //SHOULD I ADD THE BOOL HERE??                                                                                                                                               
  ////////////////////////////////////////////////////////////                                                                                                                                             
  LongDCASig = nullptr;
  TransDCASig = nullptr;
  dNdPhi_HighPurity = nullptr;
  dNdEta_HighPurity = nullptr;
  dNdPt_HighPurity = nullptr;
  NhitVsEta_HighPurity = nullptr;
  NhitVsPhi_HighPurity = nullptr;

  // IP significance
  sipDxyToBS = nullptr;
  sipDzToBS = nullptr;
  sip3dToPV = nullptr;
  sip2dToPV = nullptr;
  sipDxyToPV = nullptr;
  sipDzToPV = nullptr;

}

TrackAnalyzer::~TrackAnalyzer() 
{ 
}

void TrackAnalyzer::initHisto(DQMStore::IBooker & ibooker, const edm::EventSetup & iSetup, const edm::ParameterSet& iConfig)
{
  conf_ = &iConfig;
  bookHistosForHitProperties(ibooker);
  bookHistosForBeamSpot(ibooker);
  bookHistosForLScertification( ibooker);
  if (doEffFromHitPatternVsPU_   || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "",false);
  if (doEffFromHitPatternVsBX_   || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "VsBX",false);
  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "VsSCALLUMI",false);
  //  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "VsPIXELLUMI");
  if (doEffFromHitPatternVsPU_   || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "",true);
  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) bookHistosForEfficiencyFromHitPatter(ibooker, iSetup, "VsSCALLUMI",true);

  // book tracker specific related histograms
  // ---------------------------------------------------------------------------------//
  if(doTrackerSpecific_ || doAllPlots_) bookHistosForTrackerSpecific(ibooker);
    
  // book state related histograms
  // ---------------------------------------------------------------------------------//
  if (doMeasurementStatePlots_ || doAllPlots_) {

    
    if (stateName_ == "All") {
      bookHistosForState("OuterSurface", ibooker);
      bookHistosForState("InnerSurface", ibooker);
      bookHistosForState("ImpactPoint" , ibooker);
    } else if (
	       stateName_ != "OuterSurface" && 
	       stateName_ != "InnerSurface" && 
	       stateName_ != "ImpactPoint" &&
	       stateName_ != "default" 
	       ) {
      bookHistosForState("default", ibooker);

    } else {
      bookHistosForState(stateName_, ibooker);
    }
    conf_ = nullptr;
  }
}

void TrackAnalyzer::bookHistosForEfficiencyFromHitPatter(DQMStore::IBooker &ibooker,
                                                         const edm::EventSetup & iSetup,
							 const std::string suffix,bool useInac)
{

    ibooker.setCurrentFolder(TopFolder_ + "/HitEffFromHitPattern" + (useInac ? "All" :"") + suffix);
    
    constexpr int LUMIBin   = 300;   // conf_->getParameter<int>("LUMIBin");
    float LUMIMin = conf_->getParameter<double>("LUMIMin");
    float LUMIMax = conf_->getParameter<double>("LUMIMax");
    
    int   PVBin = conf_->getParameter<int>   ("PVBin");
    float PVMin = conf_->getParameter<double>("PVMin");
    float PVMax = conf_->getParameter<double>("PVMax");


    int NBINS[]        = { PVBin,   int(GetLumi::lastBunchCrossing),  LUMIBin, LUMIBin};
    float MIN[]        = { PVMin,     0.5,  LUMIMin, LUMIMin };
    float MAX[]        = { PVMax, float(GetLumi::lastBunchCrossing)+0.5,  LUMIMax, LUMIMax };
    std::string NAME[] = { "", "VsBX", "VsLUMI", "VsLUMI" };
   
    auto logBins = makeLogBins<float,LUMIBin>(LUMIMin,LUMIMax);
 
    int mon = -1;
    int nbins = -1;
    float min = -1.;
    float max = -1.;
    bool logQ = false;
    std::string name = "";
    for (int i=0; i<monQuantity::END; i++) {
      if (monName[i] == suffix) {
        logQ =  (i>1); // VsLUMI
	mon = i;
        if (useInac) mon+=monQuantity::END;
	nbins = NBINS[i];
	min = MIN[i];
	max = MAX[i];
	name = NAME[i];
      }
    }
  
    edm::ESHandle<TrackerGeometry> trackerGeometry;
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

    // Values are not ordered randomly, but the order is taken from
    // http://cmslxr.fnal.gov/dxr/CMSSW/source/Geometry/CommonDetUnit/interface/GeomDetEnumerators.h#15
    const char * dets[] = { "None", "PXB", "PXF", "TIB", "TID", "TOB", "TEC"};

    // Also in this case, ordering is not random but extracted from
    // http://cmslxr.fnal.gov/dxr/CMSSW/source/DataFormats/TrackReco/interface/HitPattern.h
    // The category "total" is an addition to ease the computation of
    // the efficiencies and is not part of the original HitPattern.
    const char * hit_category[] = { "valid", "missing", "inactive", "bad", "total"};

    // We set sub_det to be a 1-based index since to it is the sub-sub-structure in the HitPattern
    char title[50];
    for (unsigned int det = 1; det < sizeof(dets)/sizeof(char*); ++det ) {
      for (unsigned int sub_det = 1;
           sub_det <= trackerGeometry->numberOfLayers(det); ++sub_det) {
        for (unsigned int cat = 0;
             cat < sizeof(hit_category)/sizeof(char *); ++cat) {
          memset(title, 0, sizeof(title));
          snprintf(title, sizeof(title), "Hits%s_%s_%s_Subdet%d", name.c_str(), hit_category[cat], dets[det], sub_det);
          switch(cat) {
            case 0:
              hits_valid_.insert(std::make_pair(
		  Key(det, sub_det, mon), logQ? 
                  ibooker.book1D(title, title, nbins, &logBins[0]) :
		  ibooker.book1D(title, title, nbins, min, max)));
              break;
            case 4:
              hits_total_.insert(std::make_pair(
		  Key(det, sub_det, mon), logQ?	
                  ibooker.book1D(title, title, nbins, &logBins[0]) :
                  ibooker.book1D(title, title, nbins, min, max)));
              break;
            default:
              LogDebug("TrackAnalyzer") << "Invalid hit category used " << cat << " ignored\n";
          }
        }
      }
    }
}

#include "DataFormats/TrackReco/interface/TrajectoryStopReasons.h"
void TrackAnalyzer::bookHistosForHitProperties(DQMStore::IBooker & ibooker) {
  
    // parameters from the configuration
    std::string QualName       = conf_->getParameter<std::string>("Quality");
    std::string AlgoName       = conf_->getParameter<std::string>("AlgoName");
    std::string MEBSFolderName = conf_->getParameter<std::string>("BSFolderName"); 

    // use the AlgoName and Quality Name 
    std::string CategoryName = !QualName.empty() ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    int    TKHitBin     = conf_->getParameter<int>(   "RecHitBin");
    double TKHitMin     = conf_->getParameter<double>("RecHitMin");
    double TKHitMax     = conf_->getParameter<double>("RecHitMax");

    int    TKLostBin    = conf_->getParameter<int>(   "RecLostBin");
    double TKLostMin    = conf_->getParameter<double>("RecLostMin");
    double TKLostMax    = conf_->getParameter<double>("RecLostMax");

    int    TKLayBin     = conf_->getParameter<int>(   "RecLayBin");
    double TKLayMin     = conf_->getParameter<double>("RecLayMin");
    double TKLayMax     = conf_->getParameter<double>("RecLayMax");

    int    PhiBin       = conf_->getParameter<int>(   "PhiBin");
    double PhiMin       = conf_->getParameter<double>("PhiMin");
    double PhiMax       = conf_->getParameter<double>("PhiMax");

    int    EtaBin       = conf_->getParameter<int>(   "EtaBin");
    double EtaMin       = conf_->getParameter<double>("EtaMin");
    double EtaMax       = conf_->getParameter<double>("EtaMax");

    int    PtBin        = conf_->getParameter<int>(   "TrackPtBin");
    double PtMin        = conf_->getParameter<double>("TrackPtMin");
    double PtMax        = conf_->getParameter<double>("TrackPtMax");

    int    Phi2DBin     = conf_->getParameter<int>(   "Phi2DBin");
    int    Eta2DBin     = conf_->getParameter<int>(   "Eta2DBin");
    int    Pt2DBin      = conf_->getParameter<int>(   "TrackPt2DBin");

    int    VXBin        = conf_->getParameter<int>(   "VXBin");
    double VXMin        = conf_->getParameter<double>("VXMin");
    double VXMax        = conf_->getParameter<double>("VXMax");

    int    VYBin        = conf_->getParameter<int>(   "VYBin");
    double VYMin        = conf_->getParameter<double>("VYMin");
    double VYMax        = conf_->getParameter<double>("VYMax");

    int    VZBin        = conf_->getParameter<int>(   "VZBin");
    double VZMin        = conf_->getParameter<double>("VZMin");
    double VZMax        = conf_->getParameter<double>("VZMax");

    ibooker.setCurrentFolder(TopFolder_);

    // book the Hit Property histograms
    // ---------------------------------------------------------------------------------//

    TkParameterMEs tkmes;
    if ( doHitPropertiesPlots_ || doAllPlots_ ){

      ibooker.setCurrentFolder(TopFolder_+"/HitProperties");
      
      histname = "NumberOfRecHitsPerTrack_";
      NumberOfRecHitsPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfValidRecHitsPerTrack_";
      NumberOfValidRecHitsPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfValidRecHitsPerTrack->setAxisTitle("Number of valid RecHits for each Track");

      NumberOfValidRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLostRecHitsPerTrack_";
      NumberOfLostRecHitsPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, TKLostBin, TKLostMin, TKLostMax);
      NumberOfLostRecHitsPerTrack->setAxisTitle("Number of lost RecHits for each Track");
      NumberOfLostRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfMissingInnerRecHitsPerTrack_";
      NumberOfMIRecHitsPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, 10, -0.5, 9.5);
      NumberOfMIRecHitsPerTrack->setAxisTitle("Number of missing-inner RecHits for each Track");
      NumberOfMIRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfMissingOuterRecHitsPerTrack_";
      NumberOfMORecHitsPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, 10, -0.5, 9.5);
      NumberOfMORecHitsPerTrack->setAxisTitle("Number of missing-outer RecHits for each Track");
      NumberOfMORecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "ValidFractionPerTrack_";
      ValidFractionPerTrack = ibooker.book1D(histname+CategoryName, histname+CategoryName, 101, 0., 1.01);
      ValidFractionPerTrack->setAxisTitle("ValidFraction of RecHits for each Track");
      ValidFractionPerTrack->setAxisTitle("Number of Tracks", 2);



      if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfValidRecHitVsPhiVsEtaPerTrack_";
	NumberOfValidRecHitVsPhiVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName, 
								    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 40., "");
	NumberOfValidRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfValidRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);

        histname = "NumberOfLostRecHitVsPhiVsEtaPerTrack_";
        NumberOfLostRecHitVsPhiVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 5., "");
        NumberOfLostRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfLostRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);


        histname = "NumberMIRecHitVsPhiVsEtaPerTrack_";
        NumberOfMIRecHitVsPhiVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 15., "");
        NumberOfMIRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfMIRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);

        histname = "NumberMORecHitVsPhiVsEtaPerTrack_";
        NumberOfMORecHitVsPhiVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 15., "");
        NumberOfMORecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfMORecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);

        histname = "ValidFractionVsPhiVsEtaPerTrack_";
        ValidFractionVsPhiVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 2., "");
        ValidFractionVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        ValidFractionVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);

      }

      if ( doRecHitVsPtVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfValidRecHitVsPtVsEtaPerTrack_";
	NumberOfValidRecHitVsPtVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName, 
								    Eta2DBin, EtaMin, EtaMax, Pt2DBin, PtMin, PtMax, 0, 40., "");
	NumberOfValidRecHitVsPtVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfValidRecHitVsPtVsEtaPerTrack->setAxisTitle("Track p_{T} [GeV] ", 2);

        histname = "NumberOfLostRecHitVsPtVsEtaPerTrack_";
        NumberOfLostRecHitVsPtVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Pt2DBin, PtMin, PtMax, 0, 5., "");
        NumberOfLostRecHitVsPtVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfLostRecHitVsPtVsEtaPerTrack->setAxisTitle("Track p_{T} [GeV] ", 2);


        histname = "NumberMIRecHitVsPtVsEtaPerTrack_";
        NumberOfMIRecHitVsPtVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Pt2DBin, PtMin, PtMax, 0, 15., "");
        NumberOfMIRecHitVsPtVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfMIRecHitVsPtVsEtaPerTrack->setAxisTitle("Track p_{T} [GeV] ", 2);

        histname = "NumberMORecHitVsPtVsEtaPerTrack_";
        NumberOfMORecHitVsPtVsEtaPerTrack = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName,
                                                                    Eta2DBin, EtaMin, EtaMax, Pt2DBin, PtMin, PtMax, 0, 15., "");
        NumberOfMORecHitVsPtVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
        NumberOfMORecHitVsPtVsEtaPerTrack->setAxisTitle("Track p_{T} [GeV] ", 2);

      }

      histname = "NumberOfValidRecHitsPerTrackVsPt_";
      NumberOfValidRecHitsPerTrackVsPt = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PtBin, PtMin, PtMax, TKHitMin, TKHitMax,"");
      NumberOfValidRecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]",1);
      NumberOfValidRecHitsPerTrackVsPt->setAxisTitle("Number of valid RecHits in each Track",2);
    
      histname = "NumberOfLostRecHitsPerTrackVsPt_";
      NumberOfLostRecHitsPerTrackVsPt = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PtBin, PtMin, PtMax,TKHitMin, TKHitMax,"");
      NumberOfLostRecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]", 1);
      NumberOfLostRecHitsPerTrackVsPt->setAxisTitle("Average Number of Lost RecHits per Track", 2);
      
      
      histname = "NumberMIRecHitsPerTrackVsPt_";
      NumberOfMIRecHitsPerTrackVsPt = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PtBin, PtMin, PtMax,TKHitMin, TKHitMax,"");
      NumberOfMIRecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]", 1);
      NumberOfMIRecHitsPerTrackVsPt->setAxisTitle("Average Number of Lost RecHits per Track", 2);
      
      histname = "NumberMORecHitsPerTrackVsPt_";
      NumberOfMORecHitsPerTrackVsPt = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PtBin, PtMin, PtMax,TKHitMin, TKHitMax,"");
      NumberOfMORecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]", 1);
      NumberOfMORecHitsPerTrackVsPt->setAxisTitle("Average Number of Lost RecHits per Track", 2);

      std::string layerTypeName[5] = {"","Off","3D","Missing","Pixel"};
      for (int i=0; i<4; ++i) {
        histname = "NumberOf"+ layerTypeName[i] + "LayersPerTrack_";
        NumberOfLayersPerTrack[i] = ibooker.book1D(histname+CategoryName, histname+CategoryName, TKLayBin, TKLayMin, TKLayMax);
        NumberOfLayersPerTrack[i]->setAxisTitle("Number of " + layerTypeName[i] + " Layers of each Track", 1);
        NumberOfLayersPerTrack[i]->setAxisTitle("Number of Tracks", 2);
      }
      if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ )
	for (int i=0; i<5; ++i) {
          histname = "NumberOf"+ layerTypeName[i] + "LayersVsPhiVsEtaPerTrack_";
	  NumberOfLayersVsPhiVsEtaPerTrack[i] = ibooker.bookProfile2D(histname+CategoryName, histname+CategoryName, 
								    Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax, 0, 40., "");
	  NumberOfLayersVsPhiVsEtaPerTrack[i]->setAxisTitle("Track #eta ", 1);
	  NumberOfLayersVsPhiVsEtaPerTrack[i]->setAxisTitle("Track #phi ", 2);
      }
    }

    // book the General Property histograms
    // ---------------------------------------------------------------------------------//
    
    if (doGeneralPropertiesPlots_ || doAllPlots_){
      
      int    Chi2Bin      = conf_->getParameter<int>(   "Chi2Bin");
      double Chi2Min      = conf_->getParameter<double>("Chi2Min");
      double Chi2Max      = conf_->getParameter<double>("Chi2Max");
      
      int    Chi2NDFBin   = conf_->getParameter<int>(   "Chi2NDFBin");
      double Chi2NDFMin   = conf_->getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax   = conf_->getParameter<double>("Chi2NDFMax");
      
      int    Chi2ProbBin  = conf_->getParameter<int>(   "Chi2ProbBin");
      double Chi2ProbMin  = conf_->getParameter<double>("Chi2ProbMin");
      double Chi2ProbMax  = conf_->getParameter<double>("Chi2ProbMax");
    

      //HI PLOTS////                                                       
      int TransDCABins = conf_->getParameter<int>("TransDCABins");
      double TransDCAMin = conf_->getParameter<double>("TransDCAMin");
      double TransDCAMax = conf_->getParameter<double>("TransDCAMax");

      int LongDCABins = conf_->getParameter<int>("LongDCABins");
      double LongDCAMin = conf_->getParameter<double>("LongDCAMin");
      double LongDCAMax = conf_->getParameter<double>("LongDCAMax");
      ///////////////////////////////////////////////////////////////////  


      ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");

      histname = "Chi2_";
      Chi2 = ibooker.book1D(histname+CategoryName, histname+CategoryName, Chi2Bin, Chi2Min, Chi2Max);
      Chi2->setAxisTitle("Track #chi^{2}"  ,1);
      Chi2->setAxisTitle("Number of Tracks",2);

      histname = "Chi2Prob_";
      Chi2Prob = ibooker.book1D(histname+CategoryName, histname+CategoryName, Chi2ProbBin, Chi2ProbMin, Chi2ProbMax);
      Chi2Prob->setAxisTitle("Track #chi^{2} probability",1);
      Chi2Prob->setAxisTitle("Number of Tracks"        ,2);
      
      histname = "Chi2oNDF_";
      Chi2oNDF = ibooker.book1D(histname+CategoryName, histname+CategoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF->setAxisTitle("Number of Tracks"  ,2);


      //////////////                                                                                                                                                                                       
      //HI PLOTS///                                                                                                                                                                                       
      //////////////                                                                                                                                                                                      
      if (doHIPlots_)
        {
          histname = "LongDCASig_";
          LongDCASig = ibooker.book1D(histname+CategoryName, histname+CategoryName,LongDCABins,LongDCAMin,LongDCAMax);
          LongDCASig->setAxisTitle("dz/#sigma_{dz}",1);

          histname = "TransDCASig_";
          TransDCASig = ibooker.book1D(histname+CategoryName,histname+CategoryName,TransDCABins,TransDCAMin,TransDCAMax);
          TransDCASig->setAxisTitle("dxy/#sigma_{dxy}",1);

	  histname = "dNdPhi_HighPurity_";
	  dNdPhi_HighPurity = ibooker.book1D(histname+CategoryName,histname+CategoryName,PhiBin,PhiMin,PhiMax);
	  dNdPhi_HighPurity->setAxisTitle("#phi",1);

	  histname = "dNdEta_HighPurity_";
          dNdEta_HighPurity = ibooker.book1D(histname+CategoryName,histname+CategoryName,EtaBin,EtaMin,EtaMax);
          dNdEta_HighPurity->setAxisTitle("#eta",1);

          histname = "dNdPt_HighPurity_";
          dNdPt_HighPurity = ibooker.book1D(histname+CategoryName,histname+CategoryName,150,0,0.3);
          dNdPt_HighPurity->setAxisTitle("#sigma_{p_{T}}/p_{T}",1);

	  histname = "NhitVsEta_HighPurity_";
	  NhitVsEta_HighPurity = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,EtaBin,EtaMin,EtaMax,-0.5,39.5,"");
	  NhitVsEta_HighPurity->setAxisTitle("Track #eta",1);
	  NhitVsEta_HighPurity->setAxisTitle("Number of Valid RecHits in each Track",2);

          histname = "NhitVsPhi_HighPurity_";
          NhitVsPhi_HighPurity = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,PhiBin,PhiMin,PhiMax,-0.5,39.5,"");
          NhitVsPhi_HighPurity->setAxisTitle("Track #phi",1);
          NhitVsPhi_HighPurity->setAxisTitle("Number of Valid RecHits in each Track",2);

	  histname = "Ptdist_HighPurity_";
          Ptdist_HighPurity = ibooker.book1D(histname+CategoryName,histname+CategoryName,150,0,50.);
          Ptdist_HighPurity->setAxisTitle("p_{T} (GeV/c)",1);
          Ptdist_HighPurity->setAxisTitle("Number of Tracks",2);

          histname = "dNhitdPt_HighPurity_";
          dNhitdPt_HighPurity = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,150,0,25.,-0.5,39.5,"");
          dNhitdPt_HighPurity->setAxisTitle("p_{T} (GeV/c)",1);
          dNhitdPt_HighPurity->setAxisTitle("N_{hit}",2);
	  
        }


      
      if(doDCAPlots_ || doPVPlots_ || doSIPPlots_ || doAllPlots_)  {
	histname = "xPointOfClosestApproach_";
	xPointOfClosestApproach = ibooker.book1D(histname+CategoryName, histname+CategoryName, VXBin, VXMin, VXMax);
	xPointOfClosestApproach->setAxisTitle("x component of Track PCA to beam line (cm)",1);
	xPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "yPointOfClosestApproach_";
	yPointOfClosestApproach = ibooker.book1D(histname+CategoryName, histname+CategoryName, VYBin, VYMin, VYMax);
	yPointOfClosestApproach->setAxisTitle("y component of Track PCA to beam line (cm)",1);
	yPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "zPointOfClosestApproach_";
	zPointOfClosestApproach = ibooker.book1D(histname+CategoryName, histname+CategoryName, VZBin, VZMin, VZMax);
	zPointOfClosestApproach->setAxisTitle("z component of Track PCA to beam line (cm)",1);
	zPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "xPointOfClosestApproachToPV_";
        xPointOfClosestApproachToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, VXBin, VXMin, VXMax);
	xPointOfClosestApproachToPV->setAxisTitle("x component of Track PCA to pv (cm)",1);
	xPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
	
	histname = "yPointOfClosestApproachToPV_";
	yPointOfClosestApproachToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, VYBin, VYMin, VYMax);
	yPointOfClosestApproachToPV->setAxisTitle("y component of Track PCA to pv line (cm)",1);
	yPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
	
	histname = "zPointOfClosestApproachToPV_";
	zPointOfClosestApproachToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, VZBin, VZMin, VZMax);
	zPointOfClosestApproachToPV->setAxisTitle("z component of Track PCA to pv line (cm)",1);
	zPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
      }

      // See DataFormats/TrackReco/interface/TrackBase.h for track algorithm enum definition
      // http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DataFormats/TrackReco/interface/TrackBase.h?view=log
      histname = "algorithm_";
      algorithm = ibooker.book1D(histname+CategoryName, histname+CategoryName, reco::TrackBase::algoSize, 0., double(reco::TrackBase::algoSize));
      algorithm->setAxisTitle("Tracking algorithm",1);
      algorithm->setAxisTitle("Number of Tracks",2);

      histname = "originalAlgorithm_";
      oriAlgo = ibooker.book1D(histname+CategoryName, histname+CategoryName, reco::TrackBase::algoSize, 0., double(reco::TrackBase::algoSize));
      oriAlgo->setAxisTitle("Tracking algorithm",1);
      oriAlgo->setAxisTitle("Number of Tracks",2);

      for (size_t ibin=0; ibin<reco::TrackBase::algoSize-1; ibin++) {
	algorithm->setBinLabel(ibin+1,reco::TrackBase::algoNames[ibin]);
        oriAlgo->setBinLabel(ibin+1,reco::TrackBase::algoNames[ibin]);
      }

      size_t StopReasonNameSize = sizeof(StopReasonName::StopReasonName)/sizeof(std::string);
      histname = "stoppingSource_";
      stoppingSource = ibooker.book1D(histname+CategoryName, histname+CategoryName, StopReasonNameSize, 0., double(StopReasonNameSize));
      stoppingSource->setAxisTitle("stopping reason",1);
      stoppingSource->setAxisTitle("Number of Tracks",2);
      
      histname = "stoppingSourceVSeta_";
      stoppingSourceVSeta = ibooker.bookProfile(histname+CategoryName, histname+CategoryName,
                                            EtaBin, EtaMin, EtaMax, 2, 0., 2.);
      stoppingSourceVSeta->setAxisTitle("track #eta",1);
      stoppingSourceVSeta->setAxisTitle("stopped fraction",2);
      
      histname = "stoppingSourceVSphi_";
      stoppingSourceVSphi = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, 
                                           PhiBin, PhiMin, PhiMax, 2, 0., 2.);
      stoppingSourceVSphi->setAxisTitle("track #phi",1);
      stoppingSourceVSphi->setAxisTitle("stopped fraction",2);

      for (size_t ibin=0; ibin<StopReasonNameSize; ibin++) {
	stoppingSource->setBinLabel(ibin+1,StopReasonName::StopReasonName[ibin],1);
      }

    }

}

void TrackAnalyzer::bookHistosForLScertification(DQMStore::IBooker & ibooker) {

    // parameters from the configuration
    std::string QualName       = conf_->getParameter<std::string>("Quality");
    std::string AlgoName       = conf_->getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = !QualName.empty() ? AlgoName + "_" + QualName : AlgoName;


    // book LS analysis related histograms
    // -----------------------------------
    if ( doLumiAnalysis_ ) {

      // get binning from the configuration
      int    TKHitBin     = conf_->getParameter<int>(   "RecHitBin");
      double TKHitMin     = conf_->getParameter<double>("RecHitMin");
      double TKHitMax     = conf_->getParameter<double>("RecHitMax");

      int    Chi2NDFBin   = conf_->getParameter<int>(   "Chi2NDFBin");
      double Chi2NDFMin   = conf_->getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax   = conf_->getParameter<double>("Chi2NDFMax");

      // add by Mia in order to deal w/ LS transitions  
      ibooker.setCurrentFolder(TopFolder_+"/LSanalysis");

      histname = "NumberOfRecHitsPerTrack_lumiFlag_";
      NumberOfRecHitsPerTrack_lumiFlag = ibooker.book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of Tracks", 2);

      histname = "Chi2oNDF_lumiFlag_";
      Chi2oNDF_lumiFlag = ibooker.book1D(histname+CategoryName, histname+CategoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF_lumiFlag->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF_lumiFlag->setAxisTitle("Number of Tracks"  ,2);

    }
}

void TrackAnalyzer::bookHistosForBeamSpot(DQMStore::IBooker & ibooker) {

    // parameters from the configuration
    std::string QualName       = conf_->getParameter<std::string>("Quality");
    std::string AlgoName       = conf_->getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = !QualName.empty() ? AlgoName + "_" + QualName : AlgoName;

    // book the Beam Spot related histograms
    // ---------------------------------------------------------------------------------//
    
    if(doDCAPlots_ || doBSPlots_ || doAllPlots_) {
	
      int    DxyErrBin    = conf_->getParameter<int>(   "DxyErrBin");
      double DxyErrMax    = conf_->getParameter<double>("DxyErrMax");
      
      int    DxyBin       = conf_->getParameter<int>(   "DxyBin");
      double DxyMin       = conf_->getParameter<double>("DxyMin");
      double DxyMax       = conf_->getParameter<double>("DxyMax");
      
      int    AbsDxyBin    = conf_->getParameter<int>(   "AbsDxyBin");
      double AbsDxyMin    = conf_->getParameter<double>("AbsDxyMin");
      double AbsDxyMax    = conf_->getParameter<double>("AbsDxyMax");
      
      int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
      double PhiMin     = conf_->getParameter<double>("PhiMin");
      double PhiMax     = conf_->getParameter<double>("PhiMax");
      
      int    EtaBin     = conf_->getParameter<int>(   "EtaBin");
      double EtaMin     = conf_->getParameter<double>("EtaMin");
      double EtaMax     = conf_->getParameter<double>("EtaMax");

      int    PtBin      = conf_->getParameter<int>(   "TrackPtBin");
      double PtMin      = conf_->getParameter<double>("TrackPtMin");
      double PtMax      = conf_->getParameter<double>("TrackPtMax");

      int    X0Bin        = conf_->getParameter<int>(   "X0Bin");
      double X0Min        = conf_->getParameter<double>("X0Min");
      double X0Max        = conf_->getParameter<double>("X0Max");
      
      int    Y0Bin        = conf_->getParameter<int>(   "Y0Bin");
      double Y0Min        = conf_->getParameter<double>("Y0Min");
      double Y0Max        = conf_->getParameter<double>("Y0Max");
      
      int    Z0Bin        = conf_->getParameter<int>(   "Z0Bin");
      double Z0Min        = conf_->getParameter<double>("Z0Min");
      double Z0Max        = conf_->getParameter<double>("Z0Max");
      
      int    VZBinProf    = conf_->getParameter<int>(   "VZBinProf");
      double VZMinProf    = conf_->getParameter<double>("VZMinProf");
      double VZMaxProf    = conf_->getParameter<double>("VZMaxProf");
      
      
      ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      histname = "DistanceOfClosestApproachError_";
      DistanceOfClosestApproachError = ibooker.book1D(histname+CategoryName,histname+CategoryName,DxyErrBin,0.,DxyErrMax);
      DistanceOfClosestApproachError->setAxisTitle("Track d_{xy} error (cm)",1);
      DistanceOfClosestApproachError->setAxisTitle("Number of Tracks",2);
      
      histname = "DistanceOfClosestApproachErrorVsPt_";
      DistanceOfClosestApproachErrorVsPt = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,PtBin,PtMin,PtMax,0.,DxyErrMax);
      DistanceOfClosestApproachErrorVsPt->setAxisTitle("Track p_{T} (GeV)",1);
      DistanceOfClosestApproachErrorVsPt->setAxisTitle("Track d_{xy} error (cm)",2);
      
      histname = "DistanceOfClosestApproachErrorVsEta_";
      DistanceOfClosestApproachErrorVsEta = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,EtaBin,EtaMin,EtaMax,0.,DxyErrMax);
      DistanceOfClosestApproachErrorVsEta->setAxisTitle("Track #eta",1);
      DistanceOfClosestApproachErrorVsEta->setAxisTitle("Track d_{xy} error (cm)",2);
      
      histname = "DistanceOfClosestApproachErrorVsPhi_";
      DistanceOfClosestApproachErrorVsPhi = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,PhiBin,PhiMin,PhiMax,0.,DxyErrMax);
      DistanceOfClosestApproachErrorVsPhi->setAxisTitle("Track #phi",1);
      DistanceOfClosestApproachErrorVsPhi->setAxisTitle("Track d_{xy} error (cm)",2);
      
      histname = "DistanceOfClosestApproachErrorVsDxy_";
      DistanceOfClosestApproachErrorVsDxy = ibooker.bookProfile(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax,0.,DxyErrMax);
      DistanceOfClosestApproachErrorVsDxy->setAxisTitle("Track d_{xy}",1);
      DistanceOfClosestApproachErrorVsDxy->setAxisTitle("Track d_{xy} error (cm)",2);

      histname = "DistanceOfClosestApproachToBS_";
      DistanceOfClosestApproachToBS = ibooker.book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
      DistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
      DistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
      
      histname = "AbsDistanceOfClosestApproachToBS_";
      AbsDistanceOfClosestApproachToBS = ibooker.book1D(histname+CategoryName,histname+CategoryName,AbsDxyBin,AbsDxyMin,AbsDxyMax);
      AbsDistanceOfClosestApproachToBS->setAxisTitle("Track |d_{xy}| wrt beam spot (cm)",1);
      AbsDistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
      
      histname = "DistanceOfClosestApproachToBSVsPhi_";
      DistanceOfClosestApproachToBSVsPhi = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
      DistanceOfClosestApproachToBSVsPhi->getTH1()->SetCanExtend(TH1::kAllAxes);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrt000_";
      xPointOfClosestApproachVsZ0wrt000 = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
      xPointOfClosestApproachVsZ0wrt000->setAxisTitle("x component of Track PCA to beam line (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrt000_";
      yPointOfClosestApproachVsZ0wrt000 = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
      yPointOfClosestApproachVsZ0wrt000->setAxisTitle("y component of Track PCA to beam line (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrtBS_";
      xPointOfClosestApproachVsZ0wrtBS = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot  (cm)",1);
      xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("x component of Track PCA to BS (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrtBS_";
      yPointOfClosestApproachVsZ0wrtBS = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot (cm)",1);
      yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("y component of Track PCA to BS (cm)",2);
      
      histname = "zPointOfClosestApproachVsPhi_";
      zPointOfClosestApproachVsPhi = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, VZBinProf, VZMinProf, VZMaxProf, "");
      zPointOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
      zPointOfClosestApproachVsPhi->setAxisTitle("z component of Track PCA to beam line (cm)",2);
    }
    
    if(doDCAPlots_ || doPVPlots_ || doAllPlots_) {
      
      int    DxyBin       = conf_->getParameter<int>(   "DxyBin");
      double DxyMin       = conf_->getParameter<double>("DxyMin");
      double DxyMax       = conf_->getParameter<double>("DxyMax");
      
      int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
      double PhiMin     = conf_->getParameter<double>("PhiMin");
      double PhiMax     = conf_->getParameter<double>("PhiMax");
      
      int    X0Bin        = conf_->getParameter<int>(   "X0Bin");
      double X0Min        = conf_->getParameter<double>("X0Min");
      double X0Max        = conf_->getParameter<double>("X0Max");
      
      int    Y0Bin        = conf_->getParameter<int>(   "Y0Bin");
      double Y0Min        = conf_->getParameter<double>("Y0Min");
      double Y0Max        = conf_->getParameter<double>("Y0Max");
      
      int    Z0Bin        = conf_->getParameter<int>(   "Z0Bin");
      double Z0Min        = conf_->getParameter<double>("Z0Min");
      double Z0Max        = conf_->getParameter<double>("Z0Max");
      
      ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      histname = "DistanceOfClosestApproachToPV_";
      DistanceOfClosestApproachToPV = ibooker.book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
      DistanceOfClosestApproachToPV->setAxisTitle("Track d_{xy} w.r.t. PV (cm)",1);
      DistanceOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
      
      histname = "DistanceOfClosestApproachToPVZoom_";
      DistanceOfClosestApproachToPVZoom = ibooker.book1D(histname+CategoryName,histname+CategoryName,100,-0.08,0.08);
      DistanceOfClosestApproachToPVZoom->setAxisTitle("Track d_{xy} w.r.t. PV (cm)",1);
      DistanceOfClosestApproachToPVZoom->setAxisTitle("Number of Tracks",2);


      histname = "DeltaZToPV_";
      DeltaZToPV = ibooker.book1D(histname+CategoryName,histname+CategoryName,Z0Bin,Z0Min,Z0Max);
      DeltaZToPV->setAxisTitle("Track d_{z} w.r.t. PV (cm)",1);
      DeltaZToPV->setAxisTitle("Number of Tracks",2);

      histname = "DeltaZToPVZoom_";
      DeltaZToPVZoom = ibooker.book1D(histname+CategoryName,histname+CategoryName,100,-0.15,0.15);
      DeltaZToPVZoom->setAxisTitle("Track d_{z} w.r.t. PV (cm)",1);
      DeltaZToPVZoom->setAxisTitle("Number of Tracks",2);

      
      histname = "DistanceOfClosestApproachToPVVsPhi_";
      DistanceOfClosestApproachToPVVsPhi = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
      DistanceOfClosestApproachToPVVsPhi->getTH1()->SetCanExtend(TH1::kAllAxes);
      DistanceOfClosestApproachToPVVsPhi->setAxisTitle("Track #phi",1);
      DistanceOfClosestApproachToPVVsPhi->setAxisTitle("Track d_{xy} w.r.t. PV (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrtPV_";
      xPointOfClosestApproachVsZ0wrtPV = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrtPV->setAxisTitle("d_{z} w.r.t. PV (cm)",1);
      xPointOfClosestApproachVsZ0wrtPV->setAxisTitle("x component of Track PCA to PV (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrtPV_";
      yPointOfClosestApproachVsZ0wrtPV = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrtPV->setAxisTitle("d_{z} w.r.t. PV (cm)",1);
      yPointOfClosestApproachVsZ0wrtPV->setAxisTitle("y component of Track PCA to PV (cm)",2);
      
    }

    if (doBSPlots_ || doAllPlots_) {
      if (doTestPlots_) {
	
	int    DxyBin       = conf_->getParameter<int>(   "DxyBin");
	double DxyMin       = conf_->getParameter<double>("DxyMin");
	double DxyMax       = conf_->getParameter<double>("DxyMax");
      
	int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
	double PhiMin     = conf_->getParameter<double>("PhiMin");
	double PhiMax     = conf_->getParameter<double>("PhiMax");

	histname = "TESTDistanceOfClosestApproachToBS_";
	TESTDistanceOfClosestApproachToBS = ibooker.book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
	TESTDistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
	TESTDistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
	
	histname = "TESTDistanceOfClosestApproachToBSVsPhi_";
	TESTDistanceOfClosestApproachToBSVsPhi = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
	TESTDistanceOfClosestApproachToBSVsPhi->getTH1()->SetCanExtend(TH1::kAllAxes);
	TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
	TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
	
      }
      
    }
    
    // book the Profile plots for DCA related histograms
    // ---------------------------------------------------------------------------------//
    if(doDCAPlots_ || doAllPlots_) {

      if (doDCAwrt000Plots_) {

	int    EtaBin     = conf_->getParameter<int>(   "EtaBin");
	double EtaMin     = conf_->getParameter<double>("EtaMin");
	double EtaMax     = conf_->getParameter<double>("EtaMax");
	
	int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
	double PhiMin     = conf_->getParameter<double>("PhiMin");
	double PhiMax     = conf_->getParameter<double>("PhiMax");

	int    DxyBin       = conf_->getParameter<int>(   "DxyBin");
	double DxyMin       = conf_->getParameter<double>("DxyMin");
	double DxyMax       = conf_->getParameter<double>("DxyMax");
      
	if (doThetaPlots_) {
	  int    ThetaBin   = conf_->getParameter<int>(   "ThetaBin");
	  double ThetaMin   = conf_->getParameter<double>("ThetaMin");
	  double ThetaMax   = conf_->getParameter<double>("ThetaMax");
	  
	  ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");
	  histname = "DistanceOfClosestApproachVsTheta_";
	  DistanceOfClosestApproachVsTheta = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, ThetaBin, ThetaMin, ThetaMax, DxyMin,DxyMax,"");
	  DistanceOfClosestApproachVsTheta->setAxisTitle("Track #theta",1);
	  DistanceOfClosestApproachVsTheta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	}
	
	histname = "DistanceOfClosestApproachVsEta_";
	DistanceOfClosestApproachVsEta = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, EtaBin, EtaMin, EtaMax, DxyMin, DxyMax,"");
	DistanceOfClosestApproachVsEta->setAxisTitle("Track #eta",1);
	DistanceOfClosestApproachVsEta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	// temporary patch in order to put back those MEs in Muon Workspace
	
	histname = "DistanceOfClosestApproach_";
	DistanceOfClosestApproach = ibooker.book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
	DistanceOfClosestApproach->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",1);
	DistanceOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "DistanceOfClosestApproachVsPhi_";
	DistanceOfClosestApproachVsPhi = ibooker.bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyMin,DxyMax,"");
	DistanceOfClosestApproachVsPhi->getTH1()->SetCanExtend(TH1::kAllAxes);
	DistanceOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
	DistanceOfClosestApproachVsPhi->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
      }
    }


    if (doSIPPlots_ || doAllPlots_) {
      const double sipBins = 200;
      const double sipMin = -20;
      const double sipMax = 20;

      ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");

      // SIP wrt. beamspot
      histname = "SIPDxyToBS_";
      sipDxyToBS = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sipDxyToBS->setAxisTitle("Track dxy significance wrt beam spot",1);
      sipDxyToBS->setAxisTitle("Number of Tracks",2);

      histname = "SIPDzToBS_";
      sipDzToBS = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sipDzToBS->setAxisTitle("Track dz significance wrt beam spot",1);
      sipDzToBS->setAxisTitle("Number of Tracks",2);

      // SIP wrt. vertex
      histname = "SIP3DToPV_";
      sip3dToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sip3dToPV->setAxisTitle("3D IP significance wrt primary vertex",1);
      sip3dToPV->setAxisTitle("Number of Tracks",2);

      histname = "SIP2DToPV_";
      sip2dToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sip2dToPV->setAxisTitle("2D IP significance wrt primary vertex",1);
      sip2dToPV->setAxisTitle("Number of Tracks",2);

      histname = "SIPDxyToPV_";
      sipDxyToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sipDxyToPV->setAxisTitle("Track dxy significance wrt primary vertex",1);
      sipDxyToPV->setAxisTitle("Number of Tracks",2);

      histname = "SIPDzToPV_";
      sipDzToPV = ibooker.book1D(histname+CategoryName, histname+CategoryName, sipBins, sipMin, sipMax);
      sipDzToPV->setAxisTitle("Track dz significance wrt primary vertex",1);
      sipDzToPV->setAxisTitle("Number of Tracks",2);
    }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::setNumberOfGoodVertices(const edm::Event & iEvent) {

  good_vertices_ = 0;
  
  edm::Handle<reco::VertexCollection> recoPrimaryVerticesHandle;
  iEvent.getByToken(pvToken_, recoPrimaryVerticesHandle);
  if (recoPrimaryVerticesHandle.isValid())
    if (!recoPrimaryVerticesHandle->empty())
      for (auto v : *recoPrimaryVerticesHandle)
        if (v.ndof() >= pvNDOF_ && !v.isFake())
          ++good_vertices_;
}

void TrackAnalyzer::setBX(const edm::Event & iEvent) {
  bx_ = iEvent.bunchCrossing();
}

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
void TrackAnalyzer::setLumi(const edm::Event & iEvent, const edm::EventSetup& iSetup) {
  // as done by pixelLumi http://cmslxr.fnal.gov/source/DQM/PixelLumi/plugins/PixelLumiDQM.cc

  edm::Handle<LumiScalersCollection> lumiScalers;
  iEvent.getByToken(lumiscalersToken_, lumiScalers);
  if ( lumiScalers.isValid() && !lumiScalers->empty() ) {
    LumiScalersCollection::const_iterator scalit = lumiScalers->begin();
    scal_lumi_ = scalit->instantLumi();
  } else 
    scal_lumi_ = -1;

  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
  iEvent.getByToken(pixelClustersToken_, pixelClusters);
  if ( pixelClusters.isValid() ) {

    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // Count the number of clusters with at least a minimum
    // number of pixels per cluster and at least a minimum charge.
    size_t numClusters = 0;
    size_t tot = 0;

    edmNew::DetSetVector<SiPixelCluster>::const_iterator  pixCluDet = pixelClusters->begin();
    for ( ; pixCluDet!=pixelClusters->end(); ++pixCluDet) {
    
      DetId detid = pixCluDet->detId();
      size_t subdetid = detid.subdetId();
      //      std::cout << tTopo->print(detid) << std::endl;
      if ( subdetid == (int) PixelSubdetector::PixelBarrel ) 
	if ( tTopo->layer(detid)==1 ) 
	  continue;
      
      edmNew::DetSet<SiPixelCluster>::const_iterator  pixClu = pixCluDet->begin();    
      for ( ; pixClu != pixCluDet->end(); ++pixClu ) {
	++tot;
	if ( (pixClu->size()   >= minNumberOfPixelsPerCluster_) &&
	     (pixClu->charge() >= minPixelClusterCharge_      ) ) {
	  ++numClusters;
	}
      }
    }
    pixel_lumi_ = lumi_factor_per_bx_ * numClusters / GetLumi::CM2_TO_NANOBARN ; // ?!?!
  } else
    pixel_lumi_ = -1.;

}

void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track)
{
  auto pt    = track.pt();
  auto phi   = track.phi();
  // double eta   = track.eta();
  auto phiIn =  track.innerPosition().phi();
  auto etaIn =  track.innerPosition().eta();
  auto phiOut =  track.outerPosition().phi();
  auto etaOut =  track.outerPosition().eta();

  int nRecHits      = track.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS);
  int nValidRecHits = track.numberOfValidHits();
  int nLostRecHits  = track.numberOfLostHits();
  int nLostIn =      track.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
  int nLostOut =     track.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);

  auto chi2     = track.chi2();
  auto chi2prob = TMath::Prob(track.chi2(),(int)track.ndof());
  auto chi2oNDF = track.normalizedChi2();
  
  if ( doHitPropertiesPlots_ || doAllPlots_ ){
    // rec hits
    NumberOfRecHitsPerTrack     -> Fill(nRecHits);
    NumberOfValidRecHitsPerTrack-> Fill(nValidRecHits);
    NumberOfLostRecHitsPerTrack -> Fill(nLostRecHits);
    NumberOfMIRecHitsPerTrack -> Fill(nLostIn);
    NumberOfMORecHitsPerTrack -> Fill(nLostOut);
    ValidFractionPerTrack -> Fill(track.validFraction());


    // 2D plots    
    if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ) {
      NumberOfValidRecHitVsPhiVsEtaPerTrack->Fill(etaIn,phiIn,nValidRecHits);
      NumberOfLostRecHitVsPhiVsEtaPerTrack->Fill(etaIn,phiIn,nLostRecHits);
      NumberOfMIRecHitVsPhiVsEtaPerTrack->Fill(etaIn,phiIn,nLostIn);
      NumberOfMORecHitVsPhiVsEtaPerTrack->Fill(etaOut,phiOut,nLostOut);
      ValidFractionVsPhiVsEtaPerTrack -> Fill(etaIn,phiIn,track.validFraction());
    }
    if ( doRecHitVsPtVsEtaPerTrack_ || doAllPlots_ ) {
      NumberOfValidRecHitVsPtVsEtaPerTrack->Fill(etaIn,pt,nValidRecHits);
      NumberOfLostRecHitVsPtVsEtaPerTrack->Fill(etaIn,pt,nLostRecHits);
      NumberOfMIRecHitVsPtVsEtaPerTrack->Fill(etaIn,pt,nLostIn);
      NumberOfMORecHitVsPtVsEtaPerTrack->Fill(etaOut,pt,nLostOut);
    }
    NumberOfValidRecHitsPerTrackVsPt->Fill(pt,nValidRecHits);
    NumberOfLostRecHitsPerTrackVsPt->Fill(pt,nLostRecHits);
    NumberOfMIRecHitsPerTrackVsPt->Fill(pt,nLostIn);
    NumberOfMORecHitsPerTrackVsPt->Fill(pt,nLostOut);

    int nLayers[5]   = { track.hitPattern().trackerLayersWithMeasurement(),
                         track.hitPattern().trackerLayersTotallyOffOrBad(),
                         track.hitPattern().numberOfValidStripLayersWithMonoAndStereo() +  track.hitPattern().pixelLayersWithMeasurement(),
                         track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS),
                         track.hitPattern().pixelLayersWithMeasurement()
                       };

    // layers
    for (int i=0;i<4;++i) NumberOfLayersPerTrack[i]->Fill(nLayers[i]);

    // 2D plots    
    if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ )
      for (int i=0;i<5;++i) NumberOfLayersVsPhiVsEtaPerTrack[i]->Fill(etaIn,phiIn,nLayers[i]);

  }

  if (doEffFromHitPatternVsPU_   || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"",           float(good_vertices_), false );
  if (doEffFromHitPatternVsBX_   || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"VsBX",       float(bx_), false            );
  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"VsSCALLUMI", scal_lumi_, false            );
  //  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"VsPIXELLUMI", pixel_lumi_           );
  if (doEffFromHitPatternVsPU_   || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"",           float(good_vertices_), true );
  if (doEffFromHitPatternVsLUMI_ || doAllPlots_) fillHistosForEfficiencyFromHitPatter(track,"VsSCALLUMI", scal_lumi_, true            );


  if (doGeneralPropertiesPlots_ || doAllPlots_){
    // fitting
    Chi2     -> Fill(chi2);
    Chi2Prob -> Fill(chi2prob);
    Chi2oNDF -> Fill(chi2oNDF);

    // DCA
    // temporary patch in order to put back those MEs in Muon Workspace 
    if (doDCAPlots_) {
      if (doDCAwrt000Plots_) {
	DistanceOfClosestApproach->Fill(track.dxy());
	DistanceOfClosestApproachVsPhi->Fill(phi, track.dxy());
      }

      // PCA
      xPointOfClosestApproach->Fill(track.referencePoint().x());
      yPointOfClosestApproach->Fill(track.referencePoint().y());
      zPointOfClosestApproach->Fill(track.referencePoint().z());
    }

    // algorithm
    algorithm->Fill(static_cast<double>(track.algo()));
    oriAlgo->Fill(static_cast<double>(track.originalAlgo()));

    // stopping source
    int max = stoppingSource->getNbinsX();
    double stop = track.stopReason() > max ? double(max-1) : static_cast<double>(track.stopReason());
    double stopped = int(StopReason::NOT_STOPPED)==track.stopReason() ? 0. : 1.;
    stoppingSource->Fill(stop);
    stoppingSourceVSeta->Fill(track.eta(),stopped);
    stoppingSourceVSphi->Fill(track.phi(),stopped);
  }

  if ( doLumiAnalysis_ ) {
    NumberOfRecHitsPerTrack_lumiFlag -> Fill(nRecHits);
    Chi2oNDF_lumiFlag                -> Fill(chi2oNDF);
  }

  if(doDCAPlots_ || doBSPlots_ || doSIPPlots_ || doAllPlots_) {
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
    const reco::BeamSpot& bs = *recoBeamSpotHandle;

    DistanceOfClosestApproachError      -> Fill(track.dxyError());
    DistanceOfClosestApproachErrorVsPt  -> Fill(track.pt(),track.dxyError());
    DistanceOfClosestApproachErrorVsEta -> Fill(track.eta(),track.dxyError());
    DistanceOfClosestApproachErrorVsPhi -> Fill(track.phi(),track.dxyError());
    DistanceOfClosestApproachErrorVsDxy -> Fill(track.dxy(bs.position()),track.dxyError());

    DistanceOfClosestApproachToBS      -> Fill(track.dxy(bs.position()));
    AbsDistanceOfClosestApproachToBS   -> Fill(std::abs(track.dxy(bs.position())));
    DistanceOfClosestApproachToBSVsPhi -> Fill(track.phi(), track.dxy(bs.position()));
    zPointOfClosestApproachVsPhi       -> Fill(track.phi(), track.vz());
    xPointOfClosestApproachVsZ0wrt000  -> Fill(track.dz(),  track.vx());
    yPointOfClosestApproachVsZ0wrt000  -> Fill(track.dz(),  track.vy());
    xPointOfClosestApproachVsZ0wrtBS   -> Fill(track.dz(bs.position()),(track.vx()-bs.position(track.vz()).x()));
    yPointOfClosestApproachVsZ0wrtBS   -> Fill(track.dz(bs.position()),(track.vy()-bs.position(track.vz()).y()));
    if (doTestPlots_) {
      TESTDistanceOfClosestApproachToBS      -> Fill(track.dxy(bs.position(track.vz())));
      TESTDistanceOfClosestApproachToBSVsPhi -> Fill(track.phi(), track.dxy(bs.position(track.vz())));
    }

    if(doSIPPlots_) {
      sipDxyToBS->Fill(track.dxy(bs.position())/track.dxyError());
      sipDzToBS->Fill(track.dz(bs.position())/track.dzError());
    }
  }
  
  if(doDCAPlots_ || doPVPlots_ || doSIPPlots_ || doAllPlots_) {
    edm::Handle<reco::VertexCollection> recoPrimaryVerticesHandle;
    iEvent.getByToken(pvToken_,recoPrimaryVerticesHandle);
    if (recoPrimaryVerticesHandle.isValid() && !recoPrimaryVerticesHandle->empty()) {
      const reco::Vertex& pv = (*recoPrimaryVerticesHandle)[0];
    

      //////////////////
      //HI PLOTS/////// 
      //////////////// 

      if(doHIPlots_)
	{
	   double longDCAsig = 0, transDCAsig = 0;
	   double zerr2 = track.dzError()*track.dzError()+pv.zError()*pv.zError();
	   double xyerr2 = track.d0Error()*track.d0Error()+pv.xError()*pv.yError();
	   if(zerr2 > 0) longDCAsig = track.dz(pv.position())/zerr2;
           if(xyerr2 > 0) transDCAsig = track.dxy(pv.position())/xyerr2;	   
	   LongDCASig->Fill(longDCAsig);
	   TransDCASig->Fill(transDCAsig);


	   

	   if(track.quality(reco::TrackBase::qualityByName(qualityString_)) ==1)
	     {
	       dNdEta_HighPurity->Fill(track.eta());
	       dNdPhi_HighPurity->Fill(track.phi());
	       dNdPt_HighPurity->Fill(track.ptError()/track.pt());
	       NhitVsEta_HighPurity->Fill(track.eta(),track.numberOfValidHits());
	       NhitVsPhi_HighPurity->Fill(track.phi(),track.numberOfValidHits());
	       dNhitdPt_HighPurity->Fill(track.pt(),track.numberOfValidHits());
	       Ptdist_HighPurity->Fill(track.pt());
	     }//end of high quality tracks requirement
        }


      xPointOfClosestApproachToPV->Fill(track.vx()-pv.position().x());
      yPointOfClosestApproachToPV->Fill(track.vy()-pv.position().y());
      zPointOfClosestApproachToPV->Fill(track.dz(pv.position()));
      DistanceOfClosestApproachToPV      -> Fill(track.dxy(pv.position()));
      DeltaZToPV                         -> Fill(track.dz (pv.position()));
      DistanceOfClosestApproachToPVZoom  -> Fill(track.dxy(pv.position()));
      DeltaZToPVZoom                     -> Fill(track.dz (pv.position()));
      DistanceOfClosestApproachToPVVsPhi -> Fill(track.phi(), track.dxy(pv.position()));
      xPointOfClosestApproachVsZ0wrtPV   -> Fill(track.dz(pv.position()),(track.vx()-pv.position().x()));
      yPointOfClosestApproachVsZ0wrtPV   -> Fill(track.dz(pv.position()),(track.vy()-pv.position().y()));


      if(doSIPPlots_) {
        edm::ESHandle<TransientTrackBuilder> theB;
        iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
        reco::TransientTrack transTrack = theB->build(track);

        GlobalVector dir(track.px(), track.py(), track.pz());
        std::pair<bool, Measurement1D> ip3d = IPTools::signedImpactParameter3D(transTrack, dir, pv);
        std::pair<bool, Measurement1D> ip2d = IPTools::signedTransverseImpactParameter(transTrack, dir, pv);
        if(ip3d.first) sip3dToPV->Fill(ip3d.second.value() / ip3d.second.error());
        if(ip2d.first) sip2dToPV->Fill(ip2d.second.value() / ip2d.second.error());
        sipDxyToPV->Fill(track.dxy(pv.position())/track.dxyError());
        sipDzToPV->Fill(track.dz(pv.position())/track.dzError());
      }
    }
  }

  if(doDCAPlots_ || doAllPlots_) {
    if (doDCAwrt000Plots_) {
      if (doThetaPlots_) {
	DistanceOfClosestApproachVsTheta->Fill(track.theta(), track.d0());
      }
      DistanceOfClosestApproachVsEta->Fill(track.eta(), track.d0());
    }  
    
  }

  //Tracker Specific Histograms
  if(doTrackerSpecific_ || doAllPlots_) {
    fillHistosForTrackerSpecific(track);
  }

  if (doMeasurementStatePlots_ || doAllPlots_){

    if (stateName_ == "All") {
      fillHistosForState(iSetup, track, std::string("OuterSurface"));
      fillHistosForState(iSetup, track, std::string("InnerSurface"));
      fillHistosForState(iSetup, track, std::string("ImpactPoint"));
    } else if ( 
	       stateName_ != "OuterSurface" && 
	       stateName_ != "InnerSurface" && 
	       stateName_ != "ImpactPoint" &&
	       stateName_ != "default" 
	       ) {
      fillHistosForState(iSetup, track, std::string("default"));
    } else {
      fillHistosForState(iSetup, track, stateName_);
    }
  }
  
  if ( doAllPlots_ ) {
  }

}

void TrackAnalyzer::fillHistosForEfficiencyFromHitPatter(const reco::Track & track, const std::string suffix, const float monitoring, bool useInac) {

    int mon = -1;
    for (int i=0; i<monQuantity::END; i++) {
      if (monName[i] == suffix) mon = i;
    }
    if (useInac) mon+=monQuantity::END;

    //    if (track.pt() > 1.0 && track.dxy() < 0.1 and monitoring > 0) {
    if (track.pt() > 1.0 && track.dxy() < 0.1 and monitoring > -9.) {
      auto hp = track.hitPattern();
      // Here hit_category is meant to iterate over
      // reco::HitPattern::HitCategory, defined here:
      // http://cmslxr.fnal.gov/dxr/CMSSW/source/DataFormats/TrackReco/interface/HitPattern.h
      for (unsigned int category = 0; category < 3; ++category) {
        for (int hit = 0; hit < hp.numberOfAllHits((reco::HitPattern::HitCategory)(category)); ++hit) {
          auto pattern = hp.getHitPattern((reco::HitPattern::HitCategory)(category), hit);
          // Boolean bad is missing simply because it is inferred and the only missing case.
          bool valid = hp.validHitFilter(pattern);
          bool missing = hp.missingHitFilter(pattern);
          bool inactive = hp.inactiveHitFilter(pattern);
          int hit_type = -1;
          hit_type = valid ? 0 :
              ( missing ? 1 :
                ( inactive ? 2 : 3));
          if (hits_valid_.find(Key(hp.getSubStructure(pattern), hp.getSubSubStructure(pattern), mon)) == hits_valid_.end()) {
            LogDebug("TrackAnalyzer") << "Invalid combination of detector and subdetector: ("
                                      << hp.getSubStructure(pattern) << ", "
                                      << hp.getSubSubStructure(pattern) << ", "
                                      << mon
				      << "): ignoring it.\n";
            continue;
	  }
          switch (hit_type) {
            case 0:
              hits_valid_[Key(hp.getSubStructure(pattern), hp.getSubSubStructure(pattern), mon)]->Fill(monitoring);
              hits_total_[Key(hp.getSubStructure(pattern), hp.getSubSubStructure(pattern), mon)]->Fill(monitoring);
              break;
            case 2:
              if (!useInac) break;
            case 1:
              hits_total_[Key(hp.getSubStructure(pattern), hp.getSubSubStructure(pattern), mon)]->Fill(monitoring);
              break;
            default:
              LogDebug("TrackAnalyzer") << "Invalid hit category used " << hit_type << " ignored\n";
          }
        }
      }
    }
  
}

// book histograms at differnt measurement points
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::bookHistosForState(std::string sname, DQMStore::IBooker & ibooker) 
{

    // parameters from the configuration
    std::string QualName       = conf_->getParameter<std::string>("Quality");
    std::string AlgoName       = conf_->getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = !QualName.empty() ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    double Chi2NDFMin = conf_->getParameter<double>("Chi2NDFMin");
    double Chi2NDFMax = conf_->getParameter<double>("Chi2NDFMax");

    int    RecHitBin   = conf_->getParameter<int>(   "RecHitBin");
    double RecHitMin   = conf_->getParameter<double>("RecHitMin");
    double RecHitMax   = conf_->getParameter<double>("RecHitMax");

    int    RecLayBin   = conf_->getParameter<int>(   "RecHitBin");
    double RecLayMin   = conf_->getParameter<double>("RecHitMin");
    double RecLayMax   = conf_->getParameter<double>("RecHitMax");


    int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
    double PhiMin     = conf_->getParameter<double>("PhiMin");
    double PhiMax     = conf_->getParameter<double>("PhiMax");

    int    EtaBin     = conf_->getParameter<int>(   "EtaBin");
    double EtaMin     = conf_->getParameter<double>("EtaMin");
    double EtaMax     = conf_->getParameter<double>("EtaMax");

    int    Phi2DBin     = conf_->getParameter<int>(   "Phi2DBin");
    int    Eta2DBin     = conf_->getParameter<int>(   "Eta2DBin");

    int    ThetaBin   = conf_->getParameter<int>(   "ThetaBin");
    double ThetaMin   = conf_->getParameter<double>("ThetaMin");
    double ThetaMax   = conf_->getParameter<double>("ThetaMax");

    int    TrackQBin  = conf_->getParameter<int>(   "TrackQBin");
    double TrackQMin  = conf_->getParameter<double>("TrackQMin");
    double TrackQMax  = conf_->getParameter<double>("TrackQMax");

    int    TrackPtBin = conf_->getParameter<int>(   "TrackPtBin");
    double TrackPtMin = conf_->getParameter<double>("TrackPtMin");
    double TrackPtMax = conf_->getParameter<double>("TrackPtMax");

    int    TrackPBin  = conf_->getParameter<int>(   "TrackPBin");
    double TrackPMin  = conf_->getParameter<double>("TrackPMin");
    double TrackPMax  = conf_->getParameter<double>("TrackPMax");

    int    TrackPxBin = conf_->getParameter<int>(   "TrackPxBin");
    double TrackPxMin = conf_->getParameter<double>("TrackPxMin");
    double TrackPxMax = conf_->getParameter<double>("TrackPxMax");

    int    TrackPyBin = conf_->getParameter<int>(   "TrackPyBin");
    double TrackPyMin = conf_->getParameter<double>("TrackPyMin");
    double TrackPyMax = conf_->getParameter<double>("TrackPyMax");

    int    TrackPzBin = conf_->getParameter<int>(   "TrackPzBin");
    double TrackPzMin = conf_->getParameter<double>("TrackPzMin");
    double TrackPzMax = conf_->getParameter<double>("TrackPzMax");

    int    ptErrBin   = conf_->getParameter<int>(   "ptErrBin");
    double ptErrMin   = conf_->getParameter<double>("ptErrMin");
    double ptErrMax   = conf_->getParameter<double>("ptErrMax");

    int    pxErrBin   = conf_->getParameter<int>(   "pxErrBin");
    double pxErrMin   = conf_->getParameter<double>("pxErrMin");
    double pxErrMax   = conf_->getParameter<double>("pxErrMax");

    int    pyErrBin   = conf_->getParameter<int>(   "pyErrBin");
    double pyErrMin   = conf_->getParameter<double>("pyErrMin");
    double pyErrMax   = conf_->getParameter<double>("pyErrMax");

    int    pzErrBin   = conf_->getParameter<int>(   "pzErrBin");
    double pzErrMin   = conf_->getParameter<double>("pzErrMin");
    double pzErrMax   = conf_->getParameter<double>("pzErrMax");

    int    pErrBin    = conf_->getParameter<int>(   "pErrBin");
    double pErrMin    = conf_->getParameter<double>("pErrMin");
    double pErrMax    = conf_->getParameter<double>("pErrMax");

    int    phiErrBin  = conf_->getParameter<int>(   "phiErrBin");
    double phiErrMin  = conf_->getParameter<double>("phiErrMin");
    double phiErrMax  = conf_->getParameter<double>("phiErrMax");

    int    etaErrBin  = conf_->getParameter<int>(   "etaErrBin");
    double etaErrMin  = conf_->getParameter<double>("etaErrMin");
    double etaErrMax  = conf_->getParameter<double>("etaErrMax");


    double Chi2ProbMin  = conf_->getParameter<double>("Chi2ProbMin");
    double Chi2ProbMax  = conf_->getParameter<double>("Chi2ProbMax");

    ibooker.setCurrentFolder(TopFolder_);

    TkParameterMEs tkmes;

    std::string histTag = (sname == "default") ? CategoryName : sname + "_" + CategoryName;

    if(doAllPlots_) {

      // general properties
      ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      if (doThetaPlots_) {
	histname = "Chi2oNDFVsTheta_" + histTag;
	tkmes.Chi2oNDFVsTheta = ibooker.bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2NDFMin, Chi2NDFMax,"");
	tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #theta",1);
	tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #chi^{2}/ndf",2);
      }
      histname = "Chi2oNDFVsPhi_" + histTag;
      tkmes.Chi2oNDFVsPhi   = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #phi",1);
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2ProbVsPhi_" + histTag;
      tkmes.Chi2ProbVsPhi = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Tracks #phi"  ,1);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Track #chi^{2} probability",2);
      
      histname = "Chi2ProbVsEta_" + histTag;
      tkmes.Chi2ProbVsEta = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, EtaBin, EtaMin, EtaMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsEta->setAxisTitle("Tracks #eta"  ,1);
      tkmes.Chi2ProbVsEta->setAxisTitle("Track #chi^{2} probability",2);

    }
    
    // general properties
    ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");


    histname = "Chi2oNDFVsEta_" + histTag;
    tkmes.Chi2oNDFVsEta   = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, Chi2NDFMin, Chi2NDFMax,"");
    tkmes.Chi2oNDFVsEta->setAxisTitle("Track #eta",1);
    tkmes.Chi2oNDFVsEta->setAxisTitle("Track #chi^{2}/ndf",2);

    histname = "Chi2oNDFVsPt_" + histTag;
    tkmes.Chi2oNDFVsPt   = ibooker.bookProfile(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax, Chi2NDFMin, Chi2NDFMax,"");
    tkmes.Chi2oNDFVsPt->setAxisTitle("Track p_{T} (GeV/c)", 1);
    tkmes.Chi2oNDFVsPt->setAxisTitle("Track #chi^{2}/ndf",2);

    histname = "Chi2oNDFVsNHits_" + histTag;
    tkmes.Chi2oNDFVsNHits   = ibooker.bookProfile(histname, histname, 50, 0, 50, Chi2NDFMin, Chi2NDFMax,"");
    tkmes.Chi2oNDFVsNHits->setAxisTitle("Track NHits", 1);
    tkmes.Chi2oNDFVsNHits->setAxisTitle("Track #chi^{2}/ndf",2);

    histname = "TrackP_" + histTag;
    tkmes.TrackP = ibooker.book1D(histname, histname, TrackPBin, TrackPMin, TrackPMax);
    tkmes.TrackP->setAxisTitle("Track |p| (GeV/c)", 1);
    tkmes.TrackP->setAxisTitle("Number of Tracks",2);

    histname = "TrackPt_" + histTag;
    tkmes.TrackPt = ibooker.book1D(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax);
    tkmes.TrackPt->setAxisTitle("Track p_{T} (GeV/c)", 1);
    tkmes.TrackPt->setAxisTitle("Number of Tracks",2);

    if (doTrackPxPyPlots_) {
      histname = "TrackPx_" + histTag;
      tkmes.TrackPx = ibooker.book1D(histname, histname, TrackPxBin, TrackPxMin, TrackPxMax);
      tkmes.TrackPx->setAxisTitle("Track p_{x} (GeV/c)", 1);
      tkmes.TrackPx->setAxisTitle("Number of Tracks",2);

      histname = "TrackPy_" + histTag;
      tkmes.TrackPy = ibooker.book1D(histname, histname, TrackPyBin, TrackPyMin, TrackPyMax);
      tkmes.TrackPy->setAxisTitle("Track p_{y} (GeV/c)", 1);
      tkmes.TrackPy->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackPz_" + histTag;
    tkmes.TrackPz = ibooker.book1D(histname, histname, TrackPzBin, TrackPzMin, TrackPzMax);
    tkmes.TrackPz->setAxisTitle("Track p_{z} (GeV/c)", 1);
    tkmes.TrackPz->setAxisTitle("Number of Tracks",2);

    histname = "TrackPhi_" + histTag;
    tkmes.TrackPhi = ibooker.book1D(histname, histname, PhiBin, PhiMin, PhiMax);
    tkmes.TrackPhi->setAxisTitle("Track #phi", 1);
    tkmes.TrackPhi->setAxisTitle("Number of Tracks",2);

    histname = "TrackEta_" + histTag;
    tkmes.TrackEta = ibooker.book1D(histname, histname, EtaBin, EtaMin, EtaMax);
    tkmes.TrackEta->setAxisTitle("Track #eta", 1);
    tkmes.TrackEta->setAxisTitle("Number of Tracks",2);

    histname = "TrackEtaPhi_" + histTag;
    tkmes.TrackEtaPhi = ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
    tkmes.TrackEtaPhi->setAxisTitle("Track #eta", 1);
    tkmes.TrackEtaPhi->setAxisTitle("Track #phi", 2);

    histname = "TrackEtaPhiInner_" + histTag;
    tkmes.TrackEtaPhiInner = ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
    tkmes.TrackEtaPhiInner->setAxisTitle("Track #eta", 1);
    tkmes.TrackEtaPhiInner->setAxisTitle("Track #phi", 2);

    histname = "TrackEtaPhiOuter_" + histTag;
    tkmes.TrackEtaPhiOuter = ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
    tkmes.TrackEtaPhiOuter->setAxisTitle("Track #eta", 1);
    tkmes.TrackEtaPhiOuter->setAxisTitle("Track #phi", 2);



    if (doThetaPlots_) {  
      histname = "TrackTheta_" + histTag;
      tkmes.TrackTheta = ibooker.book1D(histname, histname, ThetaBin, ThetaMin, ThetaMax);
      tkmes.TrackTheta->setAxisTitle("Track #theta", 1);
      tkmes.TrackTheta->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackQ_" + histTag;
    tkmes.TrackQ = ibooker.book1D(histname, histname, TrackQBin, TrackQMin, TrackQMax);
    tkmes.TrackQ->setAxisTitle("Track Charge", 1);
    tkmes.TrackQ->setAxisTitle("Number of Tracks",2);

    histname = "TrackPErrOverP_" + histTag;
    tkmes.TrackPErr = ibooker.book1D(histname, histname, pErrBin, pErrMin, pErrMax);
    tkmes.TrackPErr->setAxisTitle("track error(p)/p", 1);
    tkmes.TrackPErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPt_" + histTag;
    tkmes.TrackPtErr = ibooker.book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
    tkmes.TrackPtErr->setAxisTitle("track error(p_{T})/p_{T}", 1);
    tkmes.TrackPtErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPtVsEta_" + histTag;
    tkmes.TrackPtErrVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, ptErrMin, ptErrMax);
    tkmes.TrackPtErrVsEta->setAxisTitle("Track #eta",1);
    tkmes.TrackPtErrVsEta->setAxisTitle("track error(p_{T})/p_{T}", 2);

    if (doTrackPxPyPlots_) {
      histname = "TrackPxErrOverPx_" + histTag;
      tkmes.TrackPxErr = ibooker.book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
      tkmes.TrackPxErr->setAxisTitle("track error(p_{x})/p_{x}", 1);
      tkmes.TrackPxErr->setAxisTitle("Number of Tracks",2);
      
      histname = "TrackPyErrOverPy_" + histTag;
      tkmes.TrackPyErr = ibooker.book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
      tkmes.TrackPyErr->setAxisTitle("track error(p_{y})/p_{y}", 1);
      tkmes.TrackPyErr->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackPzErrOverPz_" + histTag;
    tkmes.TrackPzErr = ibooker.book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
    tkmes.TrackPzErr->setAxisTitle("track error(p_{z})/p_{z}", 1);
    tkmes.TrackPzErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPhiErr_" + histTag;
    tkmes.TrackPhiErr = ibooker.book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
    tkmes.TrackPhiErr->setAxisTitle("track error(#phi)");
    tkmes.TrackPhiErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackEtaErr_" + histTag;
    tkmes.TrackEtaErr = ibooker.book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
    tkmes.TrackEtaErr->setAxisTitle("track error(#eta)");
    tkmes.TrackEtaErr->setAxisTitle("Number of Tracks",2);

    // rec hit profiles
    ibooker.setCurrentFolder(TopFolder_+"/GeneralProperties");
    histname = "NumberOfRecHitsPerTrackVsPhi_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsPhi = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits in each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfRecHitsPerTrackVsTheta_" + histTag;
      tkmes.NumberOfRecHitsPerTrackVsTheta = ibooker.bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecHitBin, RecHitMin, RecHitMax,"");
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track #phi",1);
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits in each Track",2);
    }
    histname = "NumberOfRecHitsPerTrackVsEta_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits in each Track",2);

    histname = "NumberOfValidRecHitsPerTrackVsPhi_" + histTag;
    tkmes.NumberOfValidRecHitsPerTrackVsPhi = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitMin, RecHitMax,"");
    tkmes.NumberOfValidRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfValidRecHitsPerTrackVsPhi->setAxisTitle("Number of valid RecHits in each Track",2);
    
    histname = "NumberOfValidRecHitsPerTrackVsEta_" + histTag;
    tkmes.NumberOfValidRecHitsPerTrackVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitMin, RecHitMax,"");
    tkmes.NumberOfValidRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfValidRecHitsPerTrackVsEta->setAxisTitle("Number of valid RecHits in each Track",2);
    
    histname = "NumberOfValidRecHitsPerTrackVsPt_" + histTag;
    tkmes.NumberOfValidRecHitsPerTrackVsPt = ibooker.bookProfile(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax, RecHitMin, RecHitMax,"");
    tkmes.NumberOfValidRecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]",1);
    tkmes.NumberOfValidRecHitsPerTrackVsPt->setAxisTitle("Number of valid RecHits in each Track",2);
    
    //////////////////////////////////////////
    histname = "NumberOfLayersPerTrackVsPhi_" + histTag;
    tkmes.NumberOfLayersPerTrackVsPhi = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Number of Layers in each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfLayersPerTrackVsTheta_" + histTag;
      tkmes.NumberOfLayersPerTrackVsTheta = ibooker.bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecLayBin, RecLayMin, RecLayMax,"");
      tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Track #phi",1);
      tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Number of Layers in each Track",2);
    }
    histname = "NumberOfLayersPerTrackVsEta_" + histTag;
    tkmes.NumberOfLayersPerTrackVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Number of Layers in each Track",2);

    if (doThetaPlots_) {
      histname = "Chi2oNDFVsTheta_" + histTag;
      tkmes.Chi2oNDFVsTheta = ibooker.bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #theta",1);
      tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #chi^{2}/ndf",2);
    }
    if (doAllPlots_) {
      histname = "Chi2oNDFVsPhi_" + histTag;
      tkmes.Chi2oNDFVsPhi   = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #phi",1);
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2oNDFVsEta_" + histTag;
      tkmes.Chi2oNDFVsEta   = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #eta",1);
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2ProbVsPhi_" + histTag;
      tkmes.Chi2ProbVsPhi = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Tracks #phi"  ,1);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Track #chi^{2} probability",2);
      
      histname = "Chi2ProbVsEta_" + histTag;
      tkmes.Chi2ProbVsEta = ibooker.bookProfile(histname+CategoryName, histname+CategoryName, EtaBin, EtaMin, EtaMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsEta->setAxisTitle("Tracks #eta"  ,1);
      tkmes.Chi2ProbVsEta->setAxisTitle("Track #chi^{2} probability",2);
    }

    // now put the MEs in the map
    TkParameterMEMap.insert( std::make_pair(sname, tkmes) );

}


// fill histograms at differnt measurement points
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname) 
{
    //get the kinematic parameters
    double p, px, py, pz, pt, theta, phi, eta, q;
    double pxerror, pyerror, pzerror, pterror, perror, phierror, etaerror;

    auto phiIn =  track.innerPosition().phi();
    auto etaIn =  track.innerPosition().eta();
    auto phiOut =  track.outerPosition().phi();
    auto etaOut =  track.outerPosition().eta();


    if (sname == "default") {

      p     = track.p();
      px    = track.px();
      py    = track.py();
      pz    = track.pz();
      pt    = track.pt();
      phi   = track.phi();
      theta = track.theta();
      eta   = track.eta();
      q     = track.charge();
      
      pterror  = (pt) ? track.ptError()/(pt*pt) : 0.0;
      pxerror  = -1.0;
      pyerror  = -1.0;
      pzerror  = -1.0;
      perror   = -1.0;
      phierror = track.phiError();
      etaerror = track.etaError();
      
    } else {
      
      edm::ESHandle<TransientTrackBuilder> theB;
      iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
      reco::TransientTrack TransTrack = theB->build(track);
      
      TrajectoryStateOnSurface TSOS;

      if      (sname == "OuterSurface")  TSOS = TransTrack.outermostMeasurementState();
      else if (sname == "InnerSurface")  TSOS = TransTrack.innermostMeasurementState();
      else if (sname == "ImpactPoint")   TSOS = TransTrack.impactPointState();

      p     = TSOS.globalMomentum().mag();
      px    = TSOS.globalMomentum().x();
      py    = TSOS.globalMomentum().y();
      pz    = TSOS.globalMomentum().z();
      pt    = TSOS.globalMomentum().perp();
      phi   = TSOS.globalMomentum().phi();
      theta = TSOS.globalMomentum().theta();
      eta   = TSOS.globalMomentum().eta();
      q     = TSOS.charge();

      //get the error of the kinimatic parameters
      AlgebraicSymMatrix66 errors = TSOS.cartesianError().matrix();
      double partialPterror = errors(3,3)*pow(TSOS.globalMomentum().x(),2) + errors(4,4)*pow(TSOS.globalMomentum().y(),2);
      pterror  = sqrt(partialPterror)/TSOS.globalMomentum().perp();
      pxerror  = sqrt(errors(3,3))/TSOS.globalMomentum().x();
      pyerror  = sqrt(errors(4,4))/TSOS.globalMomentum().y();
      pzerror  = sqrt(errors(5,5))/TSOS.globalMomentum().z();
      perror   = sqrt(partialPterror+errors(5,5)*pow(TSOS.globalMomentum().z(),2))/TSOS.globalMomentum().mag();
      phierror = sqrt(TSOS.curvilinearError().matrix()(2,2));
      etaerror = sqrt(TSOS.curvilinearError().matrix()(1,1))*fabs(sin(TSOS.globalMomentum().theta()));

    }

    std::map<std::string, TkParameterMEs>::iterator iPos = TkParameterMEMap.find(sname); 
    if (iPos != TkParameterMEMap.end()) {

      TkParameterMEs tkmes = iPos->second;
      
      // momentum
      tkmes.TrackP->Fill(p);
      if (doTrackPxPyPlots_) {
	tkmes.TrackPx->Fill(px);
	tkmes.TrackPy->Fill(py);
      }
      tkmes.TrackPz->Fill(pz);
      tkmes.TrackPt->Fill(pt);
      
      // angles
      tkmes.TrackPhi->Fill(phi);
      tkmes.TrackEta->Fill(eta);
      tkmes.TrackEtaPhi->Fill(eta,phi);
      tkmes.TrackEtaPhiInner->Fill(etaIn,phiIn);
      tkmes.TrackEtaPhiOuter->Fill(etaOut,phiOut);

      if (doThetaPlots_) {
	tkmes.TrackTheta->Fill(theta);
      }
      tkmes.TrackQ->Fill(q);
      
      // errors
      tkmes.TrackPtErr->Fill(pterror);
      tkmes.TrackPtErrVsEta->Fill(eta,pterror);
      if (doTrackPxPyPlots_) {
	tkmes.TrackPxErr->Fill(pxerror);
	tkmes.TrackPyErr->Fill(pyerror);
      }
      tkmes.TrackPzErr->Fill(pzerror);
      tkmes.TrackPErr->Fill(perror);
      tkmes.TrackPhiErr->Fill(phierror);
      tkmes.TrackEtaErr->Fill(etaerror);
      
      int nRecHits      = track.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS);
      int nValidRecHits = track.numberOfValidHits();
      // rec hits 
      tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi,    nRecHits);
      if (doThetaPlots_) {
	tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta,nRecHits);
      }
      tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta,    nRecHits);
      
      tkmes.NumberOfValidRecHitsPerTrackVsPhi->Fill(phi,    nValidRecHits);
      tkmes.NumberOfValidRecHitsPerTrackVsEta->Fill(eta,    nValidRecHits);
      tkmes.NumberOfValidRecHitsPerTrackVsPt ->Fill(pt,     nValidRecHits);

      int nLayers = track.hitPattern().trackerLayersWithMeasurement();
      // rec layers 
      tkmes.NumberOfLayersPerTrackVsPhi->Fill(phi,     nLayers);
      if (doThetaPlots_) {
	tkmes.NumberOfLayersPerTrackVsTheta->Fill(theta, nLayers);
      }
      tkmes.NumberOfLayersPerTrackVsEta->Fill(eta,     nLayers);
      
      double chi2prob = TMath::Prob(track.chi2(),(int)track.ndof());
      double chi2oNDF = track.normalizedChi2();

      tkmes.Chi2oNDFVsEta->Fill(eta, chi2oNDF);
      tkmes.Chi2oNDFVsPt->Fill(pt, chi2oNDF);
      tkmes.Chi2oNDFVsNHits->Fill(nRecHits, chi2oNDF);

      if(doAllPlots_) {
	
	// general properties
	if (doThetaPlots_) {
	  tkmes.Chi2oNDFVsTheta->Fill(theta, chi2oNDF);
	}
	tkmes.Chi2oNDFVsPhi->Fill(phi, chi2oNDF);
	tkmes.Chi2ProbVsPhi->Fill(phi, chi2prob);
	tkmes.Chi2ProbVsEta->Fill(eta, chi2prob);
      }
      
    }

}


void TrackAnalyzer::bookHistosForTrackerSpecific(DQMStore::IBooker & ibooker) 
{

    // parameters from the configuration
    std::string QualName     = conf_->getParameter<std::string>("Quality");
    std::string AlgoName     = conf_->getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = !QualName.empty() ? AlgoName + "_" + QualName : AlgoName;

    int    PhiBin     = conf_->getParameter<int>(   "PhiBin");
    double PhiMin     = conf_->getParameter<double>("PhiMin");
    double PhiMax     = conf_->getParameter<double>("PhiMax");

    int    EtaBin     = conf_->getParameter<int>(   "EtaBin");
    double EtaMin     = conf_->getParameter<double>("EtaMin");
    double EtaMax     = conf_->getParameter<double>("EtaMax");

    int    PtBin = conf_->getParameter<int>(   "TrackPtBin");
    double PtMin = conf_->getParameter<double>("TrackPtMin");
    double PtMax = conf_->getParameter<double>("TrackPtMax");

    // book hit property histograms
    // ---------------------------------------------------------------------------------//
    ibooker.setCurrentFolder(TopFolder_+"/HitProperties");



    std::vector<std::string> subdetectors = conf_->getParameter<std::vector<std::string> >("subdetectors");
    int detBin = conf_->getParameter<int>("subdetectorBin");

    for ( auto det : subdetectors ) {
      
      // hits properties
      ibooker.setCurrentFolder(TopFolder_+"/HitProperties/"+det);
      
      TkRecHitsPerSubDetMEs recHitsPerSubDet_mes;

      recHitsPerSubDet_mes.detectorTag = det;
      int detID = -1;
      if ( det == "TIB" ) detID = StripSubdetector::TIB; // 3
      if ( det == "TOB" ) detID = StripSubdetector::TOB; // 5
      if ( det == "TID" ) detID = StripSubdetector::TID; // 4
      if ( det == "TEC" ) detID = StripSubdetector::TEC; // 6
      if ( det == "PixBarrel" ) detID = PixelSubdetector::PixelBarrel; // 1
      if ( det == "PixEndcap" ) detID = PixelSubdetector::PixelEndcap; // 2
      if ( det == "Pixel" ) detID = 0;
      if ( det == "Strip" ) detID = 7;
      
      recHitsPerSubDet_mes.detectorId  = detID;

      histname = "NumberOfRecHitsPerTrack_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack = ibooker.book1D(histname, histname, detBin, -0.5, double(detBin)-0.5);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack->setAxisTitle("Number of " + det + " valid RecHits in each Track",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfRecHitsPerTrackVsPhi_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of " + det + " valid RecHits in each Track",2);

      histname = "NumberOfRecHitsPerTrackVsEta_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of " + det + " valid RecHits in each Track",2);

      histname = "NumberOfRecHitsPerTrackVsPt_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPt = ibooker.bookProfile(histname, histname, PtBin, PtMin, PtMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPt->setAxisTitle("Number of " + det + " valid RecHits in each Track",2);

      histname = "NumberOfLayersPerTrack_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrack = ibooker.book1D(histname, histname, detBin, -0.5, double(detBin)-0.5);
      recHitsPerSubDet_mes.NumberOfLayersPerTrack->setAxisTitle("Number of " + det + " valid Layers in each Track",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLayersPerTrackVsPhi_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi = ibooker.bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Track #phi",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Number of " + det + " valid Layers in each Track",2);

      histname = "NumberOfLayersPerTrackVsEta_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta = ibooker.bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta->setAxisTitle("Track #eta",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta->setAxisTitle("Number of " + det + " valid Layers in each Track",2);

      histname = "NumberOfLayersPerTrackVsPt_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPt = ibooker.bookProfile(histname, histname, PtBin, PtMin, PtMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPt->setAxisTitle("Track p_{T} [GeV]",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPt->setAxisTitle("Number of " + det + " valid Layers in each Track",2);

      TkRecHitsPerSubDetMEMap.insert(std::pair<std::string,TkRecHitsPerSubDetMEs>(det,recHitsPerSubDet_mes));

      
    }


}


void TrackAnalyzer::fillHistosForTrackerSpecific(const reco::Track & track) 
{
    
  double phi   = track.phi();
  double eta   = track.eta();
  double pt    = track.pt();

  for ( std::map<std::string,TkRecHitsPerSubDetMEs>::iterator it = TkRecHitsPerSubDetMEMap.begin();
       it != TkRecHitsPerSubDetMEMap.end(); it++ ) {

    int nValidLayers  = 0;
    int nValidRecHits = 0;
    int substr = it->second.detectorId;
    switch(substr) {
    case 0 :
      nValidLayers  = track.hitPattern().pixelBarrelLayersWithMeasurement() 
	+ track.hitPattern().pixelEndcapLayersWithMeasurement();    // case 0: pixel
      nValidRecHits = track.hitPattern().numberOfValidPixelBarrelHits()
	+ track.hitPattern().numberOfValidPixelEndcapHits();        // case 0: pixel
      break;
    case StripSubdetector::TIB :
      nValidLayers  = track.hitPattern().stripTIBLayersWithMeasurement();       // case 3: strip TIB
      nValidRecHits = track.hitPattern().numberOfValidStripTIBHits();           // case 3: strip TIB
      break;
    case StripSubdetector::TID :
      nValidLayers  = track.hitPattern().stripTIDLayersWithMeasurement();       // case 4: strip TID
      nValidRecHits = track.hitPattern().numberOfValidStripTIDHits();           // case 4: strip TID
      break;
    case StripSubdetector::TOB :
      nValidLayers  = track.hitPattern().stripTOBLayersWithMeasurement();       // case 5: strip TOB
      nValidRecHits = track.hitPattern().numberOfValidStripTOBHits();           // case 5: strip TOB
      break;
    case StripSubdetector::TEC :
      nValidLayers  = track.hitPattern().stripTECLayersWithMeasurement();       // case 6: strip TEC
      nValidRecHits = track.hitPattern().numberOfValidStripTECHits();           // case 6: strip TEC
      break;
    case PixelSubdetector::PixelBarrel :
      nValidLayers  = track.hitPattern().pixelBarrelLayersWithMeasurement();    // case 1: pixel PXB
      nValidRecHits = track.hitPattern().numberOfValidPixelBarrelHits();        // case 1: pixel PXB
      break;
    case PixelSubdetector::PixelEndcap :
      nValidLayers  = track.hitPattern().pixelEndcapLayersWithMeasurement();    // case 2: pixel PXF
      nValidRecHits = track.hitPattern().numberOfValidPixelEndcapHits();        // case 2: pixel PXF
      break;
    case 7 :
      nValidLayers  = track.hitPattern().stripTIBLayersWithMeasurement()       // case 7: strip
	+ track.hitPattern().stripTIDLayersWithMeasurement()
	+ track.hitPattern().stripTOBLayersWithMeasurement()
	+ track.hitPattern().stripTECLayersWithMeasurement();
      nValidRecHits = track.hitPattern().numberOfValidStripTIBHits()           // case 7: strip
	+ track.hitPattern().numberOfValidStripTIDHits()
	+ track.hitPattern().numberOfValidStripTOBHits()
	+ track.hitPattern().numberOfValidStripTECHits();
      break;
    default :
      break;
    }

    //Fill Layers and RecHits
    it->second.NumberOfRecHitsPerTrack      -> Fill(nValidRecHits); 
    it->second.NumberOfRecHitsPerTrackVsPhi -> Fill(phi,    nValidRecHits);
    it->second.NumberOfRecHitsPerTrackVsEta -> Fill(eta,    nValidRecHits);
    it->second.NumberOfRecHitsPerTrackVsPt  -> Fill(pt,     nValidRecHits);
    
    it->second.NumberOfLayersPerTrack      -> Fill(nValidLayers);
    it->second.NumberOfLayersPerTrackVsPhi -> Fill(phi,     nValidLayers);
    it->second.NumberOfLayersPerTrackVsEta -> Fill(eta,     nValidLayers);
    it->second.NumberOfLayersPerTrackVsPt  -> Fill(pt,      nValidLayers);
  }

}
//
// -- Set Lumi Flag
//
void TrackAnalyzer::setLumiFlag() { 

  TkParameterMEs tkmes;
  if ( Chi2oNDF_lumiFlag                ) Chi2oNDF_lumiFlag                -> setLumiFlag();
  if ( NumberOfRecHitsPerTrack_lumiFlag ) NumberOfRecHitsPerTrack_lumiFlag -> setLumiFlag();
}
//
// -- Apply SoftReset 
//
void TrackAnalyzer::doSoftReset(DQMStore * dqmStore_) {
  TkParameterMEs tkmes;
  dqmStore_->softReset(Chi2oNDF);
  dqmStore_->softReset(NumberOfRecHitsPerTrack);
}
//
// -- Apply Reset 
//
void TrackAnalyzer::doReset() {
  TkParameterMEs tkmes;
  if ( Chi2oNDF_lumiFlag                ) Chi2oNDF_lumiFlag                -> Reset();
  if ( NumberOfRecHitsPerTrack_lumiFlag ) NumberOfRecHitsPerTrack_lumiFlag -> Reset();
}
//
// -- Remove SoftReset
//
void TrackAnalyzer::undoSoftReset(DQMStore * dqmStore_) {
  TkParameterMEs tkmes;
  dqmStore_->disableSoftReset(Chi2oNDF);
  dqmStore_->disableSoftReset(NumberOfRecHitsPerTrack);
}


