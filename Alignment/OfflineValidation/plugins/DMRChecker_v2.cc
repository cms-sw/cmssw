#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariables.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariablesIORoot.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"  
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "IOMC/RandomEngine/src/RandomEngineStateProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondCore/SiPixelPlugins/interface/Phase1PixelMaps.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "TBranch.h"
#include "TFile.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TLorentzVector.h"
#include "TProfile.h"
#include "TTree.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <vector>
#include <time.h> 

#include "CommonTools/TrackerMap/interface/TrackerMap.h"

class MagneticField;

#define DEBUG 0

using namespace std;
using namespace edm;

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

namespace running{
  struct Estimators{
    int   rDirection;
    int   zDirection;
    int   hitCount;
    float runningMeanOfRes_;
    float runningVarOfRes_;
    float runningNormMeanOfRes_;
    float runningNormVarOfRes_;
  };
}

class DMRChecker_v2 : public edm::EDAnalyzer {
 public:
  DMRChecker_v2(const edm::ParameterSet& pset) {
    
    TkTag_     = pset.getParameter<string>("TkTag");
    theTrackCollectionToken = consumes<reco::TrackCollection>(TkTag_);
     
    InputTag tag("TriggerResults","","HLT"); 
    hltresultsToken = consumes<edm::TriggerResults>(tag);

    InputTag beamSpotTag("offlineBeamSpot");
    beamspotToken = consumes<reco::BeamSpot>(beamSpotTag);
    
    InputTag vertexTag("offlinePrimaryVertices");
    vertexToken   = consumes<reco::VertexCollection>(vertexTag);

    isCosmics_ = pset.getParameter<bool>("isCosmics");

    pmap = new TrackerMap("Pixel");
    pmap->onlyPixel(true);
    pmap->setTitle("Pixel Hit entries");
    pmap->setPalette(1);

    tmap = new TrackerMap("Strip");
    tmap->setTitle("Strip Hit entries");
    tmap->setPalette(1);

    pixelmap = std::make_unique<Phase1PixelMaps>("COLZ L");
    pixelmap->bookBarrelHistograms("DMRsX", "Median Residuals x-direction","Median Residuals");
    pixelmap->bookBarrelBins("DMRsX");
    pixelmap->bookForwardHistograms("DMRsX", "Median Residuals x-direction", "Median Residuals");
    pixelmap->bookForwardBins("DMRsX");
    
    pixelmap->bookBarrelHistograms("DMRsY", "Median Residuals y-direction","Median Residuals");
    pixelmap->bookBarrelBins("DMRsY");
    pixelmap->bookForwardHistograms("DMRsY", "Median Residuals y-direction", "Median Residuals");
    pixelmap->bookForwardBins("DMRsY");
   
    // set no rescale
    pixelmap->setNoRescale();

  }

  ~DMRChecker_v2(){}

  template <class OBJECT_TYPE>
  int GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name)
  {
    int result = 0;
    for (typename std::vector<OBJECT_TYPE*>::const_iterator iter = vec.begin(), iterEnd = vec.end();
	 iter != iterEnd; ++iter, ++result) {
      if (*iter && (*iter)->GetName() == name) return result;
    }
    edm::LogError("Alignment") << "@SUB=TrackerOfflineValidation::GetIndex" << " could not find " << name;
    return -1;
  }

  edm::Service<TFileService> fs;
  
  std::unique_ptr<Phase1PixelMaps> pixelmap;

  TrackerMap *tmap;
  TrackerMap *pmap;

  TH1D *hchi2ndof;
  TH1D *hNtrk;
  TH1D *hNtrkZoom;
  TH1I *htrkQuality;
  TH1I *htrkAlgo;
  TH1D *hNhighPurity;
  TH1D *hP;
  TH1D *hPPlus;
  TH1D *hPMinus;
  TH1D *hPt;
  TH1D *hMinPt;
  TH1D *hPtPlus;
  TH1D *hPtMinus;
  TH1D *hHit;
  TH1D *hHit2D; 

  TH1D *hBPixResXPrime;
  TH1D *hFPixResXPrime;
  TH1D *hFPixZPlusResXPrime;
  TH1D *hFPixZMinusResXPrime;

  TH1D *hBPixResYPrime;
  TH1D *hFPixResYPrime;
  TH1D *hFPixZPlusResYPrime;
  TH1D *hFPixZMinusResYPrime;
  
  TH1D *hBPixResXPull;
  TH1D *hFPixResXPull;
  TH1D *hFPixZPlusResXPull;
  TH1D *hFPixZMinusResXPull;

  TH1D *hBPixResYPull;
  TH1D *hFPixResYPull;
  TH1D *hFPixZPlusResYPull;
  TH1D *hFPixZMinusResYPull;

  TH1D  *hTIBResXPrime;    
  TH1D  *hTOBResXPrime;    
  TH1D  *hTIDResXPrime;    
  TH1D  *hTECResXPrime;    
    		     
  TH1D  *hTIBResXPull;     
  TH1D  *hTOBResXPull;     
  TH1D  *hTIDResXPull;     
  TH1D  *hTECResXPull;     

  TH1D *hHitCountVsXBPix;
  TH1D *hHitCountVsXFPix;
  TH1D *hHitCountVsYBPix;
  TH1D *hHitCountVsYFPix;
  TH1D *hHitCountVsZBPix;
  TH1D *hHitCountVsZFPix;
  
  TH1D *hHitCountVsThetaBPix;
  TH1D *hHitCountVsPhiBPix; 
  
  TH1D *hHitCountVsThetaFPix;
  TH1D *hHitCountVsPhiFPix; 

  TH1D *hHitCountVsXFPixPlus;
  TH1D *hHitCountVsXFPixMinus;
  TH1D *hHitCountVsYFPixPlus;
  TH1D *hHitCountVsYFPixMinus;
  TH1D *hHitCountVsZFPixPlus;
  TH1D *hHitCountVsZFPixMinus;
  
  TH1D *hHitCountVsThetaFPixPlus;
  TH1D *hHitCountVsPhiFPixPlus; 
  
  TH1D *hHitCountVsThetaFPixMinus;
  TH1D *hHitCountVsPhiFPixMinus; 

  TH1D *hHitPlus;
  TH1D *hHitMinus;

  TH1D *hPhp;    
  TH1D *hPthp;   
  TH1D *hHithp;  
  TH1D *hEtahp; 
  TH1D *hPhihp;  
  TH1D *hchi2ndofhp; 
  TH1D *hchi2Probhp;

  TH1D *hCharge;
  TH1D *hQoverP;
  TH1D *hQoverPZoom;
  TH1D *hEta;
  TH1D *hEtaPlus; 
  TH1D *hEtaMinus;
  TH1D *hPhi;
  TH1D *hPhiBarrel;
  TH1D *hPhiOverlapPlus;
  TH1D *hPhiOverlapMinus;
  TH1D *hPhiEndcapPlus;
  TH1D *hPhiEndcapMinus;
  TH1D *hPhiPlus;
  TH1D *hPhiMinus;

  TH1D *hDeltaPhi;
  TH1D *hDeltaEta;
  TH1D *hDeltaR  ;

  TH1D *hvx;
  TH1D *hvy;
  TH1D *hvz;
  TH1D *hd0;
  TH1D *hdz;
  TH1D *hdxy;

  TH2D *hd0PVvsphi ;
  TH2D *hd0PVvseta;
  TH2D *hd0PVvspt;

  TH2D *hd0vsphi;
  TH2D *hd0vseta;
  TH2D *hd0vspt;

  TH1D *hnhpxb;
  TH1D *hnhpxe;
  TH1D *hnhTIB;
  TH1D *hnhTID;
  TH1D *hnhTOB;
  TH1D *hnhTEC;

  TH1D *hHitComposition;

  TProfile* pNBpixHitsVsVx;
  TProfile* pNBpixHitsVsVy;
  TProfile* pNBpixHitsVsVz;

  TH1D *hMultCand ;

  TH1D *hdxyBS;
  TH1D *hd0BS;
  TH1D *hdzBS;
  TH1D *hdxyPV;
  TH1D *hd0PV;
  TH1D *hdzPV;
  TH1D *hrun;
  TH1D *hlumi;

  std::vector<TH1*> vTrackHistos_;
  std::vector<TH1*> vTrackProfiles_;
  std::vector<TH1*> vTrack2DHistos_;

  TH1D *tksByTrigger_;
  TH1D *evtsByTrigger_;

  TH1D *modeByRun_;
  TH1D *fieldByRun_;

  TH1D *tracksByRun_;
  TH1D *hitsByRun_;

  TH1D *trackRatesByRun_;
  TH1D *eventRatesByRun_;

  TH1D *hitsinBPixByRun_;
  TH1D *hitsinFPixByRun_;

  // Pixel

  TH1D *DMRBPixX_;
  TH1D *DMRBPixY_;
	         
  TH1D *DMRFPixX_;
  TH1D *DMRFPixY_;

  TH1D *DRnRBPixX_;
  TH1D *DRnRBPixY_;
	         
  TH1D *DRnRFPixX_;
  TH1D *DRnRFPixY_;
  
  // Strips

  TH1D *DMRTIB_;
  TH1D *DMRTOB_;
	         
  TH1D *DMRTID_;
  TH1D *DMRTEC_;

  TH1D *DRnRTIB_;
  TH1D *DRnRTOB_;
	         
  TH1D *DRnRTID_;
  TH1D *DRnRTEC_;

  std::map<unsigned int,TH1D*> barrelLayersResidualsX;
  std::map<unsigned int,TH1D*> barrelLayersPullsX;
  std::map<unsigned int,TH1D*> barrelLayersResidualsY;
  std::map<unsigned int,TH1D*> barrelLayersPullsY;

  std::map<unsigned int,TH1D*> endcapDisksResidualsX;
  std::map<unsigned int,TH1D*> endcapDisksPullsX;
  std::map<unsigned int,TH1D*> endcapDisksResidualsY;
  std::map<unsigned int,TH1D*> endcapDisksPullsY;

  std::string trigNames;
  int trigCount;
  int ievt;
  int itrks;
  int mol;
  int mode;
  float InvMass;
  double invMass;
  bool firstEvent_;
  const TrackerGeometry* trackerGeometry_;

  std::string TkTag_;
  bool isCosmics_;
  edm::EDGetTokenT<reco::TrackCollection>  theTrackCollectionToken;
  edm::EDGetTokenT<edm::TriggerResults> hltresultsToken;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken;

  std::map<std::string,std::pair<int,int> > triggerMap_;
  std::map<int,std::pair<int,float> > conditionsMap_; 
  std::map<int,std::pair<int,int> > runInfoMap_;
  std::map<int,std::array<int,6> > runHitsMap_;

  std::map<int,float> timeMap_;

  // Pixel

  std::map<int,running::Estimators > resDetailsBPixX_;
  std::map<int,running::Estimators > resDetailsBPixY_;
  std::map<int,running::Estimators > resDetailsFPixX_;
  std::map<int,running::Estimators > resDetailsFPixY_;


  // Strips

  std::map<int,running::Estimators > resDetailsTIB_;
  std::map<int,running::Estimators > resDetailsTOB_;
  std::map<int,running::Estimators > resDetailsTID_;
  std::map<int,running::Estimators > resDetailsTEC_;

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){

    //typedef cond::Time_t Time_t;
    // const unsigned int MAX_VAL(std::numeric_limits<unsigned int>::max());

    // std::cout<<"====> MAXVAL: "<<MAX_VAL<<std::endl;

    // edm::ESHandle<Alignments> globalPositionRcd;
    // edm::ValidityInterval iov(setup.get<GlobalPositionRcd>().validityInterval() );
    // if (iov.first().eventID().run()!=1 || iov.last().eventID().run()!=MAX_VAL) {
    //   throw cms::Exception("DatabaseError")
    // 	<< "@SUB=AlignmentProducer::applyDB"
    // 	<< "\nTrying to apply "<< setup.get<GlobalPositionRcd>().key().name()
    // 	<< " with multiple IOVs in tag.\n"
    // 	<< "Validity range is "
    // 	<< iov.first().eventID().run() << " - " << iov.last().eventID().run();
    // }
    
    ievt++;
    
    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByToken(theTrackCollectionToken, trackCollection);
    
    // magnetic field setup
    edm::ESHandle<MagneticField> magneticField_;
    setup.get<IdealMagneticFieldRecord>().get(magneticField_);
    float B_=magneticField_.product()->inTesla(GlobalPoint(0,0,0)).mag();

    // edm::ESHandle<RunInfo> runInfo;
    // setup.get<RunInfoRcd>().get(runInfo);
    // //float average_current = runInfo.product()->m_avg_current;
    // //float uptimeInSeconds = runInfo.product()->m_run_intervall_micros;

    edm::ESHandle<RunInfo> sum;
    setup.get<RunInfoRcd>().get(sum);
    const RunInfo* summary=sum.product();
    time_t start_time = summary->m_start_time_ll;
    ctime(&start_time);
    time_t end_time   = summary->m_stop_time_ll;
    ctime(&end_time);

    /*
      std::cout<< " start_time " << start_time << "( " << summary->m_start_time_str <<" )" 
      << " end_time "   << end_time   << "( " << summary->m_stop_time_str  <<" )" << std::endl;
    */

    double seconds = difftime(end_time,start_time)/1.0e+6;
    //std::cout<<" diff: "<< seconds << "s" << std::endl;

    timeMap_[event.run()] = seconds;

    // topology setup 
    edm::ESHandle<TrackerTopology> tTopoHandle;
    setup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();
    
    edm::ESHandle<SiStripLatency> apvlat;
    setup.get<SiStripLatencyRcd>().get(apvlat);
    if(apvlat->singleReadOutMode()==1) mode =  1; // peak mode
    if(apvlat->singleReadOutMode()==0) mode = -1; // deco mode
    
    conditionsMap_[event.run()].first  = mode;
    conditionsMap_[event.run()].second = B_;

    // geometry setup
    edm::ESHandle<TrackerGeometry> geometry;
    setup.get<TrackerDigiGeometryRecord>().get(geometry);
    const TrackerGeometry* theGeometry = &(*geometry);

    GlobalPoint zeroPoint(0,0,0);
    //std::cout << "event#" << ievt << " Event ID = "<< event.id() 
    /// << " magnetic field: " << magneticField_->inTesla(zeroPoint) << std::endl ;
    
    const reco::TrackCollection tC = *(trackCollection.product());
    itrks+=tC.size();

    runInfoMap_[event.run()].first +=1;
    runInfoMap_[event.run()].second +=tC.size();

    //std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
    TLorentzVector mother(0.,0.,0.,0.);
    
    //int iCounter=0;

    edm::Handle<edm::TriggerResults> hltresults;
    event.getByToken(hltresultsToken,hltresults);

    edm::TriggerNames triggerNames_ = event.triggerNames(*hltresults);
    int ntrigs=hltresults->size();
    vector<string> triggernames = triggerNames_.triggerNames();

    for (int itrig = 0; itrig != ntrigs; ++itrig){
      string trigName=triggerNames_.triggerName(itrig);
      bool accept = hltresults->accept(itrig);
      if (accept== 1){
	// cout << trigName << " " << accept << " ,track size: " << tC.size() << endl;
	triggerMap_[trigName].first+=1;
	triggerMap_[trigName].second+=tC.size();
	// triggerInfo.push_back(pair <string, int> (trigName, accept));
      }
    }
    
    hrun->Fill(event.run());
    hlumi->Fill(event.luminosityBlock());

    Int_t nHighPurityTracks = 0;

    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){

      auto const & residuals = track->extra()->residuals();

      unsigned int nHit2D = 0;
      int h_index=0;
      for (trackingRecHit_iterator iHit = track->recHitsBegin(); iHit != track->recHitsEnd(); ++iHit,++h_index) {
	if(this->isHit2D(**iHit)) ++nHit2D;
	
	double resX = residuals.residualX(h_index);
	double resY = residuals.residualY(h_index);
	double resErrX = residuals.pullX(h_index);
	double resErrY = residuals.pullY(h_index);

	TrackingRecHit* hit = (*iHit)->clone();
	const DetId& detId = hit->geographicalId();
	
	unsigned int subid = detId.subdetId();

	const GeomDet* geomDet( theGeometry->idToDet(detId) );

	float uOrientation(-999.F), vOrientation(-999.F);	    
	LocalPoint lPModule(0.,0.,0.), lUDirection(1.,0.,0.), lVDirection(0.,1.,0.), lWDirection(0.,0.,1.);
	
	if(!hit->detUnit()) continue; // is it a single physical module?

	if(hit->isValid() && 
	   ( subid != PixelSubdetector::PixelBarrel ) && 
	   ( subid != PixelSubdetector::PixelEndcap ) 
	   ){

	  tmap->fill(detId.rawId(),1);     
	  
	  //float uOrientation(-999.F), vOrientation(-999.F);
	  //LocalPoint lUDirection(1.,0.,0.), lVDirection(0.,1.,0.);

	  //LocalPoint lp = (*iHit)->localPosition();
	  //LocalError le = (*iHit)->localPositionError();

	  GlobalPoint gUDirection = geomDet->surface().toGlobal(lUDirection);
	  GlobalPoint gVDirection = geomDet->surface().toGlobal(lVDirection);
	  GlobalPoint gWDirection = geomDet->surface().toGlobal(lWDirection);
	  
	  //GlobalPoint GP = geomDet->surface().toGlobal(lp);
	  GlobalPoint gPModule = geomDet->surface().toGlobal(lPModule);

	  // fill DMRs and DrNRs 
	  if( subid == StripSubdetector::TIB ) {
	   
	    uOrientation = deltaPhi(gUDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;
	    vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F;

	    // if the detid has never occcurred yet, set the local orientations
	    if ( resDetailsTIB_.find(detId.rawId()) == resDetailsTIB_.end() ){
	      resDetailsTIB_[detId.rawId()].rDirection = gWDirection.perp() - gPModule.perp() >=0 ? +1 :-1;
	      resDetailsTIB_[detId.rawId()].zDirection = gVDirection.z() - gPModule.z() >=0 ? +1 : -1;
	    }	    

	    hTIBResXPrime->Fill(uOrientation*resX*10000);
	    hTIBResXPull->Fill(resErrX);

	    // update residuals
	    this->updateOnlineMomenta(resDetailsTIB_,
				      detId.rawId(),
				      uOrientation*resX*10000.,
				      resErrX
				      );

	    
	  } else if (subid == StripSubdetector::TOB){

	    uOrientation = deltaPhi(gUDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;
	    vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F;

	    hTOBResXPrime->Fill(uOrientation*resX*10000);
	    hTOBResXPull->Fill(resErrX);

	    // if the detid has never occcurred yet, set the local orientations
	    if ( resDetailsTOB_.find(detId.rawId()) == resDetailsTOB_.end() ){
	      resDetailsTOB_[detId.rawId()].rDirection = gWDirection.perp() - gPModule.perp() >=0 ? +1 :-1;
	      resDetailsTOB_[detId.rawId()].zDirection = gVDirection.z() - gPModule.z() >=0 ? +1 : -1;
	    }	    

	    // update residuals
	    this->updateOnlineMomenta(resDetailsTOB_,
				      detId.rawId(),
				      uOrientation*resX*10000.,
				      resErrX
				      );
	    
	   	    
	  } else if (subid == StripSubdetector::TID){
	    
	    uOrientation = deltaPhi(gUDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;
	    vOrientation = gVDirection.perp() - gPModule.perp() >= 0. ? +1.F : -1.F;

	    hTIDResXPrime->Fill(uOrientation*resX*10000);
	    hTIDResXPull->Fill(resErrX);

	    // update residuals
	    this->updateOnlineMomenta(resDetailsTID_,
				      detId.rawId(),
				      uOrientation*resX*10000.,
				      resErrX
				      );


	  } else if (subid == StripSubdetector::TEC){

	    uOrientation = deltaPhi(gUDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;
	    vOrientation = gVDirection.perp() - gPModule.perp() >= 0. ? +1.F : -1.F;

	    hTECResXPrime->Fill(uOrientation*resX*10000);
	    hTECResXPull->Fill(resErrX);

	    // update residuals
	    this->updateOnlineMomenta(resDetailsTEC_,
				      detId.rawId(),
				      uOrientation*resX*10000.,
				      resErrX
				      );

	  }
	}
	
	const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
	
	if(pixhit){
	  if(pixhit->isValid() ) {
	    
	    unsigned int subid = detId.subdetId();
	    int detid_db = detId.rawId();
	    
	    // not yet available for phase-I
	    // pmap->fill(detid_db,1);

	    // float uOrientation(-999.F), vOrientation(-999.F);
	    // LocalPoint lUDirection(1.,0.,0.), lVDirection(0.,1.,0.);

	    LocalPoint lp = (*iHit)->localPosition();
	    //LocalError le = (*iHit)->localPositionError();

	    GlobalPoint gUDirection = geomDet->surface().toGlobal(lUDirection);
	    GlobalPoint gVDirection = geomDet->surface().toGlobal(lVDirection);
	    GlobalPoint gWDirection = geomDet->surface().toGlobal(lWDirection);

	    GlobalPoint GP = geomDet->surface().toGlobal(lp);
	    GlobalPoint gPModule = geomDet->surface().toGlobal(lPModule);

	    if ( ( subid == PixelSubdetector::PixelBarrel ) || ( subid == PixelSubdetector::PixelEndcap ) ) {
	       // 1 = PXB, 2 = PXF
	      if ( subid == PixelSubdetector::PixelBarrel ) {
		
		int layer_num = tTopo->pxbLayer(detid_db);
		
		uOrientation = deltaPhi(gUDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;
		vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F; 

		// if the detid has never occcurred yet, set the local orientations
		if ( resDetailsBPixX_.find(detId.rawId()) == resDetailsBPixX_.end() ){
		  resDetailsBPixX_[detId.rawId()].rDirection = gWDirection.perp() - gPModule.perp() >=0 ? +1 :-1;
		  resDetailsBPixX_[detId.rawId()].zDirection = gVDirection.z() - gPModule.z() >=0 ? +1 : -1;
		}

		// if the detid has never occcurred yet, set the local orientations
		if ( resDetailsBPixY_.find(detId.rawId()) == resDetailsBPixY_.end() ){
		  resDetailsBPixY_[detId.rawId()].rDirection = gWDirection.perp() - gPModule.perp() >=0 ? +1 :-1;
		  resDetailsBPixY_[detId.rawId()].zDirection = gVDirection.z() - gPModule.z() >=0 ? +1 : -1;
		}

		hHitCountVsThetaBPix->Fill(GP.theta());
		hHitCountVsPhiBPix->Fill(GP.phi()); 

		hHitCountVsZBPix->Fill(GP.z());
		hHitCountVsXBPix->Fill(GP.x());
		hHitCountVsYBPix->Fill(GP.y());

		hBPixResXPrime->Fill(uOrientation*resX*10000.);
		hBPixResYPrime->Fill(vOrientation*resY*10000.);
		hBPixResXPull->Fill(resErrX);
		hBPixResYPull->Fill(resErrY);
		
		//std::cout<<"layer: "<<layer_num<<std::endl;
		
		// update residuals X		
		this->updateOnlineMomenta(resDetailsBPixX_,
					  detId.rawId(),
					  uOrientation*resX*10000.,
					  resErrX
					  );
	
		// update residuals Y
		this->updateOnlineMomenta(resDetailsBPixY_,
					  detId.rawId(),
					  vOrientation*resY*10000.,
					  resErrY
					  );
		
		fillByIndex(barrelLayersResidualsX,layer_num,uOrientation*resX*10000.); 
		fillByIndex(barrelLayersPullsX,layer_num,resErrX);     
		fillByIndex(barrelLayersResidualsY,layer_num,vOrientation*resY*10000.); 
		fillByIndex(barrelLayersPullsY,layer_num,resErrY);     

	      } else if ( subid == PixelSubdetector::PixelEndcap ) {
		
		uOrientation = gUDirection.perp() - gPModule.perp() >= 0 ? +1.F : -1.F;
		vOrientation = deltaPhi(gVDirection.barePhi(),gPModule.barePhi()) >= 0. ? +1.F : -1.F;

		int side_num  = tTopo->pxfSide(detid_db);
		int disk_num  = tTopo->pxfDisk(detid_db); 

		int packedTopo = disk_num+3*(side_num-1);

		//std::cout<<"side: "<< side_num <<" disk: " << disk_num << " packedTopo: " << packedTopo <<" GP.z(): "<<GP.z()<<std::endl;

		hHitCountVsThetaFPix->Fill(GP.theta());
		hHitCountVsPhiFPix->Fill(GP.phi()); 
		
		hHitCountVsZFPix->Fill(GP.z());
		hHitCountVsXFPix->Fill(GP.x());
		hHitCountVsYFPix->Fill(GP.y());
		
		hFPixResXPrime->Fill(uOrientation*resX*10000.);
		hFPixResYPrime->Fill(vOrientation*resY*10000.);
		hFPixResXPull->Fill(resErrX);
		hFPixResYPull->Fill(resErrY);

		fillByIndex(endcapDisksResidualsX,packedTopo,uOrientation*resX*10000.); 
		fillByIndex(endcapDisksPullsX,packedTopo,resErrX);     
		fillByIndex(endcapDisksResidualsY,packedTopo,vOrientation*resY*10000.); 
		fillByIndex(endcapDisksPullsY,packedTopo,resErrY);     

		// if the detid has never occcurred yet, set the local orientations
		if ( resDetailsFPixX_.find(detId.rawId()) == resDetailsFPixX_.end() ){
		  resDetailsFPixX_[detId.rawId()].rDirection = gUDirection.perp() - gPModule.perp()  >=0 ? +1 :-1;
		  resDetailsFPixX_[detId.rawId()].zDirection = gWDirection.z() - gPModule.z() >=0 ? +1 : -1;
		}	    

		// if the detid has never occcurred yet, set the local orientations
		if ( resDetailsFPixY_.find(detId.rawId()) == resDetailsFPixY_.end() ){
		  resDetailsFPixY_[detId.rawId()].rDirection = gUDirection.perp() - gPModule.perp() >=0 ? +1 :-1;
		  resDetailsFPixY_[detId.rawId()].zDirection = gWDirection.z() - gPModule.z() >=0 ? +1 : -1;
		}	    

		// update residuals X		
		this->updateOnlineMomenta(resDetailsFPixX_,
					  detId.rawId(),
					  uOrientation*resX*10000.,					 
					  resErrX
					  );
	
		// update residuals Y
		this->updateOnlineMomenta(resDetailsFPixY_,
					  detId.rawId(),
					  vOrientation*resY*10000.,
					  resErrY
					  );
		
		if(side_num==1){

		  hHitCountVsXFPixMinus->Fill(GP.x());
		  hHitCountVsYFPixMinus->Fill(GP.y());
		  hHitCountVsZFPixMinus->Fill(GP.z());
		  hHitCountVsThetaFPixMinus->Fill(GP.theta());
		  hHitCountVsPhiFPixMinus->Fill(GP.phi());
		  		
		  hFPixZMinusResXPrime->Fill(uOrientation*resX*10000.);
		  hFPixZMinusResYPrime->Fill(vOrientation*resY*10000.);
		  hFPixZMinusResXPull->Fill(resErrX);
		  hFPixZMinusResYPull->Fill(resErrY);

		} else {
		
		  hHitCountVsXFPixPlus->Fill(GP.x());
		  hHitCountVsYFPixPlus->Fill(GP.y());
		  hHitCountVsZFPixPlus->Fill(GP.z());
		  hHitCountVsThetaFPixPlus->Fill(GP.theta());
		  hHitCountVsPhiFPixPlus->Fill(GP.phi()); 
		  		
		  hFPixZPlusResXPrime->Fill(uOrientation*resX*10000.);
		  hFPixZPlusResYPrime->Fill(vOrientation*resY*10000.);
		  hFPixZPlusResXPull->Fill(resErrX);
		  hFPixZPlusResYPull->Fill(resErrY);
		 
		}  
	      }
	    }
	  }
	}
      }

      hHit2D->Fill(nHit2D);

      // std::cout << "nHit2D: "<<nHit2D<<std::endl;

      hHit->Fill(track->numberOfValidHits());
      hnhpxb->Fill(track->hitPattern().numberOfValidPixelBarrelHits());
      hnhpxe->Fill(track->hitPattern().numberOfValidPixelEndcapHits());
      hnhTIB->Fill(track->hitPattern().numberOfValidStripTIBHits());
      hnhTID->Fill(track->hitPattern().numberOfValidStripTIDHits());
      hnhTOB->Fill(track->hitPattern().numberOfValidStripTOBHits());
      hnhTEC->Fill(track->hitPattern().numberOfValidStripTECHits());
       
      runHitsMap_[event.run()][0] += track->hitPattern().numberOfValidPixelBarrelHits(); 
      runHitsMap_[event.run()][1] += track->hitPattern().numberOfValidPixelEndcapHits();
      runHitsMap_[event.run()][2] += track->hitPattern().numberOfValidStripTIBHits(); 
      runHitsMap_[event.run()][3] += track->hitPattern().numberOfValidStripTIDHits();
      runHitsMap_[event.run()][4] += track->hitPattern().numberOfValidStripTOBHits(); 
      runHitsMap_[event.run()][5] += track->hitPattern().numberOfValidStripTECHits();

      // fill hit composition histogram
      if(track->hitPattern().numberOfValidPixelBarrelHits()!=0){
	hHitComposition->Fill(0.,track->hitPattern().numberOfValidPixelBarrelHits());

	pNBpixHitsVsVx->Fill(track->vx(),track->hitPattern().numberOfValidPixelBarrelHits());
	pNBpixHitsVsVy->Fill(track->vy(),track->hitPattern().numberOfValidPixelBarrelHits());
	pNBpixHitsVsVz->Fill(track->vz(),track->hitPattern().numberOfValidPixelBarrelHits());

      }
      if(track->hitPattern().numberOfValidPixelEndcapHits()!=0){
	hHitComposition->Fill(1.,track->hitPattern().numberOfValidPixelEndcapHits());
      }
      if(track->hitPattern().numberOfValidStripTIBHits()!=0){
	hHitComposition->Fill(2.,track->hitPattern().numberOfValidStripTIBHits());
      }
      if(track->hitPattern().numberOfValidStripTIDHits()!=0){
	hHitComposition->Fill(3.,track->hitPattern().numberOfValidStripTIDHits());
      }
      if(track->hitPattern().numberOfValidStripTOBHits()!=0){
	hHitComposition->Fill(4.,track->hitPattern().numberOfValidStripTOBHits());
      }
      if(track->hitPattern().numberOfValidStripTECHits()!=0){
	hHitComposition->Fill(5.,track->hitPattern().numberOfValidStripTECHits());
      }

      hCharge->Fill(track->charge());
      hQoverP->Fill(track->qoverp());
      hQoverPZoom->Fill(track->qoverp());
      hPt->Fill(track->pt());
      hP->Fill(track->p());
      hchi2ndof->Fill(track->normalizedChi2());
      hEta->Fill(track->eta());
      hPhi->Fill(track->phi());

      //std::cout << "nHit2D: "<<nHit2D<<std::endl;
      
      if (fabs(track->eta() ) < 0.8 )                hPhiBarrel->Fill(track->phi());
      if (track->eta()>0.8 && track->eta()  < 1.4 )  hPhiOverlapPlus->Fill(track->phi());
      if (track->eta()<-0.8 && track->eta() >- 1.4 ) hPhiOverlapMinus->Fill(track->phi());
      if (track->eta()>1.4)                          hPhiEndcapPlus->Fill(track->phi());
      if (track->eta()<-1.4)                         hPhiEndcapMinus->Fill(track->phi());
     
      hd0->Fill(track->d0());
      hdz->Fill(track->dz());
      hdxy->Fill(track->dxy());
      hvx->Fill(track->vx());
      hvy->Fill(track->vy());
      hvz->Fill(track->vz());
    
      //std::cout << "nHit2D: "<<nHit2D<<std::endl;

      // int myalgo=-88;
      // if(track->algo()==reco::TrackBase::undefAlgorithm)myalgo=0;
      // if(track->algo()==reco::TrackBase::ctf)myalgo=1;
      // if(track->algo()==reco::TrackBase::iter0)myalgo=4;
      // if(track->algo()==reco::TrackBase::iter1)myalgo=5;
      // if(track->algo()==reco::TrackBase::iter2)myalgo=6;
      // if(track->algo()==reco::TrackBase::iter3)myalgo=7;
      // if(track->algo()==reco::TrackBase::iter4)myalgo=8;
      // if(track->algo()==reco::TrackBase::iter5)myalgo=9;
      // if(track->algo()==reco::TrackBase::iter6)myalgo=10;
      // if(track->algo()==reco::TrackBase::iter7)myalgo=11;
      // htrkAlgo->Fill(myalgo);
      
      int myquality=-99;
      if(track->quality(reco::TrackBase::undefQuality)){
	myquality=-1;
	htrkQuality->Fill(myquality);
      } 
      if(track->quality(reco::TrackBase::loose)){
	myquality=0;
	htrkQuality->Fill(myquality);
      }
      if(track->quality(reco::TrackBase::tight)){ 
	myquality=1;
	htrkQuality->Fill(myquality);
      }
      if(track->quality(reco::TrackBase::highPurity) && (!isCosmics_)){
	myquality=2;
	htrkQuality->Fill(myquality);
	hPhp->Fill(track->p());
	hPthp->Fill(track->pt());
	hHithp->Fill(track->numberOfValidHits()); 
	hEtahp->Fill(track->eta());
	hPhihp->Fill(track->phi());
	hchi2ndofhp->Fill(track->normalizedChi2());
	hchi2Probhp->Fill(TMath::Prob(track->chi2(),track->ndof()));
	nHighPurityTracks++;
      }
      if(track->quality(reco::TrackBase::confirmed)){
	myquality=3;
	htrkQuality->Fill(myquality);
      }
      if(track->quality(reco::TrackBase::goodIterative)){
	myquality=4;
	htrkQuality->Fill(myquality);
      }
   
      // Fill 1D track histos
      static const int etaindex = this->GetIndex(vTrackHistos_,"h_tracketa");
      vTrackHistos_[etaindex]->Fill(track->eta());
      static const int phiindex = this->GetIndex(vTrackHistos_,"h_trackphi");
      vTrackHistos_[phiindex]->Fill(track->phi());
      static const int numOfValidHitsindex = this->GetIndex(vTrackHistos_,"h_trackNumberOfValidHits");
      vTrackHistos_[numOfValidHitsindex]->Fill(track->numberOfValidHits());
      static const int numOfLostHitsindex = this->GetIndex(vTrackHistos_,"h_trackNumberOfLostHits");
      vTrackHistos_[numOfLostHitsindex]->Fill(track->numberOfLostHits());

      GlobalPoint gPoint(track->vx(), track->vy(), track->vz());
      double theLocalMagFieldInInverseGeV = magneticField_->inInverseGeV(gPoint).z();
      double kappa =  -track->charge()*theLocalMagFieldInInverseGeV/track->pt();

      static const int kappaindex = this->GetIndex(vTrackHistos_,"h_curvature");
      vTrackHistos_[kappaindex]->Fill(kappa);
      static const int kappaposindex = this->GetIndex(vTrackHistos_,"h_curvature_pos");
      if (track->charge() > 0)
      	vTrackHistos_[kappaposindex]->Fill(fabs(kappa));
      static const int kappanegindex = this->GetIndex(vTrackHistos_,"h_curvature_neg");
      if (track->charge() < 0)
      	vTrackHistos_[kappanegindex]->Fill(fabs(kappa));

      double chi2Prob= TMath::Prob(track->chi2(),track->ndof());
      double normchi2 = track->normalizedChi2();

      static const int normchi2index = this->GetIndex(vTrackHistos_,"h_normchi2");
      vTrackHistos_[normchi2index]->Fill(normchi2);
      static const int chi2index = this->GetIndex(vTrackHistos_,"h_chi2");
      vTrackHistos_[chi2index]->Fill(track->chi2());
      static const int chi2Probindex = this->GetIndex(vTrackHistos_,"h_chi2Prob");
      vTrackHistos_[chi2Probindex]->Fill(chi2Prob);
      static const int ptindex = this->GetIndex(vTrackHistos_,"h_pt");
      static const int pt2index = this->GetIndex(vTrackHistos_,"h_ptrebin");
      vTrackHistos_[ptindex]->Fill(track->pt());
      vTrackHistos_[pt2index]->Fill(track->pt());
      if (track->ptError() != 0.) {
      	static const int ptResolutionindex = this->GetIndex(vTrackHistos_,"h_ptResolution");
      	vTrackHistos_[ptResolutionindex]->Fill(track->ptError()/track->pt());
      }
      // Fill track profiles
      static const int d0phiindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_phi");
      vTrackProfiles_[d0phiindex]->Fill(track->phi(),track->d0());
      static const int dzphiindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_phi");
      vTrackProfiles_[dzphiindex]->Fill(track->phi(),track->dz());
      static const int d0etaindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_eta");
      vTrackProfiles_[d0etaindex]->Fill(track->eta(),track->d0());
      static const int dzetaindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_eta");
      vTrackProfiles_[dzetaindex]->Fill(track->eta(),track->dz());
      static const int chiProbphiindex = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_phi");
      vTrackProfiles_[chiProbphiindex]->Fill(track->phi(),chi2Prob);
      static const int chiProbabsd0index = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_d0");
      vTrackProfiles_[chiProbabsd0index]->Fill(fabs(track->d0()),chi2Prob);
      static const int chiProbabsdzindex = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_dz");
      vTrackProfiles_[chiProbabsdzindex]->Fill(track->dz(),chi2Prob);
      static const int chiphiindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_phi");
      vTrackProfiles_[chiphiindex]->Fill(track->phi(),track->chi2());
      static const int normchiphiindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_phi");
      vTrackProfiles_[normchiphiindex]->Fill(track->phi(),normchi2);
      static const int chietaindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_eta");
      vTrackProfiles_[chietaindex]->Fill(track->eta(),track->chi2());
      static const int normchiptindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_pt");
      vTrackProfiles_[normchiptindex]->Fill(track->pt(),normchi2);
      static const int normchipindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_p");
      vTrackProfiles_[normchipindex]->Fill(track->p(),normchi2);
      static const int chiProbetaindex = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_eta");
      vTrackProfiles_[chiProbetaindex]->Fill(track->eta(),chi2Prob);
      static const int normchietaindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_eta");
      vTrackProfiles_[normchietaindex]->Fill(track->eta(),normchi2);
      static const int kappaphiindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_phi");
      vTrackProfiles_[kappaphiindex]->Fill(track->phi(),kappa);
      static const int kappaetaindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_eta");
      vTrackProfiles_[kappaetaindex]->Fill(track->eta(),kappa);
      static const int ptResphiindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_phi");
      vTrackProfiles_[ptResphiindex]->Fill(track->phi(),track->ptError()/track->pt());
      static const int ptResetaindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_eta");
      vTrackProfiles_[ptResetaindex]->Fill(track->eta(),track->ptError()/track->pt());
      
      // Fill 2D track histos
      static const int d0phiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_phi");
      vTrack2DHistos_[d0phiindex_2d]->Fill(track->phi(),track->d0());
      static const int dzphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_phi");
      vTrack2DHistos_[dzphiindex_2d]->Fill(track->phi(),track->dz());
      static const int d0etaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_eta");
      vTrack2DHistos_[d0etaindex_2d]->Fill(track->eta(),track->d0());
      static const int dzetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_eta");
      vTrack2DHistos_[dzetaindex_2d]->Fill(track->eta(),track->dz());
      static const int chiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_phi");
      vTrack2DHistos_[chiphiindex_2d]->Fill(track->phi(),track->chi2());
      static const int chiProbphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_phi");
      vTrack2DHistos_[chiProbphiindex_2d]->Fill(track->phi(),chi2Prob);
      static const int chiProbabsd0index_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_d0");
      vTrack2DHistos_[chiProbabsd0index_2d]->Fill(fabs(track->d0()),chi2Prob);
      static const int normchiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_phi");
      vTrack2DHistos_[normchiphiindex_2d]->Fill(track->phi(),normchi2);
      static const int chietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_eta");
      vTrack2DHistos_[chietaindex_2d]->Fill(track->eta(),track->chi2());
      static const int chiProbetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_eta");
      vTrack2DHistos_[chiProbetaindex_2d]->Fill(track->eta(),chi2Prob);
      static const int normchietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_eta");
      vTrack2DHistos_[normchietaindex_2d]->Fill(track->eta(),normchi2);
      static const int kappaphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_phi");
      vTrack2DHistos_[kappaphiindex_2d]->Fill(track->phi(),kappa);
      static const int kappaetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_eta");
      vTrack2DHistos_[kappaetaindex_2d]->Fill(track->eta(),kappa);
      static const int normchi2kappa_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_kappa");
      vTrack2DHistos_[normchi2kappa_2d]->Fill(normchi2,kappa);
      
      //std::cout << "filling histos"<<std::endl;

      //dxy with respect to the beamspot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotHandle;
      event.getByToken(beamspotToken,beamSpotHandle);
      if(beamSpotHandle.isValid()){ 
	beamSpot = *beamSpotHandle;
	math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
	double dxy = track->dxy(point);
	double dz = track->dz(point);
	hdxyBS->Fill(dxy);
	hd0BS->Fill(-dxy);
	hdzBS->Fill(dz);
      }
      
      //dxy with respect to the primary vertex
      reco::Vertex pvtx;
      edm::Handle<reco::VertexCollection> vertexHandle;
      reco::VertexCollection vertexCollection;
      event.getByLabel("offlinePrimaryVertices",vertexHandle);
      double mindxy = 100.;
      double dz = 100;
      if(vertexHandle.isValid()) {
      	for(reco::VertexCollection::const_iterator pvtx = vertexHandle->begin(); pvtx!=vertexHandle->end(); ++pvtx) {
      	  math::XYZPoint mypoint( pvtx->x(), pvtx->y(), pvtx->z());
      	  if(abs(mindxy)>abs(track->dxy(mypoint))){
      	    mindxy = track->dxy(mypoint);
      	    dz=track->dz(mypoint);
      	    //std::cout<<"dxy: "<<mindxy<<"dz: "<<dz<<std::endl;
      	  }
      	}

      	hdxyPV->Fill(mindxy);
      	hd0PV->Fill(-mindxy);
      	hdzPV->Fill(dz);
	
      	hd0PVvsphi->Fill(track->phi(),-mindxy);
      	hd0PVvseta->Fill(track->eta(),-mindxy);
      	hd0PVvspt->Fill(track->pt(),-mindxy);

      } else {
      	hdxyPV->Fill(100);
      	hd0PV->Fill(100);
      	hdzPV->Fill(100);
      }

      // std::cout<<"end of track loop"<<std::endl;

    }

    // std::cout<<"end of analysis"<<std::endl;

    hNtrk->Fill(tC.size());
    hNtrkZoom->Fill(tC.size());
    hNhighPurity->Fill(nHighPurityTracks);
    
    // std::cout<<"end of analysis"<<std::endl;
    
  }

  virtual void beginJob() {
    if (DEBUG) cout << __LINE__ << endl;

    TH1D::SetDefaultSumw2(kTRUE);

    ievt=0;
    itrks=0;

    hrun  = fs->make<TH1D>("h_run","run",100000,230000,240000);
    hlumi = fs->make<TH1D>("h_lumi","lumi",1000,0,1000);
    
    hchi2ndof = fs->make<TH1D>("h_chi2ndof","chi2/ndf;#chi^{2}/ndf;tracks",100,0,5.);
    hCharge = fs->make<TH1D>("h_charge","charge;Charge of the track;tracks",5,-2.5,2.5);
    hNtrk        = fs->make<TH1D>("h_Ntrk","ntracks;Number of Tracks;events",200,0.,200.);
    hNtrkZoom    =fs->make<TH1D>("h_NtrkZoom","Number of tracks; number of tracks;events",10,0.,10.);
    hNhighPurity = fs->make<TH1D>("h_NhighPurity","n. high purity tracks;Number of high purity tracks;events",200,0.,200.);

    htrkAlgo     = fs->make<TH1I>("h_trkAlgo","tracking step;iterative tracking step;tracks",12,-0.5,11.5);
    TString algos[12] = {"undef","ctf","rs","cosmic","iter0","iter1","iter2","iter3","iter4","iter5","iter6","iter7"};
    for(int nbin=1;nbin<=htrkAlgo->GetNbinsX();nbin++){
      htrkAlgo->GetXaxis()->SetBinLabel(nbin,algos[nbin-1]);
    }

    htrkQuality  = fs->make<TH1I>("h_trkQuality","track quality;track quality;tracks",6,-1,5);
    TString qualities[7] = {"undef","loose","tight","highPurity","confirmed","goodIterative"};
    for(int nbin=1;nbin<=htrkQuality->GetNbinsX();nbin++){
      htrkQuality->GetXaxis()->SetBinLabel(nbin,qualities[nbin-1]);
    }
    
    hP      = fs->make<TH1D>("h_P","Momentum;track momentum [GeV];tracks",100,0.,100.);
    hQoverP = fs->make<TH1D>("h_qoverp","Track q/p; track q/p [GeV^{-1}];tracks",100,-1.,1.);
    hQoverPZoom = fs->make<TH1D>("h_qoverpZoom","Track q/p; track q/p [GeV^{-1}];tracks",100,-0.1,0.1); 
    hPt     = fs->make<TH1D>("h_Pt","Transverse Momentum;track p_{T} [GeV];tracks",100,0.,100.);
    hHit    = fs->make<TH1D>("h_nHits","Number of hits;track n. hits;tracks",50,-0.5,49.5);
    hHit2D  = fs->make<TH1D>("h_nHit2D","Number of 2D hits; number of 2D hits;tracks",20,0,20);

    // Pixel

    hBPixResXPrime       = fs->make<TH1D>("h_BPixResXPrime","BPix track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hFPixResXPrime       = fs->make<TH1D>("h_FPixResXPrime","FPix track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hFPixZPlusResXPrime  = fs->make<TH1D>("h_FPixZPlusResXPrime","FPix (Z+) track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hFPixZMinusResXPrime = fs->make<TH1D>("h_FPixZMinusResXPrime","FPix (Z-) track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    					   			
    hBPixResYPrime       = fs->make<TH1D>("h_BPixResYPrime","BPix track Y-residuals;res_{Y'} [#mum];hits",100,-1000.,1000.);     
    hFPixResYPrime       = fs->make<TH1D>("h_FPixResYPrime","FPix track Y-residuals;res_{Y'} [#mum];hits",100,-1000.,1000.);     
    hFPixZPlusResYPrime  = fs->make<TH1D>("h_FPixZPlusResYPrime","FPix (Z+) track Y-residuals;res_{Y'} [#mum];hits",100,-1000.,1000.);
    hFPixZMinusResYPrime = fs->make<TH1D>("h_FPixZMinusResYPrime","FPix (Z-) track Y-residuals;res_{Y'} [#mum];hits",100,-1000.,1000.);
    					   			
    hBPixResXPull        = fs->make<TH1D>("h_BPixResXPull","BPix track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);     
    hFPixResXPull        = fs->make<TH1D>("h_FPixResXPull","FPix track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);     
    hFPixZPlusResXPull   = fs->make<TH1D>("h_FPixZPlusResXPull","FPix (Z+) track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);
    hFPixZMinusResXPull  = fs->make<TH1D>("h_FPixZMinusResXPull","FPix (Z-) track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);
    					   			
    hBPixResYPull        = fs->make<TH1D>("h_BPixResYPull","BPix track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits",100,-5.,5.);     
    hFPixResYPull        = fs->make<TH1D>("h_FPixResYPull","FPix track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits",100,-5.,5.);     
    hFPixZPlusResYPull   = fs->make<TH1D>("h_FPixZPlusResYPull","FPix (Z+) track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits",100,-5.,5.);
    hFPixZMinusResYPull  = fs->make<TH1D>("h_FPixZMinusResYPull","FPix (Z-) track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits",100,-5.,5.);

    // Strips

    hTIBResXPrime        = fs->make<TH1D>("h_TIBResXPrime","TIB track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hTOBResXPrime        = fs->make<TH1D>("h_TOBResXPrime","TOB track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hTIDResXPrime        = fs->make<TH1D>("h_TIDResXPrime","TID track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
    hTECResXPrime        = fs->make<TH1D>("h_TECResXPrime","TEC track X-residuals;res_{X'} [#mum];hits",100,-1000.,1000.);
  
    hTIBResXPull         = fs->make<TH1D>("h_TIBResXPull","TIB track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);     
    hTOBResXPull         = fs->make<TH1D>("h_TOBResXPull","TOB track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);     
    hTIDResXPull         = fs->make<TH1D>("h_TIDResXPull","TID track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);
    hTECResXPull         = fs->make<TH1D>("h_TECResXPull","TEC track X-pulls;res_{X'}/#sigma_{res_{X'}};hits",100,-5.,5.);

    // hit counts

    hHitCountVsZBPix = fs->make<TH1D>("h_HitCountVsZBpix","Number of BPix hits vs z;hit global z;hits",60,-30,30);
    hHitCountVsZFPix = fs->make<TH1D>("h_HitCountVsZFpix","Number of FPix hits vs z;hit global z;hits",100,-100,100);

    hHitCountVsXBPix = fs->make<TH1D>("h_HitCountVsXBpix","Number of BPix hits vs x;hit global x;hits",20,-20,20);
    hHitCountVsXFPix = fs->make<TH1D>("h_HitCountVsXFpix","Number of FPix hits vs x;hit global x;hits",20,-20,20);

    hHitCountVsYBPix = fs->make<TH1D>("h_HitCountVsYBpix","Number of BPix hits vs y;hit global y;hits",20,-20,20);
    hHitCountVsYFPix = fs->make<TH1D>("h_HitCountVsYFpix","Number of FPix hits vs y;hit global y;hits",20,-20,20);

    hHitCountVsThetaBPix = fs->make<TH1D>("h_HitCountVsThetaBpix","Number of BPix hits vs #theta;hit global #theta;hits",20,0.,TMath::Pi());
    hHitCountVsPhiBPix	 = fs->make<TH1D>("h_HitCountVsPhiBpix","Number of BPix hits vs #phi;hit global #phi;hits",20,-TMath::Pi(),TMath::Pi());
                        
    hHitCountVsThetaFPix = fs->make<TH1D>("h_HitCountVsThetaFpix","Number of FPix hits vs #theta;hit global #theta;hits",40,0.,TMath::Pi());
    hHitCountVsPhiFPix	 = fs->make<TH1D>("h_HitCountVsPhiFpix","Number of FPix hits vs #phi;hit global #phi;hits",20,-TMath::Pi(),TMath::Pi());

    // two sides of FPix

    hHitCountVsZFPixPlus     = fs->make<TH1D>("h_HitCountVsZFPixPlus","Number of FPix(Z+) hits vs z;hit global z;hits",60,15.,60);
    hHitCountVsZFPixMinus    = fs->make<TH1D>("h_HitCountVsZFPixMinus","Number of FPix(Z-) hits vs z;hit global z;hits",100,-60.,-15.);

    hHitCountVsXFPixPlus     = fs->make<TH1D>("h_HitCountVsXFPixPlus","Number of FPix(Z+) hits vs x;hit global x;hits",20,-20,20);
    hHitCountVsXFPixMinus    = fs->make<TH1D>("h_HitCountVsXFPixMinus","Number of FPix(Z-) hits vs x;hit global x;hits",20,-20,20);

    hHitCountVsYFPixPlus     = fs->make<TH1D>("h_HitCountVsYFPixPlus","Number of FPix(Z+) hits vs y;hit global y;hits",20,-20,20);
    hHitCountVsYFPixMinus    = fs->make<TH1D>("h_HitCountVsYFPixMinus","Number of FPix(Z-) hits vs y;hit global y;hits",20,-20,20);

    hHitCountVsThetaFPixPlus = fs->make<TH1D>("h_HitCountVsThetaFPixPlus","Number of FPix(Z+) hits vs #theta;hit global #theta;hits",20,0.,TMath::Pi());
    hHitCountVsPhiFPixPlus   = fs->make<TH1D>("h_HitCountVsPhiFPixPlus","Number of FPix(Z+) hits vs #phi;hit global #phi;hits",20,-TMath::Pi(),TMath::Pi());
                        
    hHitCountVsThetaFPixMinus = fs->make<TH1D>("h_HitCountVsThetaFPixMinus","Number of FPix(Z+) hits vs #theta;hit global #theta;hits",40,0.,TMath::Pi());
    hHitCountVsPhiFPixMinus   = fs->make<TH1D>("h_HitCountVsPhiFPixMinus","Number of FPix(Z+) hits vs #phi;hit global #phi;hits",20,-TMath::Pi(),TMath::Pi());

    TFileDirectory ByLayerResiduals  = fs->mkdir("ByLayerResiduals");    
    barrelLayersResidualsX = bookResidualsHistogram(ByLayerResiduals,4,"X","Res","BPix");
    endcapDisksResidualsX  = bookResidualsHistogram(ByLayerResiduals,6,"X","Res","FPix");
    barrelLayersResidualsY = bookResidualsHistogram(ByLayerResiduals,4,"Y","Res","BPix");
    endcapDisksResidualsY  = bookResidualsHistogram(ByLayerResiduals,6,"Y","Res","FPix");
    
    TFileDirectory ByLayerPulls  = fs->mkdir("ByLayerPulls");    
    barrelLayersPullsX     = bookResidualsHistogram(ByLayerPulls,4,"X","Pull","BPix");
    endcapDisksPullsX      = bookResidualsHistogram(ByLayerPulls,6,"X","Pull","FPix");
    barrelLayersPullsY     = bookResidualsHistogram(ByLayerPulls,4,"Y","Pull","BPix");
    endcapDisksPullsY      = bookResidualsHistogram(ByLayerPulls,6,"Y","Pull","FPix");

    hEta    = fs->make<TH1D>("h_Eta","Track pseudorapidity; track #eta;tracks",100,-TMath::Pi(),TMath::Pi());
    hPhi    = fs->make<TH1D>("h_Phi","Track azimuth; track #phi;tracks",100,-TMath::Pi(),TMath::Pi());
    
    hPhiBarrel       = fs->make<TH1D>("h_PhiBarrel","hPhiBarrel (0<|#eta|<0.8);track #Phi;tracks",100,-TMath::Pi(),TMath::Pi());
    hPhiOverlapPlus  = fs->make<TH1D>("h_PhiOverlapPlus","hPhiOverlapPlus (0.8<#eta<1.4);track #phi;tracks",100,-TMath::Pi(),TMath::Pi());
    hPhiOverlapMinus = fs->make<TH1D>("h_PhiOverlapMinus","hPhiOverlapMinus (-1.4<#eta<-0.8);track #phi;tracks",100,-TMath::Pi(),TMath::Pi());
    hPhiEndcapPlus  = fs->make<TH1D>("h_PhiEndcapPlus","hPhiEndcapPlus (#eta>1.4);track #phi;track",100,-TMath::Pi(),TMath::Pi());
    hPhiEndcapMinus = fs->make<TH1D>("h_PhiEndcapMinus","hPhiEndcapMinus (#eta<1.4);track #phi;tracks",100,-TMath::Pi(),TMath::Pi());
    
    if(!isCosmics_){

      hPhp    = fs->make<TH1D>("h_P_hp","Momentum (high purity);track momentum [GeV];tracks",100,0.,100.);
      hPthp   = fs->make<TH1D>("h_Pt_hp","Transverse Momentum (high purity);track p_{T} [GeV];tracks",100,0.,100.);
      hHithp  = fs->make<TH1D>("h_nHit_hp","Number of hits (high purity);track n. hits;tracks",30,0,30);
      hEtahp  = fs->make<TH1D>("h_Eta_hp","Track pseudorapidity (high purity); track #eta;tracks",100,-TMath::Pi(),TMath::Pi());
      hPhihp  = fs->make<TH1D>("h_Phi_hp","Track azimuth (high purity); track #phi;tracks",100,-TMath::Pi(),TMath::Pi());
      hchi2ndofhp = fs->make<TH1D>("h_chi2ndof_hp","chi2/ndf (high purity);#chi^{2}/ndf;tracks",100,0,5.);
      hchi2Probhp = fs->make<TH1D>("hchi2_Prob_hp","#chi^{2} probability (high purity);#chi^{2}prob_{Track};Number of Tracks",100,0.0,1.);

      hvx  = fs->make<TH1D>("h_vx" ,"Track v_{x} ; track v_{x} [cm];tracks",100,-1.5,1.5);
      hvy  = fs->make<TH1D>("h_vy" ,"Track v_{y} ; track v_{y} [cm];tracks",100,-1.5,1.5);
      hvz  = fs->make<TH1D>("h_vz" ,"Track v_{z} ; track v_{z} [cm];tracks",100,-20.,20.);
      hd0  = fs->make<TH1D>("h_d0" ,"Track d_{0} ; track d_{0} [cm];tracks",100,-1. ,1.);
      hdxy = fs->make<TH1D>("h_dxy","Track d_{xy}; track d_{xy} [cm]; tracks",100,-0.5,0.5);
      hdz  = fs->make<TH1D>("h_dz" ,"Track d_{z} ; track d_{z} [cm]; tracks" ,100,-20,20);

      hd0PVvsphi = fs->make<TH2D>("h2_d0PVvsphi","hd0PVvsphi;track #phi;track d_{0}(PV) [cm]",160,-TMath::Pi(),TMath::Pi(),100,-1.,1.);
      hd0PVvseta = fs->make<TH2D>("h2_d0PVvseta","hdPV0vseta;track #eta;track d_{0}(PV) [cm]",160,-2.5,2.5,100,-1.,1.);
      hd0PVvspt  = fs->make<TH2D>("h2_d0PVvspt" ,"hdPV0vspt;track p_{T};d_{0}(PV) [cm]"      ,50,0.,100.,100,-1,1.);
     
      hdxyBS = fs->make<TH1D>("h_dxyBS","hdxyBS; track d_{xy}(BS) [cm];tracks",100,-0.1,0.1);
      hd0BS  = fs->make<TH1D>("h_d0BS" ,"hd0BS ; track d_{0}(BS) [cm];tracks" ,100,-0.1,0.1);
      hdzBS  = fs->make<TH1D>("h_dzBS" ,"hdzBS ; track d_{z}(BS) [cm];tracks" ,100,-12,12);
      hdxyPV = fs->make<TH1D>("h_dxyPV","hdxyPV; track d_{xy}(PV) [cm];tracks",100,-0.1,0.1);
      hd0PV  = fs->make<TH1D>("h_d0PV" ,"hd0PV ; track d_{0}(PV) [cm];tracks" ,100,-0.15,0.15);
      hdzPV  = fs->make<TH1D>("h_dzPV" ,"hdzPV ; track d_{z}(PV) [cm];tracks" ,100,-0.1,0.1);
      
      hnhTIB = fs->make<TH1D>("h_nHitTIB"  ,"nhTIB;# hits in TIB; tracks"  ,20,0.,20.);
      hnhTID = fs->make<TH1D>("h_nHitTID"  ,"nhTID;# hits in TID; tracks"  ,20,0.,20.);
      hnhTOB = fs->make<TH1D>("h_nHitTOB"  ,"nhTOB;# hits in TOB; tracks"  ,20,0.,20.);
      hnhTEC = fs->make<TH1D>("h_nHitTEC"  ,"nhTEC;# hits in TEC; tracks"  ,20,0.,20.);
      
    } else {
      
      hvx  =fs->make<TH1D>("h_vx","Track v_{x};track v_{x} [cm];tracks",100,-100.,100.);
      hvy  =fs->make<TH1D>("h_vy","Track v_{y};track v_{y} [cm];tracks",100,-100.,100.);
      hvz  =fs->make<TH1D>("h_vz","Track v_{z};track v_{z} [cm];track",100,-100.,100.);
      hd0  =fs->make<TH1D>("h_d0","Track d_{0};track d_{0} [cm];track",100,-100.,100.);
      hdxy =fs->make<TH1D>("h_dxy","Track d_{xy};track d_{xy} [cm];tracks",100,-100,100);
      hdz  =fs->make<TH1D>("h_dz","Track d_{z};track d_{z} [cm];tracks",100,-200,200);

      hd0vsphi=fs->make<TH2D>("h2_d0vsphi","Track d_{0} vs #phi; track #phi;track d_{0} [cm]",160,-3.20,3.20,100,-100.,100.);
      hd0vseta=fs->make<TH2D>("h2_d0vseta","Track d_{0} vs #eta; track #eta;track d_{0} [cm]",160,-3.20,3.20,100,-100.,100.);
      hd0vspt=fs->make<TH2D>("h2_d0vspt","Track d_{0} vs p_{T};track p_{T};track d_{0} [cm]",50,0.,100.,100,-100,100);
      
      hdxyBS=fs->make<TH1D>("h_dxyBS","Track d_{xy}(BS);d_{xy}(BS) [cm];tracks",100,-100.,100.);
      hd0BS=fs->make<TH1D>("h_d0BS","Track d_{0}(BS);d_{0}(BS) [cm];tracks",100,-100.,100.);
      hdzBS=fs->make<TH1D>("h_dzBS","Track d_{z}(BS);d_{z}(BS) [cm];tracks",100,-100.,100.);
      hdxyPV=fs->make<TH1D>("h_dxyPV","Track d_{xy}(PV); d_{xy}(PV) [cm];tracks",100,-100.,100.);
      hd0PV=fs->make<TH1D>("h_d0PV","Track d_{0}(PV); d_{0}(PV) [cm];tracks",100,-100.,100.);
      hdzPV=fs->make<TH1D>("h_dzPV","Track d_{z}(PV); d_{z}(PV) [cm];tracks",100,-100.,100.);

      hnhTIB = fs->make<TH1D>("h_nHitTIB"  ,"nhTIB;# hits in TIB; tracks"  ,30,0.,30.);
      hnhTID = fs->make<TH1D>("h_nHitTID"  ,"nhTID;# hits in TID; tracks"  ,30,0.,30.);
      hnhTOB = fs->make<TH1D>("h_nHitTOB"  ,"nhTOB;# hits in TOB; tracks"  ,30,0.,30.);
      hnhTEC = fs->make<TH1D>("h_nHitTEC"  ,"nhTEC;# hits in TEC; tracks"  ,30,0.,30.);
       
    }

    hnhpxb = fs->make<TH1D>("h_nHitPXB"  ,"nhpxb;# hits in Pixel Barrel; tracks"  ,10,0.,10.);
    hnhpxe = fs->make<TH1D>("h_nHitPXE"  ,"nhpxe;# hits in Pixel Endcap; tracks"  ,10,0.,10.);
    
    hHitComposition =  fs->make<TH1D>("h_hitcomposition","track hit composition;;# hits",6,-0.5,5.5);
    
    pNBpixHitsVsVx = fs->make<TProfile>("p_NpixHits_vs_Vx",
					"n. Barrel Pixel hits vs. v_{x};v_{x} (cm);n. BPix hits",
					20,-20,20);

    pNBpixHitsVsVy = fs->make<TProfile>("p_NpixHits_vs_Vy",
					"n. Barrel Pixel hits vs. v_{y};v_{y} (cm);n. BPix hits",
					20,-20,20);

    pNBpixHitsVsVz = fs->make<TProfile>("p_NpixHits_vs_Vz",
					"n. Barrel Pixel hits vs. v_{z};v_{z} (cm);n. BPix hits",
					20,-100,100);

    TString dets[6] = {"PXB","PXF","TIB","TID","TOB","TEC"};
    
    for(Int_t i=1;i<=hHitComposition->GetNbinsX();i++){
      hHitComposition->GetXaxis()->SetBinLabel(i,dets[i-1]);
    }

    vTrackHistos_.push_back(fs->make<TH1F>("h_tracketa",
					   "Track #eta;#eta_{Track};Number of Tracks",
					   90,-3.,3.));
    vTrackHistos_.push_back(fs->make<TH1F>("h_trackphi",
					   "Track #phi;#phi_{Track};Number of Tracks",
					   90,-3.15,3.15));
    vTrackHistos_.push_back(fs->make<TH1F>("h_trackNumberOfValidHits",
					   "Track # of valid hits;# of valid hits _{Track};Number of Tracks",
					   40,0.,40.));
    vTrackHistos_.push_back(fs->make<TH1F>("h_trackNumberOfLostHits",
					   "Track # of lost hits;# of lost hits _{Track};Number of Tracks",
					   10,0.,10.));
    vTrackHistos_.push_back(fs->make<TH1F>("h_curvature",
					   "Curvature #kappa;#kappa_{Track};Number of Tracks",
					   100,-.05,.05));
    vTrackHistos_.push_back(fs->make<TH1F>("h_curvature_pos",
					   "Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks",
					   100,.0,.05));
    vTrackHistos_.push_back(fs->make<TH1F>("h_curvature_neg",
					   "Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks",
					   100,.0,.05));
    vTrackHistos_.push_back(fs->make<TH1F>("h_diff_curvature",
					   "Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks",
					   100,.0,.05));
    vTrackHistos_.push_back(fs->make<TH1F>("h_chi2",
					   "Track #chi^{2};#chi^{2}_{Track};Number of Tracks",
					   500,-0.01,500.));
    vTrackHistos_.push_back(fs->make<TH1F>("h_chi2Prob",
					   "#chi^{2} probability;Track Prob(#chi^{2},ndof);Number of Tracks",
					   100,0.0,1.));	
    vTrackHistos_.push_back(fs->make<TH1F>("h_normchi2",
					   "#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks",
					   100,-0.01,10.));
    //variable binning for chi2/ndof vs. pT
    double xBins[19]={0.,0.15,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,7.,10.,15.,25.,40.,100.,200.};
    vTrackHistos_.push_back(fs->make<TH1F>("h_pt",
					   "Track p_{T};p_{T}^{track} [GeV];Number of Tracks",
					   250,0.,250));
    vTrackHistos_.push_back(fs->make<TH1F>("h_ptrebin",
					   "Track p_{T};p_{T}^{track} [GeV];Number of Tracks",
					   18,xBins));
    vTrackHistos_.push_back(fs->make<TH1F>("h_ptResolution",
					   "#delta_{p_{T}}/p_{T}^{track};#delta_{p_{T}}/p_{T}^{track};Number of Tracks",
					   100,0.,0.5));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_d0_vs_phi",
						 "Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT [cm]",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_dz_vs_phi",
						 "Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT [cm]",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_d0_vs_eta",
						 "Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT [cm]",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_dz_vs_eta",
						 "Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT [cm]",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2_vs_phi",
						 "#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2Prob_vs_phi",
						 "#chi^{2} probablility vs. #phi;#phi_{Track};#LT #chi^{2} probability#GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2Prob_vs_d0",
						 "#chi^{2} probablility vs. |d_{0}|;|d_{0}|[cm];#LT #chi^{2} probability#GT",
						 100,0,80));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2Prob_vs_dz",
						 "#chi^{2} probablility vs. dz;d_{z} [cm];#LT #chi^{2} probability#GT",
						 100,-30,30));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_normchi2_vs_phi",
						 "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2_vs_eta",
						 "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_normchi2_vs_pt",
						 "norm #chi^{2} vs. p_{T}_{Track}; p_{T}_{Track};#LT #chi^{2}/ndof #GT",
						 18,xBins));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_normchi2_vs_p",
						 "#chi^{2}/ndof vs. p_{Track};p_{Track};#LT #chi^{2}/ndof #GT",
						 18,xBins));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_chi2Prob_vs_eta",
						 "#chi^{2} probability vs. #eta;#eta_{Track};#LT #chi^{2} probability #GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_normchi2_vs_eta",
						 "#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_kappa_vs_phi",
						 "#kappa vs. #phi;#phi_{Track};#kappa",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_kappa_vs_eta",
						 "#kappa vs. #eta;#eta_{Track};#kappa",
						 100,-3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_ptResolution_vs_phi",
						 "#delta_{p_{T}}/p_{T}^{track};#phi^{track};#delta_{p_{T}}/p_{T}^{track}",
						 100, -3.15,3.15));
    vTrackProfiles_.push_back(fs->make<TProfile>("p_ptResolution_vs_eta",
						 "#delta_{p_{T}}/p_{T}^{track};#eta^{track};#delta_{p_{T}}/p_{T}^{track}",
						 100, -3.15,3.15));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_d0_vs_phi",
					     "Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0} [cm]",
					     100, -3.15, 3.15, 100,-1.,1.) );
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_dz_vs_phi",
					     "Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z} [cm]",
					     100, -3.15, 3.15, 100,-100.,100.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_d0_vs_eta",
					     "Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0} [cm]",
					     100, -3.15, 3.15, 100,-1.,1.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_dz_vs_eta",
					     "Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z} [cm]",
					     100, -3.15, 3.15, 100,-100.,100.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_chi2_vs_phi",
					     "#chi^{2} vs. #phi;#phi_{Track};#chi^{2}",
					     100, -3.15, 3.15, 500, 0., 500.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_chi2Prob_vs_phi",
					     "#chi^{2} probability vs. #phi;#phi_{Track};#chi^{2} probability",
					     100, -3.15, 3.15, 100, 0., 1.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_chi2Prob_vs_d0",
					     "#chi^{2} probability vs. |d_{0}|;|d_{0}| [cm];#chi^{2} probability",
					     100, 0, 80, 100, 0., 1.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_normchi2_vs_phi",
					     "#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof",
					     100, -3.15, 3.15, 100, 0., 10.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_chi2_vs_eta",
					     "#chi^{2} vs. #eta;#eta_{Track};#chi^{2}",
					     100, -3.15, 3.15, 500, 0., 500.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_chi2Prob_vs_eta",
					     "#chi^{2} probaility vs. #eta;#eta_{Track};#chi^{2} probability",
					     100, -3.15, 3.15, 100, 0., 1.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_normchi2_vs_eta",
					     "#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof",
					     100,-3.15,3.15, 100, 0., 10.));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_kappa_vs_phi",
					     "#kappa vs. #phi;#phi_{Track};#kappa",
					     100,-3.15,3.15, 100, .0,.05));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_kappa_vs_eta",
					     "#kappa vs. #eta;#eta_{Track};#kappa",
					     100,-3.15,3.15, 100, .0,.05));
    vTrack2DHistos_.push_back(fs->make<TH2F>("h2_normchi2_vs_kappa",
					     "#kappa vs. #chi^{2}/ndof;#chi^{2}/ndof;#kappa",
					     100,0.,10, 100,-.03,.03));
    
    firstEvent_=true;

  }//beginJob

  virtual void endJob()
  { 
    std::cout<< "*******************************" << std::endl;
    std::cout<< "Events run in total: " << ievt << std::endl;
    std::cout<< "n. tracks: "<<itrks<<std::endl;
    std::cout<< "*******************************" << std::endl;
    
    Int_t nFiringTriggers = triggerMap_.size();
    std::cout<<"firing triggers: "<<nFiringTriggers<<std::endl;
    std::cout<< "*******************************" << std::endl;

    tksByTrigger_= fs->make<TH1D>("tksByTrigger","tracks by HLT path;;% of # traks",nFiringTriggers,-0.5,nFiringTriggers-0.5);
    evtsByTrigger_= fs->make<TH1D>("evtsByTrigger","events by HLT path;;% of # events",nFiringTriggers,-0.5,nFiringTriggers-0.5);
   
    Int_t i =0;
    
    for(std::map<std::string,std::pair<int,int> >::iterator it=triggerMap_.begin(); it!=triggerMap_.end(); ++it){
      i++;
      
      Double_t trkpercent = ((it->second).second)*100./Double_t(itrks);
      Double_t evtpercent = ((it->second).first)*100./Double_t(ievt);

      std::cout.precision(4);

      std::cout<<"HLT path: "<<std::setw(60) << left << it->first << " | events firing: " << right << std::setw(8)<<(it->second).first << " ("<< setw(8) << fixed << evtpercent <<"%)"
	       <<" | tracks collected: " <<std::setw(10)<< (it->second).second << " ("<< setw(8) << fixed << trkpercent <<"%)"<<  '\n';
      
      tksByTrigger_->SetBinContent(i,trkpercent);
      tksByTrigger_->GetXaxis()->SetBinLabel(i,(it->first).c_str());
      
      evtsByTrigger_->SetBinContent(i,evtpercent);
      evtsByTrigger_->GetXaxis()->SetBinLabel(i,(it->first).c_str());
      
    }

    Int_t nRuns    = conditionsMap_.size();
  
    vector<int> theRuns_;
    for(map<int,std::pair<int,float> >::iterator it = conditionsMap_.begin(); it != conditionsMap_.end(); ++it) {
      theRuns_.push_back(it->first);
    }

    sort(theRuns_.begin(),theRuns_.end());
    Int_t runRange = theRuns_[theRuns_.size()-1] - theRuns_[0]+1;
    
    std::cout<< "*******************************" << std::endl;
    std::cout<<"first run: "<<theRuns_[0]<<std::endl;
    std::cout<<"last run:  "<<theRuns_[theRuns_.size()-1]<<std::endl;
    std::cout<<"considered runs: "<<nRuns<<std::endl;
    std::cout<< "*******************************" << std::endl;
    
    /*
      // all runs on display
      modeByRun_ = fs->make<TH1D>("modeByRun","Strip APV mode by run number;;APV mode (-1=deco,+1=peak)",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
      fieldByRun_= fs->make<TH1D>("fieldByRun","CMS B-field intensity by run number;;B-field intensity [T]",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
    
      tracksByRun_ =  fs->make<TH1D>("tracksByRun","n. AlCaReco Tracks by run number;;n. of tracks",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
      hitsByRun_   =  fs->make<TH1D>("histByRun","n. of hits by run number;;n. of hits",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
      
      hitsinBPixByRun_ =  fs->make<TH1D>("histinBPixByRun","n. of hits in BPix by run number;;n. of BPix hits",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
      hitsinFPixByRun_ =  fs->make<TH1D>("histinFPixByRun","n. of hits in FPIx by run number;;n. of FPix hits",runRange,theRuns_[0]-0.5,theRuns_[theRuns_.size()-1]+0.5);
    */


    modeByRun_       = fs->make<TH1D>("modeByRun","Strip APV mode by run number;;APV mode (-1=deco,+1=peak)",nRuns,-0.5,nRuns-0.5);
    fieldByRun_      = fs->make<TH1D>("fieldByRun","CMS B-field intensity by run number;;B-field intensity [T]",nRuns,-0.5,nRuns-0.5);
    
    tracksByRun_     =  fs->make<TH1D>("tracksByRun","n. AlCaReco Tracks by run number;;n. of tracks",nRuns,-0.5,nRuns-0.5);
    hitsByRun_       =  fs->make<TH1D>("histByRun","n. of hits by run number;;n. of hits",nRuns,-0.5,nRuns-0.5);              
    
    trackRatesByRun_ = fs->make<TH1D>("trackRatesByRun","rate of AlCaReco Tracks by run number;;n. of tracks/s",nRuns,-0.5,nRuns-0.5);  
    eventRatesByRun_ = fs->make<TH1D>("eventRatesByRun","rate of AlCaReco Events by run number;;n. of events/s",nRuns,-0.5,nRuns-0.5);              

    hitsinBPixByRun_ =  fs->make<TH1D>("histinBPixByRun","n. of hits in BPix by run number;;n. of BPix hits",nRuns,-0.5,nRuns-0.5);
    hitsinFPixByRun_ =  fs->make<TH1D>("histinFPixByRun","n. of hits in FPix by run number;;n. of FPix hits",nRuns,-0.5,nRuns-0.5);

    Int_t indexing(0);
    for(Int_t the_r=theRuns_[0];the_r<=theRuns_[theRuns_.size()-1];the_r++){

      if(conditionsMap_.find(the_r)->second.first!=0){ 
	
	indexing++;
	double runTime = timeMap_.find(the_r)->second;

	std::cout<<"run:"<<the_r<<" | isPeak: "<< std::setw(4) << conditionsMap_.find(the_r)->second.first
		 <<"| B-field: "<<conditionsMap_.find(the_r)->second.second<<" [T]"
		 <<"| events: " <<setw(10) << runInfoMap_.find(the_r)->second.first  << "(rate: "<< setw(10) << (runInfoMap_.find(the_r)->second.first)/runTime  << " ev/s)"
		 << ", tracks "<< setw(10) << runInfoMap_.find(the_r)->second.second << "(rate: "<< setw(10) << (runInfoMap_.find(the_r)->second.second)/runTime  << " trk/s)" <<std::endl;
      
	// int the_bin = modeByRun_->GetXaxis()->FindBin(the_r);
	modeByRun_->SetBinContent(indexing,conditionsMap_.find(the_r)->second.first);
	modeByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	fieldByRun_->SetBinContent(indexing,conditionsMap_.find(the_r)->second.second);
	fieldByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	
	tracksByRun_->SetBinContent(indexing,runInfoMap_.find(the_r)->second.first);
	tracksByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	hitsByRun_->SetBinContent(indexing,runInfoMap_.find(the_r)->second.second);
	hitsByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	
	hitsinBPixByRun_->SetBinContent(indexing,(runHitsMap_.find(the_r)->second)[0]);
	hitsinBPixByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	hitsinFPixByRun_->SetBinContent(indexing,(runHitsMap_.find(the_r)->second)[1]);
	hitsinFPixByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	
	trackRatesByRun_->SetBinContent(indexing,(runInfoMap_.find(the_r)->second.second)/runTime);
	trackRatesByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));
	eventRatesByRun_->SetBinContent(indexing,(runInfoMap_.find(the_r)->second.first)/runTime);
	eventRatesByRun_->GetXaxis()->SetBinLabel(indexing,Form("%d",the_r));

	
	constexpr const char *subdets[] {"BPix","FPix","TIB","TID","TOB","TEC"};

	std::cout<< "*******************************" << std::endl;
	std::cout << "Hits by Sub-det"<< std::endl;
	int si=0;
	for(const auto& entry: runHitsMap_.find(the_r)->second){
	  std::cout << subdets[si] <<" " << entry << std::endl;
	  si++;
	}
	std::cout<< "*******************************" << std::endl;
	
      }

      // modeByRun_->GetXaxis()->SetBinLabel(the_r-theRuns_[0]+1,(const char*)the_r);
    }

    // DMRs
    
    TFileDirectory DMeanR  = fs->mkdir("DMRs");

    DMRBPixX_ = DMeanR.make<TH1D>("DMRBPix-X","DMR of BPix-X;mean of X-residuals;modules",100.,-200,200);
    DMRBPixY_ = DMeanR.make<TH1D>("DMRBPix-Y","DMR of BPix-Y;mean of Y-residuals;modules",100.,-200,200);

    DMRFPixX_ = DMeanR.make<TH1D>("DMRFPix-X","DMR of FPix-X;mean of X-residuals;modules",100.,-200,200);
    DMRFPixY_ = DMeanR.make<TH1D>("DMRFPix-Y","DMR of FPix-Y;mean of Y-residuals;modules",100.,-200,200);

    DMRTIB_   = DMeanR.make<TH1D>("DMRTIB","DMR of TIB;mean of X-residuals;modules",100.,-200,200);
    DMRTOB_   = DMeanR.make<TH1D>("DMRTOB","DMR of TOB;mean of X-residuals;modules",100.,-200,200);

    DMRTID_   = DMeanR.make<TH1D>("DMRTID","DMR of TID;mean of X-residuals;modules",100.,-200,200);
    DMRTEC_   = DMeanR.make<TH1D>("DMRTEC","DMR of TEC;mean of X-residuals;modules",100.,-200,200);

    // DRnRs
    TFileDirectory DRnRs = fs->mkdir("DRnRs");

    DRnRBPixX_ = DRnRs.make<TH1D>("DRnRBPix-X","DRnR of BPix-X;rms of normalized X-residuals;modules",100.,0.,3.);
    DRnRBPixY_ = DRnRs.make<TH1D>("DRnRBPix-Y","DRnR of BPix-Y;rms of normalized Y-residuals;modules",100.,0.,3.);

    DRnRFPixX_ = DRnRs.make<TH1D>("DRnRFPix-X","DRnR of FPix-X;rms of normalized X-residuals;modules",100.,0.,3.);
    DRnRFPixY_ = DRnRs.make<TH1D>("DRnRFPix-Y","DRnR of FPix-Y;rms of normalized Y-residuals;modules",100.,0.,3.);

    DRnRTIB_   = DRnRs.make<TH1D>("DRnRTIB","DRnR of TIB;rms of normalized X-residuals;modules",100.,0.,3.);
    DRnRTOB_   = DRnRs.make<TH1D>("DRnRTOB","DRnR of TOB;rms of normalized Y-residuals;modules",100.,0.,3.);

    DRnRTID_   = DRnRs.make<TH1D>("DRnRTID","DRnR of TID;rms of normalized X-residuals;modules",100.,0.,3.);
    DRnRTEC_   = DRnRs.make<TH1D>("DRnRTEC","DRnR of TEC;rms of normalized Y-residuals;modules",100.,0.,3.);

    // pixel

    for(auto &bpixid : resDetailsBPixX_){
      DMRBPixX_->Fill(bpixid.second.runningMeanOfRes_);
      pixelmap->fillBarrelBin("DMRsX", bpixid.first,bpixid.second.runningMeanOfRes_);

      if(bpixid.second.hitCount<2)
	DRnRBPixX_->Fill(-1);
      else
	DRnRBPixX_->Fill(sqrt(bpixid.second.runningNormVarOfRes_/(bpixid.second.hitCount-1)));

    }
    
    for(auto &bpixid : resDetailsBPixY_){
      DMRBPixY_->Fill(bpixid.second.runningMeanOfRes_);
      pixelmap->fillBarrelBin("DMRsY", bpixid.first,bpixid.second.runningMeanOfRes_);

      if(bpixid.second.hitCount<2)
	DRnRBPixY_->Fill(-1);
      else
	DRnRBPixY_->Fill(sqrt(bpixid.second.runningNormVarOfRes_/(bpixid.second.hitCount-1)));
    }

    for(auto &fpixid : resDetailsFPixX_){
      DMRFPixX_->Fill(fpixid.second.runningMeanOfRes_);
      pixelmap->fillForwardBin("DMRsX", fpixid.first,fpixid.second.runningMeanOfRes_);

      if(fpixid.second.hitCount<2)
	DRnRFPixX_->Fill(-1);
      else 
	DRnRFPixX_->Fill(sqrt(fpixid.second.runningNormVarOfRes_/(fpixid.second.hitCount-1)));

    }

    for(auto &fpixid : resDetailsFPixY_){
      DMRFPixY_->Fill(fpixid.second.runningMeanOfRes_);
      pixelmap->fillForwardBin("DMRsY", fpixid.first,fpixid.second.runningMeanOfRes_);
      
      if(fpixid.second.hitCount<2)
	DRnRFPixY_->Fill(-1);
      else
	DRnRFPixY_->Fill(sqrt(fpixid.second.runningNormVarOfRes_/(fpixid.second.hitCount-1)));
      
    }
    
    // strips

    for(auto &tibid : resDetailsTIB_){
      DMRTIB_->Fill(tibid.second.runningMeanOfRes_);

          if(tibid.second.hitCount<2)
	DRnRTIB_->Fill(-1);
      else
	DRnRTIB_->Fill(sqrt(tibid.second.runningNormVarOfRes_/(tibid.second.hitCount-1)));

    }
    
    for(auto &tobid : resDetailsTOB_){
      DMRTOB_->Fill(tobid.second.runningMeanOfRes_);

      if(tobid.second.hitCount<2)
	DRnRTOB_->Fill(-1);
      else
	DRnRTOB_->Fill(sqrt(tobid.second.runningNormVarOfRes_/(tobid.second.hitCount-1)));

    }

    for(auto &tidid : resDetailsTID_){
      DMRTID_->Fill(tidid.second.runningMeanOfRes_);

          if(tidid.second.hitCount<2)
	DRnRTID_->Fill(-1);
      else 
	DRnRTID_->Fill(sqrt(tidid.second.runningNormVarOfRes_/(tidid.second.hitCount-1)));

    }

    for(auto &tecid : resDetailsTEC_){
      DMRTEC_->Fill(tecid.second.runningMeanOfRes_);

      if(tecid.second.hitCount<2)
	DRnRTEC_->Fill(-1);
      else
	DRnRTEC_->Fill(sqrt(tecid.second.runningNormVarOfRes_/(tecid.second.hitCount-1)));
    }
    

    std::cout<<"n. of bpix modules "<<resDetailsBPixX_.size() << std::endl;
    std::cout<<"n. of fpix modules "<<resDetailsFPixX_.size() << std::endl;
	
    pmap->save(true,0,0,"pixelmap.pdf",600,800);
    pmap->save(true,0,0,"pixelmap.png",500,750); 

    tmap->save(true,0,0,"trackermap.pdf");
    tmap->save(true,0,0,"trackermap.png"); 

    gStyle->SetPalette(kRainBow);
    pixelmap->beautifyAllHistograms();

    TCanvas cBX("CanvXBarrel", "CanvXBarrel", 1200, 1000);
    pixelmap->DrawBarrelMaps("DMRsX", cBX);
    cBX.SaveAs("pixelBarrelDMR_x.png");

    TCanvas cFX("CanvXForward", "CanvXForward", 1600, 1000);
    pixelmap->DrawForwardMaps("DMRsX", cFX);
    cFX.SaveAs("pixelForwardDMR_x.png");

    TCanvas cBY("CanvYBarrel", "CanvYBarrel", 1200, 1000);
    pixelmap->DrawBarrelMaps("DMRsY", cBY);
    cBY.SaveAs("pixelBarrelDMR_y.png");

    TCanvas cFY("CanvXForward", "CanvXForward", 1600, 1000);
    pixelmap->DrawForwardMaps("DMRsY", cFY);
    cFY.SaveAs("pixelForwardDMR_y.png");

  }

  bool isHit2D(const TrackingRecHit &hit) 
  {
    bool countStereoHitAs2D_ = true;
    // we count SiStrip stereo modules as 2D if selected via countStereoHitAs2D_
    // (since they provide theta information)
    if (!hit.isValid() ||
	(hit.dimension() < 2 && !countStereoHitAs2D_ && !dynamic_cast<const SiStripRecHit1D*>(&hit))){
      return false; // real RecHit1D - but SiStripRecHit1D depends on countStereoHitAs2D_
    } else {
      const DetId detId(hit.geographicalId());
      if (detId.det() == DetId::Tracker) {
	if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
	  return true; // pixel is always 2D
	} else { // should be SiStrip now
	  const SiStripDetId stripId(detId);
	  if (stripId.stereo()) return countStereoHitAs2D_; // stereo modules
	  else if (dynamic_cast<const SiStripRecHit1D*>(&hit)
		   || dynamic_cast<const SiStripRecHit2D*>(&hit)) return false; // rphi modules hit
	  //the following two are not used any more since ages...
	  else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&hit)) return true; // matched is 2D
	  else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&hit)) {
	    const ProjectedSiStripRecHit2D* pH = static_cast<const ProjectedSiStripRecHit2D*>(&hit);
	    return (countStereoHitAs2D_ && this->isHit2D(pH->originalHit())); // depends on original...
	  } else {
	    edm::LogError("UnkownType") << "@SUB=AlignmentTrackSelector::isHit2D"
					<< "Tracker hit not in pixel, neither SiStripRecHit[12]D nor "
					<< "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
	    return false;
	  }
	}
      } else { // not tracker??
	edm::LogWarning("DetectorMismatch") << "@SUB=AlignmentTrackSelector::isHit2D"
					    << "Hit not in tracker with 'official' dimension >=2.";
	return true; // dimension() >= 2 so accept that...
      }
    }
    // never reached...
  }

  //*************************************************************
  // Generic booker function
  //*************************************************************
  std::map<unsigned int, TH1D*> bookResidualsHistogram(TFileDirectory dir,
						       unsigned int theNLayers,
						       TString resType,
						       TString varType,
						       TString detType){
    TH1F::SetDefaultSumw2(kTRUE);
    
    std::pair<Double_t,Double_t> limits;
    
    if(varType.Contains("Res")){
      limits = std::make_pair(-1000.,1000);
    } else {
      limits = std::make_pair(-3.,3.);
    }

    std::map<unsigned int, TH1D*> h;
    
    for(unsigned int i=1; i<=theNLayers;i++){
      
      const char* name_;   
      const char* title_; 
      const char* xAxisTitle_;

      if(varType.Contains("Res")){
	xAxisTitle_ = (TString("res_{"+resType+"'} [#mum]")).Data();
      } else {
	xAxisTitle_ = (TString("res_{"+resType+"'}/#sigma_{res_{"+resType+"`}}")).Data();
      }
      
      unsigned int side=-1;
      unsigned int plane=i;

      if(detType.Contains("FPix")){
	side  = (i-1)/3+1;
	plane = (i-1)%3+1;

	TString theSide="";
	if(side==1) theSide="Z-";  
	else theSide="Z+"; 

	name_  =  Form("h_%s%s%s_side%i_disk%i",detType.Data(),varType.Data(),resType.Data(),side,plane);
	title_ =  Form("%s (%s, disk %i) track %s-%s;%s;hits",detType.Data(),theSide.Data(),plane,resType.Data(),varType.Data(),xAxisTitle_);

      } else {

	name_  =  Form("h_%s%s%s_layer%i",detType.Data(),varType.Data(),resType.Data(),i);
	title_ =  Form("%s (layer %i) track %s-%s;%s;hits",detType.Data(),i,resType.Data(),varType.Data(),xAxisTitle_);

	//std::cout<<"bookResidualsHistogram(): "<<i<<" layer:"<<i<<std::endl;

      }


      h[i] = dir.make<TH1D>(name_,
			    title_,
			    100,limits.first,limits.second); 
    }
  
    return h;
    
  }

  //*************************************************************
  // Generic filler function
  //*************************************************************
  void fillByIndex(std::map<unsigned int, TH1D*>& h, unsigned int index, double x)
  {
    if(h.count(index)!=0){
      //if(TString(h[index]->GetName()).Contains("BPix"))
      //std::cout<<"fillByIndex() index: "<< index << " filling histogram: "<< h[index]->GetName() << std::endl;
      
      double min = h[index]->GetXaxis()->GetXmin();
      double max = h[index]->GetXaxis()->GetXmax();
      if(x<min) h[index]->Fill(min);
      else if (x>=max) h[index]->Fill(0.99999*max);
      else h[index]->Fill(x);
    }
  }
  
  //*************************************************************
  // Implementation of the online variance algorithm
  // as in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
  //*************************************************************
  void updateOnlineMomenta( std::map<int,running::Estimators >  &myDetails, int theID, float the_data,float the_pull){

    myDetails[theID].hitCount +=1;

    float delta   = 0;
    float n_delta = 0;

    if(myDetails[theID].hitCount!=1){
      delta    = the_data - myDetails[theID].runningMeanOfRes_;
      n_delta  = the_pull - myDetails[theID].runningNormMeanOfRes_;
      myDetails[theID].runningMeanOfRes_ += (delta/myDetails[theID].hitCount);
      myDetails[theID].runningNormMeanOfRes_ += (n_delta/myDetails[theID].hitCount);
    } else {
      myDetails[theID].runningMeanOfRes_ = the_data;
      myDetails[theID].runningNormMeanOfRes_ = the_pull;
    }
    
    float delta2   = the_data - myDetails[theID].runningMeanOfRes_;
    float n_delta2 = the_pull - myDetails[theID].runningNormMeanOfRes_;

    myDetails[theID].runningVarOfRes_ += delta*delta2;
    myDetails[theID].runningNormVarOfRes_ += n_delta*n_delta2;
    
  }

};


DEFINE_FWK_MODULE(DMRChecker_v2);

