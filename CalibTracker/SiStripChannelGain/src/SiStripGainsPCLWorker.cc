#include "CalibTracker/SiStripChannelGain/interface/SiStripGainsPCLWorker.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>

//********************************************************************************//
SiStripGainsPCLWorker::SiStripGainsPCLWorker(const edm::ParameterSet& iConfig) :
  NEvent(0),
  NTrack(0),
  NClusterStrip(0),
  NClusterPixel(0),
  NStripAPVs(0),
  NPixelDets(0),
  SRun(1<<31),
  ERun(0),
  bareTkGeomPtr_(nullptr)
{
  
  MinTrackMomentum        = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"        ,  3.0);
  MaxTrackMomentum        = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"        ,  99999.0);
  MinTrackEta             = iConfig.getUntrackedParameter<double>  ("minTrackEta"             , -5.0);
  MaxTrackEta             = iConfig.getUntrackedParameter<double>  ("maxTrackEta"             ,  5.0);
  MaxNrStrips             = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"             ,  2);
  MinTrackHits            = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"            ,  8);
  MaxTrackChiOverNdf      = iConfig.getUntrackedParameter<double>  ("MaxTrackChiOverNdf"      ,  3);
  MaxTrackingIteration    = iConfig.getUntrackedParameter<int>     ("MaxTrackingIteration"    ,  7);
  AllowSaturation         = iConfig.getUntrackedParameter<bool>    ("AllowSaturation"         ,  false);
  FirstSetOfConstants     = iConfig.getUntrackedParameter<bool>    ("FirstSetOfConstants"     ,  true);
  Validation              = iConfig.getUntrackedParameter<bool>    ("Validation"              ,  false);
  OldGainRemoving         = iConfig.getUntrackedParameter<bool>    ("OldGainRemoving"         ,  false);
  useCalibration          = iConfig.getUntrackedParameter<bool>    ("UseCalibration"          ,  false);
  doChargeMonitorPerPlane = iConfig.getUntrackedParameter<bool>    ("doChargeMonitorPerPlane" ,  false);
  m_DQMdir                = iConfig.getUntrackedParameter<std::string>  ("DQMdir"             , "AlCaReco/SiStripGains");
  m_calibrationMode       = iConfig.getUntrackedParameter<std::string>  ("calibrationMode"    , "StdBunch");
  VChargeHisto            = iConfig.getUntrackedParameter<std::vector<std::string> >  ("ChargeHisto");

  //Set the monitoring element tag and store
  dqm_tag_.reserve(7);
  dqm_tag_.clear();
  dqm_tag_.push_back( "StdBunch" );      // statistic collection from Standard Collision Bunch @ 3.8 T
  dqm_tag_.push_back( "StdBunch0T" );    // statistic collection from Standard Collision Bunch @ 0 T
  dqm_tag_.push_back( "AagBunch" );      // statistic collection from First Collision After Abort Gap @ 3.8 T
  dqm_tag_.push_back( "AagBunch0T" );    // statistic collection from First Collision After Abort Gap @ 0 T
  dqm_tag_.push_back( "IsoMuon" );       // statistic collection from Isolated Muon @ 3.8 T
  dqm_tag_.push_back( "IsoMuon0T" );     // statistic collection from Isolated Muon @ 0 T
  dqm_tag_.push_back( "Harvest" );       // statistic collection: Harvest
  
  Charge_Vs_Index.insert( Charge_Vs_Index.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTIB.insert( Charge_Vs_PathlengthTIB.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTOB.insert( Charge_Vs_PathlengthTOB.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTIDP.insert( Charge_Vs_PathlengthTIDP.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTIDM.insert( Charge_Vs_PathlengthTIDM.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTECP1.insert( Charge_Vs_PathlengthTECP1.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTECP2.insert( Charge_Vs_PathlengthTECP2.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTECM1.insert( Charge_Vs_PathlengthTECM1.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTECM2.insert( Charge_Vs_PathlengthTECM2.begin(), dqm_tag_.size(), 0);

  // configure token for gathering the ntuple variables 
  edm::ParameterSet swhallowgain_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("gain");

  std::string label = swhallowgain_pset.getUntrackedParameter<std::string>("label");
  CalibPrefix_ = swhallowgain_pset.getUntrackedParameter<std::string>("prefix");
  CalibSuffix_ = swhallowgain_pset.getUntrackedParameter<std::string>("suffix");
  
  trackindex_token_     = consumes<std::vector<int>            >(edm::InputTag(label, CalibPrefix_ + "trackindex"    + CalibSuffix_)); 
  rawid_token_          = consumes<std::vector<unsigned int>   >(edm::InputTag(label, CalibPrefix_ + "rawid"         + CalibSuffix_)); 
  localdirx_token_      = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "localdirx"     + CalibSuffix_)); 
  localdiry_token_      = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "localdiry"     + CalibSuffix_)); 
  localdirz_token_      = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "localdirz"     + CalibSuffix_)); 
  firststrip_token_     = consumes<std::vector<unsigned short> >(edm::InputTag(label, CalibPrefix_ + "firststrip"    + CalibSuffix_)); 
  nstrips_token_        = consumes<std::vector<unsigned short> >(edm::InputTag(label, CalibPrefix_ + "nstrips"       + CalibSuffix_)); 
  saturation_token_     = consumes<std::vector<bool>           >(edm::InputTag(label, CalibPrefix_ + "saturation"    + CalibSuffix_)); 
  overlapping_token_    = consumes<std::vector<bool>           >(edm::InputTag(label, CalibPrefix_ + "overlapping"   + CalibSuffix_)); 
  farfromedge_token_    = consumes<std::vector<bool>           >(edm::InputTag(label, CalibPrefix_ + "farfromedge"   + CalibSuffix_)); 
  charge_token_         = consumes<std::vector<unsigned int>   >(edm::InputTag(label, CalibPrefix_ + "charge"        + CalibSuffix_)); 
  path_token_           = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "path"          + CalibSuffix_)); 
  chargeoverpath_token_ = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "chargeoverpath"+ CalibSuffix_)); 
  amplitude_token_      = consumes<std::vector<unsigned char>  >(edm::InputTag(label, CalibPrefix_ + "amplitude"     + CalibSuffix_)); 
  gainused_token_       = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "gainused"      + CalibSuffix_)); 
  gainusedTick_token_   = consumes<std::vector<double>         >(edm::InputTag(label, CalibPrefix_ + "gainusedTick"  + CalibSuffix_));
  
  edm::ParameterSet evtinfo_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("evtinfo");
  label        = evtinfo_pset.getUntrackedParameter<std::string>("label");
  EventPrefix_ = evtinfo_pset.getUntrackedParameter<std::string>("prefix");
  EventSuffix_ = evtinfo_pset.getUntrackedParameter<std::string>("suffix");
  TrigTech_token_ = consumes<std::vector<bool> >(edm::InputTag(label, EventPrefix_ + "TrigTech" + EventSuffix_));
  
  edm::ParameterSet track_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("tracks");
  label        = track_pset.getUntrackedParameter<std::string>("label");
  TrackPrefix_ = track_pset.getUntrackedParameter<std::string>("prefix");
  TrackSuffix_ = track_pset.getUntrackedParameter<std::string>("suffix");
  
  trackchi2ndof_token_  = consumes<std::vector<double>       >(edm::InputTag(label, TrackPrefix_ + "chi2ndof"  + TrackSuffix_)); 
  trackp_token_	        = consumes<std::vector<float>        >(edm::InputTag(label, TrackPrefix_ + "momentum"  + TrackSuffix_)); 
  trackpt_token_	= consumes<std::vector<float>        >(edm::InputTag(label, TrackPrefix_ + "pt"        + TrackSuffix_)); 
  tracketa_token_	= consumes<std::vector<double>       >(edm::InputTag(label, TrackPrefix_ + "eta"       + TrackSuffix_)); 
  trackphi_token_	= consumes<std::vector<double>       >(edm::InputTag(label, TrackPrefix_ + "phi"       + TrackSuffix_)); 
  trackhitsvalid_token_ = consumes<std::vector<unsigned int> >(edm::InputTag(label, TrackPrefix_ + "hitsvalid" + TrackSuffix_)); 
  trackalgo_token_      = consumes<std::vector<int>          >(edm::InputTag(label, TrackPrefix_ + "algo"      + TrackSuffix_));
  
}

//********************************************************************************//
void 
SiStripGainsPCLWorker::dqmBeginRun(edm::Run const& run, const edm::EventSetup& iSetup){
  
  using namespace edm;

  this->checkBookAPVColls(iSetup); // check whether APV colls are booked and do so if not yet done
  
  edm::ESHandle<SiStripGain> gainHandle;
  iSetup.get<SiStripGainRcd>().get(gainHandle);
  if(!gainHandle.isValid()){edm::LogError("SiStripGainPCLWorker")<< "gainHandle is not valid\n"; exit(0);}

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);

  for(unsigned int a=0;a<APVsCollOrdered.size();a++){

    std::shared_ptr<stAPVGain> APV = APVsCollOrdered[a];

    if(APV->SubDet==PixelSubdetector::PixelBarrel || APV->SubDet==PixelSubdetector::PixelEndcap) continue;
    
    APV->isMasked      = SiStripQuality_->IsApvBad(APV->DetId,APV->APVId);
    	  
    if(gainHandle->getNumberOfTags()!=2){edm::LogError("SiStripGainPCLWorker")<< "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";fflush(stdout);exit(0);};		   
    float newPreviousGain = gainHandle->getApvGain(APV->APVId,gainHandle->getRange(APV->DetId, 1),1);
    if(APV->PreviousGain!=1 and newPreviousGain!=APV->PreviousGain)edm::LogWarning("SiStripGainPCLWorker")<< "WARNING: ParticleGain in the global tag changed\n";
    APV->PreviousGain = newPreviousGain;
    
    float newPreviousGainTick = gainHandle->getApvGain(APV->APVId,gainHandle->getRange(APV->DetId, 0),0);
    if(APV->PreviousGainTick!=1 and newPreviousGainTick!=APV->PreviousGainTick){
      edm::LogWarning("SiStripGainPCLWorker")<< "WARNING: TickMarkGain in the global tag changed\n"<< std::endl
					     <<" APV->SubDet: "<< APV->SubDet << " APV->APVId:" << APV->APVId << std::endl
					     <<" APV->PreviousGainTick: "<<APV->PreviousGainTick<<" newPreviousGainTick: "<<newPreviousGainTick<<std::endl;
    }
    APV->PreviousGainTick = newPreviousGainTick;  	  
  }

}

//********************************************************************************//
// ------------ method called for each event  ------------
void
SiStripGainsPCLWorker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  //  this->checkBookAPVColls(iSetup); // check whether APV colls are booked and do so if not yet done

  eventnumber   = iEvent.id().event();
  runnumber     = iEvent.id().run();

  auto handle01 = connect(TrigTech      , TrigTech_token_      , iEvent);
  auto handle02 = connect(trackchi2ndof , trackchi2ndof_token_ , iEvent);
  auto handle03 = connect(trackp        , trackp_token_        , iEvent);
  auto handle04 = connect(trackpt       , trackpt_token_       , iEvent);
  auto handle05 = connect(tracketa      , tracketa_token_      , iEvent);
  auto handle06 = connect(trackphi      , trackphi_token_      , iEvent);
  auto handle07 = connect(trackhitsvalid, trackhitsvalid_token_, iEvent);
  auto handle08 = connect(trackindex    , trackindex_token_    , iEvent);
  auto handle09 = connect(rawid         , rawid_token_         , iEvent);
  auto handle11 = connect(localdirx     , localdirx_token_     , iEvent);
  auto handle12 = connect(localdiry     , localdiry_token_     , iEvent);
  auto handle13 = connect(localdirz     , localdirz_token_     , iEvent);
  auto handle14 = connect(firststrip    , firststrip_token_    , iEvent);
  auto handle15 = connect(nstrips       , nstrips_token_       , iEvent);
  auto handle16 = connect(saturation    , saturation_token_    , iEvent);
  auto handle17 = connect(overlapping   , overlapping_token_   , iEvent);
  auto handle18 = connect(farfromedge   , farfromedge_token_   , iEvent);
  auto handle19 = connect(charge        , charge_token_        , iEvent);
  auto handle21 = connect(path          , path_token_          , iEvent);
  auto handle22 = connect(chargeoverpath, chargeoverpath_token_, iEvent);
  auto handle23 = connect(amplitude     , amplitude_token_     , iEvent);
  auto handle24 = connect(gainused      , gainused_token_      , iEvent);
  auto handle25 = connect(gainusedTick  , gainusedTick_token_  , iEvent);
  auto handle26 = connect(trackalgo     , trackalgo_token_     , iEvent);
 
  edm::ESHandle<TrackerTopology> TopoHandle;
  iSetup.get<TrackerTopologyRcd>().get( TopoHandle );
  const TrackerTopology* topo = TopoHandle.product();

  processEvent(topo);

}

//********************************************************************************//
void SiStripGainsPCLWorker::processEvent(const TrackerTopology* topo) {

  edm::LogInfo("SiStripGainsPCLWorker") << "Processing run " << runnumber 
					<< " and event " << eventnumber 
					<< std::endl;

  if(runnumber<SRun)SRun=runnumber;
  if(runnumber>ERun)ERun=runnumber;
  
  NEvent++;
  NTrack+=(*trackp).size();

  edm::LogInfo("SiStripGainsPCLWorker")
    <<"for mode"<< m_calibrationMode 
    <<" Nevent:"<<NEvent 
    <<" NTrack:"<<NTrack
    <<" NClusterStrip:"<<NClusterStrip
    <<" NClusterPixel:"<<NClusterPixel
    <<" NStripAPVs:"<<NStripAPVs
    <<" NPixelDets:"<<NPixelDets
    <<std::endl;

  int elepos = statCollectionFromMode(m_calibrationMode.c_str());
  
  unsigned int FirstAmplitude=0;
  for(unsigned int i=0;i<(*chargeoverpath).size();i++){
    
    FirstAmplitude+=(*nstrips)[i];
    int    TI = (*trackindex)[i];
    
    if((*tracketa      )[TI]  < MinTrackEta          )continue;
    if((*tracketa      )[TI]  > MaxTrackEta          )continue;
    if((*trackp        )[TI]  < MinTrackMomentum     )continue;
    if((*trackp        )[TI]  > MaxTrackMomentum     )continue;
    if((*trackhitsvalid)[TI]  < MinTrackHits         )continue;
    if((*trackchi2ndof )[TI]  > MaxTrackChiOverNdf   )continue;
    if((*trackalgo     )[TI]  > MaxTrackingIteration )continue;
    
    std::shared_ptr<stAPVGain> APV = APVsColl[((*rawid)[i]<<4) | ((*firststrip)[i]/128)];   //works for both strip and pixel thanks to firstStrip encoding for pixel in the calibTree
    
    if(APV->SubDet>2 && (*farfromedge)[i]        == false           )continue;
    if(APV->SubDet>2 && (*overlapping)[i]        == true            )continue;
    if(APV->SubDet>2 && (*saturation )[i]        && !AllowSaturation)continue;
    if(APV->SubDet>2 && (*nstrips    )[i]      > MaxNrStrips        )continue;
    
    if(APV->SubDet>2){
      NClusterStrip++;
    } else {
      NClusterPixel++;
    }
    
    int Charge = 0;
    if(APV->SubDet>2 && (useCalibration || !FirstSetOfConstants)){
      bool Saturation = false;
      for(unsigned int s=0;s<(*nstrips)[i];s++){
	int StripCharge =  (*amplitude)[FirstAmplitude-(*nstrips)[i]+s];
	if(useCalibration && !FirstSetOfConstants){ StripCharge=(int)(StripCharge*(APV->PreviousGain/APV->CalibGain));
	}else if(useCalibration){                   StripCharge=(int)(StripCharge/APV->CalibGain);
	}else if(!FirstSetOfConstants){             StripCharge=(int)(StripCharge*APV->PreviousGain);}
	if(StripCharge>1024){
	  StripCharge = 255;
	  Saturation = true;
	}else if(StripCharge>254){
	  StripCharge = 254;
	  Saturation = true;
	}
	Charge += StripCharge;
      }
      if(Saturation && !AllowSaturation)continue;
    }else if(APV->SubDet>2){
      Charge = (*charge)[i];
    }else{
      Charge = (*charge)[i]/265.0; //expected scale factor between pixel and strip charge               
    }
    
    double ClusterChargeOverPath   =  ( (double) Charge )/(*path)[i] ;
    if(APV->SubDet>2){
      if(Validation)     {ClusterChargeOverPath/=(*gainused)[i];}
      if(OldGainRemoving){ClusterChargeOverPath*=(*gainused)[i];}
    }
      
    // real histogram for calibration
    (Charge_Vs_Index[elepos])->Fill(APV->Index,ClusterChargeOverPath);
    
    LogDebug("SiStripGainsPCLWorker") <<" for mode "<< m_calibrationMode << "\n"
				      <<" i "<< i
				      <<" NClusterStrip "<< NClusterStrip
				      <<" useCalibration "<< useCalibration
				      <<" FirstSetOfConstants "<< FirstSetOfConstants
				      <<" APV->DetId "<< APV->DetId
				      <<" APV->Index "<< APV->Index 
				      <<" Charge "<< Charge
				      <<" ClusterChargeOverPath "<< ClusterChargeOverPath
				      <<std::endl;
    
    // Fill monitoring histograms
    int mCharge1 = 0;
    int mCharge2 = 0;
    int mCharge3 = 0;
    int mCharge4 = 0;
    if(APV->SubDet>2) {
      for(unsigned int s=0;s<(*nstrips)[i];s++){
        int StripCharge =  (*amplitude)[FirstAmplitude-(*nstrips)[i]+s];
        if(StripCharge>1024)      StripCharge = 255;
        else if(StripCharge>254)  StripCharge = 254;
        mCharge1 += StripCharge;
        mCharge2 += StripCharge;
        mCharge3 += StripCharge;
        mCharge4 += StripCharge;
      }
      // Revome gains for monitoring
      mCharge2 *= (*gainused)[i];                         // remove G2
      mCharge3 *= (*gainusedTick)[i];                     // remove G1
      mCharge4 *= ( (*gainused)[i] * (*gainusedTick)[i]); // remove G1 and G2
    } 
    std::vector<APVGain::APVmon>& v1 = Charge_1[elepos];
    std::vector<MonitorElement*> cmon1 = APVGain::FetchMonitor(v1, (*rawid)[i], topo);
    for(unsigned int m=0; m<cmon1.size(); m++) cmon1[m]->Fill(( (double) mCharge1 )/(*path)[i]);

    std::vector<APVGain::APVmon>& v2 = Charge_2[elepos];
    std::vector<MonitorElement*> cmon2 = APVGain::FetchMonitor(v2, (*rawid)[i], topo);
    for(unsigned int m=0; m<cmon2.size(); m++) cmon2[m]->Fill(( (double) mCharge2 )/(*path)[i]);

    std::vector<APVGain::APVmon>& v3 = Charge_3[elepos];
    std::vector<MonitorElement*> cmon3 = APVGain::FetchMonitor(v3, (*rawid)[i], topo);
    for(unsigned int m=0; m<cmon3.size(); m++) cmon3[m]->Fill(( (double) mCharge3 )/(*path)[i]);

    std::vector<APVGain::APVmon>& v4 = Charge_4[elepos];
    std::vector<MonitorElement*> cmon4 = APVGain::FetchMonitor(v4, (*rawid)[i], topo);
    for(unsigned int m=0; m<cmon4.size(); m++) cmon4[m]->Fill(( (double) mCharge4 )/(*path)[i]);


    if(APV->SubDet==StripSubdetector::TIB){
      (Charge_Vs_PathlengthTIB[elepos])->Fill((*path)[i],Charge);  // TIB

    }else if(APV->SubDet==StripSubdetector::TOB){
      (Charge_Vs_PathlengthTOB[elepos])->Fill((*path)[i],Charge);  // TOB

    }else if(APV->SubDet==StripSubdetector::TID){
      if(APV->Eta<0)     { (Charge_Vs_PathlengthTIDM[elepos])->Fill((*path)[i],Charge); }  // TID minus
      else if(APV->Eta>0){ (Charge_Vs_PathlengthTIDP[elepos])->Fill((*path)[i],Charge); }  // TID plus

    }else if(APV->SubDet==StripSubdetector::TEC){
      if(APV->Eta<0){
        if(APV->Thickness<0.04)     { (Charge_Vs_PathlengthTECM1[elepos])->Fill((*path)[i],Charge); } // TEC minus, type 1
        else if(APV->Thickness>0.04){ (Charge_Vs_PathlengthTECM2[elepos])->Fill((*path)[i],Charge); } // TEC minus, type 2
      } else if(APV->Eta>0){
        if(APV->Thickness<0.04)     { (Charge_Vs_PathlengthTECP1[elepos])->Fill((*path)[i],Charge); } // TEC plus, type 1
        else if(APV->Thickness>0.04){ (Charge_Vs_PathlengthTECP2[elepos])->Fill((*path)[i],Charge); } // TEC plus, type 2
      }
    }

  }// END OF ON-CLUSTER LOOP

  LogDebug("SiStripGainsPCLWorker")<<" for mode"<< m_calibrationMode 
				   <<" entries in histogram:"<< (Charge_Vs_Index[elepos])->getTH2S()->GetEntries()
				   <<std::endl;

}//END OF processEvent()


//********************************************************************************//
void 
SiStripGainsPCLWorker::beginJob()
{ 
}

//********************************************************************************//
// ------------ method called once each job just before starting event loop  ------------
void
SiStripGainsPCLWorker::checkBookAPVColls(const edm::EventSetup& es){

  es.get<TrackerDigiGeometryRecord>().get( tkGeom_ );
  const TrackerGeometry *newBareTkGeomPtr = &(*tkGeom_);
  if (newBareTkGeomPtr == bareTkGeomPtr_) return; // already filled APVColls, nothing changed

  if (!bareTkGeomPtr_) { // pointer not yet set: called the first time => fill the APVColls
    auto const & Det = newBareTkGeomPtr->dets();
    
    edm::LogInfo("SiStripGainsPCLWorker")
      <<" Resetting APV struct"<<std::endl;

    unsigned int Index=0;

    for(unsigned int i=0;i<Det.size();i++){
      
      DetId  Detid  = Det[i]->geographicalId(); 
      int    SubDet = Detid.subdetId();
      
      if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
	  SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){
	
	auto DetUnit     = dynamic_cast<const StripGeomDetUnit*> (Det[i]);
	if(!DetUnit)continue;
	
	const StripTopology& Topo     = DetUnit->specificTopology();	
	unsigned int         NAPV     = Topo.nstrips()/128;
	
	for(unsigned int j=0;j<NAPV;j++){
	  auto APV = std::make_shared<stAPVGain>();
	  APV->Index         = Index;
	  APV->Bin           = -1;
	  APV->DetId         = Detid.rawId();
	  APV->APVId         = j;
	  APV->SubDet        = SubDet;
	  APV->FitMPV        = -1;
	  APV->FitMPVErr     = -1;
	  APV->FitWidth      = -1;
	  APV->FitWidthErr   = -1;
	  APV->FitChi2       = -1;
	  APV->FitNorm       = -1;
	  APV->Gain          = -1;
	  APV->PreviousGain  = 1;
	  APV->PreviousGainTick  = 1;
	  APV->x             = DetUnit->position().basicVector().x();
	  APV->y             = DetUnit->position().basicVector().y();
	  APV->z             = DetUnit->position().basicVector().z();
	  APV->Eta           = DetUnit->position().basicVector().eta();
	  APV->Phi           = DetUnit->position().basicVector().phi();
	  APV->R             = DetUnit->position().basicVector().transverse();
	  APV->Thickness     = DetUnit->surface().bounds().thickness();
	  APV->NEntries      = 0;
	  APV->isMasked      = false;
	  
	  APVsCollOrdered.push_back(APV);
	  APVsColl[(APV->DetId<<4) | APV->APVId] = APV;
	  Index++;
	  NStripAPVs++;
	} // loop on APVs
      } // if is Strips
    } // loop on dets
    
    for(unsigned int i=0;i<Det.size();i++){  //Make two loop such that the Pixel information is added at the end --> make transition simpler
      DetId  Detid  = Det[i]->geographicalId();
      int    SubDet = Detid.subdetId();
      if( SubDet == PixelSubdetector::PixelBarrel || SubDet == PixelSubdetector::PixelEndcap ){
	auto DetUnit     = dynamic_cast<const PixelGeomDetUnit*> (Det[i]);
	if(!DetUnit) continue;
	
	const PixelTopology& Topo     = DetUnit->specificTopology();
	unsigned int         NROCRow  = Topo.nrows()/(80.);
	unsigned int         NROCCol  = Topo.ncolumns()/(52.);
	
	for(unsigned int j=0;j<NROCRow;j++){
	  for(unsigned int i=0;i<NROCCol;i++){
	    
	    auto APV = std::make_shared<stAPVGain>();
	    APV->Index         = Index;
	    APV->Bin           = -1;
	    APV->DetId         = Detid.rawId();
	    APV->APVId         = (j<<3 | i);
	    APV->SubDet        = SubDet;
	    APV->FitMPV        = -1;
	    APV->FitMPVErr     = -1;
	    APV->FitWidth      = -1;
	    APV->FitWidthErr   = -1;
	    APV->FitChi2       = -1;
	    APV->Gain          = -1;
	    APV->PreviousGain  = 1;
	    APV->PreviousGainTick = 1;
	    APV->x             = DetUnit->position().basicVector().x();
	    APV->y             = DetUnit->position().basicVector().y();
	    APV->z             = DetUnit->position().basicVector().z();
	    APV->Eta           = DetUnit->position().basicVector().eta();
	    APV->Phi           = DetUnit->position().basicVector().phi();
	    APV->R             = DetUnit->position().basicVector().transverse();
	    APV->Thickness     = DetUnit->surface().bounds().thickness();
	    APV->isMasked      = false; //SiPixelQuality_->IsModuleBad(Detid.rawId());
	    APV->NEntries      = 0;
	    
	    APVsCollOrdered.push_back(APV);
	    APVsColl[(APV->DetId<<4) | APV->APVId] = APV;
	    Index++;
	    NPixelDets++;

	  } // loop on ROC cols
	} // loop on ROC rows
      } // if Pixel
    } // loop on Dets  
  }  //if (!bareTkGeomPtr_) ...
  bareTkGeomPtr_ = newBareTkGeomPtr;
}


//********************************************************************************//
void 
SiStripGainsPCLWorker::endJob() 
{
}

//********************************************************************************//
void
SiStripGainsPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//********************************************************************************//
void 
SiStripGainsPCLWorker::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & run, edm::EventSetup const & es){

  ibooker.cd();
  std::string dqm_dir = m_DQMdir;
  const char* tag = dqm_tag_[statCollectionFromMode(m_calibrationMode.c_str())].c_str();

  edm::LogInfo("SiStripGainsPCLWorker") << "Setting " << dqm_dir << "in DQM and booking histograms for tag "
					<< tag << std::endl;
  
  ibooker.setCurrentFolder(dqm_dir);
 
  std::string stag(tag);
  if(stag.size()!=0 && stag[0]!='_') stag.insert(0,1,'_');
  
  std::string cvi      = std::string("Charge_Vs_Index") + stag;
  std::string cvpTIB   = std::string("Charge_Vs_PathlengthTIB")   + stag;
  std::string cvpTOB   = std::string("Charge_Vs_PathlengthTOB")   + stag;
  std::string cvpTIDP  = std::string("Charge_Vs_PathlengthTIDP")  + stag;
  std::string cvpTIDM  = std::string("Charge_Vs_PathlengthTIDM")  + stag;
  std::string cvpTECP1 = std::string("Charge_Vs_PathlengthTECP1") + stag;
  std::string cvpTECP2 = std::string("Charge_Vs_PathlengthTECP2") + stag;
  std::string cvpTECM1 = std::string("Charge_Vs_PathlengthTECM1") + stag;
  std::string cvpTECM2 = std::string("Charge_Vs_PathlengthTECM2") + stag;
  
  int elepos = statCollectionFromMode(tag);
  
  Charge_Vs_Index[elepos]           = ibooker.book2S(cvi.c_str()     , cvi.c_str()     , 88625, 0   , 88624,2000,0,4000);
  Charge_Vs_PathlengthTIB[elepos]   = ibooker.book2S(cvpTIB.c_str()  , cvpTIB.c_str()  , 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTOB[elepos]   = ibooker.book2S(cvpTOB.c_str()  , cvpTOB.c_str()  , 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTIDP[elepos]  = ibooker.book2S(cvpTIDP.c_str() , cvpTIDP.c_str() , 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTIDM[elepos]  = ibooker.book2S(cvpTIDM.c_str() , cvpTIDM.c_str() , 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTECP1[elepos] = ibooker.book2S(cvpTECP1.c_str(), cvpTECP1.c_str(), 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTECP2[elepos] = ibooker.book2S(cvpTECP2.c_str(), cvpTECP2.c_str(), 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTECM1[elepos] = ibooker.book2S(cvpTECM1.c_str(), cvpTECM1.c_str(), 20   , 0.3 , 1.3  , 250,0,2000);
  Charge_Vs_PathlengthTECM2[elepos] = ibooker.book2S(cvpTECM2.c_str(), cvpTECM2.c_str(), 20   , 0.3 , 1.3  , 250,0,2000);

  std::vector<std::pair<std::string,std::string>> hnames = APVGain::monHnames(VChargeHisto,doChargeMonitorPerPlane,"");
  for (unsigned int i=0;i<hnames.size();i++){
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = ibooker.book1DD( htag.c_str(), (hnames[i]).second.c_str(), 100   , 0. , 1000. );
    int id    = APVGain::subdetectorId((hnames[i]).first);
    int side  = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_1[elepos].push_back( APVGain::APVmon(id,side,plane,monitor) );
  }

  hnames = APVGain::monHnames(VChargeHisto,doChargeMonitorPerPlane,"woG2");
  for (unsigned int i=0;i<hnames.size();i++){
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = ibooker.book1DD( htag.c_str(), (hnames[i]).second.c_str(), 100   , 0. , 1000. );
    int id    = APVGain::subdetectorId((hnames[i]).first);
    int side  = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_2[elepos].push_back( APVGain::APVmon(id,side,plane,monitor) );
  }

  hnames = APVGain::monHnames(VChargeHisto,doChargeMonitorPerPlane,"woG1");
  for (unsigned int i=0;i<hnames.size();i++){
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = ibooker.book1DD( htag.c_str(), (hnames[i]).second.c_str(), 100   , 0. , 1000. );
    int id    = APVGain::subdetectorId((hnames[i]).first);
    int side  = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_3[elepos].push_back( APVGain::APVmon(id,side,plane,monitor) );
  }

  hnames = APVGain::monHnames(VChargeHisto,doChargeMonitorPerPlane,"woG1G2");
  for (unsigned int i=0;i<hnames.size();i++){
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = ibooker.book1DD( htag.c_str(), (hnames[i]).second.c_str(), 100   , 0. , 1000. );
    int id    = APVGain::subdetectorId((hnames[i]).first);
    int side  = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_4[elepos].push_back( APVGain::APVmon(id,side,plane,monitor) );
  }

}
