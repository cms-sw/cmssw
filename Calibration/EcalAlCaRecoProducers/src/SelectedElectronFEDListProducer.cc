#include "Calibration/EcalAlCaRecoProducers/interface/SelectedElectronFEDListProducer.h"

template< typename TEle, typename TCand>
SelectedElectronFEDListProducer<TEle,TCand>::SelectedElectronFEDListProducer(const edm::ParameterSet & iConfig){
 
 // input electron collection
 if(iConfig.existsAs<std::vector<edm::InputTag> >("electronCollections")){
   electronCollections_ = iConfig.getParameter<std::vector<edm::InputTag>>("electronCollections");
   if(electronCollections_.empty())
      throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] empty electron collection is given --> at least one \n"; 
 }
 else throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] no electron collection are given --> need at least one \n";  

 // input RecoEcalCandidate collection
 if(iConfig.existsAs<std::vector<edm::InputTag> >("recoEcalCandidateCollections")){
   recoEcalCandidateCollections_ = iConfig.getParameter<std::vector<edm::InputTag>>("recoEcalCandidateCollections");
   if(recoEcalCandidateCollections_.empty())
      throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] empty ecal candidate collections collection is given --> at least one \n"; 
 }
 else throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] no electron reco ecal candidate collection are given --> need at least one \n";  

 // list of gsf collections
 if(iConfig.existsAs<std::vector<int>>("isGsfElectronCollection")){
    isGsfElectronCollection_ = iConfig.getParameter<std::vector<int>>("isGsfElectronCollection");
    if(isGsfElectronCollection_.empty())
      throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] empty electron flag collection --> at least one \n"; 
 }
 else throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] no electron flag are given --> need at least one \n"; 

 if(isGsfElectronCollection_.size() < electronCollections_.size()) 
    throw cms::Exception("Configuration")<<"[SelectedElectronFEDListProducer] electron flag < electron collection  --> need at equal number to understand which are Gsf and which not \n";

 // add a set of selected feds
 if(iConfig.existsAs<std::vector<int>>("addThisSelectedFEDs")){
    addThisSelectedFEDs_ = iConfig.getParameter<std::vector<int>>("addThisSelectedFEDs");
    if(addThisSelectedFEDs_.empty())
      addThisSelectedFEDs_.push_back(-1);    
 }
 else addThisSelectedFEDs_.push_back(-1); 

 // take the beam spot Tag 
 if(iConfig.existsAs<edm::InputTag>("beamSpot"))
   beamSpotTag_ = iConfig.getParameter<edm::InputTag>("beamSpot"); 
 else beamSpotTag_ = edm::InputTag("hltOnlineBeamSpot"); 

 // take the HBHE recHit Tag
 if(iConfig.existsAs<edm::InputTag>("HBHERecHitCollection"))
   HBHERecHitCollection_ = iConfig.getParameter<edm::InputTag>("HBHERecHitCollection"); 
 else HBHERecHitCollection_ = edm::InputTag("hltHbhereco"); 
   
 // ES look up table path
 if(iConfig.existsAs<std::string>("ESLookupTable"))
   ESLookupTable_    = iConfig.getParameter<edm::FileInPath>("ESLookupTable"); 
 else ESLookupTable_ = edm::FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"); 

 // Hcal look up table path
 if(iConfig.existsAs<std::string>("HCALLookupTable"))
   HCALLookupTable_    = iConfig.getParameter<edm::FileInPath>("HCALLookupTable"); 
 else HCALLookupTable_ = edm::FileInPath("Calibration/EcalAlCaRecoProducers/data/HcalElectronicsMap_v7.00_offline"); 

 // raw data collector label
 if(iConfig.existsAs<edm::InputTag>("rawDataLabel"))
   rawDataLabel_ = iConfig.getParameter<edm::InputTag>("rawDataLabel"); 
 else rawDataLabel_ = edm::InputTag("rawDataCollector") ;

 // output model label
 if(iConfig.existsAs<std::string>("outputLabelModule"))
   outputLabelModule_ = iConfig.getParameter<std::string>("outputLabelModule"); 
 else outputLabelModule_ = "streamElectronRawData" ;

 // dR for the strip region
 if(iConfig.existsAs<double>("dRStripRegion"))
   dRStripRegion_ = iConfig.getParameter<double>("dRStripRegion"); 
 else dRStripRegion_ = 0.5 ;

 // dR for the hcal region 
 if(iConfig.existsAs<double>("dRHcalRegion"))
   dRHcalRegion_ = iConfig.getParameter<double>("dRHcalRegion"); 
 else dRHcalRegion_ = 0.5 ;

 // dPhi, dEta and maxZ for pixel dump
 if(iConfig.existsAs<double>("dPhiPixelRegion"))
   dPhiPixelRegion_ = iConfig.getParameter<double>("dPhiPixelRegion"); 
 else dPhiPixelRegion_ = 0.5 ;

 if(iConfig.existsAs<double>("dEtaPixelRegion"))
   dEtaPixelRegion_ = iConfig.getParameter<double>("dEtaPixelRegion"); 
 else dEtaPixelRegion_ = 0.5 ;

 if(iConfig.existsAs<double>("maxZPixelRegion"))
   maxZPixelRegion_ = iConfig.getParameter<double>("maxZPixelRegion"); 
 else maxZPixelRegion_ = 24. ;

 // bool
 if( iConfig.existsAs<bool>("dumpSelectedEcalFed"))
   dumpSelectedEcalFed_ = iConfig.getParameter< bool >("dumpSelectedEcalFed"); 
 else dumpSelectedEcalFed_ = true ;

 if(iConfig.existsAs<bool>("dumpSelectedSiStripFed"))
   dumpSelectedSiStripFed_ = iConfig.getParameter<bool>("dumpSelectedSiStripFed"); 
 else dumpSelectedSiStripFed_ = true ;

 if(iConfig.existsAs<bool>("dumpSelectedSiPixelFed"))
    dumpSelectedSiPixelFed_ = iConfig.getParameter<bool>("dumpSelectedSiPixelFed"); 
 else dumpSelectedSiPixelFed_ = true ;

 if(iConfig.existsAs<bool>("dumpSelectedHCALFed"))
    dumpSelectedHCALFed_ = iConfig.getParameter<bool>("dumpSelectedHCALFed"); 
 else dumpSelectedHCALFed_ = true ;

 if(iConfig.existsAs<bool>("dumpAllEcalFed"))
     dumpAllEcalFed_ = iConfig.getParameter<bool>("dumpAllEcalFed"); 
 else dumpAllEcalFed_ = false ;

 if(iConfig.existsAs<bool>("dumpAllTrackerFed"))
     dumpAllTrackerFed_ = iConfig.getParameter<bool>("dumpAllTrackerFed"); 
 else dumpAllTrackerFed_ = false ;

 if(iConfig.existsAs<bool>("dumpAllHCALFed"))
     dumpAllHCALFed_ = iConfig.getParameter<bool>("dumpAllHCALFed"); 
 else dumpAllHCALFed_ = false ;

 if(iConfig.existsAs<bool>("debug"))
   debug_ = iConfig.getParameter<bool>("debug");
 else debug_ = false ;

 // only in debugging mode
 if(debug_){

  LogDebug("SelectedElectronFEDListProducer")<<"############################################################## ";
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] output Label "<<outputLabelModule_; 
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] beam spot Tag "<<beamSpotTag_;
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] dumpEcalFedList set to "<<dumpSelectedEcalFed_<<" dumpSelectedSiStripFed "<<dumpSelectedSiStripFed_<<" dumpSelectedSiPixelFed "<<dumpSelectedSiPixelFed_;
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] dumpAllEcalFed "<<dumpAllEcalFed_<<" dumpAllTrackerFed "<<dumpAllTrackerFed_<<" dump all HCAL fed "<<dumpAllHCALFed_;
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] dRStripRegion "<<dRStripRegion_;
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] dPhiPixelRegion "<<dPhiPixelRegion_<<" dEtaPixelRegion "<<dEtaPixelRegion_<<" maxZPixelRegion "<<maxZPixelRegion_;
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Electron Collections";

  std::vector<edm::InputTag>::const_iterator Tag = electronCollections_.begin();
  std::vector<int>::const_iterator Flag = isGsfElectronCollection_.begin();
  for( ; Tag !=electronCollections_.end() && Flag!=isGsfElectronCollection_.end() ; ++Tag , ++Flag)
     LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] ele collection: "<<*(Tag)<<" isGsf "<<*(Flag);

  std::vector<edm::InputTag>::const_iterator Tag2 = recoEcalCandidateCollections_.begin();
  for( ; Tag2 !=recoEcalCandidateCollections_.end() ; ++Tag2)
     LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] reco ecal candidate collection: "<<*(Tag2);

  std::vector<int>::const_iterator AddFed = addThisSelectedFEDs_.begin();
  for( ; AddFed !=addThisSelectedFEDs_.end() ; ++AddFed)
     LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] additional FED: "<<*(AddFed);
     
  LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] rawDataInput "<<rawDataLabel_; 

 }
 
 // initialize pre-shower fed id --> look up table
 for (int i=0; i<2; ++i)
  for (int j=0; j<2; ++j)
   for (int k=0 ;k<40; ++k)
    for (int m=0; m<40; m++)
          ES_fedId_[i][j][k][m] = -1;
 
 // read in look-up table
 int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
 std::ifstream ES_file;
 ES_file.open(ESLookupTable_.fullPath().c_str());
 if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Look Up table for ES "<<ESLookupTable_.fullPath().c_str();
 if( ES_file.is_open() ) {
     ES_file >> nLines;
     for (int i=0; i<nLines; ++i) {
       ES_file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx ;
       ES_fedId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = fed;
     }
 } 
 else LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Look up table file can not be found in "<<ESLookupTable_.fullPath().c_str() ;
 ES_file.close();
 
 // make the hcal map in a similar way
 int idet, cr, sl, dcc, spigot, fibcha, ieta, iphi, depth, subdet; 
 std::string subdet_tmp, buffer, tb;
 std::ifstream HCAL_file;
 
 HCAL_file.open(HCALLookupTable_.fullPath().c_str(),std::ios::in);
 if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Look Up table for HCAL "<<HCALLookupTable_.fullPath().c_str();
 if( HCAL_file.is_open() ) {
   while(!HCAL_file.eof()) {
       getline(HCAL_file,buffer);     
       if (buffer == "" || !buffer.find('#')) continue;
       std::stringstream line( buffer );
       line >> idet >> cr >> sl >> tb >> dcc >> spigot >> fiber >> fibcha >> subdet_tmp >> ieta >> iphi >> depth ;
       if (subdet_tmp == "HB")  subdet  = 1;
       else if (subdet_tmp == "HE") subdet  = 2;
       else if (subdet_tmp == "HO") subdet  = 3;
       else if (subdet_tmp == "HF") subdet  = 4;
       else subdet = 0 ;

       HCALFedId fedId (subdet,iphi,ieta,depth);
       fedId.setDCCId(dcc);
       HCAL_fedId_.push_back(fedId);
     }
 } 
 else LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Look up table file can not be found in"<<HCALLookupTable_.fullPath().c_str() ;
 HCAL_file.close();
 std::sort(HCAL_fedId_.begin(),HCAL_fedId_.end());

 if(debug_)  LogDebug("SelectedElectronFEDListProducer")<<"############################################################## ";

 produces<FEDRawDataCollection>(outputLabelModule_); // produce exit collection

}

template< typename TEle, typename TCand>
SelectedElectronFEDListProducer<TEle,TCand>::~SelectedElectronFEDListProducer(){

 if(!electronCollections_.empty()) electronCollections_.clear() ;
 if(!recoEcalCandidateCollections_.empty()) recoEcalCandidateCollections_.clear() ;
 if(!fedList_.empty()) fedList_.clear() ;
 if(!RawDataCollection_) delete RawDataCollection_ ;
}

template< typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle,TCand>::beginJob(){ 
 
  if(debug_){ LogDebug("SelectedElectronFEDListProducer")<<"############################################################## ";
              LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Begin of the Job ----> ";
              LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] event counter set to "<<eventCounter_;
	      LogDebug("SelectedElectronFEDListProducer")<<"############################################################## ";
  }
  eventCounter_ = 0 ; 
} 

template< typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle,TCand>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup){

  if(!fedList_.empty()) fedList_.clear(); 
  if(!RawDataCollection_) delete RawDataCollection_ ;

  // Build FED strip map --> just one time
  // Retrieve FED ids from cabling map and iterate through 
  if(eventCounter_ ==0 ){

   // get the ecal electronics map
   edm::ESHandle<EcalElectronicsMapping > ecalmapping;
   iSetup.get<EcalMappingRcd >().get(ecalmapping);
   TheMapping_ = ecalmapping.product();
 
   // get the calo geometry
   edm::ESHandle<CaloGeometry> caloGeometry; 
   iSetup.get<CaloGeometryRecord>().get(caloGeometry);
   geometry_ = caloGeometry.product();

   //ES geometry
   geometryES_ = caloGeometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

   // pixel tracker cabling map
   edm::ESTransientHandle<SiPixelFedCablingMap> pixelCablingMap;
   iSetup.get<SiPixelFedCablingMapRcd>().get(pixelCablingMap);

   PixelCabling_.reset();
   PixelCabling_ = pixelCablingMap->cablingTree();
   
   edm::ESHandle<TrackerGeometry> trackerGeometry;
   iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );

   if(!pixelModuleVector_.empty()) pixelModuleVector_.clear();

   // build the tracker pixel module map   
   std::vector<const GeomDet*>::const_iterator itTracker = trackerGeometry->dets().begin();   
   for( ; itTracker !=trackerGeometry->dets().end() ; ++itTracker){
       int subdet = (*itTracker)->geographicalId().subdetId();
       if(! (subdet == PixelSubdetector::PixelBarrel || subdet == PixelSubdetector::PixelEndcap) ) continue;
       PixelModule module ;
       module.x = (*itTracker)->position().x();
       module.y = (*itTracker)->position().y();
       module.z = (*itTracker)->position().z();
       module.Phi = normalizedPhi((*itTracker)->position().phi()) ; 
       module.Eta = (*itTracker)->position().eta() ;
       module.DetId  = (*itTracker)->geographicalId().rawId();
       const std::vector<sipixelobjects::CablingPathToDetUnit> path2det = PixelCabling_->pathToDetUnit(module.DetId);
       module.Fed = path2det[0].fed;
       assert(module.Fed<40);
       pixelModuleVector_.push_back(module);
   }
   std::sort(pixelModuleVector_.begin(),pixelModuleVector_.end());

   edm::ESHandle<SiStripRegionCabling> SiStripCabling ;
   iSetup.get<SiStripRegionCablingRcd>().get(SiStripCabling);
   StripRegionCabling_ = SiStripCabling.product();

   cabling_ = StripRegionCabling_->getRegionCabling();
   regionDimension_ = StripRegionCabling_->regionDimensions();
  }
  
  // event by event analysis

  // Get event raw data
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByLabel(rawDataLabel_,rawdata);

  // take the beam spot position
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel(beamSpotTag_, beamSpot); 
  if(!beamSpot.failedToGet()) beamSpotPosition_ = beamSpot->position();
  else beamSpotPosition_.SetXYZ(0,0,0);
 
  // take the calo tower collection
  edm::Handle<reco::PFRecHitCollection> hbheRecHitHandle;
  iEvent.getByLabel(HBHERecHitCollection_,hbheRecHitHandle);
  const reco::PFRecHitCollection* hcalRecHitCollection = NULL ;
  if(!hbheRecHitHandle.failedToGet()) hcalRecHitCollection = hbheRecHitHandle.product();   
  
  // loop on the input electron collection vector
  edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerRecoEcalCandidateCollection;
  edm::Handle<TEleColl> electrons;
  std::vector<edm::Ref<TCandColl>> recoEcalCandColl;
  TEle  electron ;
  edm::Ref<TCandColl> recoEcalCand ;
  
  // iterator to electron and ecal candidate collections
  std::vector<edm::InputTag>::const_iterator itElectronColl     = electronCollections_.begin();
  std::vector<int>::const_iterator itElectronCollFlag           = isGsfElectronCollection_.begin();
  std::vector<edm::InputTag>::const_iterator itRecoEcalCandColl = recoEcalCandidateCollections_.begin();

  double radTodeg = 180. / Geom::pi();

  if(dumpAllEcalFed_){
     for(uint32_t iEcalFed = FEDNumbering::MINECALFEDID ; iEcalFed <= FEDNumbering::MAXECALFEDID ; iEcalFed++)
      fedList_.push_back(iEcalFed);
     for(uint32_t iESFed = FEDNumbering::MINPreShowerFEDID ; iESFed <= FEDNumbering::MAXPreShowerFEDID ; iESFed++)
      fedList_.push_back(iESFed);
  }

  if(dumpAllTrackerFed_){
    for(uint32_t iPixelFed = FEDNumbering::MINSiPixelFEDID; iPixelFed <= FEDNumbering::MAXSiPixelFEDID ; iPixelFed++)
     fedList_.push_back(iPixelFed);
    for(uint32_t iStripFed = FEDNumbering::MINSiStripFEDID; iStripFed <= FEDNumbering::MAXSiStripFEDID ; iStripFed++)
     fedList_.push_back(iStripFed);
  }

  if(dumpAllHCALFed_){
   for(uint32_t iHcalFed = FEDNumbering::MINHCALFEDID ; iHcalFed <=   FEDNumbering::MAXHCALFEDID; iHcalFed++)
     fedList_.push_back(iHcalFed);
  } 

  // if you want to dump just FED related to the triggering electron/s
  if( !dumpAllTrackerFed_  || !dumpAllEcalFed_ ){
   for( ; itRecoEcalCandColl != recoEcalCandidateCollections_.end(); ++itRecoEcalCandColl){  

    try { iEvent.getByLabel(*itRecoEcalCandColl,triggerRecoEcalCandidateCollection);
          if(triggerRecoEcalCandidateCollection.failedToGet()) continue ;
    }
    catch (cms::Exception &exception){ continue; }
 
    triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerCluster, recoEcalCandColl);
    if(recoEcalCandColl.empty()) triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerPhoton, recoEcalCandColl);
    if(recoEcalCandColl.empty()) triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerElectron, recoEcalCandColl);

    typename std::vector<edm::Ref<TCandColl>>::const_iterator itRecoEcalCand = recoEcalCandColl.begin(); // loop on recoEcalCandidate objects

     for( ; itRecoEcalCand != recoEcalCandColl.end() ; ++itRecoEcalCand){        
       recoEcalCand = (*itRecoEcalCand); 
       reco::SuperClusterRef scRefRecoEcalCand = recoEcalCand->superCluster(); // take the supercluster in order to match with electron objects
                     
       for( ; itElectronColl != electronCollections_.end() && itElectronCollFlag != isGsfElectronCollection_.end(); ++itElectronColl , ++itElectronCollFlag){ // loop on electron collections   
        try { iEvent.getByLabel(*itElectronColl,electrons);
              if(electrons.failedToGet()) continue ;
        }
        catch (cms::Exception &exception){ continue; }

        typename TEleColl::const_iterator itEle = electrons->begin(); 
        for( ; itEle!=electrons->end() ; ++itEle){ // loop on all the electrons inside a collection
        // get electron supercluster and the associated hit -> detID
	electron = (*itEle);
        reco::SuperClusterRef scRef = electron.superCluster();
        if ( scRefRecoEcalCand != scRef ) continue ; // mathching
 
        const std::vector<std::pair<DetId,float> >& hits = scRef->hitsAndFractions();
        // start in dump the ecal FED associated to the electron
        std::vector<std::pair<DetId,float> >::const_iterator itSChits = hits.begin();
        if(!dumpAllEcalFed_){
         for( ; itSChits!=hits.end() ; ++itSChits){
           if((*itSChits).first.subdetId()== EcalBarrel){ // barrel part
            EBDetId idEBRaw ((*itSChits).first);
            GlobalPoint point = geometry_->getPosition(idEBRaw);
            int hitFED = FEDNumbering::MINECALFEDID + TheMapping_->GetFED(double(point.eta()),double(point.phi())*radTodeg);
            if( hitFED < FEDNumbering::MINECALFEDID || hitFED > FEDNumbering::MAXECALFEDID ) continue;

            if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] electron hit detID Barrel "<<(*itSChits).first.rawId()<<" eta "<<double(point.eta())<<" phi "<< double(point.phi())*radTodeg <<" FED "<<hitFED;
          
            if(dumpSelectedEcalFed_){
             if(!fedList_.empty()){ 
	       if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED); // in order not to duplicate info
             }
             else fedList_.push_back(hitFED);
	    }
	   }
           else if((*itSChits).first.subdetId()== EcalEndcap){ // endcap one
             EEDetId idEERaw ((*itSChits).first);
             GlobalPoint point = geometry_->getPosition(idEERaw);
             int hitFED = FEDNumbering::MINECALFEDID + TheMapping_->GetFED(double(point.eta()),double(point.phi())*radTodeg);
             if( hitFED < FEDNumbering::MINECALFEDID || hitFED > FEDNumbering::MAXECALFEDID ) continue;

             if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] electron hit detID Endcap "<<(*itSChits).first.rawId()<<" eta "<<double(point.eta())<<" phi "<<double(point.phi())*radTodeg <<" FED "<<hitFED;
             if(dumpSelectedEcalFed_){
              if(!fedList_.empty()){ 
               if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
              }
              else fedList_.push_back(hitFED);

              // preshower hit for each ecal endcap hit
              DetId tmpX = (dynamic_cast<const EcalPreshowerGeometry*>(geometryES_))->getClosestCellInPlane(point,1);
              ESDetId stripX = (tmpX == DetId(0)) ? ESDetId(0) : ESDetId(tmpX);          
              int hitFED = ES_fedId_[(3-stripX.zside())/2-1][stripX.plane()-1][stripX.six()-1][stripX.siy()-1];
              if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] ES hit plane X (deiID) "<<stripX.rawId()<<" six "<<stripX.six()<<" siy "<<stripX.siy()<<" plane "<<stripX.plane()<<" FED ID "<<hitFED;
              if(hitFED < FEDNumbering::MINPreShowerFEDID || hitFED > FEDNumbering::MAXPreShowerFEDID) continue;
              if(hitFED < 0) continue;
              if(!fedList_.empty()){ 
               if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
              }
              else fedList_.push_back(hitFED);
         
              DetId tmpY = (dynamic_cast<const EcalPreshowerGeometry*>(geometryES_))->getClosestCellInPlane(point,2);
              ESDetId stripY = (tmpY == DetId(0)) ? ESDetId(0) : ESDetId(tmpY);          
              hitFED = ES_fedId_[(3-stripY.zside())/2-1][stripY.plane()-1][stripY.six()-1][stripY.siy()-1];
              if(hitFED < FEDNumbering::MINPreShowerFEDID || hitFED > FEDNumbering::MAXPreShowerFEDID) continue;
              if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] ES hit plane Y (deiID) "<<stripY.rawId()<<" six "<<stripY.six()<<" siy "<<stripY.siy()<<" plane "<<stripY.plane()<<" FED ID "<<hitFED;
              if(hitFED < 0) continue;
              if(!fedList_.empty()){ 
               if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
	      }
              else fedList_.push_back(hitFED);      
	     }
	   } // end endcap  
	 } // end loop on SC hit   

         // check HCAL behind each hit    
         if(dumpSelectedHCALFed_){
  	   reco::PFRecHitCollection::const_iterator itHcalRecHit = hcalRecHitCollection->begin();
	   for( ; itHcalRecHit != hcalRecHitCollection->end() ; ++itHcalRecHit){
 	    HcalDetId id  (itHcalRecHit->detId());
            const CaloCellGeometry* cellGeometry = geometry_->getSubdetectorGeometry(id)->getGeometry(id);
            float dR = reco::deltaR(scRef->eta(),scRef->phi(),cellGeometry->getPosition().eta(),cellGeometry->getPosition().phi());
            if(dR <= dRHcalRegion_){
	     HCALFedId fedId (id.subdet(),id.iphi(),id.ieta(),id.depth());
             if(!HCAL_fedId_.empty()){ 
  	      std::vector<HCALFedId>::iterator itHcalFed = std::find(HCAL_fedId_.begin(),HCAL_fedId_.end(),fedId);
              int ishift = 1 ;
              if((*itHcalFed).fed_ == 0){
	       while((*itHcalFed).fed_ == 0 && ishift <= HBHERecHitShift_){
               int i = -ishift ;
               for( ; i <= ishift && i <= HBHERecHitShift_ ; i++){
                int j = -ishift ;
                for( ; j <= ishift && j <= HBHERecHitShift_ ; j++){                  
		 HCALFedId fedIdshifted (id.subdet(),id.iphi()+i,id.ieta()+j,id.depth());        
                 itHcalFed = std::find(HCAL_fedId_.begin(),HCAL_fedId_.end(),fedIdshifted);
		 if((*itHcalFed).fed_ != 0){ j = HBHERecHitShift_+1; continue; }
		}               
	        if(j > HBHERecHitShift_)    { i = HBHERecHitShift_+1; continue; }
	       }
               ishift++ ;
	       }
	      }
 	      int hitFED = (*itHcalFed).fed_;
              if(hitFED < FEDNumbering::MINHCALFEDID || hitFED > FEDNumbering::MAXHCALFEDID) continue; //first eighteen feds are for HBHE
              if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Hcal FED ID "<<hitFED;
              if(hitFED < 0) continue;
              if(!fedList_.empty()){ 
	      if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
	      }
	      else fedList_.push_back(hitFED);      
	     }
	    }
	   }
	 }
        }// End Ecal
     
        // get the electron track
        if( !dumpAllTrackerFed_ ){ 
         //loop on the region
         if(dumpSelectedSiStripFed_){
          double eta ;
          double phi ;
          if(*itElectronCollFlag){
           eta = electron.gsfTrack()->eta();
           phi = electron.gsfTrack()->phi();
          }
          else{
           eta = electron.track()->eta();
           phi = electron.track()->phi();
          }
 	  for(uint32_t iCabling = 0; iCabling < cabling_.size(); iCabling++){
	    SiStripRegionCabling::Position pos = StripRegionCabling_->position(iCabling);
	    double dphi=fabs(pos.second-phi);
	    if (dphi>acos(-1)) dphi=2*acos(-1)-dphi;
	    double R = sqrt(pow(pos.first-eta,2)+dphi*dphi);
	    if (R-sqrt(pow(regionDimension_.first/2,2)+pow(regionDimension_.second/2,2))>dRStripRegion_) continue;
	    //get vector of subdets within region
	    const SiStripRegionCabling::RegionCabling regSubdets = cabling_[iCabling];
	    //cycle on subdets
	    for (uint32_t idet=0; idet<SiStripRegionCabling::ALLSUBDETS; idet++){ //cicle between 1 and 4 
	      //get vector of layers whin subdet of region
	      const SiStripRegionCabling::WedgeCabling regSubdetLayers = regSubdets[idet]; // at most 10 layers        
	      for (uint32_t ilayer=0; ilayer<SiStripRegionCabling::ALLLAYERS; ilayer++){
		//get map of vectors of feds withing the layer of subdet of region
		const SiStripRegionCabling::ElementCabling fedVectorMap = regSubdetLayers[ilayer]; // vector of the fed
		SiStripRegionCabling::ElementCabling::const_iterator itFedMap = fedVectorMap.begin();
		 for( ; itFedMap!=fedVectorMap.end(); itFedMap++){
		   for (uint32_t op=0; op<(itFedMap->second).size(); op++){
		     int hitFED = (itFedMap->second)[op].fedId(); 
                     if(hitFED < FEDNumbering::MINSiStripFEDID || hitFED > FEDNumbering::MAXSiStripFEDID) continue;
                     if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] SiStrip (FedID) "<<hitFED;
                     if(!fedList_.empty()){ 
                       if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
                     }
                     else fedList_.push_back(hitFED);
                   }
		 }
	      }
	    }
	  }
	 } // end si strip
      
         if(dumpSelectedSiPixelFed_){
           math::XYZVector momentum;
           if(*itElectronCollFlag) momentum = electron.gsfTrack()->momentum();
           else momentum = electron.track()->momentum();
           PixelRegion region (momentum,dPhiPixelRegion_,dEtaPixelRegion_,maxZPixelRegion_);

           PixelModule lowerBound (normalizedPhi(region.vector.phi())-region.dPhi, region.vector.eta()-region.dEta);
           PixelModule upperBound (normalizedPhi(region.vector.phi())+region.dPhi, region.vector.eta()+region.dEta);

           std::vector<PixelModule>::const_iterator itUp, itDn ;
           if(lowerBound.Phi >= -M_PI  && upperBound.Phi <= M_PI ){
            itDn = std::lower_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),lowerBound);
            itUp = std::upper_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),upperBound);
            pixelFedDump(itDn,itUp,region);
           }
           else{
                if(lowerBound.Phi < -M_PI) lowerBound.Phi = lowerBound.Phi+2*M_PI;
                PixelModule phi_p(M_PI,region.vector.eta()-region.dEta);
                itDn = std::lower_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),lowerBound);
                itUp = std::upper_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),phi_p);
                pixelFedDump(itDn,itUp,region);
                 
                if(upperBound.Phi < -M_PI) upperBound.Phi = upperBound.Phi-2*M_PI;
                PixelModule phi_m(-M_PI,region.vector.eta()-region.dEta);
                itDn = std::lower_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),phi_m);
                itUp = std::upper_bound(pixelModuleVector_.begin(),pixelModuleVector_.end(),upperBound);
                pixelFedDump(itDn,itUp,region);
	   }
	 }
 	}// end tracker analysis
	}// end loop on the electron candidate
       } // end loop on the electron collection collection
     } // end loop on the recoEcal candidate
   } // end loop on the recoEcal candidate collection
  }  
  // add a set of chosen FED  
  for( unsigned int iFed = 0 ; iFed < addThisSelectedFEDs_.size() ; iFed++){
    if(addThisSelectedFEDs_.at(iFed) == -1 ) continue ;
    fedList_.push_back(addThisSelectedFEDs_.at(iFed));
  }

  if(debug_){
   if(!fedList_.empty()){ 
    LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] fed point ";
    for( unsigned int i =0; i< fedList_.size(); i++) 
      LogDebug("SelectedElectronFEDListProducer")<<fedList_.at(i)<<"  ";
   }
   LogDebug("SelectedElectronFEDListProducer")<<"  ";
  }

  // make the final raw data collection
  RawDataCollection_ = new FEDRawDataCollection();
  std::sort(fedList_.begin(),fedList_.end());
  std::vector<uint32_t>::const_iterator itfedList = fedList_.begin();
  for( ; itfedList!=fedList_.end() ; ++itfedList){
   const FEDRawData& data = rawdata->FEDData(*itfedList);   
   if(data.size()>0){
           FEDRawData& fedData = RawDataCollection_->FEDData(*itfedList);
           fedData.resize(data.size());
           memcpy(fedData.data(),data.data(),data.size());
    } 
  } 

  std::auto_ptr<FEDRawDataCollection> streamFEDRawProduct(RawDataCollection_);
  iEvent.put(streamFEDRawProduct,outputLabelModule_);
  eventCounter_ ++ ;
}


template< typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle,TCand>::endJob(){

 if(debug_){ LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] Counted Events "<<eventCounter_;
             LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] End of the Job ----> ";
 }

}

template< typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle,TCand>::pixelFedDump( std::vector<PixelModule>::const_iterator & itDn,  
                                                      std::vector<PixelModule>::const_iterator & itUp,
                                                      const PixelRegion & region){

  for( ; itDn != itUp ; ++itDn){
    float zmodule = itDn->z-((itDn->x-beamSpotPosition_.x())*region.cosphi+(itDn->y-beamSpotPosition_.y())*region.sinphi)*region.atantheta;
    if ( std::abs(zmodule) > region.maxZ ) continue; 
    int hitFED = itDn->Fed;
    if(hitFED < FEDNumbering::MINSiPixelFEDID || hitFED > FEDNumbering::MAXSiPixelFEDID) continue;
    if(debug_) LogDebug("SelectedElectronFEDListProducer")<<"[selectedElectronFEDListProducer] electron pixel hit "<<itDn->DetId<<" hitFED "<<hitFED;
    if(!fedList_.empty()){ 
     if(std::find(fedList_.begin(),fedList_.end(),hitFED)==fedList_.end()) fedList_.push_back(hitFED);
    }
    else fedList_.push_back(hitFED); 
 }
  
 return ;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
typedef SelectedElectronFEDListProducer<reco::Electron,reco::RecoEcalCandidate> SelectedElectronFEDListProducerGsf ;
DEFINE_FWK_MODULE(SelectedElectronFEDListProducerGsf);

