#include "HLTrigger/special/interface/HLTEcalResonanceFilter.h"

using namespace std;
using namespace edm;


HLTEcalResonanceFilter::HLTEcalResonanceFilter(const edm::ParameterSet& iConfig)
{
  barrelHits_ = iConfig.getParameter< edm::InputTag > ("barrelHits");
  barrelClusters_ = iConfig.getParameter< edm::InputTag > ("barrelClusters");
  
  endcapHits_ = iConfig.getParameter< edm::InputTag > ("endcapHits");
  endcapClusters_ = iConfig.getParameter< edm::InputTag > ("endcapClusters");
  
  doSelBarrel_ = iConfig.getParameter<bool>("doSelBarrel");  
  if(doSelBarrel_){
    edm::ParameterSet barrelSelection = iConfig.getParameter<edm::ParameterSet>( "barrelSelection" );
    
    ///for barrel selection
    selePtGamma_ = barrelSelection.getParameter<double> ("selePtGamma");  
    selePtPair_ = barrelSelection.getParameter<double> ("selePtPair");   
    seleS4S9Gamma_ = barrelSelection.getParameter<double> ("seleS4S9Gamma");  
    seleS9S25Gamma_ = barrelSelection.getParameter<double> ("seleS9S25Gamma");  
    seleMinvMaxBarrel_ = barrelSelection.getParameter<double> ("seleMinvMaxBarrel");  
    seleMinvMinBarrel_ = barrelSelection.getParameter<double> ("seleMinvMinBarrel");  
    ptMinForIsolation_ = barrelSelection.getParameter<double> ("ptMinForIsolation");
    removePi0CandidatesForEta_ = barrelSelection.getParameter<bool>("removePi0CandidatesForEta");
    if(removePi0CandidatesForEta_){
      massLowPi0Cand_ = barrelSelection.getParameter<double>("massLowPi0Cand");
      massHighPi0Cand_ = barrelSelection.getParameter<double>("massHighPi0Cand");
    }
    seleIso_ = barrelSelection.getParameter<double> ("seleIso");  
    ptMinForIsolation_ = barrelSelection.getParameter<double> ("ptMinForIsolation");
    seleBeltDR_ = barrelSelection.getParameter<double> ("seleBeltDR");  
    seleBeltDeta_ = barrelSelection.getParameter<double> ("seleBeltDeta");  
    
    store5x5RecHitEB_ = barrelSelection.getParameter<bool> ("store5x5RecHitEB");
    BarrelHits_ = barrelSelection.getParameter<string > ("barrelHitCollection");
    
    produces< EBRecHitCollection >(BarrelHits_);
    
  }
  
  
  doSelEndcap_ = iConfig.getParameter<bool>("doSelEndcap");  
  if(doSelEndcap_){
    edm::ParameterSet endcapSelection = iConfig.getParameter<edm::ParameterSet>( "endcapSelection" );
    
    ///for endcap selection
    seleMinvMaxEndCap_ = endcapSelection.getParameter<double> ("seleMinvMaxEndCap");  
    seleMinvMinEndCap_ = endcapSelection.getParameter<double> ("seleMinvMinEndCap");
    
    region1_EndCap_ = endcapSelection.getParameter<double> ("region1_EndCap");
    selePtGammaEndCap_region1_ = endcapSelection.getParameter<double> ("selePtGammaEndCap_region1");  
    selePtPairEndCap_region1_ = endcapSelection.getParameter<double> ("selePtPairEndCap_region1");   
    
    region2_EndCap_ = endcapSelection.getParameter<double> ("region2_EndCap");
    selePtGammaEndCap_region2_ = endcapSelection.getParameter<double> ("selePtGammaEndCap_region2");  
    selePtPairEndCap_region2_ = endcapSelection.getParameter<double> ("selePtPairEndCap_region2");   
    
    selePtGammaEndCap_region3_ = endcapSelection.getParameter<double> ("selePtGammaEndCap_region3");  
    selePtPairEndCap_region3_ = endcapSelection.getParameter<double> ("selePtPairEndCap_region3");
    selePtPairMaxEndCap_region3_ = endcapSelection.getParameter<double> ("selePtPairMaxEndCap_region3");
    

    seleS4S9GammaEndCap_ = endcapSelection.getParameter<double> ("seleS4S9GammaEndCap");  
    seleS9S25GammaEndCap_ = endcapSelection.getParameter<double> ("seleS9S25GammaEndCap");  
    ptMinForIsolationEndCap_ = endcapSelection.getParameter<double> ("ptMinForIsolationEndCap");
    seleBeltDREndCap_ = endcapSelection.getParameter<double> ("seleBeltDREndCap");  
    seleBeltDetaEndCap_ = endcapSelection.getParameter<double> ("seleBeltDetaEndCap");  
    seleIsoEndCap_ = endcapSelection.getParameter<double> ("seleIsoEndCap");  
    
    store5x5RecHitEE_ = endcapSelection.getParameter<bool> ("store5x5RecHitEE");
    EndcapHits_ = endcapSelection.getParameter<string > ("endcapHitCollection");
    produces< EERecHitCollection >(EndcapHits_);
    
    
  }
  
  
  useRecoFlag_ = iConfig.getParameter<bool>("useRecoFlag");
  flagLevelRecHitsToUse_ = iConfig.getParameter<int>("flagLevelRecHitsToUse"); 
  
  useDBStatus_ = iConfig.getParameter<bool>("useDBStatus");
  statusLevelRecHitsToUse_ = iConfig.getParameter<int>("statusLevelRecHitsToUse"); 
  
  
  preshHitProducer_   = iConfig.getParameter<edm::InputTag>("preshRecHitProducer");
  ///for storing rechits ES for each selected EE clusters.
  storeRecHitES_ = iConfig.getParameter<bool>("storeRecHitES");  
  if(storeRecHitES_){

    edm::ParameterSet preshowerSelection = iConfig.getParameter<edm::ParameterSet>( "preshowerSelection" );
    
    // maximum number of matched ES clusters (in each ES layer) to each BC
    preshNclust_             = preshowerSelection.getParameter<int>("preshNclust");
    // min energy of ES clusters
    preshClustECut = preshowerSelection.getParameter<double>("preshClusterEnergyCut");
    // algo params
    float preshStripECut = preshowerSelection.getParameter<double>("preshStripEnergyCut");
    int preshSeededNst = preshowerSelection.getParameter<int>("preshSeededNstrip");
    // calibration parameters:
    calib_planeX_ = preshowerSelection.getParameter<double>("preshCalibPlaneX");
    calib_planeY_ = preshowerSelection.getParameter<double>("preshCalibPlaneY");
    gamma_        = preshowerSelection.getParameter<double>("preshCalibGamma");
    mip_          = preshowerSelection.getParameter<double>("preshCalibMIP");

    // ES algo constructor:
    presh_algo_ = new PreshowerClusterAlgo(preshStripECut,preshClustECut,preshSeededNst);

    ESHits_ = preshowerSelection.getParameter< std::string > ("ESCollection");
    produces< ESRecHitCollection >(ESHits_);
  }
  
  debug_ = iConfig.getParameter<int> ("debugLevel");
  
}


HLTEcalResonanceFilter::~HLTEcalResonanceFilter()
{
  if(storeRecHitES_){
    delete presh_algo_;
  }
}



// ------------ method called to produce the data  ------------
bool
HLTEcalResonanceFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > selEBRecHitCollection( new EBRecHitCollection );
  //Create empty output collections
  std::auto_ptr< EERecHitCollection > selEERecHitCollection( new EERecHitCollection );
  
  
  ////all selected..
  vector<DetId> selectedEBDetIds;
  vector<DetId> selectedEEDetIds; 
  
  
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloSubdetectorTopology *topology_eb = pTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorTopology *topology_ee = pTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap);
  
  
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle); 
  const CaloSubdetectorGeometry *geometry_es = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  CaloSubdetectorTopology *topology_es=0;
  if (geometry_es) {
    topology_es  = new EcalPreshowerTopology(geoHandle);
  }else{
    storeRecHitES_ = false; ///if no preshower
  }

  
  ///get status from DB
  edm::ESHandle<EcalChannelStatus> csHandle;
  if ( useDBStatus_ ) iSetup.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus &channelStatus = *csHandle; 
  
  ///==============Start to process barrel part==================///
    
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  
  iEvent.getByLabel(barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaEcalResonanceProducer: Error! can't get product barrel hits!" << std::endl;
  }
  
  const EcalRecHitCollection *hitCollection_eb = barrelRecHitsHandle.product();
    
  
  Handle<reco::BasicClusterCollection> barrelClustersHandle;
  iEvent.getByLabel(barrelClusters_,barrelClustersHandle);
  if (!barrelClustersHandle.isValid()) {
    LogDebug("") << "AlCaEcalResonanceProducer: Error! can't get product barrel clusters!" << std::endl;
  }
  const reco::BasicClusterCollection *clusterCollection_eb = barrelClustersHandle.product();
  if(debug_>=1) LogDebug("")<<" barrel_input_size:  run "<<iEvent.id().run()<<" event "<<iEvent.id().event()<<" nhitEB:  "<<hitCollection_eb->size()<<" nCluster: "<< clusterCollection_eb->size() <<endl;
  
  
  if(doSelBarrel_){
    
    map<int, vector<EcalRecHit> > RecHits5x5_clus;  ///5x5 for selected pairs
    vector<int> indIsoClus; /// Iso cluster all , 5x5 rechit not yet done
    vector<int> indCandClus; ///good cluster all ,  5x5 rechit done already during the loop
    vector<int> indClusSelected;  /// saved so far, all 
    
    doSelection( EcalBarrel,clusterCollection_eb, hitCollection_eb, channelStatus,topology_eb,
		 RecHits5x5_clus,
		 indCandClus,indIsoClus, indClusSelected);
    
    
    ///Now save all rechits in the selected clusters
    for(int i = 0; i < int(indClusSelected.size()); i++){
      int ind = indClusSelected[i];
      reco::BasicClusterCollection::const_iterator it_bc3 = clusterCollection_eb->begin() + ind; 
      const std::vector< std::pair<DetId, float> > &vid =  it_bc3->hitsAndFractions(); 
      for (std::vector<std::pair<DetId,float> >::const_iterator idItr = vid.begin();idItr != vid.end ();++idItr){
	EcalRecHitCollection::const_iterator hit  = hitCollection_eb->find(idItr->first);
	if( hit == hitCollection_eb->end()) continue;  //this should not happen.
	selEBRecHitCollection->push_back(*hit); 
	selectedEBDetIds.push_back(idItr->first);
      }
    }
    
    
    if( store5x5RecHitEB_ ){
      ///stroe 5x5 of good clusters, 5x5 arleady got 
      for(int i = 0; i < int( indCandClus.size()); i++){
	int ind = indCandClus[i];
	vector<EcalRecHit> v5x5 =  RecHits5x5_clus[ind]; 
	for(int n=0; n< int(v5x5.size()); n++){
	  DetId ed = v5x5[n].id(); 
	  std::vector<DetId>::iterator itdet = find(selectedEBDetIds.begin(),selectedEBDetIds.end(),ed); 
	  if( itdet == selectedEBDetIds.end()){
	    selectedEBDetIds.push_back(ed); 
	    selEBRecHitCollection->push_back(v5x5[n]);
	  }
	}
      }
      
      
      ///store 5x5 of Iso clusters, need to getWindow of 5x5 
      for(int i = 0; i < int( indIsoClus.size()); i++){
	int ind = indIsoClus[i];
	
	std::vector<int>::iterator  it = find(indCandClus.begin(),indCandClus.end(), ind);  ///check if already saved in the good cluster vector
	if( it != indCandClus.end()) continue; 
	
	reco::BasicClusterCollection::const_iterator it_bc3 = clusterCollection_eb->begin() + ind;
	DetId seedId = it_bc3->seed();
	std::vector<DetId> clus_v5x5 = topology_eb->getWindow(seedId,5,5);
	for( std::vector<DetId>::const_iterator idItr = clus_v5x5.begin(); idItr != clus_v5x5.end(); idItr++){
	  DetId ed = *idItr;
	  EcalRecHitCollection::const_iterator rit = hitCollection_eb->find( ed );
	  if ( rit == hitCollection_eb->end() ) continue;
	  if( ! checkStatusOfEcalRecHit(channelStatus, *rit) ) continue; 
	  std::vector<DetId>::iterator itdet = find(selectedEBDetIds.begin(),selectedEBDetIds.end(),ed);
	  if( itdet == selectedEBDetIds.end()){
	    selectedEBDetIds.push_back(ed);
	    selEBRecHitCollection->push_back(*rit);
	  }
	}      
      }
    }
    
    
  }// end of selection for eta/pi0->gg in the barrel 
    
  int eb_collsize = selEBRecHitCollection->size();
  
  if(debug_>=1) LogDebug("")<<" barrel_output_size_run "<<iEvent.id().run()<<" event "<<iEvent.id().event()<<" nEBSaved "<<selEBRecHitCollection->size()<<std::endl;
  ///==============End of  barrel ==================///
  
  
  
  //===============Start of Endcap =================/////
  ///get preshower rechits
  Handle<ESRecHitCollection> esRecHitsHandle;
  iEvent.getByLabel(preshHitProducer_,esRecHitsHandle);
  if( !esRecHitsHandle.isValid()){
    LogDebug("") << "AlCaEcalResonanceProducer: Error! can't get product esRecHit!" << std::endl;
  }
  const EcalRecHitCollection* hitCollection_es = esRecHitsHandle.product();
  // make a map of rechits:
  m_esrechit_map.clear();
  EcalRecHitCollection::const_iterator iter;
  for (iter = esRecHitsHandle->begin(); iter != esRecHitsHandle->end(); iter++) {
    //Make the map of DetID, EcalRecHit pairs
    m_esrechit_map.insert(std::make_pair(iter->id(), *iter));   
  }
  // The set of used DetID's for a given event:
  m_used_strips.clear();
  std::auto_ptr<ESRecHitCollection> selESRecHitCollection(new ESRecHitCollection );

  Handle<EERecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel(endcapHits_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaEcalResonanceProducer: Error! can't get product endcap hits!" << std::endl;
  }
  const EcalRecHitCollection *hitCollection_ee = endcapRecHitsHandle.product();
  Handle<reco::BasicClusterCollection> endcapClustersHandle;
  iEvent.getByLabel(endcapClusters_,endcapClustersHandle);
  if (!endcapClustersHandle.isValid()) {
    LogDebug("") << "AlCaEcalResonanceProducer: Error! can't get product endcap clusters!" << std::endl;
  }
  const reco::BasicClusterCollection *clusterCollection_ee = endcapClustersHandle.product();
  if(debug_>=1) LogDebug("")<<" endcap_input_size:  run "<<iEvent.id().run()<<" event "<<iEvent.id().event()<<" nhitEE:  "<<hitCollection_ee->size()<<" nhitES: "<<hitCollection_es->size()<<" nCluster: "<< clusterCollection_ee->size() <<endl;
  
  
  if(doSelEndcap_){
    
    map<int, vector<EcalRecHit> > RecHits5x5_clus;  ///5x5 for selected pairs
    vector<int> indIsoClus; /// Iso cluster all , 5x5 rechit not yet done
    vector<int> indCandClus; ///good cluster all ,  5x5 rechit done already during the loop
    vector<int> indClusSelected;  /// saved so far, all 

    doSelection( EcalEndcap,clusterCollection_ee, hitCollection_ee, channelStatus,topology_ee,
		 RecHits5x5_clus,
		 indCandClus,indIsoClus, indClusSelected);
    
    ///Now save all rechits in the selected clusters
    for(int i = 0; i < int(indClusSelected.size()); i++){
      int ind = indClusSelected[i];
      reco::BasicClusterCollection::const_iterator it_bc3 = clusterCollection_ee->begin() + ind; 
      const std::vector< std::pair<DetId, float> > &vid =  it_bc3->hitsAndFractions(); 
      for (std::vector<std::pair<DetId,float> >::const_iterator idItr = vid.begin();idItr != vid.end ();++idItr){
	EcalRecHitCollection::const_iterator hit  = hitCollection_ee->find(idItr->first);
	if( hit == hitCollection_ee->end()) continue;  //this should not happen.
	selEERecHitCollection->push_back(*hit); 
	selectedEEDetIds.push_back(idItr->first);
      }
      ///save preshower rechits 
      if(storeRecHitES_){
	std::set<DetId> used_strips_before = m_used_strips;  
	makeClusterES(it_bc3->x(),it_bc3->y(),it_bc3->z(),geometry_es,topology_es);
	std::set<DetId>::const_iterator ites = m_used_strips.begin();
	for(; ites != m_used_strips.end(); ++ ites ){
	  ESDetId d1 = ESDetId(*ites);
	  std::set<DetId>::const_iterator ites2 = find(used_strips_before.begin(),used_strips_before.end(),d1);
	  if( (ites2 == used_strips_before.end())){
	    std::map<DetId, EcalRecHit>::iterator itmap = m_esrechit_map.find(d1);
	    selESRecHitCollection->push_back(itmap->second);
	  }
	}
      }
    }


    if( store5x5RecHitEE_ ){
      ///store 5x5 of good clusters, 5x5 arleady got 
      for(int i = 0; i < int( indCandClus.size()); i++){
	int ind = indCandClus[i];
	vector<EcalRecHit> v5x5 =  RecHits5x5_clus[ind]; 
	for(int n=0; n< int(v5x5.size()); n++){
	  DetId ed = v5x5[n].id(); 
	  std::vector<DetId>::iterator itdet = find(selectedEEDetIds.begin(),selectedEEDetIds.end(),ed); 
	  if( itdet == selectedEEDetIds.end()){
	    selectedEEDetIds.push_back(ed); 
	    selEERecHitCollection->push_back(v5x5[n]);
	  }
	}
      }
      
      
      ///store 5x5 of Iso clusters, need to getWindow of 5x5 
      for(int i = 0; i < int( indIsoClus.size()); i++){
	int ind = indIsoClus[i];
	
	std::vector<int>::iterator  it = find(indCandClus.begin(),indCandClus.end(), ind);  ///check if already saved in the good cluster vector
	if( it != indCandClus.end()) continue; 
	
	reco::BasicClusterCollection::const_iterator	it_bc3 = clusterCollection_ee->begin() + ind;
	DetId seedId = it_bc3->seed();
	std::vector<DetId> clus_v5x5 = topology_ee->getWindow(seedId,5,5);
	for( std::vector<DetId>::const_iterator idItr = clus_v5x5.begin(); idItr != clus_v5x5.end(); idItr++){
	  DetId ed = *idItr;
	  EcalRecHitCollection::const_iterator rit = hitCollection_ee->find( ed );
	  if ( rit == hitCollection_ee->end() ) continue;
	  if( ! checkStatusOfEcalRecHit(channelStatus, *rit) ) continue; 
	  std::vector<DetId>::iterator itdet = find(selectedEEDetIds.begin(),selectedEEDetIds.end(),ed);
	  if( itdet == selectedEEDetIds.end()){
	    selectedEEDetIds.push_back(ed);
	    selEERecHitCollection->push_back(*rit);
	  }
	}      
      }
    }
    
  }// end of selection for eta/pi0->gg in the endcap
    
 
  
  delete topology_es;
  
  ////==============End of endcap ===============///
  
  if(debug_>=1) LogDebug("")<<" endcap_output_size run "<<iEvent.id().run()<<"_"<<iEvent.id().event()<<" nEESaved "<<selEERecHitCollection->size()<<" nESSaved: " << selESRecHitCollection->size() <<endl;
  
  
  //Put selected information in the event
  int ee_collsize = selEERecHitCollection->size();
  
  if( eb_collsize < 2 && ee_collsize <2) return false; 
  
  ////Now put into events selected rechits.
  if(doSelBarrel_){
    iEvent.put( selEBRecHitCollection, BarrelHits_);
  }  
  if(doSelEndcap_){
    iEvent.put( selEERecHitCollection, EndcapHits_);
    if(storeRecHitES_){
      iEvent.put( selESRecHitCollection, ESHits_);
    }
  }
  
  return true; 
    
  
}



void HLTEcalResonanceFilter::doSelection(int detector, const reco::BasicClusterCollection *clusterCollection,
					 const EcalRecHitCollection *hitCollection,
					 const EcalChannelStatus &channelStatus,
					 const CaloSubdetectorTopology *topology_p,
					 map<int, vector<EcalRecHit> > &RecHits5x5_clus,
					 vector<int> &indCandClus, ///good cluster all ,  5x5 rechit done already during the loop
					 vector<int> &indIsoClus, /// Iso cluster all , 5x5 rechit not yet done
					 vector<int> &indClusSelected  /// saved so far, all 
					 ){
  
  
  vector<int> indClusPi0Candidates;  ///those clusters identified as pi0s
  if( detector ==EcalBarrel && removePi0CandidatesForEta_){
    for( reco::BasicClusterCollection::const_iterator it_bc = clusterCollection->begin(); it_bc != clusterCollection->end();it_bc++){
      for( reco::BasicClusterCollection::const_iterator it_bc2 = it_bc + 1 ; it_bc2 != clusterCollection->end();it_bc2++){
	float m_pair,pt_pair,eta_pair, phi_pair; 
	calcPaircluster(*it_bc,*it_bc2, m_pair, pt_pair,eta_pair,phi_pair); 
	if(m_pair > massLowPi0Cand_ && m_pair < massHighPi0Cand_){
	  int indtmp[2]={ int( it_bc - clusterCollection->begin()), int( it_bc2 - clusterCollection->begin())};
	  for( int k=0;k<2; k++){
	    std::vector<int>::iterator it = find(indClusPi0Candidates.begin(),indClusPi0Candidates.end(),indtmp[k]);
	    if( it == indClusPi0Candidates.end()) indClusPi0Candidates.push_back(indtmp[k]);
	  }
	}
      }
    }//end of loop over finding pi0's clusters
  }
  
  
  
  
  
  double m_minCut = seleMinvMinBarrel_; 
  double m_maxCut = seleMinvMaxBarrel_; 
  double ptminforIso = ptMinForIsolation_; 
  double isoCut = seleIso_; 
  double isoBeltdrCut = seleBeltDR_ ; 
  double isoBeltdetaCut =  seleBeltDeta_; 
  double s4s9Cut = seleS4S9Gamma_; 
  double s9s25Cut = seleS9S25Gamma_; 
  bool store5x5 = store5x5RecHitEB_; 
  
  if( detector ==EcalEndcap){
    m_minCut = seleMinvMinEndCap_;
    m_maxCut = seleMinvMaxEndCap_; 
    ptminforIso = ptMinForIsolationEndCap_; 
    isoCut = seleIsoEndCap_; 
    isoBeltdrCut =  seleBeltDREndCap_; 
    isoBeltdetaCut =  seleBeltDetaEndCap_; 
    s4s9Cut = seleS4S9GammaEndCap_; 
    s9s25Cut = seleS9S25GammaEndCap_ ; 
    store5x5 = store5x5RecHitEE_; 
    
  }
  
  
  

  map<int,bool> passShowerShape_clus;  //if this cluster passed showershape cut, so no need to compute the quantity again for each loop
  
  for( reco::BasicClusterCollection::const_iterator it_bc = clusterCollection->begin(); it_bc != clusterCollection->end();it_bc++){
    
    if( detector ==EcalBarrel && removePi0CandidatesForEta_){
      std::vector<int>::iterator it = find(indClusPi0Candidates.begin(),indClusPi0Candidates.end(),int(it_bc - clusterCollection->begin()) );
      if( it != indClusPi0Candidates.end()) continue; 
    }
    
    float en1 = it_bc->energy();
    float theta1 = 2. * atan(exp(-it_bc->position().eta()));
    float pt1 = en1*sin(theta1);
 
    int ind1 = int( it_bc - clusterCollection->begin() );
    std::map<int,bool>::iterator  itmap = passShowerShape_clus.find(ind1);
    if( itmap != passShowerShape_clus.end()){
      if( itmap->second == false){
	continue; 
      }
    }
    
    for( reco::BasicClusterCollection::const_iterator it_bc2 = it_bc + 1 ; it_bc2 != clusterCollection->end();it_bc2++){
      if( detector ==EcalBarrel && removePi0CandidatesForEta_){
	std::vector<int>::iterator it = find(indClusPi0Candidates.begin(),indClusPi0Candidates.end(),int(it_bc2 - clusterCollection->begin()) );
	if( it != indClusPi0Candidates.end()) continue; 
      }
      float en2 = it_bc2 ->energy();
      float theta2 = 2. * atan(exp(-it_bc2->position().eta()));
      float pt2 = en2*sin(theta2);
      
      int ind2 = int( it_bc2 - clusterCollection->begin() );
      std::map<int,bool>::iterator  itmap = passShowerShape_clus.find(ind2);
      if( itmap != passShowerShape_clus.end()){
	if( itmap->second == false){
	  continue; 
	}
      }
      
      float m_pair,pt_pair,eta_pair, phi_pair; 
      calcPaircluster(*it_bc,*it_bc2, m_pair, pt_pair,eta_pair,phi_pair);  
      /// pt & Pt pair Cut 
      float ptmin = pt1< pt2 ? pt1:pt2; 
      if(  detector ==EcalBarrel ){
	if( ptmin < selePtGamma_ ) continue; 
	if (pt_pair < selePtPair_ )continue;
      }else{
	float etapair = fabs(eta_pair);
	if(etapair <= region1_EndCap_){
	  if(ptmin < selePtGammaEndCap_region1_ || pt_pair < selePtPairEndCap_region1_) continue; 
	}else if( etapair <= region2_EndCap_){
	  if(ptmin < selePtGammaEndCap_region2_ || pt_pair < selePtPairEndCap_region2_) continue;
	}else{
	  if(ptmin < selePtGammaEndCap_region3_ || pt_pair < selePtPairEndCap_region3_) continue;
	  if(pt_pair > selePtPairMaxEndCap_region3_ ) continue; 
	}
      }
      

      
      /// Mass window Cut 
      if( m_pair < m_minCut || m_pair > m_maxCut) continue; 
      
      
      
      
      //// Loop on cluster to measure isolation:
      vector<int> IsoClus;
      IsoClus.push_back(ind1);  //first two are good clusters
      IsoClus.push_back(ind2); 
      
      float Iso = 0;
      for( reco::BasicClusterCollection::const_iterator it_bc3 = clusterCollection->begin(); it_bc3 != clusterCollection->end();it_bc3++){
	if( it_bc3->seed() == it_bc->seed() || it_bc3->seed() == it_bc2->seed()) continue; 
	float drcl = GetDeltaR(eta_pair,it_bc3->eta(),phi_pair,it_bc3->phi()); 
	float dretacl = fabs( eta_pair - it_bc3->eta()); 
	if( drcl > isoBeltdrCut ||  dretacl > isoBeltdetaCut ) continue; 
	float pt3 = it_bc3->energy()*sin(it_bc3->position().theta()) ; 
	if( pt3 < ptminforIso) continue; 
	Iso += pt3; 
	int ind3 = int(it_bc3 - clusterCollection->begin());  /// remember which Iso cluster used 
	IsoClus.push_back( ind3);
      }
      /// Isolation cut
      if(Iso/pt_pair > isoCut) continue; 
      
      
      bool failShowerShape = false;
      for(int n=0; n<2; n++){
	int ind = IsoClus[n];
	reco::BasicClusterCollection::const_iterator it_bc3 = clusterCollection->begin() + ind; 
	std::map<int,bool>::iterator  itmap = passShowerShape_clus.find(ind);
	if( itmap != passShowerShape_clus.end()){
	  if( itmap->second == false){
	    failShowerShape = true; 
	    n=2; //exit the loop 
	  }
	}else{
	  vector<EcalRecHit> RecHitsCluster_5x5;
	  float res[3];
	  calcShowerShape(*it_bc3, channelStatus,hitCollection, topology_p,store5x5, RecHitsCluster_5x5, res);
	  float s4s9 = res[1] >0 ? res[0]/ res[1] : -1; 
	  float s9s25 = res[2] >0 ? res[1] / res[2] : -1; 
	  bool passed = s4s9 > s4s9Cut && s9s25 > s9s25Cut; 
	  passShowerShape_clus.insert(pair<int, bool>(ind,passed));
	  if( !passed ){
	    failShowerShape = true; 
	    n=2; //exit the loop 
	  }else{
	    RecHits5x5_clus.insert(pair<int, vector<EcalRecHit> >(ind,RecHitsCluster_5x5) );
	  }
	}
      }
      
      if( failShowerShape == true) continue;  //if any of the two clusters fail shower shape
      
      
      ///Save two good clusters' index( plus Iso cluster )  if not yet saved
      for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
	int ind = IsoClus[iii];
	
	if( iii < 2){
	  std::vector<int>::iterator it = find(indCandClus.begin(),indCandClus.end(), ind);  ///good cluster 
	  if( it == indCandClus.end()) indCandClus.push_back(ind);
	  else continue; 
	}else{
	  std::vector<int>::iterator it = find(indIsoClus.begin(),indIsoClus.end(), ind);  //iso cluster 
	  if( it == indIsoClus.end()) indIsoClus.push_back(ind);
	  else continue; 
	}
	
	std::vector<int>::iterator it = find(indClusSelected.begin(),indClusSelected.end(),ind); 
	if( it != indClusSelected.end()) continue; 
	indClusSelected.push_back(ind);
      }
      
    }
  }
  
  
  
  
}







void HLTEcalResonanceFilter::convxtalid(Int_t &nphi,Int_t &neta)
{
  // Barrel only
  // Output nphi 0...359; neta 0...84; nside=+1 (for eta>0), or 0 (for eta<0).
  // neta will be [-85,-1] , or [0,84], the minus sign indicates the z<0 side.
  
  if(neta > 0) neta -= 1;
  if(nphi > 359) nphi=nphi-360;
  
  // final check
  if(nphi >359 || nphi <0 || neta< -85 || neta > 84)
    {
      LogError("") <<" unexpected fatal error in HLTEcalResonanceFilter::convxtalid "<<  nphi <<  " " << neta <<  " " <<std::endl;
    }
} //end of convxtalid




int HLTEcalResonanceFilter::diff_neta_s(Int_t neta1, Int_t neta2){
  Int_t mdiff;
  mdiff=(neta1-neta2);
  return mdiff;
}

// Calculate the distance in xtals taking into account the periodicity of the Barrel
int HLTEcalResonanceFilter::diff_nphi_s(Int_t nphi1,Int_t nphi2) {
   Int_t mdiff;
   if(std::abs(nphi1-nphi2) < (360-std::abs(nphi1-nphi2))) {
     mdiff=nphi1-nphi2;
   }
   else {
   mdiff=360-std::abs(nphi1-nphi2);
   if(nphi1>nphi2) mdiff=-mdiff;
   }
   return mdiff;
}


void HLTEcalResonanceFilter::calcShowerShape(const reco::BasicCluster &bc,   const EcalChannelStatus &channelStatus, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology *topology_p, bool calc5x5, vector<EcalRecHit> &rechit5x5, float res[]){
  
  
  const std::vector< std::pair<DetId, float> > &vid =  bc.hitsAndFractions(); 
  
  
  float e2x2[4]={0};
  float e3x3 =0 ; 
  float e5x5 =0; 
  int seedx ;
  int seedy; 
  DetId seedId = bc.seed(); 
  
  bool InBarrel = true; 
  if( seedId.subdetId() == EcalBarrel){
    EBDetId ebd = EBDetId(seedId);
    seedx = ebd.ieta();
    seedy = ebd.iphi();
    convxtalid(seedy,seedx);
  }else{
    InBarrel = false; 
    EEDetId eed = EEDetId(seedId);
    seedx = eed.ix();
    seedy = eed.iy();
  }
  int x,y, dx, dy; 
  
  

  if( calc5x5 ){
    rechit5x5.clear();
    std::vector<DetId> clus_v5x5; 
    clus_v5x5 = topology_p->getWindow(seedId,5,5);
    
    for( std::vector<DetId>::const_iterator idItr = clus_v5x5.begin(); idItr != clus_v5x5.end(); idItr++){
      DetId ed = *idItr; 
      if( InBarrel == true){
	EBDetId ebd = EBDetId(ed); 
	x = ebd.ieta();
	y = ebd.iphi();
	convxtalid(y,x);
	dx = diff_neta_s(x,seedx);
	dy = diff_nphi_s(y,seedy);
      }else{
	EEDetId eed = EEDetId(ed); 
	x = eed.ix();
	y = eed.iy();
	dx = x-seedx;
	dy = y-seedy; 
      }
      EcalRecHitCollection::const_iterator rit = recHits->find( ed );
      if ( rit == recHits->end() ) continue; 
      if( ! checkStatusOfEcalRecHit(channelStatus, *rit) ) continue; 
      
      float energy = (*rit).energy();
      e5x5 += energy; 
      
      std::vector<std::pair<DetId,float> >::const_iterator idItrF = std::find( vid.begin(),vid.end(), std::make_pair(ed,1.0f));  ///has to add "f", make it float 
      if( idItrF == vid.end()){ ///only store those not belonging to this cluster
	rechit5x5.push_back(*rit);
      }else{ /// S4, S9 are defined inside the cluster, the same as below. 
	if(std::abs(dx)<=1 && std::abs(dy)<=1) {
	  if(dx <= 0 && dy <=0) e2x2[0] += energy; 
	  if(dx >= 0 && dy <=0) e2x2[1] += energy; 
	  if(dx <= 0 && dy >=0) e2x2[2] += energy; 
	  if(dx >= 0 && dy >=0) e2x2[3] += energy; 
	  e3x3 += energy; 
	}
      }
    }
    
    
  }else{
    
    for (std::vector<std::pair<DetId,float> >::const_iterator idItr = vid.begin();idItr != vid.end ();++idItr){
      DetId ed = idItr->first; 
      if( InBarrel == true){
	EBDetId ebd = EBDetId(ed); 
	x = ebd.ieta();
	y = ebd.iphi();
	convxtalid(y,x);
	dx = diff_neta_s(x,seedx);
	dy = diff_nphi_s(y,seedy);
      }else{
	EEDetId eed = EEDetId(ed); 
	x = eed.ix();
	y = eed.iy();
	dx = x-seedx;
	dy = y-seedy; 
      }
      EcalRecHitCollection::const_iterator rit = recHits->find( ed );
      if ( rit == recHits->end() ) {
	continue; 
      }
      
      float energy = (*rit).energy();
      if(std::abs(dx)<=1 && std::abs(dy)<=1){
	if(dx <= 0 && dy <=0) e2x2[0] += energy; 
	if(dx >= 0 && dy <=0) e2x2[1] += energy; 
	if(dx <= 0 && dy >=0) e2x2[2] += energy; 
	if(dx >= 0 && dy >=0) e2x2[3] += energy; 
        e3x3 += energy; 
      }
      
    }
    e5x5 = e3x3; ///if not asked to calculte 5x5, then just make e5x5 = e3x3
  }
  
  
  
  ///e2x2
  res[0] = *max_element( e2x2,e2x2+4); 
  res[1] = e3x3; 
  res[2] = e5x5; 
  
  
}




void HLTEcalResonanceFilter::calcPaircluster(const reco::BasicCluster &bc1, const reco::BasicCluster &bc2,float &m_pair,float &pt_pair,float &eta_pair, float &phi_pair){
    
  
  float theta1 = 2. * atan(exp(-bc1.eta())); 
  float en1 = bc1.energy();
  float pt1 = en1 * sin(theta1); 
  TLorentzVector v1( pt1 *cos(bc1.phi()),pt1 * sin(bc1.phi()),en1*cos(theta1),en1);
  
  float theta2 = 2. * atan(exp(-bc2.eta())); 
  float en2 = bc2.energy();
  float pt2 = en2 * sin(theta2); 
  TLorentzVector v2( pt2 *cos(bc2.phi()),pt2 * sin(bc2.phi()),en2*cos(theta2),en2);
  
  TLorentzVector v = v1 + v2; 
  
  m_pair = v.M();
  pt_pair = v.Pt();
  eta_pair = v.Eta();
  phi_pair = v.Phi();
  
  
}





bool HLTEcalResonanceFilter::checkStatusOfEcalRecHit(const EcalChannelStatus &channelStatus,const EcalRecHit &rh){
  
  if(useRecoFlag_ ){ ///from recoFlag()
    int flag = rh.recoFlag();
    if( flagLevelRecHitsToUse_ ==0){ ///good 
      if( flag != 0) return false; 
    }
    else if( flagLevelRecHitsToUse_ ==1){ ///good || PoorCalib 
      if( flag !=0 && flag != 4 ) return false; 
    }
    else if( flagLevelRecHitsToUse_ ==2){ ///good || PoorCalib || LeadingEdgeRecovered || kNeighboursRecovered,
      if( flag !=0 && flag != 4 && flag != 6 && flag != 7) return false; 
    }
  }
  if ( useDBStatus_){ //// from DB
    int status =  int(channelStatus[rh.id().rawId()].getStatusCode()); 
    if ( status > statusLevelRecHitsToUse_ ) return false; 
  }
  
  return true; 
}


float HLTEcalResonanceFilter::DeltaPhi(float phi1, float phi2){

  float diff = fabs(phi2 - phi1);
  
  while (diff >acos(-1)) diff -= 2*acos(-1);
  while (diff <= -acos(-1)) diff += 2*acos(-1);
  
  return diff; 

}


float HLTEcalResonanceFilter::GetDeltaR(float eta1, float eta2, float phi1, float phi2){
  
  return sqrt( (eta1-eta2)*(eta1-eta2) 
	       + DeltaPhi(phi1, phi2)*DeltaPhi(phi1, phi2) );
  
}


void HLTEcalResonanceFilter::makeClusterES(float x, float y, float z,const CaloSubdetectorGeometry*& geometry_es,
					CaloSubdetectorTopology*& topology_es
					){
  
  
  ///get assosicated ES clusters of this endcap cluster
  const GlobalPoint point(x,y,z);
  DetId tmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_es))->getClosestCellInPlane(point, 1);
  DetId tmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_es))->getClosestCellInPlane(point, 2);
  ESDetId strip1 = (tmp1 == DetId(0)) ? ESDetId(0) : ESDetId(tmp1);
  ESDetId strip2 = (tmp2 == DetId(0)) ? ESDetId(0) : ESDetId(tmp2);     
  
  // Get ES clusters (found by the PreshSeeded algorithm) associated with a given EE cluster.           
  for (int i2=0; i2<preshNclust_; i2++) {
    reco::PreshowerCluster cl1 = presh_algo_->makeOneCluster(strip1,&m_used_strips,&m_esrechit_map,geometry_es,topology_es);   
    reco::PreshowerCluster cl2 = presh_algo_->makeOneCluster(strip2,&m_used_strips,&m_esrechit_map,geometry_es,topology_es); 
  } // end of cycle over ES clusters
  
}
