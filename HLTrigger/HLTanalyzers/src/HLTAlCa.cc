#include <iostream> 
#include <sstream> 
#include <istream> 
#include <fstream> 
#include <iomanip> 
#include <string> 
#include <cmath> 
#include <functional> 
#include <stdlib.h> 
#include <string.h> 

#include "HLTrigger/HLTanalyzers/interface/HLTAlCa.h"

HLTAlCa::HLTAlCa() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;

  TheMapping = new EcalElectronicsMapping(); 
  first_ = true;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTAlCa::setup(const edm::ParameterSet& pSet, TTree* HltTree) {
  
  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myEmParams.getParameterNames() ;

  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }

  // AlCa-specific parameters
  clusSeedThr_ = pSet.getParameter<double> ("clusSeedThr"); 
  clusSeedThrEndCap_ = pSet.getParameter<double> ("clusSeedThrEndCap"); 
  clusEtaSize_ = pSet.getParameter<int> ("clusEtaSize"); 
  clusPhiSize_ = pSet.getParameter<int> ("clusPhiSize"); 
  seleXtalMinEnergy_ = pSet.getParameter<double> ("seleXtalMinEnergy"); 
  RegionalMatch_ = pSet.getUntrackedParameter<bool>("RegionalMatch",true); 
  ptMinEMObj_ = pSet.getParameter<double>("ptMinEMObj"); 
  EMregionEtaMargin_ = pSet.getParameter<double>("EMregionEtaMargin"); 
  EMregionPhiMargin_ = pSet.getParameter<double>("EMregionPhiMargin"); 
  Jets_ = pSet.getUntrackedParameter<bool>("Jets",false); 
  JETSdoCentral_ = pSet.getUntrackedParameter<bool>("JETSdoCentral",true); 
  JETSdoForward_ = pSet.getUntrackedParameter<bool>("JETSdoForward",true); 
  JETSdoTau_ = pSet.getUntrackedParameter<bool>("JETSdoTau",true); 
  JETSregionEtaMargin_ = pSet.getUntrackedParameter<double>("JETS_regionEtaMargin",1.0); 
  JETSregionPhiMargin_ = pSet.getUntrackedParameter<double>("JETS_regionPhiMargin",1.0); 
  Ptmin_jets_ = pSet.getUntrackedParameter<double>("Ptmin_jets",0.); 

   
  // Parameters for the position calculation:
  edm::ParameterSet posCalcParameters = 
    pSet.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);
  

  const int kMaxClus = 2000; 
  ptClusAll = new float[kMaxClus]; 
  etaClusAll = new float[kMaxClus]; 
  phiClusAll = new float[kMaxClus]; 
  s4s9ClusAll = new float[kMaxClus]; 

  // AlCa-specific branches of the tree 
  HltTree->Branch("ohHighestEnergyEERecHit",&ohHighestEnergyEERecHit,"ohHighestEnergyEERecHit/F");
  HltTree->Branch("ohHighestEnergyEBRecHit",&ohHighestEnergyEBRecHit,"ohHighestEnergyEBRecHit/F"); 
  HltTree->Branch("ohHighestEnergyHBHERecHit",&ohHighestEnergyHBHERecHit,"ohHighestEnergyHBHERecHit/F");  
  HltTree->Branch("ohHighestEnergyHORecHit",&ohHighestEnergyHORecHit,"ohHighestEnergyHORecHit/F");   
  HltTree->Branch("ohHighestEnergyHFRecHit",&ohHighestEnergyHFRecHit,"ohHighestEnergyHFRecHit/F");   
  HltTree->Branch("Nalcapi0clusters",&Nalcapi0clusters,"Nalcapi0clusters/I");
  HltTree->Branch("ohAlcapi0ptClusAll",ptClusAll,"ohAlcapi0ptClusAll[Nalcapi0clusters]/F");
  HltTree->Branch("ohAlcapi0etaClusAll",etaClusAll,"ohAlcapi0etaClusAll[Nalcapi0clusters]/F"); 
  HltTree->Branch("ohAlcapi0phiClusAll",phiClusAll,"ohAlcapi0phiClusAll[Nalcapi0clusters]/F"); 
  HltTree->Branch("ohAlcapi0s4s9ClusAll",s4s9ClusAll,"ohAlcapi0s4s9ClusAll[Nalcapi0clusters]/F"); 
}

/* **Analyze the event** */
void HLTAlCa::analyze(const edm::Handle<EBRecHitCollection>               & ebrechits, 
		      const edm::Handle<EERecHitCollection>               & eerechits, 
		      const edm::Handle<HBHERecHitCollection>             & hbherechits,
		      const edm::Handle<HORecHitCollection>               & horechits,
		      const edm::Handle<HFRecHitCollection>               & hfrechits,
		      const edm::Handle<EBRecHitCollection>               & pi0ebrechits,  
		      const edm::Handle<EERecHitCollection>               & pi0eerechits,  
		      const edm::Handle<l1extra::L1EmParticleCollection>  & l1EGIso,   
		      const edm::Handle<l1extra::L1EmParticleCollection>  & l1EGNonIso,   
		      const edm::Handle<l1extra::L1JetParticleCollection> & l1extjetc,  
		      const edm::Handle<l1extra::L1JetParticleCollection> & l1extjetf,
		      const edm::Handle<l1extra::L1JetParticleCollection> & l1exttaujet,
		      const edm::ESHandle< EcalElectronicsMapping >       & ecalmapping,  
		      const edm::ESHandle<CaloGeometry>                   & geoHandle,  
		      const edm::ESHandle<CaloTopology>                   & pTopology,   
		      const edm::ESHandle<L1CaloGeometry>                 & l1CaloGeom, 
		      TTree* HltTree) {

  //  std::cout << " Beginning HLTAlCa " << std::endl;

  ohHighestEnergyEERecHit = -1.0;
  ohHighestEnergyEBRecHit = -1.0;
  ohHighestEnergyHBHERecHit = -1.0; 
  ohHighestEnergyHORecHit = -1.0; 
  ohHighestEnergyHFRecHit = -1.0; 

  if (ebrechits.isValid()) {
    EBRecHitCollection myebrechits;
    myebrechits = * ebrechits;
    
    float ebrechitenergy = -1.0;

    typedef EBRecHitCollection::const_iterator ebrechititer;

    for (ebrechititer i=myebrechits.begin(); i!=myebrechits.end(); i++) {
      ebrechitenergy = i->energy();
      if(ebrechitenergy > ohHighestEnergyEBRecHit)
	ohHighestEnergyEBRecHit = ebrechitenergy;
    }
  }

  if (eerechits.isValid()) { 
    EERecHitCollection myeerechits; 
    myeerechits = * eerechits; 
 
    float eerechitenergy = -1.0; 

    typedef EERecHitCollection::const_iterator eerechititer; 
 
    for (eerechititer i=myeerechits.begin(); i!=myeerechits.end(); i++) { 
      eerechitenergy = i->energy(); 
      if(eerechitenergy > ohHighestEnergyEERecHit) 
        ohHighestEnergyEERecHit = eerechitenergy; 
    } 
  } 

  if (hbherechits.isValid()) {  
    HBHERecHitCollection myhbherechits;  
    myhbherechits = * hbherechits;  
  
    float hbherechitenergy = -1.0;  
 
    typedef HBHERecHitCollection::const_iterator hbherechititer;  
  
    for (hbherechititer i=myhbherechits.begin(); i!=myhbherechits.end(); i++) {  
      hbherechitenergy = i->energy();  
      if(hbherechitenergy > ohHighestEnergyHBHERecHit)  
        ohHighestEnergyHBHERecHit = hbherechitenergy;  
    }  
  }  

  if (horechits.isValid()) {   
    HORecHitCollection myhorechits;   
    myhorechits = * horechits;   
   
    float horechitenergy = -1.0;   
  
    typedef HORecHitCollection::const_iterator horechititer;   
   
    for (horechititer i=myhorechits.begin(); i!=myhorechits.end(); i++) {   
      horechitenergy = i->energy();   
      if(horechitenergy > ohHighestEnergyHORecHit)   
        ohHighestEnergyHORecHit = horechitenergy;   
    }   
  }   

  if (hfrechits.isValid()) {   
    HFRecHitCollection myhfrechits;   
    myhfrechits = * hfrechits;   
   
    float hfrechitenergy = -1.0;   
  
    typedef HFRecHitCollection::const_iterator hfrechititer;   
   
    for (hfrechititer i=myhfrechits.begin(); i!=myhfrechits.end(); i++) {   
      hfrechitenergy = i->energy();   
      if(hfrechitenergy > ohHighestEnergyHFRecHit)   
        ohHighestEnergyHFRecHit = hfrechitenergy;   
    }   
  }   

  //////////////////////////////////////////////////////////////////////////////
  // Start of AlCa pi0 trigger variables here
  ////////////////////////////////////////////////////////////////////////////// 

  if (first_) { 

  const EcalElectronicsMapping* TheMapping_ = ecalmapping.product(); 
  *TheMapping = *TheMapping_; 
  first_ = false; 

  geometry_eb = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel); 
  geometry_ee = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalEndcap); 
  geometry_es = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower); 
   
  topology_eb = pTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel); 
  topology_ee = pTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap); 
  }

  // End pi0 setup 
    
  //first get all the FEDs around EM objects with PT > defined value. 
  FEDListUsed.clear();
  std::vector<int>::iterator it; 
  if( RegionalMatch_){
    
    for( l1extra::L1EmParticleCollection::const_iterator emItr = l1EGIso->begin();
	 emItr != l1EGIso->end() ;++emItr ){
      
      float pt = emItr -> pt();
      
      if( pt< ptMinEMObj_ ) continue; 
	
      int etaIndex = emItr->gctEmCand()->etaIndex() ;
      int phiIndex = emItr->gctEmCand()->phiIndex() ;
      double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
      double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
      double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;
      
      std::vector<int> feds = ListOfFEDS(etaLow, etaHigh, phiLow, phiHigh, EMregionEtaMargin_, EMregionPhiMargin_);
      for (int n=0; n < (int)feds.size(); n++) {
	int fed = feds[n];
	it = find(FEDListUsed.begin(),FEDListUsed.end(),fed);
	if( it == FEDListUsed.end()){
	  FEDListUsed.push_back(fed);
	}
      }
    }
    
    for( l1extra::L1EmParticleCollection::const_iterator emItr = l1EGNonIso->begin();
	 emItr != l1EGNonIso->end() ;++emItr ){
      
      float pt = emItr -> pt();

      if( pt< ptMinEMObj_ ) continue; 

      int etaIndex = emItr->gctEmCand()->etaIndex() ;
      int phiIndex = emItr->gctEmCand()->phiIndex() ;
      double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
      double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
      double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;
      
      std::vector<int> feds = ListOfFEDS(etaLow, etaHigh, phiLow, phiHigh, EMregionEtaMargin_, EMregionPhiMargin_);
      for (int n=0; n < (int)feds.size(); n++) {
	int fed = feds[n];
	it = find(FEDListUsed.begin(),FEDListUsed.end(),fed);
	if( it == FEDListUsed.end()){
	  FEDListUsed.push_back(fed);
	}
      }
    }

    if( Jets_ ){
      
      double epsilon = 0.01;
      
      if (JETSdoCentral_) {
	
	for (l1extra::L1JetParticleCollection::const_iterator jetItr=l1extjetc->begin(); jetItr != l1extjetc->end(); jetItr++) {
	  
	  double pt    =   jetItr-> pt();
	  double eta   =   jetItr-> eta();
	  double phi   =   jetItr-> phi();
	  
	  if (pt < Ptmin_jets_ ) continue;
	  
	  std::vector<int> feds = ListOfFEDS(eta, eta, phi-epsilon, phi+epsilon, JETSregionEtaMargin_, JETSregionPhiMargin_);
	  for (int n=0; n < (int)feds.size(); n++) {
	    int fed = feds[n];
	    it = find(FEDListUsed.begin(),FEDListUsed.end(),fed);
	    if( it == FEDListUsed.end()){
	      FEDListUsed.push_back(fed);
	    }
	  }
	}
	
      }
      
      if (JETSdoForward_) {
	
	for (l1extra::L1JetParticleCollection::const_iterator jetItr=l1extjetf->begin(); jetItr != l1extjetf->end(); jetItr++) {
	  
	  double pt    =  jetItr -> pt();
	  double eta   =  jetItr -> eta();
	  double phi   =  jetItr -> phi();
	  
	  if (pt < Ptmin_jets_ ) continue;
	  
	  std::vector<int> feds = ListOfFEDS(eta, eta, phi-epsilon, phi+epsilon, JETSregionEtaMargin_, JETSregionPhiMargin_);
	  
	  for (int n=0; n < (int)feds.size(); n++) {
	    int fed = feds[n];
	    it = find(FEDListUsed.begin(),FEDListUsed.end(),fed);
	    if( it == FEDListUsed.end()){
	      FEDListUsed.push_back(fed);
	    }
	  }
	  
	}
      }
      
      if (JETSdoTau_) {
	
	for (l1extra::L1JetParticleCollection::const_iterator jetItr=l1exttaujet->begin(); jetItr != l1exttaujet->end(); jetItr++) {
	  
	  double pt    =  jetItr -> pt();
	  double eta   =  jetItr -> eta();
	  double phi   =  jetItr -> phi();
	  
	  if (pt < Ptmin_jets_ ) continue;
	  
	  std::vector<int> feds = ListOfFEDS(eta, eta, phi-epsilon, phi+epsilon, JETSregionEtaMargin_, JETSregionPhiMargin_);
	  for (int n=0; n < (int)feds.size(); n++) {
	    int fed = feds[n];
	    it = find(FEDListUsed.begin(),FEDListUsed.end(),fed);
	    if( it == FEDListUsed.end()){
	      FEDListUsed.push_back(fed);
	    }
	  }
	  
	}
      }
    }
  } //// end of getting FED List if asked to do regional match ( for 21x ecalRawtoDigi. etc)

  //// end of getting FED List 
  //separate into barrel and endcap to speed up when checking 
  FEDListUsedBarrel.clear(); 
  FEDListUsedEndcap.clear(); 
  for(  int j=0; j< int(FEDListUsed.size());j++){ 
    int fed = FEDListUsed[j]; 
    
    if( fed >= 10 && fed <= 45){ 
      FEDListUsedBarrel.push_back(fed); 
    }else FEDListUsedEndcap.push_back(fed); 
  } 
  
  //==============Start to process barrel part==================/// 

  if (pi0ebrechits.isValid()) { 

    const EcalRecHitCollection *hitCollection_p = pi0ebrechits.product(); 
    
    std::vector<EcalRecHit> seeds; 
    seeds.clear(); 
    
    std::vector<EBDetId> usedXtals; 
    usedXtals.clear(); 
    
    detIdEBRecHits.clear(); //// EBDetId 
    EBRecHits.clear();  // EcalRecHit 
  
    detIdEBRecHits.clear(); //// EBDetId 
    EBRecHits.clear();  // EcalRecHit 
    
    ////make seeds.  
    EBRecHitCollection::const_iterator itb; 
    
    for (itb=pi0ebrechits->begin(); itb!=pi0ebrechits->end(); itb++) { 
      double energy = itb->energy(); 
      if( energy < seleXtalMinEnergy_) continue;  
      
      EBDetId det = itb->id(); 
      
      if (RegionalMatch_){ 
	int fed = TheMapping->DCCid(det); 
	it = find(FEDListUsedBarrel.begin(),FEDListUsedBarrel.end(),fed); 
	if(it == FEDListUsedBarrel.end()) continue;  
      } 
      
      detIdEBRecHits.push_back(det); 
      EBRecHits.push_back(*itb); 
      
      if (energy > clusSeedThr_) seeds.push_back(*itb);       
    } 
    
    int nClus;
    std::vector<float> eClus;
    std::vector<float> etClus;
    std::vector<float> etaClus;
    std::vector<float> phiClus;
    std::vector<EBDetId> max_hit;
    std::vector< std::vector<EcalRecHit> > RecHitsCluster;
    std::vector<float> s4s9Clus;
    
    nClus=0;
    
    // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
    sort(seeds.begin(), seeds.end(), eecalRecHitLess());
    
    Nalcapi0clusters = 0;
    nClusAll = 0; 
    
    for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
      EBDetId seed_id = itseed->id();
      std::vector<EBDetId>::const_iterator usedIds;
      
      bool seedAlreadyUsed=false;
      for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	if(*usedIds==seed_id){
	  seedAlreadyUsed=true;
	  break; 
	}
      }
      
      if(seedAlreadyUsed)continue;
      
      std::vector<DetId> clus_v = topology_eb->getWindow(seed_id,clusEtaSize_,clusPhiSize_);
      std::vector< std::pair<DetId, float> > clus_used;
      
      std::vector<EcalRecHit> RecHitsInWindow;
      
      double simple_energy = 0; 
      
      for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
	EBDetId EBdet = *det;
	
	bool  HitAlreadyUsed=false;
	for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	  if(*usedIds==*det){
	    HitAlreadyUsed=true;
	    break;
	  }
	}
	
	if(HitAlreadyUsed)continue;
      
	///once again. check FED of this det.
	  if (RegionalMatch_){
	    int fed = TheMapping->DCCid(EBdet);
	    it = find(FEDListUsedBarrel.begin(),FEDListUsedBarrel.end(),fed);
	    if(it == FEDListUsedBarrel.end()) continue; 
	  }
	  
	  
	  std::vector<EBDetId>::iterator itdet = find( detIdEBRecHits.begin(),detIdEBRecHits.end(),EBdet);
	  if(itdet == detIdEBRecHits.end()) continue; 
	  
	  int nn = int(itdet - detIdEBRecHits.begin());
	  usedXtals.push_back(*det);
	  RecHitsInWindow.push_back(EBRecHits[nn]);
	  clus_used.push_back( std::pair<DetId, float>(*det, 1) );
	  simple_energy = simple_energy + EBRecHits[nn].energy();
      }
      
      math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_eb,geometry_es);
      
      float theta_s = 2. * atan(exp(-clus_pos.eta()));
      float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
      float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
      float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);
      
      eClus.push_back(simple_energy);
      etClus.push_back(et_s);
      etaClus.push_back(clus_pos.eta());
      phiClus.push_back(clus_pos.phi());
      max_hit.push_back(seed_id);
      RecHitsCluster.push_back(RecHitsInWindow);
      
      //Compute S4/S9 variable
      //We are not sure to have 9 RecHits so need to check eta and phi:
      
      //check s4s9
      float s4s9_tmp[4];
      for(int i=0;i<4;i++)s4s9_tmp[i]= 0;
      
      int seed_ieta = seed_id.ieta();
      int seed_iphi = seed_id.iphi();
      
      convxtalid( seed_iphi,seed_ieta);
      
      for(unsigned int j=0; j<RecHitsInWindow.size();j++){
	EBDetId det = (EBDetId)RecHitsInWindow[j].id(); 
	
	int ieta = det.ieta();
	int iphi = det.iphi();
	
	convxtalid(iphi,ieta);
	
	float en = RecHitsInWindow[j].energy(); 
	
	int dx = diff_neta_s(seed_ieta,ieta);
	int dy = diff_nphi_s(seed_iphi,iphi);
	
	if(dx <= 0 && dy <=0) s4s9_tmp[0] += en; 
	if(dx >= 0 && dy <=0) s4s9_tmp[1] += en; 
	if(dx <= 0 && dy >=0) s4s9_tmp[2] += en; 
	if(dx >= 0 && dy >=0) s4s9_tmp[3] += en; 
      }
      
      float s4s9_max = *std::max_element( s4s9_tmp,s4s9_tmp+4)/simple_energy; 
      s4s9Clus.push_back(s4s9_max);
      
      if(nClusAll<MAXCLUS){
	ptClusAll[nClusAll] = et_s;
	etaClusAll[nClusAll] = clus_pos.eta();
	phiClusAll[nClusAll] = clus_pos.phi();
	s4s9ClusAll[nClusAll] = s4s9_max;
	nClusAll++; 
	Nalcapi0clusters++;
      }
      
      nClus++;
    }
  }

  //==============Start of  Endcap ==================//

  if (pi0eerechits.isValid()) {  
    
    const EcalRecHitCollection *hitCollection_e = pi0eerechits.product();  
    
    detIdEERecHits.clear(); //// EEDetId
    EERecHits.clear();  // EcalRecHit
    
    std::vector<EcalRecHit> seedsEndCap;
    seedsEndCap.clear();
    
    std::vector<EEDetId> usedXtalsEndCap;
    usedXtalsEndCap.clear();
    
    ////make seeds. 
    EERecHitCollection::const_iterator ite;
    for (ite=pi0eerechits->begin(); ite!=pi0eerechits->end(); ite++) {
      double energy = ite->energy();
      if( energy < seleXtalMinEnergy_) continue; 
      
      EEDetId det = ite->id();
      if (RegionalMatch_){
	EcalElectronicsId elid = TheMapping->getElectronicsId(det);
	int fed = elid.dccId();
	it = find(FEDListUsedEndcap.begin(),FEDListUsedEndcap.end(),fed);
	if(it == FEDListUsedEndcap.end()) continue; 
      }
      
      detIdEERecHits.push_back(det);
      EERecHits.push_back(*ite);
      
    
      if (energy > clusSeedThrEndCap_) seedsEndCap.push_back(*ite);
      
    }
    
    //Create empty output collections
    std::auto_ptr< EERecHitCollection > pi0EERecHitCollection( new EERecHitCollection );
    
    int nClusEndCap;
    std::vector<float> eClusEndCap;
    std::vector<float> etClusEndCap;
    std::vector<float> etaClusEndCap;
    std::vector<float> phiClusEndCap;
    std::vector< std::vector<EcalRecHit> > RecHitsClusterEndCap;
    std::vector<float> s4s9ClusEndCap;
    nClusEndCap=0;
    
    // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
    sort(seedsEndCap.begin(), seedsEndCap.end(), eecalRecHitLess());
    
    for (std::vector<EcalRecHit>::iterator itseed=seedsEndCap.begin(); itseed!=seedsEndCap.end(); itseed++) {
      EEDetId seed_id = itseed->id();
      std::vector<EEDetId>::const_iterator usedIds;
      
      bool seedAlreadyUsed=false;
      for(usedIds=usedXtalsEndCap.begin(); usedIds!=usedXtalsEndCap.end(); usedIds++){
	if(*usedIds==seed_id){
	  seedAlreadyUsed=true;
	  break; 
	}
      }
      
      if(seedAlreadyUsed)continue;
      
      std::vector<DetId> clus_v = topology_ee->getWindow(seed_id,clusEtaSize_,clusPhiSize_);
      std::vector< std::pair<DetId, float> > clus_used;
      
      std::vector<EcalRecHit> RecHitsInWindow;
      
      double simple_energy = 0; 
      
      for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
	EEDetId EEdet = *det;
	
	bool  HitAlreadyUsed=false;
	for(usedIds=usedXtalsEndCap.begin(); usedIds!=usedXtalsEndCap.end(); usedIds++){
	  if(*usedIds==*det){
	    HitAlreadyUsed=true;
	    break;
	  }
	}
	
	if(HitAlreadyUsed)continue;
	
	//once again. check FED of this det.
	if (RegionalMatch_){
	  EcalElectronicsId elid = TheMapping->getElectronicsId(EEdet);
	  int fed = elid.dccId();
	  it = find(FEDListUsedEndcap.begin(),FEDListUsedEndcap.end(),fed);
	  if(it == FEDListUsedEndcap.end()) continue; 
	}
	
	std::vector<EEDetId>::iterator itdet = find( detIdEERecHits.begin(),detIdEERecHits.end(),EEdet);
	if(itdet == detIdEERecHits.end()) continue; 
	
	int nn = int(itdet - detIdEERecHits.begin());
	usedXtalsEndCap.push_back(*det);
	RecHitsInWindow.push_back(EERecHits[nn]);
	clus_used.push_back( std::pair<DetId, float>(*det, 1) );
	simple_energy = simple_energy + EERecHits[nn].energy();
      }
      
      math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_e,geometry_ee,geometry_es);
      
      float theta_s = 2. * atan(exp(-clus_pos.eta()));
      float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
      float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
      float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);
      
      eClusEndCap.push_back(simple_energy);
      etClusEndCap.push_back(et_s);
      etaClusEndCap.push_back(clus_pos.eta());
      phiClusEndCap.push_back(clus_pos.phi());
      RecHitsClusterEndCap.push_back(RecHitsInWindow);
      
      //Compute S4/S9 variable
      //We are not sure to have 9 RecHits so need to check eta and phi:
      float s4s9_tmp[4];
      for(int i=0;i<4;i++) s4s9_tmp[i]= 0; 
      
      int ixSeed = seed_id.ix();
      int iySeed = seed_id.iy();
      for(unsigned int j=0; j<RecHitsInWindow.size();j++){
	EEDetId det_this = (EEDetId)RecHitsInWindow[j].id(); 
	int dx = ixSeed - det_this.ix();
	int dy = iySeed - det_this.iy();
	
	float en = RecHitsInWindow[j].energy(); 
	
	if(dx <= 0 && dy <=0) s4s9_tmp[0] += en; 
	if(dx >= 0 && dy <=0) s4s9_tmp[1] += en; 
	if(dx <= 0 && dy >=0) s4s9_tmp[2] += en; 
	if(dx >= 0 && dy >=0) s4s9_tmp[3] += en; 
      }
      
      float s4s9_max = *std::max_element( s4s9_tmp,s4s9_tmp+4)/simple_energy;
      s4s9ClusEndCap.push_back(s4s9_max);
      
      if(nClusAll<MAXCLUS){
	ptClusAll[nClusAll] = et_s;
	etaClusAll[nClusAll] = clus_pos.eta();
	phiClusAll[nClusAll] = clus_pos.phi();
	s4s9ClusAll[nClusAll] = s4s9_max;
	nClusAll++;
	Nalcapi0clusters++;
      }
      
      nClusEndCap++;
    }
  }

  // If no barrel OR endcap rechits, set everything to 0
  if(!(pi0eerechits.isValid()) && !(pi0ebrechits.isValid()))
    Nalcapi0clusters = 0;
}


////////////////////////////////////////////////////////////////////////////// 
// Below here are helper functions for the pi0 variables
//////////////////////////////////////////////////////////////////////////////  


/////FED list 
std::vector<int> HLTAlCa::ListOfFEDS(double etaLow, double etaHigh, double phiLow, 
				    double phiHigh, double etamargin, double phimargin)
{

  std::vector<int> FEDs;

  if (phimargin > Geom::pi()) phimargin =  Geom::pi() ;

  etaLow -= etamargin;
  etaHigh += etamargin;
  double phiMinus = phiLow - phimargin;
  double phiPlus = phiHigh + phimargin;

  bool all = false;
  double dd = fabs(phiPlus-phiMinus);

  if (dd > 2.*Geom::pi() ) all = true;

  while (phiPlus > Geom::pi()) { phiPlus -= 2.*Geom::pi() ; }
  while (phiMinus < 0) { phiMinus += 2.*Geom::pi() ; }
  if ( phiMinus > Geom::pi()) phiMinus -= 2.*Geom::pi() ;

  double dphi = phiPlus - phiMinus;
  if (dphi < 0) dphi += 2.*Geom::pi() ;

  if (dphi > Geom::pi()) {
    int fed_low1 = TheMapping->GetFED(etaLow,phiMinus*180./Geom::pi());
    int fed_low2 = TheMapping->GetFED(etaLow,phiPlus*180./Geom::pi());

    if (fed_low1 == fed_low2) all = true;
    int fed_hi1 = TheMapping->GetFED(etaHigh,phiMinus*180./Geom::pi());
    int fed_hi2 = TheMapping->GetFED(etaHigh,phiPlus*180./Geom::pi());
    
    if (fed_hi1 == fed_hi2) all = true;
  }

  if (all) {

    phiMinus = -20 * Geom::pi() / 180.;  // -20 deg
    phiPlus = -40 * Geom::pi() / 180.;  // -20 deg
  }

  const EcalEtaPhiRegion ecalregion(etaLow,etaHigh,phiMinus,phiPlus);

  FEDs = TheMapping->GetListofFEDs(ecalregion);

  return FEDs;

}


////already existing , int EcalElectronicsMapping::DCCid(const EBDetId& id)
int HLTAlCa::convertSmToFedNumbBarrel(int ieta, int smId){
    
  if( ieta<=-1) return smId - 9; 
  else return smId + 27; 
  
  
}


void HLTAlCa::convxtalid(Int_t &nphi,Int_t &neta)
{
  // Barrel only
  // Output nphi 0...359; neta 0...84; nside=+1 (for eta>0), or 0 (for eta<0).
  // neta will be [-85,-1] , or [0,84], the minus sign indicates the z<0 side.
  
  if(neta > 0) neta -= 1;
  if(nphi > 359) nphi=nphi-360;
  
  // final check
  if(nphi >359 || nphi <0 || neta< -85 || neta > 84)
    {
      std::cout <<" unexpected fatal error in HLTPi0::convxtalid "<<  nphi <<  " " << neta <<  " " <<std::endl;
      //exit(1);
    }
} //end of convxtalid




int HLTAlCa::diff_neta_s(Int_t neta1, Int_t neta2){
  Int_t mdiff;
  mdiff=(neta1-neta2);
  return mdiff;
}


// Calculate the distance in xtals taking into account the periodicity of the Barrel 
int HLTAlCa::diff_nphi_s(Int_t nphi1,Int_t nphi2) { 
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

