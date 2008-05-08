#include "HLTrigger/special/interface/HLTPi0RecHitsFilter.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TVector3.h"

HLTPi0RecHitsFilter::HLTPi0RecHitsFilter(const edm::ParameterSet& iConfig)
{
  barrelHits_ = iConfig.getParameter< edm::InputTag > ("barrelHits");

  pi0BarrelHits_ = iConfig.getParameter< std::string > ("pi0BarrelHitCollection");

  gammaCandEtaSize_ = iConfig.getParameter<int> ("gammaCandEtaSize");
  gammaCandPhiSize_ = iConfig.getParameter<int> ("gammaCandPhiSize");
  if ( gammaCandPhiSize_ % 2 == 0 ||  gammaCandEtaSize_ % 2 == 0)
    edm::LogError("AlCaPi0RecHitsProducerError") << "Size of eta/phi for sliding window should be odd numbers";

  clusSeedThr_ = iConfig.getParameter<double> ("clusSeedThr");
  clusEtaSize_ = iConfig.getParameter<int> ("clusEtaSize");
  clusPhiSize_ = iConfig.getParameter<int> ("clusPhiSize");
  if ( clusPhiSize_ % 2 == 0 ||  clusEtaSize_ % 2 == 0)
    edm::LogError("AlCaPi0RecHitsProducerError") << "Size of eta/phi for simple clustering should be odd numbers";

  selePtGammaOne_ = iConfig.getParameter<double> ("selePtGammaOne");  
  selePtGammaTwo_ = iConfig.getParameter<double> ("selePtGammaTwo");  
  selePtPi0_ = iConfig.getParameter<double> ("selePtPi0");  
  seleMinvMaxPi0_ = iConfig.getParameter<double> ("seleMinvMaxPi0");  
  seleMinvMinPi0_ = iConfig.getParameter<double> ("seleMinvMinPi0");  
  seleXtalMinEnergy_ = iConfig.getParameter<double> ("seleXtalMinEnergy");
  seleNRHMax_ = iConfig.getParameter<int> ("seleNRHMax");
  //New Selection
  seleS4S9GammaOne_ = iConfig.getParameter<double> ("seleS4S9GammaOne");  
  seleS4S9GammaTwo_ = iConfig.getParameter<double> ("seleS4S9GammaTwo");  
  selePi0Iso_ = iConfig.getParameter<double> ("selePi0Iso");  
  selePi0BeltDR_ = iConfig.getParameter<double> ("selePi0BeltDR");  
  selePi0BeltDeta_ = iConfig.getParameter<double> ("selePi0BeltDeta");  

  ParameterLogWeighted_ = iConfig.getParameter<bool> ("ParameterLogWeighted");
  ParameterX0_ = iConfig.getParameter<double> ("ParameterX0");
  ParameterT0_barl_ = iConfig.getParameter<double> ("ParameterT0_barl");
  ParameterW0_ = iConfig.getParameter<double> ("ParameterW0");


  //register your products
  produces< EBRecHitCollection >(pi0BarrelHits_);
}


HLTPi0RecHitsFilter::~HLTPi0RecHitsFilter()
{
 
  //  TimingReport::current()->dump(std::cout);

}


// ------------ method called to produce the data  ------------
bool
HLTPi0RecHitsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

 //To Deal with Geometry:
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);

  // Timer
  //  const std::string category = "AlCaPi0RecHitsProducer";
  //TimerStack timers;
  //string timerName = category + "::Total";
  //timers.push(timerName);

  bool accept=false;

  Handle<EBRecHitCollection> barrelRecHitsHandle;

  iEvent.getByLabel(barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product!" << std::endl;
  }

  recHitsEB_map= new std::map<DetId, EcalRecHit>();

  std::vector<EcalRecHit> seeds;
  seeds.clear();

  vector<EBDetId> usedXtals;
  usedXtals.clear();


  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > pi0EBRecHitCollection( new EBRecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  //cout<< "   EB RecHits #: "<<barrelRecHitsHandle->size()<<endl;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {

    double energy = itb->energy();
    if (energy > seleXtalMinEnergy_) {
      std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
      recHitsEB_map->insert(map_entry);
      if (energy > clusSeedThr_) seeds.push_back(*itb);
    }
  }

  //timerName = category + "::readEBRecHitsCollection";
  //timers.push(timerName);


  // Initialize the Position Calc
  const CaloSubdetectorGeometry *geometry_p;    
  const CaloSubdetectorTopology *topology_p;
  const CaloSubdetectorGeometry *geometryES_p;
  const EcalRecHitCollection *hitCollection_p = barrelRecHitsHandle.product();

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);     
  geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",ParameterLogWeighted_));
  providedParameters.insert(std::make_pair("X0",ParameterX0_));
  providedParameters.insert(std::make_pair("T0_barl",ParameterT0_barl_));
  providedParameters.insert(std::make_pair("W0",ParameterW0_));

  PositionCalc posCalculator_ = PositionCalc(providedParameters);
  //  PositionCalc posCalculator_;

  static const int MAXCLUS = 2000;
  int nClus;
  vector<float> eClus;
  vector<float> etClus;
  vector<float> etaClus;
  vector<float> phiClus;
  vector<EBDetId> max_hit;
  vector< vector<EcalRecHit> > RecHitsCluster;
  vector<float> s4s9Clus;

  nClus=0;

  // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
  sort(seeds.begin(), seeds.end(), ecalRecHitLess());

  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    EBDetId seed_id = itseed->id();
    std::vector<EBDetId>::const_iterator usedIds;
    
//     cout<< " Start: Seed with energy "<<itseed->energy()<<endl;
//     cout<< " Start: Seed with z,ieta,iphi : "<<seed_id.zside()<<" "<<seed_id.ieta()<<" " <<seed_id.iphi()<<endl;
    bool seedAlreadyUsed=false;
    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
      if(*usedIds==seed_id){
	seedAlreadyUsed=true;
	//cout<< " Seed with energy "<<itseed->energy()<<" was used !"<<endl;
	break; 
      }
    }
    if(seedAlreadyUsed)continue;
    topology_p = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
    std::vector<DetId> clus_v = topology_p->getWindow(seed_id,clusEtaSize_,clusPhiSize_);	
    std::vector<DetId> clus_used;
    //Reject the seed if not able to build the cluster around it correctly
    //if(clus_v.size() < clusEtaSize_*clusPhiSize_){cout<<" Not enough RecHits "<<endl; continue;}
    vector<EcalRecHit> RecHitsInWindow;
    
    //    cout<<" clus_v.size() "<<clus_v.size()<<" clusEtaSize_*clusPhiSize_ "<<clusEtaSize_*clusPhiSize_<<endl;
    double simple_energy = 0; 

    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
      EBDetId EBdet = *det;
      //      cout<<" det "<< EBdet<<" ieta "<<EBdet.ieta()<<" iphi "<<EBdet.iphi()<<endl;
     bool  HitAlreadyUsed=false;
      for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	if(*usedIds==*det){
	  HitAlreadyUsed=true;
	  break;
	}
      }

      if(HitAlreadyUsed)continue;
      if (recHitsEB_map->find(*det) != recHitsEB_map->end()){
	//	cout<<" Used det "<< EBdet<<endl;
	std::map<DetId, EcalRecHit>::iterator aHit;
	aHit = recHitsEB_map->find(*det);
	usedXtals.push_back(*det);
	RecHitsInWindow.push_back(aHit->second);
	clus_used.push_back(*det);
	simple_energy = simple_energy + aHit->second.energy();
	//	cout<<" simple_energy "<<simple_energy <<"  aHit->second.energy() "<< aHit->second.energy()<<endl;
      }
    }
   
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_p,geometryES_p);
    //      cout<< "       Simple Clustering: Total energy for this simple cluster : "<<simple_energy<<endl; 
//      cout<< "       Simple Clustering: eta phi : "<<clus_pos.eta()<<" "<<clus_pos.phi()<<endl; 
//      cout<< "       Simple Clustering: x y z : "<<clus_pos.x()<<" "<<clus_pos.y()<<" "<<clus_pos.z()<<endl; 

    float theta_s = 2. * atan(exp(-clus_pos.eta()));
    float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
    float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
    //	    float p0z_s = simple_energy * cos(theta_s);
    float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);
    
    //    cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<endl;
    
    eClus.push_back(simple_energy);
    etClus.push_back(et_s);
    etaClus.push_back(clus_pos.eta());
    phiClus.push_back(clus_pos.phi());
    max_hit.push_back(seed_id);
    RecHitsCluster.push_back(RecHitsInWindow);
    //Compute S4/S9 variable
    //We are not sure to have 9 RecHits so need to check eta and phi:
    float s4s9_[4];
    for(int i=0;i<4;i++)s4s9_[i]= itseed->energy();
    for(int j=0; j<RecHitsInWindow.size();j++){
      //cout << " Simple cluster rh, ieta, iphi : "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" "<<((EBDetId)RecHitsInWindow[j].id()).iphi()<<endl;
      if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-1 && seed_id.ieta()!=1 ) || ( seed_id.ieta()==1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-2))){
	if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	  s4s9_[0]+=RecHitsInWindow[j].energy();
	}else{
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[1]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy(); 
	    }
	  }
	}
      }else{
	if(((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()){
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[3]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy(); 
	      s4s9_[2]+=RecHitsInWindow[j].energy(); 
	    }
	  }
	}else{
	  if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+1 && seed_id.ieta()!=-1 ) || ( seed_id.ieta()==-1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+2))){
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	      s4s9_[3]+=RecHitsInWindow[j].energy();
	    }else{
	      if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
		s4s9_[2]+=RecHitsInWindow[j].energy();
		s4s9_[3]+=RecHitsInWindow[j].energy();
	      }else{
		if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
		  s4s9_[2]+=RecHitsInWindow[j].energy(); 
		}
	      }
	    }
	  }else{
	    cout<<" (EBDetId)RecHitsInWindow[j].id()).ieta() "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" seed_id.ieta() "<<seed_id.ieta()<<endl;
	    cout<<" Problem with S4 calculation "<<endl;return accept;
	  }
	}
      }
    }
    s4s9Clus.push_back(*max_element( s4s9_,s4s9_+4)/simple_energy);
    //    cout<<" s4s9Clus[0] "<<s4s9_[0]/simple_energy<<" s4s9Clus[1] "<<s4s9_[1]/simple_energy<<" s4s9Clus[2] "<<s4s9_[2]/simple_energy<<" s4s9Clus[3] "<<s4s9_[3]/simple_energy<<endl;
    //    cout<<" Max "<<*max_element( s4s9_,s4s9_+4)/simple_energy<<endl;
    nClus++;
    if (nClus == MAXCLUS) return accept;
  }
 
 
  //timerName = category + "::makeSimpleClusters";
  //timers.pop_and_push(timerName);


  // Selection, based on Simple clustering
  //pi0 candidates
  static const int MAXPI0S = 200;
  int npi0_s=0;

  vector<EBDetId> scXtals;
  scXtals.clear();

  if (nClus <= 1) return accept;
  for(Int_t i=0 ; i<nClus ; i++){
    for(Int_t j=i+1 ; j<nClus ; j++){
      //      cout<<" i "<<i<<"  etClus[i] "<<etClus[i]<<" j "<<j<<"  etClus[j] "<<etClus[j]<<endl;
     if( etClus[i]>selePtGammaOne_ && etClus[j]>selePtGammaTwo_ && s4s9Clus[i]>seleS4S9GammaOne_ && s4s9Clus[j]>seleS4S9GammaTwo_){
	float theta_0 = 2. * atan(exp(-etaClus[i]));
	float theta_1 = 2. * atan(exp(-etaClus[j]));
        
	float p0x = eClus[i] * sin(theta_0) * cos(phiClus[i]);
	float p1x = eClus[j] * sin(theta_1) * cos(phiClus[j]);
	float p0y = eClus[i] * sin(theta_0) * sin(phiClus[i]);
	float p1y = eClus[j] * sin(theta_1) * sin(phiClus[j]);
	float p0z = eClus[i] * cos(theta_0);
	float p1z = eClus[j] * cos(theta_1);
        
	float pt_pi0 = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	//	cout<<" pt_pi0 "<<pt_pi0<<endl;
	if (pt_pi0 < selePtPi0_)continue;
	float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	//	cout<<" m_inv "<<m_inv<<endl;
	if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) ){

	  //cout <<"  Simple Clustering: pi0 Candidate (pt>2.5 GeV, m_inv<0.2) pt,m_inv,i,j :   "<<pt_pi0<<" "<<m_inv<<" "<<i<<" "<<j<<" "<<endl;  


	  //New Loop on cluster to measure isolation:
	  vector<int> IsoClus;
	  IsoClus.clear();
	  float Iso = 0;
	  TVector3 pi0vect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	  for(Int_t k=0 ; k<nClus ; k++){
	    if(k==i || k==j)continue;
	    TVector3 Clusvect = TVector3(eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * cos(phiClus[k]), eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * sin(phiClus[k]) , eClus[k] * cos(2. * atan(exp(-etaClus[k]))));
	    float dretaclpi0 = fabs(etaClus[k] - pi0vect.Eta());
	    float drclpi0 = Clusvect.DeltaR(pi0vect);
	    //	    cout<< "   Iso: k, E, drclpi0, detaclpi0, dphiclpi0 "<<k<<" "<<eClus[k]<<" "<<drclpi0<<" "<<dretaclpi0<<endl;
	    if((drclpi0<selePi0BeltDR_) && (dretaclpi0<selePi0BeltDeta_) ){
	      //	      cout<< "   ... good iso cluster #: "<<k<<" etClus[k] "<<etClus[k] <<endl;
	      Iso = Iso + etClus[k];
	      IsoClus.push_back(k);
	    }
	  }
	  //	  cout<<"  Iso/pt_pi0 "<<Iso/pt_pi0<<endl;
	  if(Iso/pt_pi0<selePi0Iso_){
	    for(int Rec=0;Rec<RecHitsCluster[i].size();Rec++)pi0EBRecHitCollection->push_back(RecHitsCluster[i][Rec]);
	    for(int Rec2=0;Rec2<RecHitsCluster[j].size();Rec2++)pi0EBRecHitCollection->push_back(RecHitsCluster[j][Rec2]);
	    
     
	    for(Int_t iii=0 ; iii<IsoClus.size() ; iii++){   
	      //cout<< "    Iso cluster: # "<<iii<<" "<<IsoClus[iii]<<endl;  
	      for(int Rec3=0;Rec3<RecHitsCluster[iii].size();Rec3++)pi0EBRecHitCollection->push_back(RecHitsCluster[iii][Rec3]);
	    }   
	    
	    npi0_s++;
	  }
	  
	  if(npi0_s == MAXPI0S) return accept;
	}
      }
    } // End of the "j" loop over Simple Clusters
  } // End of the "i" loop over Simple Clusters

  //timerName = category + "::makePi0Cand";
  //timers.pop_and_push(timerName);


  //  cout<<"  (Simple Clustering) Pi0 candidates #: "<<npi0_s<<endl;



      //timerName = category + "::preparePi0RecHitsCollection";
      //timers.pop_and_push(timerName);


      //Put selected information in the event
      int pi0_collsize = pi0EBRecHitCollection->size();
      //cout<< "   EB RecHits # in Collection: "<<pi0EBRecHitCollection->size()<<endl;
      if ( pi0_collsize > seleNRHMax_ ) return accept;
      if ( pi0_collsize < 1 ) return accept;
      if( npi0_s ==0 )return accept; 
      //     cout<<" Full RecHit Collection "<<hitCollection_p->size()<<endl;
      iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);
      accept = true;
      
      //timerName = category + "::storePi0RecHitsCollection";
      //timers.pop_and_push(timerName);

      //timers.clear_stack();
      
      delete recHitsEB_map;

      return accept;

}
