#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM include files

#include "DQMServices/Core/interface/MonitorElement.h"

// work on collections
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
/// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

#include "TVector3.h"

#define TWOPI 6.283185308



using namespace std;
using namespace edm;


// ******************************************
// constructors
// *****************************************

DQMSourcePi0::DQMSourcePi0( const edm::ParameterSet& ps ) :
eventCounter_(0)
{
  dbe_ = Service<DQMStore>().operator->();
  folderName_ = ps.getUntrackedParameter<string>("FolderName","HLT/AlCaEcalPi0");
  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor",1);
  productMonitoredEB_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEBTag");
  productMonitoredEE_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEETag");
  isMonEB_ = ps.getUntrackedParameter<bool>("isMonEB",false);
  isMonEE_ = ps.getUntrackedParameter<bool>("isMonEE",false);

  saveToFile_=ps.getUntrackedParameter<bool>("SaveToFile",false);
  fileName_=  ps.getUntrackedParameter<string>("FileName","MonitorAlCaEcalPi0.root");

  gammaCandEtaSize_ = ps.getParameter<int> ("gammaCandEtaSize");
  gammaCandPhiSize_ = ps.getParameter<int> ("gammaCandPhiSize");
  if ( gammaCandPhiSize_ % 2 == 0 ||  gammaCandEtaSize_ % 2 == 0)
    edm::LogError("AlCaPi0RecHitsProducerError") << "Size of eta/phi for sliding window should be odd numbers";

  clusSeedThr_ = ps.getParameter<double> ("clusSeedThr");
  clusEtaSize_ = ps.getParameter<int> ("clusEtaSize");
  clusPhiSize_ = ps.getParameter<int> ("clusPhiSize");
  if ( clusPhiSize_ % 2 == 0 ||  clusEtaSize_ % 2 == 0)
    edm::LogError("AlCaPi0RecHitsProducerError") << "Size of eta/phi for simple clustering should be odd numbers";

  selePtGammaOne_ = ps.getParameter<double> ("selePtGammaOne");  
  selePtGammaTwo_ = ps.getParameter<double> ("selePtGammaTwo");  
  selePtPi0_ = ps.getParameter<double> ("selePtPi0");  
  seleMinvMaxPi0_ = ps.getParameter<double> ("seleMinvMaxPi0");  
  seleMinvMinPi0_ = ps.getParameter<double> ("seleMinvMinPi0");  
  seleXtalMinEnergy_ = ps.getParameter<double> ("seleXtalMinEnergy");
  seleNRHMax_ = ps.getParameter<int> ("seleNRHMax");
  //New Selection
  seleS4S9GammaOne_ = ps.getParameter<double> ("seleS4S9GammaOne");  
  seleS4S9GammaTwo_ = ps.getParameter<double> ("seleS4S9GammaTwo");  
  selePi0Iso_ = ps.getParameter<double> ("selePi0Iso");  
  selePi0BeltDR_ = ps.getParameter<double> ("selePi0BeltDR");  
  selePi0BeltDeta_ = ps.getParameter<double> ("selePi0BeltDeta");  

  ParameterLogWeighted_ = ps.getParameter<bool> ("ParameterLogWeighted");
  ParameterX0_ = ps.getParameter<double> ("ParameterX0");
  ParameterT0_barl_ = ps.getParameter<double> ("ParameterT0_barl");
  ParameterW0_ = ps.getParameter<double> ("ParameterW0");



}


DQMSourcePi0::~DQMSourcePi0()
{}


//--------------------------------------------------------
void DQMSourcePi0::beginJob(const EventSetup& context){


  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D

  hiPhiDistrEB_ = dbe_->book1D("iphiDistributionEB", "RechitEB iphi", 361, 1,361);

  hiPhiDistrEB_->setAxisTitle("i#phi ", 1);
  hiPhiDistrEB_->setAxisTitle("# rechits", 2);


  hiEtaDistrEB_ = dbe_->book1D("iEtaDistributionEB", "RechitEB ieta", 171, -85, 86);
  hiEtaDistrEB_->setAxisTitle("eta", 1);
  hiEtaDistrEB_->setAxisTitle("#rechits", 2);


  hRechitEnergyEB_ = dbe_->book1D("rhEnergyEB","rechits energy EB",160,0.,2.0);
  hRechitEnergyEB_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEB_->setAxisTitle("#rechits",2);

  hEventEnergyEB_ = dbe_->book1D("eventEnergyEB","event energy EB",100,0.,20.0);
  hEventEnergyEB_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEB_ = dbe_->book1D("nRechitsEB","#rechits in event EB",100,0.,250.);
  hNRecHitsEB_->setAxisTitle("rechits ",1);
  
  hMeanRecHitEnergyEB_ = dbe_->book1D("meanEnergyEB","Mean rechit energy EB",50,0.,2.);
  hMeanRecHitEnergyEB_->setAxisTitle("Mean Energy [GeV] ",1);
  
  hMinvPi0EB_ = dbe_->book1D("Pi0InvmassEB","Pi0 Invariant Mass in EB",100,0.,0.5);
  hMinvPi0EB_->setAxisTitle("Inv Mass [GeV] ",1);

  
  hPt1Pi0EB_ = dbe_->book1D("Pt1Pi0EB","Pt 1st most energetic Pi0 photon in EB",100,0.,20.);
  hPt1Pi0EB_->setAxisTitle("1st photon Pt [GeV] ",1);
  
  hPt2Pi0EB_ = dbe_->book1D("Pt2Pi0EB","Pt 2nd most energetic Pi0 photon in EB",100,0.,20.);
  hPt2Pi0EB_->setAxisTitle("2nd photon Pt [GeV] ",1);

  
  hPtPi0EB_ = dbe_->book1D("PtPi0EB","Pi0 Pt in EB",100,0.,20.);
  hPtPi0EB_->setAxisTitle("Pi0 Pt [GeV] ",1);

  hIsoPi0EB_ = dbe_->book1D("IsoPi0EB","Pi0 Iso in EB",50,0.,1.);
  hIsoPi0EB_->setAxisTitle("Pi0 Iso",1);

  hS4S91EB_ = dbe_->book1D("S4S91EB","S4S9 1st most energetic Pi0 photon in EB",50,0.,1.);
  hS4S91EB_->setAxisTitle("S4S9 of the 1st Pi0 Photon ",1);

  hS4S92EB_ = dbe_->book1D("S4S92EB","S4S9 2nd most energetic Pi0 photon in EB",50,0.,1.);
  hS4S92EB_->setAxisTitle("S4S9 of the 2nd Pi0 Photon",1);

  

  hRechitEnergyEE_ = dbe_->book1D("rhEnergyEE","rechits energy EE",160,0.,3.0);
  hRechitEnergyEE_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEE_->setAxisTitle("#rechits",2);

  hEventEnergyEE_ = dbe_->book1D("eventEnergyEE","event energy EE",100,0.,20.0);
  hEventEnergyEE_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEE_ = dbe_->book1D("nRechitsEE","#rechits in event EE" ,100,0.,250.);
  hNRecHitsEE_->setAxisTitle("rechits ",1);
 
  hMeanRecHitEnergyEE_ = dbe_->book1D("meanEnergyEE","Mean rechit energy EE",50,0.,5.);
  hMeanRecHitEnergyEE_-> setAxisTitle("Mean Energy [GeV] ",1);
  

}

//--------------------------------------------------------
void DQMSourcePi0::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMSourcePi0::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

//-------------------------------------------------------------

void DQMSourcePi0::analyze(const Event& iEvent, 
			       const EventSetup& iSetup ){  
 
  if (eventCounter_% prescaleFactor_ ) return; 
  eventCounter_++;

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  //  bool accept=false;

  recHitsEB_map= new std::map<DetId, EcalRecHit>();

  std::vector<EcalRecHit> seeds;
  seeds.clear();

  vector<EBDetId> usedXtals;
  usedXtals.clear();






    
  edm::Handle<EcalRecHitCollection> rhEB;
  edm::Handle<EcalRecHitCollection> rhEE;
 
  if(isMonEB_) iEvent.getByLabel(productMonitoredEB_, rhEB); 
  if(isMonEE_) iEvent.getByLabel(productMonitoredEE_, rhEE);

  EcalRecHitCollection::const_iterator itb;

  // fill EB histos
  if(isMonEB_){
    if (rhEB.isValid()){
      float etot =0;
      for(itb=rhEB->begin(); itb!=rhEB->end(); ++itb){
	
	EBDetId id(itb->id());
	double energy = itb->energy();
	if (energy > seleXtalMinEnergy_) {
	  std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
	  recHitsEB_map->insert(map_entry);
	}
	if (energy > clusSeedThr_) seeds.push_back(*itb);

	hiPhiDistrEB_->Fill(id.iphi());
	hiEtaDistrEB_->Fill(id.ieta());
	hRechitEnergyEB_->Fill(itb->energy());
	
	etot+= itb->energy();	 
      } // Eb rechits
      
      hNRecHitsEB_->Fill(rhEB->size());
      hMeanRecHitEnergyEB_->Fill(etot/rhEB->size());
      hEventEnergyEB_->Fill(etot);

      // Pi0 maker

      //cout<< " RH coll size: "<<rhEB->size()<<endl;
      //cout<< " Pi0 seeds: "<<seeds.size()<<endl;

      // Initialize the Position Calc
      const CaloSubdetectorGeometry *geometry_p;    
      const CaloSubdetectorTopology *topology_p;
      const CaloSubdetectorGeometry *geometryES_p;
      const EcalRecHitCollection *hitCollection_p = rhEB.product();

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);     
      geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
      geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

      std::map<std::string,double> providedParameters;  
      providedParameters.insert(std::make_pair("LogWeighted",ParameterLogWeighted_));
      providedParameters.insert(std::make_pair("X0",ParameterX0_));
      providedParameters.insert(std::make_pair("T0_barl",ParameterT0_barl_));
      providedParameters.insert(std::make_pair("W0",ParameterW0_));

      PositionCalc posCalculator_ = PositionCalc(providedParameters);


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
	    //      cout<<" Used det "<< EBdet<<endl;
	    std::map<DetId, EcalRecHit>::iterator aHit;
	    aHit = recHitsEB_map->find(*det);
	    usedXtals.push_back(*det);
	    RecHitsInWindow.push_back(aHit->second);
	    clus_used.push_back(*det);
	    simple_energy = simple_energy + aHit->second.energy();
	  }
	}
   
	math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_p,geometryES_p);
	//cout<< "       Simple Clustering: Total energy for this simple cluster : "<<simple_energy<<endl; 
	//cout<< "       Simple Clustering: eta phi : "<<clus_pos.eta()<<" "<<clus_pos.phi()<<endl; 
	//cout<< "       Simple Clustering: x y z : "<<clus_pos.x()<<" "<<clus_pos.y()<<" "<<clus_pos.z()<<endl; 

	float theta_s = 2. * atan(exp(-clus_pos.eta()));
	float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
	float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
	//      float p0z_s = simple_energy * cos(theta_s);
	float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);
    
	//cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<endl;

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
	for(unsigned int j=0; j<RecHitsInWindow.size();j++){
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
	      cout<<" Problem with S4 calculation "<<endl;return;
	    }
	  }
	}
	}
	s4s9Clus.push_back(*max_element( s4s9_,s4s9_+4)/simple_energy);
	//    cout<<" s4s9Clus[0] "<<s4s9_[0]/simple_energy<<" s4s9Clus[1] "<<s4s9_[1]/simple_energy<<" s4s9Clus[2] "<<s4s9_[2]/simple_energy<<" s4s9Clus[3] "<<s4s9_[3]/simple_energy<<endl;
    //    cout<<" Max "<<*max_element( s4s9_,s4s9_+4)/simple_energy<<endl;
    nClus++;
    if (nClus == MAXCLUS) return;
  }
 
      // cout<< " Pi0 clusters: "<<nClus<<endl;

      // Selection, based on Simple clustering
      //pi0 candidates
      static const int MAXPI0S = 200;
      int npi0_s=0;

      vector<EBDetId> scXtals;
      scXtals.clear();

      if (nClus <= 1) return;
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
	    //      cout<<" pt_pi0 "<<pt_pi0<<endl;
	    if (pt_pi0 < selePtPi0_)continue;
	    float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	    if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) ){

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
		//      cout<< "   Iso: k, E, drclpi0, detaclpi0, dphiclpi0 "<<k<<" "<<eClus[k]<<" "<<drclpi0<<" "<<dretaclpi0<<endl;
		if((drclpi0<selePi0BeltDR_) && (dretaclpi0<selePi0BeltDeta_) ){
		  //              cout<< "   ... good iso cluster #: "<<k<<" etClus[k] "<<etClus[k] <<endl;
		  Iso = Iso + etClus[k];
		  IsoClus.push_back(k);
		}
	      }

	      //      cout<<"  Iso/pt_pi0 "<<Iso/pt_pi0<<endl;
	      if(Iso/pt_pi0<selePi0Iso_){
		//for(unsigned int Rec=0;Rec<RecHitsCluster[i].size();Rec++)pi0EBRecHitCollection->push_back(RecHitsCluster[i][Rec]);
		//for(unsigned int Rec2=0;Rec2<RecHitsCluster[j].size();Rec2++)pi0EBRecHitCollection->push_back(RecHitsCluster[j][Rec2]);
		
		
		hMinvPi0EB_->Fill(m_inv);
		hPt1Pi0EB_->Fill(etClus[i]);
		hPt2Pi0EB_->Fill(etClus[j]);
		hPtPi0EB_->Fill(pt_pi0);
		hIsoPi0EB_->Fill(Iso/pt_pi0);
		hS4S91EB_->Fill(s4s9Clus[i]);
		hS4S92EB_->Fill(s4s9Clus[j]);
		
		npi0_s++;
	      }
	      //		for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
		//cout<< "    Iso cluster: # "<<iii<<" "<<IsoClus[iii]<<endl;  
	      //		  for(unsigned int Rec3=0;Rec3<RecHitsCluster[IsoClus[iii]].size();Rec3++)pi0EBRecHitCollection->push_back(RecHitsCluster[IsoClus[iii]][Rec3]);
	      //}   
            
		//cout <<"  Simple Clustering: pi0 Candidate pt,m_inv,i,j :   "<<pt_pi0<<" "<<m_inv<<" "<<i<<" "<<j<<" "<<endl;  

          
	      if(npi0_s == MAXPI0S) return;
	    }
	  }
	} // End of the "j" loop over Simple Clusters
      } // End of the "i" loop over Simple Clusters

      //cout<<"  (Simple Clustering) Pi0 candidates #: "<<npi0_s<<endl;

      delete recHitsEB_map;









      
    } // if valid

  } // if isMonEB

  // fill EE histos

  if(isMonEE_){  
    EcalRecHitCollection::const_iterator ite;
    
    if (rhEE.isValid()){
      
      float etot =0;
      for(ite=rhEE->begin(); ite!=rhEE->end(); ++ite){
	
	EEDetId id(ite->id());
	hRechitEnergyEE_->Fill(ite->energy());
	etot+= ite->energy();	 
      } // EE rechits
      
      hNRecHitsEE_->Fill(rhEE->size());
      hMeanRecHitEnergyEE_->Fill(etot/rhEE->size());
      hEventEnergyEE_->Fill(etot);
    }
    
  }//isMonEE
} //analyze




//--------------------------------------------------------
void DQMSourcePi0::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void DQMSourcePi0::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void DQMSourcePi0::endJob(){

  if(dbe_) {  
    if (saveToFile_) {
      dbe_->save(fileName_);
    }
  }
}


