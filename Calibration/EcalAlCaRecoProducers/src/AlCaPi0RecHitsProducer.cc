#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPi0RecHitsProducer.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlCaPi0RecHitsProducer::AlCaPi0RecHitsProducer(const edm::ParameterSet& iConfig)
{
  ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");
  barrelHits_ = iConfig.getParameter< std::string > ("barrelHitCollection");
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


  //register your products
  produces< EBRecHitCollection >(pi0BarrelHits_);
}


AlCaPi0RecHitsProducer::~AlCaPi0RecHitsProducer()
{
 
  TimingReport::current()->dump(std::cout);

}


// ------------ method called to produce the data  ------------
void
AlCaPi0RecHitsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  // Timer
  const std::string category = "AlCaPi0RecHitsProducer";
  TimerStack timers;
  string timerName = category + "::Total";
  timers.push(timerName);


  Handle<EBRecHitCollection> barrelRecHitsHandle;

  try {
    iEvent.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  } catch ( std::exception& ex ) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product!" << std::endl;
  }

  recHitsEB_map = new std::map<DetId, EcalRecHit>();

  std::vector<EcalRecHit> seeds;
  seeds.clear();

  vector<EBDetId> usedXtals;
  usedXtals.clear();


  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > pi0EBRecHitCollection( new EBRecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  cout<< "   EB RecHits #: "<<barrelRecHitsHandle->size()<<endl;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {

    std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
    recHitsEB_map->insert(map_entry);

    double energy = itb->energy();
    if (energy > clusSeedThr_) seeds.push_back(*itb);
  }

  timerName = category + "::readEBRecHitsCollection";
  timers.push(timerName);


  // Initialize the Position Calc
  const CaloSubdetectorGeometry *geometry_p;    
  const CaloSubdetectorTopology *topology_p;
  const CaloSubdetectorGeometry *geometryES_p;
  const EcalRecHitCollection *hitCollection_p = barrelRecHitsHandle.product();

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);     
  geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  topology_p = new EcalBarrelTopology(geoHandle);
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",1));
  providedParameters.insert(std::make_pair("X0",0.89));
  providedParameters.insert(std::make_pair("T0_barl",5.7));
  providedParameters.insert(std::make_pair("W0",4.2));

  PositionCalc posCalculator_ = PositionCalc(providedParameters);
  //PositionCalc::Initialize(providedParameters, &recHitsEB_map, &(*geometry_p));

  static const int MAXCLUS = 200;
  int nClus;
  float eClus[MAXCLUS];
  float etClus[MAXCLUS];
  float etaClus[MAXCLUS];
  float phiClus[MAXCLUS];
  EBDetId max_hit[MAXCLUS];  

  nClus=0;
  for(int i=0; i<MAXCLUS; i++){
    eClus[i] = 0;
    etClus[i] = 0;
    etaClus[i] = 0;
    phiClus[i] = 0;
    max_hit[i] = EBDetId(0);
  }

  // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
  sort(seeds.begin(), seeds.end(), ecalRecHitLess());


  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    EBDetId seed_id = itseed->id();
    std::vector<EBDetId>::const_iterator usedIds;
    
    //cout<< " Start: Seed with energy "<<itseed->energy()<<endl;
    //cout<< " Start: Seed with z,ieta,iphi : "<<seed_id.zside()<<" "<<seed_id.ieta()<<" " <<seed_id.iphi()<<endl;
    bool seedAlreadyUsed=false;
    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++)
      if(*usedIds==seed_id)
	{
	  seedAlreadyUsed=true;
	  //cout<< " Seed with energy "<<itseed->energy()<<" was used !"<<endl;
	  break;
	}
    if(!seedAlreadyUsed)	    
      {
	double simple_energy = 0; 
	std::vector<DetId> clus_v;
	clus_v.clear();
    for (int icry=0;icry< clusEtaSize_*clusPhiSize_;icry++)
      {
	
	int row = icry / clusEtaSize_;
	int column= icry % clusEtaSize_;
	int curr_eta=seed_id.ieta() + column - (clusEtaSize_/2);
	int curr_phi=seed_id.iphi() + row - (clusPhiSize_/2);
	
	if (curr_eta * seed_id.ieta() <= 0) {if (seed_id.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
	if (curr_phi < 1) curr_phi += 360;
	if (curr_phi > 360) curr_phi -= 360;
	
	try
	  {
	    EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);
	    std::vector<EBDetId>::const_iterator usedIds;
			  
	    std::map<DetId, EcalRecHit>::iterator aHit;
	    bool HitAlreadyUsed=false;
	    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++)
	      if(*usedIds==det)
		{
		  HitAlreadyUsed=true;
		  break;
		}
	    
	    if(!HitAlreadyUsed)
	      if (recHitsEB_map->find(det) != recHitsEB_map->end())
		{
		  aHit = recHitsEB_map->find(det);
		  usedXtals.push_back(det);
		  clus_v.push_back(det);

		  simple_energy = simple_energy + aHit->second.energy();
		  
		  //EBDetId sel_rh = aHit->second.detid();
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : z,ieta,iphi "<<sel_rh.zside()<<" "<<sel_rh.ieta()<<" "<<sel_rh.iphi()<<endl;    
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : tower_ieta,tower_iphi "<<sel_rh.tower_ieta()<<" "<<sel_rh.tower_iphi()<<endl;   
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : iSM, ic "<<sel_rh.ism()<<" "<<sel_rh.ic()<<endl;
		  
		}
	  }
	catch (...)
	  {
	  }
      }
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_v,hitCollection_p,geometry_p,geometryES_p);
    //cout<< "       Simple Clustering: Total energy for this simple cluster : "<<simple_energy<<endl; 
    //cout<< "       Simple Clustering: eta phi : "<<clus_pos.eta()<<" "<<clus_pos.phi()<<endl; 
    //cout<< "       Simple Clustering: x y z : "<<clus_pos.x()<<" "<<clus_pos.y()<<" "<<clus_pos.z()<<endl; 

	    float theta_s = 2. * atan(exp(-clus_pos.eta()));
	    float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
	    float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
	    float p0z_s = simple_energy * cos(theta_s);
	    float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);

	    //cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<p0z_s<<endl;
    
	    eClus[nClus] = simple_energy;
	    etClus[nClus] = et_s;
	    etaClus[nClus] = clus_pos.eta();
	    phiClus[nClus] = clus_pos.phi();
	    max_hit[nClus] = seed_id;
	    
	    nClus++;
      }
  }
  
  timerName = category + "::makeSimpleClusters";
  timers.pop_and_push(timerName);


  // Selection, based on Simple clustering
  //pi0 candidates
  static const int MAXPI0S = 20;
  int npi0_s=0;
  int sClus_1[MAXPI0S],sClus_2[MAXPI0S];
  for(int i=0; i<MAXPI0S; i++){
    sClus_1[i]=0;
    sClus_2[i]=0;
  }

  if (nClus > 1) 
    {
      for(Int_t i=0 ; i<nClus ; i++)
        {
          for(Int_t j=i+1 ; j<nClus ; j++)
            {

              if( etClus[i]>selePtGammaOne_ && etClus[j]>selePtGammaTwo_) 
		{
                
		  float theta_0 = 2. * atan(exp(-etaClus[i]));
		  float theta_1 = 2. * atan(exp(-etaClus[j]));
                
		  float p0x = eClus[i] * sin(theta_0) * cos(phiClus[i]);
		  float p1x = eClus[j] * sin(theta_1) * cos(phiClus[j]);
		  float p0y = eClus[i] * sin(theta_0) * sin(phiClus[i]);
		  float p1y = eClus[j] * sin(theta_1) * sin(phiClus[j]);
		  float p0z = eClus[i] * cos(theta_0);
		  float p1z = eClus[j] * cos(theta_1);
                
		  //
                
		  float pt_pi0 = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
		  //float dr_pi0 = sqrt ( (etaIslandBCEB[i]-etaIslandBCEB[j])*(etaIslandBCEB[i]-etaIslandBCEB[j]) + (phiIslandBCEB[i]-phiIslandBCEB[j])*(phiIslandBCEB[i]-phiIslandBCEB[j]) );
		  if (pt_pi0 > selePtPi0_) 
		    {
		      float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
		      if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) )
			{
			  cout <<"  Simple Clustering: pi0 Candidate (pt>2.5 GeV, m_inv<0.2) pt,m_inv,i,j :   "<<pt_pi0<<" "<<m_inv<<" "<<i<<" "<<j<<" "<<endl;  
			  sClus_1[npi0_s]=i;
			  sClus_2[npi0_s]=j;
			  npi0_s++;
			}

		    }
                
		}
            } // End of the "j" loop over Simple Clusters
        } // End of the "i" loop over Simple Clusters

    }

  timerName = category + "::makePi0Cand";
  timers.pop_and_push(timerName);


      cout<<" "<<endl;
      cout<<"  (Simple Clustering) Pi0 candidates #: "<<npi0_s<<endl;

      for(Int_t jj=0;jj<nClus;jj++)
	{
	  EBDetId maxHit = max_hit[jj];
	  //cout << "        Simple Clustering: maxHit jj eta phi : "<<jj<<" "<<maxHit.ieta()<<" "<<maxHit.iphi()<<endl;
	}

      vector<EBDetId> scXtals;
      scXtals.clear();
      for(Int_t i=0 ; i<npi0_s ; i++)
	{
	  //cout<<"     Pi0: i, Bc1, Bc2 "<<i<<" "<<sClus_1[i]<<" "<<sClus_2[i]<<endl; 

	  for(Int_t j=0; j<nClus;j++)
	    {

	      EBDetId maxHit = max_hit[j];
	      if( (sClus_1[i] == j) || (sClus_2[i] == j) )
		{

		  for (int icry=0;icry< gammaCandEtaSize_*gammaCandPhiSize_;icry++)
		    {
              
		      int row = icry / gammaCandEtaSize_;
		      int column= icry % gammaCandEtaSize_;
		      int curr_eta=maxHit.ieta() + column - (gammaCandEtaSize_/2);
		      int curr_phi=maxHit.iphi() + row - (gammaCandPhiSize_/2);
              
		      if (curr_eta * maxHit.ieta() <= 0) {if (maxHit.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
		      if (curr_phi < 1) curr_phi += 360;
		      if (curr_phi > 360) curr_phi -= 360;
              
		      try
			{
			  EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);
			  std::vector<EBDetId>::const_iterator usedIds;
			  
			  bool HitAlreadyUsed=false;
			  for(usedIds=scXtals.begin(); usedIds!=scXtals.end(); usedIds++)
			    if(*usedIds==det)
			      {
				HitAlreadyUsed=true;
				break;
			      }
			  
			  if(!HitAlreadyUsed)
			    if (recHitsEB_map->find(det) != recHitsEB_map->end())
			      {
				std::map<DetId, EcalRecHit>::iterator aHit;
				aHit = recHitsEB_map->find(det);
				pi0EBRecHitCollection->push_back(aHit->second);
				scXtals.push_back(det);

				//EBDetId sel_rh = aHit->second.detid();
				//cout << "       RecHit Ok 20x20 matrix outside cluster : z,ieta,iphi "<<sel_rh.zside()<<" "<<sel_rh.ieta()<<" "<<sel_rh.iphi()<<endl;    
				//cout << "       RecHit Ok 20x20 matrix outside cluster : tower_ieta,tower_iphi "<<sel_rh.tower_ieta()<<" "<<sel_rh.tower_iphi()<<endl;   
				//cout << "       RecHit Ok 20x20 matrix outside cluster : iSM, ic "<<sel_rh.ism()<<" "<<sel_rh.ic()<<endl;
				
			      }
			}
		      catch (...)
			{
			}
		    }

		}
	
	    }
	  //	    intbc++;
	  
	}



      timerName = category + "::preparePi0RecHitsCollection";
      timers.pop_and_push(timerName);


      //Put selected information in the event
      //      if (npi0>0) iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);
      cout<< "   EB RecHits # in Collection: "<<pi0EBRecHitCollection->size()<<endl;
      iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);
  
      timerName = category + "::storePi0RecHitsCollection";
      timers.pop_and_push(timerName);

      timers.clear_stack();
      
      delete recHitsEB_map;
      delete topology_p;

}
