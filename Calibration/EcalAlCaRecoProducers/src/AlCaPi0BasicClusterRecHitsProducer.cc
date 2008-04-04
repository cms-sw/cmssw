#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPi0BasicClusterRecHitsProducer.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlCaPi0BasicClusterRecHitsProducer::AlCaPi0BasicClusterRecHitsProducer(const edm::ParameterSet& iConfig)
{
  ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");
  barrelHits_ = iConfig.getParameter< std::string > ("barrelHitCollection");
  pi0BarrelHits_ = iConfig.getParameter< std::string > ("pi0BarrelHitCollection");
  islandBCProd_ = iConfig.getParameter< std::string > ("islandBCProd");
  islandBCColl_ = iConfig.getParameter< std::string > ("islandBCColl");

  gammaCandEtaSize_ = iConfig.getParameter<int> ("gammaCandEtaSize");
  gammaCandPhiSize_ = iConfig.getParameter<int> ("gammaCandPhiSize");
  if ( gammaCandPhiSize_ % 2 == 0 ||  gammaCandEtaSize_ % 2 == 0)
    edm::LogError("AlCaPi0BasicClusterRecHitsProducerError") << "Size of eta/phi should be odd numbers";

  selePtGammaOne_ = iConfig.getParameter<double> ("selePtGammaOne");  
  selePtGammaTwo_ = iConfig.getParameter<double> ("selePtGammaTwo");  
  selePtPi0_ = iConfig.getParameter<double> ("selePtPi0");  
  seleMinvMaxPi0_ = iConfig.getParameter<double> ("seleMinvMaxPi0");  
  seleMinvMinPi0_ = iConfig.getParameter<double> ("seleMinvMinPi0");  


  //register your products
  produces< EBRecHitCollection >(pi0BarrelHits_);
}


AlCaPi0BasicClusterRecHitsProducer::~AlCaPi0BasicClusterRecHitsProducer()
{
 
  TimingReport::current()->dump(std::cout);

}


// ------------ method called to produce the data  ------------
void
AlCaPi0BasicClusterRecHitsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  // Timer
  const std::string category = "AlCaPi0BasicClusterRecHitsProducer";
  TimerStack timers;
  string timerName = category + "::Total";
  timers.push(timerName);



  Handle<EBRecHitCollection> barrelRecHitsHandle;

  iEvent.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product!" << std::endl;
  }

  recHitsEB_map = new std::map<DetId, EcalRecHit>();


  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > pi0EBRecHitCollection( new EBRecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  cout<< "   EB RecHits #: "<<barrelRecHitsHandle->size()<<endl;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {

    std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
    recHitsEB_map->insert(map_entry);

  }

  timerName = category + "::readEBRecHitsCollection";
  timers.push(timerName);



  // Get ECAL Barrel Island Basic Clusters collection
  // ECAL Barrel Island Basic Clusters 
     static const int MAXBCEB = 200;
     int nIslandBCEB;
     float eIslandBCEB[MAXBCEB];
     float etIslandBCEB[MAXBCEB];
     float etaIslandBCEB[MAXBCEB];
     float phiIslandBCEB[MAXBCEB];

  nIslandBCEB=0;
  for(int i=0; i<MAXBCEB; i++){
    eIslandBCEB[i] = 0;
    etIslandBCEB[i] = 0;
    etaIslandBCEB[i] = 0;
    phiIslandBCEB[i] = 0;
  }
  //pi0 candidates
  static const int MAXPI0 = 20;
  int npi0=0;
  int BC1[MAXPI0],BC2[MAXPI0];
  for(int i=0; i<MAXPI0; i++){
    BC1[i]=0;
    BC2[i]=0;
  }

  Handle<reco::BasicClusterCollection> pIslandBarrelBasicClusters;
  iEvent.getByLabel(islandBCProd_, islandBCColl_, pIslandBarrelBasicClusters);
  const reco::BasicClusterCollection* islandBarrelBasicClusters = pIslandBarrelBasicClusters.product();
  cout << " island Basic Clusters # "<< islandBarrelBasicClusters->size() <<endl;

  for(reco::BasicClusterCollection::const_iterator aClus = islandBarrelBasicClusters->begin();
      aClus != islandBarrelBasicClusters->end(); aClus++) {

    //cout << " island Basic Cluster E, #xtals: "<<  aClus->energy()<<" "<< aClus->getHitsByDetId().size() <<endl;
    //cout << " island Basic Cluster Position (x,y,z): "<<aClus->position().X()<<" "<<aClus->position().Y()<<" "<<aClus->position().Z()<<" "<<endl;
    //cout << " island Basic Cluster Position (eta,phi): "<<aClus->position().eta()<<" "<<aClus->position().phi()<<endl;

    float theta = 2. * atan(exp(-aClus->position().eta()));
    float p0x = aClus->energy() * sin(theta) * cos(aClus->position().phi());
    float p0y = aClus->energy() * sin(theta) * sin(aClus->position().phi());
    float p0z = aClus->energy() * cos(theta);
    float et = sqrt( p0x*p0x + p0y*p0y);

    //cout << " island Basic Cluster E,Et,px,py,pz: "<<aClus->energy()<<" "<<et<<" "<<p0x<<" "<<p0y<<" "<<p0z<<endl; 

    eIslandBCEB[nIslandBCEB] = aClus->energy();
    etIslandBCEB[nIslandBCEB] = et;
    etaIslandBCEB[nIslandBCEB] = aClus->position().eta();
    phiIslandBCEB[nIslandBCEB] = aClus->position().phi();
     
    nIslandBCEB++;
  }

  timerName = category + "::readIslandBasicClustersCollection";
  timers.push(timerName);



  // Selection, based on ECAL Barrel Basic Clusters

  if (nIslandBCEB > 1) 
    {
      for(Int_t i=0 ; i<nIslandBCEB ; i++)
        {
          for(Int_t j=i+1 ; j<nIslandBCEB ; j++)
            {

              if( etIslandBCEB[i]>selePtGammaOne_ && etIslandBCEB[j]>selePtGammaTwo_) 
		{
                
		  float theta_0 = 2. * atan(exp(-etaIslandBCEB[i]));
		  float theta_1 = 2. * atan(exp(-etaIslandBCEB[j]));
                
		  float p0x = eIslandBCEB[i] * sin(theta_0) * cos(phiIslandBCEB[i]);
		  float p1x = eIslandBCEB[j] * sin(theta_1) * cos(phiIslandBCEB[j]);
                
		  float p0y = eIslandBCEB[i] * sin(theta_0) * sin(phiIslandBCEB[i]);
		  float p1y = eIslandBCEB[j] * sin(theta_1) * sin(phiIslandBCEB[j]);
		  float p0z = eIslandBCEB[i] * cos(theta_0);
		  float p1z = eIslandBCEB[j] * cos(theta_1);
                
		  //
                
		  float pt_pi0 = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
		  //float dr_pi0 = sqrt ( (etaIslandBCEB[i]-etaIslandBCEB[j])*(etaIslandBCEB[i]-etaIslandBCEB[j]) + (phiIslandBCEB[i]-phiIslandBCEB[j])*(phiIslandBCEB[i]-phiIslandBCEB[j]) );
		  if (pt_pi0 > selePtPi0_) 
		    {
		      float m_inv = sqrt ( (eIslandBCEB[i] + eIslandBCEB[j])*(eIslandBCEB[i] + eIslandBCEB[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
		      if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) )
			{
			  cout <<" pi0 Candidate (pt>2.5 GeV, m_inv<0.2) pt,m_inv,i,j :   "<<pt_pi0<<" "<<m_inv<<" "<<i<<" "<<j<<" "<<endl;  
			  BC1[npi0]=i;
			  BC2[npi0]=j;
			  npi0++;
			}

		    }
                
		}
            } // End of the "j" loop over BCEB
        } // End of the "i" loop over BCEB


      cout<<" "<<endl;
      cout<<"  Pi0 candidates #: "<<npi0<<endl;

      timerName = category + "::makePi0Cand";
      timers.pop_and_push(timerName);



      vector<EBDetId> scXtals;
      scXtals.clear();
      for(Int_t i=0 ; i<npi0 ; i++)
	{
	  // cout<<"     Pi0 i,Bc1, Bc2 "<<i<<" "<<BC1[i]<<" "<<BC2[i]<<endl; 

	  int intbc=0;

	  for(reco::BasicClusterCollection::const_iterator aClus = islandBarrelBasicClusters->begin();
	      aClus != islandBarrelBasicClusters->end(); aClus++) {

	    //   cout << " intbc, BC1[i], BC2[i] "<<intbc<<" "<<BC1[i]<<" "<<BC2[i]<<endl; 
	    if((intbc==BC1[i]) || (intbc==BC2[i]))
	      {

		std::vector<DetId> hits = aClus->getHitsByDetId();
		//std::vector<DetId>::iterator hit;
		std::map<DetId, EcalRecHit>::iterator aHit;

		// New addition - all existed xtals inside a sliding window will be added 

		double currEnergy = 0.;
		EBDetId maxHit(0);

		for( std::vector<DetId>::const_iterator idsIt = hits.begin(); idsIt != hits.end(); ++idsIt) {
          
		  if((*idsIt).subdetId()!=EcalBarrel || (*idsIt).det()!= DetId::Ecal) continue;
          
		  if(((recHitsEB_map->find(*idsIt))->second).energy() > currEnergy) {
		    currEnergy=((recHitsEB_map->find(*idsIt))->second).energy();
		    maxHit=*idsIt;
		  }
		  aHit = recHitsEB_map->find(*idsIt);
		  pi0EBRecHitCollection->push_back(aHit->second);
		  scXtals.push_back(*idsIt);

		  //EBDetId sel_rh_bc = aHit->second.detid();
		  // cout << "       RecHit Ok, belongs to cluster # "<< intbc<<" : z,ieta,iphi "<<sel_rh_bc.zside()<<" "<<sel_rh_bc.ieta()<<" "<<sel_rh_bc.iphi()<<endl;    
		  //cout << "       RecHit Ok, belongs to cluster # "<< intbc<<" : tower_ieta,tower_iphi "<<sel_rh_bc.tower_ieta()<<" "<<sel_rh_bc.tower_iphi()<<endl;   
		  //cout << "       RecHit Ok, belongs to  cluster # "<<intbc <<" : iSM, ic "<<sel_rh_bc.ism()<<" "<<sel_rh_bc.ic()<<endl;

		}

		if (!maxHit.null())
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
				aHit = recHitsEB_map->find(det);
				pi0EBRecHitCollection->push_back(aHit->second);
				scXtals.push_back(det);

				//EBDetId sel_rh = aHit->second.detid();
				//cout << "       RecHit Ok 20x20 matrix outside cluster # "<<intbc<<" : z,ieta,iphi "<<sel_rh.zside()<<" "<<sel_rh.ieta()<<" "<<sel_rh.iphi()<<endl;    
				//cout << "       RecHit Ok 20x20 matrix outside cluster # "<<intbc<<" : tower_ieta,tower_iphi "<<sel_rh.tower_ieta()<<" "<<sel_rh.tower_iphi()<<endl;   
				//cout << "       RecHit Ok 20x20 matrix outside cluster # "<<intbc<<" : iSM, ic "<<sel_rh.ism()<<" "<<sel_rh.ic()<<endl;

			      }
			}
		      catch (...)
			{
			}
		    }

	      }
	    intbc++;
		
	  }
	}
    }

  timerName = category + "::preparePi0RecHitsCollection";
  timers.pop_and_push(timerName);


      //Put selected information in the event
      //if (npi0>0) iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);

  cout<< "   EB RecHits # in Collection: "<<pi0EBRecHitCollection->size()<<endl;
  iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);

  timerName = category + "::storePi0RecHitsCollection";
  timers.pop_and_push(timerName);
  
  timers.clear_stack();
  delete recHitsEB_map;

}
