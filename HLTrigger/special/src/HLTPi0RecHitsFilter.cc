#include "HLTrigger/special/interface/HLTPi0RecHitsFilter.h"
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

HLTPi0RecHitsFilter::HLTPi0RecHitsFilter(const edm::ParameterSet& iConfig)
{
  //  ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");
  //  barrelHits_ = iConfig.getParameter< std::string > ("barrelHitCollection");
 // replace the 2 strings with 1 InputTag of form label:instance 
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
  seleS4S9GammaOne_ = iConfig.getParameter<double> ("seleS4S9GammaOne");  
  seleS4S9GammaTwo_ = iConfig.getParameter<double> ("seleS4S9GammaTwo");  
  selePi0Iso_ = iConfig.getParameter<double> ("selePi0Iso");  
  selePi0BeltDR_ = iConfig.getParameter<double> ("selePi0BeltDR");  
  selePi0BeltDeta_ = iConfig.getParameter<double> ("selePi0BeltDeta");  
  seleMinvMaxPi0_ = iConfig.getParameter<double> ("seleMinvMaxPi0");  
  seleMinvMinPi0_ = iConfig.getParameter<double> ("seleMinvMinPi0");  
  seleXtalMinEnergy_ = iConfig.getParameter<double> ("seleXtalMinEnergy");
  seleNRHMax_ = iConfig.getParameter<int> ("seleNRHMax");

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

  // Timer
  //  const std::string category = "AlCaPi0RecHitsProducer";
  //TimerStack timers;
  //string timerName = category + "::Total";
  //timers.push(timerName);

  bool accept=false;

  Handle<EBRecHitCollection> barrelRecHitsHandle;

  //  iEvent.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  // replace the 2 strings with 1 InputTag of form label:instance 	 
  iEvent.getByLabel(barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product!" << std::endl;
  }

  recHitsEB_map = new std::map<DetId, EcalRecHit>();

  std::vector<EcalRecHit> seeds;
  seeds.clear();

  vector<EBDetId> usedXtals;
  usedXtals.clear();


  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > pi0EBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EBRecHitCollection > pi0EBDummyRecHitCollection( new EBRecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  //cout<< " Unpacked EB RecHits #: "<<barrelRecHitsHandle->size()<<endl;

    int ipositive = 0;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {

    double energy = itb->energy();

    if (energy > seleXtalMinEnergy_) 
     {
	std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
	recHitsEB_map->insert(map_entry);
	
	if (energy > clusSeedThr_) seeds.push_back(*itb);

	ipositive++;
     }
  }


  //cout<< " Unpacked positive EB RecHits #: "<<ipositive<<endl;
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
  topology_p = new EcalBarrelTopology(geoHandle);
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",1));
  providedParameters.insert(std::make_pair("X0",0.89));
  providedParameters.insert(std::make_pair("T0_barl",5.7));
  providedParameters.insert(std::make_pair("W0",4.2));

  PositionCalc posCalculator_ = PositionCalc(providedParameters);
  //PositionCalc::Initialize(providedParameters, &recHitsEB_map, &(*geometry_p));

  static const int MAXCLUS = 2000;
  int nClus;
  float eClus[MAXCLUS];
  float etClus[MAXCLUS];
  float etaClus[MAXCLUS];
  float phiClus[MAXCLUS];
  float s4s9Clus[MAXCLUS];
  EBDetId max_hit[MAXCLUS];  
  int goodIsoClus[MAXCLUS];

  vector<vector<DetId> > detids_in_clusters;

  nClus=0;
  for(int i=0; i<MAXCLUS; i++){
    eClus[i] = 0;
    etClus[i] = 0;
    etaClus[i] = 0;
    phiClus[i] = 0;
    goodIsoClus[i] = 0;
    s4s9Clus[i] = 0;
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
        double extals3x3[3][3];


        for(int i=0; i<3; i++){
          for(int j=0; j<3; j++){
            extals3x3[i][j] = 0;
          }
        }
    for (int icry=0;icry< clusEtaSize_*clusPhiSize_;icry++)
      {
	
	int row = icry / clusEtaSize_;
	int column= icry % clusEtaSize_;
	int curr_eta=seed_id.ieta() + column - (clusEtaSize_/2);
	int curr_phi=seed_id.iphi() + row - (clusPhiSize_/2);
	
	if (curr_eta * seed_id.ieta() <= 0) {if (seed_id.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
	if (curr_phi < 1) curr_phi += 360;
	if (curr_phi > 360) curr_phi -= 360;
	
	//try
	//  {
	//    EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);

        EBDetId det;

	if (!EBDetId::validDetId(curr_eta,curr_phi) ) 
	  {
	    continue;
	  } else {
	    det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);
	  }


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
		  extals3x3[row][column] = aHit->second.energy();
		  
		  EBDetId sel_rh = aHit->second.detid();
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : E,z,ieta,iphi "<<aHit->second.energy()<<" "<<sel_rh.zside()<<" "<<sel_rh.ieta()<<" "<<sel_rh.iphi()<<endl;    
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : tower_ieta,tower_iphi "<<sel_rh.tower_ieta()<<" "<<sel_rh.tower_iphi()<<endl;   
		  //cout << "       Simple Clustering: RecHit Ok 3x3 matrix inside cluster : iSM, ic "<<sel_rh.ism()<<" "<<sel_rh.ic()<<endl;
		  
		}
	    //	  }
	    //	catch (...)
	    //  {
	    // }
      }
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_v,hitCollection_p,geometry_p,geometryES_p);
    //cout<< "       Simple Clustering: Total energy for this simple cluster : "<<simple_energy<<endl; 
    //cout<< "       Simple Clustering: eta phi : "<<clus_pos.eta()<<" "<<clus_pos.phi()<<endl; 
    //cout<< "       Simple Clustering: x y z : "<<clus_pos.x()<<" "<<clus_pos.y()<<" "<<clus_pos.z()<<endl; 

    detids_in_clusters.push_back(clus_v);

	    float theta_s = 2. * atan(exp(-clus_pos.eta()));
	    float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
	    float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
	    float p0z_s = simple_energy * cos(theta_s);
	    float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);

	    //cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<p0z_s<<endl;
    
            float s4s9_1 = 0.;
            float s4s9_max = -999.;
            s4s9_1 = extals3x3[0][0] + extals3x3[0][1] + extals3x3[1][0] + extals3x3[1][1];
            if(s4s9_1 >= s4s9_max) s4s9_max = s4s9_1; 
            //cout<< " s4s9: 1 cand: s4s9_max s4s9_1 "<<s4s9_max<<" "<<s4s9_1<<endl;
            s4s9_1 = extals3x3[0][1] + extals3x3[0][2] + extals3x3[1][1] + extals3x3[1][2];
            if(s4s9_1 >= s4s9_max) s4s9_max = s4s9_1; 
            //cout<< " s4s9: 2 cand: s4s9_max s4s9_1 "<<s4s9_max<<" "<<s4s9_1<<endl;
            s4s9_1 = extals3x3[1][0] + extals3x3[1][1] + extals3x3[2][0] + extals3x3[2][1];
            if(s4s9_1 >= s4s9_max) s4s9_max = s4s9_1; 
            //cout<< " s4s9: 3 cand: s4s9_max s4s9_1 "<<s4s9_max<<" "<<s4s9_1<<endl;
            s4s9_1 = extals3x3[1][1] + extals3x3[1][2] + extals3x3[2][1] + extals3x3[2][2];
            if(s4s9_1 >= s4s9_max) s4s9_max = s4s9_1; 
            //      cout<< " s4s9: 4 cand: s4s9_max s4s9_1 "<<s4s9_max<<" "<<s4s9_1<<endl;

	    //cout<< "       Simple Clustering: s4s9 : "<<(s4s9_max/simple_energy)<<endl;
	    eClus[nClus] = simple_energy;
	    etClus[nClus] = et_s;
	    etaClus[nClus] = clus_pos.eta();
	    phiClus[nClus] = clus_pos.phi();
	    max_hit[nClus] = seed_id;
            s4s9Clus[nClus] = s4s9_max/simple_energy;
	    
	    nClus++;
	    if (nClus == MAXCLUS) return accept;
      }
  }
  
  //timerName = category + "::makeSimpleClusters";
  //timers.pop_and_push(timerName);


  /*
  int iclusters = detids_in_clusters.size();
  cout<< "  detids_in_clusters size: "<<iclusters<<endl;
  for(int i=0 ; i<iclusters ; i++)
    {
      int iclusize = detids_in_clusters[i].size();
      cout<< "   detids_in_clusters: cluster #, cluster size "<<i<<" "<<iclusize<<endl;
      for (int j = 0 ; j<iclusize ; j++)
	{
	  std::map<DetId, EcalRecHit>::iterator aHit;
	  aHit = recHitsEB_map->find(detids_in_clusters[i][j]);
	  cout<< "    detids_in_clusters: cluster #, rechit E "<<aHit->second.energy()<<endl;

	}

    }
    */

  // Selection, based on Simple clustering
  //pi0 candidates
  static const int MAXPI0S = 200;
  int npi0_s=0;

  if (nClus > 1) 
    {
      for(Int_t i=0 ; i<nClus ; i++)
        {
          for(Int_t j=i+1 ; j<nClus ; j++)
            {

              if( etClus[i]>selePtGammaOne_ && etClus[j]>selePtGammaTwo_ && s4s9Clus[i]>seleS4S9GammaOne_ && s4s9Clus[j]>seleS4S9GammaTwo_) 
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

		  if (pt_pi0 > selePtPi0_) 
		    {
		      float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
		      if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) )
			{

			  math::XYZPoint pi0vect = math::XYZPoint((p0x+p1x), (p0y+p1y), (p0z+p1z));

			  int icliso = 0;
			  vector<int> IsoClus;
			  IsoClus.clear();

                          float Iso = 0;
                          for(Int_t k=0 ; k<nClus ; k++)
                            {
			      float dretaclpi0 = fabs(etaClus[k] - pi0vect.eta());
                             
                              float drclpi0 = sqrt( (etaClus[k] - pi0vect.eta())*(etaClus[k] - pi0vect.eta()) + (phiClus[k] - pi0vect.phi())*(phiClus[k] - pi0vect.phi()));
			      //cout<< "   Iso: k, E, drclpi0, detaclpi0, dphiclpi0 "<<k<<" "<<eClus[k]<<" "<<drclpi0<<" "<<fabs(etaClus[k] - pi0vect.eta())<<" "<<fabs(phiClus[k] - pi0vect.phi())<<endl;
                              if( (i!=k) && (j!=k) && (drclpi0<selePi0BeltDR_) && (dretaclpi0<selePi0BeltDeta_) )
				{
				  // cout<< "   ... good iso cluster #: "<<k<<endl;
				  Iso = Iso + etClus[k];
				  IsoClus.push_back(k);
				}
                            }

                          //cout << "  After Iso: Iso, pt_pi0, Isovar "<< Iso<< " "<<pt_pi0<<" "<<(Iso/pt_pi0)<<endl;

			  /*
			  int isize = IsoClus.size();
			  for(Int_t iii=0 ; iii<isize ; iii++){
			    cout<< "    Iso cluster: # "<<iii<<" "<<IsoClus[iii]<<endl; 
			  }
			  */
                          float Isovar = Iso/pt_pi0;

                          if(Isovar<selePi0Iso_)
                            {

			      //cout <<"  Simple Clustering: pi0 Candidate (pt>2.5 GeV, m_inv<0.2) pt,m_inv,s4s9_i, s4s9_j, Iso, i,j :   "<<pt_pi0<<" "<<m_inv<<" "<<s4s9Clus[i]<<" "<<s4s9Clus[j]<<" "<<Isovar<<" "<<i<<" "<<j<<" "<<endl;  

			       int isize = IsoClus.size();
			       
			       for(Int_t iii=0 ; iii<isize ; iii++){
				 //cout<< "    Iso cluster: # "<<iii<<" "<<IsoClus[iii]<<endl; 
				 goodIsoClus[IsoClus[iii]] = 1;
			       }

			       goodIsoClus[i] = 1;
			       goodIsoClus[j] = 1;

			      npi0_s++;
			      if(npi0_s == MAXPI0S) return accept;
			    }
			}

		    }
                
		}
            } // End of the "j" loop over Simple Clusters
        } // End of the "i" loop over Simple Clusters

    }

  //timerName = category + "::makePi0Cand";
  //timers.pop_and_push(timerName);


  //cout<<" "<<endl;
  //cout<<"  (Simple Clustering) Pi0 candidates #: "<<npi0_s<<endl;

     for(Int_t jj=0;jj<nClus;jj++)
       {
	 if (goodIsoClus[jj] == 1) 
	   {
	     int iclusize = detids_in_clusters[jj].size();
	     //cout<< "   detids_in_clusters: cluster #, cluster size "<<jj<<" "<<iclusize<<endl;
	     for (int j = 0 ; j<iclusize ; j++)
	       {
		 
		 std::map<DetId, EcalRecHit>::iterator aHit;
		 aHit = recHitsEB_map->find(detids_in_clusters[jj][j]);
		 //cout<< "    detids_in_clusters: cluster #, rechit E "<<aHit->second.energy()<<endl;
		 
		 pi0EBRecHitCollection->push_back(aHit->second);
		 
	       }
	   }
       }



      //timerName = category + "::preparePi0RecHitsCollection";
      //timers.pop_and_push(timerName);


      //Put selected information in the event
      int pi0_collsize = pi0EBRecHitCollection->size();
      if ( pi0_collsize > seleNRHMax_ )
	{
	  //cout<< "   Max RH limit exceeded: "<<pi0EBRecHitCollection->size()<<" Max RH: "<<seleNRHMax_<<endl;
	  iEvent.put( pi0EBDummyRecHitCollection, pi0BarrelHits_);
	  //cout<< "   EB RecHits # in Collection: "<<0<<endl;
	} else {

	  //cout<< "   EB RecHits # in Collection: "<<pi0EBRecHitCollection->size()<<endl;
	  if (pi0_collsize > 0 ) {
	    accept = true;
	    iEvent.put( pi0EBRecHitCollection, pi0BarrelHits_);
	  }
	}
  
      //timerName = category + "::storePi0RecHitsCollection";
      //timers.pop_and_push(timerName);

      //timers.clear_stack();
      
      delete recHitsEB_map;
      delete topology_p;

      return accept;

}
