#include "RecoMET/METAlgorithms/interface/EcalHaloAlgo.h"
#include "DataFormats/Common/interface/ValueMap.h"

/*
  [class]:  EcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See EcalHaloAlgo.h
  [date]: October 15, 2009
*/

using namespace std;
using namespace reco;
using namespace edm;

bool CompareTime(const EcalRecHit* x, const EcalRecHit* y){ return x->time() < y->time(); }

EcalHaloAlgo::EcalHaloAlgo()
{
  for(unsigned int i = 0  ; i < 3 ; i++)
    {
      RoundnessCut = 0 ;
      AngleCut = 0 ;
    }
  EBRecHitEnergyThreshold = 0.;
  EERecHitEnergyThreshold = 0.;
  ESRecHitEnergyThreshold = 0.;
  SumEnergyThreshold = 0.;
  NHitsThreshold =0;
}

EcalHaloData EcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<reco::PhotonCollection>& ThePhotons, edm::Handle<reco::SuperClusterCollection>& TheSuperClusters, edm::Handle<EBRecHitCollection>& TheEBRecHits, edm::Handle<EERecHitCollection>& TheEERecHits, edm::Handle<ESRecHitCollection>& TheESRecHits)
{  
  EcalHaloData TheEcalHaloData;

  /*
  // Store energy sum of rechits as a function of iPhi (iphi goes from 1 to 72)
  float SumE[73];
  // Store number of rechits as a function of iPhi 
  int NumHits[73];
  // Store minimum time of rechit as a function of iPhi
  float MinTimeHits[73];
  // Store maximum time of rechit as a function of iPhi
  float MaxTimeHits[73];
  */

  // Store energy sum of rechits as a function of iPhi (iphi goes from 1 to 72)       
  float SumE[361];           
  // Store number of rechits as a function of iPhi
  int NumHits[361];               
  // Store minimum time of rechit as a function of iPhi
  float MinTimeHits[361];                             
  // Store maximum time of rechit as a function of iPhi 
  float MaxTimeHits[361];                      

  // initialize
  for(int i = 0 ; i < 361 ; i++ )
    {
      SumE[i] = 0.;
      NumHits[i] = 0 ;
      MinTimeHits[i] = 9999.;
      MaxTimeHits[i] = -9999.;
    }

  // Loop over EB RecHits
  for(EBRecHitCollection::const_iterator hit = TheEBRecHits->begin() ; hit != TheEBRecHits->end() ; hit++ )
    {
      // Arbitrary threshold to kill noise (needs to be optimized with data)
      if (hit->energy() < EBRecHitEnergyThreshold ) continue;
      
      // Get Det Id of the rechit
      DetId id = DetId(hit->id()); 
      const CaloSubdetectorGeometry* TheSubGeometry = 0;                                                                         
      const CaloCellGeometry* cell = 0 ;                                                                                    

      // Get EB geometry 
      TheSubGeometry = TheCaloGeometry.getSubdetectorGeometry(DetId::Ecal, 1);                                                         
      EBDetId EcalID(id.rawId());
      if( TheSubGeometry )
	cell = TheSubGeometry->getGeometry(id);
  
      if(cell)
	{
	  GlobalPoint globalpos = cell->getPosition();
	  //	  float r = TMath::Sqrt ( globalpos.y()*globalpos.y() + globalpos.x()*globalpos.x());
	  int iPhi = EcalID.iphi();

	  if( iPhi < 361 )	  // just to be safe
 	    {
	      //iPhi = (iPhi-1)/5 +1;  // convert ecal iphi to phiwedge iphi  (e.g. there are 5 crystal per phi wedge, as in calotowers )
	      SumE[iPhi] += hit->energy();
	      NumHits[iPhi] ++;

	      float time = hit->time();
	      MinTimeHits[iPhi] = time < MinTimeHits[iPhi] ? time : MinTimeHits[iPhi];
	      MaxTimeHits[iPhi] = time > MaxTimeHits[iPhi] ? time : MaxTimeHits[iPhi];
	    }
	}
    }
  
  //for( int iPhi = 1 ; iPhi < 73; iPhi++ )
  for( int iPhi = 1 ; iPhi < 361; iPhi++ )
    {
      if( SumE[iPhi] >= SumEnergyThreshold || NumHits[iPhi] > NHitsThreshold )
	{
	  // Build PhiWedge and store to EcalHaloData if energy or #hits pass thresholds
	  PhiWedge wedge(SumE[iPhi], iPhi, NumHits[iPhi], MinTimeHits[iPhi], MaxTimeHits[iPhi]);
	  
	  // Loop over rechits again to calculate direction based on timing info
	  
	  // Loop over EB RecHits
	  std::vector<const EcalRecHit*>  Hits;
	  for(EBRecHitCollection::const_iterator hit = TheEBRecHits->begin() ; hit != TheEBRecHits->end() ; hit++ )
	    {
	      if (hit->energy() < EBRecHitEnergyThreshold ) continue;
	      
	      // Get Det Id of the rechit
	      DetId id = DetId(hit->id()); 
	      EBDetId EcalID(id.rawId());
	      int Hit_iPhi = EcalID.iphi();
	      //Hit_iPhi = (Hit_iPhi-1)/5 +1; // convert ecal iphi to phiwedge iphi
	      if( Hit_iPhi != iPhi ) continue;
	      Hits.push_back( &(*hit) );
	      
	    }
	  std::sort( Hits.begin() , Hits.end(), CompareTime);
	  float MinusToPlus = 0.;
	  float PlusToMinus = 0.;
	  for( unsigned int i = 0 ; i < Hits.size() ; i++ )
	    {
	      DetId id_i = DetId(Hits[i]->id()); 
	      EBDetId EcalID_i(id_i.rawId());
	      int ieta_i = EcalID_i.ieta();
	      for( unsigned int j = (i+1) ; j < Hits.size() ; j++ )
		{
		  DetId id_j = DetId(Hits[j]->id() );
		  EBDetId EcalID_j(id_j.rawId());
		  int ieta_j = EcalID_j.ieta();
		  if( ieta_i > ieta_j ) PlusToMinus += TMath::Abs(ieta_j - ieta_i );
		  else MinusToPlus += TMath::Abs(ieta_j -ieta_i) ;
		}
	    }
	  
	  float PlusZOriginConfidence = (PlusToMinus+MinusToPlus) ? PlusToMinus / (PlusToMinus+MinusToPlus) : -1.;
	  wedge.SetPlusZOriginConfidence(PlusZOriginConfidence);
	  TheEcalHaloData.GetPhiWedges().push_back(wedge);
	}
    }

  std::vector<float> vShowerShapes_Roundness;
  std::vector<float> vShowerShapes_Angle ;
  for(reco::SuperClusterCollection::const_iterator cluster = TheSuperClusters->begin() ; cluster != TheSuperClusters->end() ; cluster++ )
    {
      /* R. Remington :  Commenting out until we debug the showerRoundness() function for 34X. 
 
	 if( abs(cluster->eta()) <= 1.47 )
	{ 
	  vector<float> shapes = EcalClusterTools::showerRoundness( *cluster, &(*TheEBRecHits.product()) );
	  float roundness = shapes[0];
	  float angle = shapes[1];
	  
	  // Check if supercluster passes the cuts on roundness and angle, if so store the reference to it
	  if( (roundness >=0 && roundness < GetRoundnessCut()) &&  angle >= 0 && angle < GetAngleCut() )
	    {
	      edm::Ref<SuperClusterCollection> TheClusterRef( TheSuperClusters, cluster - TheSuperClusters->begin() ) ;
	      TheEcalHaloData.GetSuperClusters().push_back( TheClusterRef ) ; 
	    }
	  
	  vShowerShapes_Roundness.push_back(shapes[0]);
	  vShowerShapes_Angle.push_back(shapes[1]);
	}
      else
	{ 
	  vShowerShapes_Roundness.push_back(-1.);
	  vShowerShapes_Angle.push_back(-1.);
	}
      */
      
      
      vShowerShapes_Roundness.push_back(-1.);         
      vShowerShapes_Angle.push_back(-1.);      
    }
  
  edm::ValueMap<float>::Filler TheRoundnessFiller( TheEcalHaloData.GetShowerShapesRoundness() );
  TheRoundnessFiller.insert( TheSuperClusters, vShowerShapes_Roundness.begin(), vShowerShapes_Roundness.end() );
  TheRoundnessFiller.fill();  

  edm::ValueMap<float>::Filler TheAngleFiller( TheEcalHaloData.GetShowerShapesAngle() );
  TheAngleFiller.insert( TheSuperClusters, vShowerShapes_Angle.begin() , vShowerShapes_Angle.end() );
  TheAngleFiller.fill();

  return TheEcalHaloData;
}



