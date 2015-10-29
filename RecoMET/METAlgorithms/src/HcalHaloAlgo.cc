#include "RecoMET/METAlgorithms/interface/HcalHaloAlgo.h"

/*
  [class]:  HcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See HcalHaloAlgo.h
  [date]: October 15, 2009
*/

using namespace std;
using namespace edm;
using namespace reco;

#include <iomanip>
bool CompareTime(const HBHERecHit* x, const HBHERecHit* y ){ return x->time() < y->time() ;}
bool CompareTowers(const CaloTower* x, const CaloTower* y ){ 
  return x->iphi()*1000 + x->ieta() < y->iphi()*1000 + y->ieta();
}

HcalHaloAlgo::HcalHaloAlgo()
{
  HBRecHitEnergyThreshold = 0.;
  HERecHitEnergyThreshold = 0.;
  SumEnergyThreshold = 0.;
  NHitsThreshold = 0;
}

HcalHaloData HcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits) {
    edm::Handle<CaloTowerCollection> TheCaloTowers;
    return Calculate(TheCaloGeometry, TheHBHERecHits, TheCaloTowers);
}

HcalHaloData HcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits, edm::Handle<CaloTowerCollection>& TheCaloTowers)
{
  HcalHaloData TheHcalHaloData;
  
  // Store Energy sum of rechits as a function of iPhi (iPhi goes from 1 to 72)
  float SumE[73];
  // Store Number of rechits as a function of iPhi 
  int NumHits[73];
  // Store minimum time of rechit as a function of iPhi
  float MinTimeHits[73];
  // Store maximum time of rechit as a function of iPhi
  float MaxTimeHits[73];
  for(unsigned int i = 0 ; i < 73 ; i++ ) 
    {
      SumE[i] = 0;
      NumHits[i]= 0;
      MinTimeHits[i] = 0.;
      MaxTimeHits[i] = 0.;
    }
  
  for( HBHERecHitCollection::const_iterator hit = TheHBHERecHits->begin() ; hit != TheHBHERecHits->end() ; hit++ )
    {
      HcalDetId id = HcalDetId(hit->id());                                                                                                    
      switch ( id.subdet() )                                                                                         
	{      
	case HcalBarrel:                                                                           
	  if(hit->energy() < HBRecHitEnergyThreshold )continue;
	  break;                                                                                                                  
	case HcalEndcap:                                                                                          
	  if(hit->energy() < HERecHitEnergyThreshold ) continue;
	  break;
	default:
	  continue;
	}
      
      int iEta = id.ieta();
      int iPhi = id.iphi();
      if(iPhi < 73 && TMath::Abs(iEta) < 23 )
	{ 
	  SumE[iPhi]+= hit->energy();
	  NumHits[iPhi] ++;
	  
	  float time = hit->time();
	  MinTimeHits[iPhi] = time < MinTimeHits[iPhi] ? time : MinTimeHits[iPhi];
	  MaxTimeHits[iPhi] = time > MaxTimeHits[iPhi] ? time : MaxTimeHits[iPhi];
	}
    }
  
  for( int iPhi = 1 ; iPhi < 73 ; iPhi++ )
    {
      if( SumE[iPhi] >= SumEnergyThreshold && NumHits[iPhi] > NHitsThreshold )
	{
	  // Build PhiWedge and store to HcalHaloData if energy or #hits pass thresholds
	  PhiWedge wedge(SumE[iPhi], iPhi, NumHits[iPhi], MinTimeHits[iPhi], MaxTimeHits[iPhi]);
	  
	  // Loop over rechits again to calculate direction based on timing info
	  std::vector<const HBHERecHit*> Hits;
	  for( HBHERecHitCollection::const_iterator hit = TheHBHERecHits->begin() ; hit != TheHBHERecHits->end() ; hit++ )
	    {

	      HcalDetId id = HcalDetId(hit->id());
	      if( id.iphi() != iPhi ) continue;
	      if( TMath::Abs(id.ieta() ) > 22 ) continue;  // has to overlap geometrically w/ HB
	      switch ( id.subdet() )
		{
		case HcalBarrel:
		  if(hit->energy() < HBRecHitEnergyThreshold )continue;
		  break;
		case HcalEndcap:
		  if(hit->energy() < HERecHitEnergyThreshold ) continue;
		  break;
		default:
		  continue;
		}
	      Hits.push_back(&(*hit));
	    }
	      
	  std::sort( Hits.begin() , Hits.end() , CompareTime);
	  float MinusToPlus = 0.;
	  float PlusToMinus = 0.;
	  for( unsigned int i = 0 ; i < Hits.size() ; i++ )
	    {
	      HcalDetId id_i = HcalDetId(Hits[i]->id() );
	      int ieta_i = id_i.ieta();
	      for( unsigned int j = (i+1) ; j < Hits.size() ; j++ )
		{
		  HcalDetId id_j = HcalDetId(Hits[j]->id() );
		  int ieta_j = id_j.ieta();
		  if( ieta_i > ieta_j ) PlusToMinus += TMath::Abs(ieta_i - ieta_j ) ;
		  else MinusToPlus += TMath::Abs(ieta_i - ieta_j);
		}
	    }
	  float PlusZOriginConfidence = (PlusToMinus + MinusToPlus )? PlusToMinus / ( PlusToMinus + MinusToPlus ) : -1. ;
	  wedge.SetPlusZOriginConfidence( PlusZOriginConfidence );
	  TheHcalHaloData.GetPhiWedges().push_back( wedge );
	}
    }

  // Don't use HF.
  int maxAbsIEta = 29;
  std::vector<const CaloTower*> sortedCaloTowers;
  for(CaloTowerCollection::const_iterator tower = TheCaloTowers->begin(); tower != TheCaloTowers->end(); tower++) {
    if(abs(tower->ieta()) <= maxAbsIEta && tower->numProblematicHcalCells() > 0)
      sortedCaloTowers.push_back(&(*tower));
  }

  // Sort towers such that lowest iphi and ieta are first, highest last, and towers
  // with same iphi value are consecutive. Then we can do everything else in one loop.
  std::sort(sortedCaloTowers.begin(), sortedCaloTowers.end(), CompareTowers);

  HaloTowerStrip strip;

  int prevIEta = -99, prevIPhi = -99;
  float prevHadEt = 0.;
  std::pair<uint8_t, CaloTowerDetId> prevPair, towerPair;
  bool wasContiguous = true;
  // Loop through and store a vector of pairs (problematicCells, DetId) for each contiguous strip we find
  for(unsigned int i = 0; i < sortedCaloTowers.size(); i++) {
    const CaloTower* tower = sortedCaloTowers[i];

    towerPair = std::make_pair((uint8_t)tower->numProblematicHcalCells(), tower->id());

    bool newIPhi = tower->iphi() != prevIPhi;
    bool isContiguous = tower->ieta() == 1 ? tower->ieta() - 2 == prevIEta : tower->ieta() - 1 == prevIEta;

    isContiguous = isContiguous || (tower->ieta() == -maxAbsIEta);
    if(newIPhi) isContiguous = false;

    if(!wasContiguous && isContiguous) {
      strip.cellTowerIds.push_back(prevPair);
      strip.cellTowerIds.push_back(towerPair);
      strip.hadEt += prevHadEt + tower->hadEt();
    }

    if(wasContiguous && isContiguous) {
      strip.cellTowerIds.push_back(towerPair);
      strip.hadEt += tower->hadEt();
    }

    if((wasContiguous && !isContiguous) || i == sortedCaloTowers.size()-1) { //ended the strip, so flush it
      if(strip.cellTowerIds.size() > 2) {
        TheHcalHaloData.getProblematicStrips().push_back( strip );
      }
      strip = HaloTowerStrip();
    }

    wasContiguous = isContiguous;
    prevPair = towerPair;
    prevIPhi = tower->iphi();
    prevIEta = tower->ieta();
    prevHadEt = tower->hadEt();
  }

  return TheHcalHaloData;
  
}


