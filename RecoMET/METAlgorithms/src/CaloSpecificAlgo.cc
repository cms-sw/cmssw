#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace reco;
using namespace std;

//-------------------------------------------------------------------------
// This algorithm adds calorimeter specific global event information to 
// the MET object which may be useful/needed for MET Data Quality Monitoring
// and MET cleaning.  This list is not exhaustive and additional 
// information will be added in the future. 
//-------------------------------------
reco::CaloMET CaloSpecificAlgo::addInfo(const TowerCollection &towers, CommonMETData met)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificCaloMETData specific;
  // Initialise the container 
  specific.mMaxEInEmTowers = 0.0;         // Maximum energy in EM towers
  specific.mMaxEInHadTowers = 0.0;        // Maximum energy in HCAL towers
  specific.mHadEnergyInHO = 0.0;          // Hadronic energy fraction in HO
  specific.mHadEnergyInHB = 0.0;          // Hadronic energy in HB
  specific.mHadEnergyInHF = 0.0;          // Hadronic energy in HF
  specific.mHadEnergyInHE = 0.0;          // Hadronic energy in HE
  specific.mEmEnergyInEB = 0.0;           // Em energy in EB
  specific.mEmEnergyInEE = 0.0;           // Em energy in EE
  specific.mEmEnergyInHF = 0.0;           // Em energy in HF
  specific.mEnergyFractionHadronic = 0.0; // Hadronic energy fraction
  specific.mEnergyFractionEm = 0.0;       // Em energy fraction
  double totalEnergy = 0.0; 
  double totalEm     = 0.0;
  double totalHad    = 0.0;
  double MaxTowerEm  = 0.0;
  double MaxTowerHad = 0.0;
  //retreive calo tower information from candidates
  //start with the first element of the candidate list
  TowerCollection::const_iterator tower = towers.begin();
  //get the EDM references to the CaloTowers from the candidate list
  edm::Ref<CaloTowerCollection> towerRef = (*tower)->get<CaloTowerRef>();
  //finally instantiate now, a list of pointers to the CaloTowers
  const CaloTowerCollection *towerCollection = towerRef.product();
  //iterate over all CaloTowers and record information
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  for( ; calotower != towerCollection->end(); calotower++ ) 
    {
      totalEnergy += calotower->energy();
      totalEm     += calotower->emEnergy();
      totalHad    += calotower->hadEnergy();

      if( MaxTowerEm  > calotower->emEnergy() )  MaxTowerEm  = calotower->emEnergy();
      if( MaxTowerHad > calotower->hadEnergy() ) MaxTowerHad = calotower->hadEnergy();

      specific.mHadEnergyInHO   += calotower->outerEnergy();

      bool hadIsDone = false;
      bool emIsDone = false;
      int cell = calotower->constituentsSize();
      while ( --cell >= 0 && (!hadIsDone || !emIsDone) ) 
	{
	  DetId id = calotower->constituent( cell );
	  if( !hadIsDone && id.det() == DetId::Hcal ) 
	    {
	      HcalSubdetector subdet = HcalDetId(id).subdet();
	      if( subdet == HcalBarrel || subdet == HcalOuter )
		{
		  specific.mHadEnergyInHB   += calotower->hadEnergy();
		  specific.mHadEnergyInHO   += calotower->outerEnergy();
		}
	      else if( subdet == HcalEndcap )
		{
		  specific.mHadEnergyInHE   += calotower->hadEnergy();
		}
	      else if( subdet == HcalForward )
		{
		  specific.mHadEnergyInHF   += calotower->hadEnergy();
		  specific.mEmEnergyInHF    += calotower->emEnergy(); 
		}
	      hadIsDone = true;
	    }
	  else if( !emIsDone && id.det() == DetId::Ecal )
	    {
	      EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
	      if( subdet == EcalBarrel )
		{
		  specific.mEmEnergyInEB    += calotower->emEnergy(); 
		}
	      else if( subdet == EcalEndcap ) 
		{
		  specific.mEmEnergyInEE    += calotower->emEnergy();
		}
	    }
	}
    }
  specific.mMaxEInEmTowers         = MaxTowerEm;  
  specific.mMaxEInHadTowers        = MaxTowerHad;         
  specific.mEnergyFractionHadronic = totalEm  / totalEnergy; 
  specific.mEnergyFractionEm       = totalHad / totalEnergy;       
  // Instantiate containers for the MET candidate and initialise them with
  // the MET information in "met" (of type CommonMETData)
  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type CaloMET, which is a MET object with 
  // the extra calorimeter specfic information added
  CaloMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
