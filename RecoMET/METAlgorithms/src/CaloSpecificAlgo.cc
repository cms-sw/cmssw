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
//reco::CaloMET CaloSpecificAlgo::addInfo(const CandidateCollection *towers, CommonMETData met)
reco::CaloMET CaloSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > towers, CommonMETData met)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificCaloMETData specific;
  // Initialise the container 
  specific.MaxEtInEmTowers = 0.0;         // Maximum energy in EM towers
  specific.MaxEtInHadTowers = 0.0;        // Maximum energy in HCAL towers
  specific.HadEtInHO = 0.0;          // Hadronic energy fraction in HO
  specific.HadEtInHB = 0.0;          // Hadronic energy in HB
  specific.HadEtInHF = 0.0;          // Hadronic energy in HF
  specific.HadEtInHE = 0.0;          // Hadronic energy in HE
  specific.EmEtInEB = 0.0;           // Em energy in EB
  specific.EmEtInEE = 0.0;           // Em energy in EE
  specific.EmEtInHF = 0.0;           // Em energy in HF
  specific.EtFractionHadronic = 0.0; // Hadronic energy fraction
  specific.EtFractionEm = 0.0;       // Em energy fraction
  double totalEt = 0.0; 
  double totalEm     = 0.0;
  double totalHad    = 0.0;
  double MaxTowerEm  = 0.0;
  double MaxTowerHad = 0.0;
  if( towers->size() == 0 )  // if there are no towers, return specific = 0
    {
      cout << "[CaloMET] Number of Candidate CaloTowers is zero : Unable to calculate calo specific info. " << endl;
      const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
      const Point vtx( 0.0, 0.0, 0.0 );
      CaloMET specificmet( specific, met.sumet, p4, vtx );
      return specificmet;
    }
  /*
  //retreive calo tower information from candidates
  //start with the first element of the candidate list
  CandidateCollection::const_iterator tower = towers->begin();
  //get the EDM references to the CaloTowers from the candidate list
  edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
  */
  edm::Ref<CaloTowerCollection> towerRef = towers->begin()->get<CaloTowerRef>();
  //finally instantiate now, a list of pointers to the CaloTowers
  const CaloTowerCollection *towerCollection = towerRef.product();

  //iterate over all CaloTowers and record information
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  for( ; calotower != towerCollection->end(); calotower++ ) 
    {
      totalEt  += calotower->et();
      totalEm  += calotower->emEt();
      totalHad += calotower->hadEt();

      if( calotower->emEt()  > MaxTowerEm  ) MaxTowerEm  = calotower->emEt();
      if( calotower->hadEt() > MaxTowerHad ) MaxTowerHad = calotower->hadEt();

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
		  specific.HadEtInHB   += calotower->hadEt();
		  specific.HadEtInHO   += calotower->outerEt();
		}
	      else if( subdet == HcalEndcap )
		{
		  specific.HadEtInHE   += calotower->hadEt();
		}
	      else if( subdet == HcalForward )
		{
		  specific.HadEtInHF   += calotower->hadEt();
		  specific.EmEtInHF    += calotower->emEt(); 
		}
	      hadIsDone = true;
	    }
	  else if( !emIsDone && id.det() == DetId::Ecal )
	    {
	      EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
	      if( subdet == EcalBarrel )
		{
		  specific.EmEtInEB    += calotower->emEt(); 
		}
	      else if( subdet == EcalEndcap ) 
		{
		  specific.EmEtInEE    += calotower->emEt();
		}
	      emIsDone = true;
	    }
	}
    }
  specific.MaxEtInEmTowers         = MaxTowerEm;  
  specific.MaxEtInHadTowers        = MaxTowerHad;         
  specific.EtFractionHadronic = totalEm  / totalEt; 
  specific.EtFractionEm       = totalHad / totalEt;       
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
