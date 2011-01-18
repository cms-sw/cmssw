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

reco::CaloMET CaloSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > towers, CommonMETData met, bool noHF, double globalThreshold)
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
  specific.CaloSETInpHF = 0.0;        // CaloSET in HF+ 
  specific.CaloSETInmHF = 0.0;        // CaloSET in HF- 
  specific.CaloMETInpHF = 0.0;        // CaloMET in HF+ 
  specific.CaloMETInmHF = 0.0;        // CaloMET in HF- 
  specific.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+ 
  specific.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF- 
  specific.METSignificance = 0.0;

  double totalEt = 0.0; 
  double totalEm     = 0.0;
  double totalHad    = 0.0;
  double MaxTowerEm  = 0.0;
  double MaxTowerHad = 0.0;
  double sumEtInpHF = 0.0;
  double sumEtInmHF = 0.0;
  double MExInpHF = 0.0;
  double MEyInpHF = 0.0;
  double MExInmHF = 0.0;
  double MEyInmHF = 0.0;

  if( towers->size() == 0 )  // if there are no towers, return specific = 0
    {
      //   LogDebug("CaloMET") << "Number of Candidate CaloTowers is zero : Unable to calculate calo specific info. " ;
      const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
      const Point vtx( 0.0, 0.0, 0.0 );
      CaloMET specificmet( specific, met.sumet, p4, vtx );
      return specificmet;
    }

  edm::View<Candidate>::const_iterator towerCand = towers->begin();
  for( ; towerCand != towers->end(); towerCand++ ) 
    {
      const Candidate* candidate = &(*towerCand);
      if (candidate) {
	const CaloTower* calotower = dynamic_cast<const CaloTower*> (candidate);
	if (calotower)
	  {
	    double caloTowerEt=calotower->et();
	    double caloTowerHadEt=calotower->hadEt();
	    double caloTowerEmEt=calotower->emEt();
	    if(caloTowerEt < globalThreshold) continue;
	    totalEt  += caloTowerEt;
	    totalEm  += caloTowerEmEt;
	           
	    //totalHad += caloTowerHadEt + calotower->outerEt() ;
	           
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
			if( caloTowerHadEt > MaxTowerHad ) MaxTowerHad = caloTowerHadEt;
			specific.HadEtInHB   += caloTowerHadEt;
			specific.HadEtInHO   += calotower->outerEt();
		      }
		    else if( subdet == HcalEndcap )
		      {
			if( caloTowerHadEt > MaxTowerHad ) MaxTowerHad = caloTowerHadEt;
			specific.HadEtInHE   += caloTowerHadEt;
		      }
		    else if( subdet == HcalForward )
		      {
			if (!noHF)
			  {
			    if( caloTowerHadEt > MaxTowerHad ) MaxTowerHad = caloTowerHadEt;
			    if( caloTowerEmEt  > MaxTowerEm  ) MaxTowerEm  = caloTowerEmEt;
			    //These quantities should be nonzero only if HF is included, i.e., noHF == false
			    specific.HadEtInHF   += caloTowerHadEt;
			    specific.EmEtInHF    += caloTowerEmEt;
			  }
			else
			  {
			    //These quantities need to be corrected from above if HF is excluded
			    // totalHad             -= caloTowerHadEt;  
			    totalEm              -= caloTowerEmEt;
			    totalEt              -= caloTowerEt;
			  }
			// These get calculate regardless of NoHF == true or not.
			// They are needed below for either case. 
			if (calotower->eta()>=0)
			  {
			    sumEtInpHF  += caloTowerEt;
			    MExInpHF    -= (caloTowerEt * cos(calotower->phi()));
			    MEyInpHF    -= (caloTowerEt * sin(calotower->phi()));
			  }
			else
			  {
			    sumEtInmHF  += caloTowerEt;
			    MExInmHF    -= (caloTowerEt * cos(calotower->phi()));
			    MEyInmHF    -= (caloTowerEt * sin(calotower->phi()));
			  }
		      }
		    hadIsDone = true;
		  }
		else if( !emIsDone && id.det() == DetId::Ecal )
		  {
		    EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
		    if( caloTowerEmEt  > MaxTowerEm  ) MaxTowerEm  = caloTowerEmEt;
		    if( subdet == EcalBarrel )
		      {
			specific.EmEtInEB    += caloTowerEmEt; 
		      }
		    else if( subdet == EcalEndcap ) 
		      {
			specific.EmEtInEE    += caloTowerEmEt;
		      }
		    emIsDone = true;
		  }
	      }
	  }
      }
    }
  
  //Following Greg L's suggestion to calculate this quantity outside of the loop and to avoid confusion. 
  //This should work regardless of HO's inclusion / exclusion .
  totalHad += (totalEt - totalEm);
  
  if(!noHF)
    { // Form sub-det specific MET-vectors
      LorentzVector METpHF(MExInpHF, MEyInpHF, 0, sqrt(MExInpHF*MExInpHF + MEyInpHF*MEyInpHF));
      LorentzVector METmHF(MExInmHF, MEyInmHF, 0, sqrt(MExInmHF*MExInmHF + MEyInmHF*MEyInmHF));
      specific.CaloMETInpHF = METpHF.pt();
      specific.CaloMETInmHF = METmHF.pt();
      specific.CaloMETPhiInpHF = METpHF.Phi();
      specific.CaloMETPhiInmHF = METmHF.Phi();
      specific.CaloSETInpHF = sumEtInpHF;
      specific.CaloSETInmHF = sumEtInmHF;
    }
  else
    { // remove HF from MET calculation 
      met.mex   -= (MExInmHF + MExInpHF);
      met.mey   -= (MEyInmHF + MEyInpHF);
      met.sumet -= (sumEtInpHF + sumEtInmHF);
      met.met    = sqrt(met.mex*met.mex + met.mey*met.mey);   
    } 

  specific.MaxEtInEmTowers         = MaxTowerEm;  
  specific.MaxEtInHadTowers        = MaxTowerHad;         
  specific.EtFractionHadronic = totalHad / totalEt; 
  specific.EtFractionEm       =  totalEm / totalEt;       

  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type CaloMET, which is a MET object with 
  // the extra calorimeter specfic information added
  CaloMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
