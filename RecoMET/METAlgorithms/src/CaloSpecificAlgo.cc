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
reco::CaloMET CaloSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > towers, CommonMETData met, bool noHF)
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
  specific.METSignificance = -1.0;    // MET Significance
  specific.CaloMETInpHF = 0.0;        // CaloMET in HF+ 
  specific.CaloMETInmHF = 0.0;        // CaloMET in HF- 
  specific.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+ 
  specific.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF- 
  specific.CaloMETInpHE = 0.0;        // CaloMET in HE+ 
  specific.CaloMETInmHE = 0.0;        // CaloMET in HE- 
  specific.CaloMETPhiInpHE= 0.0;      // CaloMET-phi in HE+ 
  specific.CaloMETPhiInmHE = 0.0;     // CaloMET-phi in HE- 
  specific.CaloMETInpHB = 0.0;        // CaloMET in HB+ 
  specific.CaloMETInmHB = 0.0;        // CaloMET in HB- 
  specific.CaloMETPhiInpHB = 0.0;     // CaloMET-phi in HB+ 
  specific.CaloMETPhiInmHB = 0.0;     // CaloMET-phi in HB- 
  
  double totalEt = 0.0; 
  double totalEm     = 0.0;
  double totalHad    = 0.0;
  double MaxTowerEm  = 0.0;
  double MaxTowerHad = 0.0;
  double sumEtInHF = 0.0;
  double MExInpHF = 0.0;
  double MEyInpHF = 0.0;
  double MExInmHF = 0.0;
  double MEyInmHF = 0.0;
  double MExInpHE = 0.0;
  double MEyInpHE = 0.0;
  double MExInmHE = 0.0;
  double MEyInmHE = 0.0;
  double MExInpHB = 0.0;
  double MEyInpHB = 0.0;
  double MExInmHB = 0.0;
  double MEyInmHB = 0.0;

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
		  if (calotower->eta()>=0)
                    {
                      MExInpHB -= (calotower->et() * cos(calotower->phi()));
                      MEyInpHB -= (calotower->et() * sin(calotower->phi()));
                    }
                  else
                    {
                      MExInmHB -= (calotower->et() * cos(calotower->phi()));
                      MEyInmHB -= (calotower->et() * sin(calotower->phi()));
                    }
		}
	      else if( subdet == HcalEndcap )
		{
		  specific.HadEtInHE   += calotower->hadEt();
		  if (calotower->eta()>=0)
                    {
                      MExInpHE -= (calotower->et() * cos(calotower->phi()));
                      MEyInpHE -= (calotower->et() * sin(calotower->phi()));
                    }
                  else
                    {
                      MExInmHE -= (calotower->et() * cos(calotower->phi()));
                      MEyInmHE -= (calotower->et() * sin(calotower->phi()));
                    }

		}
	      else if( subdet == HcalForward )
		{
		  sumEtInHF            += calotower->et();
		  specific.HadEtInHF   += calotower->hadEt();
		  specific.EmEtInHF    += calotower->emEt(); 
		  if (calotower->eta()>=0)
		    {
		      MExInpHF -= (calotower->et() * cos(calotower->phi()));
		      MEyInpHF -= (calotower->et() * sin(calotower->phi()));
		    }
		  else
		    {
		      MExInmHF -= (calotower->et() * cos(calotower->phi()));
		      MEyInmHF -= (calotower->et() * sin(calotower->phi()));
		    }
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

  // Form sub-det specific MET-vectors
  LorentzVector METpHF(MExInpHF, MEyInpHF, 0, sqrt(MExInpHF*MExInpHF + MEyInpHF*MEyInpHF));
  LorentzVector METmHF(MExInmHF, MEyInmHF, 0, sqrt(MExInmHF*MExInmHF + MEyInmHF*MEyInmHF));
  LorentzVector METpHE(MExInpHE, MEyInpHE, 0, sqrt(MExInpHE*MExInpHE + MEyInpHE*MEyInpHE));
  LorentzVector METmHE(MExInmHE, MEyInmHE, 0, sqrt(MExInmHE*MExInmHE + MEyInmHE*MEyInmHE));
  LorentzVector METpHB(MExInpHB, MEyInpHB, 0, sqrt(MExInpHB*MExInpHB + MEyInpHB*MEyInpHB));
  LorentzVector METmHB(MExInmHB, MEyInmHB, 0, sqrt(MExInmHB*MExInmHB + MEyInmHB*MEyInmHB));
  
  specific.CaloMETInpHF = METpHF.pt();
  specific.CaloMETInmHF = METmHF.pt();
  specific.CaloMETInpHE = METpHE.pt();
  specific.CaloMETInmHE = METmHE.pt();
  specific.CaloMETInpHB = METpHB.pt();
  specific.CaloMETInmHB = METmHB.pt();

  specific.CaloMETPhiInpHF = METpHF.Phi();
  specific.CaloMETPhiInmHF = METmHF.Phi();
  specific.CaloMETPhiInpHE = METpHE.Phi();
  specific.CaloMETPhiInmHE = METmHE.Phi();
  specific.CaloMETPhiInpHB = METpHB.Phi();
  specific.CaloMETPhiInmHB = METmHB.Phi();

  specific.MaxEtInEmTowers         = MaxTowerEm;  
  specific.MaxEtInHadTowers        = MaxTowerHad;         
  specific.EtFractionHadronic = totalHad / totalEt; 
  specific.EtFractionEm       =  totalEm / totalEt;       

  
  // Instantiate containers for the MET candidate and initialise them with
  // the MET information in "met" (of type CommonMETData)

  // remove HF from MET calculation if specified
  if (noHF)
    {
      met.mex -= (MExInmHF + MEyInpHF);
      met.mey -= (MEyInmHF + MEyInpHF);
      met.met = sqrt(met.mex*met.mex + met.mey*met.mey);
      met.sumet -= sumEtInHF;
    } 

  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type CaloMET, which is a MET object with 
  // the extra calorimeter specfic information added
  CaloMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
