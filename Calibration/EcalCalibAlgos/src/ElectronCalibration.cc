
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"
#include "Calibration/EcalCalibAlgos/interface/ElectronCalibration.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>



ElectronCalibration::ElectronCalibration(const edm::ParameterSet& iConfig)
{

//   rootfile_                  = iConfig.getUntrackedParameter<std::string>("rootfile","electronCalibration.root");
   recHitLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
   electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");
   trackLabel_ = iConfig.getParameter< edm::InputTag > ("trackLabel");
   calibAlgo_       = iConfig.getParameter<std::string>("CALIBRATION_ALGO");
   cout << "Read Inputs" << endl;
}


ElectronCalibration::~ElectronCalibration()
{
 

}

//========================================================================
void
ElectronCalibration::beginJob(edm::EventSetup const& iSetup) {
//========================================================================
 calibClusterSize=5; 
 etaMin = 1;
 etaMax = 85;
 phiMin = 11; 
 phiMax = 50;
 MyL3Algo1 = new MinL3Algorithm(calibClusterSize, etaMin, etaMax, phiMin, phiMax);


 }

//========================================================================
void
ElectronCalibration::endJob() {
//========================================================================
int nIterations =10;
solution = MyL3Algo1->iterate(EventMatrix, MaxCCeta, MaxCCphi, EnergyVector, nIterations);

int ii=0;
for (int ii=0;ii<solution.size();ii++)
{
  cout << "solution[" << ii << "] = " << solution[ii] << endl;
}
////////////////////////       FINAL STATISTICS           ////////////////////

   std::cout << " " << std::endl;
   std::cout << "************* STATISTICS **************" << std::endl;


/////////////////////////////////////////////////////////////////////////////
//  TFile f(rootfile_.c_str(),"RECREATE");

//  e25->Write(); 
//  f.Close();
 delete MyL3Algo1; 
}


//=================================================================================
EBDetId ElectronCalibration::findMaxHit(edm::Handle<EBRecHitCollection> &  phits) {
//=================================================================================

     EcalRecHitCollection ecrh = *phits;
     EcalRecHitCollection::iterator it;
     int count=0;
     EBDetId save;
     float en_save=0;
     for (it = ecrh.begin(); it != ecrh.end(); it++)
     {
       EBDetId p = EBDetId(it->id().rawId());
       // std::cout << "Hit list " << p.ieta() << " " << p.iphi() << " " << it->energy() << std::endl;
        if(it->energy()> en_save){
	  en_save=it->energy();
	  save=p;
	  
	}
      count++;
     }
     //return save.ic();
      return save;

}

//=================================================================================
EBDetId ElectronCalibration::findMaxHit2(std::vector<DetId> & v1,const EBRecHitCollection* hits) {
//=================================================================================

	double currEnergy = 0.;
	EBDetId maxHit(0);
 
	for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
	  	  
	  if((hits->find(*idsIt))->energy() > currEnergy) {
	    currEnergy=(hits->find(*idsIt))->energy();
	    maxHit=*idsIt;
	  }
	}
	

      return maxHit;

}


//=================================================================================
void
ElectronCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//=================================================================================
   using namespace edm;

  // Get EBRecHits
   Handle<EBRecHitCollection> phits;
   try {
    std::cout << "Taken EBRecHitCollection " << std::endl;
     iEvent.getByLabel( recHitLabel_, phits);
   } catch ( std::exception& ex ) {
//     std::cerr << "Error! can't get the product EBRecHitCollection: " << hitCollection_.c_str() << std::endl;
   }
     const EBRecHitCollection* hits = phits.product(); // get a ptr to the product

  // Get pixelElectrons
  
  Handle<reco::ElectronCollection> pElectrons;
  try {
    iEvent.getByLabel(electronLabel_, pElectrons);
   } catch ( std::exception& ex ) {
//     std::cerr << "Error! can't get the product ElectronCollection: " << hitCollection_.c_str() << std::endl;
   }
    const reco::ElectronCollection* electronCollection = pElectrons.product();


  // Get pixelElectrons
  Handle<reco::TrackCollection> pTracks;
  try {
    iEvent.getByLabel(trackLabel_, pTracks);
   } catch ( std::exception& ex ) {
//     std::cerr << "Error! can't get the product TrackCollection: " << hitCollection_.c_str() << std::endl;
   }
    const reco::TrackCollection* trackCollection = pTracks.product();

   if (!hits)
     return;

   if (hits->size() == 0)
     return;

   if (!electronCollection)
     return;

   if (electronCollection->size() == 0)
     return;

   if (!trackCollection)
     return;

   if (trackCollection->size() == 0)
     return;

////////////////////////////////////////////////////////////////////////////////////////
///                          START HERE....
///////////////////////////////////////////////////////////////////////////////////////
  read_events++;

  float highestElePt=0.;
  float highestEleEta=-999.;
  float highestElePhi=-999.;
  reco::TrackCollection::const_iterator itTrk;
  for(itTrk=trackCollection->begin();itTrk!=trackCollection->end();itTrk++)
  {
    float elePt = itTrk->pt();
    cout << "track pt " << elePt << endl;
    if(elePt>highestElePt) {
       highestElePt=elePt;
       highestEleEta=itTrk->eta();
       highestElePhi=itTrk->phi();
    }
  }
  
    cout << "Highest Pt " << highestElePt << endl;
    cout << "Highest Pt eta " << highestEleEta << endl;
    cout << "Highest Pt phi " << highestElePhi << endl;
  
  reco::ElectronCollection::const_iterator eleIt = electronCollection->begin();

  reco::Electron highPtElectron;
  
  
  cout << "Electron Size " << electronCollection->size() << endl;
   
  float minDist =99999.;
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
		
	float deltaR = sqrt((eleIt->eta()-highestEleEta)*(eleIt->eta()-highestEleEta) + 
	                    (eleIt->phi()-highestElePhi)*(eleIt->phi()-highestElePhi));
	
	if(deltaR < minDist) {
	   highPtElectron = *eleIt;
	   minDist=deltaR;
	   }
	  
  }
	
    cout << "min distance = " << minDist << endl;
	const reco::SuperCluster & sc = *(highPtElectron.superCluster()) ;
	
	std::vector<DetId> & v1 = sc.getHitsByDetId();

        EBDetId maxHitId = findMaxHit2(v1,hits); 

//  EBDetId maxHitId = findMaxHit(phits); 

  int maxCC_Eta = maxHitId.ieta();
  int maxCC_Phi = maxHitId.iphi();
  
    cout << "maxCC_Eta = " << maxCC_Eta << endl;
    cout << "maxCC_Phi = " << maxCC_Phi << endl;
  

  vector<EBDetId> Xtals5x5 = calibCluster.get5x5Id(maxHitId);
  
// fill cluster energy
  vector<float> energy;
  float energy3x3=0.;  
  float energy5x5=0.;  

   for (unsigned int icry=0;icry<25;icry++)
     {
       
       cout << "energy = " << (hits->find(Xtals5x5[icry]))->energy() << 
       " eta = " << Xtals5x5[icry].ieta() << " phi = " << Xtals5x5[icry].iphi() << endl;
      
       energy.push_back((hits->find(Xtals5x5[icry]))->energy());
       energy5x5 += energy[icry];
 
       if ( icry == 6  || icry == 7  || icry == 8 ||
	    icry == 11 || icry == 12 || icry ==13 ||
	    icry == 16 || icry == 17 || icry ==18   )
	 {
	   energy3x3+=energy[icry];
	 }
     }

   cout << "track Size " << trackCollection->size() << endl;
 
   
////////////////////////////////////////////////////

    for (int yk=0; yk<energy.size(); yk++) cout << "input energy [" << yk << "] = "<< energy[yk] << endl;
    cout << "input pt " << highestElePt << endl;
    cout << "input eta " << maxCC_Eta << endl;
    cout << "input phi " << maxCC_Phi << endl;
// maxCC_Eta=20;
// maxCC_Phi=30;

  EventMatrix.push_back(energy);
  EnergyVector.push_back(highestElePt);
  MaxCCeta.push_back(maxCC_Eta);
  MaxCCphi.push_back(maxCC_Phi);

}

