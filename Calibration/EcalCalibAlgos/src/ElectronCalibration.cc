
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

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

ElectronCalibration::ElectronCalibration(const edm::ParameterSet& iConfig)
{

   rootfile_                  = iConfig.getParameter<std::string>("rootfile");
   recHitLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
   electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");
   trackLabel_ = iConfig.getParameter< edm::InputTag > ("trackLabel");
   calibAlgo_       = iConfig.getParameter<std::string>("CALIBRATION_ALGO");
}


ElectronCalibration::~ElectronCalibration()
{
  
  
}

//========================================================================
void ElectronCalibration::beginJob(edm::EventSetup const& iSetup) {
  //========================================================================
  
  // Book histograms 
  e9 = new TH1F("e9","E9 energy", 150, 0., 150.);
  e25 = new TH1F("e25","E25 energy", 150, 0., 150.);
  scE = new TH1F("scE","SC energy", 150, 0., 150.);
  trP = new TH1F("trP","Trk momentum", 150, 0., 150.);
  EoP = new TH1F("EoP","EoP", 300, 0., 3.);
  calibs = new TH1F("calib","Calibration constants", 400, 0., 2.);
  e25OverScE = new TH1F("e25OverscE","E25 / SC energy", 200, 0., 2.);
  E25oP = new TH1F("E25oP","E25 / P", 300, 0., 3.);
  Map = new TH2F("Map","Nb Events in Crystal",85,1, 85,70 ,5, 75);
  e9Overe25 = new TH1F("e9Overe25","E9 / E25", 200, 0., 2.);//
  // Book histograms 
  e9NoCuts = new TH1F("e9NoCuts","E9 energy (Before Cuts)", 150, 0., 150.);
  e25NoCuts = new TH1F("e25NoCuts","E25 energy (Before Cuts)", 150, 0., 150.);
  scENoCuts = new TH1F("scENoCuts","SC energy (Before Cuts)", 150, 0., 150.);
  trPNoCuts = new TH1F("trPNoCuts","Trk momentum (Before Cuts)", 150, 0., 150.);
  EoPNoCuts = new TH1F("EoPNoCuts","EoP (Before Cuts)", 300, 0., 3.);
  calibsNoCuts = new TH1F("calibNoCuts","Calibration constants (Before Cuts)", 400, 0., 2.);
  e25OverScENoCuts = new TH1F("e25OverscENoCuts","E25 / SC energy (Before Cuts)", 200, 0., 2.);
  E25oPNoCuts = new TH1F("E25oPNoCuts","E25 / P (Before Cuts)", 300, 0., 3.);
  MapNoCuts = new TH2F("MapNoCuts","Nb Events in Crystal (Before Cuts)",85,1, 85,70 ,5, 75);
  e9Overe25NoCuts = new TH1F("e9Overe25NoCuts","E9 / E25 (Before Cuts)", 200, 0., 2.);//
  
  calibClusterSize=5; 
  etaMin = 1;
  etaMax = 85;
  phiMin = 11; 
  phiMax = 71;
  MyL3Algo1 = new MinL3Algorithm(2,calibClusterSize, etaMin, etaMax, phiMin, phiMax);
  read_events=0;
  
  // get Region to be calibrated  
  ReducedMap = calibCluster.getMap(etaMin, etaMax, phiMin, phiMax);
  
  oldCalibs.resize(ReducedMap.size(),0.);
  
  
  CalibrationCluster::CalibMap::iterator itmap;
}


//========================================================================
void
ElectronCalibration::endJob() {
//========================================================================

int nIterations =10;
 solution = MyL3Algo1->iterate(EventMatrix, MaxCCeta, MaxCCphi, EnergyVector, nIterations);

for (int ii=0;ii<solution.size();ii++)
{
  cout << "solution[" << ii << "] = " << solution[ii] << endl;
  calibs->Fill(solution[ii]); 
}

newCalibs.resize(ReducedMap.size(),0.);

 calibXMLwriter write_calibrations;

int icry=0;
CalibrationCluster::CalibMap::iterator itmap;
for (itmap=ReducedMap.begin(); itmap != ReducedMap.end();itmap++){
 
//      newCalibs[icry] = oldCalibs[icry]*solution[icry];
      newCalibs[icry] = solution[icry];

      write_calibrations.writeLine(itmap->first,newCalibs[icry]);
 
      icry++;
}

 solutionNoCuts = MyL3Algo1->iterate(EventMatrixNoCuts, MaxCCetaNoCuts, MaxCCphiNoCuts, EnergyVectorNoCuts, nIterations);
for (int ii=0;ii<solutionNoCuts.size();ii++)
{
  calibsNoCuts->Fill(solutionNoCuts[ii]); 
}

////////////////////////       FINAL STATISTICS           ////////////////////

   std::cout << " " << std::endl;
   std::cout << "************* STATISTICS **************" << std::endl;
   std::cout << " Events Studied "<<read_events<< std::endl;

/////////////////////////////////////////////////////////////////////////////
  TFile f(rootfile_.c_str(),"RECREATE");

  e9NoCuts->Write();
  e25NoCuts->Write(); 
  scENoCuts->Write(); 
  trPNoCuts->Write(); 
  EoPNoCuts->Write(); 
  calibsNoCuts->Write(); 
  e25OverScENoCuts->Write(); 
  e9Overe25NoCuts->Write(); 
  E25oPNoCuts->Write(); 
  MapNoCuts->Write();

  e9->Write();
  e25->Write(); 
  scE->Write(); 
  trP->Write(); 
  EoP->Write(); 
  calibs->Write(); 
  e25OverScE->Write(); 
  e9Overe25->Write(); 
  E25oP->Write(); 
  Map->Write();

  f.Close();
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
        if(it->energy()> en_save){
	  en_save=it->energy();
	  save=p;
	  
	}
      count++;
     }
      return save;

}

//=================================================================================
EBDetId  ElectronCalibration::findMaxHit2(const std::vector<DetId> & v1,const EBRecHitCollection* hits) {
//=================================================================================

	double currEnergy = 0.;
	EBDetId maxHit;
 
	for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
           
	  // if(idsIt->subdetId()!=1) continue;
	   
	   EBRecHitCollection::const_iterator itrechit;
	   
	   try{
	      itrechit = hits->find(*idsIt);
	    
	    } catch ( ... ) {
	      
	      std::cout << "ElectronCalibration::findMaxHit2: rechit not found! " << std::endl;
	      
	      continue;
	   
	    }
	     
	   if(itrechit->energy() > currEnergy) {
 	      currEnergy=itrechit->energy();
 	      maxHit= *idsIt;
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
     //    std::cout << "Taken EBRecHitCollection " << std::endl;
         iEvent.getByLabel( recHitLabel_, phits);
   } catch ( ... ) {
     std::cerr << "Error! can't get the product EBRecHitCollection: " << std::endl;
   }
   const EBRecHitCollection* hits = phits.product(); // get a ptr to the product

  // Get pixelElectrons
   Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  try {
    iEvent.getByLabel(electronLabel_, pElectrons);
   } catch ( ... ) {
     std::cerr << "Error! can't get the product ElectronCollection: " << std::endl;
   }
  const reco::PixelMatchGsfElectronCollection* electronCollection = pElectrons.product();
  read_events++;
  if(read_events%1000 ==0)cout << "read_events = " << read_events << endl;

/*
  // Get pixelElectrons
  Handle<reco::TrackCollection> pTracks;
  try {
    iEvent.getByLabel(trackLabel_, pTracks);
   } catch ( ... ) {
//     std::cerr << "Error! can't get the product TrackCollection: " << hitCollection_.c_str() << std::endl;
   }
    const reco::TrackCollection* trackCollection = pTracks.product();
*/

   if (!hits)
     return;

   if (hits->size() == 0)
     return;

   if (!electronCollection)
     return;
   if (electronCollection->size() == 0)
     return;
/*
   if (!trackCollection)
     return;

   if (trackCollection->size() == 0)
     return;
*/
////////////////////////////////////////////////////////////////////////////////////////
///                          START HERE....
///////////////////////////////////////////////////////////////////////////////////////

  reco::PixelMatchGsfElectronCollection::const_iterator eleIt = electronCollection->begin();

  reco::PixelMatchGsfElectron highPtElectron;

  float highestElePt=0.;
  bool found=false;
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {

     if(abs(eleIt->eta())>1.479) continue;
     if(eleIt->eta()<0.0) continue;
//     if(eleIt->phi()>0.870) continue;
//     if(eleIt->phi()<0.174) continue;
//     if(eleIt->eSuperClusterOverP()<0.5 || eleIt->eSuperClusterOverP()>1.5) continue;
      
     if(eleIt->pt()>highestElePt) {
       highestElePt=eleIt->pt();
       highPtElectron = *eleIt;
       found =true;
     }

  }
  if(highestElePt<5)return;
      if(!found) return;

//     cout << "track eta = " << highPtElectron.eta() << endl;
//     cout << "track phi = " << highPtElectron.phi() << endl;
    
       const reco::SuperCluster & sc = *(highPtElectron.superCluster()) ;
       const std::vector<DetId> & v1 = sc.getHitsByDetId();
     EBDetId maxHitId;
      
       maxHitId = findMaxHit2(v1,hits); 
      
       if(maxHitId.null()){cout<<" Null "<<endl; return;}
 
  int maxCC_Eta = maxHitId.ieta();
  int maxCC_Phi = maxHitId.iphi();
  
    if(maxCC_Eta>83) return;
    if(maxCC_Eta<3)  return;

  vector<EBDetId> Xtals5x5 = calibCluster.get5x5Id(maxHitId);
 

// fill cluster energy
  vector<float> energy;
  float energy3x3=0.;  
  float energy5x5=0.;  

   for (unsigned int icry=0;icry<25;icry++)
     {
      
	   EBRecHitCollection::const_iterator itrechit;
	   
	   if(Xtals5x5[icry].subdetId()!=1) continue;
	   
	   try{

	      itrechit = hits->find(Xtals5x5[icry]);
	    
	      } catch ( ... ) {
	      	      
              cout << "DetId not is e25" << endl;
	      continue;
	      
	      }
 	      energy.push_back(itrechit->energy());
 	      energy5x5 += energy[icry];
 
 	      if ( icry == 6  || icry == 7  || icry == 8 ||
	           icry == 11 || icry == 12 || icry ==13 ||
	           icry == 16 || icry == 17 || icry ==18   )
	        {
	          energy3x3+=energy[icry];
	        }
		
     }
   //Once we have the matrix 5x5, we have to correct for gaps/cracks/umbrella and maincontainement  



   EventMatrixNoCuts.push_back(energy);
   EnergyVectorNoCuts.push_back(highPtElectron.p());
   MaxCCetaNoCuts.push_back(maxCC_Eta);
   MaxCCphiNoCuts.push_back(maxCC_Phi);

   e9NoCuts->Fill(energy3x3); 
   e25NoCuts->Fill(energy5x5); 
   e9Overe25NoCuts->Fill(energy3x3/energy5x5);
   scENoCuts->Fill(sc.energy()); 
   trPNoCuts->Fill(highPtElectron.p()); 
   EoPNoCuts->Fill(sc.energy()/highPtElectron.p()); 
   e25OverScENoCuts->Fill(energy5x5/sc.energy());
   E25oPNoCuts->Fill(energy5x5/highPtElectron.p());
   MapNoCuts->Fill(maxCC_Eta,maxCC_Phi);

   //Cuts!
   if((energy3x3/energy5x5)<0.92)return;
   if((energy5x5/highPtElectron.p())<0.8 || (energy5x5/highPtElectron.p())>1.2)return;

  e9->Fill(energy3x3); 
  e25->Fill(energy5x5); 
  e9Overe25->Fill(energy3x3/energy5x5);
  scE->Fill(sc.energy()); 
  trP->Fill(highPtElectron.p()); 
  EoP->Fill(sc.energy()/highPtElectron.p()); 
  e25OverScE->Fill(energy5x5/sc.energy());
  E25oP->Fill(energy5x5/highPtElectron.p());
  Map->Fill(maxCC_Eta,maxCC_Phi);

////////////////////////////////////////////////////


  EventMatrix.push_back(energy);
  EnergyVector.push_back(highPtElectron.p());
  MaxCCeta.push_back(maxCC_Eta);
  MaxCCphi.push_back(maxCC_Phi);

}

