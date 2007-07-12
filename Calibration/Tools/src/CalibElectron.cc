#include "Calibration/Tools/interface/CalibElectron.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//#include <TMath.h>
#include <iostream>

using namespace calib;
using namespace std;

#define NXTAL_ETA_BAR 170
#define NXTAL_PHI_BAR 360
#define NMOD_BAR 144

CalibElectron::CalibElectron() : theElectron_(0), theHits_(0)
{
}


std::vector< std::pair<int,float> > CalibElectron::getCalibModulesWeights(TString calibtype)
{
  std::vector< std::pair<int,float> > theWeights;
  if (calibtype == "RING")
    {
      float w_ring[NXTAL_ETA_BAR];

      for (int i=0;i<NXTAL_ETA_BAR;i++)
	w_ring[i]=0.;
      
      std::vector<DetId> scDetIds = theElectron_->superCluster()->getHitsByDetId();

      
      for(std::vector<DetId>::const_iterator idIt=scDetIds.begin(); idIt!=scDetIds.end(); idIt++){
    
	//my eta index goes from 0 to 169
	
	int etaIndex(0);
	
	if(EBDetId(*idIt).ieta()<0) 
	  etaIndex = EBDetId(*idIt).ieta() + 85; 
	else 
	  etaIndex = EBDetId(*idIt).ieta() + 84; 
	
	const EcalRecHit* rh = &*(theHits_->find(*idIt));

	w_ring[etaIndex]+=rh->energy();
	
      }

      for (int i=0;i<NXTAL_ETA_BAR;i++)
	if (w_ring[i]!=0.) 
	  theWeights.push_back(std::pair<int,float>(i,w_ring[i]));
	  // cout << " ring " << i << " - energy sum " << w_ring[i] << endl;
    }
  else
    {
      cout << "CalibType not yet implemented" << endl;
    }
  
  return theWeights;
}

  
// TLorentzVector Electron::getEnergy4Vector()
// {
//   TLorentzVector pvec_;
//   pvec_.SetPtEtaPhiE(electronSCEnocorr_/TMath::CosH(electronSCEta_),electronSCEta_,electronSCPhi_,electronSCEnocorr_);
//   return pvec_;
// }  

// bool Electron::isInCrack()
// {

//   float x = fabs(electronSCEta_);
//   float y = fabs(electronSCPhi_)- ((int)(fabs(electronSCPhi_)/0.349)*0.349);
//   return (x < 0.018 || 
// 	  (x>0.423 && x<0.461) ||
// 	  (x>0.770 && x<0.806) ||
// 	  (x>1.127 && x<1.163) ||
// 	  (x>1.460 && x<1.558) ||
// 	  (y> 0.157 && y<0.193));
// }

// bool Electron::isInBarrel(){

//   float x = fabs(electronSCEta_);
//   return (x <= 1.460);

// }

// bool Electron::isInEndcap(){

//   float x = fabs(electronSCEta_);
//   return (x >= 1.558 && x < 2.7);

// }

// float Electron::fNCrystals(int nCry){
  
//   float p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

//   if (nCry<=10) {
// 	p0 =  6.32879e-01; 
// 	p1 =  1.14893e-01; 
// 	p2 = -2.45705e-02; 
// 	p3 =  2.53074e-03; 
// 	p4 = -9.29654e-05; 
//   } else if (nCry>10 && nCry<=30) {
// 	p0 =  6.93196e-01; 
// 	p1 =  4.44034e-02; 
// 	p2 = -2.82229e-03; 
// 	p3 =  8.19495e-05; 
// 	p4 = -8.96645e-07; 
//   } else {
// 	p0 =  5.65474e+00; 
// 	p1 = -6.31640e-01; 
// 	p2 =  3.14218e-02; 
// 	p3 = -6.84256e-04; 
// 	p4 =  5.50659e-06; 
//   }

//   float x  = (float) nCry;
//   return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
  
// }

// float Electron::fEtaBarrelBad(float scEta){
  
//   // f(eta) for the class = 3 (estimated from 1Mevt single e sample)
//   float p0 =  1.00160e+00; 
//   float p1 = -1.54445e-02; 
//   float p2 = -7.95510e-04; 
//   float p3 =  8.89824e-03; 
//   float p4 = -1.83390e-02; 

//   float x  = (float) fabs(scEta);
//   return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;  

// }
  
// float Electron::fEtaBarrelGood(float scEta){

//   // f(eta) for the first 3 classes (0, 1 and 2) (estimated from 1Mevt single e sample)
//   float p0 =  1.00525e+00; 
//   float p1 =  5.30274e-03; 
//   float p2 = -4.54321e-02; 
//   float p3 =  5.86581e-02; 
//   float p4 = -2.84469e-02; 

//   float x  = (float) fabs(scEta);
//   return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; 

// }

// float Electron::fEtaBarrelGood25(float scEta){

//   // f(eta) for the first 3 classes (0, 1 and 2) (estimated from 1Mevt single e sample)
//   float p0 =  1.00525e+00; 
//   float p1 =  5.30274e-03; 
//   float p2 = -4.54321e-02; 
//   float p3 =  5.86581e-02; 
//   float p4 = -2.84469e-02; 
//   float scale = 1.04340e+00; 
//   float x  = (float) fabs(scEta);
//   return scale/(p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x); 

// }
  
// float Electron::fEtaEndcap(float scEta){
//   // f(eta) for endcap 
//   float p0 =         3.34835e+00;
//   float p1 =        -5.21591e+00;
//   float p2 =         4.08598e+00;
//   float p3 =        -1.36064e+00;
//   float p4 =         1.64725e-01;
  
//   float x  = (float) fabs(scEta);
//   return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
// }

// int Electron::nCrystalsGT2Sigma(const reco::SuperCluster *seed, float sigmaNoise){
  
//   int nCry = 0;
//   for(vector<CellID>::const_iterator it=seed->usedCrystals.begin();
//       it!=seed->usedCrystals.end();it++)
//     {
//       if (seed->getArray()->getEnergy(*it) > 2*sigmaNoise)
// 	nCry++;
//     }
//   return nCry;
// }

// Float_t Electron::getE25()
// {
// //   CellID CurrentPosition((NCRY_PHI-1)/2,(NCRY_ETA-1)/2);

// //   int startEta = 0;
// //   int startPhi = 0;
// //   int _etaBins=5;
// //   int _phiBins=5;
// //   float ed=0.;	  	  

// //   // go to the starting position
// //   // ---------------------------
// //   for (int i=0; i<((_etaBins-1)/2); ++i)
// //     {
// //     }
	  
// //   for (int i=0; i<((_phiBins-1)/2); ++i)
// //     {
// //       if (!CurrentPosition.south()) break;
// //       --startPhi;
// //     }
	  
	  
// //   // loop over eta
// //   for (int i=startEta; i<=(_etaBins-1)/2; i++)
// //     {          
// //       // loop over phi
// //       int j = 0;
// //       for (j=startPhi; j<=(_phiBins-1)/2; j++)
// // 	{
// // 	  if(theArray->getEnergy(CurrentPosition)!=-999.)
// // 		ed+=theArray->getEnergy(CurrentPosition);
	  
// // 	  if (!CurrentPosition.north()) 
// // 	    {
// // 	      cout << "[Electron]::[getE25] North Position not found" << endl;
// // 	      continue;
// // 	    }
// // 	} // loop over phi
      
// //       // back in phi to the starting point
// //       for (int k=((_phiBins-1)/2); k>=startPhi; --k) CurrentPosition.south();
      
// //       if (!CurrentPosition.west()) continue;
      
// //     }// loop over eta

// //   return ed;
//   return -1.;
// }

// Float_t Electron::getE9()
// {
// //   CellID CurrentPosition((NCRY_PHI-1)/2,(NCRY_ETA-1)/2);

// //   int startEta = 0;
// //   int startPhi = 0;
// //   int _etaBins=5;
// //   int _phiBins=5;
// //   float ed=0.;	  	  

// //   // go to the starting position
// //   // ---------------------------
// //   for (int i=0; i<((_etaBins-1)/2); ++i)
// //     {
// //       if (!CurrentPosition.east()) break;
// //       --startEta;
// //     }
	  
// //   for (int i=0; i<((_phiBins-1)/2); ++i)
// //     {
// //       if (!CurrentPosition.south()) break;
// //       --startPhi;
// //     }
	  
	  
// //   // loop over eta
// //   for (int i=startEta; i<=(_etaBins-1)/2; i++)
// //     {          
// //       // loop over phi
// //       int j = 0;
// //       for (j=startPhi; j<=(_phiBins-1)/2; j++)
// // 	{
// // 	  if(theArray->getEnergy(CurrentPosition)!=-999.)
// // 		ed+=theArray->getEnergy(CurrentPosition);
	  
// // 	  if (!CurrentPosition.north()) 
// // 	    {
// // 	      cout << "[Electron]::[getE9] North Position not found" << endl;
// // 	      continue;
// // 	    }
// // 	} // loop over phi
      
// //       // back in phi to the starting point
// //       for (int k=((_phiBins-1)/2); k>=startPhi; --k) CurrentPosition.south();
      
// //       if (!CurrentPosition.west()) continue;
      
// //     }// loop over eta

// //   return ed;
//   return -1;
// }


// Float_t Electron::getE1()
// {
//   EcalRecHitCollection::const_iterator maxHit=theHits->find(maxEnergyHitId_);
//   return (*maxHit).energy();
// }

// void Electron::fillTrackInfo(const reco::PixelMatchGsfElectron* anEle)
// {
//   // Valid, Lost and Invalid number of hits
//   electronTRvhits_=anEle->track()->found();
//   electronTRlhits_=anEle->track()->lost();
//   electronTRihits_=anEle->track()->invalid();

//   // sigma(1/p)
//   //  electronTRinvps_=anEle->track()->

//   //Chi quadro e gradi di libertà
//   electronTRChisq_=anEle->track()->chi2();
//   electronTRdegOfFreedom_=anEle->track()->ndof();

//   //Carica dell'elettrone
//   electronTRcharge_=anEle->track()->charge();
//   //Transverse and Longitudinal impact parameter
//   electronTRtip_=anEle->track()->innerPosition().perp();
//   electronTRlip_=anEle->track()->innerPosition().z();
 
//   //Isolamento dell'elettrone
//   //  electronISiso_=anEle->track()->electronISiso_;
//   //  electronPXLlines_=anEle->track()->electronPXLlines_;
 
//   // Impulso dello stato iniziale della traccia
//   electron_Tr_Pmag_=anEle->track()->innerMomentum().p();
//   electron_Tr_Peta_=anEle->track()->innerMomentum().eta();
//   electron_Tr_Pphi_=anEle->track()->innerMomentum().phi();

//   // Impulso dello stato ad ECAL della traccia
//   electron_Tr_Out_Pmag_=anEle->track()->outerMomentum().p();
//   electron_Tr_Out_Peta_=anEle->track()->outerMomentum().eta();
//   electron_Tr_Out_Pphi_=anEle->track()->outerMomentum().phi();

//   //  electronHoE_=anEle->electronHoE_ ; //H/E

//   electronTRstat_=anEle->track()->innerOk();; //stat indi
// }

// void Electron::fillSCInfo(const reco::PixelMatchGsfElectron* anEle, const EcalRecHitCollection* myHits)
// {
//   float maxHitEnergy=-999999.;

//   vector<DetId> theHitsIds=anEle->superClutster()->getHitsByDetId();

//   EcalRecHitCollection theNewHits;

//   float ESCnocorr=0.;
//   float ESCmiscalib=0.;
  
//   for (int i=0;i<theHitsIds.size(); i++)
//     {
//       EcalRecHitCollection::const_iterator aHit = myHits->find(theHitsIds[i]);
//       if (aHit != myHits->end())
// 	{
// 	  ESCnocorr+=(*aHit).energy();
// 	  float newEnergy=(*aHit).energy()*EcalBarrelCalibMap::getMap()->readCalib(EBDetId(theHitsIds[i]))*EcalBarrelCalibMap::getMap()->readMisCalib(EBDetId(theHitsIds[i]));
// 	  if (newEnergy>=maxHitEnergy)
// 	    {
// 	      maxHitEnergy=newEnergy;
// 	      maxEnergyHitId_=theHitsIds[i];
// 	    }
// 	  ESCmiscalib+=newEnergy;
// 	  theHits->push_back(EcalRecHit(theHitsIds[i],newEnergy,(*aHit).time()));
// 	}
//     }
    
//   electronSCEnocorr_ = ESCmiscalib;    //not corrected
//   electronSCE_ =  ESCmiscalib*(anEle->superClutster()->energy()/ESCnocorr);          
//   //Recalculate positions
//   electronSCEta_ = LogC;
//   electronSCPhi_ = LogC;
//   electronNDigis_ = theNewHits.size();
//   electronSCNClus_ = anEle->superClutster()->nCluse(); ?? 

//   electronSCE25_ =  getE25();
//   electronSCE25Corr_ =  electronSCE25_*fEtaBarrelGood25(electronSCEta_); //To be tuned
//   electronSCE9_ =  getE9();
//   electronSCE1_ =  getE1();  
// }

// void Electron::fillElectronClass()
// {
//   //Define electron Class. Code Taken from F. Ferri
//   // first look whether it's in crack, barrel or endcap
//   electronClass_ = 0;
//   int prefix=0;
  
//   if (isInCrack()) {
//     //    std::cout << "Electron is in crack! " << electronSCEta_ << std::endl;
//     electronClass_ = 4;
//   } else if (isInBarrel()) {
//     prefix = 0;
//   } else if (isInEndcap()) {
//     prefix = 10;
//   } else {
//     electronClass_ = -1;
//     cout << "Electron: Undefined electron, eta = " << 
//       electronSCEta_ << "!!!!" << endl;
//   }
  

//   //Not in Crack or not undentified
//   if (electronClass_ != -1 && electronClass_ != 4)
//     {
//       //Determining electron class
      
//       // then decide to which class it belongs
//       float p0 = 7.20583e-04;
//       float p1 = 9.20913e-02;
//       float p2 = 8.97269e+00;
      
//       float x  = electron_Tr_Pmag_; 
      
//       float peak = p0 + p1/(x-p2);
      
//       float Phi = electronSCPhi_;
//       float Phi_ref = electron_Tr_Out_Pphi_;
//       if (Phi<0) Phi = 2*TMath::Pi() + Phi;
//       if (Phi_ref<0) Phi_ref = 2*TMath::Pi() + Phi_ref;
//       float DPhi = Phi- Phi_ref;
      
//       //Golden electrons
//       if ( electronSCNClus_ == 1 &&  
// 	   (x - electronSCEnocorr_)/x < 0.1 &&
// 	   fabs(DPhi - peak) < 0.15 &&
// 	   fabs(x - electron_Tr_Out_Pmag_)/x < 0.2) {
	
// 	electronClass_ = prefix + 0;
// 	//High bremming electrons
//       } else if (electronSCNClus_ == 1 &&
// 		 fabs(x - electron_Tr_Out_Pmag_)/x > 0.5 &&
// 		 fabs(x - electronSCEnocorr_)/x < 0.1) {
	
// 	electronClass_ =  prefix + 1;
// 	//Narrow electrons
//       } else if (electronSCNClus_ == 1 &&
// 		 fabs(x - electronSCEnocorr_)/x < 0.1) {
// 	electronClass_ =  prefix + 2;
// 	//Showering electrons	
//       } else {   
// 	electronClass_ = prefix + 3;
//       }
//     }
// }

// void Electron::fillCorrections()
// {
//   if (electronClass_ >= 0 && electronClass_ < 5) 
//     {    // barrel
//       int nCryGT2Sigma = nCrystalsGT2Sigma(theSuperClusters[0].seed_,0.03);
//       electronSCNCryGT2Sigma_ = nCryGT2Sigma;
      
//       if (electronClass_ == 0 ||  electronClass_ == 1 || electronClass_ == 2) 
// 	{
// 	  //	  if ( correctionType == "Full")
// 	  electronSCEfullCorr_ = theSuperClusters[0].energy_/fNCrystals(nCryGT2Sigma)/fEtaBarrelGood(electronSCEta_);
// 	  // 	  else if ( correctionType == "NCryOnly")
// 	  // 	    electronSCEfullCorr_ = theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma);
// 	  // 	  else if ( correctionType == "")
// 	  // 	    electronSCEfullCorr_ = theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma)/fEtaBarrelGood(electronSCEta_);
	  
// 	  electronSCEnoEtaCorr_ = theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma);
// 	} else if (electronClass_ == 3) 
// 	{
// 	  float bremsEnergy = electronSCEnocorr_ - theSuperClusters[0].seed_->energy_;
// 	  //	    if ( correctionType == "Full")
// 	  electronSCEfullCorr_ = (theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma) + bremsEnergy)/
// 	    fEtaBarrelBad(electronSCEta_);
// 	  // 	    else if ( correctionType == "NCryOnly")
// 	  // 	      electronSCEfullCorr_ = (theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma) + bremsEnergy);
// 	  // 	    else if ( correctionType == "")
// 	  // 	      electronSCEfullCorr_ = (theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma) + bremsEnergy)/
// 	  // 		fEtaBarrelBad(electronSCEta_);
// 	  electronSCEnoEtaCorr_ = (theSuperClusters[0].seed_->energy_/fNCrystals(nCryGT2Sigma) + bremsEnergy);
// 	} 
//       else 
// 	{
// 	  // cracks, newEnergy = oldEnergy;
// 	  electronSCEfullCorr_ = electronSCE_;
// 	}
      
//     } 
//   else 
//     {   // endcap, if in crack do nothing, otherwise just correct for eta effect
//       if (electronClass_ == 4)  electronSCEfullCorr_ = electronSCE_; 
//       else electronSCEfullCorr_ = electronSCE_/fEtaEndcap(electronSCEta_);
//       electronSCEnoEtaCorr_ = electronSCE_;
//       // something wroing with fEtaEndcap, so dont correct
//       // else electronSCEfullCorr_ = myElectronCandidate->getSuperClusterEnergy();
//     }
// }
