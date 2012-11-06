// File: MuonMETAlgo.cc
// Description:  see MuonMETAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include <vector>
#include "RecoMET/METAlgorithms/interface/MuonMETAlgo.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"


using namespace std;
using namespace reco;

typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;

CaloMET MuonMETAlgo::makeMET (const CaloMET& fMet, 
			      double fSumEt, 
			      const std::vector<CorrMETData>& fCorrections, 
			      const MET::LorentzVector& fP4) {
  return CaloMET (fMet.getSpecific (), fSumEt, fCorrections, fP4, fMet.vertex ());
}
  
  
  
MET MuonMETAlgo::makeMET (const MET& fMet, 
			  double fSumEt, 
			  const std::vector<CorrMETData>& fCorrections, 
			  const MET::LorentzVector& fP4) {
  return MET (fSumEt, fCorrections, fP4, fMet.vertex ()); 
}

 

template <class T> void MuonMETAlgo::MuonMETAlgo_run(const edm::View<reco::Muon>& inputMuons,
						     const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
						     const edm::View<T>& v_uncorMET,
						     std::vector<T>* v_corMET) {
  T uncorMETObj = v_uncorMET.front();
  
  double uncorMETX = uncorMETObj.px();
  double uncorMETY = uncorMETObj.py();
  
  double corMETX = uncorMETX;
  double corMETY = uncorMETY;
  
  CorrMETData delta;
  double sumMuPx    = 0.;
  double sumMuPy    = 0.;
  double sumMuPt    = 0.;
  double sumMuDepEx = 0.;
  double sumMuDepEy = 0.;
  double sumMuDepEt = 0.;

  unsigned int nMuons = inputMuons.size();
  for(unsigned int iMu = 0; iMu<nMuons; iMu++) {
    const reco::Muon *mu = &inputMuons[iMu]; //new
    reco::MuonMETCorrectionData muCorrData = (vm_muCorrData)[inputMuons.refAt(iMu)];
    int flag   = muCorrData.type();      
    float deltax = muCorrData.corrX();      
    float deltay = muCorrData.corrY();      
            
    LorentzVector mup4;
    if (flag == 0) //this muon is not used to correct the MET
      continue;

    //if we're here, then the muon should be used to correct the MET using the default fit
    mup4 = mu->p4();
    
    sumMuPx    += mup4.px();
    sumMuPy    += mup4.py();
    sumMuPt    += mup4.pt();
    sumMuDepEx += deltax;
    sumMuDepEy += deltay;
    sumMuDepEt += sqrt(deltax*deltax + deltay*deltay);
    corMETX    = corMETX - mup4.px() + deltax;
    corMETY    = corMETY - mup4.py() + deltay;
  
  }
  delta.mex = sumMuDepEx - sumMuPx;
  delta.mey = sumMuDepEy - sumMuPy;
  delta.sumet = sumMuPt - sumMuDepEt;
  MET::LorentzVector correctedMET4vector(corMETX, corMETY, 0., sqrt(corMETX*corMETX + corMETY*corMETY));
  std::vector<CorrMETData> corrections = uncorMETObj.mEtCorr();
  corrections.push_back(delta);
   
  T result = makeMET(uncorMETObj, uncorMETObj.sumEt() + delta.sumet, corrections, correctedMET4vector);
  v_corMET->push_back(result);
}

void MuonMETAlgo::GetMuDepDeltas(const reco::Muon* inputMuon,
				  TrackDetMatchInfo& info,
				  bool useTrackAssociatorPositions,
				  bool useRecHits,
				  bool useHO,
				  double towerEtThreshold,
				  double& deltax, double& deltay,
				  double Bfield) {
  
        
  bool useAverage = false;
  //decide whether or not we want to correct on average based 
  //on isolation information from the muon
  double sumPt   = inputMuon->isIsolationValid()? inputMuon->isolationR03().sumPt       : 0.0;
  double sumEtEcal = inputMuon->isIsolationValid() ? inputMuon->isolationR03().emEt     : 0.0;
  double sumEtHcal    = inputMuon->isIsolationValid() ? inputMuon->isolationR03().hadEt : 0.0;
    
  if(sumPt > 3 || sumEtEcal + sumEtHcal > 5) useAverage = true;
  
  //get the energy using TrackAssociator if
  //the muon turns out to be isolated
  MuonMETInfo muMETInfo;
  muMETInfo.useAverage = useAverage;
  muMETInfo.useTkAssociatorPositions = useTrackAssociatorPositions;
  muMETInfo.useHO = useHO;
  
  
  TrackRef mu_track;
  if(inputMuon->isGlobalMuon()) {
    mu_track = inputMuon->globalTrack();
  } else if(inputMuon->isTrackerMuon()) {
    mu_track = inputMuon->innerTrack();
  } else 
    mu_track = inputMuon->outerTrack();

  if(useTrackAssociatorPositions) {
    muMETInfo.ecalPos  = info.trkGlobPosAtEcal;
    muMETInfo.hcalPos  = info.trkGlobPosAtHcal;
    muMETInfo.hoPos    = info.trkGlobPosAtHO;
  }
  
  if(!useAverage) {
    
    if(useRecHits) {
      muMETInfo.ecalE = inputMuon->calEnergy().emS9;
      muMETInfo.hcalE = inputMuon->calEnergy().hadS9;
      if(useHO) //muMETInfo.hoE is 0 by default
	muMETInfo.hoE   = inputMuon->calEnergy().hoS9;
    } else {// use Towers (this is the default)
      //only include towers whose Et > 0.5 since 
      //by default the MET only includes towers with Et > 0.5
      std::vector<const CaloTower*> towers = info.crossedTowers;
      for(vector<const CaloTower*>::const_iterator it = towers.begin();
	  it != towers.end(); it++) {
	if((*it)->et() < towerEtThreshold) continue;
	muMETInfo.ecalE += (*it)->emEnergy();
	muMETInfo.hcalE += (*it)->hadEnergy();
	if(useHO)
	  muMETInfo.hoE +=(*it)->outerEnergy();
      }
    }//use Towers
  }
  
  

  //This needs to be fixed!!!!!
  //The tracker has better resolution for pt < 200 GeV
  math::XYZTLorentzVector mup4;
  if(inputMuon->isGlobalMuon()) {
    if(inputMuon->globalTrack()->pt() < 200) {
      mup4 = LorentzVector(inputMuon->innerTrack()->px(), inputMuon->innerTrack()->py(),
			   inputMuon->innerTrack()->pz(), inputMuon->innerTrack()->p());
    } else {
      mup4 = LorentzVector(inputMuon->globalTrack()->px(), inputMuon->globalTrack()->py(),
			   inputMuon->globalTrack()->pz(), inputMuon->globalTrack()->p());
    }	
  } else if(inputMuon->isTrackerMuon()) {
    mup4 = LorentzVector(inputMuon->innerTrack()->px(), inputMuon->innerTrack()->py(),
			 inputMuon->innerTrack()->pz(), inputMuon->innerTrack()->p());
  } else 
    mup4 = LorentzVector(inputMuon->outerTrack()->px(), inputMuon->outerTrack()->py(),
			 inputMuon->outerTrack()->pz(), inputMuon->outerTrack()->p());
  
  
  //call function that does the work 
  correctMETforMuon(deltax, deltay, Bfield, inputMuon->charge(),
		    mup4, inputMuon->vertex(),
		    muMETInfo);
}
   
//----------------------------------------------------------------------------

void MuonMETAlgo::correctMETforMuon(double& deltax, double& deltay, double bfield, int muonCharge,
				    math::XYZTLorentzVector muonP4,math::XYZPoint muonVertex,
				    MuonMETInfo& muonMETInfo) {
  
  double mu_p     = muonP4.P();
  double mu_pt    = muonP4.Pt();
  double mu_phi   = muonP4.Phi();
  double mu_eta   = muonP4.Eta();
  double mu_vz    = muonVertex.z()/100.;
  double mu_pz    = muonP4.Pz();
  
  double ecalPhi, ecalTheta;
  double hcalPhi, hcalTheta;
  double hoPhi, hoTheta;
  

  //should always be false for FWLite
  //unless you want to supply co-ordinates at 
  //the calorimeter sub-detectors yourself
  if(muonMETInfo.useTkAssociatorPositions) {
    ecalPhi   = muonMETInfo.ecalPos.Phi();
    ecalTheta = muonMETInfo.ecalPos.Theta();
    hcalPhi   = muonMETInfo.hcalPos.Phi();
    hcalTheta = muonMETInfo.hcalPos.Theta();
    hoPhi     = muonMETInfo.hoPos.Phi();
    hoTheta   = muonMETInfo.hoPos.Theta();
  } else {
    
    /*
      use the analytical solution for the
      intersection of a helix with a cylinder
      to find the positions of the muon
      at the various calo surfaces
    */
    
    //radii of subdetectors in meters
    double rEcal = 1.290;
    double rHcal = 1.9;
    double rHo   = 3.82;
    if(abs(mu_eta) > 0.3) rHo = 4.07;
    //distance from the center of detector to face of Ecal
    double zFaceEcal = 3.209;
    if(mu_eta < 0 ) zFaceEcal = -1*zFaceEcal;
    //distance from the center of detector to face of Hcal
    double zFaceHcal = 3.88;
    if(mu_eta < 0 ) zFaceHcal = -1*zFaceHcal;    
    
    //now we have to get Phi
    //bending radius of the muon (units are meters)
    double bendr = mu_pt*1000/(300*bfield);

    double tb_ecal = TMath::ACos(1-rEcal*rEcal/(2*bendr*bendr)); //helix time interval parameter
    double tb_hcal = TMath::ACos(1-rHcal*rHcal/(2*bendr*bendr)); //helix time interval parameter
    double tb_ho   = TMath::ACos(1-rHo*rHo/(2*bendr*bendr));     //helix time interval parameter
    double xEcal,yEcal,zEcal;
    double xHcal,yHcal,zHcal; 
    double xHo, yHo,zHo;
    //Ecal
    //in the barrel and if not a looper
    if(fabs(mu_pz*bendr*tb_ecal/mu_pt+mu_vz) < fabs(zFaceEcal) && rEcal < 2*bendr) { 
      xEcal = bendr*(TMath::Sin(tb_ecal+mu_phi)-TMath::Sin(mu_phi));
      yEcal = bendr*(-TMath::Cos(tb_ecal+mu_phi)+TMath::Cos(mu_phi));
      zEcal = bendr*tb_ecal*mu_pz/mu_pt + mu_vz;
    } else { //endcap 
      if(mu_pz > 0) {
        double te_ecal = (fabs(zFaceEcal) - mu_vz)*mu_pt/(bendr*mu_pz);
        xEcal = bendr*(TMath::Sin(te_ecal+mu_phi) - TMath::Sin(mu_phi));
        yEcal = bendr*(-TMath::Cos(te_ecal+mu_phi) + TMath::Cos(mu_phi));
        zEcal = fabs(zFaceEcal);
      } else {
        double te_ecal = -(fabs(zFaceEcal) + mu_vz)*mu_pt/(bendr*mu_pz);
        xEcal = bendr*(TMath::Sin(te_ecal+mu_phi) - TMath::Sin(mu_phi));
	yEcal = bendr*(-TMath::Cos(te_ecal+mu_phi) + TMath::Cos(mu_phi));
        zEcal = -fabs(zFaceEcal);
      }
    }

    //Hcal
    if(fabs(mu_pz*bendr*tb_hcal/mu_pt+mu_vz) < fabs(zFaceHcal) && rEcal < 2*bendr) { //in the barrel
      xHcal = bendr*(TMath::Sin(tb_hcal+mu_phi)-TMath::Sin(mu_phi));
      yHcal = bendr*(-TMath::Cos(tb_hcal+mu_phi)+TMath::Cos(mu_phi));
      zHcal = bendr*tb_hcal*mu_pz/mu_pt + mu_vz;
    } else { //endcap 
      if(mu_pz > 0) {
        double te_hcal = (fabs(zFaceHcal) - mu_vz)*mu_pt/(bendr*mu_pz);
        xHcal = bendr*(TMath::Sin(te_hcal+mu_phi) - TMath::Sin(mu_phi));
        yHcal = bendr*(-TMath::Cos(te_hcal+mu_phi) + TMath::Cos(mu_phi));
        zHcal = fabs(zFaceHcal);
      } else {
        double te_hcal = -(fabs(zFaceHcal) + mu_vz)*mu_pt/(bendr*mu_pz);
        xHcal = bendr*(TMath::Sin(te_hcal+mu_phi) - TMath::Sin(mu_phi));
        yHcal = bendr*(-TMath::Cos(te_hcal+mu_phi) + TMath::Cos(mu_phi));
        zHcal = -fabs(zFaceHcal);
      }
    }
    
    //Ho - just the barrel
    xHo = bendr*(TMath::Sin(tb_ho+mu_phi)-TMath::Sin(mu_phi));
    yHo = bendr*(-TMath::Cos(tb_ho+mu_phi)+TMath::Cos(mu_phi));
    zHo = bendr*tb_ho*mu_pz/mu_pt + mu_vz;  
    
    ecalTheta = TMath::ACos(zEcal/sqrt(pow(xEcal,2) + pow(yEcal,2)+pow(zEcal,2)));
    ecalPhi   = atan2(yEcal,xEcal);
    hcalTheta = TMath::ACos(zHcal/sqrt(pow(xHcal,2) + pow(yHcal,2)+pow(zHcal,2)));
    hcalPhi   = atan2(yHcal,xHcal);
    hoTheta   = TMath::ACos(zHo/sqrt(pow(xHo,2) + pow(yHo,2)+pow(zHo,2)));
    hoPhi     = atan2(yHo,xHo);

    //2d radius in x-y plane
    double r2dEcal = sqrt(pow(xEcal,2)+pow(yEcal,2));
    double r2dHcal = sqrt(pow(xHcal,2)+pow(yHcal,2));
    double r2dHo   = sqrt(pow(xHo,2)  +pow(yHo,2));
    
    /*
      the above prescription is for right handed helicies only
      Positively charged muons trace a left handed helix
      so we correct for that 
    */
    if(muonCharge > 0) {
         
      //Ecal
      double dphi = mu_phi - ecalPhi;
      if(fabs(dphi) > TMath::Pi()) 
        dphi = 2*TMath::Pi() - fabs(dphi);
      ecalPhi = mu_phi - fabs(dphi);
      if(fabs(ecalPhi) > TMath::Pi()) {
        double temp = 2*TMath::Pi() - fabs(ecalPhi);
        ecalPhi = -1*temp*ecalPhi/fabs(ecalPhi);
      }
      xEcal = r2dEcal*TMath::Cos(ecalPhi);
      yEcal = r2dEcal*TMath::Sin(ecalPhi);
      
      //Hcal
      dphi = mu_phi - hcalPhi;
      if(fabs(dphi) > TMath::Pi()) 
        dphi = 2*TMath::Pi() - fabs(dphi);
      hcalPhi = mu_phi - fabs(dphi);
      if(fabs(hcalPhi) > TMath::Pi()) {
        double temp = 2*TMath::Pi() - fabs(hcalPhi);
	hcalPhi = -1*temp*hcalPhi/fabs(hcalPhi);
      }
      xHcal = r2dHcal*TMath::Cos(hcalPhi);
      yHcal = r2dHcal*TMath::Sin(hcalPhi);

         
      //Ho
      dphi = mu_phi - hoPhi;
      if(fabs(dphi) > TMath::Pi())
        dphi = 2*TMath::Pi() - fabs(dphi);
      hoPhi = mu_phi - fabs(dphi);
      if(fabs(hoPhi) > TMath::Pi()) {
        double temp = 2*TMath::Pi() - fabs(hoPhi);
        hoPhi = -1*temp*hoPhi/fabs(hoPhi);
      }
      xHo = r2dHo*TMath::Cos(hoPhi);
      yHo = r2dHo*TMath::Sin(hoPhi);
      
    }
  }
  
  //for isolated muons
  if(!muonMETInfo.useAverage) {
    
    double mu_Ex =  muonMETInfo.ecalE*sin(ecalTheta)*cos(ecalPhi)
      + muonMETInfo.hcalE*sin(hcalTheta)*cos(hcalPhi)
      + muonMETInfo.hoE*sin(hoTheta)*cos(hoPhi);
    double mu_Ey =  muonMETInfo.ecalE*sin(ecalTheta)*sin(ecalPhi)
      + muonMETInfo.hcalE*sin(hcalTheta)*sin(hcalPhi)
      + muonMETInfo.hoE*sin(hoTheta)*sin(hoPhi);

    deltax += mu_Ex;
    deltay += mu_Ey;
    
  } else { //non-isolated muons - derive the correction
    
    //dE/dx in matter for iron:
    //-(11.4 + 0.96*fabs(log(p0*2.8)) + 0.033*p0*(1.0 - pow(p0, -0.33)) )*1e-3
    //from http://cmslxr.fnal.gov/lxr/source/TrackPropagation/SteppingHelixPropagator/src/SteppingHelixPropagator.ccyes,
    //line ~1100
    //normalisation is at 50 GeV
    double dEdx_normalization = -(11.4 + 0.96*fabs(log(50*2.8)) + 0.033*50*(1.0 - pow(50, -0.33)) )*1e-3;
    double dEdx_numerator     = -(11.4 + 0.96*fabs(log(mu_p*2.8)) + 0.033*mu_p*(1.0 - pow(mu_p, -0.33)) )*1e-3;
    
    double temp = 0.0;
    
    if(muonMETInfo.useHO) {
      //for the Towers, with HO
      if(fabs(mu_eta) < 0.2)
	temp = 2.75*(1-0.00003*mu_p);
      if(fabs(mu_eta) > 0.2 && fabs(mu_eta) < 1.0)
	temp = (2.38+0.0144*fabs(mu_eta))*(1-0.0003*mu_p);
      if(fabs(mu_eta) > 1.0 && fabs(mu_eta) < 1.3)
	temp = 7.413-5.12*fabs(mu_eta);
      if(fabs(mu_eta) > 1.3)
	temp = 2.084-0.743*fabs(mu_eta);
    } else {
      if(fabs(mu_eta) < 1.0)
	temp = 2.33*(1-0.0004*mu_p);
      if(fabs(mu_eta) > 1.0 && fabs(mu_eta) < 1.3)
	temp = (7.413-5.12*fabs(mu_eta))*(1-0.0003*mu_p);
      if(fabs(mu_eta) > 1.3)
	temp = 2.084-0.743*fabs(mu_eta);
    }

    double dep = temp*dEdx_normalization/dEdx_numerator;
    if(dep < 0.5) dep = 0;
    //use the average phi of the 3 subdetectors
    if(fabs(mu_eta) < 1.3) {
      deltax += dep*cos((ecalPhi+hcalPhi+hoPhi)/3);
      deltay += dep*sin((ecalPhi+hcalPhi+hoPhi)/3);
    } else {
      deltax += dep*cos( (ecalPhi+hcalPhi)/2);
      deltay += dep*cos( (ecalPhi+hcalPhi)/2);
    }
  }

  
}
//----------------------------------------------------------------------------
void MuonMETAlgo::run(const edm::View<reco::Muon>& inputMuons,
		      const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
		      const edm::View<reco::MET>& uncorMET,
		      METCollection *corMET) {

  MuonMETAlgo_run(inputMuons, vm_muCorrData, uncorMET, corMET);
}

//----------------------------------------------------------------------------
void MuonMETAlgo::run(const edm::View<reco::Muon>& inputMuons,
		      const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
		      const edm::View<reco::CaloMET>& uncorMET,
		      CaloMETCollection *corMET) {
  
  MuonMETAlgo_run(inputMuons, vm_muCorrData, uncorMET, corMET);
  
}


//----------------------------------------------------------------------------
MuonMETAlgo::MuonMETAlgo() {}
//----------------------------------------------------------------------------
  
//----------------------------------------------------------------------------
MuonMETAlgo::~MuonMETAlgo() {}
//----------------------------------------------------------------------------



