// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1EGammaCrystalsEmulatorProducer
//
/**\class L1EGammaCrystalsEmulatorProducer L1EGammaCrystalsEmulatorProducer.cc L1Trigger/L1CaloTrigger/plugins/L1EGammaCrystalsEmulatorProducer.cc
Description: Produces crystal clusters using crystal-level information and hardware constraints
Implementation:
[Notes on implementation]
*/
//
// Original Author: Cecile Caillol
// Created: Tue Aug 10 2018
// $Id$
//
//

// system include files
#include <memory>
#include <array>

// user include files
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iostream>
#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TTree.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "DataFormats/Phase2L1CaloTrig/src/classes.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "L1Trigger/L1CaloTrigger/interface/L1TkElectronTrackMatchAlgo.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalEBTrigPrimProducers/plugins/EcalEBTrigPrimProducer.h"
#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


#include "Geometry/Records/interface/IdealGeometryRecord.h"

// ECAL TPs
#include "SimCalorimetry/EcalEBTrigPrimProducers/plugins/EcalEBTrigPrimProducer.h"
#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"

// HCAL TPs
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

// Output tower collection
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"

bool do_brem=true;

float get_calibration(float pt, float eta){
   if (pt<15){
        if (fabs(eta)<0.05) return 1.284;
        else if (fabs(eta)<0.15) return 1.261;
        else if (fabs(eta)<0.25) return 1.241;
        else if (fabs(eta)<0.35) return 1.260;
        else if (fabs(eta)<0.45) return 1.264;
        else if (fabs(eta)<0.55) return 1.284;
        else if (fabs(eta)<0.65) return 1.270;
        else if (fabs(eta)<0.75) return 1.295;
        else if (fabs(eta)<0.85) return 1.306;
        else if (fabs(eta)<0.95) return 1.292;
        else if (fabs(eta)<1.05) return 1.300;
        else if (fabs(eta)<1.15) return 1.317;
        else if (fabs(eta)<1.25) return 1.331;
        else if (fabs(eta)<1.35) return 1.345;
        else if (fabs(eta)<1.45) return 1.358;
        else return 1.389;
   }
   else if (pt<25){
        if (fabs(eta)<0.05) return 1.204;
        else if (fabs(eta)<0.15) return 1.154;
        else if (fabs(eta)<0.25) return 1.164;
        else if (fabs(eta)<0.35) return 1.150;
        else if (fabs(eta)<0.45) return 1.160;
        else if (fabs(eta)<0.55) return 1.170;
        else if (fabs(eta)<0.65) return 1.167;
        else if (fabs(eta)<0.75) return 1.166;
        else if (fabs(eta)<0.85) return 1.207;
        else if (fabs(eta)<0.95) return 1.166;
        else if (fabs(eta)<1.05) return 1.174;
        else if (fabs(eta)<1.15) return 1.198;
        else if (fabs(eta)<1.25) return 1.210;
        else if (fabs(eta)<1.35) return 1.205;
        else if (fabs(eta)<1.45) return 1.211;
        else return 1.265;
   }
   else if (pt<35){
        if (fabs(eta)<0.05) return 1.154;
        else if (fabs(eta)<0.15) return 1.117;
        else if (fabs(eta)<0.25) return 1.115;
        else if (fabs(eta)<0.35) return 1.124;
        else if (fabs(eta)<0.45) return 1.133;
        else if (fabs(eta)<0.55) return 1.138;
        else if (fabs(eta)<0.65) return 1.121;
        else if (fabs(eta)<0.75) return 1.133;
        else if (fabs(eta)<0.85) return 1.151;
        else if (fabs(eta)<0.95) return 1.134;
        else if (fabs(eta)<1.05) return 1.136;
        else if (fabs(eta)<1.15) return 1.167;
        else if (fabs(eta)<1.25) return 1.165;
        else if (fabs(eta)<1.35) return 1.161;
        else if (fabs(eta)<1.45) return 1.154;
        else return 1.195;
   }
   else if (pt<45){
        if (fabs(eta)<0.05) return 1.135;
        else if (fabs(eta)<0.15) return 1.097;
        else if (fabs(eta)<0.25) return 1.101;
        else if (fabs(eta)<0.35) return 1.095;
        else if (fabs(eta)<0.45) return 1.117;
        else if (fabs(eta)<0.55) return 1.114;
        else if (fabs(eta)<0.65) return 1.106;
        else if (fabs(eta)<0.75) return 1.107;
        else if (fabs(eta)<0.85) return 1.144;
        else if (fabs(eta)<0.95) return 1.113;
        else if (fabs(eta)<1.05) return 1.117;
        else if (fabs(eta)<1.15) return 1.142;
        else if (fabs(eta)<1.25) return 1.137;
        else if (fabs(eta)<1.35) return 1.129;
        else if (fabs(eta)<1.45) return 1.144;
        else return 1.169;
   }
   else if (pt<55){
        if (fabs(eta)<0.05) return 1.105;
        else if (fabs(eta)<0.15) return 1.086;
        else if (fabs(eta)<0.25) return 1.090;
        else if (fabs(eta)<0.35) return 1.081;
        else if (fabs(eta)<0.45) return 1.105;
        else if (fabs(eta)<0.55) return 1.101;
        else if (fabs(eta)<0.65) return 1.093;
        else if (fabs(eta)<0.75) return 1.096;
        else if (fabs(eta)<0.85) return 1.122;
        else if (fabs(eta)<0.95) return 1.098;
        else if (fabs(eta)<1.05) return 1.095;
        else if (fabs(eta)<1.15) return 1.124;
        else if (fabs(eta)<1.25) return 1.120;
        else if (fabs(eta)<1.35) return 1.110;
        else if (fabs(eta)<1.45) return 1.116;
        else return 1.164;
   }
   else if (pt<65){
        if (fabs(eta)<0.05) return 1.106;
        else if (fabs(eta)<0.15) return 1.082;
        else if (fabs(eta)<0.25) return 1.078;
        else if (fabs(eta)<0.35) return 1.077;
        else if (fabs(eta)<0.45) return 1.085;
        else if (fabs(eta)<0.55) return 1.084;
        else if (fabs(eta)<0.65) return 1.082;
        else if (fabs(eta)<0.75) return 1.087;
        else if (fabs(eta)<0.85) return 1.104;
        else if (fabs(eta)<0.95) return 1.086;
        else if (fabs(eta)<1.05) return 1.089;
        else if (fabs(eta)<1.15) return 1.107;
        else if (fabs(eta)<1.25) return 1.107;
        else if (fabs(eta)<1.35) return 1.099;
        else if (fabs(eta)<1.45) return 1.099;
        else return 1.136;
   }
   else if (pt<75){
        if (fabs(eta)<0.05) return 1.086;
        else if (fabs(eta)<0.15) return 1.071;
        else if (fabs(eta)<0.25) return 1.069;
        else if (fabs(eta)<0.35) return 1.068;
        else if (fabs(eta)<0.45) return 1.074;
        else if (fabs(eta)<0.55) return 1.078;
        else if (fabs(eta)<0.65) return 1.073;
        else if (fabs(eta)<0.75) return 1.075;
        else if (fabs(eta)<0.85) return 1.085;
        else if (fabs(eta)<0.95) return 1.077;
        else if (fabs(eta)<1.05) return 1.078;
        else if (fabs(eta)<1.15) return 1.089;
        else if (fabs(eta)<1.25) return 1.090;
        else if (fabs(eta)<1.35) return 1.088;
        else if (fabs(eta)<1.45) return 1.091;
        else return 1.128;
   }
   else if (pt<85){
        if (fabs(eta)<0.05) return 1.075;
        else if (fabs(eta)<0.15) return 1.063;
        else if (fabs(eta)<0.25) return 1.062;
        else if (fabs(eta)<0.35) return 1.064;
        else if (fabs(eta)<0.45) return 1.068;
        else if (fabs(eta)<0.55) return 1.068;
        else if (fabs(eta)<0.65) return 1.067;
        else if (fabs(eta)<0.75) return 1.067;
        else if (fabs(eta)<0.85) return 1.079;
        else if (fabs(eta)<0.95) return 1.066;
        else if (fabs(eta)<1.05) return 1.073;
        else if (fabs(eta)<1.15) return 1.078;
        else if (fabs(eta)<1.25) return 1.080;
        else if (fabs(eta)<1.35) return 1.081;
        else if (fabs(eta)<1.45) return 1.079;
        else return 1.102;
   }
   else{
        if (fabs(eta)<0.05) return 1.053;
        else if (fabs(eta)<0.15) return 1.053;
        else if (fabs(eta)<0.25) return 1.053;
        else if (fabs(eta)<0.35) return 1.051;
        else if (fabs(eta)<0.45) return 1.057;
        else if (fabs(eta)<0.55) return 1.054;
        else if (fabs(eta)<0.65) return 1.053;
        else if (fabs(eta)<0.75) return 1.056;
        else if (fabs(eta)<0.85) return 1.056;
        else if (fabs(eta)<0.95) return 1.058;
        else if (fabs(eta)<1.05) return 1.055;
        else if (fabs(eta)<1.15) return 1.061;
        else if (fabs(eta)<1.25) return 1.060;
        else if (fabs(eta)<1.35) return 1.065;
        else if (fabs(eta)<1.45) return 1.064;
        else return 1.070;
   }
}

float mydeltarv2(float eta1, float eta2, float phi1, float phi2){
   float deta2=(eta1-eta2)*(eta1-eta2);
   float dphi=(phi1-phi2);
   while (dphi>3.14) dphi=dphi-6.28;
   while (dphi<-3.14) dphi=dphi+6.28;
   return sqrt(deta2+dphi*dphi);
}

float getEta_fromL2LinkCardTowerCrystal(int link, int card, int tower, int crystal){
    int etaID=5*(17*((link/4)%2)+(tower%17))+crystal%5;
    float size_cell=2*1.4841/(5*34);
    return etaID*size_cell-1.4841+0.00873;
}

float getPhi_fromL2LinkCardTowerCrystal(int link, int card, int tower, int crystal){
    int phiID=5*((card*24)+(4*(link/8))+(tower/17))+crystal/5;
    float size_cell=2*3.14159/(5*72);
    return phiID*size_cell-3.14259+0.00873;
}

int getCrystal_etaID(float eta){
   float size_cell=2*1.4841/(5*34);
   int etaID=int((eta+1.4841)/size_cell);
   return etaID;
}

int get_insideEta(float eta, float centereta){
   float size_cell=2*1.4841/(5*34);
   if (eta-centereta<-1.0*size_cell/4) return 0;
   else if (eta-centereta<0) return 1;
   else if (eta-centereta<1.0*size_cell/4) return 2;
   else return 3;
}

int get_insidePhi(float phi, float centerphi){
   float size_cell=2*3.14159/(5*72);
   if (phi-centerphi<-1*size_cell/4) return 0;
   else if (phi-centerphi<0) return 1;
   else if (phi-centerphi<1*size_cell/4) return 2;
   else return 3;
}

int convert_L2toL1_link(int link){
   return link%4;
}

int convert_L2toL1_tower(int tower){
   return tower;
}

int convert_L2toL1_card(int card, int link){
   return card*12+link/4;
}

int getCrystal_phiID(float phi){
   float size_cell=2*3.14159/(5*72);
   int phiID=int((phi+3.14159)/size_cell);
   return phiID;
}

int getTower_absoluteEtaID(float eta){
   float size_cell=2*1.4841/(34);
   int etaID=int((eta+1.4841)/size_cell);
   return etaID;
}

int getTower_absolutePhiID(float phi){
   float size_cell=2*3.14159/(72);
   int phiID=int((phi+3.14159)/size_cell);
   return phiID;
}

// absolue IDs range from 0-33
// 0-16 are iEta -17 to -1
// 17 to 33 are iEta 1 to 17
int getToweriEta_fromAbsoluteID(int id){
   if (id < 17) return id - 17; 
   else return id - 16;
}

// absolue IDs range from 0-71.
// To align with detector tower IDs (1 - 72)
// shift all indices by 37 and loop over after 72
int getToweriPhi_fromAbsoluteID(int id){
   if (id < 36) return id + 37;
   else return id - 35;
}


float getTowerEta_fromAbsoluteID(int id){
   float size_cell=2*1.4841/(34);
   float eta=(id*size_cell)-1.4841+0.5*size_cell;
   return eta;
}

float getTowerPhi_fromAbsoluteID(int id){
   float size_cell=2*3.14159/(72);
   float phi=(id*size_cell)-3.14159+0.5*size_cell;
   return phi;
}

int getCrystalIDInTower(int etaID, int phiID){
   return int(5*(phiID%5)+(etaID%5));
}

int getTowerInFullDetector_etaID(float eta){
   float size_cell=2*1.4841/(34);
   int etaID=int((eta+1.4841)/size_cell);
   return etaID;
}

int getTowerInFullDetector_phiID(float phi){
   float size_cell=2*3.14159/(72);
   int phiID=int((phi+3.14159)/size_cell);
   return phiID;
}

int get_towerEta_fromCardTowerInCard(int card, int towerincard){
   return 17*(card%2)+towerincard%17;
}

int get_towerPhi_fromCardTowerInCard(int card, int towerincard){
   return 4*(card/2)+towerincard/17;
}

int get_towerEta_fromCardLinkTower(int card, int link, int tower){
   return 17*(card%2)+tower;
}

int get_towerPhi_fromCardLinkTower(int card, int link, int tower){
   return 4*(card/2)+link;
}

int getTowerID(int etaID, int phiID){
   return int(17*((phiID/5)%4)+(etaID/5)%17);
}

int getTower_phiID(int cluster_phiID){ // Tower ID in card given crystal ID in total detector
   return int((cluster_phiID/5)%4);
}

int getTower_etaID(int cluster_etaID){ // Tower ID in card given crystal ID in total detector
   return int((cluster_etaID/5)%17);
}

int getEtaMax_card(int card){
    int etamax=0;
    if (card%2==0) etamax=17*5-1; // First eta half. 5 crystals in eta in 1 tower.
    else etamax=34*5-1;
    return etamax;
}

int getEtaMin_card(int card){
    int etamin=0;
    if (card%2==0) etamin=0*5; // First eta half. 5 crystals in eta in 1 tower.
    else etamin=17*5;
    return etamin;
}

int getPhiMax_card(int card){
    int phimax=((card/2)+1)*4*5-1; 
    return phimax;
}

int getPhiMin_card(int card){
    int phimin=(card/2)*4*5;
    return phimin;
}

int getEtaMaxTower_card(int card){
    int etamax=0;
    if (card%2==0) etamax=17-1; // First eta half. 5 crystals in eta in 1 tower.
    else etamax=34-1;
    return etamax;
}

int getEtaMinTower_card(int card){
    int etamin=0;
    if (card%2==0) etamin=0; // First eta half. 5 crystals in eta in 1 tower.
    else etamin=17;
    return etamin;
}

int getPhiMaxTower_card(int card){
    int phimax=(card+1)*4-1;
    return phimax;
}

int getPhiMinTower_card(int card){
    int phimin=card*4;
    return phimin;
}

class L1EGCrystalClusterEmulatorProducer : public edm::EDProducer {
   public:
      explicit L1EGCrystalClusterEmulatorProducer(const edm::ParameterSet&);
      ~L1EGCrystalClusterEmulatorProducer();

   private:
      void produce(edm::Event&, const edm::EventSetup&);
      bool passes_he(float pt, float he);
      bool passes_ss(float pt, float ss);
      bool passes_photon(float pt, float pss);
      bool passes_iso(float pt, float iso);
      bool passes_looseTkss(float pt, float ss);
      bool passes_looseTkiso(float pt, float iso);
      float get_calibrate(float uncorr);

      edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> ecalTPEBToken_;
      edm::EDGetTokenT< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hcalTPToken_;
      edm::ESHandle<CaloTPGTranscoder> decoder_;

      edm::ESHandle<CaloGeometry> caloGeometry_;
      const CaloSubdetectorGeometry * ebGeometry;
      const CaloSubdetectorGeometry * hbGeometry;
      edm::ESHandle<HcalTopology> hbTopology;
      const HcalTopology * hcTopology_;

      struct mycluster{
	float c2x2;
	float c2x5;
	float c5x5;
        int cshowershape;
        int cshowershapeloosetk;
        float cvalueshowershape;
	int cphotonshowershape;
	float cpt; // ECAL pt
        int cbrem; // if brem corrections were applied
	float cWeightedEta;
	float cWeightedPhi;
	float ciso; // pt of cluster divided by 7x7 ECAL towers
	float chovere; // 5x5 HCAL towers divided by the ECAL cluster pt
	float craweta; // coordinates between -1.44 and 1.44
	float crawphi; // coordinates between -pi and pi
	float chcal; // 5x5 HCAL towers
	float ceta; // eta ID in the whole detector (between 0 and 5*34-1)
	float cphi; // phi ID in the whole detector (between 0 and 5*72-1)
	int ccrystalid; // crystal ID inside tower (between 0 and 24)
        int cinsidecrystalid;
	int ctowerid; // tower ID inside card (between 0 and 4*17-1)
      };

      bool order_clusters(mycluster c1, mycluster c2){
	  return c1.cpt<c2.cpt;
      }

      class SimpleCaloHit
      {
         public:
            EBDetId id;
            HcalDetId id_hcal;
            GlobalVector position; // As opposed to GlobalPoint, so we can add them (for weighted average)
            float energy=0.;
	    bool used=false;
            bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
            bool isEndcapHit=false; // If using endcap, we won't be using integer crystal indices

         // tool functions
            inline float pt() const{return (position.mag2()>0) ? energy*sin(position.theta()) : 0.;};
            inline float deta(SimpleCaloHit& other) const{return position.eta() - other.position.eta();};
            int dieta(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               if (id.ieta() * other.id.ieta() > 0)
                  return id.ieta()-other.id.ieta();
               return id.ieta()-other.id.ieta()-1;
            };
            inline float dphi(SimpleCaloHit& other) const{return reco::deltaPhi(static_cast<float>(position.phi()), static_cast<float>(other.position.phi()));};
            int diphi(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               // Logic from EBDetId::distancePhi() without the abs()
               int PI = 180;
               int result = id.iphi() - other.id.iphi();
               while (result > PI) result -= 2*PI;
               while (result <= -PI) result += 2*PI;
               return result;
            };
            float distanceTo(SimpleCaloHit& other) const
            {
               // Treat position as a point, measure 3D distance
               // This is used for endcap hits, where we don't have a rectangular mapping
               return (position-other.position).mag();
            };
            bool operator==(SimpleCaloHit& other) const
            {
               if ( id == other.id &&
                    position == other.position &&
                    energy == other.energy &&
                    isEndcapHit == other.isEndcapHit
                  ) return true;

               return false;
            };
      };
            
};

L1EGCrystalClusterEmulatorProducer::L1EGCrystalClusterEmulatorProducer(const edm::ParameterSet& iConfig) :
   ecalTPEBToken_(consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPEB"))),
   hcalTPToken_(consumes< edm::SortedCollection<HcalTriggerPrimitiveDigi> >(iConfig.getParameter<edm::InputTag>("hcalTP")))
{
   produces<l1slhc::L1EGCrystalClusterCollection>("L1EGXtalClusterEmulator");
   produces< BXVector<l1t::EGamma> >("L1EGammaCollectionBXVEmulator");
   produces< L1CaloTowerCollection >("L1CaloTowerCollection");

}


L1EGCrystalClusterEmulatorProducer::~L1EGCrystalClusterEmulatorProducer()
{
}


void L1EGCrystalClusterEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Handle<EcalEBTrigPrimDigiCollection> pcalohits;
   iEvent.getByToken(ecalTPEBToken_,pcalohits);

   iSetup.get<CaloGeometryRecord>().get(caloGeometry_);
   ebGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
   hbGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
   iSetup.get<HcalRecNumberingRecord>().get(hbTopology);
   hcTopology_ = hbTopology.product();
   HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);
   iSetup.get<CaloTPGRecord>().get(decoder_);

//****************************************************************
//******************* Get all the hits ***************************
//****************************************************************

      // Get all the ECAL hits
      iEvent.getByToken(ecalTPEBToken_,pcalohits);
      std::vector<SimpleCaloHit> ecalhits;

      for(auto& hit : *pcalohits.product())
      {
         if(hit.encodedEt() > 0) // hit.encodedEt() returns an int corresponding to 2x the crystal Et
         {
            float et = hit.encodedEt()/8.; // Et is 10 bit, by keeping the ADC saturation Et at 120 GeV it means that you have to divide by 8
            if (et < 0.5) continue; // keep the 500 MeV ET Cut

            auto cell = ebGeometry->getGeometry(hit.id());

            SimpleCaloHit ehit;
            ehit.id = hit.id();
            ehit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
            ehit.energy = et / sin(ehit.position.theta());
            ecalhits.push_back(ehit);
         }
      }

      // Get all the HCAL hits
      std::vector<SimpleCaloHit> hcalhits;
      edm::Handle< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hbhecoll;
      iEvent.getByToken(hcalTPToken_,hbhecoll);
      for (auto& hit : *hbhecoll.product())
      {
         //if ( hit.SOI_compressedEt() == 0 ) continue; 
         float et = decoder_->hcaletValue(hit.id(), hit.t0());
         if ( et <= 0 ) continue; 
         if (!(hcTopology_->validHT(hit.id()))) {
           std::cout << " -- Hcal hit DetID not present in HCAL Geom: " << hit.id() << std::endl;
           continue;
         }
         std::vector<HcalDetId> hcId = theTrigTowerGeometry.detIds(hit.id());
         if (hcId.size() == 0) {
           std::cout << "Cannot find any HCalDetId corresponding to " << hit.id() << std::endl;
           continue;
         }
         if (hcId[0].subdetId() > 1) continue;
         GlobalVector hcal_tp_position = GlobalVector(0., 0., 0.);
         for (auto &hcId_i : hcId) {
           if (hcId_i.subdetId() > 1) continue;
           auto cell = hbGeometry->getGeometry(hcId_i);
           if (cell == 0) continue;
           GlobalVector tmpVector = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
           hcal_tp_position = tmpVector;
           break;
         }
         SimpleCaloHit hhit;
         hhit.id = hit.id();
         hhit.id_hcal = hit.id();
         hhit.position = hcal_tp_position;
         //float et = hit.SOI_compressedEt() / 2.;
         hhit.energy = et / sin(hhit.position.theta());
         hcalhits.push_back(hhit);
      }


//*******************************************************************
//********************** Do layer 1 *********************************
//*******************************************************************

      // Definition of L1 outputs
      //float energy_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      float ECAL_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      float HCAL_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      int iEta_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      int iPhi_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      //float HE_tower_L1Card[4][17][36]; // 36 L1 cards send each 4 links with 17 towers
      float energy_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      int brem_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      int towerID_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      int crystalID_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      //int insideCrystalID_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      int showerShape_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      int showerShapeLooseTk_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters
      //float valueShowerShape_cluster_L1Card[4][3][36];
      int photonShowerShape_cluster_L1Card[4][3][36]; // 36 L1 cards send each 4 links with 3 clusters

      for (int ii=0; ii<4; ++ii){
	 for (int jj=0; jj<17; ++jj){
	    for (int ll=0; ll<36; ++ll){
	       //energy_tower_L1Card[ii][jj][ll]=0;
               ECAL_tower_L1Card[ii][jj][ll]=0;
               HCAL_tower_L1Card[ii][jj][ll]=0;
               iPhi_tower_L1Card[ii][jj][ll]=-999;
               iEta_tower_L1Card[ii][jj][ll]=-999;
	       //HE_tower_L1Card[ii][jj][ll]=0;
	    }
	 }
      }
      for (int ii=0; ii<4; ++ii){
         for (int jj=0; jj<3; ++jj){
            for (int ll=0; ll<36; ++ll){
               energy_cluster_L1Card[ii][jj][ll]=0;
               brem_cluster_L1Card[ii][jj][ll]=0;
               towerID_cluster_L1Card[ii][jj][ll]=0;
               crystalID_cluster_L1Card[ii][jj][ll]=0;
               //insideCrystalID_cluster_L1Card[ii][jj][ll]=0;
	       //valueShowerShape_cluster_L1Card[ii][jj][ll]=0;
	    }
         }
      }

      vector<mycluster> cluster_list[36]; // There is one list of clusters per card. We take the 12 highest pt per card
      vector<mycluster> cluster_list_merged[36]; // After merging the clusters in different regions of a single L1 card

      for (int cc=0; cc<36; ++cc){ // Loop over 36 L1 cards

          for (int nregion=0; nregion<6; ++nregion){ // Loop over 3x4 etaxphi regions to search for max 5 clusters
             int nclusters=0;
             bool build_cluster=true;
	
	     while (nclusters<5 && build_cluster){ // Continue until 5 clusters have been built or there is no cluster left 
	        build_cluster=false;
                SimpleCaloHit centerhit;

                for(const auto& hit : ecalhits)
                {
                   if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)){  // Check that the hit is in the good card
	               if ( getCrystal_etaID(hit.position.eta())<getEtaMin_card(cc)+3*5*(nregion+1) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)+3*5*nregion && !hit.used && hit.pt()>=1.0 && hit.pt() > centerhit.pt() ) // 3 towers x 5 crystals
                       { // Highest hit in good region with pt>1 and not used in any other cluster
                          centerhit = hit;
                          build_cluster=true;
                       }
	           }
                }
                if (build_cluster) nclusters++;

                if (build_cluster && nclusters>0 && nclusters<6){ // Use only the 5 most energetic clusters
                   mycluster mc1;
                   mc1.cpt=0.0;
                   mc1.cWeightedEta=0.0;
                   mc1.cWeightedPhi=0.0;
                   float leftlobe=0;
                   float rightlobe=0;
		   float e5x5=0;
		   float n5x5=0;
                   float e2x5_1=0;
                   float n2x5_1=0;
                   float e2x5_2=0;
                   float n2x5_2=0;
                   float e2x2_1=0;
                   float n2x2_1=0;
                   float e2x2_2=0;
                   float n2x2_2=0;
                   float e2x2_3=0;
                   float n2x2_3=0;
                   float e2x2_4=0;
                   float n2x2_4=0;
                   for(auto& hit : ecalhits)
                   {
		      if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc) && hit.pt()>0 && getCrystal_etaID(hit.position.eta())<getEtaMin_card(cc)+3*5*(nregion+1) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)+3*5*nregion){
                           if (abs(hit.dieta(centerhit))<=1 && hit.diphi(centerhit)>2 && hit.diphi(centerhit)<=7){
                               rightlobe+=hit.pt();
                           }
                           if (abs(hit.dieta(centerhit))<=1 && hit.diphi(centerhit)<-2 && hit.diphi(centerhit)>=-7){
                               leftlobe+=hit.pt();
                           }
			   if (abs(hit.dieta(centerhit))<=2 && abs(hit.diphi(centerhit))<=2){
			       e5x5+=hit.energy;
			       n5x5++;
			   }
                           if ((hit.dieta(centerhit)==1 or hit.dieta(centerhit)==0) && (hit.diphi(centerhit)==1 or hit.diphi(centerhit)==0)){
                               e2x2_1+=hit.energy;
                               n2x2_1++;
                           }
                           if ((hit.dieta(centerhit)==0 or hit.dieta(centerhit)==-1) && (hit.diphi(centerhit)==0 or hit.diphi(centerhit)==1)){
                               e2x2_2+=hit.energy;
                               n2x2_2++;
                           }
                           if ((hit.dieta(centerhit)==0 or hit.dieta(centerhit)==1) && (hit.diphi(centerhit)==0 or hit.diphi(centerhit)==-1)){
                               e2x2_3+=hit.energy;
                               n2x2_3++;
                           }
                           if ((hit.dieta(centerhit)==0 or hit.dieta(centerhit)==-1) && (hit.diphi(centerhit)==0 or hit.diphi(centerhit)==-1)){
                               e2x2_4+=hit.energy;
                               n2x2_4++;
                           }
                           if ((hit.dieta(centerhit)==0 or hit.dieta(centerhit)==1) && abs(hit.diphi(centerhit))<=2){
                               e2x5_1+=hit.energy;
                               n2x5_1++;
                           }
                           if ((hit.dieta(centerhit)==0 or hit.dieta(centerhit)==-1) && abs(hit.diphi(centerhit))<=2){
                               e2x5_2+=hit.energy;
                               n2x5_2++;
                           }
		      }
                      if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)
                           && !hit.used && hit.pt()>0
                           && abs(hit.dieta(centerhit))<=1 && abs(hit.diphi(centerhit))<=2 
                           && getCrystal_etaID(hit.position.eta())<getEtaMin_card(cc)+3*5*(nregion+1) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)+3*5*nregion){ // clusters 3x5 in etaxphi using only the hits in the corresponding card and in the corresponding 3x4 region
                          hit.used=true;
                          mc1.cpt+=hit.pt();
                          mc1.cWeightedEta+=float(hit.pt())*float(hit.position.eta());
                          mc1.cWeightedPhi=mc1.cWeightedPhi+(float(hit.pt())*float(hit.position.phi()));
                      }
                   }
                   if (do_brem && (rightlobe>0.10*mc1.cpt or leftlobe>0.10*mc1.cpt)){
                      for(auto& hit : ecalhits){
                         if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc) && hit.pt()>0 && getCrystal_etaID(hit.position.eta())<getEtaMin_card(cc)+3*5*(nregion+1) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc)+3*5*nregion && !hit.used){
                           if (rightlobe>0.10*mc1.cpt && (leftlobe<0.10*mc1.cpt or rightlobe>leftlobe) && abs(hit.dieta(centerhit))<=1 && hit.diphi(centerhit)>2 && hit.diphi(centerhit)<=7){
                               mc1.cpt+=hit.pt();
                               hit.used=true;
			       mc1.cbrem=1;
                           }
                           if (leftlobe>0.10*mc1.cpt && (rightlobe<0.10*mc1.cpt or leftlobe>=rightlobe) && abs(hit.dieta(centerhit))<=1 && hit.diphi(centerhit)<-2 && hit.diphi(centerhit)>=-7){
                               mc1.cpt+=hit.pt();
                               hit.used=true;
			       mc1.cbrem=1;
                           }

                         }
                      }
                   }
		   mc1.c5x5=e5x5;
		   mc1.c2x5=e2x5_1;
		   if (e2x5_2>mc1.c2x5) mc1.c2x5=e2x5_2;
                   mc1.c2x2=e2x2_1;
                   if (e2x2_2>mc1.c2x2) mc1.c2x2=e2x2_2;
                   if (e2x2_3>mc1.c2x2) mc1.c2x2=e2x2_3;
                   if (e2x2_4>mc1.c2x2) mc1.c2x2=e2x2_4;
		   mc1.cWeightedEta=mc1.cWeightedEta/mc1.cpt;
                   mc1.cWeightedPhi=mc1.cWeightedPhi/mc1.cpt;
		   mc1.ceta=getCrystal_etaID(centerhit.position.eta());
                   mc1.cphi=getCrystal_phiID(centerhit.position.phi());
                   mc1.crawphi=centerhit.position.phi();
                   mc1.craweta=centerhit.position.eta();
                   cluster_list[cc].push_back(mc1);
                } // End if 5 clusters per region
	     } // End while to find the 5 clusters
          } // End loop over regions to search for clusters
          std::sort(begin(cluster_list[cc]), end(cluster_list[cc]), [](mycluster a, mycluster b){return a.cpt > b.cpt;});

	  // Merge clusters from different regions
	  for (unsigned int jj=0; jj<unsigned(cluster_list[cc].size()); ++jj){
	     for (unsigned int kk=jj+1; kk<unsigned(cluster_list[cc].size()); ++kk){
		 if (fabs(cluster_list[cc][jj].ceta-cluster_list[cc][kk].ceta)<2 && fabs(cluster_list[cc][jj].cphi-cluster_list[cc][kk].cphi)<2){ //Diagonale + exact neighbors
		     if (cluster_list[cc][kk].cpt>cluster_list[cc][jj].cpt){
			cluster_list[cc][kk].cpt+=cluster_list[cc][jj].cpt;
                        cluster_list[cc][kk].c5x5+=cluster_list[cc][jj].c5x5;
                        cluster_list[cc][kk].c2x5+=cluster_list[cc][jj].c2x5; 
			cluster_list[cc][jj].cpt=0;
                        cluster_list[cc][jj].c5x5=0;
                        cluster_list[cc][jj].c2x5=0;
                        cluster_list[cc][jj].c2x2=0;
		     }
		     else{
                        cluster_list[cc][jj].cpt+=cluster_list[cc][kk].cpt;
                        cluster_list[cc][jj].c5x5+=cluster_list[cc][kk].c5x5;
                        cluster_list[cc][jj].c2x5+=cluster_list[cc][kk].c2x5; 
                        cluster_list[cc][kk].cpt=0;
                        cluster_list[cc][kk].c2x2=0;
                        cluster_list[cc][kk].c2x5=0;
                        cluster_list[cc][kk].c5x5=0;
		     }
		 }
	     }
	     if (cluster_list[cc][jj].cpt>0){
		 cluster_list[cc][jj].cpt=cluster_list[cc][jj].cpt*0.963*get_calibration(cluster_list[cc][jj].cpt,cluster_list[cc][jj].ceta); //Mark's calibration as a function of eta and pt
	         cluster_list_merged[cc].push_back(cluster_list[cc][jj]);
	      }
	  }
          std::sort(begin(cluster_list_merged[cc]), end(cluster_list_merged[cc]), [](mycluster a, mycluster b){return a.cpt > b.cpt;});

	  // Fill cluster information in the arrays. We keep max 12 clusters (distributed between 4 links)
          for (unsigned int jj=0; jj<unsigned(cluster_list_merged[cc].size()) && jj<12; ++jj){
              crystalID_cluster_L1Card[jj%4][jj/4][cc]=getCrystalIDInTower(cluster_list_merged[cc][jj].ceta,cluster_list_merged[cc][jj].cphi);
              towerID_cluster_L1Card[jj%4][jj/4][cc]=getTowerID(cluster_list_merged[cc][jj].ceta,cluster_list_merged[cc][jj].cphi);
              energy_cluster_L1Card[jj%4][jj/4][cc]=cluster_list_merged[cc][jj].cpt;
              brem_cluster_L1Card[jj%4][jj/4][cc]=cluster_list_merged[cc][jj].cbrem;
	      //valueShowerShape_cluster_L1Card[jj%4][jj/4][cc]=(cluster_list_merged[cc][jj].c2x5/cluster_list_merged[cc][jj].c5x5);
	      if (passes_ss(cluster_list_merged[cc][jj].cpt,cluster_list_merged[cc][jj].c2x5/cluster_list_merged[cc][jj].c5x5)) showerShape_cluster_L1Card[jj%4][jj/4][cc]=1;
	      else showerShape_cluster_L1Card[jj%4][jj/4][cc]=0;
              if (passes_looseTkss(cluster_list_merged[cc][jj].cpt,cluster_list_merged[cc][jj].c2x5/cluster_list_merged[cc][jj].c5x5)) showerShapeLooseTk_cluster_L1Card[jj%4][jj/4][cc]=1;
              else showerShapeLooseTk_cluster_L1Card[jj%4][jj/4][cc]=0;
	      if (passes_photon(cluster_list_merged[cc][jj].cpt,cluster_list_merged[cc][jj].c2x2/cluster_list_merged[cc][jj].c2x5)) photonShowerShape_cluster_L1Card[jj%4][jj/4][cc]=1;
              else photonShowerShape_cluster_L1Card[jj%4][jj/4][cc]=0;
              //insideCrystalID_cluster_L1Card[jj%4][jj/4][cc]=get_insideEta(cluster_list_merged[cc][jj].cWeightedEta,cluster_list_merged[cc][jj].craweta);
              //insideCrystalID_cluster_L1Card[jj%4][jj/4][cc]+=4*get_insidePhi(cluster_list_merged[cc][jj].cWeightedPhi,cluster_list_merged[cc][jj].crawphi);
          }

          // Loop over calo ecal hits to get the ECAL towers. Take only hits that have not been used to make clusters
          for(const auto& hit : ecalhits)
          {
	    //cout<<"ieta, iphi: "<<hit.id.ieta()<<" "<<hit.id.iphi()<<endl;
            if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc) && !hit.used){ // Take all the hits inside the card that have not been used yet 
                for (int jj=0; jj<4; ++jj){ // loop over 4 links per card
                   if ((getCrystal_phiID(hit.position.phi())/5)%4==jj){ // Go to ID tower modulo 4
             	      for (int ii=0; ii<17; ++ii){
			 //Apply Mark's calibration at the same time (row of the lowest pT, as a function of eta)
            	         if ((getCrystal_etaID(hit.position.eta())/5)%17==ii){ 
			     ECAL_tower_L1Card[jj][ii][cc]+=hit.pt()*0.963*get_calibration(0,hit.position.eta());
                             iEta_tower_L1Card[jj][ii][cc]=getTower_absoluteEtaID(hit.position.eta());//hit.id.ieta();
                             iPhi_tower_L1Card[jj][ii][cc]=getTower_absolutePhiID(hit.position.phi());//hit.id.iphi();
			 }
            	      } // end of loop over eta towers
                   } 
                } // end of loop over phi links
             } // end of check if inside card
          } // end of loop over hits to build towers

          // Loop over hcal hits to get the HCAL towers.
          for(const auto& hit : hcalhits)
          {
            //cout<<"H ieta, iphi: "<<hit.id.ieta()<<" "<<hit.id.iphi()<<endl;
            if (getCrystal_phiID(hit.position.phi())<=getPhiMax_card(cc) && getCrystal_phiID(hit.position.phi())>=getPhiMin_card(cc) && getCrystal_etaID(hit.position.eta())<=getEtaMax_card(cc) && getCrystal_etaID(hit.position.eta())>=getEtaMin_card(cc) && hit.pt()>0){ 
                for (int jj=0; jj<4; ++jj){
                   if ((getCrystal_phiID(hit.position.phi())/5)%4==jj){
                      for (int ii=0; ii<17; ++ii){
                         if ((getCrystal_etaID(hit.position.eta())/5)%17==ii){ 
			     HCAL_tower_L1Card[jj][ii][cc]+=hit.pt();
			     iEta_tower_L1Card[jj][ii][cc]=getTower_absoluteEtaID(hit.position.eta());//hit.id.ieta();
                             iPhi_tower_L1Card[jj][ii][cc]=getTower_absolutePhiID(hit.position.phi());//hit.id.iphi();
			 }
                      } // end of loop over eta towers
                   } 
                } // end of loop over phi links
             } // end of check if inside card
          } // end of loop over hits to build towers

          // Give back energy of not used clusters to the towers (if there are more than 12 clusters)
          for (unsigned int kk=12; kk<cluster_list_merged[cc].size(); ++kk){
	      if (cluster_list_merged[cc][kk].cpt>0){
                 ECAL_tower_L1Card[getTower_phiID(cluster_list_merged[cc][kk].cphi)][getTower_etaID(cluster_list_merged[cc][kk].ceta)][cc]+=cluster_list_merged[cc][kk].cpt;
	      }
          }
     } //End of loop over cards

     /*
     // From ECAL and HCAL energies of the towers compute H/E and H+E
      for (int ii=0; ii<4; ++ii){
         for (int jj=0; jj<17; ++jj){
            for (int ll=0; ll<36; ++ll){
               energy_tower_L1Card[ii][jj][ll]=ECAL_tower_L1Card[ii][jj][ll]+HCAL_tower_L1Card[ii][jj][ll];
               if (ECAL_tower_L1Card[ii][jj][ll]!=0) HE_tower_L1Card[ii][jj][ll]=HCAL_tower_L1Card[ii][jj][ll]/ECAL_tower_L1Card[ii][jj][ll];
	       else HE_tower_L1Card[ii][jj][ll]=-1;
            }
         }
      }*/

     //cout<<"Default printout: "<<energy_tower_L1Card[0][0][0]<<" "<<insideCrystalID_cluster_L1Card[0][0][0]<<" "<<crystalID_cluster_L1Card[0][0][0]<<" "<<towerID_cluster_L1Card[0][0][0]<<" "<<energy_cluster_L1Card[0][0][0]<<" "<<HE_tower_L1Card[0][0][0]<<" "<<showerShape_cluster_L1Card[0][0][0]<<" "<<photonShowerShape_cluster_L1Card[0][0][0]<<" "<<valueShowerShape_cluster_L1Card[0][0][0]<<endl;

//*********************************************************
//******************** Do layer 2 *************************
//*********************************************************

      // Definition of L2 outputs
      //float energy_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      //float HE_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      float HCAL_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      float ECAL_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      int iEta_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      int iPhi_tower_L2Card[48][17][3]; // 3 L2 cards send each 48 links with 17 towers
      float energy_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      float brem_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      int towerID_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      int crystalID_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      //int insideCrystalID_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      float isolation_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      float HE_cluster_L2Card[48][2][3]; // 3 L2 cards send each 48 links with 2 clusters
      int showerShape_cluster_L2Card[48][2][3];
      int showerShapeLooseTk_cluster_L2Card[48][2][3];
      //float valueShowerShape_cluster_L2Card[48][2][3];
      int photonShowerShape_cluster_L2Card[48][2][3];

      for (int ii=0; ii<48; ++ii){
         for (int jj=0; jj<17; ++jj){
            for (int ll=0; ll<3; ++ll){
               //energy_tower_L2Card[ii][jj][ll]=0;
               //HE_tower_L2Card[ii][jj][ll]=0;
               HCAL_tower_L2Card[ii][jj][ll]=0;
               ECAL_tower_L2Card[ii][jj][ll]=0;
               iEta_tower_L2Card[ii][jj][ll]=0;
               iPhi_tower_L2Card[ii][jj][ll]=0;
            }
         }
      }
      for (int ii=0; ii<48; ++ii){
         for (int jj=0; jj<2; ++jj){
            for (int ll=0; ll<3; ++ll){
               energy_cluster_L2Card[ii][jj][ll]=0;
               brem_cluster_L2Card[ii][jj][ll]=0;
               towerID_cluster_L2Card[ii][jj][ll]=0;
               crystalID_cluster_L2Card[ii][jj][ll]=0;
               //insideCrystalID_cluster_L2Card[ii][jj][ll]=0;
               isolation_cluster_L2Card[ii][jj][ll]=0;
               HE_cluster_L2Card[ii][jj][ll]=0;
               photonShowerShape_cluster_L2Card[ii][jj][ll]=0;
               showerShape_cluster_L2Card[ii][jj][ll]=0;
               showerShapeLooseTk_cluster_L2Card[ii][jj][ll]=0;
	       //valueShowerShape_cluster_L2Card[ii][jj][ll]=0;
            }
         }
      }

      vector<mycluster> cluster_list_L2[36]; // There is one list of clusters per equivalent of L1 card. We take the 8 highest pt.

      // Merge clusters on the phi edges
      for (int ii=0; ii<18; ++ii){ // 18 borders in phi
	 for (int jj=0; jj<2; ++jj){ // 2 eta bins
	     int card_left=2*ii+jj;
	     int card_right=2*ii+jj+2;
	     if (card_right>35) card_right=card_right-36;
	     for (int kk=0; kk<12; ++kk){ // 12 clusters in the first card. We check the right side
		  if (towerID_cluster_L1Card[kk%4][kk/4][card_left]>50 && crystalID_cluster_L1Card[kk%4][kk/4][card_left]>19 && energy_cluster_L1Card[kk%4][kk/4][card_left]>0){
		       for (int ll=0; ll<12; ++ll){ // We check the 12 clusters in the card on the right
			    if (towerID_cluster_L1Card[ll%4][ll/4][card_right]<17 && crystalID_cluster_L1Card[ll%4][ll/4][card_right]<5 && fabs(5*(towerID_cluster_L1Card[ll%4][ll/4][card_right])%17+crystalID_cluster_L1Card[ll%4][ll/4][card_right]%5-5*(towerID_cluster_L1Card[kk%4][kk/4][card_left])%17-crystalID_cluster_L1Card[kk%4][kk/4][card_left]%5)<2){ 
				if (energy_cluster_L1Card[kk%4][kk/4][card_left]>energy_cluster_L1Card[ll%4][ll/4][card_right]){
				    energy_cluster_L1Card[kk%4][kk/4][card_left]+=energy_cluster_L1Card[ll%4][ll/4][card_right];
				    energy_cluster_L1Card[ll%4][ll/4][card_right]=0;
				} // The least energetic cluster is merged to the most energetic one
				else{
                                    energy_cluster_L1Card[ll%4][ll/4][card_right]+=energy_cluster_L1Card[kk%4][kk/4][card_left];
                                    energy_cluster_L1Card[kk%4][kk/4][card_left]=0;
                                }
			    }
			}
		   }
		}
	     }
	}

      // Bremsstrahlung corrections. Merge clusters on the phi edges depending on pT (pt more than 10 percent, dphi leq 5, deta leq 1)
      if (do_brem){
        for (int ii=0; ii<18; ++ii){ // 18 borders in phi
           for (int jj=0; jj<2; ++jj){ // 2 eta bins
             int card_left=2*ii+jj;
             int card_right=2*ii+jj+2;
             if (card_right>35) card_right=card_right-36;
             for (int kk=0; kk<12; ++kk){ // 12 clusters in the first card. We check the right side crystalID_cluster_L1Card[kk%4][kk/4][card_left] 
                  if (towerID_cluster_L1Card[kk%4][kk/4][card_left]>50 && energy_cluster_L1Card[kk%4][kk/4][card_left]>0){ // if the tower is at the edge there might be brem corrections, whatever the crystal ID
                       for (int ll=0; ll<12; ++ll){ // We check the 12 clusters in the card on the right
                            if (towerID_cluster_L1Card[ll%4][ll/4][card_right]<17 && fabs(5*(towerID_cluster_L1Card[ll%4][ll/4][card_right])%17+crystalID_cluster_L1Card[ll%4][ll/4][card_right]%5-5*(towerID_cluster_L1Card[kk%4][kk/4][card_left])%17-crystalID_cluster_L1Card[kk%4][kk/4][card_left]%5)<=1){ //Distance of 1 max in eta
                            if (towerID_cluster_L1Card[ll%4][ll/4][card_right]<17 && fabs(5+crystalID_cluster_L1Card[ll%4][ll/4][card_right]/5-(crystalID_cluster_L1Card[kk%4][kk/4][card_left]/5))<=5){ //Distance of 5 max in phi
                                if (energy_cluster_L1Card[kk%4][kk/4][card_left]>energy_cluster_L1Card[ll%4][ll/4][card_right] && energy_cluster_L1Card[ll%4][ll/4][card_right]>0.10*energy_cluster_L1Card[kk%4][kk/4][card_left]){
                                    energy_cluster_L1Card[kk%4][kk/4][card_left]+=energy_cluster_L1Card[ll%4][ll/4][card_right];                            
                                    energy_cluster_L1Card[ll%4][ll/4][card_right]=0;
                                    brem_cluster_L1Card[kk%4][kk/4][card_left]=1;
                                } // The least energetic cluster is merged to the most energetic one, if its pt is at least ten percent
                                else if (energy_cluster_L1Card[kk%4][kk/4][card_right]>=energy_cluster_L1Card[ll%4][ll/4][card_left] && energy_cluster_L1Card[ll%4][ll/4][card_left]>0.10*energy_cluster_L1Card[kk%4][kk/4][card_right]){
                                    energy_cluster_L1Card[ll%4][ll/4][card_right]+=energy_cluster_L1Card[kk%4][kk/4][card_left];                            
                                    energy_cluster_L1Card[kk%4][kk/4][card_left]=0;
                                    brem_cluster_L1Card[ll%4][ll/4][card_right]=1;
                                }
                              } //max distance eta
                            } //max distance phi
                          } //max distance phi
                       }
                   }
                }
           }
      }      

      // Merge clusters on the eta edges
      for (int ii=0; ii<18; ++ii){ // 18 borders in eta
	     int card_bottom=2*ii;
	     int card_top=2*ii+1;
             for (int kk=0; kk<12; ++kk){ // 12 clusters in the first card. We check the top side
                  if (towerID_cluster_L1Card[kk%4][kk/4][card_bottom]%17==16 && crystalID_cluster_L1Card[kk%4][kk/4][card_bottom]%5==4 && energy_cluster_L1Card[kk%4][kk/4][card_bottom]>0){ // If there is one cluster on the right side of the first card
                       for (int ll=0; ll<12; ++ll){ // We check the card on the right
                            if (fabs(5*(towerID_cluster_L1Card[kk%4][kk/4][card_bottom]/17)+crystalID_cluster_L1Card[kk%4][kk/4][card_bottom]/5-5*(towerID_cluster_L1Card[ll%4][ll/4][card_top]/17)-crystalID_cluster_L1Card[ll%4][ll/4][card_top]/5)<2){
                                if (energy_cluster_L1Card[kk%4][kk/4][card_bottom]>energy_cluster_L1Card[ll%4][ll/4][card_bottom]){
                                    energy_cluster_L1Card[kk%4][kk/4][card_bottom]+=energy_cluster_L1Card[ll%4][ll/4][card_top];
                                    energy_cluster_L1Card[ll%4][ll/4][card_top]=0;
                                }
                                else{
                                    energy_cluster_L1Card[ll%4][ll/4][card_top]+=energy_cluster_L1Card[kk%4][kk/4][card_bottom];
                                    energy_cluster_L1Card[kk%4][kk/4][card_bottom]=0;
                                }
                            }
                        }
                   }
                }
        }

	// Regroup the new clusters per equivalent of L1 card geometry
	for (int ii=0; ii<36; ++ii){
	   for (int jj=0; jj<12; ++jj){
	       if (energy_cluster_L1Card[jj%4][jj/4][ii]>0){
		  mycluster mc1;
                  mc1.cpt=energy_cluster_L1Card[jj%4][jj/4][ii];
		  mc1.cbrem=brem_cluster_L1Card[jj%4][jj/4][ii];
                  mc1.ctowerid=towerID_cluster_L1Card[jj%4][jj/4][ii];
                  mc1.ccrystalid=crystalID_cluster_L1Card[jj%4][jj/4][ii];
		  //mc1.cinsidecrystalid=insideCrystalID_cluster_L1Card[jj%4][jj/4][ii];
                  mc1.cshowershape=showerShape_cluster_L1Card[jj%4][jj/4][ii];
                  mc1.cshowershapeloosetk=showerShapeLooseTk_cluster_L1Card[jj%4][jj/4][ii];
                  //mc1.cvalueshowershape=valueShowerShape_cluster_L1Card[jj%4][jj/4][ii];
                  mc1.cphotonshowershape=photonShowerShape_cluster_L1Card[jj%4][jj/4][ii];
		  cluster_list_L2[ii].push_back(mc1);
		}
	   }
          std::sort(begin(cluster_list_L2[ii]), end(cluster_list_L2[ii]), [](mycluster a, mycluster b){return a.cpt > b.cpt;});
	}

	// If there are more than 8 clusters per equivalent of L1 card we need to put them back in the towers
	for (int ii=0; ii<36; ++ii){
	    for (unsigned int jj=8; jj<12 && jj<cluster_list_L2[ii].size(); ++jj){
		if (cluster_list_L2[ii][jj].cpt>0){
		   ECAL_tower_L1Card[cluster_list_L2[ii][jj].ctowerid/17][cluster_list_L2[ii][jj].ctowerid%17][ii]+=cluster_list_L2[ii][jj].cpt; 
		   cluster_list_L2[ii][jj].cpt=0;
                   cluster_list_L2[ii][jj].ctowerid=0;
                   cluster_list_L2[ii][jj].ccrystalid=0;
	        }
	    }
	}

	// Compute isolation (7*7 ECAL towers) and HCAL energy (5x5 HCAL towers)
	for (int ii=0; ii<36; ++ii){ // Loop over the new cluster list (stored in 36x8 format)
	    for (unsigned int jj=0; jj<8 && jj<cluster_list_L2[ii].size(); ++jj){
                int cluster_etaOfTower_fullDetector=get_towerEta_fromCardTowerInCard(ii,cluster_list_L2[ii][jj].ctowerid);
                int cluster_phiOfTower_fullDetector=get_towerPhi_fromCardTowerInCard(ii,cluster_list_L2[ii][jj].ctowerid);
		float hcal_nrj=0.0;
		float isolation=0.0;
		int ntowers=0;
                for (int iii=0; iii<36; ++iii){ // The clusters have to be added to the isolation
                    for (unsigned int jjj=0; jjj<8 && jjj<cluster_list_L2[iii].size(); ++jjj){
			if (!(iii==ii && jjj==jj)){
			    int cluster2_eta=get_towerEta_fromCardTowerInCard(iii,cluster_list_L2[iii][jjj].ctowerid);
                	    int cluster2_phi=get_towerPhi_fromCardTowerInCard(iii,cluster_list_L2[iii][jjj].ctowerid);
                            if (abs(cluster2_eta-cluster_etaOfTower_fullDetector)<=2 && (abs(cluster2_phi-cluster_phiOfTower_fullDetector)<=2 or abs(cluster2_phi-72-cluster_phiOfTower_fullDetector)<=2)){
			       isolation+=cluster_list_L2[iii][jjj].cpt;
			    }
			}
		    }
		}
		for (int kk=0; kk<36; ++kk){ // 36 cards
		    for (int ll=0; ll<4; ++ll){ // 4 links per card
			for (int mm=0; mm<17; ++mm){ // 17 towers per link
			    int etaOftower_fullDetector=get_towerEta_fromCardLinkTower(kk,ll,mm);
                            int phiOftower_fullDetector=get_towerPhi_fromCardLinkTower(kk,ll,mm);
			    // First do ECAL
			    if (abs(etaOftower_fullDetector-cluster_etaOfTower_fullDetector)<=2 && (abs(phiOftower_fullDetector-cluster_phiOfTower_fullDetector)<=2 or abs(phiOftower_fullDetector-72-cluster_phiOfTower_fullDetector)<=2)){ // The towers are within 3. Needs to stitch the two phi sides together
				if (!((cluster_phiOfTower_fullDetector==0 && phiOftower_fullDetector==71) or (cluster_phiOfTower_fullDetector==23 && phiOftower_fullDetector==26) or (cluster_phiOfTower_fullDetector==24 && phiOftower_fullDetector==21) or (cluster_phiOfTower_fullDetector==47 && phiOftower_fullDetector==50) or (cluster_phiOfTower_fullDetector==48 && phiOftower_fullDetector==45) or(cluster_phiOfTower_fullDetector==71 && phiOftower_fullDetector==2))){ // Remove the column outside of the L2 card
			           isolation+=ECAL_tower_L1Card[ll][mm][kk];
			           ntowers++;
				}
			    }
			    // Now do HCAL
                            if (abs(etaOftower_fullDetector-cluster_etaOfTower_fullDetector)<=2 && (abs(phiOftower_fullDetector-cluster_phiOfTower_fullDetector)<=2 or abs(phiOftower_fullDetector-72-cluster_phiOfTower_fullDetector)<=2)){ // The towers are within 2. Needs to stitch the two phi sides together
                                hcal_nrj+=HCAL_tower_L1Card[ll][mm][kk];
                            }
			}
		    }
		}
		cluster_list_L2[ii][jj].ciso=((isolation)*(25.0/ntowers))/cluster_list_L2[ii][jj].cpt;
                cluster_list_L2[ii][jj].chovere=hcal_nrj/cluster_list_L2[ii][jj].cpt;
	    }
	}

      //Reformat the information inside the 3 L2 cards
      //First let's fill the towers
      for (int ii=0; ii<48; ++ii){
         for (int jj=0; jj<17; ++jj){
            for (int ll=0; ll<3; ++ll){
		ECAL_tower_L2Card[ii][jj][ll]=ECAL_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll,ii)];
                HCAL_tower_L2Card[ii][jj][ll]=HCAL_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll,ii)];
                iEta_tower_L2Card[ii][jj][ll]=iEta_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll,ii)];
                iPhi_tower_L2Card[ii][jj][ll]=iPhi_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll,ii)];
            }
         }
      }

      //Second let's fill the clusters
      for (int ii=0; ii<36; ++ii){ // The cluster list is still in the L1 like geometry
          for (unsigned int jj=0; jj<unsigned(cluster_list_L2[ii].size()) && jj<12; ++jj){
              //insideCrystalID_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cinsidecrystalid;
              crystalID_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].ccrystalid;
              towerID_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].ctowerid;
              energy_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cpt;
              brem_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cbrem;
              isolation_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].ciso;
              HE_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].chovere;
              showerShape_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cshowershape;
              showerShapeLooseTk_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cshowershapeloosetk;
              //valueShowerShape_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cvalueshowershape;
              photonShowerShape_cluster_L2Card[4*(ii%12)+jj%4][jj/4][ii/12]=cluster_list_L2[ii][jj].cphotonshowershape;
          }
      }

      /* // Needed at hardware level but not software
      // From ECAL and HCAL energies of the towers compute H/E and H+E
      for (int ii=0; ii<48; ++ii){
         for (int jj=0; jj<17; ++jj){
            for (int ll=0; ll<3; ++ll){
               energy_tower_L2Card[ii][jj][ll]=ECAL_tower_L2Card[ii][jj][ll]+HCAL_tower_L2Card[ii][jj][ll];
               if (ECAL_tower_L2Card[ii][jj][ll]>0) HE_tower_L2Card[ii][jj][ll]=HCAL_tower_L2Card[ii][jj][ll]/ECAL_tower_L2Card[ii][jj][ll];
	       else HE_tower_L2Card[ii][jj][ll]=-1;
            }
         }
      }*/

     //cout<<energy_tower_L2Card[0][0][0]<<" "<<HE_tower_L2Card[0][0][0]<<" "<<crystalID_cluster_L2Card[0][0][0]<<" "<<towerID_cluster_L2Card[0][0][0]<<" "<<energy_cluster_L2Card[0][0][0]<<" "<<HE_cluster_L2Card[0][0][0]<<" "<<isolation_cluster_L2Card[0][0][0]<<" "<<insideCrystalID_cluster_L2Card[0][0][0]<<" "<<photonShowerShape_cluster_L2Card[0][0][0]<<" "<<showerShape_cluster_L2Card[0][0][0]<<" "<<valueShowerShape_cluster_L2Card[0][0][0]<<endl;

   std::unique_ptr<l1slhc::L1EGCrystalClusterCollection> L1EGXtalClusterEmulator( new l1slhc::L1EGCrystalClusterCollection );
   std::unique_ptr< BXVector<l1t::EGamma> > L1EGammaCollectionBXVEmulator(new l1t::EGammaBxCollection);
   std::unique_ptr< L1CaloTowerCollection > l1CaloTowerCollection(new L1CaloTowerCollection);

   // Fill the cluster collection
   for (int ii=0; ii<48; ++ii){
       for (int jj=0; jj<2; ++jj){
           for (int ll=0; ll<3; ++ll){
               if (energy_cluster_L2Card[ii][jj][ll]>0.45 ){
		   reco::Candidate::PolarLorentzVector p4calibrated(energy_cluster_L2Card[ii][jj][ll],getEta_fromL2LinkCardTowerCrystal(ii,ll,towerID_cluster_L2Card[ii][jj][ll],crystalID_cluster_L2Card[ii][jj][ll]),getPhi_fromL2LinkCardTowerCrystal(ii,ll,towerID_cluster_L2Card[ii][jj][ll],crystalID_cluster_L2Card[ii][jj][ll]), 0.);
		   SimpleCaloHit centerhit;
		   bool is_iso=passes_iso(energy_cluster_L2Card[ii][jj][ll],isolation_cluster_L2Card[ii][jj][ll]);
                   bool is_looseTkiso=passes_looseTkiso(energy_cluster_L2Card[ii][jj][ll],isolation_cluster_L2Card[ii][jj][ll]);
		   bool is_ss=(showerShape_cluster_L2Card[ii][jj][ll]==1);
		   bool is_photon=(photonShowerShape_cluster_L2Card[ii][jj][ll]==1) && is_ss && is_iso;
                   bool is_looseTkss=(showerShapeLooseTk_cluster_L2Card[ii][jj][ll]==1);
		   //bool is_he=passes_he(energy_cluster_L2Card[ii][jj][ll],HE_cluster_L2Card[ii][jj][ll]); //not used so far
		   // All the ID set to Standalone WP! Some dummy values for non calculated variables
		   l1slhc::L1EGCrystalCluster cluster(p4calibrated, energy_cluster_L2Card[ii][jj][ll], HE_cluster_L2Card[ii][jj][ll], isolation_cluster_L2Card[ii][jj][ll], centerhit.id,-1000, float(brem_cluster_L2Card[ii][jj][ll]),-1000,-1000, energy_cluster_L2Card[ii][jj][ll],-1, is_iso && is_ss, is_iso && is_ss, is_photon, is_iso && is_ss, is_looseTkiso && is_looseTkss, is_iso && is_ss); 
           // Experimental parameters, don't want to bother with hardcoding them in data format
           std::map<std::string, float> params;
           params["standaloneWP_showerShape"] = is_ss;
           params["standaloneWP_isolation"] = is_iso;
           params["trkMatchWP_showerShape"] = is_looseTkss;
           params["trkMatchWP_isolation"] = is_looseTkiso;
           cluster.SetExperimentalParams(params);
		   L1EGXtalClusterEmulator->push_back(cluster);

                   // BXVector l1t::EGamma quality defined with respect to these WPs
                   // FIXME, need to defaul some of these to 0 I think...
                   int standaloneWP = (int)(is_iso && is_ss);
                   int looseL1TkMatchWP = (int)(is_looseTkiso && is_looseTkss); 
                   int photonWP = (int)(is_photon);
                   int quality = (standaloneWP*2^0) + (looseL1TkMatchWP*2^1) + (photonWP*2^2);
                   L1EGammaCollectionBXVEmulator->push_back(0,l1t::EGamma(p4calibrated, p4calibrated.pt(), p4calibrated.eta(), p4calibrated.phi(),quality,1 ));

		}
	    }
	}
    }

   // Fill the tower collection
   for (int ii=0; ii<48; ++ii){
       for (int jj=0; jj<17; ++jj){
           for (int ll=0; ll<3; ++ll){
              L1CaloTower l1CaloTower;
              if (HCAL_tower_L2Card[ii][jj][ll]>0 or ECAL_tower_L2Card[ii][jj][ll]>0){
                 l1CaloTower.ecal_tower_et = ECAL_tower_L2Card[ii][jj][ll];
                 l1CaloTower.hcal_tower_et = HCAL_tower_L2Card[ii][jj][ll];
                 l1CaloTower.tower_iEta = getToweriEta_fromAbsoluteID(iEta_tower_L2Card[ii][jj][ll]);
                 l1CaloTower.tower_iPhi = getToweriPhi_fromAbsoluteID(iPhi_tower_L2Card[ii][jj][ll]);
                 l1CaloTower.tower_eta = getTowerEta_fromAbsoluteID(iEta_tower_L2Card[ii][jj][ll]);
                 l1CaloTower.tower_phi = getTowerPhi_fromAbsoluteID(iPhi_tower_L2Card[ii][jj][ll]);
                 //std::cout << "Tower TP Position eta,iEta: " << l1CaloTower.tower_eta << "," << l1CaloTower.tower_iEta << std::endl;
                 //std::cout << "Tower TP Position phi,iPhi: " << l1CaloTower.tower_phi << "," << l1CaloTower.tower_iPhi << std::endl;
                 
                 //for(const auto& hit : hcalhits)
                 //{
                 //   if (l1CaloTower.tower_iPhi == hit.id_hcal.iphi() && l1CaloTower.tower_iEta == hit.id_hcal.ieta() )
                 //   {
                 //       //printf("\nMatching Towers iEta:iPhi %i %i - eta1:eta2 %f %f - phi1:phi2 %f %f", l1CaloTower.tower_iEta,l1CaloTower.tower_iPhi,l1CaloTower.tower_eta,hit.position.eta(),l1CaloTower.tower_phi,(float)hit.position.phi());
                 //       if (l1CaloTower.hcal_tower_et == hit.pt()) std::cout << ".";
                 //       else printf("\n - Mismatch in HCAL energy, l1CaloTower ET: %f hit direct ET: %f", l1CaloTower.hcal_tower_et, hit.pt());
                 //   }
                 //}
                 l1CaloTowerCollection->push_back( l1CaloTower );
	      }
       	   }
	}
   }



   iEvent.put(std::move(L1EGXtalClusterEmulator),"L1EGXtalClusterEmulator");
   iEvent.put(std::move(L1EGammaCollectionBXVEmulator),"L1EGammaCollectionBXVEmulator");
   iEvent.put(std::move(l1CaloTowerCollection),"L1CaloTowerCollection");

}

bool L1EGCrystalClusterEmulatorProducer::passes_iso(float pt, float iso) {
   if (pt<80){
        if (!((0.85-0.0080*pt)>iso)) return false;
   }
   if (pt>=80){
        if (iso>0.21) return false;
   }
   return true;
}

bool L1EGCrystalClusterEmulatorProducer::passes_looseTkiso(float pt, float iso) {
   if (( 0.38 + 1.9 * TMath::Exp( -0.05 * pt ) > iso)) return true;
   else return false;
}

bool L1EGCrystalClusterEmulatorProducer::passes_ss(float pt, float ss) {
   if ((0.94+0.052*TMath::Exp(-0.044*pt))>ss) return false;
   else return true;
}

bool L1EGCrystalClusterEmulatorProducer::passes_photon(float pt, float pss) {
   if ( pss > 0.96 - 0.0003 * pt ) return true;
   return false;
}


bool L1EGCrystalClusterEmulatorProducer::passes_looseTkss(float pt, float ss) {
   if ((0.944-0.65*TMath::Exp(-0.4*pt))>ss) return false;
   else return true;
}

bool L1EGCrystalClusterEmulatorProducer::passes_he(float pt, float he) {
   if (he<(0.269+TMath::Exp(1.2504-0.0478*pt))) return true;
   else return false;
}

float L1EGCrystalClusterEmulatorProducer::get_calibrate(float uncorr){
   return uncorr*(1.0712 + TMath::Exp(-0.6819-0.0462*uncorr));
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1EGCrystalClusterEmulatorProducer);
