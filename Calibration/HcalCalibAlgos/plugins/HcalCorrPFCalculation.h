#ifndef HcalCorrPFCalculation_H
#define HcalCorrPFCalculation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "FWCore/Framework/interface/Selector.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalPFCorrs.h"
#include "CondFormats/DataRecord/interface/HcalPFCorrsRcd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TProfile.h"

class HcalCorrPFCalculation : public edm::EDAnalyzer {
 public:
  HcalCorrPFCalculation(edm::ParameterSet const& conf);
  ~HcalCorrPFCalculation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 private:
  
  virtual void fillRecHitsTmp(int subdet_, edm::Event const& ev);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1,double phi2);  

double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint);

  DQMStore* dbe_;
  
  std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string eventype_;
  std::string sign_;
  std::string mc_;
  bool        Respcorr_;
  bool        PFcorr_;
  bool        Conecorr_;
  bool        famos_;
  double        radius_;

  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  int subdet_;

  // single/multi-particle sample (1/2)
  int etype_;
  int iz;
  int imc;

  // for single monoenergetic particles - cone collection profile vs ieta.
  MonitorElement* meEnConeEtaProfile_depth1;
  MonitorElement* meEnConeEtaProfile_depth1Noise;
  MonitorElement* meEnConeEtaProfile;
  MonitorElement* meEnConeEtaProfileNoise;

  edm::ESHandle<CaloGeometry> geometry ;

 // Filling vectors with essential RecHits data
  std::vector<int>    csub;
  std::vector<int>    cieta;
  std::vector<int>    ciphi;
  std::vector<int>    cdepth;
  std::vector<double> cen;
  std::vector<double> ceta;
  std::vector<double> cphi;
  std::vector<double> ctime;
  std::vector<double> cx;
  std::vector<double> cy;
  std::vector<double> cz;

  // counter
  int nevtot;
  int hasresp;

  const HcalRespCorrs* myRecalib;
  const HcalPFCorrs* pfRecalib;

  SteppingHelixPropagator* stepPropF;
  MagneticField *theMagField;
  
  TProfile *nCells, *nCellsNoise, *enHcal, *enHcalNoise;
  TFile *rootFile;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;
  double taECALCone_;
  double taHCALCone_;

  const CaloGeometry* geo;

  Float_t xTrkEcal;
  Float_t yTrkEcal;
  Float_t zTrkEcal;

  Float_t xTrkHcal;
  Float_t yTrkHcal;
  Float_t zTrkHcal;

};

#endif
