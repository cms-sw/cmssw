#ifndef DQMSourcePi0_H
#define DQMSourcePi0_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"


typedef std::map<DetId, EcalRecHit> RecHitsMap;
// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
  { 
    return (x.energy() > y.energy()); 
  }
};




class DQMStore;
class MonitorElement;

class DQMSourcePi0 : public edm::EDAnalyzer {

public:

  DQMSourcePi0( const edm::ParameterSet& );
  ~DQMSourcePi0();

protected:
   
  void beginJob();

  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void endRun(const edm::Run& r, const edm::EventSetup& c);

  void endJob();

  void convxtalid(int & , int &);
  int diff_neta_s(int,int);
  int diff_nphi_s(int,int);



private:
 

  DQMStore*   dbe_;  
  int eventCounter_;      
  PositionCalc posCalculator_ ;                        

  /// Distribution of rechits in iPhi (pi0)   
  MonitorElement * hiPhiDistrEBpi0_;

  /// Distribution of rechits in ix EE  (pi0)
  MonitorElement * hiXDistrEEpi0_;

  /// Distribution of rechits in iPhi (eta)   
  MonitorElement * hiPhiDistrEBeta_;

  /// Distribution of rechits in ix EE  (eta)
  MonitorElement * hiXDistrEEeta_;

  /// Distribution of rechits in iEta (pi0)
  MonitorElement * hiEtaDistrEBpi0_;

  /// Distribution of rechits in iy EE (pi0)
  MonitorElement * hiYDistrEEpi0_;

  /// Distribution of rechits in iEta (eta)
  MonitorElement * hiEtaDistrEBeta_;

  /// Distribution of rechits in iy EE (eta)
  MonitorElement * hiYDistrEEeta_;

  /// Energy Distribution of rechits EB (pi0)
  MonitorElement * hRechitEnergyEBpi0_;

  /// Energy Distribution of rechits EE (pi0) 
  MonitorElement * hRechitEnergyEEpi0_;

  /// Energy Distribution of rechits EB (eta)  
  MonitorElement * hRechitEnergyEBeta_;

  /// Energy Distribution of rechits EE (eta) 
  MonitorElement * hRechitEnergyEEeta_;

  /// Distribution of total event energy EB (pi0)
  MonitorElement * hEventEnergyEBpi0_;

  /// Distribution of total event energy EE (pi0)
  MonitorElement * hEventEnergyEEpi0_;
  
  /// Distribution of total event energy EB (eta)
  MonitorElement * hEventEnergyEBeta_;

  /// Distribution of total event energy EE (eta)
  MonitorElement * hEventEnergyEEeta_;
  
  /// Distribution of number of RecHits EB (pi0)
  MonitorElement * hNRecHitsEBpi0_;

  /// Distribution of number of RecHits EE (pi0)
  MonitorElement * hNRecHitsEEpi0_;

  /// Distribution of number of RecHits EB (eta)
  MonitorElement * hNRecHitsEBeta_;

  /// Distribution of number of RecHits EE (eta)
  MonitorElement * hNRecHitsEEeta_;

  /// Distribution of Mean energy per rechit EB (pi0)
  MonitorElement * hMeanRecHitEnergyEBpi0_;

  /// Distribution of Mean energy per rechit EE (pi0)
  MonitorElement * hMeanRecHitEnergyEEpi0_;

  /// Distribution of Mean energy per rechit EB (eta)
  MonitorElement * hMeanRecHitEnergyEBeta_;

  /// Distribution of Mean energy per rechit EE (eta)
  MonitorElement * hMeanRecHitEnergyEEeta_;

  /// Pi0 invariant mass in EB
  MonitorElement * hMinvPi0EB_;

  /// Pi0 invariant mass in EE
  MonitorElement * hMinvPi0EE_;

  /// Eta invariant mass in EB
  MonitorElement * hMinvEtaEB_;

  /// Eta invariant mass in EE
  MonitorElement * hMinvEtaEE_;

  /// Pt of the 1st most energetic Pi0 photon in EB 
  MonitorElement *hPt1Pi0EB_;

  /// Pt of the 1st most energetic Pi0 photon in EE
  MonitorElement *hPt1Pi0EE_;

  /// Pt of the 1st most energetic Eta photon in EB
  MonitorElement *hPt1EtaEB_;

  /// Pt of the 1st most energetic Eta photon in EE
  MonitorElement *hPt1EtaEE_;

  
  /// Pt of the 2nd most energetic Pi0 photon in EB
  MonitorElement *hPt2Pi0EB_;

  /// Pt of the 2nd most energetic Pi0 photon in EE
  MonitorElement *hPt2Pi0EE_;

  /// Pt of the 2nd most energetic Eta photon in EB
  MonitorElement *hPt2EtaEB_;

  /// Pt of the 2nd most energetic Eta photon in EE
  MonitorElement *hPt2EtaEE_;

  
  /// Pi0 Pt in EB
  MonitorElement * hPtPi0EB_;

  /// Pi0 Pt in EE
  MonitorElement * hPtPi0EE_;

  /// Eta Pt in EB
  MonitorElement * hPtEtaEB_;

  /// Eta Pt in EE
  MonitorElement * hPtEtaEE_;

  /// Pi0 Iso EB
  MonitorElement * hIsoPi0EB_;

  /// Pi0 Iso EE
  MonitorElement * hIsoPi0EE_;

  /// Eta Iso EB
  MonitorElement * hIsoEtaEB_;

  /// Eta Iso EE
  MonitorElement * hIsoEtaEE_;

  /// S4S9 of the 1st most energetic pi0 photon
  MonitorElement * hS4S91Pi0EB_;

  /// S4S9 of the 1st most energetic pi0 photon EE
  MonitorElement * hS4S91Pi0EE_;

  /// S4S9 of the 1st most energetic eta photon
  MonitorElement * hS4S91EtaEB_;

  /// S4S9 of the 1st most energetic eta photon EE
  MonitorElement * hS4S91EtaEE_;

  /// S4S9 of the 2nd most energetic pi0 photon
  MonitorElement * hS4S92Pi0EB_;
  
  /// S4S9 of the 2nd most energetic pi0 photon EE
  MonitorElement * hS4S92Pi0EE_;
  
  /// S4S9 of the 2nd most energetic eta photon
  MonitorElement * hS4S92EtaEB_;
  
  /// S4S9 of the 2nd most energetic eta photon EE
  MonitorElement * hS4S92EtaEE_;
  



  /// object to monitor
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEBpi0_;
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEBeta_;

 /// object to monitor
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEEpi0_;
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEEeta_;

      int gammaCandEtaSize_;
      int gammaCandPhiSize_;

      double seleXtalMinEnergy_;
      double seleXtalMinEnergyEndCap_;

  double clusSeedThr_;
  int clusEtaSize_;
  int clusPhiSize_;

  double clusSeedThrEndCap_;

      //// for pi0->gg barrel 
      double selePtGamma_;
      double selePtPi0_;
      double seleMinvMaxPi0_;
      double seleMinvMinPi0_;
      double seleS4S9Gamma_;
      double selePi0BeltDR_;
      double selePi0BeltDeta_;
      double selePi0Iso_;
      double ptMinForIsolation_; 

      ///for pi0->gg endcap
      double selePtGammaEndCap_;
      double selePtPi0EndCap_;
      double seleMinvMaxPi0EndCap_;
      double seleMinvMinPi0EndCap_;
      double seleS4S9GammaEndCap_;
      double selePi0IsoEndCap_;
      double selePi0BeltDREndCap_;
      double selePi0BeltDetaEndCap_;
      double ptMinForIsolationEndCap_; 

      ///for eta->gg barrel
      double selePtGammaEta_;
      double selePtEta_;
      double seleS4S9GammaEta_; 
      double seleS9S25GammaEta_; 
      double seleMinvMaxEta_; 
      double seleMinvMinEta_; 
      double ptMinForIsolationEta_; 
      double seleEtaIso_; 
      double seleEtaBeltDR_; 
      double seleEtaBeltDeta_; 

      ///for eta->gg endcap
      double selePtGammaEtaEndCap_;
      double seleS4S9GammaEtaEndCap_;
      double seleS9S25GammaEtaEndCap_;
      double selePtEtaEndCap_;
      double seleMinvMaxEtaEndCap_;
      double seleMinvMinEtaEndCap_;
      double ptMinForIsolationEtaEndCap_;
      double seleEtaIsoEndCap_;
      double seleEtaBeltDREndCap_;
      double seleEtaBeltDetaEndCap_;


  bool ParameterLogWeighted_;
  double ParameterX0_;
  double ParameterT0_barl_;
  double ParameterT0_endc_;
  double ParameterT0_endcPresh_;
  double ParameterW0_;



  std::vector<EBDetId> detIdEBRecHits; 
  std::vector<EcalRecHit> EBRecHits; 
 
  
  std::vector<EEDetId> detIdEERecHits; 
  std::vector<EcalRecHit> EERecHits; 



  /// Monitor every prescaleFactor_ events
  unsigned int prescaleFactor_;
  
  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  /// which subdet will be monitored
  bool isMonEBpi0_;
  bool isMonEBeta_;
  bool isMonEEpi0_;
  bool isMonEEeta_;

  /// Output file name if required
  std::string fileName_;
};

#endif

