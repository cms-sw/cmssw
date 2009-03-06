#ifndef DQMSourcePi0_H
#define DQMSourcePi0_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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
   
  void beginJob(const edm::EventSetup& c);

  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void endRun(const edm::Run& r, const edm::EventSetup& c);

  void endJob();

private:
 

  DQMStore*   dbe_;  
  int eventCounter_;      
                        

  /// Distribution of rechits in iPhi   
  MonitorElement * hiPhiDistrEB_;

  /// Distribution of rechits in iEta
  MonitorElement * hiEtaDistrEB_;

  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEB_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEB_;
  
  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEB_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEB_;

  /// Pi0 invariant mass in EB
  MonitorElement * hMinvPi0EB_;

  /// Pt of the 1st most energetic Pi0 photon in EB
  MonitorElement *hPt1Pi0EB_;

  
  /// Pt of the 2nd most energetic Pi0 photon in EB
  MonitorElement *hPt2Pi0EB_;

  
  /// Pi0 Pt in EB
  MonitorElement * hPtPi0EB_;

  /// Pi0 Iso
  MonitorElement * hIsoPi0EB_;

  /// S4S9 of the 1st most energetic pi0 photon
  MonitorElement * hS4S91EB_;

  /// S4S9 of the 2nd most energetic pi0 photon
  MonitorElement * hS4S92EB_;
  


  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEE_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEE_;
  
  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEE_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEE_;

  /// object to monitor
  edm::InputTag productMonitoredEB_;

 /// object to monitor
  edm::InputTag productMonitoredEE_;

  int gammaCandEtaSize_;
  int gammaCandPhiSize_;

  double clusSeedThr_;
  int clusEtaSize_;
  int clusPhiSize_;

  double selePtGammaOne_;
  double selePtGammaTwo_;
  double selePtPi0_;
  double seleMinvMaxPi0_;
  double seleMinvMinPi0_;
  double seleXtalMinEnergy_;
  int seleNRHMax_;
  //New criteria
  double seleS4S9GammaOne_;
  double seleS4S9GammaTwo_;
  double selePi0BeltDR_;
  double selePi0BeltDeta_;
  double selePi0Iso_;
  bool ParameterLogWeighted_;
  double ParameterX0_;
  double ParameterT0_barl_;
  double ParameterW0_;

  std::map<DetId, EcalRecHit> *recHitsEB_map;


  /// Monitor every prescaleFactor_ events
  unsigned int prescaleFactor_;
  
  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  /// which subdet will be monitored
  bool isMonEB_;
  bool isMonEE_;

  /// Output file name if required
  std::string fileName_;
};

#endif

