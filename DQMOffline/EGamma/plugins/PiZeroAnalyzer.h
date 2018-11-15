#ifndef PiZeroAnalyzer_H
#define PiZeroAnalyzer_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// DataFormats
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
/// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
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

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//DQM services
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <map>
#include <vector>

/** \class PiZeroAnalyzer
 **  
 **
 **  $Id: PiZeroAnalyzer
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **     
 ***/

// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;

class PiZeroAnalyzer : public DQMEDAnalyzer
{

 public:
  explicit PiZeroAnalyzer( const edm::ParameterSet& ) ;
  ~PiZeroAnalyzer() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze( const edm::Event&, const edm::EventSetup& ) override ;
 
 private:
  void makePizero(const edm::EventSetup& es, const edm::Handle<EcalRecHitCollection> eb, const edm::Handle<EcalRecHitCollection> ee ); 

  std::string fName_;
  unsigned int prescaleFactor_;

  int nEvt_;

  edm::ParameterSet posCalcParameters_;

  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > barrelEcalHits_token_;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > endcapEcalHits_token_;
  double minPhoEtCut_;

  double cutStep_;
  int numberOfSteps_;

  /// parameters needed for pizero finding
  double clusSeedThr_;
  int clusEtaSize_;
  int clusPhiSize_;

  double seleXtalMinEnergy_;

  bool ParameterLogWeighted_;
  double ParameterX0_;
  double ParameterT0_barl_;
  double ParameterW0_;

  double selePtGammaOne_;
  double selePtGammaTwo_;
  double selePtPi0_;
  double seleS4S9GammaOne_;
  double seleS4S9GammaTwo_;
  double selePi0BeltDR_;
  double selePi0BeltDeta_;
  double selePi0Iso_;
  double seleMinvMaxPi0_;
  double seleMinvMinPi0_;

  std::stringstream currentFolder_;

  MonitorElement*  hMinvPi0EB_;
  MonitorElement*  hPt1Pi0EB_;
  MonitorElement*  hPt2Pi0EB_;
  MonitorElement*  hIsoPi0EB_;
  MonitorElement*  hPtPi0EB_;
};

#endif
