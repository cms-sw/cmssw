#ifndef EcalPileUpDepMonitor_H
#define EcalPileUpDepMonitor_H

/*
 * \file EcalPileUpDepMonitor.h
 *
 * \author Ben Carlson - CMU
 *
 */

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class EcalPileUpDepMonitor: public DQMEDAnalyzer{

 public:

  /// Constructor
  EcalPileUpDepMonitor(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~EcalPileUpDepMonitor();

 protected:

  /// Analyze
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

 private:
	
  //profiles
  MonitorElement* bcEB_PV; //basic clusters Ecal-Barrel vs Number of Primary Vertices 
  MonitorElement* bcEE_PV;
  MonitorElement* scEB_PV;
  MonitorElement* scEE_PV;
	
  MonitorElement* scEtEB_PV;//super cluster Et profiles vs Number of vertices 
  MonitorElement* scEtEE_PV;
	
  MonitorElement* recHitEtEB_PV; // reconstructed hit Et profiles vs number of vertices
  MonitorElement* recHitEtEE_PV;

  MonitorElement* emIso_PV;
	
  // histograms of reconstructed hit Et and supercluster Et
  MonitorElement* emIso; 
  MonitorElement* recHitEtEB;
  MonitorElement* recHitEtEE;
	
  MonitorElement* scHitEtEB;
  MonitorElement* scHitEtEE;

  // SC energy TH1Fs already exist in ClusterTask	
/*   MonitorElement* scHitE_EB; */
/*   MonitorElement* scHitE_EE; */
	
  //Eta
  // Exists in ClusterTask
/*   MonitorElement* scEta_EB; */
/*   MonitorElement* scEta_EE; */
	
  //Phi
  // Exists in ClusterTask
/*   MonitorElement* scPhi_EB; */
/*   MonitorElement* scPhi_EE; */
	
  //sc sigma eta_eta and eta phi
	
  MonitorElement* scSigmaIetaIeta_EB;
  MonitorElement* scSigmaIetaIeta_EE;

  MonitorElement* scSigmaIetaIphi_EB;
  MonitorElement* scSigmaIetaIphi_EE;

  //R9 
  MonitorElement* r9_EB; 
  MonitorElement* r9_EE;
	
  edm::ESHandle<CaloGeometry> geomH;
  edm::ESHandle<CaloTopology> caloTop;

  edm::EDGetTokenT<reco::VertexCollection> VertexCollection_; //vertex collection
	
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > basicClusterCollection_EB_; // Ecal Barrel Basic Clusters 
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > basicClusterCollection_EE_; // Ecal Endcap Basic Clusters
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > basicClusterCollection_; // Ecal Barrel & Endcap Basic Clusters (for >= 70X, BC collection is merged)

  edm::EDGetTokenT<reco::SuperClusterCollection> superClusterCollection_EB_;
  edm::EDGetTokenT<reco::SuperClusterCollection> superClusterCollection_EE_;
  edm::EDGetTokenT<reco::GsfElectronCollection> EleTag_;
	
  edm::EDGetTokenT<EcalRecHitCollection> RecHitCollection_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> RecHitCollection_EE_;
};

#endif
