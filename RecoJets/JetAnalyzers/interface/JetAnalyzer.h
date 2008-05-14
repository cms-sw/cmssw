#ifndef ANALYSIS_JET_MYJET_H
#define ANALYSIS_JET_MYJET_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include "TLorentzVector.h"
#include <vector>
#include <map>


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"


#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

//#include "DataFormats/JetReco/interface/CaloJet.h"
//#include "DataFormats/JetReco/interface/CaloJetfwd.h"
//#include "DataFormats/JetReco/interface/GenJet.h"
//#include "DataFormats/JetReco/interface/GenJetfwd.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


//#include "DataFormats/EgammaReco/interface/BasicCluster.h"
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"



#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "RecoJets/JetAnalyzers/interface/CaloTowerBoundries.h"
#include "RecoJets/JetAnalyzers/interface/MyCluster.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"


/** \class JetAnalyzer
  *  
  * $Date: 2008/01/28 21:52:42 $
  * $Revision: 1.7 $
  * \author L. Apanasevich - UIC and Anwar Bhatti
  */
class JetAnalyzer : public edm::EDAnalyzer {
public:
  //  JetAnalyzer(); 

  JetAnalyzer(const edm::ParameterSet& pSet);
  virtual void endJob();

  /** Setup the analysis to put the histograms into HistoFile and focus on
      ieta,iphi for analysis.
  */
  void setup(const edm::ParameterSet& pSet);

  void fillHist1D(const TString histName, const Double_t x,const Double_t wt=1.0);
  void fillHist2D(const TString histName, const Double_t x,const Double_t y,const Double_t wt=1.0);


  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup);


  /** Analyze the Data */
  void analyze(const CaloJetCollection& rjets,
	       const GenJetCollection& gjets,
	       const CaloMETCollection& rmets,
	       const GenMETCollection& gmets,
	       const CaloTowerCollection& caloTowers,
	       const HepMC::GenEvent genEvent,
	       const EBRecHitCollection& EBRecHits,
	       const EERecHitCollection& EERecHits,
	       const HBHERecHitCollection& hbhe_hits, 
	       const HBHEDigiCollection& hbhe_digis, 
	       const HORecHitCollection& ho_hits, 
	       const HODigiCollection& ho_digis, 
	       const HFRecHitCollection& hf_hits, 
	       const HFDigiCollection& hf_digis, 
	       const HcalTBTriggerData& trigger,
	       const CaloGeometry& geom);

  void MakeLocalClusters(const CaloGeometry& caloGeometry,
			 const CaloJetCollection& calojets,
			 const CaloMETCollection& recmets,
			 const CaloTowerCollection& caloTowers,
			 const EBRecHitCollection& EBRecHits,
			 const EERecHitCollection& EERecHits,
			 const HBHERecHitCollection& HBHERecHits,
			 const HORecHitCollection& HORecHits,
			 const HFRecHitCollection& HFRecHits);


  void dummyAnalyze(
	       const CaloGeometry& geom);
  /** Finalization (close files, etc) */
  void done();


  void bookHistograms();
  void bookGeneralHistograms();
  void bookFillEoPCorrectionPlots();

  void bookRecHitHists(const TString subDetName);
  template <typename T> void fillRecHitHists(const CaloGeometry& caloGeometry,const T& hits,const TString subDetName,double& sumEnergy);
  //  void fillRecHitHists(const CaloGeometry& caloGeometry,const EBRecHitCollection& EBRecHits,const EERecHitCollection& EERecHits);


  void bookClusterPlot(TString name);
  void fillClusterPlot(TString name, std::vector<CalCluster> ClusterList);

  void bookTBTriggerHists();
  void fillTBTriggerHists(const HcalTBTriggerData& trigger);

  void bookCaloTowerHists();
  void fillCaloTowerHists(const CaloTowerCollection& caloTowers);


  void MakeCellListFromCaloTowers(const CaloGeometry& caloGeometry,
                                      const CaloTowerCollection& caloTowers,
				      const EBRecHitCollection& EBRecHits,
   			              const EERecHitCollection& EERecHits,
				      const HBHERecHitCollection& HBHERecHits,
				      const HORecHitCollection& HORecHits,
				      const HFRecHitCollection& HFRecHits,
				      std::vector<CalCell>& EmCellList,
				      std::vector<CalCell>& HdCellList);


  void MakeCaloTowerList(const CaloGeometry& caloGeometry,const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList);

  void MakeEmCellList(const CaloGeometry& caloGeometry,
		      const EBRecHitCollection& EBRecHits,
		      const EERecHitCollection& EERecHits,
		      std::vector<CalCell>& CellList);


  void MakeHadCellList(const CaloGeometry& caloGeometry,
		       const HBHERecHitCollection& HBHERecHits,
		       const HORecHitCollection& HORecHits,
		       const HFRecHitCollection& HFRecHits,
		       std::vector<CalCell>& CellList);

  void CalculateSumEtMET(std::vector<CalCluster> EmHdClusterList,double& SumEt,double& MET);

  void Convert2HepLorentzVector(GlobalPoint position,double energy,CLHEP::HepLorentzVector& P4);

  void bookForId(const HcalDetId& id);
  void bookForId_TS(const HcalDetId& id);

  void bookMetHists(const TString& prefix);
  template <typename T> void fillMetHists(const T& mets, const TString& prefx);

  void bookJetHistograms(const TString& prefix);

  template <typename T> void fillJetHists(const T& jets, const TString& prefx);
  void fillJetHists(const CaloJetCollection& calojets,const TString& prefix);

  template <typename T> void fillRecHits(const T& hits);
  template <typename T> void fillDigis(const T& digis);


  void bookCalculateEfficiency();
  void CalculateEfficiency(const GenJetCollection& genJets,const CaloJetCollection& calojets);

  void bookDiJetBalance(const TString& prefix);
  template <typename T> void DiJetBalance(const T& jets, const TString& prefix);

  void bookMCParticles();
  //  void fillMCParticles(const HepMC::GenEvent mctruth);
  void fillMCParticles(edm::Handle<CandidateCollection> genParticles);
  void bookMCParticles(const TString& prefix );
  void fillMCParticlesInsideJet(const HepMC::GenEvent genEvent,const GenJetCollection& genJets);

  void GetIntegratedEnergy(GenJetCollection::const_iterator ijet,int nbin,const HepMC::GenEvent genEvent,std::vector<double>& Bins,std::vector<double>& e,std::vector<double>& pt);

  void GetGenPhoton(math::XYZTLorentzVector& momentum);
  template <typename T> void DarkEnergyPlots(const T& jets, const TString& prefix,const CaloTowerCollection& caloTowers );

  void bookDarkMetPlots(const TString& prefix );

  void MakeHadCellList(const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList);
  void MakeEmCellList(const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList);
  void SimpleConeCluster(const int type,const double SEEDCUT, const double TOWERCUT,const double RADIUS,std::vector<CalCell> CellList,std::vector<CalCluster>& ClusterList);
  void MakeIRConeJets(const double SEEDCUT, const double TOWERCUT,const double RADIUS,std::vector<CalCell> CellList,std::vector<CalCluster>& ClusterList);

  bool VectorsAreEqual(std::vector<int> VecA,std::vector<int> VecB);

  void MatchEmHadClusters(std::vector<CalCluster> EmClusterList,std::vector<CalCluster> HdClusterList,std::vector<CalCluster>& EmHdClusterList);

  void bookSubClusterHistograms();

  void bookPtSpectrumInAJet();
  void PtSpectrumInAJet(GenJetCollection::const_iterator ijet,const HepMC::GenEvent genEvent,const double response);

  void fillSubClusterPlot(std::vector<CalCluster> CaloClusterR05List,
			  std::vector<CalCluster> CaloClusterR03List,
			  std::vector<CalCluster> CaloClusterR15List,
			  std::vector<CalCluster> HdClusterR015List,
			  std::vector<CalCluster> HdClusterR025List,
			  std::vector<CalCluster> EmClusterR003List,
			  std::vector<CalCluster> EmClusterR006List,
			  std::vector<CalCluster> EmHdClusterList);


  void PtSpectrumInSideAJet(const GenJetCollection& genJets,const HepMC::GenEvent genEvent);
  void GetParentPartons(std::vector<Candidate*>& ParentParton);

  int GetPtBin(double GenJetPt);

private:

  std::string calojets_,genJets_,recmet_,genmet_,calotowers_;
  int errCnt;
  const int errMax(){return 100;}

  int evtCounter;

  HepMC::GenEvent genEvent;

  edm::Handle<CandidateCollection> genParticles;
  edm::SimVertexContainer simVertex;
  edm::SimTrackContainer simTrack;


  double _EtaMin,_EtaMax;

  std::map <std::string, edm::PCaloHitContainer> caloSimHits_;

  edm::Handle<CaloJetCollection> calojets, calojetsDummy;
  edm::Handle<GenJetCollection>  genJets, genJetsDummy;
  edm::Handle<CaloMETCollection> recmet, recmetDummy;
  edm::Handle<GenMETCollection>  genmet, genmetDummy;
  edm::Handle<CaloTowerCollection> caloTowers,caloTowersDummy;

  edm::Handle<EBRecHitCollection> EBRecHits, EBRecHitsDummy;
  edm::Handle<EERecHitCollection> EERecHits, EERecHitsDummy;

  edm::Handle<HBHERecHitCollection> HBHERecHits, HBHERecHitsDummy;
  edm::Handle<HORecHitCollection> HORecHits, HORecHitsDummy;
  edm::Handle<HFRecHitCollection> HFRecHits, HFRecHitsDummy;

  edm::Handle<HBHEDigiCollection> HBHEDigis,HBHEDigisDummy;
  edm::Handle<HODigiCollection> HODigis,HODigisDummy;
  edm::Handle<HFDigiCollection> HFDigis,HFDigisDummy;

  edm::Handle<HcalTBTriggerData> trigger,triggerDummy;



  // input variables
  string _HistName; // Name of histogram file

  bool _Monte;
  bool _PlotTrigger;
  bool _PlotRecHits;
  bool _PlotDigis;
  bool _PlotDijets;
  bool _PlotMCParticles;
  bool _PlotLocalClusters;
  bool ExcludeInteractions_;

  bool IsItMIP_;

  const double etaBarrel() {return 1.4;}

  TFile* m_file; // pointer to Histogram file
  const TString EnergyDir() {return "Channel Energies";}
  const TString PulseDir(){ return "Pulse Shapes";}

  // Trigger histogram labels
  static const char* trigBeam(){return "Beam";}
  static const char* trigIped(){return "In-Spill Ped";}
  static const char* trigOped(){return "Out-Spill Ped";}
  static const char* trigLED(){return "LED";}
  static const char* trigLaser(){return "Laser";}

  // use the map function to access the rest of the histograms
  std::map<TString, TH1*> m_HistNames;
  std::map<TString, TH1*>::iterator hid;

  std::map<TString, TH2*> m_HistNames2D;
  std::map<TString, TH2*>::iterator hid2D;

  //create maps linking histogram pointers to HCAL Channel hits and digis

  std::map<HcalDetId, TH1*> channelmap1;  
  std::map<HcalDetId, TH1*> channelmap2;  
  std::map<HcalDetId, TH1*> digimap1;  
  std::map<HcalDetId, TH1*> digimap2;  

  TString gjetpfx, rjetpfx,gmetpfx, rmetpfx,calopfx;
};

#endif
