#ifndef ANALYSIS_JET_MYJET_H
#define ANALYSIS_JET_MYJET_H 1

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"

#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

/** \class JetAnalysis
  *  
  * $Date: 2006/08/31 15:08:26 $
  * $Revision: 1.1 $
  * \author L. Apanasevich - UIC and Anwar Bhatti
  */
class JetAnalysis {
public:
  JetAnalysis(); 
  /** Setup the analysis to put the histograms into HistoFile and focus on
      ieta,iphi for analysis.
  */
  void setup(const edm::ParameterSet& pSet);
  void fillHist(const TString& histName, const Double_t& x,const Double_t& wt=1.0);
  void fillHist1D(const TString& histName, const Double_t& x,const Double_t& wt=1.0);
  void fillHist2D(const TString& histName, const Double_t& x,const Double_t& y,const Double_t& wt=1.0);


  /** Analyze the Data */
  void analyze(const CaloJetCollection& rjets,
	       const GenJetCollection& gjets,
	       const CaloMETCollection& rmets,
	       const METCollection& gmets,
	       const CaloTowerCollection& caloTowers,
	       const HepMC::GenEvent mctruth,
	       const HBHERecHitCollection& hbhe_hits, 
	       const HBHEDigiCollection& hbhe_digis, 
	       const HORecHitCollection& ho_hits, 
	       const HODigiCollection& ho_digis, 
	       const HFRecHitCollection& hf_hits, 
	       const HFDigiCollection& hf_digis, 
	       const HcalTBTriggerData& trigger,
	       const CaloGeometry& geom);

  void dummyAnalyze(
	       const CaloGeometry& geom);
  /** Finalization (close files, etc) */
  void done();


  void bookHistograms();
  void bookGeneralHistograms();

  void bookTBTriggerHists();
  void fillTBTriggerHists(const HcalTBTriggerData& trigger);

  void bookCaloTowerHists();
  void fillCaloTowerHists(const CaloTowerCollection& caloTowers);

  void bookForId(const HcalDetId& id);
  void bookForId_TS(const HcalDetId& id);

  void bookMetHists(const TString& prefix);
  template <typename T> void fillMetHists(const T& mets, const TString& prefx);

  void bookJetHistograms(const TString& prefix);

  template <typename T> void fillJetHists(const T& jets, const TString& prefx);
  template <typename T> void fillRecHits(const T& hits);
  template <typename T> void fillDigis(const T& digis);

  void bookCalculateEfficiency();
  void CalculateEfficiency(GenJetCollection& genjets,CaloJetCollection& calojets);

  void bookDiJetBalance(const TString& prefix);
  template <typename T> void DiJetBalance(const T& jets, const TString& prefix);

  void bookMCParticles();
  void fillMCParticles(const HepMC::GenEvent mctruth);
  void bookMCParticles(const TString& prefix );
  void fillMCParticlesInsideJet(const HepMC::GenEvent genEvent,const GenJetCollection& genjets);


private:

  // input variables
  string _HistName; // Name of histogram file
  bool _Monte,_PlotRecHits,_PlotDigis;
  double _EtaMin,_EtaMax;

  int evtCounter;
  bool doGenJets, doGenMets, doMCTruth, doTBTrigger;

  const float etaBarrel() {return 1.4;}

  TFile* m_file; // pointer to Histogram file
  const TString EnergyDir() {return "Channel Energies";}
  const TString PulseDir(){ return "Pulse Shapes";}

  // Trigger histogram labels
  static const char* trigBeam(){return "Beam";}
  static const char* trigIped(){return "In-Spill Ped";}
  static const char* trigOped(){return "Out-Spill Ped";}
  static const char* trigLED(){return "LED";}
  static const char* trigLaser(){return "Laser";}

  // histogram declarations
  TH1* m_Cntr; // Simple single histogram

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
