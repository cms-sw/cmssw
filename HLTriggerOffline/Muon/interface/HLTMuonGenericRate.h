#ifndef HLTriggerOffline_Muon_HLTMuonGenericRate_H
#define HLTriggerOffline_Muon_HLTMuonGenericRate_H

/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author  M. Vander Donckt, J. Klukas  (copied from J. Alcaraz)
 */

// Base Class Headers

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <vector>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class HLTMuonGenericRate {
public:
  /// Constructor
  HLTMuonGenericRate(const edm::ParameterSet& pset, int index);

  /// Destructor
  virtual ~HLTMuonGenericRate();

  // Operations

  void analyze(const edm::Event & event);

  void BookHistograms() ;
  void WriteHistograms() ;
  void SetCurrentFolder( TString folder );
  MonitorElement* BookIt( char name[], char title[], int bins, float min, 
			  float max) ;
  MonitorElement* BookIt( TString name, TString title, 
					      int Nbins, float Min, float Max);

private:

  // Input from cfg file
  edm::InputTag theL1CollectionLabel, theGenLabel, theRecoLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  double theL1ReferenceThreshold,theHLTReferenceThreshold;
  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  bool useMuonFromGenerator,useMuonFromReco;
  double theCrossSection;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  unsigned int theNbins;
  int thisEventWeight;

  // Histograms
  DQMStore* dbe_;  
  MonitorElement* hL1DR, *hL2DR, *hL3DR;
  MonitorElement* hL1eff;
  MonitorElement* hL1MCeff;
  MonitorElement* hL1RECOeff;
  MonitorElement* hMCptnor;
  MonitorElement* hMCphinor;
  MonitorElement* hMCetanor;
  MonitorElement* hRECOptnor;
  MonitorElement* hRECOphinor;
  MonitorElement* hRECOetanor;
  MonitorElement* hL1rate;
  MonitorElement* hL1pt;
  MonitorElement* hL1etaMC;
  MonitorElement* hL1phiMC;
  MonitorElement* hL1etaRECO;
  MonitorElement* hL1phiRECO;
  MonitorElement* hSteps;
  std::vector <MonitorElement*> hHLTeff;
  std::vector <MonitorElement*> hHLTMCeff;
  std::vector <MonitorElement*> hHLTRECOeff;
  std::vector <MonitorElement*> hHLTrate;
  std::vector <MonitorElement*> hHLTpt;
  std::vector <MonitorElement*> hHLTetaMC;
  std::vector <MonitorElement*> hHLTphiMC;
  std::vector <MonitorElement*> hHLTetaRECO;
  std::vector <MonitorElement*> hHLTphiRECO;

  HepMC::GenEvent::particle_const_iterator theAssociatedGenPart;
  reco::TrackCollection::const_iterator theAssociatedRecoPart;
  const HepMC::GenEvent* evt;

  std::pair<double,double> getGenAngle( double eta, double phi, 
			   HepMC::GenEvent evt, double DR=0.4 );
  std::pair<double,double> getRecAngle( double eta, double phi, 
			   reco::TrackCollection tracks, double DR=0.4 );

  MonitorElement *NumberOfEvents, *NumberOfL1Events;
  int theNumberOfEvents,theNumberOfL1Events;
  std::string theRootFileName;

};
#endif
