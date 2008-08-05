#ifndef HLTriggerOffline_Muon_HLTMuonRate_H
#define HLTriggerOffline_Muon_HLTMuonRate_H

/** \class HLTMuonRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author  Sho Maruyama  (copied from J. Alcaraz
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

class HLTMuonRate {
public:
  HLTMuonRate(const edm::ParameterSet& pset, int Index);
  virtual ~HLTMuonRate();
  void analyze(const edm::Event & event);
  void BookHistograms() ;
  void WriteHistograms() ;
  MonitorElement* BookIt(char name[], char title[], int bins, float min, float max) ;
private:
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
  int this_event_weight;
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
  std::pair<double,double> getGenAngle(double eta, double phi, HepMC::GenEvent evt, double DR=0.4 );
  std::pair<double,double> getRecoAngle(double eta, double phi,reco::TrackCollection tracks, double DR=0.4 );
  MonitorElement *NumberOfEvents,*NumberOfL1Events;
  int theNumberOfEvents,theNumberOfL1Events;
  std::string theRootFileName;
};
#endif
