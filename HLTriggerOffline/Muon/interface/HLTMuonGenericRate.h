#ifndef HLTriggerOffline_Muon_HLTMuonGenericRate_H
#define HLTriggerOffline_Muon_HLTMuonGenericRate_H

/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author  M. Vander Donckt  (copied fromJ. Alcaraz
 */

// Base Class Headers

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <vector>
#include "TDirectory.h"

class TH1F;

class HLTMuonGenericRate {
public:
  /// Constructor
  HLTMuonGenericRate(const edm::ParameterSet& pset, int Index);

  /// Destructor
  virtual ~HLTMuonGenericRate();

  // Operations

  void analyze(const edm::Event & event);

  virtual void BookHistograms() ;
  virtual void FillHistograms() ;
  virtual void WriteHistograms() ;

private:
  // Input from cfg file
  edm::InputTag theL1CollectionLabel, theGenLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  double theL1ReferenceThreshold,theHLTReferenceThreshold;
  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  bool useMuonFromGenerator;
  double theCrossSection;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  unsigned int theNbins;
  int this_event_weight;
  // Histograms
  TH1F* hL1eff;
  TH1F* hL1MCeff;
  TH1F* hMCptnor;
  TH1F* hMCphinor;
  TH1F* hMCetanor;
  TH1F* hL1rate;
  TH1F* hL1pt;
  TH1F* hL1eta;
  TH1F* hL1phi;
  std::vector <TH1F*> hHLTeff;
  std::vector <TH1F*> hHLTMCeff;
  std::vector <TH1F*> hHLTrate;
  std::vector <TH1F*> hHLTpt;
  std::vector <TH1F*> hHLTeta;
  std::vector <TH1F*> hHLTphi;
  HepMC::GenEvent::particle_const_iterator theAssociatedGenPart;
  const HepMC::GenEvent* evt;
  std::pair<double,double> getGenAngle(edm::RefToBase<reco::Candidate> candref, HepMC::GenEvent evt );
  double theNumberOfEvents;
  TDirectory *ratedir;  
  TDirectory *distribdir;
  TDirectory *top;

};
#endif
