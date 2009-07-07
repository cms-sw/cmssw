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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <vector>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;
using namespace std;
using namespace math;
class HLTMuonRate {
public:
  HLTMuonRate(const edm::ParameterSet& pset, int Index);
  virtual ~HLTMuonRate();
  void analyze(const edm::Event & event);
  void BookHistograms() ;
  void WriteHistograms() ;
  MonitorElement* BookIt(char name[], char title[], int bins, float min, float max) ;
private:
  edm::InputTag theL1CollectionLabel, InputLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  double theL1ReferenceThreshold,theHLTReferenceThreshold;
  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  double theCrossSection;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  unsigned int theNbins;
  int this_event_weight;
  DQMStore* dbe_;  
  MonitorElement* hL1DR, *hL2DR, *hL3DR;
  MonitorElement* hL1eff;
  MonitorElement* hPtNor;
  MonitorElement* hPhiNor;
  MonitorElement* hEtaNor;
  MonitorElement* hL1rate;
  MonitorElement* hL1pt;
  MonitorElement* hL1Eta;
  MonitorElement* hL1Phi;
  MonitorElement* hSteps;
  std::vector <MonitorElement*> hHLTeff;
  std::vector <MonitorElement*> hHLTrate;
  std::vector <MonitorElement*> hHLTpt;
  std::vector <MonitorElement*> hHLTeta;
  std::vector <MonitorElement*> hHLTphi;
  std::pair<double,double> getAngle(double eta, double phi, Handle< vector<XYZTLorentzVectorD> >& refVector );
  MonitorElement *NumberOfEvents,*NumberOfL1Events;
  int theNumberOfEvents,theNumberOfL1Events;
  std::string theRootFileName;
};
#endif
