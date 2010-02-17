#ifndef DQMOffline_Trigger_HLTMuonOverlap_H
#define DQMOffline_Trigger_HLTMuonOverlap_H

/** \class HLTMuonOverlap
 *
 *  \author  M. Vander Donckt  (starting from Christos analyzeTrigegrResults)
 */

// Base Class Headers
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

#include "TH1F.h"

class HLTMuonOverlap {

public:
  HLTMuonOverlap(const edm::ParameterSet&);
  void begin  ( );
  void analyze(const edm::Event & event);
  void finish ( );

private:
  bool TrigResultsIn;
  edm::InputTag TrigResLabel_;
  unsigned int size;
  unsigned int Ntp; // # of trigger paths (should be the same for all events!)
  unsigned int Nall_trig; // # of all triggered events
  unsigned int Nevents; // # of analyzed events 
  typedef std::map<std::string, unsigned int> trigPath;
  double theCrossSection, theLuminosity;
  bool init_; 
  trigPath Ntrig; // # of triggered events per path

  // # of cross-triggered events per path
  // (pairs with same name correspond to unique trigger rates for that path)
  std::map<std::string, trigPath> Ncross;

  // whether a trigger path has fired for given event
  // (variable with event-scope)
  std::map<std::string, bool> fired; 
};
#endif
