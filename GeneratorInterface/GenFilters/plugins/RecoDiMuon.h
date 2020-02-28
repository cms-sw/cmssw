#ifndef RecoDiMuon_h
#define RecoDiMuon_h

/** \class RecoDiMuon
 *
 *  
 *  This class is an EDFilter choosing reconstructed di-muons
 *
 *
 *  \author Chang Liu  -  Purdue University
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RecoDiMuon : public edm::EDFilter {
public:
  explicit RecoDiMuon(const edm::ParameterSet&);
  ~RecoDiMuon() override;
  void endJob() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  edm::InputTag muonLabel_;
  double singleMuonPtMin_;
  double diMuonPtMin_;
  unsigned int nEvents_;
  unsigned int nAccepted_;
};
#endif
