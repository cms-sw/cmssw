#ifndef VBFHiggsTo2TauLJetSkim_h
#define VBFHiggsTo2TauLJetSkim_h

/** \class VBFHiggsTo2TauLJetSkim
 *
 * From HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkim.h
 *
 * \author S.Greder
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
#include "FWCore/ParameterSet/interface/InputTag.h"
   
using namespace edm;
using namespace std;   


class VBFHiggsTo2TauLJetSkim : public edm::EDFilter {
  
 public:
  // Constructor
  explicit VBFHiggsTo2TauLJetSkim(const edm::ParameterSet&);

  // Destructor
  ~VBFHiggsTo2TauLJetSkim();

  /// Get event properties to send to builder to fill seed collection
  virtual bool filter(edm::Event&, const edm::EventSetup&);

 private:
  bool debug;

  // To keep track of statistics:
  int nEvents, nSelectedEvents;

  // Cuts 
};

#endif
