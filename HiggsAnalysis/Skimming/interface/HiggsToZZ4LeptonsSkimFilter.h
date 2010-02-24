#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkimFilter
#define HiggsAnalysis_HiggsToZZ4LeptonsSkimFilter

/* \class HiggsTo4LeptonsSkimFilter
 *
 *
 * Filter to select 4 lepton events based on the
 * 1 or 2 electron or 1 or 2 muon HLT trigger, 
 * and four leptons (no flavour requirement).
 * No charge requirements are applied on event.
 *
 * \author Dominique Fortin - UC Riverside
 *modified by N. De Filippis - LLR - Ecole Polytechnique
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace edm;
using namespace std;

class HiggsToZZ4LeptonsSkimFilter : public edm::EDFilter {
  
 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkimFilter(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsSkimFilter();

  /// Get event properties to send to builder to fill seed collection
  virtual bool filter(edm::Event&, const edm::EventSetup& );

 private:
  string HLTinst_;
  vector<string> HLTflag_;
  string Skiminst_,Skimflag_;
  int nSelectedSkimEvents;
  vector<int> nSelectedEvents;

};

#endif
