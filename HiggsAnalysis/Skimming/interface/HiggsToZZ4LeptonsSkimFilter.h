#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkimFilter
#define HiggsAnalysis_HiggsToZZ4LeptonsSkimFilter

/* \class HiggsTo4LeptonsSkimFilter
 *
 * Author: N. De Filippis - Politecnico and INFN Bari
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
  bool useHLT;
  string HLTinst_;
  vector<string> HLTflag_;

  bool useDiLeptonSkim,useTriLeptonSkim;
  string SkimDiLeptoninst_,SkimDiLeptonflag_;
  string SkimTriLeptoninst_,SkimTriLeptonflag_;

  int nDiLeptonSkimEvents,nTriLeptonSkimEvents,nSelectedSkimEvents;
  vector<int> nSelectedEvents;

};

#endif
