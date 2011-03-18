#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkimDiLeptonProducer
#define HiggsAnalysis_HiggsToZZ4LeptonsSkimDiLeptonProducer

/* \class HiggsTo4LeptonsSkimDiLeptonProducer
 *
 * Author: N. De Filippis - Politecnico and INFN Bari
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace edm;
using namespace std;

class HiggsToZZ4LeptonsSkimDiLeptonProducer : public edm::EDProducer {
  
 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkimDiLeptonProducer(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsSkimDiLeptonProducer();

 private:
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
  virtual void endJob();

  float cutPt, cutEta;
  edm::InputTag RECOcollOS,RECOcollSS,RECOcollZMM,RECOcollZEE;

};

#endif
