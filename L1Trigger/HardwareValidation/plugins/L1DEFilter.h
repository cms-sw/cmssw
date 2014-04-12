#ifndef L1DEFILTER_H
#define L1DEFILTER_H

/*\class L1DEFilter
 *\description L1 trigger data|emulation event filter
 *\author Nuno Leonardo (CERN)
 *\date 07.06
 */

// system includes
#include <memory>

// common includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// d|e record
#include "DataFormats/L1Trigger/interface/L1DataEmulRecord.h"


class L1DEFilter : public edm::EDFilter {
  
 public:
  explicit L1DEFilter(const edm::ParameterSet&);
  ~L1DEFilter();
  
 private:
  virtual void beginJob(void) {};
  //virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  edm::InputTag DEsource_;
  std::vector<unsigned int> flagSys_;
  int nEvt_;
  int nAgree_;

};

#endif
