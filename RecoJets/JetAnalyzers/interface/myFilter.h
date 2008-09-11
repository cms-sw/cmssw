#ifndef RecoJets_myFilter_h
#define RecoJets_myFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <string>

class myFilter : public edm::EDFilter {

public:
  myFilter(const edm::ParameterSet& );
  virtual ~myFilter();
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
  virtual void beginJob(edm::EventSetup const&);
  virtual void endJob();

private:
  int _rejectedEvt;
  int _nEvent;
  std::string CaloJetAlgorithm;

};

#endif
