#ifndef EVENTFILTER_UTILITIES_PLUGINS_EVFRECORDUNPACKER_H
#define EVENTFILTER_UTILITIES_PLUGINS_EVFRECORDUNPACKER_H

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>

namespace evf{

  class EvFRecordUnpacker : public edm::EDAnalyzer
  {
  public:
    EvFRecordUnpacker(const edm::ParameterSet&);
    ~EvFRecordUnpacker();
    void analyze(const edm::Event & e, const edm::EventSetup& c);

  private:
    edm::InputTag label_;
    TH1F *node_usage_;
    TH1F *l1_rb_delay_;
    TH2F *corre_;
    TFile * f_;
  };
}
#endif
