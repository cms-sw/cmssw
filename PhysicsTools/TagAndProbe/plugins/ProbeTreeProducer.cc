// -*- C++ -*-
//
// Package:    ProbeTreeProducer
// Class:      ProbeTreeProducer
// 
/**\class ProbeTreeProducer ProbeTreeProducer.cc 

 Description: TTree producer based on input probe parameters

 Implementation:
     <Notes on implementation>
*/

#include <memory>
#include <ctype.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/TagAndProbe/interface/BaseTreeFiller.h"
#include <set>
#include "FWCore/ParameterSet/interface/Registry.h"

class ProbeTreeProducer : public edm::EDFilter {
  public:
    explicit ProbeTreeProducer(const edm::ParameterSet&);
    ~ProbeTreeProducer();

  private:
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    /// InputTag to the collection of all probes
    edm::InputTag probesTag_;

    /// The selector object
    StringCutObjectSelector<reco::Candidate, true> cut_;

    /// Specifies whether this module should filter
    bool filter_;

    /// The object that actually computes variables and fills the tree for the probe
    std::auto_ptr<tnp::BaseTreeFiller> probeFiller_;
};

ProbeTreeProducer::ProbeTreeProducer(const edm::ParameterSet& iConfig) :
  probesTag_(iConfig.getParameter<edm::InputTag>("src")),
  cut_(iConfig.existsAs<std::string>("cut") ? iConfig.getParameter<std::string>("cut") : ""),
  filter_(iConfig.existsAs<bool>("filter") ? iConfig.getParameter<bool>("filter") : false),
  probeFiller_(new tnp::BaseTreeFiller("probe_tree", iConfig))
{
}

ProbeTreeProducer::~ProbeTreeProducer(){
}

bool ProbeTreeProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  bool result = !filter_;
  edm::Handle<reco::CandidateView> probes;
  iEvent.getByLabel(probesTag_, probes);
  probeFiller_->init(iEvent);
  for (size_t i = 0; i < probes->size(); ++i){
    const reco::CandidateBaseRef &probe = probes->refAt(i);
    if(cut_(*probe)){
      probeFiller_->fill(probe);
      result = true;
    }
  }
  return result;
}

void ProbeTreeProducer::endJob(){
    // ask to write the current PSet info into the TTree header
    probeFiller_->writeProvenance(edm::getProcessParameterSet());
}

//define this as a plug-in
DEFINE_FWK_MODULE(ProbeTreeProducer);

