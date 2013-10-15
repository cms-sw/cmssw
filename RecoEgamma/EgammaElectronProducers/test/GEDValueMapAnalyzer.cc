#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <iostream>

class GEDValueMapAnalyzer : public edm::EDAnalyzer
{
public: 
  GEDValueMapAnalyzer(const edm::ParameterSet&);
  ~GEDValueMapAnalyzer();
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void analyze(const edm::Event & iEvent,const edm::EventSetup & c);

private:
  edm::InputTag inputTagValueMapElectrons_;
  edm::InputTag inputTagPFCandidates_;
  
};

GEDValueMapAnalyzer::GEDValueMapAnalyzer(const edm::ParameterSet& iConfig) {
  inputTagPFCandidates_ = iConfig.getParameter<edm::InputTag>("PFCandidates");
  inputTagValueMapElectrons_ = iConfig.getParameter<edm::InputTag>("ElectronValueMap"); 
}

GEDValueMapAnalyzer::~GEDValueMapAnalyzer() {;}

void GEDValueMapAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& ){;}


void GEDValueMapAnalyzer::analyze(const edm::Event & iEvent,const edm::EventSetup & c) {
  edm::Handle<reco::PFCandidateCollection> pfCandidatesH;
  bool found=iEvent.getByLabel(inputTagPFCandidates_,pfCandidatesH);
  if(!found ) {
    std::ostringstream err;
    err<<" cannot get PFCandidates: "
       <<inputTagPFCandidates_<<std::endl;
    edm::LogError("PFIsoReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }

  // Get the value maps

  edm::Handle<edm::ValueMap<reco::GsfElectronRef> > electronValMapH;
  found = iEvent.getByLabel(inputTagValueMapElectrons_,electronValMapH);
  const edm::ValueMap<reco::GsfElectronRef> & myElectronValMap(*electronValMapH);
  
  std::cout << " Read Electron Value Map " << myElectronValMap.size() << std::endl;


  unsigned ncandidates=pfCandidatesH->size();
  for(unsigned ic=0 ; ic < ncandidates ; ++ic) {
    // check if it has a GsfTrack
    const reco::PFCandidate & cand((*pfCandidatesH)[ic]);
    if(!cand.gsfTrackRef().isNonnull()) continue;
    
    reco::PFCandidateRef pfRef(pfCandidatesH,ic);
    // get the GsfElectronRef from the ValueMap
    reco::GsfElectronRef gsfRef=myElectronValMap[pfRef];
    
    //basic check
    std::cout << " Comparing GsfTrackRef from GsfElectron and PFCandidate " ;
    if(gsfRef->gsfTrack()==cand.gsfTrackRef()) std::cout << " OK " << std::endl;
    else
      std::cout << " Problem different Ref" << std::endl;
  }

}


DEFINE_FWK_MODULE(GEDValueMapAnalyzer);
