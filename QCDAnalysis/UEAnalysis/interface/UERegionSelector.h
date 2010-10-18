#ifndef UEAnalysis_UERegionSelector_h
#define UEAnalysis_UERegionSelector_h

// system include files
#include <memory>

// user include files
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <CommonTools/UtilAlgos/interface/TFileService.h>
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>
#include <numeric>
#include <TROOT.h>
#include <TFile.h>
#include <TMath.h>

#include <DataFormats/Common/interface/Handle.h>
#include <DataFormats/Candidate/interface/Candidate.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/TrackReco/interface/TrackFwd.h>

class UERegionSelector : public edm::EDProducer {
public:
  explicit UERegionSelector(const edm::ParameterSet&);
  ~UERegionSelector();
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::InputTag jetCollName;
  //  edm::InputTag particleCollName;
  edm::InputTag trackCollName;

  edm::Handle< edm::View<reco::Candidate> > jetHandle;
  //  edm::Handle< edm::View<reco::Candidate> > particleHandle;
  edm::Handle< edm::View<reco::Track> > trackHandle;

  double deltaPhiByPiMinJetParticle;
  double deltaPhiByPiMaxJetParticle;
};

#endif
