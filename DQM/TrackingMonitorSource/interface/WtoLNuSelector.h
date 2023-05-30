#ifndef DQM_TrackingMonitorSource_WtoLNuSelector_h
#define DQM_TrackingMonitorSource_WtoLNuSelector_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TLorentzVector.h"

// Forward declaration
class TH1D;
namespace {
  class BeamSpot;
}

class WtoLNuSelector : public edm::stream::EDFilter<> {
public:
  explicit WtoLNuSelector(const edm::ParameterSet&);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;
  double getMt(const TLorentzVector& vlep, const reco::PFMET& obj);

private:
  // module config parameters
  const edm::InputTag electronTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag muonTag_;
  const edm::InputTag pfmetTag_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const edm::EDGetTokenT<reco::PFMETCollection> pfmetToken_;
};
#endif
