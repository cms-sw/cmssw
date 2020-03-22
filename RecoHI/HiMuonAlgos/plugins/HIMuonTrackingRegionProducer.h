#ifndef RecoHI_HiTracking_HIMuonTrackingRegionProducer_H
#define RecoHI_HiTracking_HIMuonTrackingRegionProducer_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

class HIMuonTrackingRegionProducer : public TrackingRegionProducer {
public:
  HIMuonTrackingRegionProducer(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) {
    // get parameters from PSet
    theMuonSource = cfg.getParameter<edm::InputTag>("MuonSrc");
    theMuonSourceToken = iC.consumes<reco::TrackCollection>(theMuonSource);

    // initialize region builder
    edm::ParameterSet regionBuilderPSet = cfg.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
    theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet, iC);

    // initialize muon service proxy
    edm::ParameterSet servicePSet = cfg.getParameter<edm::ParameterSet>("ServiceParameters");
  }

  ~HIMuonTrackingRegionProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("MuonSrc", edm::InputTag(""));

    edm::ParameterSetDescription descRegion;
    MuonTrackingRegionBuilder::fillDescriptionsOffline(descRegion);
    desc.add("MuonTrackingRegionBuilder", descRegion);

    edm::ParameterSetDescription descService;
    descService.setAllowAnything();
    desc.add<edm::ParameterSetDescription>("ServiceParameters", descService);

    descriptions.add("HiTrackingRegionEDProducer", desc);
  }

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& ev,
                                                        const edm::EventSetup& es) const override {
    // initialize output vector of tracking regions
    std::vector<std::unique_ptr<TrackingRegion> > result;

    // initialize the region builder
    theRegionBuilder->setEvent(ev);

    // get stand-alone muon collection
    edm::Handle<reco::TrackCollection> muonH;
    ev.getByToken(theMuonSourceToken, muonH);

    // loop over all muons and add a tracking region for each
    // that passes the requirements specified to theRegionBuilder
    unsigned int nMuons = muonH->size();
    //std::cout << "there are " << nMuons << " muon(s)" << std::endl;

    // TO DO: this can be extended further to a double-loop
    // over all combinations of muons, returning tracking regions
    // for pairs that pass some loose invariant mass cuts
    for (unsigned int imu = 0; imu < nMuons; imu++) {
      reco::TrackRef muRef(muonH, imu);
      //std::cout << "muon #" << imu << ": pt=" << muRef->pt() << std::endl;
      result.push_back(theRegionBuilder->region(muRef));
    }

    return result;
  }

private:
  edm::InputTag theMuonSource;
  edm::EDGetTokenT<reco::TrackCollection> theMuonSourceToken;
  MuonTrackingRegionBuilder* theRegionBuilder;
};

#endif
