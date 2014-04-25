#ifndef ELECTRONMAKER_H
#define ELECTRONMAKER_H

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DQM/PhysicsHWW/interface/HWW.h"


class ElectronMaker {

  public:

    ElectronMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    double electronIsoValuePF(const reco::GsfElectron& el, const reco::Vertex& vtx, float coner, float minptn, float dzcut,
                              float footprintdr, float gammastripveto, float elestripveto, int filterId);
    void PFIsolation2012(const reco::GsfElectron& el, const reco::VertexCollection* vertexCollection,
                             const int vertexIndex, const float &R, float &pfiso_ch, float &pfiso_em, float &pfiso_nh);


    edm::EDGetTokenT<reco::TrackCollection>         TrackCollection_;
    edm::EDGetTokenT<reco::GsfTrackCollection>      GSFTrackCollection_;
    edm::EDGetTokenT<edm::View<reco::GsfElectron> > GSFElectron_;
    edm::EDGetTokenT<reco::GsfElectronCollection>   GSFElectronCollection_;
    edm::EDGetTokenT<reco::PFCandidateCollection>   PFCandidateCollection_;
    edm::EDGetTokenT<reco::VertexCollection>        thePVCollection_;
    edm::EDGetTokenT<reco::BeamSpot>                BeamSpot_;
    edm::EDGetTokenT<reco::ConversionCollection>    ConversionCollection_;
    edm::EDGetTokenT<EcalRecHitCollection>          ClusterToken1_;
    edm::EDGetTokenT<EcalRecHitCollection>          ClusterToken2_;

    edm::Handle<reco::PFCandidateCollection> pfCand_h;

};

#endif

