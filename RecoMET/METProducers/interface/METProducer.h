// -*- C++ -*-
//
// Package:    METProducers
// Class:      METProducer
//
/**\class METProducer METProducer.h RecoMET/METProducers/interface/METProducer.h

 Description: An EDProducer which produces MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Rick Cavanaugh
//         Created:  May 14, 2005
//
//

#ifndef METProducer_h
#define METProducer_h

#include <string.h>
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TCMETAlgo;

namespace metsig {
    class SignAlgoResolutions;
}

namespace cms
{
  class METProducer: public edm::EDProducer
    {
    public:
      explicit METProducer(const edm::ParameterSet&);
      virtual ~METProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      void produce_CaloMET(edm::Event& event);
      void produce_TCMET(edm::Event& event, const edm::EventSetup& setup);
      void produce_PFMET(edm::Event& event);
      void produce_PFClusterMET(edm::Event& event);
      void produce_GenMET(edm::Event& event);
      void produce_else(edm::Event& event);


      edm::InputTag inputLabel;
      std::string inputType;
      std::string METtype;
      std::string alias;

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      //Calculate MET Significance (not necessary at HLT)
      bool calculateSignificance_;
      metsig::SignAlgoResolutions *resolutions_;
      edm::InputTag jetsLabel_; //used for jet-based significance calculation

      //Use HF in CaloMET calculation?
      bool noHF;

      //Use an Et threshold on all of the objects in the CaloMET calculation?
      double globalThreshold;

      //Use only fiducial GenParticles in GenMET calculation?
      bool onlyFiducial;

      //Use only fiducial GenParticles and apply thresholdin GenMET fraction calculation?
      bool applyFiducialThresholdForFractions;

      //Use Pt instaed of Et
      bool usePt;

      // for pfMET
      edm::EDGetTokenT<edm::View<reco::PFJet> > jetToken_;


      TCMETAlgo tcMetAlgo_;

      // for tcMET
      edm::EDGetTokenT<reco::MuonCollection> muonToken_;
      edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
      edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
      edm::EDGetTokenT<reco::TrackCollection> trackToken_;
      edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
      edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
      edm::EDGetTokenT<reco::PFClusterCollection> clustersECALToken_;
      edm::EDGetTokenT<reco::PFClusterCollection> clustersHCALToken_;
      edm::EDGetTokenT<reco::PFClusterCollection> clustersHFEMToken_;
      edm::EDGetTokenT<reco::PFClusterCollection> clustersHFHADToken_;
      edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > muonDepValueMapToken_;
      edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > tcmetDepValueMapToken_;

    };
}

#endif // METProducer_h
