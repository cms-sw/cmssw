#ifndef RecoTauTag_TauTagTools_PFTauQualityCutWrapper_h
#define RecoTauTag_TauTagTools_PFTauQualityCutWrapper_h

/*
 * THIS CLASS IS DEPRECATED!!
 *
 */

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

class PFTauQualityCutWrapper {
   public:
      PFTauQualityCutWrapper(const edm::ParameterSet& pset)
      {
         isoQCuts.fill(pset.getParameter<edm::ParameterSet>("isolationQualityCuts"));
         signalQCuts.fill(pset.getParameter<edm::ParameterSet>("signalQualityCuts"));
      }

      struct QualityCutSet {
         // charged hadron & track cuts
         bool useTracksInsteadOfPF;
         double minTrackPt;
         double maxTrackChi2;
         uint32_t minTrackPixelHits;
         uint32_t minTrackHits; 
         double maxTransverseImpactParameter;
         double maxDeltaZ;
         // gamma cuts
         double minGammaEt;
         void fill(const edm::ParameterSet& pset) {
            useTracksInsteadOfPF         = pset.getParameter<bool>("useTracksInsteadOfPFHadrons");
            minTrackPt                   = pset.getParameter<double>("minTrackPt");
            maxTrackChi2                 = pset.getParameter<double>("maxTrackChi2");
            minTrackPixelHits            = pset.getParameter<uint32_t>("minTrackPixelHits");
            minTrackHits                 = pset.getParameter<uint32_t>("minTrackHits");
            maxTransverseImpactParameter = pset.getParameter<double>("maxTransverseImpactParameter");
            maxDeltaZ                    = pset.getParameter<double>("maxDeltaZ");
            minGammaEt                   = pset.getParameter<double>("minGammaEt"); 
         }
      };

      /// retrieve filtered isolation charged objects from the pfTau
      void isolationChargedObjects(const reco::PFTau&, const reco::Vertex&, std::vector<reco::LeafCandidate>&);
      void isolationPUObjects(const reco::PFTau&, const reco::Vertex&, std::vector<reco::LeafCandidate>&);
      /// retrieve filtered isolation gamma objects from the pfTau
      void isolationGammaObjects(const reco::PFTau&, std::vector<reco::LeafCandidate>&);

      /// retrieve filtered signal charged objects from the pfTau
      void signalChargedObjects(const reco::PFTau&, const reco::Vertex&, std::vector<reco::LeafCandidate>&);
      /// retrieve filtered signal gamma objects from the pfTau
      void signalGammaObjects(const reco::PFTau&, std::vector<reco::LeafCandidate>&);

   private:
      QualityCutSet isoQCuts;
      QualityCutSet signalQCuts;
};

#endif
