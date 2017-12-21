#include "RecoBTag/DeepFlavour/interface/JetConverter.h"
#include "DataFormats/BTauReco/interface/JetFeatures.h"

#include "DataFormats/JetReco/interface/Jet.h"

namespace btagbtvdeep {


      static void JetToFeatures(const reco::Jet & jet,
                                JetFeatures & jet_features) {
        jet_features.pt = jet.pt(); // uncorrected
        jet_features.eta = jet.eta();
        jet_features.mass = jet.mass();
        jet_features.energy = jet.energy();
      }

}


