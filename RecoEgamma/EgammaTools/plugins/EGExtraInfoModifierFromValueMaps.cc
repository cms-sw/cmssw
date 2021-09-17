#include "RecoEgamma/EgammaTools/interface/EGExtraInfoModifierFromValueMaps.h"

#include "DataFormats/EgammaCandidates/interface/HIPhotonIsolation.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

using EGExtraInfoModifierFromFloatValueMaps = EGExtraInfoModifierFromValueMaps<float>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromFloatValueMaps,
                  "EGExtraInfoModifierFromFloatValueMaps");

using EGExtraInfoModifierFromIntValueMaps = EGExtraInfoModifierFromValueMaps<int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGExtraInfoModifierFromIntValueMaps, "EGExtraInfoModifierFromIntValueMaps");

using EGExtraInfoModifierFromBoolValueMaps = EGExtraInfoModifierFromValueMaps<bool>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromBoolValueMaps,
                  "EGExtraInfoModifierFromBoolValueMaps");

using EGExtraInfoModifierFromBoolToIntValueMaps = EGExtraInfoModifierFromValueMaps<bool, int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromBoolToIntValueMaps,
                  "EGExtraInfoModifierFromBoolToIntValueMaps");

using EGExtraInfoModifierFromUIntValueMaps = EGExtraInfoModifierFromValueMaps<unsigned int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromUIntValueMaps,
                  "EGExtraInfoModifierFromUIntValueMaps");

using EGExtraInfoModifierFromUIntToIntValueMaps = EGExtraInfoModifierFromValueMaps<unsigned int, int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromUIntToIntValueMaps,
                  "EGExtraInfoModifierFromUIntToIntValueMaps");

using EGExtraInfoModifierFromHIPhotonIsolationValueMaps = EGExtraInfoModifierFromValueMaps<reco::HIPhotonIsolation>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromHIPhotonIsolationValueMaps,
                  "EGExtraInfoModifierFromHIPhotonIsolationValueMaps");

#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
using EGExtraInfoModifierFromVIDCutFlowResultValueMaps = EGExtraInfoModifierFromValueMaps<vid::CutFlowResult>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromVIDCutFlowResultValueMaps,
                  "EGExtraInfoModifierFromVIDCutFlowResultValueMaps");

using EGExtraInfoModifierFromEGIDValueMaps = EGExtraInfoModifierFromValueMaps<float, egmodifier::EGID>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromEGIDValueMaps,
                  "EGExtraInfoModifierFromEGIDValueMaps");

using EGExtraInfoModifierFromPackedCandPtrValueMaps = EGExtraInfoModifierFromValueMaps<edm::Ptr<pat::PackedCandidate> >;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGExtraInfoModifierFromPackedCandPtrValueMaps,
                  "EGExtraInfoModifierFromPackedCandPtrValueMaps");
