#include "RecoEgamma/EgammaTools/plugins/EGExtraInfoModifierFromValueMaps.h"

using EGExtraInfoModifierFromFloatValueMapsTest = EGExtraInfoModifierFromValueMaps<float>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromFloatValueMapsTest,
		  "EGExtraInfoModifierFromFloatValueMapsTest");


using EGExtraInfoModifierFromIntValueMapsTest = EGExtraInfoModifierFromValueMaps<int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromIntValueMapsTest,
		  "EGExtraInfoModifierFromIntValueMapsTest");

using EGExtraInfoModifierFromBoolValueMaps = EGExtraInfoModifierFromValueMaps<bool>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromBoolValueMaps,
		  "EGExtraInfoModifierFromBoolValueMaps");

using EGExtraInfoModifierFromBoolToIntValueMaps = EGExtraInfoModifierFromValueMaps<bool,int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromBoolToIntValueMaps,
		  "EGExtraInfoModifierFromBoolToIntValueMaps");

using EGExtraInfoModifierFromUIntValueMaps = EGExtraInfoModifierFromValueMaps<unsigned int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromUIntValueMaps,
		  "EGExtraInfoModifierFromUIntValueMaps");

using EGExtraInfoModifierFromUIntToIntValueMaps = EGExtraInfoModifierFromValueMaps<unsigned int,int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromUIntToIntValueMaps,
		  "EGExtraInfoModifierFromUIntToIntValueMaps");



#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
using EGExtraInfoModifierFromVIDCutFlowResultValueMaps = EGExtraInfoModifierFromValueMaps<vid::CutFlowResult>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromVIDCutFlowResultValueMaps,
		  "EGExtraInfoModifierFromVIDCutFlowResultValueMaps");
