#include "RecoEgamma/EgammaTools/plugins/EGExtraInfoModifierFromValueMaps.h"

using EGExtraInfoModifierFromFloatValueMapsTest = EGExtraInfoModifierFromValueMaps<float>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromFloatValueMapsTest,
		  "EGExtraInfoModifierFromFloatValueMapsTest");


using EGExtraInfoModifierFromIntValueMapsTest = EGExtraInfoModifierFromValueMaps<int>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromIntValueMapsTest,
		  "EGExtraInfoModifierFromIntValueMapsTest");

using EGExtraInfoModifierFromBoolValueMapsTest = EGExtraInfoModifierFromValueMaps<bool>;
DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromBoolValueMapsTest,
		  "EGExtraInfoModifierFromBoolValueMapsTest");
