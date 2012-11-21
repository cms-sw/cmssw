#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/Ecal/interface/EcalCondHandler.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondTools/Ecal/interface/EcalPedestalsXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondTools/Ecal/interface/EcalWeightGroupXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondTools/Ecal/interface/EcalDAQTowerStatusXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondTools/Ecal/interface/EcalDCSTowerStatusXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"

#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondTools/Ecal/interface/EcalTimeOffsetXMLTranslator.h"


typedef EcalCondHandler<EcalGainRatios,
			EcalGainRatiosXMLTranslator> EcalGainRatiosHandler;
typedef popcon::PopConAnalyzer<EcalGainRatiosHandler>  
                                         EcalGainRatiosAnalyzer;

typedef EcalCondHandler<EcalPedestals,
			EcalPedestalsXMLTranslator> EcalPedestalsHandler;
typedef popcon::PopConAnalyzer<EcalPedestalsHandler>
                                         EcalPedestalsAnalyzer;

typedef EcalCondHandler<EcalADCToGeVConstant,
			EcalADCToGeVXMLTranslator> EcalADCToGeVConstantHandler;
typedef popcon::PopConAnalyzer<EcalADCToGeVConstantHandler>  
                                         EcalADCToGeVConstantAnalyzer;

typedef EcalCondHandler<EcalWeightXtalGroups,
			EcalWeightGroupXMLTranslator> EcalWeightGroupHandler;
typedef popcon::PopConAnalyzer<EcalWeightGroupHandler>  
                                         EcalWeightGroupAnalyzer;

typedef EcalCondHandler<EcalChannelStatus,
			EcalChannelStatusXMLTranslator> EcalChannelStatusHandler;
typedef popcon::PopConAnalyzer<EcalChannelStatusHandler>  
                                         EcalChannelStatusAnalyzer;

typedef EcalCondHandler<EcalDAQTowerStatus,
			EcalDAQTowerStatusXMLTranslator> EcalDAQTowerStatusHandler;
typedef popcon::PopConAnalyzer<EcalDAQTowerStatusHandler>  
                                         EcalDAQTowerStatusAnalyzer;

typedef EcalCondHandler<EcalDCSTowerStatus,
			EcalDCSTowerStatusXMLTranslator> EcalDCSTowerStatusHandler;
typedef popcon::PopConAnalyzer<EcalDCSTowerStatusHandler>  
                                         EcalDCSTowerStatusAnalyzer;

typedef EcalCondHandler<EcalTBWeights,
			EcalTBWeightsXMLTranslator> EcalTBWeightsHandler;

typedef popcon::PopConAnalyzer<EcalTBWeightsHandler>  
                                         EcalTBWeightsAnalyzer;


typedef EcalCondHandler<EcalLinearCorrections,
			EcalFloatCondObjectContainerXMLTranslator> EcalLinearCorrectionsHandler;

typedef popcon::PopConAnalyzer<EcalLinearCorrectionsHandler>  
                                         EcalLinearCorrectionsAnalyzer;

typedef EcalCondHandler<EcalIntercalibConstants,
			EcalFloatCondObjectContainerXMLTranslator> EcalIntercalibConstantsHandler;

typedef popcon::PopConAnalyzer<EcalIntercalibConstantsHandler>  
                                         EcalIntercalibConstantsAnalyzer;


typedef EcalCondHandler<EcalIntercalibErrors,
			EcalFloatCondObjectContainerXMLTranslator> EcalIntercalibErrorsHandler;

typedef popcon::PopConAnalyzer<EcalIntercalibErrorsHandler>  
                                         EcalIntercalibErrorsAnalyzer;

typedef EcalCondHandler<EcalIntercalibConstantsMC,
			EcalFloatCondObjectContainerXMLTranslator> EcalIntercalibConstantsMCHandler;

typedef popcon::PopConAnalyzer<EcalIntercalibConstantsMCHandler>  
                                         EcalIntercalibConstantsMCAnalyzer;


typedef EcalCondHandler<EcalTimeCalibConstants,
			EcalFloatCondObjectContainerXMLTranslator> EcalTimeCalibConstantsHandler;

typedef popcon::PopConAnalyzer<EcalTimeCalibConstantsHandler>
                                         EcalTimeCalibConstantsAnalyzer;

typedef EcalCondHandler<EcalTimeCalibErrors,
			EcalFloatCondObjectContainerXMLTranslator> EcalTimeCalibErrorsHandler;

typedef popcon::PopConAnalyzer<EcalTimeCalibErrorsHandler>
                                         EcalTimeCalibErrorsAnalyzer;

typedef EcalCondHandler<EcalTimeOffsetConstant,
			EcalTimeOffsetXMLTranslator> EcalTimeOffsetConstantHandler;
typedef popcon::PopConAnalyzer<EcalTimeOffsetConstantHandler>  
                                         EcalTimeOffsetConstantAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalGainRatiosAnalyzer);
DEFINE_FWK_MODULE(EcalADCToGeVConstantAnalyzer);
DEFINE_FWK_MODULE(EcalChannelStatusAnalyzer);
DEFINE_FWK_MODULE(EcalDAQTowerStatusAnalyzer);
DEFINE_FWK_MODULE(EcalDCSTowerStatusAnalyzer);
DEFINE_FWK_MODULE(EcalTBWeightsAnalyzer);
DEFINE_FWK_MODULE(EcalWeightGroupAnalyzer);
DEFINE_FWK_MODULE(EcalLinearCorrectionsAnalyzer);
DEFINE_FWK_MODULE(EcalIntercalibConstantsAnalyzer);
DEFINE_FWK_MODULE(EcalIntercalibErrorsAnalyzer);
DEFINE_FWK_MODULE(EcalIntercalibConstantsMCAnalyzer);
DEFINE_FWK_MODULE(EcalTimeCalibConstantsAnalyzer);
DEFINE_FWK_MODULE(EcalTimeCalibErrorsAnalyzer);
DEFINE_FWK_MODULE(EcalTimeOffsetConstantAnalyzer);

