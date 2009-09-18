#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/Ecal/interface/EcalCondHandler.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondTools/Ecal/interface/EcalWeightGroupXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"


typedef EcalCondHandler<EcalGainRatios,
			EcalGainRatiosXMLTranslator> EcalGainRatiosHandler;
typedef popcon::PopConAnalyzer<EcalGainRatiosHandler>  
                                         EcalGainRatiosAnalyzer;


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


typedef EcalCondHandler<EcalTBWeights,
			EcalTBWeightsXMLTranslator> EcalTBWeightsHandler;

typedef popcon::PopConAnalyzer<EcalTBWeightsHandler>  
                                         EcalTBWeightsAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalGainRatiosAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalADCToGeVConstantAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalChannelStatusAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalTBWeightsAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalWeightGroupAnalyzer);

