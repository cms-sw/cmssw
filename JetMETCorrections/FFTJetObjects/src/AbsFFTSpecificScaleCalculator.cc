#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTGenericScaleCalculator.h"
#include "JetMETCorrections/FFTJetObjects/interface/L2AbsScaleCalculator.h"
#include "JetMETCorrections/FFTJetObjects/interface/L2ResScaleCalculator.h"

FFTSpecificScaleCalculatorFactory::FFTSpecificScaleCalculatorFactory()
{
    (*this)["L2ResScaleCalculator"] = new ConcreteFFTJetObjectFactory<
        AbsFFTSpecificScaleCalculator,L2ResScaleCalculator>();

    (*this)["L2AbsScaleCalculator"] = new ConcreteFFTJetObjectFactory<
        AbsFFTSpecificScaleCalculator,L2AbsScaleCalculator>();

    (*this)["FFTGenericScaleCalculator"] = new ConcreteFFTJetObjectFactory<
        AbsFFTSpecificScaleCalculator,FFTGenericScaleCalculator>();
}

AbsFFTSpecificScaleCalculator* parseFFTSpecificScaleCalculator(
    const edm::ParameterSet& ps, const std::string& tableDescription)
{
    std::string mapper_type(ps.getParameter<std::string>("Class"));
    if (!mapper_type.compare("auto"))
        mapper_type = tableDescription;

    return StaticFFTSpecificScaleCalculatorFactory::instance().create(
        mapper_type, ps);
}
