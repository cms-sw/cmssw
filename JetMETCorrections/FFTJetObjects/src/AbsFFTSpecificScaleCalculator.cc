#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

FFTSpecificScaleCalculatorFactory::FFTSpecificScaleCalculatorFactory()
{
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
