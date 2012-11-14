#ifndef JetMETCorrections_FFTJetModules_FFTJetESParameterParser_h
#define JetMETCorrections_FFTJetModules_FFTJetESParameterParser_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetAdjusters.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetScaleCalculators.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectionsTypemap.h"
#include "JetMETCorrections/InterpolationTables/interface/CoordinateSelector.h"

#include "JetMETCorrections/FFTJetModules/interface/loadFFTJetInterpolationTable.h"


// Parser for the adjuster of the adjustable :-)
template<class Jet, class Adjustable>
boost::shared_ptr<AbsFFTJetAdjuster<Jet, Adjustable> >
parseFFTJetAdjuster(const edm::ParameterSet& ps, const bool /* verbose */)
{
    typedef boost::shared_ptr<AbsFFTJetAdjuster<Jet,Adjustable> > Result;

    const std::string& adjuster_type = ps.getParameter<std::string>("Class");

    if (!adjuster_type.compare("FFTSimpleScalingAdjuster"))
        return Result(new FFTSimpleScalingAdjuster<Jet,Adjustable>());

    else if (!adjuster_type.compare("FFTUncertaintyAdjuster"))
        return Result(new FFTUncertaintyAdjuster<Jet,Adjustable>());

    else if (!adjuster_type.compare("FFTScalingAdjusterWithUncertainty"))
        return Result(new FFTScalingAdjusterWithUncertainty<Jet,Adjustable>());

    else
        throw cms::Exception("FFTJetBadConfig")
            << "In parseFFTJetAdjuster: unknown adjuster type \""
            << adjuster_type << "\"\n";
}


// Parser for the mapper/scaler
template<class Jet, class Adjustable>
boost::shared_ptr<AbsFFTJetScaleCalculator<Jet, Adjustable> >
parseFFTJetScaleCalculator(const edm::ParameterSet& ps,
                           gs::StringArchive& ar,
                           const bool verbose)
{
    typedef boost::shared_ptr<AbsFFTJetScaleCalculator<Jet,Adjustable> > Result;

    std::string mapper_type(ps.getParameter<std::string>("Class"));

    // Initially, check for mappers which do not need to load
    // a data table from the archive
//     if (!mapper_type.compare("SomeClass"))
//     {
//         Do something ....
//         return Result(...);
//     }

    // Load the table from the archive
    CPP11_auto_ptr<npstat::StorableMultivariateFunctor> autof = 
        loadFFTJetInterpolationTable(ps, ar, verbose);
    CPP11_shared_ptr<npstat::StorableMultivariateFunctor> f(autof.release());

    // Swap the class name if it is supposed to be determined
    // from the table description
    if (!mapper_type.compare("auto"))
        mapper_type = f->description();

    if (!mapper_type.compare("FFTEtaLogPtConeRadiusMapper"))
        return Result(new FFTEtaLogPtConeRadiusMapper<Jet,Adjustable>(f));

    else if (!mapper_type.compare("FFTSpecificScaleCalculator"))
    {
        const edm::ParameterSet& subclass = 
            ps.getParameter<edm::ParameterSet>("Subclass");
        AbsFFTSpecificScaleCalculator* p = parseFFTSpecificScaleCalculator(
            subclass, f->description());
        return Result(new FFTSpecificScaleCalculator<Jet,Adjustable>(f, p));
    }

    else
        throw cms::Exception("FFTJetBadConfig")
            << "In parseFFTJetScaleCalculator: unknown mapper type \""
            << mapper_type << '"' << std::endl;
}


// Parser for a single corrector of FFTJets
template<class Corrector>
Corrector parseFFTJetCorrector(const edm::ParameterSet& ps,
                               gs::StringArchive& ar,
                               const bool verbose)
{
    typedef typename Corrector::jet_type MyJet;
    typedef typename Corrector::adjustable_type Adjustable;

    // "level" is an unsigned
    const unsigned level(ps.getParameter<unsigned>("level"));

    // "applyTo" is a string
    const std::string& applyTo(ps.getParameter<std::string>("applyTo"));

    // "adjuster" is a PSet
    const edm::ParameterSet& adjuster = ps.getParameter<edm::ParameterSet>("adjuster");
    boost::shared_ptr<AbsFFTJetAdjuster<MyJet,Adjustable> > adj = 
        parseFFTJetAdjuster<MyJet,Adjustable>(adjuster, verbose);

    // "scalers" is a VPSet
    const std::vector<edm::ParameterSet>& scalers = 
        ps.getParameter<std::vector<edm::ParameterSet> >("scalers");
    const unsigned nScalers = scalers.size();
    std::vector<boost::shared_ptr<AbsFFTJetScaleCalculator<MyJet,Adjustable> > > sVec;
    sVec.reserve(nScalers);
    for (unsigned i=0; i<nScalers; ++i)
    {
        boost::shared_ptr<AbsFFTJetScaleCalculator<MyJet,Adjustable> > s = 
            parseFFTJetScaleCalculator<MyJet,Adjustable>(scalers[i], ar, verbose);
        sVec.push_back(s);
    }
    return Corrector(adj, sVec, level, parseFFTJetCorrectorApp(applyTo));
}

#endif // JetMETCorrections_FFTJetModules_FFTJetESParameterParser_h
