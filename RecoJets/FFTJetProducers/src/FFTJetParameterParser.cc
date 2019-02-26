#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

#include "RecoJets/FFTJetAlgorithms/interface/JetConvergenceDistance.h"
#include "RecoJets/FFTJetAlgorithms/interface/ScaleCalculators.h"
#include "RecoJets/FFTJetAlgorithms/interface/EtaAndPtDependentPeakSelector.h"
#include "RecoJets/FFTJetAlgorithms/interface/EtaDependentPileup.h"
#include "RecoJets/FFTJetAlgorithms/interface/PileupGrid2d.h"
#include "RecoJets/FFTJetAlgorithms/interface/JetAbsEta.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "fftjet/PeakSelectors.hh"
#include "fftjet/Kernel2dFactory.hh"
#include "fftjet/GaussianNoiseMembershipFcn.hh"
#include "fftjet/EquidistantSequence.hh"
#include "fftjet/PeakEtaPhiDistance.hh"
#include "fftjet/PeakEtaDependentDistance.hh"
#include "fftjet/JetProperty.hh"
#include "fftjet/InterpolatedMembershipFcn.hh"
#include "fftjet/CompositeKernel.hh"
#include "fftjet/InterpolatedKernel.hh"
#include "fftjet/InterpolatedKernel3d.hh"
#include "fftjet/MagneticSmearingKernel.hh"

#define make_param(type, varname) const \
    type & varname (ps.getParameter< type >( #varname ))

using namespace fftjetcms;

typedef fftjet::RecombinedJet<VectorLike> RecoFFTJet;

static bool parse_peak_member_function(const char* fname,
    fftjet::JetProperty<fftjet::Peak>::JetMemberFunction *f)
{
    if (strcmp(fname, "eta") == 0)
        *f = &fftjet::Peak::eta;
    else if (strcmp(fname, "phi") == 0)
        *f = &fftjet::Peak::phi;
    else if (strcmp(fname, "magnitude") == 0)
        *f = &fftjet::Peak::magnitude;
    else if (strcmp(fname, "driftSpeed") == 0)
        *f = &fftjet::Peak::driftSpeed;
    else if (strcmp(fname, "magSpeed") == 0)
        *f = &fftjet::Peak::magSpeed;
    else if (strcmp(fname, "lifetime") == 0)
        *f = &fftjet::Peak::lifetime;
    else if (strcmp(fname, "mergeTime") == 0)
        *f = &fftjet::Peak::mergeTime;
    else if (strcmp(fname, "splitTime") == 0)
        *f = &fftjet::Peak::splitTime;
    else if (strcmp(fname, "scale") == 0)
        *f = &fftjet::Peak::scale;
    else if (strcmp(fname, "nearestNeighborDistance") == 0)
        *f = &fftjet::Peak::nearestNeighborDistance;
    else if (strcmp(fname, "membershipFactor") == 0)
        *f = &fftjet::Peak::membershipFactor;
    else if (strcmp(fname, "recoScale") == 0)
        *f = &fftjet::Peak::recoScale;
    else if (strcmp(fname, "recoScaleRatio") == 0)
        *f = &fftjet::Peak::recoScaleRatio;
    else if (strcmp(fname, "laplacian") == 0)
        *f = &fftjet::Peak::laplacian;
    else if (strcmp(fname, "hessianDeterminant") == 0)
        *f = &fftjet::Peak::hessianDeterminant;
    else if (strcmp(fname, "clusterRadius") == 0)
        *f = &fftjet::Peak::clusterRadius;
    else if (strcmp(fname, "clusterSeparation") == 0)
        *f = &fftjet::Peak::clusterSeparation;
    else
    {
        return false;
    }
    return true;
}

static bool parse_jet_member_function(const char* fname,
    fftjet::JetProperty<RecoFFTJet>::JetMemberFunction *f)
{
    if (strcmp(fname, "peakEta") == 0)
        *f = &RecoFFTJet::peakEta;
    else if (strcmp(fname, "peakPhi") == 0)
        *f = &RecoFFTJet::peakPhi;
    else if (strcmp(fname, "peakMagnitude") == 0)
        *f = &RecoFFTJet::peakMagnitude;
    else if (strcmp(fname, "magnitude") == 0)
        *f = &RecoFFTJet::magnitude;
    else if (strcmp(fname, "driftSpeed") == 0)
        *f = &RecoFFTJet::driftSpeed;
    else if (strcmp(fname, "magSpeed") == 0)
        *f = &RecoFFTJet::magSpeed;
    else if (strcmp(fname, "lifetime") == 0)
        *f = &RecoFFTJet::lifetime;
    else if (strcmp(fname, "mergeTime") == 0)
        *f = &RecoFFTJet::mergeTime;
    else if (strcmp(fname, "splitTime") == 0)
        *f = &RecoFFTJet::splitTime;
    else if (strcmp(fname, "scale") == 0)
        *f = &RecoFFTJet::scale;
    else if (strcmp(fname, "nearestNeighborDistance") == 0)
        *f = &RecoFFTJet::nearestNeighborDistance;
    else if (strcmp(fname, "membershipFactor") == 0)
        *f = &RecoFFTJet::membershipFactor;
    else if (strcmp(fname, "recoScale") == 0)
        *f = &RecoFFTJet::recoScale;
    else if (strcmp(fname, "recoScaleRatio") == 0)
        *f = &RecoFFTJet::recoScaleRatio;
    else if (strcmp(fname, "laplacian") == 0)
        *f = &RecoFFTJet::laplacian;
    else if (strcmp(fname, "hessianDeterminant") == 0)
        *f = &RecoFFTJet::hessianDeterminant;
    else if (strcmp(fname, "clusterRadius") == 0)
        *f = &RecoFFTJet::clusterRadius;
    else if (strcmp(fname, "clusterSeparation") == 0)
        *f = &RecoFFTJet::clusterSeparation;
    else
    {
        return false;
    }
    return true;
}


namespace fftjetcms {

std::unique_ptr<fftjet::Grid2d<Real> >
fftjet_Grid2d_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::Grid2d<Real> > return_type;
    fftjet::Grid2d<Real> *g = nullptr;

    // Check if the grid should be read from file
    if (ps.exists("file"))
    {
        const std::string file = ps.getParameter<std::string>("file");
        std::ifstream in(file.c_str(),
	                 std::ios_base::in | std::ios_base::binary);
        if (!in.is_open())
	    throw cms::Exception("FFTJetBadConfig")
	        << "Failed to open file " << file << std::endl;
        g = fftjet::Grid2d<Real>::read(in);
	if (g == nullptr)
	    throw cms::Exception("FFTJetBadConfig")
	        << "Failed to read file " << file << std::endl;
    }
    else
    {
        const unsigned nEtaBins = ps.getParameter<unsigned>("nEtaBins");
        const Real etaMin = ps.getParameter<Real>("etaMin");
        const Real etaMax = ps.getParameter<Real>("etaMax");
        const unsigned nPhiBins = ps.getParameter<unsigned>("nPhiBins");
        const Real phiBin0Edge = ps.getParameter<Real>("phiBin0Edge");
        const std::string& title = ps.getUntrackedParameter<std::string>(
            "title", "");

        if (nEtaBins == 0 || nPhiBins == 0 || etaMin >= etaMax)
            return return_type(nullptr);
        
        g = new fftjet::Grid2d<Real>(
            nEtaBins,
            etaMin,
            etaMax,
            nPhiBins,
            phiBin0Edge,
            title.c_str()
        );

        // Check if the grid data is provided
        if (ps.exists("data"))
	{
	    const std::vector<Real>& data = 
	        ps.getParameter<std::vector<Real> >("data");
            if (data.size() == nEtaBins*nPhiBins)
	        g->blockSet(&data[0], nEtaBins, nPhiBins);
            else
	    {
	        delete g;
                g = nullptr;
            }
        }
    }

    return return_type(g);
}


std::unique_ptr<fftjet::Functor1<bool,fftjet::Peak> >
fftjet_PeakSelector_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::Functor1<bool,fftjet::Peak> > return_type;

    const std::string peakselector_type = ps.getParameter<std::string>(
        "Class");

    if (!peakselector_type.compare("AllPeaksPass"))
    {
        return return_type(new fftjet::AllPeaksPass());
    }

    if (!peakselector_type.compare("EtaAndPtDependentPeakSelector"))
    {
        const std::string file = ps.getParameter<std::string>("file");
        std::ifstream in(file.c_str(),
                         std::ios_base::in | std::ios_base::binary);
        if (!in.is_open())
            throw cms::Exception("FFTJetBadConfig")
                << "Failed to open file " << file << std::endl;
        EtaAndPtDependentPeakSelector* ptr =
            new EtaAndPtDependentPeakSelector(in);
        if (!ptr->isValid())
            throw cms::Exception("FFTJetBadConfig")
                << "Failed to read file " << file << std::endl;
        return return_type(ptr);
    }

    if (!peakselector_type.compare("EtaAndPtLookupPeakSelector"))
    {
        make_param(unsigned, nx);
        make_param(unsigned, ny);
        make_param(double, xmin);
        make_param(double, xmax);
        make_param(double, ymin);
        make_param(double, ymax);
        make_param(std::vector<double>, data);

        if (xmin >= xmax || ymin >= ymax || !nx || !ny || data.size() != nx*ny)
            throw cms::Exception("FFTJetBadConfig")
                << "Failed to configure EtaAndPtLookupPeakSelector" << std::endl;

        return return_type(new EtaAndPtLookupPeakSelector(
                               nx, xmin, xmax, ny, ymin, ymax, data));
    }

    if (!peakselector_type.compare("SimplePeakSelector"))
    {
        const double magCut = ps.getParameter<double>("magCut");
        const double driftSpeedCut = ps.getParameter<double>("driftSpeedCut");
        const double magSpeedCut = ps.getParameter<double>("magSpeedCut");
        const double lifeTimeCut = ps.getParameter<double>("lifeTimeCut");
        const double NNDCut = ps.getParameter<double>("NNDCut");
        const double etaCut = ps.getParameter<double>("etaCut");
        const double splitTimeCut = ps.getParameter<double>("splitTimeCut");
        const double mergeTimeCut = ps.getParameter<double>("mergeTimeCut");

        return return_type(new fftjet::SimplePeakSelector(
            magCut, driftSpeedCut, magSpeedCut, lifeTimeCut, NNDCut,
            etaCut, splitTimeCut, mergeTimeCut));
    }

    if (!peakselector_type.compare("ScalePowerPeakSelector"))
    {
        const double a = ps.getParameter<double>("a");
        const double p = ps.getParameter<double>("p");
        const double b = ps.getParameter<double>("b");
        const double etaCut = ps.getParameter<double>("etaCut");

        return return_type(new fftjet::ScalePowerPeakSelector(
                               a, p, b, etaCut));
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::ScaleSpaceKernel>
fftjet_MembershipFunction_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::ScaleSpaceKernel> return_type;

    const std::string MembershipFunction_type = ps.getParameter<std::string>(
        "Class");

    // Parse special cases first
    if (!MembershipFunction_type.compare("InterpolatedMembershipFcn"))
    {
        // This is a kernel defined by a 4d (sparsified) lookup table.
        // Here, it is simply loaded from a file using a built-in
        // method from fftjet. Note that the table representation
        // must be native binary (this will not work on platforms with
        // different endianity of floating point standard).
        const std::string file = ps.getParameter<std::string>("file");
        std::ifstream in(file.c_str(),
                         std::ios_base::in | std::ios_base::binary);
        if (!in.is_open())
            throw cms::Exception("FFTJetBadConfig")
                << "Failed to open file " << file << std::endl;
        return return_type(fftjet::InterpolatedMembershipFcn<float>::read(in));
    }

    if (!MembershipFunction_type.compare("Composite"))
    {
        throw cms::Exception("FFTJetBadConfig")
            << "Parsing of CompositeKernel objects is not implemented yet"
            << std::endl;
    }

    if (!MembershipFunction_type.compare("MagneticSmearing"))
    {
        // This kernel represents smearing of a jet in phi
        // in a magnetic field. The meaning of the parameters
        // is explained in the comments in the MagneticSmearingKernel.hh
        // header file of the fftjet package.
        make_param(std::vector<double>, fragmentationData);
        make_param(double, numeratorConst);
        make_param(double, charge1Fraction);
        make_param(double, charge2Fraction);
        make_param(unsigned, samplesPerBin);

        if (fragmentationData.empty())
            throw cms::Exception("FFTJetBadConfig")
                << "Fragmentation function data not defined for "
                "MagneticSmearingKernel" << std::endl;
        if (samplesPerBin < 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "Bad number of samples per bin in "
                "MagneticSmearingKernel" << std::endl;

        fftjet::LinearInterpolator1d* fragmentationFunction = 
            new fftjet::LinearInterpolator1d(
                &fragmentationData[0], fragmentationData.size(), 0.0, 1.0);

        return return_type(
            new fftjet::MagneticSmearingKernel<fftjet::LinearInterpolator1d>(
                fragmentationFunction, numeratorConst,
                charge1Fraction, charge2Fraction,
                samplesPerBin, true));
    }

    if (!MembershipFunction_type.compare("Interpolated"))
    {
        // This is a kernel defined by a histogram-like 2d lookup table
        make_param(double, sx);
        make_param(double, sy);
        make_param(int, scalePower);
        make_param(unsigned, nxbins);
        make_param(double, xmin);
        make_param(double, xmax);
        make_param(unsigned, nybins);
        make_param(double, ymin);
        make_param(double, ymax);
        make_param(std::vector<double>, data);

        if (data.size() != nxbins*nybins)
            throw cms::Exception("FFTJetBadConfig")
                << "Bad number of data points for Interpolated kernel"
                << std::endl;

        return return_type(new fftjet::InterpolatedKernel(
                               sx, sy, scalePower,
                               &data[0], nxbins, xmin, xmax,
                               nybins, ymin, ymax));
    }

    if (!MembershipFunction_type.compare("Interpolated3d"))
    {
        // This is a kernel defined by a histogram-like 3d lookup table
        make_param(std::vector<double>, data);
        make_param(std::vector<double>, scales);
        make_param(bool, useLogSpaceForScale);
        make_param(unsigned, nxbins);
        make_param(double, xmin);
        make_param(double, xmax);
        make_param(unsigned, nybins);
        make_param(double, ymin);
        make_param(double, ymax);

        if (data.size() != nxbins*nybins*scales.size())
            throw cms::Exception("FFTJetBadConfig")
                << "Bad number of data points for Interpolated3d kernel"
                << std::endl;

        return return_type(new fftjet::InterpolatedKernel3d(
                               &data[0], scales, useLogSpaceForScale,
                               nxbins, xmin, xmax, nybins, ymin, ymax));
    }

    // This is not a special kernel. Try one of the classes
    // in the kernel factory provided by FFTJet.
    fftjet::DefaultKernel2dFactory factory;
    if (factory[MembershipFunction_type] == nullptr) {
        return return_type(nullptr);
    }

    make_param(double, sx);
    make_param(double, sy);
    make_param(int, scalePower);
    make_param(std::vector<double>, kernelParameters);

    const int n_expected = factory[MembershipFunction_type]->nParameters();
    if (n_expected >= 0)
        if (static_cast<unsigned>(n_expected) != kernelParameters.size())
            throw cms::Exception("FFTJetBadConfig")
                << "Bad number of kernel parameters" << std::endl;

    return std::unique_ptr<fftjet::ScaleSpaceKernel>(
        factory[MembershipFunction_type]->create(
            sx, sy, scalePower, kernelParameters));
}


std::unique_ptr<AbsBgFunctor>
fftjet_BgFunctor_parser(const edm::ParameterSet& ps)
{
    const std::string bg_Membership_type = ps.getParameter<std::string>(
        "Class");

    if (!bg_Membership_type.compare("GaussianNoiseMembershipFcn"))
    {
        const double minWeight = ps.getParameter<double>("minWeight");
        const double prior = ps.getParameter<double>("prior");
        return std::unique_ptr<AbsBgFunctor>(
            new fftjet::GaussianNoiseMembershipFcn(minWeight,prior));
    }

    return std::unique_ptr<AbsBgFunctor>(nullptr);
}


std::unique_ptr<std::vector<double> >
fftjet_ScaleSet_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<std::vector<double> > return_type;

    const std::string className = ps.getParameter<std::string>("Class");

    if (!className.compare("EquidistantInLinearSpace") ||
        !className.compare("EquidistantInLogSpace"))
    {
        const double minScale = ps.getParameter<double>("minScale");
        const double maxScale = ps.getParameter<double>("maxScale");
        const unsigned nScales = ps.getParameter<unsigned>("nScales");

        if (minScale <= 0.0 || maxScale <= 0.0 ||
            nScales == 0 || minScale == maxScale)
            return return_type(nullptr);

        // Can't return pointers to EquidistantInLinearSpace
        // or EquidistantInLogSpace directly because std::vector
        // destructor is not virtual.
        if (!className.compare("EquidistantInLinearSpace"))
            return return_type(new std::vector<double>(
                                   fftjet::EquidistantInLinearSpace(
                                       minScale, maxScale, nScales)));
        else
            return return_type(new std::vector<double>(
                                   fftjet::EquidistantInLogSpace(
                                       minScale, maxScale, nScales)));
    }

    if (!className.compare("UserSet"))
    {
        return_type scales(new std::vector<double>(
            ps.getParameter<std::vector<double> >("scales")));

        // Verify that all scales are positive and unique
        const unsigned nscales = scales->size();
        for (unsigned i=0; i<nscales; ++i)
            if ((*scales)[i] <= 0.0)
                return return_type(nullptr);

        for (unsigned i=1; i<nscales; ++i)
            for (unsigned j=0; j<i; ++j)
                if ((*scales)[i] == (*scales)[j])
                    return return_type(nullptr);

        return scales;
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::ClusteringTreeSparsifier<fftjet::Peak,long> >
fftjet_ClusteringTreeSparsifier_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::ClusteringTreeSparsifier<fftjet::Peak,long> > return_type;

    const int maxLevelNumber = ps.getParameter<int>("maxLevelNumber");
    const unsigned filterMask = ps.getParameter<unsigned>("filterMask");
    const std::vector<double> userScalesV = 
        ps.getParameter<std::vector<double> >("userScales");
    const unsigned nUserScales = userScalesV.size();
    const double* userScales = nUserScales ? &userScalesV[0] : nullptr;

    return return_type(
        new fftjet::ClusteringTreeSparsifier<fftjet::Peak,long>(
            maxLevelNumber,
            filterMask,
            userScales,
            nUserScales
            )
        );
}


std::unique_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> >
fftjet_DistanceCalculator_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> > return_type;

    const std::string calc_type = ps.getParameter<std::string>("Class");

    if (!calc_type.compare("PeakEtaPhiDistance"))
    {
        const double etaToPhiBandwidthRatio = ps.getParameter<double>(
            "etaToPhiBandwidthRatio");
        return return_type(new fftjet::PeakEtaPhiDistance(etaToPhiBandwidthRatio));
    }

    if (!calc_type.compare("PeakEtaDependentDistance"))
    {
        std::unique_ptr<fftjet::LinearInterpolator1d> interp = 
            fftjet_LinearInterpolator1d_parser(
                ps.getParameter<edm::ParameterSet>("Interpolator"));
        const fftjet::LinearInterpolator1d* ip = interp.get();
        if (ip == nullptr)
            return return_type(nullptr);

        // Check that the interpolator is always positive
        const unsigned n = ip->nx();
        const double* data = ip->getData();
        for (unsigned i=0; i<n; ++i)
            if (data[i] <= 0.0)
                return return_type(nullptr);
        if (ip->fLow() <= 0.0 || ip->fHigh() <= 0.0)
            return return_type(nullptr);

        return return_type(new fftjet::PeakEtaDependentDistance(*ip));
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjetcms::LinInterpolatedTable1D>
fftjet_LinInterpolatedTable1D_parser(const edm::ParameterSet& ps)
{
    const double xmin = ps.getParameter<double>("xmin");
    const double xmax = ps.getParameter<double>("xmax");
    const bool leftExtrapolationLinear = 
        ps.getParameter<bool>("leftExtrapolationLinear");
    const bool rightExtrapolationLinear = 
        ps.getParameter<bool>("rightExtrapolationLinear");
    const std::vector<double> data(
        ps.getParameter<std::vector<double> >("data"));
    if (data.empty())
        return std::unique_ptr<fftjetcms::LinInterpolatedTable1D>(nullptr);
    else
        return std::unique_ptr<fftjetcms::LinInterpolatedTable1D>(
            new fftjetcms::LinInterpolatedTable1D(
                &data[0], data.size(), xmin, xmax,
                leftExtrapolationLinear, rightExtrapolationLinear));
}


std::unique_ptr<fftjet::LinearInterpolator1d>
fftjet_LinearInterpolator1d_parser(const edm::ParameterSet& ps)
{
    const double xmin = ps.getParameter<double>("xmin");
    const double xmax = ps.getParameter<double>("xmax");
    const double flow = ps.getParameter<double>("flow");
    const double fhigh = ps.getParameter<double>("fhigh");
    const std::vector<double> data(
        ps.getParameter<std::vector<double> >("data"));
    if (data.empty())
        return std::unique_ptr<fftjet::LinearInterpolator1d>(nullptr);
    else
        return std::unique_ptr<fftjet::LinearInterpolator1d>(
            new fftjet::LinearInterpolator1d(
                &data[0], data.size(), xmin, xmax, flow, fhigh));
}


std::unique_ptr<fftjet::LinearInterpolator2d>
fftjet_LinearInterpolator2d_parser(const edm::ParameterSet& ps)
{
    const std::string file = ps.getParameter<std::string>("file");
    std::ifstream in(file.c_str(),
                     std::ios_base::in | std::ios_base::binary);
    if (!in.is_open())
        throw cms::Exception("FFTJetBadConfig")
            << "Failed to open file " << file << std::endl;
    fftjet::LinearInterpolator2d* ip = fftjet::LinearInterpolator2d::read(in);
    if (!ip)
        throw cms::Exception("FFTJetBadConfig")
            << "Failed to read file " << file << std::endl;
    return std::unique_ptr<fftjet::LinearInterpolator2d>(ip);
}


std::unique_ptr<fftjet::Functor1<double,fftjet::Peak> >
fftjet_PeakFunctor_parser(const edm::ParameterSet& ps)
{
    typedef fftjet::Functor1<double,fftjet::Peak> ptr_type;
    typedef std::unique_ptr<ptr_type> return_type;

    const std::string property_type = ps.getParameter<std::string>("Class");

    if (!property_type.compare("Log"))
    {
        return_type wrapped = fftjet_PeakFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function"));
        ptr_type* wr = wrapped.get();
        if (wr)
        {
            return_type result = return_type(
                new fftjet::LogProperty<fftjet::Peak>(wr, true));
            wrapped.release();
            return result;
        }
    }

    if (!property_type.compare("PeakProperty"))
    {
        const std::string member = ps.getParameter<std::string>("member");
        fftjet::JetProperty<fftjet::Peak>::JetMemberFunction fcn;
        if (parse_peak_member_function(member.c_str(), &fcn))
            return return_type(
                new fftjet::JetProperty<fftjet::Peak>(fcn));
        else
            return return_type(nullptr);
    }

    if (!property_type.compare("MinusScaledLaplacian"))
    {
        const double sx = ps.getParameter<double>("sx");
        const double sy = ps.getParameter<double>("sx");
        return return_type(
            new fftjet::MinusScaledLaplacian<fftjet::Peak>(sx, sy));
    }

    if (!property_type.compare("ScaledHessianDet"))
    {
        return return_type(
            new fftjet::ScaledHessianDet<fftjet::Peak>());
    }

    if (!property_type.compare("ScaledMagnitude"))
    {
        return return_type(
            new fftjet::ScaledMagnitude<fftjet::Peak>());
    }

    if (!property_type.compare("ScaledMagnitude2"))
    {
        return return_type(
            new fftjet::ScaledMagnitude2<fftjet::Peak>());
    }

    if (!property_type.compare("ConstDouble"))
    {
        const double value = ps.getParameter<double>("value");
        return return_type(new ConstDouble<fftjet::Peak>(value));
    }

    if (!property_type.compare("ProportionalToScale"))
    {
        const double value = ps.getParameter<double>("value");
        return return_type(
            new ProportionalToScale<fftjet::Peak>(value));
    }

    if (!property_type.compare("MultiplyByConst"))
    {
        const double factor = ps.getParameter<double>("factor");
        return_type function = fftjet_PeakFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function"));
        ptr_type* ptr = function.get();
        if (ptr)
        {
            return_type result = return_type(
                new MultiplyByConst<fftjet::Peak>(factor, ptr, true));
            function.release();
            return result;
        }
    }

    if (!property_type.compare("CompositeFunctor"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function1"));
        return_type fcn2 = fftjet_PeakFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function2"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        ptr_type* f2 = fcn2.get();
        if (f1 && f2)
        {
            return_type result = return_type(
                new CompositeFunctor<fftjet::Peak>(f1, f2, true));
            fcn1.release();
            fcn2.release();
            return result;
        }
    }

    if (!property_type.compare("ProductFunctor"))
    {
        return_type fcn1 = fftjet_PeakFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function1"));
        return_type fcn2 = fftjet_PeakFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function2"));
        ptr_type* f1 = fcn1.get();
        ptr_type* f2 = fcn2.get();
        if (f1 && f2)
        {
            return_type result = return_type(
                new ProductFunctor<fftjet::Peak>(f1, f2, true));
            fcn1.release();
            fcn2.release();
            return result;
        }
    }

    if (!property_type.compare("MagnitudeDependent"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        if (f1)
        {
            return_type result = return_type(
                new MagnitudeDependent<fftjet::Peak>(f1, true));
            fcn1.release();
            return result;
        }
    }

    if (!property_type.compare("PeakEtaDependent"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        if (f1)
        {
            return_type result = return_type(new PeakEtaDependent(f1, true));
            fcn1.release();
            return result;
        }
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::Functor1<double,fftjet::RecombinedJet<VectorLike> > >
fftjet_JetFunctor_parser(const edm::ParameterSet& ps)
{
    typedef fftjet::Functor1<double,RecoFFTJet> ptr_type;
    typedef std::unique_ptr<ptr_type> return_type;

    const std::string property_type = ps.getParameter<std::string>("Class");

    if (!property_type.compare("Log"))
    {
        return_type wrapped = fftjet_JetFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function"));
        fftjet::Functor1<double,RecoFFTJet>* wr = wrapped.get();
        if (wr)
        {
            return_type result = return_type(
                new fftjet::LogProperty<RecoFFTJet>(wr, true));
            wrapped.release();
            return result;
        }
    }

    if (!property_type.compare("JetEtaDependent"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        if (f1)
        {
            return_type result = return_type(new JetEtaDependent(f1, true));
            fcn1.release();
            return result;
        }
    }

    if (!property_type.compare("JetProperty"))
    {
        const std::string member = ps.getParameter<std::string>("member");
        fftjet::JetProperty<RecoFFTJet>::JetMemberFunction fcn;
        if (parse_jet_member_function(member.c_str(), &fcn))
            return return_type(
                new fftjet::JetProperty<RecoFFTJet>(fcn));
        else
            return return_type(nullptr);
    }

    if (!property_type.compare("ConstDouble"))
    {
        const double value = ps.getParameter<double>("value");
        return return_type(new ConstDouble<RecoFFTJet>(value));
    }

    if (!property_type.compare("ProportionalToScale"))
    {
        const double value = ps.getParameter<double>("value");
        return return_type(
            new ProportionalToScale<RecoFFTJet>(value));
    }

    if (!property_type.compare("MultiplyByConst"))
    {
        const double factor = ps.getParameter<double>("factor");
        return_type function = fftjet_JetFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function"));
        ptr_type* ptr = function.get();
        if (ptr)
        {
            return_type result = return_type(
                new MultiplyByConst<RecoFFTJet>(factor, ptr, true));
            function.release();
            return result;
        }
    }

    if (!property_type.compare("CompositeFunctor"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function1"));
        return_type fcn2 = fftjet_JetFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function2"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        ptr_type* f2 = fcn2.get();
        if (f1 && f2)
        {
            return_type result = return_type(
                new CompositeFunctor<RecoFFTJet>(f1, f2, true));
            fcn1.release();
            fcn2.release();
            return result;
        }
    }

    if (!property_type.compare("ProductFunctor"))
    {
        return_type fcn1 = fftjet_JetFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function1"));
        return_type fcn2 = fftjet_JetFunctor_parser(
            ps.getParameter<edm::ParameterSet>("function2"));
        ptr_type* f1 = fcn1.get();
        ptr_type* f2 = fcn2.get();
        if (f1 && f2)
        {
            return_type result = return_type(
                new ProductFunctor<RecoFFTJet>(f1, f2, true));
            fcn1.release();
            fcn2.release();
            return result;
        }
    }

    if (!property_type.compare("MagnitudeDependent"))
    {
        std::unique_ptr<fftjet::Functor1<double,double> > fcn1 = 
            fftjet_Function_parser(
                ps.getParameter<edm::ParameterSet>("function"));
        fftjet::Functor1<double,double>* f1 = fcn1.get();
        if (f1)
        {
            return_type result = return_type(
                new MagnitudeDependent<RecoFFTJet>(f1, true));
            fcn1.release();
            return result;
        }
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::Functor2<double,
                               fftjet::RecombinedJet<VectorLike>,
                               fftjet::RecombinedJet<VectorLike> > >
fftjet_JetDistance_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::Functor2<
        double,
        fftjet::RecombinedJet<VectorLike>,
        fftjet::RecombinedJet<VectorLike> > > return_type;

    const std::string distance_type = ps.getParameter<std::string>(
        "Class");

    if (!distance_type.compare("JetConvergenceDistance"))
    {
        make_param(double, etaToPhiBandwidthRatio);
        make_param(double, relativePtBandwidth);

        if (etaToPhiBandwidthRatio > 0.0 && relativePtBandwidth > 0.0)
            return return_type(new JetConvergenceDistance(
                etaToPhiBandwidthRatio, relativePtBandwidth));
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::Functor1<double,double> >
fftjet_Function_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<fftjet::Functor1<double,double> > return_type;

    const std::string fcn_type = ps.getParameter<std::string>("Class");

    if (!fcn_type.compare("LinearInterpolator1d"))
    {
        std::unique_ptr<fftjet::LinearInterpolator1d> p = 
            fftjet_LinearInterpolator1d_parser(ps);
        fftjet::LinearInterpolator1d* ptr = p.get();
        if (ptr)
        {
            p.release();
            return return_type(ptr);
        }
    }

    if (!fcn_type.compare("LinInterpolatedTable1D"))
    {
        std::unique_ptr<fftjetcms::LinInterpolatedTable1D> p = 
            fftjet_LinInterpolatedTable1D_parser(ps);
        fftjetcms::LinInterpolatedTable1D* ptr = p.get();
        if (ptr)
        {
            p.release();
            return return_type(ptr);
        }
    }

    if (!fcn_type.compare("Polynomial"))
    {
        std::vector<double> coeffs;
        for (unsigned i=0; ; ++i)
        {
            std::ostringstream s;
            s << 'c' << i;
            if (ps.exists(s.str()))
                coeffs.push_back(ps.getParameter<double>(s.str()));
            else
                break;
        }
        return return_type(new Polynomial(coeffs));
    }

    return return_type(nullptr);
}


std::unique_ptr<AbsPileupCalculator>
fftjet_PileupCalculator_parser(const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<AbsPileupCalculator> return_type;

    const std::string fcn_type = ps.getParameter<std::string>("Class");

    if (!fcn_type.compare("EtaDependentPileup"))
    {
        std::unique_ptr<fftjet::LinearInterpolator2d> interp = 
            fftjet_LinearInterpolator2d_parser(
                ps.getParameter<edm::ParameterSet>("Interpolator2d"));
        const double inputRhoFactor = ps.getParameter<double>("inputRhoFactor");
        const double outputRhoFactor = ps.getParameter<double>("outputRhoFactor");

        const fftjet::LinearInterpolator2d* ip = interp.get();
        if (ip)
            return return_type(new EtaDependentPileup(
                                   *ip, inputRhoFactor, outputRhoFactor));
        else
            return return_type(nullptr);
    }

    if (!fcn_type.compare("PileupGrid2d"))
    {
        std::unique_ptr<fftjet::Grid2d<Real> > grid = 
	    fftjet_Grid2d_parser(
		ps.getParameter<edm::ParameterSet>("Grid2d"));
        const double rhoFactor = ps.getParameter<double>("rhoFactor");

        const fftjet::Grid2d<Real>* g = grid.get();
        if (g)
	    return return_type(new PileupGrid2d(*g, rhoFactor));
        else
	    return return_type(nullptr);
    }

    return return_type(nullptr);
}


std::unique_ptr<fftjet::JetMagnitudeMapper2d <fftjet::Peak> >
fftjet_PeakMagnitudeMapper2d_parser (const edm::ParameterSet& ps)
{
    std::unique_ptr<fftjet::LinearInterpolator2d> responseCurve =
        fftjet_LinearInterpolator2d_parser(
            ps.getParameter<edm::ParameterSet>("responseCurve"));

    const double minPredictor = ps.getParameter<double>("minPredictor");
    const double maxPredictor = ps.getParameter<double>("maxPredictor");
    const unsigned nPredPoints = ps.getParameter<unsigned>("nPredPoints");
    const double maxMagnitude = ps.getParameter<double>("maxMagnitude");
    const unsigned nMagPoints = ps.getParameter<unsigned>("nMagPoints");

    return (std::unique_ptr<fftjet::JetMagnitudeMapper2d<fftjet::Peak> >
             (new fftjet::JetMagnitudeMapper2d<fftjet::Peak>(
                  *responseCurve,
                  new fftjetcms::PeakAbsEta<fftjet::Peak>(),
                  true,minPredictor,maxPredictor,nPredPoints,
                  maxMagnitude,nMagPoints)));
}


std::unique_ptr<fftjet::JetMagnitudeMapper2d <fftjet::RecombinedJet<VectorLike> > >
fftjet_JetMagnitudeMapper2d_parser (const edm::ParameterSet& ps)
{
    std::unique_ptr<fftjet::LinearInterpolator2d> responseCurve =
        fftjet_LinearInterpolator2d_parser(
            ps.getParameter<edm::ParameterSet>("responseCurve"));

    const double minPredictor = ps.getParameter<double>("minPredictor");
    const double maxPredictor = ps.getParameter<double>("maxPredictor");
    const unsigned nPredPoints = ps.getParameter<unsigned>("nPredPoints");
    const double maxMagnitude = ps.getParameter<double>("maxMagnitude");
    const unsigned nMagPoints = ps.getParameter<unsigned>("nMagPoints");

    return (std::unique_ptr<fftjet::JetMagnitudeMapper2d<RecoFFTJet> >
            (new fftjet::JetMagnitudeMapper2d<RecoFFTJet>(
                 *responseCurve,
                 new fftjetcms::JetAbsEta<RecoFFTJet>(),
                 true,minPredictor,maxPredictor,nPredPoints,
                 maxMagnitude,nMagPoints)));
}

}
