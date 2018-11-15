// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetEFlowSmoother
// 
/**\class FFTJetEFlowSmoother FFTJetEFlowSmoother.cc RecoJets/FFTJetProducers/plugins/FFTJetEFlowSmoother.cc

 Description: Runs FFTJet filtering code for multiple scales and saves the results

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Jun  2 18:49:49 CDT 2011
//
//

#include <cmath>

// FFTJet headers
#include "fftjet/FrequencyKernelConvolver.hh"
#include "fftjet/DiscreteGauss2d.hh"
#include "fftjet/EquidistantSequence.hh"
#include "fftjet/interpolate.hh"
#include "fftjet/FrequencyKernelConvolver.hh"
#include "fftjet/FrequencySequentialConvolver.hh"
#include "fftjet/DiscreteGauss1d.hh"
#include "fftjet/DiscreteGauss2d.hh"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include <TH3F.h>

// parameter parser header
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// useful utilities collected in the second base
#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

using namespace fftjetcms;

//
// class declaration
//
class FFTJetEFlowSmoother : public FFTJetInterface
{
public:
    explicit FFTJetEFlowSmoother(const edm::ParameterSet&);
    ~FFTJetEFlowSmoother() override;

protected:
    // methods
    void beginJob() override ;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

private:
    FFTJetEFlowSmoother() = delete;
    FFTJetEFlowSmoother(const FFTJetEFlowSmoother&) = delete;
    FFTJetEFlowSmoother& operator=(const FFTJetEFlowSmoother&) = delete;

    void buildKernelConvolver(const edm::ParameterSet&);

    // Storage for convolved energy flow
    std::unique_ptr<fftjet::Grid2d<fftjetcms::Real> > convolvedFlow;

    // Filtering scales
    std::unique_ptr<std::vector<double> > iniScales;

    // The FFT engine(s)
    std::unique_ptr<MyFFTEngine> engine;
    std::unique_ptr<MyFFTEngine> anotherEngine;

    // The pattern recognition kernel(s)
    std::unique_ptr<fftjet::AbsFrequencyKernel> kernel2d;
    std::unique_ptr<fftjet::AbsFrequencyKernel1d> etaKernel;
    std::unique_ptr<fftjet::AbsFrequencyKernel1d> phiKernel;

    // The kernel convolver
    std::unique_ptr<fftjet::AbsConvolverBase<Real> > convolver;

    // The scale power to use for scaling Et
    double scalePower;

    // Overall factor to multiply Et with
    double etConversionFactor;
};

//
// constructors and destructor
//
FFTJetEFlowSmoother::FFTJetEFlowSmoother(const edm::ParameterSet& ps)
    : FFTJetInterface(ps),
      scalePower(ps.getParameter<double>("scalePower")),
      etConversionFactor(ps.getParameter<double>("etConversionFactor"))
{
    // Build the discretization grid
    energyFlow = fftjet_Grid2d_parser(
        ps.getParameter<edm::ParameterSet>("GridConfiguration"));
    checkConfig(energyFlow, "invalid discretization grid");

    // Copy of the grid which will be used for convolutions
    convolvedFlow = std::unique_ptr<fftjet::Grid2d<fftjetcms::Real> >(
        new fftjet::Grid2d<fftjetcms::Real>(*energyFlow));

    // Build the initial set of pattern recognition scales
    iniScales = fftjet_ScaleSet_parser(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    checkConfig(iniScales, "invalid set of scales");

    // Build the FFT engine(s), pattern recognition kernel(s),
    // and the kernel convolver
    buildKernelConvolver(ps);

    // Build the set of pattern recognition scales
    produces<TH3F>(outputLabel);
}


void FFTJetEFlowSmoother::buildKernelConvolver(const edm::ParameterSet& ps)
{
    // Check the parameter named "etaDependentScaleFactors". If the vector
    // of scales is empty we will use 2d kernel, otherwise use 1d kernels
    const std::vector<double> etaDependentScaleFactors(
        ps.getParameter<std::vector<double> >("etaDependentScaleFactors"));

    // Make sure that the number of scale factors provided is correct
    const bool use2dKernel = etaDependentScaleFactors.empty();
    if (!use2dKernel)
        if (etaDependentScaleFactors.size() != energyFlow->nEta())
            throw cms::Exception("FFTJetBadConfig")
                << "invalid number of eta-dependent scale factors"
                << std::endl;

    // Get the eta and phi scales for the kernel(s)
    double kernelEtaScale = ps.getParameter<double>("kernelEtaScale");
    const double kernelPhiScale = ps.getParameter<double>("kernelPhiScale");
    if (kernelEtaScale <= 0.0 || kernelPhiScale <= 0.0)
        throw cms::Exception("FFTJetBadConfig")
            << "invalid kernel scale" << std::endl;

    // FFT assumes that the grid extent in eta is 2*Pi. Adjust the
    // kernel scale in eta to compensate.
    kernelEtaScale *= (2.0*M_PI/(energyFlow->etaMax() - energyFlow->etaMin()));

    // Are we going to try to fix the efficiency near detector edges?
    const bool fixEfficiency = ps.getParameter<bool>("fixEfficiency");

    // Minimum and maximum eta bin for the convolver
    unsigned convolverMinBin = 0, convolverMaxBin = 0;
    if (fixEfficiency || !use2dKernel)
    {
        convolverMinBin = ps.getParameter<unsigned>("convolverMinBin");
        convolverMaxBin = ps.getParameter<unsigned>("convolverMaxBin");
    }

    if (use2dKernel)
    {
        // Build the FFT engine
        engine = std::unique_ptr<MyFFTEngine>(
            new MyFFTEngine(energyFlow->nEta(), energyFlow->nPhi()));

        // 2d kernel
        kernel2d = std::unique_ptr<fftjet::AbsFrequencyKernel>(
            new fftjet::DiscreteGauss2d(
                kernelEtaScale, kernelPhiScale,
                energyFlow->nEta(), energyFlow->nPhi()));

        // 2d convolver
        convolver = std::unique_ptr<fftjet::AbsConvolverBase<Real> >(
            new fftjet::FrequencyKernelConvolver<Real,Complex>(
                engine.get(), kernel2d.get(),
                convolverMinBin, convolverMaxBin));
    }
    else
    {
        // Need separate FFT engines for eta and phi
        engine = std::unique_ptr<MyFFTEngine>(
            new MyFFTEngine(1, energyFlow->nEta()));
        anotherEngine = std::unique_ptr<MyFFTEngine>(
            new MyFFTEngine(1, energyFlow->nPhi()));

        // 1d kernels
        etaKernel = std::unique_ptr<fftjet::AbsFrequencyKernel1d>(
            new fftjet::DiscreteGauss1d(kernelEtaScale, energyFlow->nEta()));

        phiKernel = std::unique_ptr<fftjet::AbsFrequencyKernel1d>(
            new fftjet::DiscreteGauss1d(kernelPhiScale, energyFlow->nPhi()));

        // Sequential convolver
        convolver = std::unique_ptr<fftjet::AbsConvolverBase<Real> >(
            new fftjet::FrequencySequentialConvolver<Real,Complex>(
                engine.get(), anotherEngine.get(),
                etaKernel.get(), phiKernel.get(),
                etaDependentScaleFactors, convolverMinBin,
                convolverMaxBin, fixEfficiency));
    }
}


FFTJetEFlowSmoother::~FFTJetEFlowSmoother()
{
}


// ------------ method called to produce the data  ------------
void FFTJetEFlowSmoother::produce(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    loadInputCollection(iEvent);
    discretizeEnergyFlow();

    // Various useful variables
    const fftjet::Grid2d<Real>& g(*energyFlow);
    const unsigned nScales = iniScales->size();
    const double* scales = &(*iniScales)[0];
    Real* convData = const_cast<Real*>(convolvedFlow->data());
    const unsigned nEta = g.nEta();
    const unsigned nPhi = g.nPhi();
    const double bin0edge = g.phiBin0Edge();

    // We will fill the following histo
    auto pTable = std::make_unique<TH3F>(
                 "FFTJetEFlowSmoother", "FFTJetEFlowSmoother",
                 nScales+1U, -1.5, nScales-0.5,
                 nEta, g.etaMin(), g.etaMax(),
                 nPhi, bin0edge, bin0edge+2.0*M_PI);
    TH3F* h = pTable.get();
    h->SetDirectory(nullptr);
    h->GetXaxis()->SetTitle("Scale");
    h->GetYaxis()->SetTitle("Eta");
    h->GetZaxis()->SetTitle("Phi");

    // Fill the original thing
    double factor = etConversionFactor*pow(getEventScale(), scalePower);
    for (unsigned ieta=0; ieta<nEta; ++ieta)
        for (unsigned iphi=0; iphi<nPhi; ++iphi)
            h->SetBinContent(1U, ieta+1U, iphi+1U, 
                             factor*g.binValue(ieta, iphi));

    // Go over all scales and perform the convolutions
    convolver->setEventData(g.data(), nEta, nPhi);
    for (unsigned iscale=0; iscale<nScales; ++iscale)
    {
        factor = etConversionFactor*pow(scales[iscale], scalePower);

        // Perform the convolution
        convolver->convolveWithKernel(
            scales[iscale], convData,
            convolvedFlow->nEta(), convolvedFlow->nPhi());

        // Fill the output histo
        for (unsigned ieta=0; ieta<nEta; ++ieta)
        {
            const Real* etaData = convData + ieta*nPhi;
            for (unsigned iphi=0; iphi<nPhi; ++iphi)
                h->SetBinContent(iscale+2U, ieta+1U, iphi+1U,
                                 factor*etaData[iphi]);
        }
    }

    iEvent.put(std::move(pTable), outputLabel);
}


// ------------ method called once each job just before starting event loop
void FFTJetEFlowSmoother::beginJob()
{
}


// ------------ method called once each job just after ending the event loop
void FFTJetEFlowSmoother::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetEFlowSmoother);
