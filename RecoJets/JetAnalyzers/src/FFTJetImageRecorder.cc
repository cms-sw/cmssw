// -*- C++ -*-
//
// Package:    JetAnalyzers
// Class:      FFTJetImageRecorder
// 
/**\class FFTJetImageRecorder FFTJetImageRecorder.cc RecoJets/FFTJetProducers/plugins/FFTJetImageRecorder.cc

 Description: Runs FFTJet filtering code for multiple scales and saves the results

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Jun  2 18:49:49 CDT 2011
// $Id: FFTJetImageRecorder.cc,v 1.1 2011/06/03 05:05:44 igv Exp $
//
//

#include <cmath>
#include <cassert>
#include <sstream>
#include <numeric>

#include "TNtuple.h"

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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

// parameter parser header
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// useful utilities collected in the second base
#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

using namespace fftjetcms;

//
// class declaration
//
class FFTJetImageRecorder : public edm::EDAnalyzer, public FFTJetInterface
{
public:
    explicit FFTJetImageRecorder(const edm::ParameterSet&);
    ~FFTJetImageRecorder();

protected:
    // methods
    void beginJob() ;
    void analyze(const edm::Event&, const edm::EventSetup&);
    void endJob() ;

private:
    FFTJetImageRecorder();
    FFTJetImageRecorder(const FFTJetImageRecorder&);
    FFTJetImageRecorder& operator=(const FFTJetImageRecorder&);

    void buildKernelConvolver(const edm::ParameterSet&);
    void writeToExternalFile(const edm::Event&, std::auto_ptr<TH3F>);

    // Storage for convolved energy flow
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> > convolvedFlow;

    // Filtering scales
    std::auto_ptr<std::vector<double> > iniScales;

    // The FFT engine(s)
    std::auto_ptr<MyFFTEngine> engine;
    std::auto_ptr<MyFFTEngine> anotherEngine;

    // The pattern recognition kernel(s)
    std::auto_ptr<fftjet::AbsFrequencyKernel> kernel2d;
    std::auto_ptr<fftjet::AbsFrequencyKernel1d> etaKernel;
    std::auto_ptr<fftjet::AbsFrequencyKernel1d> phiKernel;

    // The kernel convolver
    std::auto_ptr<fftjet::AbsConvolverBase<Real> > convolver;

    // The scale power to use for scaling Et
    double scalePower;

    // Overall factor to multiply Et with
    double etConversionFactor;

    // Event counter
    unsigned long counter;
};

//
// constructors and destructor
//
FFTJetImageRecorder::FFTJetImageRecorder(const edm::ParameterSet& ps)
    : FFTJetInterface(ps),
      scalePower(ps.getParameter<double>("scalePower")),
      etConversionFactor(ps.getParameter<double>("etConversionFactor")),
      counter(0)
{
    // Build the discretization grid
    energyFlow = fftjet_Grid2d_parser(
        ps.getParameter<edm::ParameterSet>("GridConfiguration"));
    checkConfig(energyFlow, "invalid discretization grid");

    // Copy of the grid which will be used for convolutions
    convolvedFlow = std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >(
        new fftjet::Grid2d<fftjetcms::Real>(*energyFlow));

    // Build the initial set of pattern recognition scales
    iniScales = fftjet_ScaleSet_parser(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    checkConfig(iniScales, "invalid set of scales");

    // Build the FFT engine(s), pattern recognition kernel(s),
    // and the kernel convolver
    buildKernelConvolver(ps);
}


void FFTJetImageRecorder::buildKernelConvolver(const edm::ParameterSet& ps)
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
        engine = std::auto_ptr<MyFFTEngine>(
            new MyFFTEngine(energyFlow->nEta(), energyFlow->nPhi()));

        // 2d kernel
        kernel2d = std::auto_ptr<fftjet::AbsFrequencyKernel>(
            new fftjet::DiscreteGauss2d(
                kernelEtaScale, kernelPhiScale,
                energyFlow->nEta(), energyFlow->nPhi()));

        // 2d convolver
        convolver = std::auto_ptr<fftjet::AbsConvolverBase<Real> >(
            new fftjet::FrequencyKernelConvolver<Real,Complex>(
                engine.get(), kernel2d.get(),
                convolverMinBin, convolverMaxBin));
    }
    else
    {
        // Need separate FFT engines for eta and phi
        engine = std::auto_ptr<MyFFTEngine>(
            new MyFFTEngine(1, energyFlow->nEta()));
        anotherEngine = std::auto_ptr<MyFFTEngine>(
            new MyFFTEngine(1, energyFlow->nPhi()));

        // 1d kernels
        etaKernel = std::auto_ptr<fftjet::AbsFrequencyKernel1d>(
            new fftjet::DiscreteGauss1d(kernelEtaScale, energyFlow->nEta()));

        phiKernel = std::auto_ptr<fftjet::AbsFrequencyKernel1d>(
            new fftjet::DiscreteGauss1d(kernelPhiScale, energyFlow->nPhi()));

        // Sequential convolver
        convolver = std::auto_ptr<fftjet::AbsConvolverBase<Real> >(
            new fftjet::FrequencySequentialConvolver<Real,Complex>(
                engine.get(), anotherEngine.get(),
                etaKernel.get(), phiKernel.get(),
                etaDependentScaleFactors, convolverMinBin,
                convolverMaxBin, fixEfficiency));
    }
}


FFTJetImageRecorder::~FFTJetImageRecorder()
{
}


// ------------ method called to analyze the data  ------------
void FFTJetImageRecorder::analyze(
    const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
    std::auto_ptr<TH3F> pTable(
        new TH3F("FFTJetImageRecorder", "FFTJetImageRecorder",
                 nScales+1U, -1.5, nScales-0.5,
                 nEta, g.etaMin(), g.etaMax(),
                 nPhi, bin0edge, bin0edge+2.0*M_PI));
    TH3F* h = pTable.get();
    h->SetDirectory(0);
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

    writeToExternalFile(iEvent, pTable);
    ++counter;
}


void FFTJetImageRecorder::writeToExternalFile(const edm::Event& iEvent,
                                              std::auto_ptr<TH3F> hptr)
{
    const long runnumber = iEvent.id().run();
    const long eventnumber = iEvent.id().event();

    TH3F* histo = hptr.release();

    std::ostringstream os;
    os << histo->GetName() << '_' << counter << '_'
       << runnumber << '_' << eventnumber;
    const std::string& newname(os.str());
    histo->SetNameTitle(newname.c_str(), newname.c_str());

    edm::Service<TFileService> fs;
    histo->SetDirectory(fs->getBareDirectory());
}


// ------------ method called once each job just before starting event loop
void FFTJetImageRecorder::beginJob()
{
    edm::Service<TFileService> fs;
    fs->make<TNtuple>("dummy", "dummy", "var");
}


// ------------ method called once each job just after ending the event loop
void FFTJetImageRecorder::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetImageRecorder);
