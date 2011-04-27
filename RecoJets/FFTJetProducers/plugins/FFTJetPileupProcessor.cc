// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetPileupProcessor
// 
/**\class FFTJetPileupProcessor FFTJetPileupProcessor.cc RecoJets/FFTJetProducers/plugins/FFTJetPileupProcessor.cc

 Description: Runs FFTJet multiscale pileup filtering code and saves the results

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Apr 20 13:52:23 CDT 2011
// $Id: FFTJetPileupProcessor.cc,v 1.0 2011/04/20 00:19:43 igv Exp $
//
//

#include <cmath>

// FFTJet headers
#include "fftjet/FrequencyKernelConvolver.hh"
#include "fftjet/DiscreteGauss2d.hh"
#include "fftjet/EquidistantSequence.hh"
#include "fftjet/interpolate.hh"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

// parameter parser header
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// useful utilities collected in the second base
#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

using namespace fftjetcms;

//
// class declaration
//
class FFTJetPileupProcessor : public edm::EDProducer, public FFTJetInterface
{
public:
    explicit FFTJetPileupProcessor(const edm::ParameterSet&);
    ~FFTJetPileupProcessor();

protected:
    // methods
    void beginJob() ;
    void produce(edm::Event&, const edm::EventSetup&);
    void endJob() ;

private:
    FFTJetPileupProcessor();
    FFTJetPileupProcessor(const FFTJetPileupProcessor&);
    FFTJetPileupProcessor& operator=(const FFTJetPileupProcessor&);

    void buildKernelConvolver(const edm::ParameterSet&);

    // The FFT engine
    std::auto_ptr<MyFFTEngine> engine;

    // The pattern recognition kernel(s)
    std::auto_ptr<fftjet::AbsFrequencyKernel> kernel2d;

    // The kernel convolver
    std::auto_ptr<fftjet::AbsConvolverBase<Real> > convolver;

    // Storage for convolved energy flow
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> > convolvedFlow;

    // Filtering scales
    std::auto_ptr<fftjet::EquidistantInLogSpace> filterScales;

    // Number of percentile points to use
    unsigned nPercentiles;
};

//
// constructors and destructor
//
FFTJetPileupProcessor::FFTJetPileupProcessor(const edm::ParameterSet& ps)
    : FFTJetInterface(ps),
      nPercentiles(ps.getParameter<unsigned>("nPercentiles"))
{
    // Build the discretization grid
    energyFlow = fftjet_Grid2d_parser(
        ps.getParameter<edm::ParameterSet>("GridConfiguration"));
    checkConfig(energyFlow, "invalid discretization grid");

    // Copy of the grid which will be used for convolutions
    convolvedFlow = std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >(
        new fftjet::Grid2d<fftjetcms::Real>(*energyFlow));

    // Build the FFT engine(s), pattern recognition kernel(s),
    // and the kernel convolver
    buildKernelConvolver(ps);

    // Build the set of pattern recognition scales
    const unsigned nScales = ps.getParameter<unsigned>("nScales");
    const double minScale = ps.getParameter<double>("minScale");
    const double maxScale = ps.getParameter<double>("maxScale");
    if (minScale <= 0.0 || maxScale < minScale || nScales == 0U)
        throw cms::Exception("FFTJetBadConfig")
            << "invalid filter scales" << std::endl;

    filterScales = std::auto_ptr<fftjet::EquidistantInLogSpace>(
        new fftjet::EquidistantInLogSpace(minScale, maxScale, nScales));

    produces<TH2D>(outputLabel);
}


void FFTJetPileupProcessor::buildKernelConvolver(const edm::ParameterSet& ps)
{
    // Get the eta and phi scales for the kernel(s)
    double kernelEtaScale = ps.getParameter<double>("kernelEtaScale");
    const double kernelPhiScale = ps.getParameter<double>("kernelPhiScale");
    if (kernelEtaScale <= 0.0 || kernelPhiScale <= 0.0)
        throw cms::Exception("FFTJetBadConfig")
            << "invalid kernel scales" << std::endl;

    // FFT assumes that the grid extent in eta is 2*Pi. Adjust the
    // kernel scale in eta to compensate.
    kernelEtaScale *= (2.0*M_PI/(energyFlow->etaMax() - energyFlow->etaMin()));

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
            engine.get(), kernel2d.get()));
}


FFTJetPileupProcessor::~FFTJetPileupProcessor()
{
}


// ------------ method called to produce the data  ------------
void FFTJetPileupProcessor::produce(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    loadInputCollection(iEvent);
    discretizeEnergyFlow();

    // Various useful variables
    const fftjet::Grid2d<Real>& g(*energyFlow);
    const unsigned nScales = filterScales->size();
    const double* scales = &(*filterScales)[0];
    Real* convData = const_cast<Real*>(convolvedFlow->data());
    const unsigned dataLen = convolvedFlow->nEta()*convolvedFlow->nPhi();

    // We will fill the following histo
    std::auto_ptr<TH2D> pTable(new TH2D("FFTJetPileupProcessor",
                                        "FFTJetPileupProcessor",
                                        nScales, -0.5, nScales-0.5,
                                        nPercentiles, 0.0, 1.0));
    pTable->SetDirectory(0);
    pTable->GetXaxis()->SetTitle("Filter Number");
    pTable->GetYaxis()->SetTitle("Et CDF");
    pTable->GetZaxis()->SetTitle("Et Density");

    // Go over all scales and perform the convolutions
    convolver->setEventData(g.data(), g.nEta(), g.nPhi());
    for (unsigned iscale=0; iscale<nScales; ++iscale)
    {
        // Perform the convolution
        convolver->convolveWithKernel(
            scales[iscale], convData,
            convolvedFlow->nEta(), convolvedFlow->nPhi());

        // Sort the convolved data
        std::sort(convData, convData+dataLen);

        // Determine the percentile points
        for (unsigned iper=0; iper<nPercentiles; ++iper)
        {
            // Map percentile 0 into point number 0, 
            // 1 into point number dataLen-1
            const double q = (iper + 0.5)/nPercentiles;
            const double dindex = q*(dataLen-1U);
            const unsigned ilow = static_cast<unsigned>(std::floor(dindex));
            const double percentile = fftjet::lin_interpolate_1d(
                ilow, ilow+1U, convData[ilow], convData[ilow+1U], dindex);

            // Store the calculated percentile
            pTable->SetBinContent(iscale+1U, iper+1U, percentile);
        }
    }

    iEvent.put(pTable, outputLabel);
}


// ------------ method called once each job just before starting event loop
void FFTJetPileupProcessor::beginJob()
{
}


// ------------ method called once each job just after ending the event loop
void FFTJetPileupProcessor::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetPileupProcessor);
