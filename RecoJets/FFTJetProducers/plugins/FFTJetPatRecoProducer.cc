// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetPatRecoProducer
// 
/**\class FFTJetPatRecoProducer FFTJetPatRecoProducer.cc RecoJets/FFTJetProducer/plugins/FFTJetPatRecoProducer.cc

 Description: Runs FFTJet pattern recognition stage and saves the results

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Tue Jun 15 12:45:45 CDT 2010
//
//

#include <fstream>

// FFTJet headers
#include "fftjet/ProximityClusteringTree.hh"
#include "fftjet/ClusteringSequencer.hh"
#include "fftjet/ClusteringTreeSparsifier.hh"
#include "fftjet/FrequencyKernelConvolver.hh"
#include "fftjet/FrequencySequentialConvolver.hh"
#include "fftjet/DiscreteGauss1d.hh"
#include "fftjet/DiscreteGauss2d.hh"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// Energy flow object
#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"

// parameter parser header
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// functions which manipulate storable trees
#include "RecoJets/FFTJetAlgorithms/interface/clusteringTreeConverters.h"

// functions which manipulate energy discretization grids
#include "RecoJets/FFTJetAlgorithms/interface/gridConverters.h"

// useful utilities collected in the second base
#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

using namespace fftjetcms;

//
// class declaration
//
class FFTJetPatRecoProducer : public FFTJetInterface
{
public:
    explicit FFTJetPatRecoProducer(const edm::ParameterSet&);
    ~FFTJetPatRecoProducer() override;

protected:
    // Useful local typedefs
    typedef fftjet::ProximityClusteringTree<fftjet::Peak,long> ClusteringTree;
    typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;
    typedef fftjet::ClusteringSequencer<Real> Sequencer;
    typedef fftjet::ClusteringTreeSparsifier<fftjet::Peak,long> Sparsifier;

    // methods
    void beginJob() override ;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

    void buildKernelConvolver(const edm::ParameterSet&);
    fftjet::PeakFinder buildPeakFinder(const edm::ParameterSet&);

    template<class Real>
    void buildSparseProduct(edm::Event&) const;

    template<class Real>
    void buildDenseProduct(edm::Event&) const;

    // The complete clustering tree
    ClusteringTree* clusteringTree;

    // Basically, we need to create FFTJet objects
    // ClusteringSequencer and ClusteringTreeSparsifier
    // which will subsequently perform most of the work
    std::unique_ptr<Sequencer> sequencer;
    std::unique_ptr<Sparsifier> sparsifier;

    // The FFT engine(s)
    std::unique_ptr<MyFFTEngine> engine;
    std::unique_ptr<MyFFTEngine> anotherEngine;

    // The pattern recognition kernel(s)
    std::unique_ptr<fftjet::AbsFrequencyKernel> kernel2d;
    std::unique_ptr<fftjet::AbsFrequencyKernel1d> etaKernel;
    std::unique_ptr<fftjet::AbsFrequencyKernel1d> phiKernel;

    // The kernel convolver
    std::unique_ptr<fftjet::AbsConvolverBase<Real> > convolver;

    // The peak selector for the clustering tree
    std::unique_ptr<fftjet::Functor1<bool,fftjet::Peak> > peakSelector;

    // Distance calculator for the clustering tree
    std::unique_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> > distanceCalc;

    // The sparse clustering tree
    SparseTree sparseTree;

    // The following parameters will define the behavior
    // of the algorithm wrt insertion of the complete event
    // into the clustering tree
    const double completeEventDataCutoff;

    // Are we going to make clustering trees?
    const bool makeClusteringTree;

    // Are we going to verify the data conversion for double precision
    // storage?
    const bool verifyDataConversion;

    // Are we going to store the discretization grid?
    const bool storeDiscretizationGrid;

    // Sparsify the clustering tree?
    const bool sparsify;

private:
    FFTJetPatRecoProducer() = delete;
    FFTJetPatRecoProducer(const FFTJetPatRecoProducer&) = delete;
    FFTJetPatRecoProducer& operator=(const FFTJetPatRecoProducer&) = delete;

    // Members needed for storing grids externally
    std::ofstream externalGridStream;
    bool storeGridsExternally;
    fftjet::Grid2d<float>* extGrid;
};

//
// constructors and destructor
//
FFTJetPatRecoProducer::FFTJetPatRecoProducer(const edm::ParameterSet& ps)
    : FFTJetInterface(ps),
      clusteringTree(nullptr),
      completeEventDataCutoff(ps.getParameter<double>("completeEventDataCutoff")),
      makeClusteringTree(ps.getParameter<bool>("makeClusteringTree")),
      verifyDataConversion(ps.getUntrackedParameter<bool>("verifyDataConversion",false)),
      storeDiscretizationGrid(ps.getParameter<bool>("storeDiscretizationGrid")),
      sparsify(ps.getParameter<bool>("sparsify")),
      extGrid(nullptr)
{
    // register your products
    if (makeClusteringTree)
    {
        if (storeInSinglePrecision())
            produces<reco::PattRecoTree<float,reco::PattRecoPeak<float> > >(outputLabel);
        else
            produces<reco::PattRecoTree<double,reco::PattRecoPeak<double> > >(outputLabel);
    }
    if (storeDiscretizationGrid)
        produces<reco::DiscretizedEnergyFlow>(outputLabel);

    // Check if we want to write the grids into an external file
    const std::string externalGridFile(ps.getParameter<std::string>("externalGridFile"));
    storeGridsExternally = !externalGridFile.empty();
    if (storeGridsExternally)
    {
        externalGridStream.open(externalGridFile.c_str(), std::ios_base::out | 
                                                          std::ios_base::binary);
        if (!externalGridStream.is_open())
            throw cms::Exception("FFTJetBadConfig")
                << "FFTJetPatRecoProducer failed to open file "
                << externalGridFile << std::endl;
    }

    if (!makeClusteringTree && !storeDiscretizationGrid && !storeGridsExternally)
    {
        throw cms::Exception("FFTJetBadConfig")
            << "FFTJetPatRecoProducer is not configured to produce anything"
            << std::endl;
    }

    // now do whatever other initialization is needed

    // Build the discretization grid
    energyFlow = fftjet_Grid2d_parser(
        ps.getParameter<edm::ParameterSet>("GridConfiguration"));
    checkConfig(energyFlow, "invalid discretization grid");

    // Build the FFT engine(s), pattern recognition kernel(s),
    // and the kernel convolver
    buildKernelConvolver(ps);

    // Build the peak selector
    peakSelector = fftjet_PeakSelector_parser(
        ps.getParameter<edm::ParameterSet>("PeakSelectorConfiguration"));
    checkConfig(peakSelector, "invalid peak selector");

    // Build the initial set of pattern recognition scales
    std::unique_ptr<std::vector<double> > iniScales = fftjet_ScaleSet_parser(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    checkConfig(iniScales, "invalid set of scales");

    // Do we want to use the adaptive clustering tree algorithm?
    const unsigned maxAdaptiveScales = 
        ps.getParameter<unsigned>("maxAdaptiveScales");
    const double minAdaptiveRatioLog = 
        ps.getParameter<double>("minAdaptiveRatioLog");
    if (minAdaptiveRatioLog <= 0.0)
        throw cms::Exception("FFTJetBadConfig")
            << "bad adaptive ratio logarithm limit" << std::endl;

    // Make sure that all standard scales are larger than the
    // complete event scale
    if (getEventScale() > 0.0)
    {
      const double cs = getEventScale();
      const unsigned nscales = iniScales->size();
      for (unsigned i=0; i<nscales; ++i)
        if (cs >= (*iniScales)[i])
	  throw cms::Exception("FFTJetBadConfig")
	    << "incompatible scale for complete event" << std::endl;
    }

    // At this point we are ready to build the clustering sequencer
    sequencer = std::unique_ptr<Sequencer>(new Sequencer(
        convolver.get(), peakSelector.get(), buildPeakFinder(ps),
        *iniScales, maxAdaptiveScales, minAdaptiveRatioLog));

    // Build the clustering tree sparsifier
    const edm::ParameterSet& SparsifierConfiguration(
        ps.getParameter<edm::ParameterSet>("SparsifierConfiguration"));
    sparsifier = fftjet_ClusteringTreeSparsifier_parser(
        SparsifierConfiguration);
    checkConfig(sparsifier, "invalid sparsifier parameters");

    // Build the distance calculator for the clustering tree
    const edm::ParameterSet& TreeDistanceCalculator(
        ps.getParameter<edm::ParameterSet>("TreeDistanceCalculator"));
    distanceCalc = fftjet_DistanceCalculator_parser(TreeDistanceCalculator);
    checkConfig(distanceCalc, "invalid tree distance calculator");

    // Build the clustering tree itself
    clusteringTree = new ClusteringTree(distanceCalc.get());
}


void FFTJetPatRecoProducer::buildKernelConvolver(const edm::ParameterSet& ps)
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


fftjet::PeakFinder FFTJetPatRecoProducer::buildPeakFinder(const edm::ParameterSet& ps)
{
    const double peakFinderMaxEta = ps.getParameter<double>("peakFinderMaxEta");
    if (peakFinderMaxEta <= 0.0)
        throw cms::Exception("FFTJetBadConfig")
            << "invalid peak finder eta cut" << std::endl;
    const double maxMagnitude = ps.getParameter<double>("peakFinderMaxMagnitude");
    int minBin = energyFlow->getEtaBin(-peakFinderMaxEta);
    if (minBin < 0)
        minBin = 0;
    int maxBin = energyFlow->getEtaBin(peakFinderMaxEta) + 1;
    if (maxBin < 0)
        maxBin = 0;
    return fftjet::PeakFinder(maxMagnitude, true, minBin, maxBin);
}


FFTJetPatRecoProducer::~FFTJetPatRecoProducer()
{
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
    delete clusteringTree;
    delete extGrid;
}


//
// member functions
//
template<class Real>
void FFTJetPatRecoProducer::buildSparseProduct(edm::Event& ev) const
{
    typedef reco::PattRecoTree<Real,reco::PattRecoPeak<Real> > StoredTree;

    auto tree = std::make_unique<StoredTree>();

    sparsePeakTreeToStorable(sparseTree,
                             sequencer->maxAdaptiveScales(),
                             tree.get());

    // Check that we can restore the tree
    if (verifyDataConversion && !storeInSinglePrecision())
    {
        SparseTree check;
        const std::vector<double>& scalesUsed(sequencer->getInitialScales());
        sparsePeakTreeFromStorable(*tree, &scalesUsed, getEventScale(), &check);
        if (sparseTree != check)
            throw cms::Exception("FFTJetInterface")
                << "Data conversion failed for sparse clustering tree"
                << std::endl;
    }

    ev.put(std::move(tree), outputLabel);
}


template<class Real>
void FFTJetPatRecoProducer::buildDenseProduct(edm::Event& ev) const
{
    typedef reco::PattRecoTree<Real,reco::PattRecoPeak<Real> > StoredTree;

    auto tree = std::make_unique<StoredTree>();

    densePeakTreeToStorable(*clusteringTree,
                            sequencer->maxAdaptiveScales(),
                            tree.get());

    // Check that we can restore the tree
    if (verifyDataConversion && !storeInSinglePrecision())
    {
        ClusteringTree check(distanceCalc.get());
        const std::vector<double>& scalesUsed(sequencer->getInitialScales());
        densePeakTreeFromStorable(*tree, &scalesUsed, getEventScale(), &check);
        if (*clusteringTree != check)
            throw cms::Exception("FFTJetInterface")
                << "Data conversion failed for dense clustering tree"
                << std::endl;
    }

    ev.put(std::move(tree), outputLabel);
}


// ------------ method called to produce the data  ------------
void FFTJetPatRecoProducer::produce(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    loadInputCollection(iEvent);
    discretizeEnergyFlow();

    if (makeClusteringTree)
    {
        sequencer->run(*energyFlow, clusteringTree);
        if (getEventScale() > 0.0)
	    sequencer->insertCompleteEvent(getEventScale(), *energyFlow,
                                           clusteringTree, completeEventDataCutoff);

        if (sparsify)
        {
            sparsifier->sparsify(*clusteringTree, &sparseTree);

            // Do not call the "sortNodes" method of the sparse tree here.
            // Currently, the nodes are sorted by daughter number.
            // This is the way we want it in storage because the stored
            // tree does not include daughter ordering info explicitly.

            if (storeInSinglePrecision())
                buildSparseProduct<float>(iEvent);
            else
                buildSparseProduct<double>(iEvent);
        }
        else
        {
	    if (storeInSinglePrecision())
                buildDenseProduct<float>(iEvent);
            else
                buildDenseProduct<double>(iEvent);
        }
    }

    if (storeDiscretizationGrid)
    {
        const fftjet::Grid2d<Real>& g(*energyFlow);

        auto flow = std::make_unique<reco::DiscretizedEnergyFlow>(
                g.data(), g.title(), g.etaMin(), g.etaMax(),
                g.phiBin0Edge(), g.nEta(), g.nPhi());

        if (verifyDataConversion)
        {
            fftjet::Grid2d<Real> check(
                flow->nEtaBins(), flow->etaMin(), flow->etaMax(),
                flow->nPhiBins(), flow->phiBin0Edge(), flow->title());
            check.blockSet(flow->data(), flow->nEtaBins(), flow->nPhiBins());
            assert(g == check);
        }

        iEvent.put(std::move(flow), outputLabel);
    }

    if (storeGridsExternally)
    {
        if (extGrid)
            copy_Grid2d_data(extGrid, *energyFlow);
        else
            extGrid = convert_Grid2d_to_float(*energyFlow);
        if (!extGrid->write(externalGridStream))
        {
            throw cms::Exception("FFTJetPatRecoProducer::produce")
                << "Failed to write grid data into an external file"
                << std::endl;
        }
    }
}


// ------------ method called once each job just before starting event loop
void FFTJetPatRecoProducer::beginJob()
{
}


// ------------ method called once each job just after ending the event loop
void FFTJetPatRecoProducer::endJob()
{
    if (storeGridsExternally)
        externalGridStream.close();
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetPatRecoProducer);
