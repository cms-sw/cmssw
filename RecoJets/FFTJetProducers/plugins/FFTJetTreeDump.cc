// -*- C++ -*-
//
// Package:    FFTJetTreeDump
// Class:      FFTJetTreeDump
// 
/**\class FFTJetTreeDump FFTJetTreeDump.cc RecoJets/FFTJetProducer/plugins/FFTJetTreeDump.cc

 Description: formats FFTJet clustering trees for subsequent visualization by OpenDX

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Sun Jun 20 14:32:36 CDT 2010
//
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <functional>

// FFTJet headers
#include "fftjet/ProximityClusteringTree.hh"
#include "fftjet/OpenDXPeakTree.hh"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// parameter parser header
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// functions which manipulate storable trees
#include "RecoJets/FFTJetAlgorithms/interface/clusteringTreeConverters.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

using namespace fftjetcms;

//
// class declaration
//
class FFTJetTreeDump : public edm::EDAnalyzer
{
public:
    explicit FFTJetTreeDump(const edm::ParameterSet&);
    ~FFTJetTreeDump() override;

private:
    // Useful local typedefs
    typedef fftjet::ProximityClusteringTree<fftjet::Peak,long> ClusteringTree;
    typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;
    typedef fftjet::OpenDXPeakTree<long,fftjet::AbsClusteringTree> DXFormatter;
    typedef fftjet::OpenDXPeakTree<long,fftjet::SparseClusteringTree> SparseFormatter;
    typedef fftjet::Functor1<double,fftjet::Peak> PeakProperty;
    typedef reco::PattRecoTree<Real,reco::PattRecoPeak<Real> > StoredTree;

    FFTJetTreeDump() = delete;
    FFTJetTreeDump(const FFTJetTreeDump&) = delete;
    FFTJetTreeDump& operator=(const FFTJetTreeDump&) = delete;

    void beginJob() override ;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

    template<class Real>
    void processTreeData(const edm::Event&, std::ofstream&);

    template<class Ptr>
    void checkConfig(const Ptr& ptr, const char* message)
    {
        if (ptr.get() == nullptr)
            throw cms::Exception("FFTJetBadConfig") << message << std::endl;
    }

    // ----------member data ---------------------------
    // The complete clustering tree
    ClusteringTree* clusteringTree;

    const edm::InputTag treeLabel;
    edm::EDGetTokenT<StoredTree> treeToken;

    const std::string outputPrefix;
    const double etaMax;
    const bool storeInSinglePrecision;
    const bool insertCompleteEvent;
    const double completeEventScale;

    // Distance calculator for the clustering tree
    std::unique_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> > distanceCalc;

    // Scales used
    std::unique_ptr<std::vector<double> > iniScales;

    // The sparse clustering tree
    SparseTree sparseTree;

    // Functors which define OpenDX glyph size and color
    std::unique_ptr<PeakProperty> glyphSize;
    std::unique_ptr<PeakProperty> glyphColor;

    // OpenDX formatters
    std::unique_ptr<DXFormatter> denseFormatter;
    std::unique_ptr<SparseFormatter> sparseFormatter;

    unsigned counter;
};

//
// constructors and destructor
//
FFTJetTreeDump::FFTJetTreeDump(const edm::ParameterSet& ps)
    : clusteringTree(nullptr),
      treeLabel(ps.getParameter<edm::InputTag>("treeLabel")),
      outputPrefix(ps.getParameter<std::string>("outputPrefix")),
      etaMax(ps.getParameter<double>("etaMax")),
      storeInSinglePrecision(true),
      insertCompleteEvent(ps.getParameter<bool>("insertCompleteEvent")),
      completeEventScale(ps.getParameter<double>("completeEventScale")),
      counter(0)
{
    if (etaMax < 0.0)
        throw cms::Exception("FFTJetBadConfig")
            << "etaMax can not be negative" << std::endl;

    // Build the set of pattern recognition scales
    const edm::ParameterSet& InitialScales(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    iniScales = fftjet_ScaleSet_parser(InitialScales);
    checkConfig(iniScales, "invalid set of scales");
    std::sort(iniScales->begin(), iniScales->end(), std::greater<double>());

    // Distance calculator for the clustering tree
    const edm::ParameterSet& TreeDistanceCalculator(
        ps.getParameter<edm::ParameterSet>("TreeDistanceCalculator"));
    distanceCalc = fftjet_DistanceCalculator_parser(TreeDistanceCalculator);
    checkConfig(distanceCalc, "invalid tree distance calculator");

    // Determine representations for the OpenDX glyph size and color
    const edm::ParameterSet& GlyphSize(
        ps.getParameter<edm::ParameterSet>("GlyphSize"));
    glyphSize = fftjet_PeakFunctor_parser(GlyphSize);
    checkConfig(glyphSize, "invalid glyph size parameters");

    const edm::ParameterSet& GlyphColor(
        ps.getParameter<edm::ParameterSet>("GlyphColor"));
    glyphColor = fftjet_PeakFunctor_parser(GlyphColor);
    checkConfig(glyphColor, "invalid glyph color parameters");

    // Build the tree formatters
    denseFormatter = std::unique_ptr<DXFormatter>(new DXFormatter(
        glyphSize.get(), glyphColor.get(), etaMax));
    sparseFormatter = std::unique_ptr<SparseFormatter>(new SparseFormatter(
        glyphSize.get(), glyphColor.get(), etaMax));

    // Build the clustering tree
    clusteringTree = new ClusteringTree(distanceCalc.get());

    treeToken = consumes<StoredTree>(treeLabel);
}


FFTJetTreeDump::~FFTJetTreeDump()
{
    delete clusteringTree;
}


//
// member functions
//
template<class Real>
void FFTJetTreeDump::processTreeData(const edm::Event& iEvent,
                                     std::ofstream& file)
{
    // Get the event number
    edm::RunNumber_t const  runNum = iEvent.id().run();
    edm::EventNumber_t const evNum = iEvent.id().event();

    // Get the input
    edm::Handle<StoredTree> input;
    iEvent.getByToken(treeToken, input);

    const double eventScale = insertCompleteEvent ? completeEventScale : 0.0;
    if (input->isSparse())
    {
        sparsePeakTreeFromStorable(*input, iniScales.get(),
                                   eventScale, &sparseTree);
        sparseFormatter->setTree(sparseTree, runNum, evNum);
        file << *sparseFormatter << std::endl;
    }
    else
    {
        densePeakTreeFromStorable(*input, iniScales.get(),
                                  eventScale, clusteringTree);
        denseFormatter->setTree(*clusteringTree, runNum, evNum);
        file << *denseFormatter << std::endl;
    }
}


// ------------ method called to for each event  ------------
void FFTJetTreeDump::analyze(const edm::Event& iEvent,
                             const edm::EventSetup& iSetup)
{
    // Create the file name
    std::ostringstream filename;
    filename << outputPrefix << '_' << counter++ << ".dx";

    // Open the file
    std::ofstream file(filename.str().c_str());
    if (!file)
        throw cms::Exception("FFTJetBadConfig") 
            << "Failed to open file \""
            << filename.str() << "\"" << std::endl;

    if (storeInSinglePrecision)
        processTreeData<float>(iEvent, file);
    else
        processTreeData<double>(iEvent, file);
}


// ------------ method called once each job just before starting event loop
void FFTJetTreeDump::beginJob()
{
}

// ------------ method called once each job just after ending the event loop
void FFTJetTreeDump::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetTreeDump);
