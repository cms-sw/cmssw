// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetDijetFilter
// 
/**\class FFTJetDijetFilter FFTJetDijetFilter.cc RecoJets/FFTJetProducers/plugins/FFTJetDijetFilter.cc

 Description: selects good dijet events looking only at the clustering tree

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Jul 18 19:19:40 CDT 2012
//
//
#include <cmath>
#include <cassert>
#include <iostream>

#include "fftjet/EquidistantSequence.hh"
#include "fftjet/ProximityClusteringTree.hh"
#include "fftjet/SparseClusteringTree.hh"
#include "fftjet/PeakEtaPhiDistance.hh"
#include "fftjet/peakEtLifetime.hh"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"
#include "RecoJets/FFTJetAlgorithms/interface/clusteringTreeConverters.h"
#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

using namespace fftjetcms;

//
// class declaration
//
class FFTJetDijetFilter : public edm::EDFilter
{
public:
    typedef fftjet::ProximityClusteringTree<fftjet::Peak,long> ClusteringTree;
    typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;

    explicit FFTJetDijetFilter(const edm::ParameterSet&);
    ~FFTJetDijetFilter() override;

private:
    typedef reco::PattRecoTree<float,reco::PattRecoPeak<float> > StoredTree;

    FFTJetDijetFilter() = delete;
    FFTJetDijetFilter(const FFTJetDijetFilter&) = delete;
    FFTJetDijetFilter& operator=(const FFTJetDijetFilter&) = delete;

    void beginJob() override;
    bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    template<class Ptr>
    inline void checkConfig(const Ptr& ptr, const char* message) const
    {
        if (ptr.get() == nullptr)
            throw cms::Exception("FFTJetBadConfig") << message << std::endl;
    }

    inline double peakPt(const fftjet::Peak& peak) const
    {
        const double s = peak.scale();
        return ptConversionFactor*s*s*peak.magnitude();
    }

    // Module parameters
    edm::InputTag treeLabel;
    double ptConversionFactor;
    double fixedScale;
    double completeEventScale;
    double min1to0PtRatio;
    double minDeltaPhi;
    double maxThirdJetFraction;
    double minPt0;
    double minPt1;
    double maxPeakEta;
    bool insertCompleteEvent;

    // Distance calculator for the clustering tree
    std::unique_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> > distanceCalc;

    // Scales used
    std::unique_ptr<std::vector<double> > iniScales;

    // The clustering trees
    ClusteringTree* clusteringTree;
    SparseTree* sparseTree;

    // Space for peaks
    std::vector<fftjet::Peak> peaks;

    // Space for sparse tree nodes
    std::vector<unsigned> nodes;

    // pass/fail decision counters
    unsigned long nPassed;
    unsigned long nRejected;

    edm::EDGetTokenT<StoredTree> treeToken;
};   


//
// constructors and destructor
//
FFTJetDijetFilter::FFTJetDijetFilter(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, treeLabel),
      init_param(double, ptConversionFactor),
      init_param(double, fixedScale),
      init_param(double, completeEventScale),
      init_param(double, min1to0PtRatio),
      init_param(double, minDeltaPhi),
      init_param(double, maxThirdJetFraction),
      init_param(double, minPt0),
      init_param(double, minPt1),
      init_param(double, maxPeakEta),
      init_param(bool, insertCompleteEvent),
      clusteringTree(nullptr),
      sparseTree(nullptr)
{
    // Parse the set of scales
    iniScales = fftjet_ScaleSet_parser(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    checkConfig(iniScales, "invalid set of scales");
    std::sort(iniScales->begin(), iniScales->end(), std::greater<double>());

    // Parse the distance calculator
    const edm::ParameterSet& TreeDistanceCalculator(
        ps.getParameter<edm::ParameterSet>("TreeDistanceCalculator"));
    distanceCalc = fftjet_DistanceCalculator_parser(TreeDistanceCalculator);
    checkConfig(distanceCalc, "invalid tree distance calculator");

    // Create the clustering tree
    clusteringTree = new ClusteringTree(distanceCalc.get());
    sparseTree = new SparseTree();

    treeToken = consumes<StoredTree>(treeLabel);
}


FFTJetDijetFilter::~FFTJetDijetFilter()
{
    delete sparseTree;
    delete clusteringTree;
}


void FFTJetDijetFilter::beginJob()
{
    nPassed = 0;
    nRejected = 0;
}


void FFTJetDijetFilter::endJob()
{
//     std::cout << "In FTJetDijetFilter::endJob: nPassed = " << nPassed
//               << ", nRejected = " << nRejected << std::endl;
}


// ------------ method called to produce the data  ------------
bool FFTJetDijetFilter::filter(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<StoredTree> input;
    iEvent.getByToken(treeToken, input);

    // Convert the stored tree into a normal FFTJet clustering tree
    // and extract the set of peaks at the requested scale
    const double eventScale = insertCompleteEvent ? completeEventScale : 0.0;
    if (input->isSparse())
    {
        sparsePeakTreeFromStorable(*input, iniScales.get(),
                                   eventScale, sparseTree);
        sparseTree->sortNodes();
        fftjet::updateSplitMergeTimes(*sparseTree, sparseTree->minScale(),
                                      sparseTree->maxScale());
        const unsigned usedLevel = sparseTree->getLevel(fixedScale);
        sparseTree->getLevelNodes(usedLevel, &nodes);
        const unsigned numNodes = nodes.size();
        peaks.clear();
        peaks.reserve(numNodes);
        for (unsigned i=0; i<numNodes; ++i)
            peaks.push_back(sparseTree->uncheckedNode(nodes[i]).getCluster());
    }
    else
    {
        densePeakTreeFromStorable(*input, iniScales.get(),
                                  eventScale, clusteringTree);
        const unsigned usedLevel = clusteringTree->getLevel(fixedScale);
        double actualScale = 0.0;
        long dummyInfo;
        clusteringTree->getLevelData(usedLevel,&actualScale,&peaks,&dummyInfo);
    }

    // Get out if we don't have two clusters
    const unsigned nClusters = peaks.size();
    if (nClusters < 2)
    {
        ++nRejected;
        return false;
    }
    std::sort(peaks.begin(), peaks.end(), std::greater<fftjet::Peak>());

    // Calculate all quantities needed to make the pass/fail decision
    const double pt0 = peakPt(peaks[0]);
    const double pt1 = peakPt(peaks[1]);
    const double dphi = reco::deltaPhi(peaks[0].phi(), peaks[1].phi());
    const double ptratio = pt1/pt0;
    double thirdJetFraction = 0.0;
    if (nClusters > 2)
        thirdJetFraction = peakPt(peaks[2])/(pt0 + pt1);

    // Calculate the cut
    const bool pass = pt0 > minPt0 &&
                      pt1 > minPt1 &&
                      ptratio > min1to0PtRatio &&
                      std::abs(dphi) > minDeltaPhi &&
                      thirdJetFraction < maxThirdJetFraction &&
                      std::abs(peaks[0].eta()) < maxPeakEta &&
                      std::abs(peaks[1].eta()) < maxPeakEta;
    if (pass)
        ++nPassed;
    else
        ++nRejected;
    return pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetDijetFilter);
