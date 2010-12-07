// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetProducer
// 
/**\class FFTJetProducer FFTJetProducer.cc RecoJets/FFTJetProducers/plugins/FFTJetProducer.cc

 Description: makes jets using FFTJet clustering tree

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Sun Jun 20 14:32:36 CDT 2010
// $Id: FFTJetProducer.cc,v 1.1 2010/12/06 17:33:19 igv Exp $
//
//

#include <iostream>
#include <fstream>
#include <algorithm>

// Header for this class
#include "RecoJets/FFTJetProducers/plugins/FFTJetProducer.h"

// Additional FFTJet headers
#include "fftjet/VectorRecombinationAlgFactory.hh"
#include "fftjet/RecombinationAlgFactory.hh"

// Framework include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data formats
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/FFTCaloJetCollection.h"
#include "DataFormats/JetReco/interface/FFTGenJetCollection.h"
#include "DataFormats/JetReco/interface/FFTPFJetCollection.h"
#include "DataFormats/JetReco/interface/FFTJPTJetCollection.h"
#include "DataFormats/JetReco/interface/FFTBasicJetCollection.h"
#include "DataFormats/JetReco/interface/FFTTrackJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/FFTJetProducerSummary.h"

#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

#include "RecoJets/FFTJetAlgorithms/interface/clusteringTreeConverters.h"
#include "RecoJets/FFTJetAlgorithms/interface/jetConverters.h"
#include "RecoJets/FFTJetAlgorithms/interface/DiscretizedEnergyFlow.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"


#define make_param(type, varname) const \
    type & varname (ps.getParameter< type >( #varname ))

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

// A generic switch statement based on jet type.
// Defining it in a single place avoids potential errors
// in case another jet type is introduced in the future.
//
// JPTJet is omitted for now: there is no reco::writeSpecific method
// for it (see header JetSpecific.h in the JetProducers package) 
//
#define jet_type_switch(method, arg1, arg2) do {\
    switch (jetType)\
    {\
    case CALOJET:\
        method <reco::CaloJet> ( arg1 , arg2 );\
        break;\
    case PFJET:\
        method <reco::PFJet> ( arg1 , arg2 );\
        break;\
    case GENJET:\
        method <reco::GenJet> ( arg1 , arg2 );\
        break;\
    case TRACKJET:\
        method <reco::TrackJet> ( arg1 , arg2 );\
        break;\
    case BASICJET:\
        method <reco::BasicJet> ( arg1 , arg2 );\
        break;\
    default:\
        assert(!"ERROR in FFTJetProducer : invalid jet type."\
               " This is a bug. Please report.");\
    }\
} while(0);

using namespace fftjetcms;

FFTJetProducer::Resolution FFTJetProducer::parse_resolution(
    const std::string& name)
{
    if (!name.compare("fixed"))
        return FIXED;
    else if (!name.compare("maximallyStable"))
        return MAXIMALLY_STABLE;
    else if (!name.compare("globallyAdaptive"))
        return GLOBALLY_ADAPTIVE;
    else if (!name.compare("locallyAdaptive"))
        return LOCALLY_ADAPTIVE;
    else
        throw cms::Exception("FFTJetBadConfig")
            << "Invalid resolution specification \""
            << name << "\"" << std::endl;
}


template <typename T>
void FFTJetProducer::makeProduces(
    const std::string& alias, const std::string& tag)
{
    produces<std::vector<reco::FFTAnyJet<T> > >(tag).setBranchAlias(alias);
}

//
// constructors and destructor
//
FFTJetProducer::FFTJetProducer(const edm::ParameterSet& ps)
    : FFTJetInterface(ps),
      myConfiguration(ps),
      init_param(edm::InputTag, treeLabel),
      init_param(bool, useGriddedAlgorithm),
      init_param(bool, reuseExistingGrid),
      init_param(unsigned, maxIterations),
      init_param(unsigned, nJetsRequiredToConverge),
      init_param(double, convergenceDistance),
      init_param(bool, assignConstituents),
      init_param(bool, resumConstituents),
      init_param(double, fixedScale),
      init_param(double, minStableScale),
      init_param(double, maxStableScale),
      init_param(double, stabilityAlpha),
      init_param(double, noiseLevel),
      init_param(unsigned, nClustersRequested),
      init_param(double, gridScanMaxEta),
      init_param(std::string, recombinationAlgorithm),
      init_param(bool, isCrisp),
      init_param(double, unlikelyBgWeight),
      init_param(double, recombinationDataCutoff),
      resolution(parse_resolution(ps.getParameter<std::string>("resolution"))),

      minLevel(0),
      maxLevel(0),
      usedLevel(0),
      unused(0.0),
      iterationsPerformed(1U),
      constituents(200)
{
    // Check that the settings make sense
    if (resumConstituents && !assignConstituents)
        throw cms::Exception("FFTJetBadConfig") 
            << "Can't resum constituents if they are not assigned"
            << std::endl;

    produces<reco::FFTJetProducerSummary>(outputLabel);
    const std::string alias(ps.getUntrackedParameter<std::string>(
                                "alias", outputLabel));
    jet_type_switch(makeProduces, alias, outputLabel);

    // Build the set of pattern recognition scales.
    // This is needed in order to read the clustering tree
    // from the event record.
    iniScales = fftjet_ScaleSet_parser(
        ps.getParameter<edm::ParameterSet>("InitialScales"));
    checkConfig(iniScales, "invalid set of scales");
    std::sort(iniScales->begin(), iniScales->end(), std::greater<double>());

    // Most of the configuration has to be performed inside
    // the "beginJob" method. This is because chaining of the
    // parsers between this base class and the derived classes
    // can not work from the constructor of the base class.
}


FFTJetProducer::~FFTJetProducer()
{
}

//
// member functions
//
template<class Real>
void FFTJetProducer::loadSparseTreeData(const edm::Event& iEvent)
{
    typedef reco::PattRecoTree<Real,reco::PattRecoPeak<Real> > StoredTree;

    // Get the input
    edm::Handle<StoredTree> input;
    iEvent.getByLabel(treeLabel, input);

    if (!input->isSparse())
        throw cms::Exception("FFTJetBadConfig") 
            << "The stored clustering tree is not sparse" << std::endl;

    sparsePeakTreeFromStorable(*input, iniScales.get(), getEventScale(), &sparseTree);
    sparseTree.sortNodes();
}


void FFTJetProducer::selectPreclusters(
    const SparseTree& tree,
    const fftjet::Functor1<bool,fftjet::Peak>& peakSelector,
    std::vector<fftjet::Peak>* preclusters)
{
    nodes.clear();
    selectTreeNodes(tree, peakSelector, &nodes);

    // Fill out the vector of preclusters using the tree node ids
    const unsigned nNodes = nodes.size();
    const SparseTree::NodeId* pnodes = nNodes ? &nodes[0] : 0;
    preclusters->reserve(nNodes);
    for (unsigned i=0; i<nNodes; ++i)
        preclusters->push_back(
            sparseTree.uncheckedNode(pnodes[i]).getCluster());

    // Set the status word to indicate the resolution scheme used
    fftjet::Peak* clusters = nNodes ? &(*preclusters)[0] : 0;
    for (unsigned i=0; i<nNodes; ++i)
        clusters[i].setStatus(resolution);
}


void FFTJetProducer::selectTreeNodes(
    const SparseTree& tree,
    const fftjet::Functor1<bool,fftjet::Peak>& peakSelect,
    std::vector<SparseTree::NodeId>* mynodes)
{
    minLevel = maxLevel = usedLevel = 0;

    // Get the tree nodes which pass the cuts
    // (according to the selected resolution strategy)
    switch (resolution)
    {
    case FIXED:
    {
        usedLevel = tree.getLevel(fixedScale);
        tree.getPassingNodes(usedLevel, peakSelect, mynodes);
    }
    break;

    case MAXIMALLY_STABLE:
    {
        const unsigned minStartingLevel = maxStableScale > 0.0 ? 
            tree.getLevel(maxStableScale) : 0;
        const unsigned maxStartingLevel = minStableScale > 0.0 ?
            tree.getLevel(minStableScale) : UINT_MAX;

        if (tree.stableClusterCount(
                peakSelect, &minLevel, &maxLevel, stabilityAlpha,
                minStartingLevel, maxStartingLevel))
        {
            usedLevel = (minLevel + maxLevel)/2;
            tree.getPassingNodes(usedLevel, peakSelect, mynodes);
        }
    }
    break;

    case GLOBALLY_ADAPTIVE:
    {
        const bool stable = tree.clusterCountLevels(
            nClustersRequested, peakSelect, &minLevel, &maxLevel);
        if (minLevel || maxLevel)
        {
            usedLevel = (minLevel + maxLevel)/2;
            if (!stable)
            {
                const int maxlev = tree.maxStoredLevel();
                bool levelFound = false;
                for (int delta=0; delta<=maxlev && !levelFound; ++delta)
                    for (int ifac=1; ifac>-2 && !levelFound; ifac-=2)
                    {
                        const int level = usedLevel + ifac*delta;
                        if (level > 0 && level <= maxlev)
                            if (occupancy[level] == nClustersRequested)
                            {
                                usedLevel = level;
                                levelFound = true;
                            }
                    }
                assert(levelFound);
            }
        }
        else
        {
            // Can't find that exact number of preclusters.
            // Try to get the next best thing.
            usedLevel = 1;
            const unsigned occ1 = occupancy[1];
            if (nClustersRequested >= occ1)
            {
                const unsigned maxlev = tree.maxStoredLevel();
                if (nClustersRequested > occupancy[maxlev])
                    usedLevel = maxlev;
                else
                {
                    // It would be nice to use "lower_bound" here,
                    // but the occupancy is not necessarily monotonous.
                    unsigned bestDelta = nClustersRequested > occ1 ?
                        nClustersRequested - occ1 : occ1 - nClustersRequested;
                    for (unsigned level=2; level<=maxlev; ++level)
                    {
                        const unsigned n = occupancy[level];
                        const unsigned d = nClustersRequested > n ? 
                            nClustersRequested - n : n - nClustersRequested;
                        if (d < bestDelta)
                        {
                            bestDelta = d;
                            usedLevel = level;
                        }
                    }
                }
            }
        }
        tree.getPassingNodes(usedLevel, peakSelect, mynodes);
    }
    break;

    case LOCALLY_ADAPTIVE:
    {
        usedLevel = tree.getLevel(fixedScale);
        tree.getMagS2OptimalNodes(peakSelect, nClustersRequested,
                                  usedLevel, mynodes, &thresholds);
    }
    break;

    default:
        assert(!"ERROR in FFTJetProducer::selectTreeNodes : "
               "should never get here! This is a bug. Please report.");
    }
}


void FFTJetProducer::prepareRecombinationScales()
{
    const unsigned nClus = preclusters.size();
    if (nClus)
    {
        fftjet::Peak* clus = &preclusters[0];
        fftjet::Functor1<double,fftjet::Peak>& 
            scaleCalc(*recoScaleCalcPeak);
        fftjet::Functor1<double,fftjet::Peak>& 
            ratioCalc(*recoScaleRatioCalcPeak);
        fftjet::Functor1<double,fftjet::Peak>& 
            factorCalc(*memberFactorCalcPeak);

        for (unsigned i=0; i<nClus; ++i)
        {
            clus[i].setRecoScale(scaleCalc(clus[i]));
            clus[i].setRecoScaleRatio(ratioCalc(clus[i]));
            clus[i].setMembershipFactor(factorCalc(clus[i]));
        }
    }
}


void FFTJetProducer::buildGridAlg()
{
    int minBin = energyFlow->getEtaBin(-gridScanMaxEta);
    if (minBin < 0)
        minBin = 0;
    int maxBin = energyFlow->getEtaBin(gridScanMaxEta) + 1;
    if (maxBin < 0)
        maxBin = 0;
    
    fftjet::DefaultRecombinationAlgFactory<
        Real,VectorLike,BgData,VBuilder> factory;
    if (factory[recombinationAlgorithm] == NULL)
        throw cms::Exception("FFTJetBadConfig")
            << "Invalid grid recombination algorithm \""
            << recombinationAlgorithm << "\"" << std::endl;
    gridAlg = std::auto_ptr<GridAlg>(
        factory[recombinationAlgorithm]->create(
            jetMembershipFunction.get(),
            bgMembershipFunction.get(),
            unlikelyBgWeight, recombinationDataCutoff,
            isCrisp, false, assignConstituents, minBin, maxBin));
}


void FFTJetProducer::loadEnergyFlow(const edm::Event& iEvent)
{
    edm::Handle<DiscretizedEnergyFlow> input;
    iEvent.getByLabel(treeLabel, input);

    // Make sure that the grid is compatible with the stored one
    bool rebuildGrid = energyFlow.get() == NULL;
    if (!rebuildGrid)
        rebuildGrid = 
            !(energyFlow->nEta() == input->nEtaBins() &&
              energyFlow->nPhi() == input->nPhiBins() &&
              energyFlow->etaMin() == input->etaMin() &&
              energyFlow->etaMax() == input->etaMax() &&
              energyFlow->phiBin0Edge() == input->phiBin0Edge());
    if (rebuildGrid)
    {
        // We should not get here very often...
        energyFlow = std::auto_ptr<fftjet::Grid2d<Real> >(
            new fftjet::Grid2d<Real>(
                input->nEtaBins(), input->etaMin(), input->etaMax(),
                input->nPhiBins(), input->phiBin0Edge(), input->title()));
        buildGridAlg();
    }
    energyFlow->blockSet(input->data(), input->nEtaBins(), input->nPhiBins());
}


bool FFTJetProducer::checkConvergence(const std::vector<RecoFFTJet>& previous,
                                      std::vector<RecoFFTJet>& nextSet)
{
    fftjet::Functor2<double,RecoFFTJet,RecoFFTJet>&
        distanceCalc(*jetDistanceCalc);

    const unsigned nJets = previous.size();
    const RecoFFTJet* prev = &previous[0];
    RecoFFTJet* next = &nextSet[0];

    // Calculate convergence distances for all jets
    bool converged = true;
    for (unsigned i=0; i<nJets; ++i)
    {
        const double d = distanceCalc(prev[i], next[i]);
        next[i].setConvergenceDistance(d);
        if (i < nJetsRequiredToConverge && d > convergenceDistance)
            converged = false;
    }

    return converged;
}


unsigned FFTJetProducer::iterateJetReconstruction()
{
    fftjet::Functor1<double,RecoFFTJet>& scaleCalc(*recoScaleCalcJet);
    fftjet::Functor1<double,RecoFFTJet>& ratioCalc(*recoScaleRatioCalcJet);
    fftjet::Functor1<double,RecoFFTJet>& factorCalc(*memberFactorCalcJet);

    const unsigned nJets = recoJets.size();

    unsigned iterNum = 1U;
    bool converged = false;
    for (; iterNum<maxIterations && !converged; ++iterNum)
    {
        // Recreate the vector of preclusters using the jets
        const RecoFFTJet* jets = &recoJets[0];
        iterPreclusters.clear();
        iterPreclusters.reserve(nJets);
        for (unsigned i=0; i<nJets; ++i)
        {
            const RecoFFTJet& jet(jets[i]);
            fftjet::Peak p(jet.precluster());
            p.setEtaPhi(jet.vec().Eta(), jet.vec().Phi());
            p.setRecoScale(scaleCalc(jet));
            p.setRecoScaleRatio(ratioCalc(jet));
            p.setMembershipFactor(factorCalc(jet));
            iterPreclusters.push_back(p);
        }

        // Run the algorithm
        int status = 0;
        if (useGriddedAlgorithm)
            status = gridAlg->run(iterPreclusters, *energyFlow,
                                  &noiseLevel, 1U, 1U,
                                  &iterJets, &unclustered, &unused);
        else
            status = recoAlg->run(iterPreclusters, eventData, &noiseLevel, 1U,
                                  &iterJets, &unclustered, &unused);
        if (status)
            throw cms::Exception("FFTJetInterface")
                << "FFTJet algorithm failed" << std::endl;
        assert(iterJets.size() == nJets);

        // Figure out if the iterations have converged
        converged = checkConvergence(recoJets, iterJets);

        // Prepare for the next cycle
        iterJets.swap(recoJets);
    }

    // Plug in the original precluster coordinates into the result
    assert(preclusters.size() == nJets);
    RecoFFTJet* jets = &recoJets[0];
    for (unsigned i=0; i<nJets; ++i)
    {
        const fftjet::Peak& oldp(preclusters[i]);
        jets[i].setPeakEtaPhi(oldp.eta(), oldp.phi());
    }

    // If we have converged on the last cycle, the result
    // would be indistinguishable from no convergence.
    // Because of this, raise the counter by one to indicate
    // the case when the convergence is not achieved.
    if (!converged)
        ++iterNum;

    return iterNum;
}


void FFTJetProducer::determineGriddedConstituents()
{
    const unsigned nJets = recoJets.size();
    const unsigned* clusterMask = gridAlg->getClusterMask();
    const int nEta = gridAlg->getLastNEta();
    const int nPhi = gridAlg->getLastNPhi();
    const fftjet::Grid2d<Real>& g(*energyFlow);

    const unsigned nInputs = eventData.size();
    const VectorLike* inp = nInputs ? &eventData[0] : 0;
    const unsigned* candIdx = nInputs ? &candidateIndex[0] : 0;
    for (unsigned i=0; i<nInputs; ++i)
    {
        const VectorLike& item(inp[i]);
        const int iPhi = g.getPhiBin(item.Phi());
        const int iEta = g.getEtaBin(item.Eta());
        const unsigned mask = iEta >= 0 && iEta < nEta ?
            clusterMask[iEta*nPhi + iPhi] : 0;
        assert(mask <= nJets);
        constituents[mask].push_back(inputCollection->ptrAt(candIdx[i]));
    }
}


void FFTJetProducer::determineVectorConstituents()
{
    const unsigned nJets = recoJets.size();
    const unsigned* clusterMask = recoAlg->getClusterMask();
    const unsigned maskLength = recoAlg->getLastNData();
    assert(maskLength == eventData.size());

    const unsigned* candIdx = maskLength ? &candidateIndex[0] : 0;
    for (unsigned i=0; i<maskLength; ++i)
    {
        // In FFTJet, the mask value of 0 corresponds to unclustered
        // energy. We will do the same here. Jet numbers are therefore
        // shifted by 1 wrt constituents vector, and constituents[1]
        // corresponds to jet number 0.
        const unsigned mask = clusterMask[i];
        assert(mask <= nJets);
        constituents[mask].push_back(inputCollection->ptrAt(candIdx[i]));
    }
}


// The following code more or less coincides with the similar method
// implemented in VirtualJetProducer
template <typename T>
void FFTJetProducer::writeJets(edm::Event& iEvent,
                               const edm::EventSetup& iSetup)
{
    using namespace reco;

    typedef FFTAnyJet<T> OutputJet;
    typedef std::vector<OutputJet> OutputCollection;

    // Area of a single eta-phi cell for jet area calculations.
    // Set it to 0 in case the module configuration does not allow
    // us to calculate jet areas reliably.
    const double cellArea = useGriddedAlgorithm && 
                            recombinationDataCutoff < 0.0 ?
        energyFlow->etaBinWidth() * energyFlow->phiBinWidth() : 0.0;

    // allocate output jet collection
    std::auto_ptr<OutputCollection> jets(new OutputCollection());
    const unsigned nJets = recoJets.size();
    jets->reserve(nJets);

    for (unsigned ijet=0; ijet<nJets; ++ijet)
    {
        const RecoFFTJet& myjet(recoJets[ijet]);

        // Check if we should resum jet constituents
        VectorLike jet4vec(myjet.vec());
        if (resumConstituents)
        {
            VectorLike sum(0.0, 0.0, 0.0, 0.0);
            const unsigned nCon = constituents[ijet+1].size();
            const reco::CandidatePtr* cn = nCon ? &constituents[ijet+1][0] : 0;
            for (unsigned i=0; i<nCon; ++i)
                sum += cn[i]->p4();
            jet4vec = sum;
        }

        // Write the specifics to the jet (simultaneously sets 4-vector,
        // vertex, constituents). These are overridden functions that will
        // call the appropriate specific code.
        T jet;
        writeSpecific(jet, jet4vec, vertexUsed(),
                      constituents[ijet+1], iSetup);

        // calcuate the jet area
        jet.setJetArea(cellArea*myjet.ncells());

        // add jet to the list
        jets->push_back(OutputJet(jet, jetToStorable<float>(myjet)));
    }

    // put the collection into the event
    iEvent.put(jets, outputLabel);
}


void FFTJetProducer::saveResults(edm::Event& ev, const edm::EventSetup& iSetup)
{
    // Write recombined jets
    jet_type_switch(writeJets, ev, iSetup);

    // Check if we should resum unclustered energy constituents
    VectorLike unclusE(unclustered);
    if (resumConstituents)
    {
        VectorLike sum(0.0, 0.0, 0.0, 0.0);
        const unsigned nCon = constituents[0].size();
        const reco::CandidatePtr* cn = nCon ? &constituents[0][0] : 0;
        for (unsigned i=0; i<nCon; ++i)
            sum += cn[i]->p4();
        unclusE = sum;
    }

    // Write the jet reconstruction summary
    const double minScale = minLevel ? sparseTree.getScale(minLevel) : 0.0;
    const double maxScale = maxLevel ? sparseTree.getScale(maxLevel) : 0.0;
    const double scaleUsed = usedLevel ? sparseTree.getScale(usedLevel) : 0.0;

    std::auto_ptr<reco::FFTJetProducerSummary> summary(
        new reco::FFTJetProducerSummary(
            thresholds, occupancy, unclusE,
            constituents[0], unused,
            minScale, maxScale, scaleUsed,
            preclusters.size(), iterationsPerformed,
            iterationsPerformed == 1U ||
            iterationsPerformed <= maxIterations));
    ev.put(summary, outputLabel);
}


// ------------ method called to for each event  ------------
void FFTJetProducer::produce(edm::Event& iEvent,
                             const edm::EventSetup& iSetup)
{
    // Load the clustering tree made by FFTJetPatRecoProducer
    if (storeInSinglePrecision())
        loadSparseTreeData<float>(iEvent);
    else
        loadSparseTreeData<double>(iEvent);

    // Do we need to load the candidate collection?
    if (assignConstituents || !(useGriddedAlgorithm && reuseExistingGrid))
        loadInputCollection(iEvent);

    // Do we need to have discretized energy flow?
    if (useGriddedAlgorithm)
    {
        if (reuseExistingGrid)
            loadEnergyFlow(iEvent);
        else
            discretizeEnergyFlow();
    }

    // Calculate cluster occupancy as a function of level number
    sparseTree.occupancyInScaleSpace(*peakSelector, &occupancy);

    // Select the preclusters using the requested resolution scheme
    preclusters.clear();
    selectPreclusters(sparseTree, *peakSelector, &preclusters);

    // Prepare to run the jet recombination procedure
    prepareRecombinationScales();

    // Assign membership functions to preclusters. If this function
    // is not overriden in a derived class, default algorithm membership
    // function will be used for every cluster.
    assignMembershipFunctions(&preclusters);

    // Run the recombination algorithm once
    int status = 0;
    if (useGriddedAlgorithm)
        status = gridAlg->run(preclusters, *energyFlow,
                              &noiseLevel, 1U, 1U,
                              &recoJets, &unclustered, &unused);
    else
        status = recoAlg->run(preclusters, eventData, &noiseLevel, 1U,
                              &recoJets, &unclustered, &unused);
    if (status)
        throw cms::Exception("FFTJetInterface")
            << "FFTJet algorithm failed (first iteration)" << std::endl;

    // If requested, iterate the jet recombination procedure
    if (maxIterations > 1U && !recoJets.empty())
        iterationsPerformed = iterateJetReconstruction();
    else
        iterationsPerformed = 1U;

    // Determine jet constituents. FFTJet returns a map
    // of constituents which is inverse to what we need here.
    const unsigned nJets = recoJets.size();
    if (constituents.size() <= nJets)
        constituents.resize(nJets + 1U);
    if (assignConstituents)
    {
        for (unsigned i=0; i<=nJets; ++i)
            constituents[i].clear();
        if (useGriddedAlgorithm)
            determineGriddedConstituents();
        else
            determineVectorConstituents();
    }

    // Write out the results
    saveResults(iEvent, iSetup);
}


std::auto_ptr<fftjet::Functor1<bool,fftjet::Peak> >
FFTJetProducer::parse_peakSelector(const edm::ParameterSet& ps)
{
    return fftjet_PeakSelector_parser(
        ps.getParameter<edm::ParameterSet>("PeakSelectorConfiguration"));
}


// Parser for the jet membership function
std::auto_ptr<fftjet::ScaleSpaceKernel>
FFTJetProducer::parse_jetMembershipFunction(const edm::ParameterSet& ps)
{
    return fftjet_MembershipFunction_parser(
        ps.getParameter<edm::ParameterSet>("jetMembershipFunction"));
}


// Parser for the background membership function
std::auto_ptr<AbsBgFunctor>
FFTJetProducer::parse_bgMembershipFunction(const edm::ParameterSet& ps)
{
    return fftjet_BgFunctor_parser(
        ps.getParameter<edm::ParameterSet>("bgMembershipFunction"));
}


// Calculator for the recombination scale
std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
FFTJetProducer::parse_recoScaleCalcPeak(const edm::ParameterSet& ps)
{
    return fftjet_PeakFunctor_parser(
        ps.getParameter<edm::ParameterSet>("recoScaleCalcPeak"));
}


// Calculator for the recombination scale ratio
std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
FFTJetProducer::parse_recoScaleRatioCalcPeak(const edm::ParameterSet& ps)
{
    return fftjet_PeakFunctor_parser(
        ps.getParameter<edm::ParameterSet>("recoScaleRatioCalcPeak"));
}


// Calculator for the membership function factor
std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
FFTJetProducer::parse_memberFactorCalcPeak(const edm::ParameterSet& ps)
{
    return fftjet_PeakFunctor_parser(
        ps.getParameter<edm::ParameterSet>("memberFactorCalcPeak"));
}


std::auto_ptr<fftjet::Functor1<double,FFTJetProducer::RecoFFTJet> >
FFTJetProducer::parse_recoScaleCalcJet(const edm::ParameterSet& ps)
{
    return fftjet_JetFunctor_parser(
        ps.getParameter<edm::ParameterSet>("recoScaleCalcJet"));
}


std::auto_ptr<fftjet::Functor1<double,FFTJetProducer::RecoFFTJet> >
FFTJetProducer::parse_recoScaleRatioCalcJet(const edm::ParameterSet& ps)
{
    return fftjet_JetFunctor_parser(
        ps.getParameter<edm::ParameterSet>("recoScaleRatioCalcJet"));
}


std::auto_ptr<fftjet::Functor1<double,FFTJetProducer::RecoFFTJet> >
FFTJetProducer::parse_memberFactorCalcJet(const edm::ParameterSet& ps)
{
    return fftjet_JetFunctor_parser(
        ps.getParameter<edm::ParameterSet>("memberFactorCalcJet"));
}


std::auto_ptr<fftjet::Functor2<
    double,FFTJetProducer::RecoFFTJet,FFTJetProducer::RecoFFTJet> >
FFTJetProducer::parse_jetDistanceCalc(const edm::ParameterSet& ps)
{
    return fftjet_JetDistance_parser(
        ps.getParameter<edm::ParameterSet>("jetDistanceCalc"));
}


void FFTJetProducer::assignMembershipFunctions(std::vector<fftjet::Peak>*)
{
}


// ------------ method called once each job just before starting event loop
void FFTJetProducer::beginJob()
{
    const edm::ParameterSet& ps(myConfiguration);

    // Parse the peak selector definition
    peakSelector = parse_peakSelector(ps);
    checkConfig(peakSelector, "invalid peak selector");

    jetMembershipFunction = parse_jetMembershipFunction(ps);
    checkConfig(jetMembershipFunction, "invalid jet membership function");

    bgMembershipFunction = parse_bgMembershipFunction(ps);
    checkConfig(bgMembershipFunction, "invalid noise membership function");

    // Build the energy recombination algorithm
    if (!useGriddedAlgorithm)
    {
        fftjet::DefaultVectorRecombinationAlgFactory<
            VectorLike,BgData,VBuilder> factory;
        if (factory[recombinationAlgorithm] == NULL)
            throw cms::Exception("FFTJetBadConfig")
                << "Invalid vector recombination algorithm \""
                << recombinationAlgorithm << "\"" << std::endl;
        recoAlg = std::auto_ptr<RecoAlg>(
            factory[recombinationAlgorithm]->create(
                jetMembershipFunction.get(),
                &VectorLike::Et, &VectorLike::Eta, &VectorLike::Phi,
                bgMembershipFunction.get(),
                unlikelyBgWeight, isCrisp, false, assignConstituents));
    }
    else if (!reuseExistingGrid)
    {
        energyFlow = fftjet_Grid2d_parser(
            ps.getParameter<edm::ParameterSet>("GridConfiguration"));
        checkConfig(energyFlow, "invalid discretization grid");
        buildGridAlg();
    }

    // Parse the calculator of the recombination scale
    recoScaleCalcPeak = parse_recoScaleCalcPeak(ps);
    checkConfig(recoScaleCalcPeak, "invalid spec for the "
                "reconstruction scale calculator from peaks");

    // Parse the calculator of the recombination scale ratio
    recoScaleRatioCalcPeak = parse_recoScaleRatioCalcPeak(ps);
    checkConfig(recoScaleRatioCalcPeak, "invalid spec for the "
                "reconstruction scale ratio calculator from peaks");

    // Calculator for the membership function factor
    memberFactorCalcPeak = parse_memberFactorCalcPeak(ps);
    checkConfig(memberFactorCalcPeak, "invalid spec for the "
                "membership function factor calculator from peaks");

    if (maxIterations > 1)
    {
        // We are going to run iteratively. Make required objects.
        recoScaleCalcJet = parse_recoScaleCalcJet(ps);
        checkConfig(recoScaleCalcJet, "invalid spec for the "
                    "reconstruction scale calculator from jets");

        recoScaleRatioCalcJet = parse_recoScaleRatioCalcJet(ps);
        checkConfig(recoScaleRatioCalcJet, "invalid spec for the "
                    "reconstruction scale ratio calculator from jets");

        memberFactorCalcJet = parse_memberFactorCalcJet(ps);
        checkConfig(memberFactorCalcJet, "invalid spec for the "
                    "membership function factor calculator from jets");

        jetDistanceCalc = parse_jetDistanceCalc(ps);
        checkConfig(memberFactorCalcJet, "invalid spec for the "
                    "jet distance calculator");
    }
}


// ------------ method called once each job just after ending the event loop
void FFTJetProducer::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetProducer);
