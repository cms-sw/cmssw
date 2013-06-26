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
// $Id: FFTJetProducer.cc,v 1.16 2012/11/21 03:13:26 igv Exp $
//
//

#include <iostream>
#include <fstream>
#include <functional>
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
#include "RecoJets/FFTJetAlgorithms/interface/matchOneToOne.h"
#include "RecoJets/FFTJetAlgorithms/interface/JetToPeakDistance.h"
#include "RecoJets/FFTJetAlgorithms/interface/adjustForPileup.h"

#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

// Loader for the lookup tables
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequenceLoader.h"


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

namespace {
    struct LocalSortByPt
    {
        template<class Jet>
        inline bool operator()(const Jet& l, const Jet& r) const
            {return l.pt() > r.pt();}
    };
}

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
    else if (!name.compare("fromGenJets"))
        return FROM_GENJETS;
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
      init_param(bool, calculatePileup),
      init_param(bool, subtractPileup),
      init_param(bool, subtractPileupAs4Vec),
      init_param(edm::InputTag, pileupLabel),
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
      init_param(edm::InputTag, genJetsLabel),
      init_param(unsigned, maxInitialPreclusters),
      resolution(parse_resolution(ps.getParameter<std::string>("resolution"))),
      init_param(std::string, pileupTableRecord),
      init_param(std::string, pileupTableName),
      init_param(std::string, pileupTableCategory),
      init_param(bool, loadPileupFromDB),

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

    sparsePeakTreeFromStorable(*input, iniScales.get(),
                               getEventScale(), &sparseTree);
    sparseTree.sortNodes();
}


void FFTJetProducer::genJetPreclusters(
    const SparseTree& /* tree */,
    edm::Event& iEvent, const edm::EventSetup& /* iSetup */,
    const fftjet::Functor1<bool,fftjet::Peak>& peakSelect,
    std::vector<fftjet::Peak>* preclusters)
{
    typedef reco::FFTAnyJet<reco::GenJet> InputJet;
    typedef std::vector<InputJet> InputCollection;

    edm::Handle<InputCollection> input;
    iEvent.getByLabel(genJetsLabel, input);

    const unsigned sz = input->size();
    preclusters->reserve(sz);
    for (unsigned i=0; i<sz; ++i)
    {
        const RecoFFTJet& jet(jetFromStorable((*input)[i].getFFTSpecific()));
        fftjet::Peak p(jet.precluster());
        const double scale(p.scale());
        p.setEtaPhi(jet.vec().Eta(), jet.vec().Phi());
        p.setMagnitude(jet.vec().Pt()/scale/scale);
        p.setStatus(resolution);
        if (peakSelect(p))
            preclusters->push_back(p);
    }
}


void FFTJetProducer::selectPreclusters(
    const SparseTree& tree,
    const fftjet::Functor1<bool,fftjet::Peak>& peakSelect,
    std::vector<fftjet::Peak>* preclusters)
{
    nodes.clear();
    selectTreeNodes(tree, peakSelect, &nodes);

    // Fill out the vector of preclusters using the tree node ids
    const unsigned nNodes = nodes.size();
    const SparseTree::NodeId* pnodes = nNodes ? &nodes[0] : 0;
    preclusters->reserve(nNodes);
    for (unsigned i=0; i<nNodes; ++i)
        preclusters->push_back(
            sparseTree.uncheckedNode(pnodes[i]).getCluster());

    // Remember the node id in the precluster and set
    // the status word to indicate the resolution scheme used
    fftjet::Peak* clusters = nNodes ? &(*preclusters)[0] : 0;
    for (unsigned i=0; i<nNodes; ++i)
    {
        clusters[i].setCode(pnodes[i]);
        clusters[i].setStatus(resolution);
    }
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


bool FFTJetProducer::loadEnergyFlow(
    const edm::Event& iEvent, const edm::InputTag& label,
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& flow)
{
    edm::Handle<reco::DiscretizedEnergyFlow> input;
    iEvent.getByLabel(label, input);

    // Make sure that the grid is compatible with the stored one
    bool rebuildGrid = flow.get() == NULL;
    if (!rebuildGrid)
        rebuildGrid = 
            !(flow->nEta() == input->nEtaBins() &&
              flow->nPhi() == input->nPhiBins() &&
              flow->etaMin() == input->etaMin() &&
              flow->etaMax() == input->etaMax() &&
              flow->phiBin0Edge() == input->phiBin0Edge());
    if (rebuildGrid)
    {
        // We should not get here very often...
        flow = std::auto_ptr<fftjet::Grid2d<Real> >(
            new fftjet::Grid2d<Real>(
                input->nEtaBins(), input->etaMin(), input->etaMax(),
                input->nPhiBins(), input->phiBin0Edge(), input->title()));
    }
    flow->blockSet(input->data(), input->nEtaBins(), input->nPhiBins());
    return rebuildGrid;
}


bool FFTJetProducer::checkConvergence(const std::vector<RecoFFTJet>& previous,
                                      std::vector<RecoFFTJet>& nextSet)
{
    fftjet::Functor2<double,RecoFFTJet,RecoFFTJet>&
        distanceCalc(*jetDistanceCalc);

    const unsigned nJets = previous.size();
    if (nJets != nextSet.size())
        return false;

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

    unsigned nJets = recoJets.size();
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

        // As it turns out, it is possible, in very rare cases,
        // to have iterJets.size() != nJets at this point

        // Figure out if the iterations have converged
        converged = checkConvergence(recoJets, iterJets);

        // Prepare for the next cycle
        iterJets.swap(recoJets);
	nJets = recoJets.size();
    }

    // Check that we have the correct number of preclusters
    if (preclusters.size() != nJets)
    {
	assert(nJets < preclusters.size());
	removeFakePreclusters();
        assert(preclusters.size() == nJets);
    }

    // Plug in the original precluster coordinates into the result
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
    double cellArea = useGriddedAlgorithm && 
                      recombinationDataCutoff < 0.0 ?
        energyFlow->etaBinWidth() * energyFlow->phiBinWidth() : 0.0;

    if (calculatePileup)
        cellArea = pileupEnergyFlow->etaBinWidth() *
                   pileupEnergyFlow->phiBinWidth();

    // allocate output jet collection
    std::auto_ptr<OutputCollection> jets(new OutputCollection());
    const unsigned nJets = recoJets.size();
    jets->reserve(nJets);

    bool sorted = true;
    double previousPt = DBL_MAX;
    for (unsigned ijet=0; ijet<nJets; ++ijet)
    {
        RecoFFTJet& myjet(recoJets[ijet]);

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
            setJetStatusBit(&myjet, CONSTITUENTS_RESUMMED, true);
        }

        // Subtract the pile-up
        if (calculatePileup && subtractPileup)
        {
            jet4vec = adjustForPileup(jet4vec, pileup[ijet], 
                                      subtractPileupAs4Vec);
            if (subtractPileupAs4Vec)
                setJetStatusBit(&myjet, PILEUP_SUBTRACTED_4VEC, true);
            else
                setJetStatusBit(&myjet, PILEUP_SUBTRACTED_PT, true);
        }

        // Write the specifics to the jet (simultaneously sets 4-vector,
        // vertex, constituents). These are overridden functions that will
        // call the appropriate specific code.
        T jet;
        writeSpecific(jet, jet4vec, vertexUsed(),
                      constituents[ijet+1], iSetup);

        // calcuate the jet area
        double ncells = myjet.ncells();
        if (calculatePileup)
        {
            ncells = cellCountsVec[ijet];
            setJetStatusBit(&myjet, PILEUP_CALCULATED, true);
        }
        jet.setJetArea(cellArea*ncells);

        // add jet to the list
        FFTJet<float> fj(jetToStorable<float>(myjet));
        fj.setFourVec(jet4vec);
        if (calculatePileup)
        {
            fj.setPileup(pileup[ijet]);
            fj.setNCells(ncells);
        }
        jets->push_back(OutputJet(jet, fj));

        // Check whether the sequence remains sorted by pt
        const double pt = jet.pt();
        if (pt > previousPt)
            sorted = false;
        previousPt = pt;
    }

    // Sort the collection
    if (!sorted)
        std::sort(jets->begin(), jets->end(), LocalSortByPt());

    // put the collection into the event
    iEvent.put(jets, outputLabel);
}


void FFTJetProducer::saveResults(edm::Event& ev, const edm::EventSetup& iSetup,
                                 const unsigned nPreclustersFound)
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
            nPreclustersFound, iterationsPerformed,
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
        {
            if (loadEnergyFlow(iEvent, treeLabel, energyFlow))
                buildGridAlg();
        }
        else
            discretizeEnergyFlow();
    }

    // Calculate cluster occupancy as a function of level number
    sparseTree.occupancyInScaleSpace(*peakSelector, &occupancy);

    // Select the preclusters using the requested resolution scheme
    preclusters.clear();
    if (resolution == FROM_GENJETS)
        genJetPreclusters(sparseTree, iEvent, iSetup,
                          *peakSelector, &preclusters);
    else
        selectPreclusters(sparseTree, *peakSelector, &preclusters);
    if (preclusters.size() > maxInitialPreclusters)
    {
        std::sort(preclusters.begin(), preclusters.end(), std::greater<fftjet::Peak>());
        preclusters.erase(preclusters.begin()+maxInitialPreclusters, preclusters.end());
    }

    // Prepare to run the jet recombination procedure
    prepareRecombinationScales();

    // Assign membership functions to preclusters. If this function
    // is not overriden in a derived class, default algorithm membership
    // function will be used for every cluster.
    assignMembershipFunctions(&preclusters);

    // Count the preclusters going in
    unsigned nPreclustersFound = 0U;
    const unsigned npre = preclusters.size();
    for (unsigned i=0; i<npre; ++i)
        if (preclusters[i].membershipFactor() > 0.0)
            ++nPreclustersFound;

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
    {
        // It is possible to have a smaller number of jets than we had
        // preclusters. Fake preclusters are possible, but for a good
        // choice of pattern recognition kernel their presence should
        // be infrequent. However, any fake preclusters will throw the
        // iterative reconstruction off balance. Deal with the problem now.
        const unsigned nJets = recoJets.size();
        if (preclusters.size() != nJets)
        {
            assert(nJets < preclusters.size());
            removeFakePreclusters();
        }
        iterationsPerformed = iterateJetReconstruction();
    }
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

    // Figure out the pile-up
    if (calculatePileup)
    {
        if (loadPileupFromDB)
            determinePileupDensityFromDB(iEvent, iSetup,
                                         pileupLabel, pileupEnergyFlow);
        else
            determinePileupDensityFromConfig(iEvent, pileupLabel,
                                             pileupEnergyFlow);
        determinePileup();
        assert(pileup.size() == recoJets.size());
    }

    // Write out the results
    saveResults(iEvent, iSetup, nPreclustersFound);
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


// Pile-up density calculator
std::auto_ptr<fftjetcms::AbsPileupCalculator>
FFTJetProducer::parse_pileupDensityCalc(const edm::ParameterSet& ps)
{
    return fftjet_PileupCalculator_parser(
        ps.getParameter<edm::ParameterSet>("pileupDensityCalc"));
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

    // Create the grid subsequently used for pile-up subtraction
    if (calculatePileup)
    {
        pileupEnergyFlow = fftjet_Grid2d_parser(
            ps.getParameter<edm::ParameterSet>("PileupGridConfiguration"));
        checkConfig(pileupEnergyFlow, "invalid pileup density grid");

        if (!loadPileupFromDB)
        {
            pileupDensityCalc = parse_pileupDensityCalc(ps);
            checkConfig(pileupDensityCalc, "invalid pile-up density calculator");
        }
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


void FFTJetProducer::removeFakePreclusters()
{
    // There are two possible reasons for fake preclusters:
    // 1. Membership factor was set to 0
    // 2. Genuine problem with pattern recognition
    //
    // Anyway, we need to match jets to preclusters and keep
    // only those preclusters that have been matched
    //
    std::vector<int> matchTable;
    const unsigned nmatched = matchOneToOne(
        recoJets, preclusters, JetToPeakDistance(), &matchTable);

    // Ensure that all jets have been matched.
    // If not, we must have a bug somewhere.
    assert(nmatched == recoJets.size());

    // Collect all matched preclusters
    iterPreclusters.clear();
    iterPreclusters.reserve(nmatched);
    for (unsigned i=0; i<nmatched; ++i)
        iterPreclusters.push_back(preclusters[matchTable[i]]);
    iterPreclusters.swap(preclusters);
}


void FFTJetProducer::setJetStatusBit(RecoFFTJet* jet,
                                     const int mask, const bool value)
{
    int status = jet->status();
    if (value)
        status |= mask;
    else
        status &= ~mask;
    jet->setStatus(status);
}


void FFTJetProducer::determinePileupDensityFromConfig(
    const edm::Event& iEvent, const edm::InputTag& label,
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& density)
{
    edm::Handle<reco::FFTJetPileupSummary> summary;
    iEvent.getByLabel(label, summary);

    const reco::FFTJetPileupSummary& s(*summary);
    const AbsPileupCalculator& calc(*pileupDensityCalc);
    const bool phiDependent = calc.isPhiDependent();

    fftjet::Grid2d<Real>& g(*density);
    const unsigned nEta = g.nEta();
    const unsigned nPhi = g.nPhi();

    for (unsigned ieta=0; ieta<nEta; ++ieta)
    {
        const double eta(g.etaBinCenter(ieta));

        if (phiDependent)
        {
            for (unsigned iphi=0; iphi<nPhi; ++iphi)
            {
                const double phi(g.phiBinCenter(iphi));
                g.uncheckedSetBin(ieta, iphi, calc(eta, phi, s));
            }
        }
        else
        {
            const double pil = calc(eta, 0.0, s);
            for (unsigned iphi=0; iphi<nPhi; ++iphi)
                g.uncheckedSetBin(ieta, iphi, pil);
        }
    }
}


void FFTJetProducer::determinePileupDensityFromDB(
    const edm::Event& iEvent, const edm::EventSetup& iSetup,
    const edm::InputTag& label,
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& density)
{
    edm::ESHandle<FFTJetLookupTableSequence> h;
    StaticFFTJetLookupTableSequenceLoader::instance().load(
        iSetup, pileupTableRecord, h);
    boost::shared_ptr<npstat::StorableMultivariateFunctor> f =
        (*h)[pileupTableCategory][pileupTableName];

    edm::Handle<reco::FFTJetPileupSummary> summary;
    iEvent.getByLabel(label, summary);

    const float rho = summary->pileupRho();
    const bool phiDependent = f->minDim() == 3U;

    fftjet::Grid2d<Real>& g(*density);
    const unsigned nEta = g.nEta();
    const unsigned nPhi = g.nPhi();

    double functorArg[3] = {0.0, 0.0, 0.0};
    if (phiDependent)
        functorArg[2] = rho;
    else
        functorArg[1] = rho;

    for (unsigned ieta=0; ieta<nEta; ++ieta)
    {
        const double eta(g.etaBinCenter(ieta));
        functorArg[0] = eta;

        if (phiDependent)
        {
            for (unsigned iphi=0; iphi<nPhi; ++iphi)
            {
                functorArg[1] = g.phiBinCenter(iphi);
                g.uncheckedSetBin(ieta, iphi, (*f)(functorArg, 3U));
            }
        }
        else
        {
            const double pil = (*f)(functorArg, 2U);
            for (unsigned iphi=0; iphi<nPhi; ++iphi)
                g.uncheckedSetBin(ieta, iphi, pil);
        }
    }
}


void FFTJetProducer::determinePileup()
{
    // This function works with crisp clustering only
    if (!isCrisp)
        assert(!"Pile-up subtraction for fuzzy clustering "
               "is not implemented yet");

    // Clear the pileup vector
    const unsigned nJets = recoJets.size();
    pileup.resize(nJets);
    if (nJets == 0)
        return;
    const VectorLike zero;
    for (unsigned i=0; i<nJets; ++i)
        pileup[i] = zero;

    // Pileup energy flow grid
    const fftjet::Grid2d<Real>& g(*pileupEnergyFlow);
    const unsigned nEta = g.nEta();
    const unsigned nPhi = g.nPhi();
    const double cellArea = g.etaBinWidth() * g.phiBinWidth();

    // Various calculators
    fftjet::Functor1<double,RecoFFTJet>& scaleCalc(*recoScaleCalcJet);
    fftjet::Functor1<double,RecoFFTJet>& ratioCalc(*recoScaleRatioCalcJet);
    fftjet::Functor1<double,RecoFFTJet>& factorCalc(*memberFactorCalcJet);

    // Make sure we have enough memory
    memFcns2dVec.resize(nJets);
    fftjet::AbsKernel2d** memFcns2d = &memFcns2dVec[0];

    doubleBuf.resize(nJets*4U + nJets*nPhi);
    double* recoScales = &doubleBuf[0];
    double* recoScaleRatios = recoScales + nJets;
    double* memFactors = recoScaleRatios + nJets;
    double* dEta = memFactors + nJets;
    double* dPhiBuf = dEta + nJets;

    cellCountsVec.resize(nJets);
    unsigned* cellCounts = &cellCountsVec[0];

    // Go over jets and collect the necessary info
    for (unsigned ijet=0; ijet<nJets; ++ijet)
    {
        const RecoFFTJet& jet(recoJets[ijet]);
        const fftjet::Peak& peak(jet.precluster());

        // Make sure we are using 2-d membership functions.
        // Pile-up subtraction scheme for 3-d functions should be different.
        fftjet::AbsMembershipFunction* m3d = 
            dynamic_cast<fftjet::AbsMembershipFunction*>(
                peak.membershipFunction());
        if (m3d == 0)
            m3d = dynamic_cast<fftjet::AbsMembershipFunction*>(
                jetMembershipFunction.get());
        if (m3d)
        {
            assert(!"Pile-up subtraction for 3-d membership functions "
                   "is not implemented yet");
        }
        else
        {
            fftjet::AbsKernel2d* m2d =
                dynamic_cast<fftjet::AbsKernel2d*>(
                    peak.membershipFunction());
            if (m2d == 0)
                m2d = dynamic_cast<fftjet::AbsKernel2d*>(
                    jetMembershipFunction.get());
            assert(m2d);
            memFcns2d[ijet] = m2d;
        }
        recoScales[ijet] = scaleCalc(jet);
        recoScaleRatios[ijet] = ratioCalc(jet);
        memFactors[ijet] = factorCalc(jet);
        cellCounts[ijet] = 0U;

        const double jetPhi = jet.vec().Phi();
        for (unsigned iphi=0; iphi<nPhi; ++iphi)
        {
            double dphi = g.phiBinCenter(iphi) - jetPhi;
            while (dphi > M_PI)
                dphi -= (2.0*M_PI);
            while (dphi < -M_PI)
                dphi += (2.0*M_PI);
            dPhiBuf[iphi*nJets+ijet] = dphi;
        }
    }

    // Go over all grid points and integrate
    // the pile-up energy density
    VBuilder vMaker;
    for (unsigned ieta=0; ieta<nEta; ++ieta)
    {
        const double eta(g.etaBinCenter(ieta));
        const Real* databuf = g.data() + ieta*nPhi;

        // Figure out dEta for each jet
        for (unsigned ijet=0; ijet<nJets; ++ijet)
            dEta[ijet] = eta - recoJets[ijet].vec().Eta();

        for (unsigned iphi=0; iphi<nPhi; ++iphi)
        {
            double maxW(0.0);
            unsigned maxWJet(nJets);
            const double* dPhi = dPhiBuf + iphi*nJets;

            for (unsigned ijet=0; ijet<nJets; ++ijet)
            {
                if (recoScaleRatios[ijet] > 0.0)
                    memFcns2d[ijet]->setScaleRatio(recoScaleRatios[ijet]);
                const double f = memFactors[ijet]*
                    (*memFcns2d[ijet])(dEta[ijet], dPhi[ijet],
                                       recoScales[ijet]);
                if (f > maxW)
                {
                    maxW = f;
                    maxWJet = ijet;
                }
            }

            if (maxWJet < nJets)
            {
                pileup[maxWJet] += vMaker(cellArea*databuf[iphi],
                                          eta, g.phiBinCenter(iphi));
                cellCounts[maxWJet]++;
            }
        }
    }
}


// ------------ method called once each job just after ending the event loop
void FFTJetProducer::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetProducer);
