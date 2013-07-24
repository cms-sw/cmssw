// -*- C++ -*-
//
// Package:    RecoJets/FFTJetProducers
// Class:      FFTJetProducer
// 
/**\class FFTJetProducer FFTJetProducer.h RecoJets/FFTJetProducers/plugins/FFTJetProducer.h

 Description: makes jets using FFTJet clustering tree

 Implementation:
     If you want to change the jet algorithm functionality (for example,
     by providing your own jet membership function), derive from this
     class and override the appropriate parser method (for example,
     parse_jetMembershipFunction). At the end of your own parser, don't
     forget to call the parser of the base class in order to get the default
     behavior when your special configuration is not provided (this is known
     as the "chain-of-responsibility" design pattern). If you also need to
     override "beginJob" and/or "produce" methods, the first thing to do in
     your method is to call the corresponding method of this base.
*/
//
// Original Author:  Igor Volobouev
//         Created:  Sun Jun 20 14:32:36 CDT 2010
// $Id: FFTJetProducer.h,v 1.12 2012/11/21 03:13:26 igv Exp $
//
//

#ifndef RecoJets_FFTJetProducers_FFTJetProducer_h
#define RecoJets_FFTJetProducers_FFTJetProducer_h

#include <memory>

// FFTJet headers
#include "fftjet/AbsRecombinationAlg.hh"
#include "fftjet/AbsVectorRecombinationAlg.hh"
#include "fftjet/SparseClusteringTree.hh"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

// local FFTJet-related definitions
#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"
#include "RecoJets/FFTJetAlgorithms/interface/AbsPileupCalculator.h"
#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

namespace fftjetcms {
    class DiscretizedEnergyFlow;
}

//
// class declaration
//
class FFTJetProducer : public edm::EDProducer, 
                       public fftjetcms::FFTJetInterface
{
public:
    typedef fftjet::RecombinedJet<fftjetcms::VectorLike> RecoFFTJet;
    typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;

    // Masks for the status bits. Do not add anything
    // here -- higher bits (starting with 0x1000) will be
    // used to indicate jet correction levels applied.
    enum StatusBits
    {
        RESOLUTION = 0xff,
        CONSTITUENTS_RESUMMED = 0x100,
        PILEUP_CALCULATED = 0x200,
        PILEUP_SUBTRACTED_4VEC = 0x400,
        PILEUP_SUBTRACTED_PT = 0x800
    };

    enum Resolution
    {
        FIXED = 0,
        MAXIMALLY_STABLE,
        GLOBALLY_ADAPTIVE,
        LOCALLY_ADAPTIVE,
        FROM_GENJETS
    };

    explicit FFTJetProducer(const edm::ParameterSet&);
    virtual ~FFTJetProducer();

    // Parser for the resolution enum
    static Resolution parse_resolution(const std::string& name);

protected:
    // Functions which should be overriden from the base
    virtual void beginJob();
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    // The following functions can be overriden by derived classes 
    // in order to adjust jet reconstruction algorithm behavior.

    // Override the following method in order to implement
    // your own precluster selection strategy
    virtual void selectPreclusters(
        const SparseTree& tree,
        const fftjet::Functor1<bool,fftjet::Peak>& peakSelector,
        std::vector<fftjet::Peak>* preclusters);

    // Precluster maker from GenJets (useful in calibration)
    virtual void genJetPreclusters(
        const SparseTree& tree,
        edm::Event&, const edm::EventSetup&,
        const fftjet::Functor1<bool,fftjet::Peak>& peakSelector,
        std::vector<fftjet::Peak>* preclusters);

    // Override the following method (which by default does not do
    // anything) in order to implement your own process-dependent
    // assignment of membership functions to preclusters. This method
    // will be called once per event, just before the main algorithm.
    virtual void assignMembershipFunctions(
        std::vector<fftjet::Peak>* preclusters);

    // Parser for the peak selector
    virtual std::auto_ptr<fftjet::Functor1<bool,fftjet::Peak> >
    parse_peakSelector(const edm::ParameterSet&);

    // Parser for the default jet membership function
    virtual std::auto_ptr<fftjet::ScaleSpaceKernel>
    parse_jetMembershipFunction(const edm::ParameterSet&);

    // Parser for the background membership function
    virtual std::auto_ptr<fftjetcms::AbsBgFunctor>
    parse_bgMembershipFunction(const edm::ParameterSet&);

    // Calculator for the recombination scale
    virtual std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
    parse_recoScaleCalcPeak(const edm::ParameterSet&);

    // Calculator for the recombination scale ratio
    virtual std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
    parse_recoScaleRatioCalcPeak(const edm::ParameterSet&);

    // Calculator for the membership function factor
    virtual std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
    parse_memberFactorCalcPeak(const edm::ParameterSet&);

    // Similar calculators for the iterative algorithm
    virtual std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> >
    parse_recoScaleCalcJet(const edm::ParameterSet&);

    virtual std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> >
    parse_recoScaleRatioCalcJet(const edm::ParameterSet&);

    virtual std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> >
    parse_memberFactorCalcJet(const edm::ParameterSet&);

    // Calculator of the distance between jets which is used to make
    // the decision about convergence of the iterative algorithm
    virtual std::auto_ptr<fftjet::Functor2<double,RecoFFTJet,RecoFFTJet> >
    parse_jetDistanceCalc(const edm::ParameterSet&);

    // Pile-up density calculator
    virtual std::auto_ptr<fftjetcms::AbsPileupCalculator>
    parse_pileupDensityCalc(const edm::ParameterSet& ps);

    // The following function performs most of the precluster selection
    // work in this module. You might want to reuse it if only a slight
    // modification of the "selectPreclusters" method is desired.
    void selectTreeNodes(const SparseTree& tree,
                         const fftjet::Functor1<bool,fftjet::Peak>& peakSelect,
                         std::vector<SparseTree::NodeId>* nodes);
private:
    typedef fftjet::AbsVectorRecombinationAlg<
        fftjetcms::VectorLike,fftjetcms::BgData> RecoAlg;
    typedef fftjet::AbsRecombinationAlg<
        fftjetcms::Real,fftjetcms::VectorLike,fftjetcms::BgData> GridAlg;

    // Explicitly disable other ways to construct this object
    FFTJetProducer();
    FFTJetProducer(const FFTJetProducer&);
    FFTJetProducer& operator=(const FFTJetProducer&);

    // Useful local utilities
    template<class Real>
    void loadSparseTreeData(const edm::Event&);

    void removeFakePreclusters();

    // The following methods do most of the work.
    // The following function tells us if the grid was rebuilt.
    static bool loadEnergyFlow(
        const edm::Event& iEvent, const edm::InputTag& label,
        std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& flow);
    void buildGridAlg();
    void prepareRecombinationScales();
    bool checkConvergence(const std::vector<RecoFFTJet>& previousIterResult,
                          std::vector<RecoFFTJet>& thisIterResult);
    void determineGriddedConstituents();
    void determineVectorConstituents();
    void saveResults(edm::Event& iEvent, const edm::EventSetup&,
                     unsigned nPreclustersFound);

    template <typename Jet>
    void writeJets(edm::Event& iEvent, const edm::EventSetup&);

    template <typename Jet>
    void makeProduces(const std::string& alias, const std::string& tag);

    // The following function scans the pile-up density
    // and fills the pile-up grid. Can be overriden if
    // necessary.
    virtual void determinePileupDensityFromConfig(
        const edm::Event& iEvent, const edm::InputTag& label,
        std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& density);

    // Similar function for getting pile-up shape from the database
    virtual void determinePileupDensityFromDB(
        const edm::Event& iEvent, const edm::EventSetup& iSetup,
        const edm::InputTag& label,
        std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> >& density);

    // The following function builds the pile-up estimate
    // for each jet
    void determinePileup();

    // The following function returns the number of iterations
    // performed. If this number equals to or less than "maxIterations"
    // then the iterations have converged. If the number larger than
    // "maxIterations" then the iterations failed to converge (note,
    // however, that only "maxIterations" iterations would still be
    // performed).
    unsigned iterateJetReconstruction();

    // A function to set jet status bits
    static void setJetStatusBit(RecoFFTJet* jet, int mask, bool value);

    //
    // ----------member data ---------------------------
    //

    // Local copy of the module configuration 
    const edm::ParameterSet myConfiguration;

    // Label for the tree produced by FFTJetPatRecoProducer
    const edm::InputTag treeLabel;

    // Are we going to use energy flow discretization grid as input
    // to jet reconstruction?
    const bool useGriddedAlgorithm;

    // Are we going to rebuild the energy flow discretization grid
    // or to reuse the grid made by FFTJetPatRecoProducer?
    const bool reuseExistingGrid;

    // Are we iterating?
    const unsigned maxIterations;

    // Parameters which affect iteration convergence
    const unsigned nJetsRequiredToConverge;
    const double convergenceDistance;

    // Are we building assignments of collection members to jets?
    const bool assignConstituents;

    // Are we resumming the constituents to determine jet 4-vectors?
    // This might make sense if FFTJet is used in the crisp, gridded
    // mode to determine jet areas, and vector recombination is desired.
    const bool resumConstituents;

    // Are we going to subtract the pile-up? Note that
    // pile-up subtraction does not modify eta and phi moments.
    const bool calculatePileup;
    const bool subtractPileup;
    const bool subtractPileupAs4Vec;

    // Label for the pile-up energy flow. Must be specified
    // if the pile-up is subtracted.
    const edm::InputTag pileupLabel;

    // Scale for the peak selection (if the scale is fixed)
    const double fixedScale;

    // Minimum and maximum scale for searching stable configurations
    const double minStableScale;
    const double maxStableScale;

    // Stability "alpha"
    const double stabilityAlpha;

    // Not sure at this point how to treat noise... For now, this is
    // just a single configurable number...
    const double noiseLevel;

    // Number of clusters requested (if the scale is adaptive)
    const unsigned nClustersRequested;

    // Maximum eta for the grid-based algorithm
    const double gridScanMaxEta;

    // Parameters related to the recombination algorithm
    const std::string recombinationAlgorithm;
    const bool isCrisp;
    const double unlikelyBgWeight;
    const double recombinationDataCutoff;

    // Label for the genJets used as seeds for jets
    const edm::InputTag genJetsLabel;
    
    // Maximum number of preclusters to use as jet seeds.
    // This does not take into account the preclusters
    // for which the value of the membership factor is 0.
    const unsigned maxInitialPreclusters;

    // Resolution. The corresponding parameter value
    // should be one of "fixed", "maximallyStable",
    // "globallyAdaptive", "locallyAdaptive", or "fromGenJets".
    Resolution resolution;

    // Parameters related to the pileup shape stored
    // in the database
    std::string pileupTableRecord;
    std::string pileupTableName;
    std::string pileupTableCategory;
    bool loadPileupFromDB;

    // Scales used
    std::auto_ptr<std::vector<double> > iniScales;

    // The sparse clustering tree
    SparseTree sparseTree;

    // Peak selector for the peaks already in the tree
    std::auto_ptr<fftjet::Functor1<bool,fftjet::Peak> > peakSelector;

    // Recombination algorithms and related quantities
    std::auto_ptr<RecoAlg> recoAlg;
    std::auto_ptr<GridAlg> gridAlg;
    std::auto_ptr<fftjet::ScaleSpaceKernel> jetMembershipFunction;
    std::auto_ptr<fftjetcms::AbsBgFunctor> bgMembershipFunction;

    // Calculator for the recombination scale
    std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> > recoScaleCalcPeak;

    // Calculator for the recombination scale ratio
    std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
    recoScaleRatioCalcPeak;

    // Calculator for the membership function factor
    std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> > memberFactorCalcPeak;

    // Similar calculators for the iterative algorithm
    std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> > recoScaleCalcJet;
    std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> > recoScaleRatioCalcJet;
    std::auto_ptr<fftjet::Functor1<double,RecoFFTJet> > memberFactorCalcJet;

    // Calculator for the jet distance used to estimate convergence
    // of the iterative algorithm
    std::auto_ptr<fftjet::Functor2<double,RecoFFTJet,RecoFFTJet> >
    jetDistanceCalc;

    // Vector of selected tree nodes
    std::vector<SparseTree::NodeId> nodes;

    // Vector of selected preclusters
    std::vector<fftjet::Peak> preclusters;

    // Vector of reconstructed jets (we will refill it in every event)
    std::vector<RecoFFTJet> recoJets;

    // Cluster occupancy calculated as a function of level number
    std::vector<unsigned> occupancy;

    // The thresholds obtained by the LOCALLY_ADAPTIVE method
    std::vector<double> thresholds;

    // Minimum, maximum and used level calculated by some algorithms
    unsigned minLevel, maxLevel, usedLevel;

    // Unclustered/unused energy produced during recombination
    fftjetcms::VectorLike unclustered;
    double unused;

    // Quantities defined below are used in the iterative mode only
    std::vector<fftjet::Peak> iterPreclusters;
    std::vector<RecoFFTJet> iterJets;
    unsigned iterationsPerformed;

    // Vectors of constituents
    std::vector<std::vector<reco::CandidatePtr> > constituents;

    // Vector of pile-up. We will subtract it from the
    // 4-vectors of reconstructed jets.
    std::vector<fftjetcms::VectorLike> pileup;

    // The pile-up transverse energy density discretization grid.
    // Note that this is _density_, not energy. To get energy, 
    // multiply by cell area.
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> > pileupEnergyFlow;

    // The functor that calculates the pile-up density
    std::auto_ptr<fftjetcms::AbsPileupCalculator> pileupDensityCalc;

    // Memory buffers related to pile-up subtraction
    std::vector<fftjet::AbsKernel2d*> memFcns2dVec;
    std::vector<double> doubleBuf;
    std::vector<unsigned> cellCountsVec;
};

#endif // RecoJets_FFTJetProducers_FFTJetProducer_h
