#ifndef PhysicsTools_TagAndProbe_BaseTreeFiller_h
#define PhysicsTools_TagAndProbe_BaseTreeFiller_h

#include <vector>
#include <string>
#include <stdint.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <TTree.h>
#include <boost/utility.hpp>


/** Class that makes a TTree with:
     - variables from a Probe
     - flags saying if the Probe passed one or more selections
     - MC flag saying if the Probe is to be used in MC efficiency
*/

namespace tnp {

/// A variable for the probe: can be a string expression or an external ValueMap<float>
class ProbeVariable {
    public:
        /// Create a ProbeVariable to be evaluated on the fly from a string expression
        ProbeVariable(const std::string &name, const std::string &expression) :
            name_(name), external_(false), function_(expression) {}

        /// Create a ProbeVariable to be read from a ValueMap
        ProbeVariable(const std::string &name, const edm::EDGetTokenT<edm::ValueMap<float> > &srcToken) :
            name_(name), external_(true), function_("-1"), srcToken_(srcToken) {}

        /// Destructor (does nothing)
        ~ProbeVariable() ;

        /// Addess for ROOT Branch
        float * address() const { return &value_; }

        /// name
        const std::string & name() const { return name_; }

        /// To be called at the beginning of the event (will fetch ValueMap if needed)
        void init(const edm::Event &iEvent) const {
            if (external_) iEvent.getByToken(srcToken_, handle_);
        }

        /// To be called for each item
        void fill(const reco::CandidateBaseRef &probe) const {
            value_ = external_ ? (*handle_)[probe] : function_(*probe);
        }

    private:
        /// the name of the variable, which becomes the ROOT branch name
        std::string name_;
        /// the place where we store the value, and that ROOT uses to fill the tree
        mutable float value_;

        /// true if it's an external ValueMap, false if it's a StringParser function
        bool external_;
        // ---- this below is used if 'external_' is false
        /// a lazy-parsed StringObjectFunction<reco::Candidate> that gets all the methods of daughter classes too
        StringObjectFunction<reco::Candidate,true> function_;
        // In releases older than 3.4.X, it can be parially emulated by PATStringObjectFunction (PhysicsTools/PatUtils/interface/StringParserTools.h)
        // Or you can use StringObjectFunction<reco::Candidate> and get only reco::Candidate methods

        // ---- this below is used if 'external_' is true
        /// the external valuemap
        edm::EDGetTokenT<edm::ValueMap<float> > srcToken_;
        /// the handle to keep the ValueMap
        mutable edm::Handle<edm::ValueMap<float> > handle_;
};

class ProbeFlag {
    public:
        /// Create a ProbeFlag to be evaluated on the fly from a string cut
        ProbeFlag(const std::string &name, const std::string &cut) :
            name_(name), external_(false), cut_(cut) {}

        /// Create a ProbeFlag to be read from a ValueMap
        ProbeFlag(const std::string &name, const edm::EDGetTokenT<edm::View<reco::Candidate> > &srcToken) :
            name_(name), external_(true), cut_(""), srcToken_(srcToken) {}

        /// Destructor (does nothing)
        ~ProbeFlag() ;

        /// Addess for ROOT Branch
        int32_t * address() const { return &value_; }

        /// name
        const std::string & name() const { return name_; }

        /// To be called at the beginning of the event (will fetch Candidate View if needed)
        void init(const edm::Event &iEvent) const ;

        /// To be called for each item
        void fill(const reco::CandidateBaseRef &probe) const ;

    private:
        /// the name of the variable, which becomes the ROOT branch name
        std::string name_;
        /// the place where we store the value, and that ROOT uses to fill the tree
        mutable int32_t value_;

        /// true if it's an external Candidate View, false if it's a StringParser cut
        bool external_;
        // ---- this below is used if 'external_' is false
        /// implementation of the cut using a string selector
        StringCutObjectSelector<reco::Candidate,true> cut_;

        // ---- this below is used if 'external_' is true
        /// the external collection
        edm::EDGetTokenT<edm::View<reco::Candidate> > srcToken_;
        /// the handle to keep the refs to the passing probes
        mutable std::vector<reco::CandidateBaseRef> passingProbes_;
};


// This class inherits from boost::noncopyable, as copying it would break the addresses in the TTree
class BaseTreeFiller : boost::noncopyable {
    public:
        /// specify the name of the TTree, and the configuration for it
        BaseTreeFiller(const char *name, const edm::ParameterSet& config, edm::ConsumesCollector && iC) : BaseTreeFiller(name, config, iC) {};
        BaseTreeFiller(const char *name, const edm::ParameterSet& config, edm::ConsumesCollector & iC);

        /// Add branches to an existing TTree managed by another BaseTreeFiller
        BaseTreeFiller(BaseTreeFiller &main, const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC, const std::string &branchNamePrefix);

        /// Destructor, does nothing but it's out-of-line as we have complex data members
        ~BaseTreeFiller();

        //implementation notice: we declare 'const' the methods which don't change the configuration
        //                       and that can't mess up the addresses in the TTree.

        /// To be called once per event, to load possible external variables
        void init(const edm::Event &iEvent) const ;

        /// To be called once per probe, to fill the values for this probe
        void fill(const reco::CandidateBaseRef &probe) const ;

        /// Write a string dump of this PSet into the TTree header.
        /// see macro in test directory for how to retrieve it from the output root file
        void writeProvenance(const edm::ParameterSet &pset) const ;

	//get the pileup weight informations
	bool storePUweight() const {return storePUweight_;};

    protected:

        std::vector<ProbeVariable> vars_;
        std::vector<ProbeFlag>     flags_;

        /// How event weights are defined: 'None' = no weights, 'Fixed' = one value specified in cfg file, 'External' = read weight from the event (as double)
        enum WeightMode { None, Fixed, External };
        WeightMode weightMode_;
        edm::EDGetTokenT<double> weightSrcToken_;
	edm::EDGetTokenT<double> PUweightSrcToken_;
        edm::EDGetTokenT<reco::VertexCollection> recVtxsToken_;
        edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
        edm::EDGetTokenT<reco::CaloMETCollection> metToken_;
        edm::EDGetTokenT<reco::METCollection> tcmetToken_;
        edm::EDGetTokenT<reco::PFMETCollection> pfmetToken_;

        /// Ignore exceptions when evaluating variables
        bool ignoreExceptions_;

        /// Add branches with run and lumisection number
        bool addRunLumiInfo_;

	/// Store Pileup weight when running over Monte Carlo
	bool storePUweight_;

        /// Add branches with event variables: met, sum ET, .. etc.
	bool addEventVariablesInfo_;

        void addBranches_(TTree *tree, const edm::ParameterSet &iConfig, edm::ConsumesCollector & iC, const std::string &branchNamePrefix="") ;

        //implementation notice: these two are 'mutable' because we will fill them from a 'const' method
        mutable TTree * tree_;
        mutable float weight_;
	mutable float PUweight_;
        mutable uint32_t run_, lumi_, mNPV_;
        mutable uint64_t event_;

        mutable float mPVx_,mPVy_,mPVz_,mBSx_,mBSy_,mBSz_;

        mutable float mMET_,mSumET_,mMETSign_,mtcMET_,mtcSumET_,
	  mtcMETSign_,mpfMET_,mpfSumET_,mpfMETSign_;
};


} //namespace

#endif
