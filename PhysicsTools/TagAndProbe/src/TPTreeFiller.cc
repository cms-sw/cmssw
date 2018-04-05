#include "PhysicsTools/TagAndProbe/interface/TPTreeFiller.h"

tnp::TPTreeFiller::TPTreeFiller(const edm::ParameterSet& config, edm::ConsumesCollector & iC) :
    tnp::BaseTreeFiller("fitter_tree",config, iC)
{
    // Add extra branch for the mass
    tree_->Branch("mass",   &mass_,   "mass/F");

    // set up MC if needed
    if (config.getParameter<bool>("isMC")) {
        tree_->Branch("mcTrue", &mcTrue_, "mcTrue/I");
	tree_->Branch("mcMass", &mcMass_, "mcMass/F");
    }
   
}

tnp::TPTreeFiller::~TPTreeFiller() {}

void tnp::TPTreeFiller::init(const edm::Event &iEvent) const {
    tnp::BaseTreeFiller::init(iEvent);
}

void tnp::TPTreeFiller::fill(const reco::CandidateBaseRef &probe, double mass, bool mcTrue, float mcMass) const {
    mass_ = mass;
    mcTrue_ = mcTrue;
    mcMass_ = mcMass;
    tnp::BaseTreeFiller::fill(probe);
}
