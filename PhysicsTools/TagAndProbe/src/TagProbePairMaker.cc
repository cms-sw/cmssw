#include "PhysicsTools/TagAndProbe/interface/TagProbePairMaker.h"
#include "DataFormats/Candidate/interface/Candidate.h"

tnp::TagProbePairMaker::TagProbePairMaker(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) :
  srcToken_(iC.consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("tagProbePairs"))),
  randGen_(0)
{
  std::string arbitration = iConfig.getParameter<std::string>("arbitration");
  if (arbitration == "None") {
    arbitration_ = None;
  } else if (arbitration == "OneProbe") {
    arbitration_ = OneProbe;
  } else if (arbitration == "OnePair") {
    arbitration_ = OnePair;
  } else if (arbitration == "NonDuplicate") {
    arbitration_ = NonDuplicate;
  } else if (arbitration == "BestMass") {
    arbitration_ = BestMass;
    arbitrationMass_ = iConfig.getParameter<double>("massForArbitration");
  } else if (arbitration == "Random2") {
    arbitration_ = Random2;
    randGen_ = new TRandom2(7777);
  } else throw cms::Exception("Configuration") << "TagProbePairMakerOnTheFly: the only currently "
					       << "allowed values for 'arbitration' are "
					       << "'None', 'OneProbe', 'BestMass', 'Random2'\n";

  if (iConfig.existsAs<bool>("phiCutForTwoLeg")) {
    phiCutForTwoLeg_ = iConfig.getParameter<bool>("phiCutForTwoLeg");
    //std::cout << "Set phiCutForTwoLeg_ to " << phiCutForTwoLeg_ << std::endl;
  } else {
    phiCutForTwoLeg_ = false;
    //std::cout << "Set phiCutForTwoLeg_ to default " << phiCutForTwoLeg_ << std::endl;
  }
}


tnp::TagProbePairs
tnp::TagProbePairMaker::run(const edm::Event &iEvent) const
{
  // declare output
  tnp::TagProbePairs pairs;

  // read from event
  edm::Handle<reco::CandidateView> src;
  iEvent.getByToken(srcToken_, src);

  // convert
  for (reco::CandidateView::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
    const reco::Candidate & mother = *it;
    if (mother.numberOfDaughters() != 2) throw cms::Exception("CorruptData") << "Tag&Probe pair with " << mother.numberOfDaughters() << " daughters\n";
    pairs.push_back(tnp::TagProbePair(mother.daughter(0)->masterClone(), mother.daughter(1)->masterClone(),
				      src->refAt(it - src->begin()), mother.mass()));
  }

  if ((arbitration_ != None) && (pairs.size() > 1)) {
    // might need to clean up
    arbitrate(pairs);
  }

  if (phiCutForTwoLeg_ && pairs.size() > 0) {
    int eventNum = iEvent.id().event();
    std::cout << "Calling phiCutByEventNumber on eventNum=" << eventNum << std::endl;
    phiCutByEventNumber(pairs,eventNum);
  }

  // return
  return pairs;
}

void
tnp::TagProbePairMaker::phiCutByEventNumber(TagProbePairs &pairs, int eventNumber) const
{
  unsigned int currentNum = 0;

  size_t nclean = pairs.size();
  for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
    if (it->tag.isNull()) continue; // skip already invalidated pairs
    if (eventNumber%2) {
      std::cout << "Odd event number " << eventNumber << ", require 0 < phi(tag) < pi... ";
      if (!(it->tag->phi() > 0. && it->tag->phi() < 3.141592654)) {
	std::cout  << "Rejecting pair number " << currentNum++ << " with tag phi " << it->tag->phi();
	nclean--;
	it->tag = reco::CandidateBaseRef(); --nclean;
      } else {
	std::cout  << "Keeping pair number " << currentNum++ << " with tag phi " << it->tag->phi();
      }
    } else {
      std::cout << "Even event number " << eventNumber << ", require -pi < phi(tag) < 0... ";
      //      if (!(it->tag->phi() > 3.141592654 && it->tag->phi() < 2*3.141592654)) {
      if (!(it->tag->phi() > -3.141592654 && it->tag->phi() < 0)) {
	std::cout  << "Rejecting pair number " << currentNum++ << " with tag phi " << it->tag->phi();
        nclean--;
        it->tag = reco::CandidateBaseRef(); --nclean;
      } else {
	std::cout  << "Keeping pair number " << currentNum++ << " with tag phi " << it->tag->phi();
      }
    }
    std::cout << std::endl;
  }

  if (nclean == 0) {
    pairs.clear();
  } else if (nclean < pairs.size()) {
    TagProbePairs cleaned; cleaned.reserve(nclean);
    for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
      if (it->tag.isNonnull()) cleaned.push_back(*it);
    }
    pairs.swap(cleaned);
  }
}

void
tnp::TagProbePairMaker::arbitrate(TagProbePairs &pairs) const
{
  size_t nclean = pairs.size();
  for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
    if (it->tag.isNull()) continue; // skip already invalidated pairs

    bool TTpair=false;
    for (TagProbePairs::iterator it2 = pairs.begin(); it2 != ed; ++it2) {   // first check for Tag-Tag pairs
      if(it!=it2 && it->probe==it2->tag && it->tag==it2->probe){
	//std::cout << "----------> probe is tag!!!!" << std::endl;
	TTpair=true;
      }
    }
    //if(!TTpair) std::cout << "this is not TT!" << std::endl;
    bool invalidateThis = false;
    int numberOfProbes=0;
    for (TagProbePairs::iterator it2 = it + 1; it2 != ed; ++it2) {   // it+1 <= ed, otherwise we don't arrive here
      // arbitrate the case where multiple probes are matched to the same tag
      if ((arbitration_ != NonDuplicate) && (it->tag == it2->tag)) {
	if(TTpair){ // we already have found a TT pair, no need to guess
	  it2->tag = reco::CandidateBaseRef(); --nclean;
	  //std::cout << "remove unnecessary pair! -----------" << std::endl;
	  continue;
	}

	if (arbitration_ == OneProbe) {
	  // invalidate this one
	  it2->tag = reco::CandidateBaseRef(); --nclean;
	  // remember to invalidate also the other one (but after finishing to check for duplicates)
	  invalidateThis = true;
	} else if (arbitration_ == BestMass) {
	  // but the best one in the first  iterator
	  if (fabs(it2->mass-arbitrationMass_) < fabs(it->mass-arbitrationMass_)) {
	    std::swap(*it, *it2);
	  }
	  // and invalidate it2
	  it2->tag = reco::CandidateBaseRef(); --nclean;
	} else if (arbitration_ == Random2) {
	  numberOfProbes++;
	  if (numberOfProbes>1) {
	    //std::cout << "more than 2 probes!" << std::endl;
	    invalidateThis=true;
	    it2->tag = reco::CandidateBaseRef();
	    --nclean;
	  } else{
	    // do a coin toss to decide if we want to swap them
	    if (randGen_->Rndm()>0.5) {
	      std::swap(*it, *it2);
	    }
	    // and invalidate it2
	    it2->tag = reco::CandidateBaseRef();
	    --nclean;
	  }
	}
      }
      // arbitrate the case where the same probe is associated to more then one tag
      if ((arbitration_ == NonDuplicate) && (it->probe == it2->probe)) {
        // invalidate the pair in which the tag has lower pT
        if (it2->tag->pt() > it->tag->pt()) std::swap(*it, *it2);
        it2->tag = reco::CandidateBaseRef(); --nclean;
      }
      // arbitrate the OnePair case: disallow the same pair to enter the lineshape twice
      // this can't be done by reference, unfortunately, so we resort to a simple matching
      if ((arbitration_ == OnePair) &&
          std::abs(it->mass - it2->mass) < 1e-4 &&
          std::abs( it->probe->phi() - it2->tag->phi()) < 1e-5 &&
          std::abs( it->probe->eta() - it2->tag->eta()) < 1e-5 &&
          std::abs( it->probe->pt()  - it2->tag->pt() ) < 1e-5 &&
          std::abs(it2->probe->phi() -  it->tag->phi()) < 1e-5 &&
          std::abs(it2->probe->eta() -  it->tag->eta()) < 1e-5 &&
          std::abs(it2->probe->pt()  -  it->tag->pt() ) < 1e-5) {
          it2->tag = reco::CandidateBaseRef(); --nclean;
      }
    }
    if (invalidateThis) { it->tag = reco::CandidateBaseRef(); --nclean; }
  }

  if (nclean == 0) {
    pairs.clear();
  } else if (nclean < pairs.size()) {
    TagProbePairs cleaned; cleaned.reserve(nclean);
    for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
      if (it->tag.isNonnull()) cleaned.push_back(*it);
    }
    pairs.swap(cleaned);
  }
}
