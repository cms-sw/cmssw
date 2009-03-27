#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/HepMCCandAlgos/interface/PdgEntryReplacer.h"

namespace helper {
  struct SelectCode {
    enum KeepOrDrop { kKeep, kDrop };
    enum FlagDepth { kNone, kFirst, kAll };
    KeepOrDrop keepOrDrop_;
    FlagDepth daughtersDepth_, mothersDepth_;
    bool all_;
  };
}

class GenParticlePruner : public edm::EDProducer {
public:
  GenParticlePruner(const edm::ParameterSet&);
private:
  void produce(edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  edm::InputTag src_;
  int keepOrDropAll_;
  std::vector<std::string> selection_;
  std::vector<std::pair<StringCutObjectSelector<reco::GenParticle>, helper::SelectCode> > select_;
  std::vector<int> flags_;
  std::vector<size_t> indices_;
  void parse(const std::string & selection, helper::SelectCode & code, std::string & cut) const;
  void flagDaughters(const reco::GenParticle &, int); 
  void flagMothers(const reco::GenParticle &, int); 
  void recursiveFlagDaughters(size_t, const reco::GenParticleCollection &, int); 
  void recursiveFlagMothers(size_t, const reco::GenParticleCollection &, int); 
  void addDaughterRefs(reco::GenParticle&, reco::GenParticleRefProd, const reco::GenParticleRefVector&) const;
  void addMotherRefs(reco::GenParticle&, reco::GenParticleRefProd, const reco::GenParticleRefVector&) const;
};

using namespace edm;
using namespace std;
using namespace reco;

const int keep = 1, drop = -1;

void GenParticlePruner::parse(const std::string & selection, ::helper::SelectCode & code, std::string & cut) const {
  using namespace ::helper;
  size_t f =  selection.find_first_not_of(' ');
  size_t n = selection.size();
  string command;
  char c;
  for(; (c = selection[f]) != ' ' && f < n; ++f) {
    command.push_back(c);
  }
  if(command[0] == '+') {
    command.erase(0, 1);
    if(command[0] == '+') {
      command.erase(0, 1);
      code.mothersDepth_ = SelectCode::kAll;
    } else {
      code.mothersDepth_ = SelectCode::kFirst;
    }
  } else 
    code.mothersDepth_ = SelectCode::kNone;

  if(command[command.size() - 1] == '+') {
    command.erase(command.size() - 1);
    if(command[command.size()-1] == '+') {
      command.erase(command.size() - 1);
      code.daughtersDepth_ = SelectCode::kAll;
    } else {
      code.daughtersDepth_ = SelectCode::kFirst;
    }
  } else
    code.daughtersDepth_ = SelectCode::kNone;

  if(command == "keep") code.keepOrDrop_ = SelectCode::kKeep;
  else if(command == "drop") code.keepOrDrop_ = SelectCode::kDrop;
  else {
    throw Exception(errors::Configuration)
      << "invalid selection command: " << command << "\n" << endl;
  }
  for(; f < n; ++f) {
    if(selection[f] != ' ') break;
  }
  cut = string(selection, f);
  if(cut[0] == '*')
    cut = string(cut, 0, cut.find_first_of(' '));
  code.all_ = cut == "*";
}

GenParticlePruner::GenParticlePruner(const ParameterSet& cfg) :
  src_(cfg.getParameter<InputTag>("src")), keepOrDropAll_(drop),
  selection_(cfg.getParameter<vector<string> >("select")) {
  using namespace ::helper;
  produces<GenParticleCollection>();
}

void GenParticlePruner::beginJob(const EventSetup& es) {
  PdgEntryReplacer rep(es);
  for(vector<string>::const_iterator i = selection_.begin(); i != selection_.end(); ++i) {
    string cut;
    ::helper::SelectCode code;
    parse(*i, code, cut);
    if(code.all_) {
      if(i != selection_.begin()) 
	throw Exception(errors::Configuration)
	  << "selections \"keep *\" and \"drop *\" can be used only as first options. Here used in position # " 
	  << (i - selection_.begin()) + 1 << "\n" << endl;
      switch(code.keepOrDrop_) {
          case ::helper::SelectCode::kDrop :
              keepOrDropAll_ = drop; break;
          case ::helper::SelectCode::kKeep :
              keepOrDropAll_ = keep; 
      };
    } else {
      cut = rep.replace(cut);
      select_.push_back(make_pair(StringCutObjectSelector<GenParticle>(cut), code));
    }
  }
}

void GenParticlePruner::flagDaughters(const reco::GenParticle & gen, int keepOrDrop) {
  GenParticleRefVector daughters = gen.daughterRefVector();
  for(GenParticleRefVector::const_iterator i = daughters.begin(); i != daughters.end(); ++i) 
    flags_[i->key()] = keepOrDrop;
}

void GenParticlePruner::flagMothers(const reco::GenParticle & gen, int keepOrDrop) {
  GenParticleRefVector mothers = gen.motherRefVector();
  for(GenParticleRefVector::const_iterator i = mothers.begin(); i != mothers.end(); ++i) 
    flags_[i->key()] = keepOrDrop;
}

void GenParticlePruner::recursiveFlagDaughters(size_t index, const reco::GenParticleCollection & src, int keepOrDrop) {
  GenParticleRefVector daughters = src[index].daughterRefVector();
  for(GenParticleRefVector::const_iterator i = daughters.begin(); i != daughters.end(); ++i) {
    index = i->key();
    flags_[index] = keepOrDrop;
    recursiveFlagDaughters(index, src, keepOrDrop);
  }
}

void GenParticlePruner::recursiveFlagMothers(size_t index, const reco::GenParticleCollection & src, int keepOrDrop) {
  GenParticleRefVector mothers = src[index].motherRefVector();
  for(GenParticleRefVector::const_iterator i = mothers.begin(); i != mothers.end(); ++i) {
    index = i->key();
    flags_[index] = keepOrDrop;
    recursiveFlagMothers(index, src, keepOrDrop);
  }
}

void GenParticlePruner::produce(Event& evt, const EventSetup&) {
  using namespace ::helper;
  Handle<GenParticleCollection> src;
  evt.getByLabel(src_, src);
  const size_t n = src->size();
  flags_.clear();
  flags_.resize(n, keepOrDropAll_);
  for(size_t j = 0; j < select_.size(); ++j) { 
    const pair<StringCutObjectSelector<GenParticle>, SelectCode> & sel = select_[j];
    SelectCode code = sel.second;
    const StringCutObjectSelector<GenParticle> & cut = sel.first;
    for(size_t i = 0; i < n; ++i) {
      const GenParticle & p = (*src)[i];
      if(cut(p)) {
	int keepOrDrop = keep;
	switch(code.keepOrDrop_) {
	case SelectCode::kKeep:
	  keepOrDrop = keep; break;
	case SelectCode::kDrop:
	  keepOrDrop = drop; 
	};
	flags_[i] = keepOrDrop;
	switch(code.daughtersDepth_) {
	case SelectCode::kAll : 
	  recursiveFlagDaughters(i, *src, keepOrDrop); break;
	case SelectCode::kFirst :
	  flagDaughters(p, keepOrDrop); break;
	case SelectCode::kNone:
	  ;
	};
	switch(code.mothersDepth_) {
	case SelectCode::kAll :
	  recursiveFlagMothers(i, *src, keepOrDrop); break;
	case SelectCode::kFirst :
	  flagMothers(p, keepOrDrop); break;
	case SelectCode::kNone:
	  ;
	};
      }
    }
  }
  indices_.clear();
  int counter = 0;
  for(size_t i = 0; i < n; ++i) {
    if(flags_[i] == keep) {
      indices_.push_back(i);
      flags_[i] = counter++;
    }
  }

  auto_ptr<GenParticleCollection> out(new GenParticleCollection);
  GenParticleRefProd outRef = evt.getRefBeforePut<GenParticleCollection>();
  out->reserve(counter);
  for(vector<size_t>::const_iterator i = indices_.begin(); i != indices_.end(); ++i) {
    size_t index = *i;
    const GenParticle & gen = (*src)[index];
    const Particle & part = gen;
    out->push_back(GenParticle(part));
    GenParticle & newGen = out->back();
    addDaughterRefs(newGen, outRef, gen.daughterRefVector());
    addMotherRefs(newGen, outRef, gen.motherRefVector());
  }

  evt.put(out);
}


void GenParticlePruner::addDaughterRefs(GenParticle& newGen, GenParticleRefProd outRef, 
					const GenParticleRefVector& daughters) const {
  for(GenParticleRefVector::const_iterator j = daughters.begin();
      j != daughters.end(); ++j) {
    GenParticleRef dau = *j;
    int idx = flags_[dau.key()];
    if(idx > 0) {
      GenParticleRef newDau(outRef, static_cast<size_t>(idx));
      newGen.addDaughter(newDau);
    } else {
      const GenParticleRefVector daus = dau->daughterRefVector();
      if(daus.size()>0)
	addDaughterRefs(newGen, outRef, daus);
    }
  }
}

void GenParticlePruner::addMotherRefs(GenParticle& newGen, GenParticleRefProd outRef, 
				      const GenParticleRefVector& mothers) const {
  for(GenParticleRefVector::const_iterator j = mothers.begin();
      j != mothers.end(); ++j) {
    GenParticleRef mom = *j;
    int idx = flags_[mom.key()];
    if(idx > 0) {
      GenParticleRef newMom(outRef, static_cast<size_t>(idx));
      newGen.addMother(newMom);
    } else {
      const GenParticleRefVector moms = mom->motherRefVector();
      if(moms.size()>0)
	addMotherRefs(newGen, outRef, moms);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticlePruner);
