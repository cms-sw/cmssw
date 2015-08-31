#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/HepMCCandAlgos/interface/PdgEntryReplacer.h"
#include "DataFormats/Common/interface/Association.h"

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
  void produce(edm::Event&, const edm::EventSetup&) override;
  bool firstEvent_;
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  int keepOrDropAll_;
  std::vector<std::string> selection_;
  std::vector<std::pair<StringCutObjectSelector<reco::GenParticle>, helper::SelectCode> > select_;
  std::vector<int> flags_;
  std::vector<size_t> indices_;
  void parse(const std::string & selection, helper::SelectCode & code, std::string & cut) const;
  void flagDaughters(const reco::GenParticle &, int);
  void flagMothers(const reco::GenParticle &, int);
  void recursiveFlagDaughters(size_t, const reco::GenParticleCollection &, int, std::vector<size_t> &);
  void recursiveFlagMothers(size_t, const reco::GenParticleCollection &, int, std::vector<size_t> &);
  void getDaughterKeys(std::vector<size_t> &, std::vector<size_t> &, const reco::GenParticleRefVector&) const;
  void getMotherKeys(std::vector<size_t> &, std::vector<size_t> &, const reco::GenParticleRefVector&) const;
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
  firstEvent_(true),
  srcToken_(consumes<GenParticleCollection>(cfg.getParameter<InputTag>("src"))), keepOrDropAll_(drop),
  selection_(cfg.getParameter<vector<string> >("select")) {
  using namespace ::helper;
  produces<GenParticleCollection>();
  produces<edm::Association<reco::GenParticleCollection> >();
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

void GenParticlePruner::recursiveFlagDaughters(size_t index, const reco::GenParticleCollection & src, int keepOrDrop,
					       std::vector<size_t> & allIndices ) {
  GenParticleRefVector daughters = src[index].daughterRefVector();
  // avoid infinite recursion if the daughters are set to "this" particle.
  size_t cachedIndex = index;
  for(GenParticleRefVector::const_iterator i = daughters.begin(); i != daughters.end(); ++i) {
    index = i->key();
    // To also avoid infinite recursion if a "loop" is found in the daughter list,
    // check to make sure the index hasn't already been added.
    if ( find( allIndices.begin(), allIndices.end(), index ) == allIndices.end() ) {
      allIndices.push_back( index );
      if ( cachedIndex != index ) {
	flags_[index] = keepOrDrop;
	recursiveFlagDaughters(index, src, keepOrDrop, allIndices);
      }
    }
  }
}

void GenParticlePruner::recursiveFlagMothers(size_t index, const reco::GenParticleCollection & src, int keepOrDrop,
					     std::vector<size_t> & allIndices ) {
  GenParticleRefVector mothers = src[index].motherRefVector();
  // avoid infinite recursion if the mothers are set to "this" particle.
  size_t cachedIndex = index;
  for(GenParticleRefVector::const_iterator i = mothers.begin(); i != mothers.end(); ++i) {
    index = i->key();
    // To also avoid infinite recursion if a "loop" is found in the daughter list,
    // check to make sure the index hasn't already been added.
    if ( find( allIndices.begin(), allIndices.end(), index ) == allIndices.end() ) {
      allIndices.push_back( index );
      if ( cachedIndex != index ) {
	flags_[index] = keepOrDrop;
	recursiveFlagMothers(index, src, keepOrDrop, allIndices);
      }
    }
  }
}

void GenParticlePruner::produce(Event& evt, const EventSetup& es) {
  if (firstEvent_) {
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
    firstEvent_ = false;
  }

  using namespace ::helper;
  Handle<GenParticleCollection> src;
  evt.getByToken(srcToken_, src);
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
	std::vector<size_t> allIndicesDa;
	std::vector<size_t> allIndicesMo;
	switch(code.daughtersDepth_) {
	case SelectCode::kAll :
	  recursiveFlagDaughters(i, *src, keepOrDrop, allIndicesDa); break;
	case SelectCode::kFirst :
	  flagDaughters(p, keepOrDrop); break;
	case SelectCode::kNone:
	  ;
	};
	switch(code.mothersDepth_) {
	case SelectCode::kAll :
	  recursiveFlagMothers(i, *src, keepOrDrop, allIndicesMo); break;
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
    } else
    {
      flags_[i]=-1; //set to invalid ref	
    }
  }

  auto_ptr<GenParticleCollection> out(new GenParticleCollection);
  GenParticleRefProd outRef = evt.getRefBeforePut<GenParticleCollection>();
  out->reserve(counter);
  
  for(vector<size_t>::const_iterator i = indices_.begin(); i != indices_.end(); ++i) {
    size_t index = *i;
    const GenParticle & gen = (*src)[index];
    const LeafCandidate & part = gen;
    out->push_back(GenParticle(part));
    GenParticle & newGen = out->back();
    //fill status flags
    newGen.statusFlags() = gen.statusFlags();
    // The "daIndxs" and "moIndxs" keep a list of the keys for the mother/daughter
    // parentage/descendency. In some cases, a circular referencing is encountered,
    // which would result in an infinite loop. The list is checked to
    // avoid this.
    vector<size_t> daIndxs, daNewIndxs;
    getDaughterKeys(daIndxs, daNewIndxs, gen.daughterRefVector());
    std::sort(daNewIndxs.begin(),daNewIndxs.end());
    for(size_t i=0; i<daNewIndxs.size(); ++i)
      newGen.addDaughter( GenParticleRef(outRef, daNewIndxs[i]) );

    vector<size_t> moIndxs, moNewIndxs;
    getMotherKeys(moIndxs, moNewIndxs, gen.motherRefVector());
    std::sort(moNewIndxs.begin(),moNewIndxs.end());
    for(size_t i=0; i<moNewIndxs.size(); ++i)
      newGen.addMother( GenParticleRef(outRef, moNewIndxs[i]) );
  }


    edm::OrphanHandle<reco::GenParticleCollection> oh = evt.put(out);
    std::auto_ptr<edm::Association<reco::GenParticleCollection> > orig2new(new edm::Association<reco::GenParticleCollection>(oh   ));
    edm::Association<reco::GenParticleCollection>::Filler orig2newFiller(*orig2new);
    orig2newFiller.insert(src, flags_.begin(), flags_.end());
    orig2newFiller.fill();
    evt.put(orig2new);
   

}


void GenParticlePruner::getDaughterKeys(vector<size_t> & daIndxs, vector<size_t> & daNewIndxs,
					const GenParticleRefVector& daughters) const {
  for(GenParticleRefVector::const_iterator j = daughters.begin();
      j != daughters.end(); ++j) {
    GenParticleRef dau = *j;
    if (find(daIndxs.begin(), daIndxs.end(), dau.key()) == daIndxs.end()) {
      daIndxs.push_back( dau.key() );
      int idx = flags_[dau.key()];
      if (idx > 0 ) {
        daNewIndxs.push_back( idx );
      } else {
        const GenParticleRefVector & daus = dau->daughterRefVector();
        if(daus.size()>0)
          getDaughterKeys(daIndxs, daNewIndxs, daus);
      }
    }
  }
}



void GenParticlePruner::getMotherKeys(vector<size_t> & moIndxs, vector<size_t> & moNewIndxs,
				      const GenParticleRefVector& mothers) const {
  for(GenParticleRefVector::const_iterator j = mothers.begin();
      j != mothers.end(); ++j) {
    GenParticleRef mom = *j;
    if (find(moIndxs.begin(), moIndxs.end(), mom.key()) == moIndxs.end()) {
      moIndxs.push_back( mom.key() );
      int idx = flags_[mom.key()];
      if (idx >= 0 ) {
        moNewIndxs.push_back( idx );
      } else {
        const GenParticleRefVector & moms = mom->motherRefVector();
        if(moms.size()>0)
          getMotherKeys(moIndxs, moNewIndxs, moms);
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticlePruner);
