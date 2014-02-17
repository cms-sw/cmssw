// $Id: CompositeCandidate.cc,v 1.15 2010/12/06 20:04:17 wmtan Exp $
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeCandidate::CompositeCandidate(const Candidate & c,
				       const std::string& name) :
  LeafCandidate(c), name_(name) {
  size_t n = c.numberOfDaughters();
  for(size_t i = 0; i != n; ++i) {
      addDaughter(*c.daughter(i));
  }
}

CompositeCandidate::CompositeCandidate(const Candidate & c,
				       const std::string& name,
				       role_collection const & roles) :
  LeafCandidate(c), name_(name), roles_(roles) {
  size_t n = c.numberOfDaughters();
  size_t r = roles_.size();
  bool sameSize = (n == r);
  for(size_t i = 0; i != n; ++i) {
    if (sameSize && r > 0) 
      addDaughter(*c.daughter(i), roles_[i]);
    else
      addDaughter(*c.daughter(i));
  }
}

CompositeCandidate::~CompositeCandidate() { }

CompositeCandidate * CompositeCandidate::clone() const { return new CompositeCandidate(* this); }

Candidate::const_iterator CompositeCandidate::begin() const { return const_iterator(new const_iterator_imp_specific(dau.begin())); }

Candidate::const_iterator CompositeCandidate::end() const { return const_iterator(new const_iterator_imp_specific(dau.end())); }    

Candidate::iterator CompositeCandidate::begin() { return iterator(new iterator_imp_specific(dau.begin())); }

Candidate::iterator CompositeCandidate::end() { return iterator(new iterator_imp_specific(dau.end())); }    

const Candidate * CompositeCandidate::daughter(size_type i) const { 
  return (i < numberOfDaughters()) ? & dau[ i ] : 0; // i >= 0, since i is unsigned
}

Candidate * CompositeCandidate::daughter(size_type i) { 
  Candidate * d = (i < numberOfDaughters()) ? & dau[ i ] : 0; // i >= 0, since i is unsigned
  return d;
}

const Candidate * CompositeCandidate::mother(size_type i) const { 
  return 0;
}

size_t CompositeCandidate::numberOfDaughters() const { return dau.size(); }

size_t CompositeCandidate::numberOfMothers() const { return 0; }

bool CompositeCandidate::overlap(const Candidate & c2) const {
  throw cms::Exception("Error") << "can't check overlap internally for CompositeCanddate";
}

void CompositeCandidate::applyRoles() {

  if (roles_.size() == 0)
    return;

  // Check if there are the same number of daughters and roles
  int N1 = roles_.size();
  int N2 = numberOfDaughters();
  if (N1 != N2) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::applyRoles : Number of roles and daughters differ, this is an error.\n";
  }
  // Set up the daughter roles
  for (int i = 0 ; i < N1; ++i) {
    std::string role = roles_[i];
    Candidate * c = CompositeCandidate::daughter(i);

    CompositeCandidate * c1 = dynamic_cast<CompositeCandidate *>(c);
    if (c1 != 0) {
      c1->setName(role);
    }
  }
}

Candidate * CompositeCandidate::daughter(const std::string& s) {
  int ret = -1;
  int i = 0, N = roles_.size();
  bool found = false;
  for (; i < N && !found; ++i) {
    if (s == roles_[i]) {
      found = true;
      ret = i;
    }
  }

  if (ret < 0) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

const Candidate * CompositeCandidate::daughter(const std::string& s) const  {
  int ret = -1;
  int i = 0, N = roles_.size();
  bool found = false;
  for (; i < N && !found; ++i) {
    if (s == roles_[i]) {
      found = true;
      ret = i;
    }
  }

  if (ret < 0) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

void CompositeCandidate::addDaughter(const Candidate & cand, const std::string& s) {
  Candidate * c = cand.clone();
  if (s != "") {
    role_collection::iterator begin = roles_.begin(), end = roles_.end();
    bool isFound = (find(begin, end, s) != end);
    if (isFound) {
      throw cms::Exception("InvalidReference")
	<< "CompositeCandidate::addDaughter: Already have role with name \"" << s 
	<< "\", please clearDaughters, or use a new name\n";
    }
    roles_.push_back(s);
    CompositeCandidate * c1 = dynamic_cast<CompositeCandidate*>(&*c);
    if (c1 != 0) {
      c1->setName(s);
    }
  }
  dau.push_back(c);
}

void CompositeCandidate::addDaughter(std::auto_ptr<Candidate> cand, const std::string& s) {
  if (s != "") {
    role_collection::iterator begin = roles_.begin(), end = roles_.end();
    bool isFound = (find(begin, end, s) != end);
    if (isFound) {
      throw cms::Exception("InvalidReference")
	<< "CompositeCandidate::addDaughter: Already have role with name \"" << s 
	<< "\", please clearDaughters, or use a new name\n";
    }
    roles_.push_back(s);  
    CompositeCandidate * c1 = dynamic_cast<CompositeCandidate*>(&*cand);
    if (c1 != 0) {
      c1->setName(s);
    }
  }
  dau.push_back(cand);
}


