#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm
{
  MESetMulti::MESetMulti(MESet const& _seed, unsigned _nClones) :
    MESet(_seed),
    current_(0),
    sets_(_nClones, 0),
    use_(_nClones, false)
  {
    if(_nClones == 0) return;

    for(unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS] = _seed.clone();

    current_ = sets_[0];
  }

  MESetMulti::MESetMulti(MESetMulti const& _orig) :
    MESet(_orig),
    current_(0),
    sets_(_orig.sets_.size()),
    use_(_orig.use_)
  {
    if(sets_.size() == 0) return;

    unsigned currentIndex(-1);
    for(unsigned iS(0); iS < sets_.size(); ++iS){
      if(_orig.sets_[iS] == _orig.current_) currentIndex = iS;
      sets_[iS] = _orig.sets_[iS]->clone();
    }
    if(currentIndex != unsigned(-1)) current_ = sets_[currentIndex];
    else current_ = sets_[0];
  }

  MESetMulti::~MESetMulti()
  {
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      delete sets_[iS];
  }

  MESet&
  MESetMulti::operator=(MESet const& _rhs)
  {
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      delete sets_[iS];
    sets_.clear();
    current_ = 0;

    MESetMulti const* pRhs(dynamic_cast<MESetMulti const*>(&_rhs));
    if(pRhs){
      if(pRhs->sets_.size() == 0) return *this;

      use_.resize(pRhs->sets_.size(), false);

      unsigned currentIndex(-1);
      for(unsigned iS(0); iS < pRhs->sets_.size(); ++iS){
        if(pRhs->sets_[iS] == pRhs->current_) currentIndex = iS;
        sets_.push_back(pRhs->sets_[iS]->clone());
        use_[iS] = pRhs->use_[iS];
      }
      if(currentIndex != unsigned(-1)) current_ = sets_[currentIndex];
      else current_ = sets_[0];
    }
    return MESet::operator=(_rhs);
  }

  MESet*
  MESetMulti::clone() const
  {
    return new MESetMulti(*this);
  }

  void
  MESetMulti::book()
  {
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      if(use_[iS]) sets_[iS]->book();

    active_ = true;
  }

  bool
  MESetMulti::retrieve() const
  {
    bool retrieved(true);
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      if(use_[iS]) retrieved &= sets_[iS]->retrieve();

    active_ = retrieved;
    return retrieved;
  }

  void
  MESetMulti::clear() const
  {
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      if(use_[iS]) sets_[iS]->clear();

    active_ = false;
  }

  bool
  MESetMulti::use(unsigned _iSet) const
  {
    if(_iSet >= sets_.size())
      throw_("MESetMulti index out of range");

    // use_ vector is frozen once activated
    if(active_ && !use_[_iSet]) return false;

    use_[_iSet] = true;

    current_ = sets_[_iSet];

    return true;
  }
}
