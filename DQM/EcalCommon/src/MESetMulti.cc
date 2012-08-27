#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm
{
  MESetMulti::MESetMulti(MESet const& _seed, unsigned _nClones) :
    MESet(_seed),
    current_(0),
    sets_(_nClones)
  {
    if(_nClones == 0)
      throw_("Zero-plet MESetMulti");

    for(unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS] = _seed.clone();

    current_ = sets_[0];
  }

  MESetMulti::MESetMulti(MESetMulti const& _orig) :
    MESet(_orig),
    current_(0),
    sets_(_orig.sets_.size())
  {
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

    MESetMulti const* pRhs(dynamic_cast<MESetMulti const*>(&_rhs));
    if(pRhs){
      unsigned currentIndex(-1);
      for(unsigned iS(0); iS < pRhs->sets_.size(); ++iS){
        if(pRhs->sets_[iS] == pRhs->current_) currentIndex = iS;
        sets_.push_back(pRhs->sets_[iS]->clone());
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
      sets_[iS]->book();
  }

  bool
  MESetMulti::retrieve() const
  {
    bool retrieved(true);
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      retrieved &= sets_[iS]->retrieve();

    return retrieved;
  }

  void
  MESetMulti::clear() const
  {
    for(unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS]->clear();
  }

  void
  MESetMulti::use(unsigned _iSet) const
  {
    if(_iSet >= sets_.size()) return;

    current_ = sets_[_iSet];
  }
}
