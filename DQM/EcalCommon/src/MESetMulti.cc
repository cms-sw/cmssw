#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {
  MESetMulti::MESetMulti(MESet const &_seed, ReplCandidates const &_replCandidates)
      : MESet(_seed), current_(nullptr), sets_(), replCandidates_(_replCandidates) {
    PathReplacements replacements;
    std::map<std::string, unsigned> indices;
    // recursive function to set replacements
    // indices gives the multi index in each dimension
    // dimensions are alphanumerically ordered from the use of std::map
    std::function<bool(typename ReplCandidates::const_iterator &)> setReplacements(
        [&setReplacements, &replacements, &indices, this](typename ReplCandidates::const_iterator &_rItr) {
          unsigned &index(indices[_rItr->first]);
          replacements[_rItr->first] = _rItr->second[index];
          // one dimension set, go to next
          ++_rItr;
          if (_rItr == this->replCandidates_.end()) {
            // this is the last dimension. Increment the index and retutn to the
            // first
            _rItr = this->replCandidates_.begin();
            ++index;
          } else if (setReplacements(_rItr))
            ++index;

          if (index != _rItr->second.size())
            return false;
          // index has counted to the maximum of this dimension, carry over
          index = 0;
          return true;
        });

    // [dim0 = 0, dim1 = 0] -> 0, [dim0 = 0, dim1 = 1] -> 1, ...
    while (true) {
      replacements.clear();
      typename ReplCandidates::const_iterator rItr(replCandidates_.begin());
      bool last(setReplacements(rItr));
      sets_.push_back(_seed.clone(formPath(replacements)));
      if (last)
        break;
    }

    current_ = sets_[0];
  }

  MESetMulti::MESetMulti(MESetMulti const &_orig)
      : MESet(_orig), current_(nullptr), sets_(_orig.sets_.size(), nullptr), replCandidates_(_orig.replCandidates_) {
    if (sets_.empty())
      return;

    for (unsigned iS(0); iS < sets_.size(); ++iS) {
      if (!_orig.sets_[iS])
        continue;
      sets_[iS] = _orig.sets_[iS]->clone();
      if (_orig.sets_[iS] == _orig.current_)
        current_ = sets_[iS];
    }
  }

  MESetMulti::~MESetMulti() {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      delete sets_[iS];
  }

  MESet &MESetMulti::operator=(MESet const &_rhs) {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      delete sets_[iS];
    sets_.clear();
    current_ = nullptr;

    MESetMulti const *pRhs(dynamic_cast<MESetMulti const *>(&_rhs));
    if (pRhs) {
      sets_.assign(pRhs->sets_.size(), nullptr);

      for (unsigned iS(0); iS < pRhs->sets_.size(); ++iS) {
        sets_[iS] = pRhs->sets_[iS]->clone();
        if (pRhs->sets_[iS] == pRhs->current_)
          current_ = sets_[iS];
      }

      replCandidates_ = pRhs->replCandidates_;
    }

    return MESet::operator=(_rhs);
  }

  MESet *MESetMulti::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESetMulti(*this));
    path_ = path;
    return copy;
  }

  void MESetMulti::book(DQMStore::IBooker &_ibooker, EcalElectronicsMapping const *electronicsMap) {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS]->book(_ibooker, electronicsMap);

    active_ = true;
  }

  bool MESetMulti::retrieve(EcalElectronicsMapping const *electronicsMap,
                            DQMStore::IGetter &_igetter,
                            std::string *_failedPath /* = 0*/) const {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      if (!sets_[iS]->retrieve(electronicsMap, _igetter, _failedPath))
        return false;

    active_ = true;
    return true;
  }

  void MESetMulti::clear() const {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS]->clear();

    active_ = false;
  }

  void MESetMulti::reset(EcalElectronicsMapping const *electronicsMap,
                         double _content /* = 0*/,
                         double _error /* = 0.*/,
                         double _entries /* = 0.*/) {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS]->reset(electronicsMap, _content, _error, _entries);
  }

  void MESetMulti::resetAll(double _content /* = 0*/, double _error /* = 0.*/, double _entries /* = 0.*/) {
    for (unsigned iS(0); iS < sets_.size(); ++iS)
      sets_[iS]->resetAll(_content, _error, _entries);
  }

  void MESetMulti::use(unsigned _iSet) const {
    if (_iSet >= sets_.size())
      throw_("MESetMulti index out of range");

    current_ = sets_[_iSet];
  }

  unsigned MESetMulti::getIndex(PathReplacements const &_replacements) const {
    unsigned index(0);
    unsigned base(1);
    for (typename ReplCandidates::const_reverse_iterator cItr(replCandidates_.rbegin()); cItr != replCandidates_.rend();
         ++cItr) {
      typename PathReplacements::const_iterator rItr(_replacements.find(cItr->first));
      if (rItr == _replacements.end())
        throw_(cItr->first + " not given in the key for getIndex");
      unsigned nC(cItr->second.size());
      unsigned iR(0);
      for (; iR != nC; ++iR)
        if (rItr->second == cItr->second[iR])
          break;
      if (iR == nC)
        throw_(rItr->second + " not found in replacement candidates");
      index += iR * base;
      base *= nC;
    }

    return index;
  }
}  // namespace ecaldqm
