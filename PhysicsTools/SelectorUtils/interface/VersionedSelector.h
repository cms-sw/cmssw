#ifndef PhysicsTools_SelectorUtils_VersionedSelector_h
#define PhysicsTools_SelectorUtils_VersionedSelector_h

/**
  \class    VersionedSelector VersionedSelector.h "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
  \brief    cut-flow versioning info in the event provenance

  class template to implement versioning for IDs that's available in the 
  event provenance or available by hash-code in the event record

  \author Lindsey Gray
*/

#if (!defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__) && !defined(__CLING__))

#define REGULAR_CPLUSPLUS 1
#define CINT_GUARD(CODE) CODE
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <memory>
#define SHARED_PTR(T) std::shared_ptr<T>

#else

#define CINT_GUARD(CODE)

#define SHARED_PTR(T) std::shared_ptr<T>

#endif

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/SelectorUtils/interface/CandidateCut.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
// because we need to be able to validate the ID
#include "Utilities/OpenSSL/interface/openssl_init.h"

namespace candf = candidate_functions;

namespace vid {
  class CutFlowResult;
}

template <class T>
class VersionedSelector : public Selector<T> {
public:
  VersionedSelector() : Selector<T>(), initialized_(false) {}

  VersionedSelector(const edm::ParameterSet& conf) : Selector<T>(), initialized_(false) {
    validateParamsAreTracked(conf);

    name_ = conf.getParameter<std::string>("idName");

    // now setup the md5 and cute accessor functions
    cms::openssl_init();
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    const EVP_MD* md = EVP_get_digestbyname("MD5");
    unsigned int md_len = 0;
    std::string tracked(conf.trackedPart().dump());

    EVP_DigestInit_ex(mdctx, md, nullptr);
    EVP_DigestUpdate(mdctx, tracked.c_str(), tracked.size());
    EVP_DigestFinal_ex(mdctx, id_md5_, &md_len);
    EVP_MD_CTX_free(mdctx);
    id_md5_[md_len] = 0;
    char tmp[EVP_MAX_MD_SIZE * 2 + 1];
    for (unsigned int i = 0; i < md_len; i++) {
      ::sprintf(&tmp[i * 2], "%02x", id_md5_[i]);
    }
    tmp[md_len * 2] = 0;
    md5_string_ = tmp;
    initialize(conf);
    this->retInternal_ = this->getBitTemplate();
  }

  bool operator()(const T& ref, pat::strbitset& ret) CINT_GUARD(final) {
    howfar_ = 0;
    bitmap_ = 0;
    values_.clear();
    bool failed = false;
    if (!initialized_) {
      throw cms::Exception("CutNotInitialized") << "VersionedGsfElectronSelector not initialized!" << std::endl;
    }
    for (unsigned i = 0; i < cuts_.size(); ++i) {
      reco::CandidatePtr temp(ref);
      const bool result = (*cuts_[i])(temp);
      values_.push_back(cuts_[i]->value(temp));
      if (result || this->ignoreCut(cut_indices_[i])) {
        this->passCut(ret, cut_indices_[i]);
        bitmap_ |= 1 << i;
        if (!failed)
          ++howfar_;
      } else {
        failed = true;
      }
    }
    this->setIgnored(ret);
    return (bool)ret;
  }

  bool operator()(const T& ref, edm::EventBase const& e, pat::strbitset& ret) CINT_GUARD(final) {
    // setup isolation needs
    for (size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i) {
      if (needs_event_content_[i]) {
        CutApplicatorWithEventContentBase* needsEvent = static_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
        needsEvent->getEventContent(e);
      }
    }
    return this->operator()(ref, ret);
  }

  //repeat the other operator() we left out here
  //in the base class here so they are exposed to ROOT

  /* VID BY VALUE */
  bool operator()(typename T::value_type const& t) {
    const T temp(&t, 0);  // assuming T is edm::Ptr
    return this->operator()(temp);
  }

  bool operator()(typename T::value_type const& t, edm::EventBase const& e) {
    const T temp(&t, 0);
    return this->operator()(temp, e);
  }

  bool operator()(T const& t) CINT_GUARD(final) {
    this->retInternal_.set(false);
    this->operator()(t, this->retInternal_);
    this->setIgnored(this->retInternal_);
    return (bool)this->retInternal_;
  }

  bool operator()(T const& t, edm::EventBase const& e) CINT_GUARD(final) {
    this->retInternal_.set(false);
    this->operator()(t, e, this->retInternal_);
    this->setIgnored(this->retInternal_);
    return (bool)this->retInternal_;
  }

  const unsigned char* md55Raw() const { return id_md5_; }
  bool operator==(const VersionedSelector& other) const {
    constexpr unsigned length = EVP_MAX_MD_SIZE;
    return (0 == memcmp(id_md5_, other.id_md5_, length * sizeof(unsigned char)));
  }
  const std::string& md5String() const { return md5_string_; }

  const std::string& name() const { return name_; }

  const unsigned howFarInCutFlow() const { return howfar_; }

  const unsigned bitMap() const { return bitmap_; }

  const size_t cutFlowSize() const { return cuts_.size(); }

  vid::CutFlowResult cutFlowResult() const;

  void initialize(const edm::ParameterSet&);

  CINT_GUARD(void setConsumes(edm::ConsumesCollector));

private:
  //here we check that the parameters of the VID cuts are tracked
  //we allow exactly one parameter to be untracked "isPOGApproved"
  //as if its tracked, its a pain for the md5Sums
  //due to the mechanics of PSets, it was demined easier just to
  //create a new config which doesnt have an untracked isPOGApproved
  //if isPOGApproved is tracked (if we decide to do that in the future), it keeps it
  //see https://github.com/cms-sw/cmssw/issues/19799 for the discussion
  static void validateParamsAreTracked(const edm::ParameterSet& conf) {
    edm::ParameterSet trackedPart = conf.trackedPart();
    edm::ParameterSet confWithoutIsPOGApproved;
    for (auto& paraName : conf.getParameterNames()) {
      if (paraName != "isPOGApproved")
        confWithoutIsPOGApproved.copyFrom(conf, paraName);
      else if (conf.existsAs<bool>(paraName, true))
        confWithoutIsPOGApproved.copyFrom(conf, paraName);  //adding isPOGApproved if its a tracked bool
    }
    std::string tracked(conf.trackedPart().dump()), untracked(confWithoutIsPOGApproved.dump());
    if (tracked != untracked) {
      throw cms::Exception("InvalidConfiguration") << "VersionedSelector does not allow untracked parameters"
                                                   << " in the cutflow ParameterSet!";
    }
  }

protected:
  bool initialized_;
  std::vector<SHARED_PTR(candf::CandidateCut)> cuts_;
  std::vector<bool> needs_event_content_;
  std::vector<typename Selector<T>::index_type> cut_indices_;
  unsigned howfar_, bitmap_;
  std::vector<double> values_;

private:
  unsigned char id_md5_[EVP_MAX_MD_SIZE];
  std::string md5_string_, name_;
};

template <class T>
void VersionedSelector<T>::initialize(const edm::ParameterSet& conf) {
  if (initialized_) {
    edm::LogWarning("VersionedPatElectronSelector") << "ID was already initialized!";
    return;
  }
  const std::vector<edm::ParameterSet>& cutflow = conf.getParameterSetVector("cutFlow");
  if (cutflow.empty()) {
    throw cms::Exception("InvalidCutFlow") << "You have supplied a null/empty cutflow to VersionedIDSelector,"
                                           << " please add content to the cuflow and try again.";
  }

  // this lets us keep track of cuts without knowing what they are :D
  std::vector<edm::ParameterSet>::const_iterator cbegin(cutflow.begin()), cend(cutflow.end());
  std::vector<edm::ParameterSet>::const_iterator icut = cbegin;
  std::map<std::string, unsigned> cut_counter;
  std::vector<std::string> ignored_cuts;
  for (; icut != cend; ++icut) {
    std::stringstream realname;
    const std::string& name = icut->getParameter<std::string>("cutName");
    if (!cut_counter.count(name))
      cut_counter[name] = 0;
    realname << name << "_" << cut_counter[name];
    const bool needsContent = icut->getParameter<bool>("needsAdditionalProducts");
    const bool ignored = icut->getParameter<bool>("isIgnored");
    CINT_GUARD(cuts_.emplace_back(CutApplicatorFactory::get()->create(name, *icut)));
    needs_event_content_.push_back(needsContent);
    const std::string therealname = realname.str();
    this->push_back(therealname);
    this->set(therealname);
    if (ignored)
      ignored_cuts.push_back(therealname);
    cut_counter[name]++;
  }
  this->setIgnoredCuts(ignored_cuts);

  //have to loop again to set cut indices after all are filled
  icut = cbegin;
  cut_counter.clear();
  for (; icut != cend; ++icut) {
    std::stringstream realname;
    const std::string& name = cuts_[std::distance(cbegin, icut)]->name();
    if (!cut_counter.count(name))
      cut_counter[name] = 0;
    realname << name << "_" << cut_counter[name];
    cut_indices_.push_back(typename Selector<T>::index_type(&(this->bits_), realname.str()));
    cut_counter[name]++;
  }

  initialized_ = true;
}

#ifdef REGULAR_CPLUSPLUS
#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
template <class T>
vid::CutFlowResult VersionedSelector<T>::cutFlowResult() const {
  std::map<std::string, unsigned> names_to_index;
  std::map<std::string, unsigned> cut_counter;
  for (unsigned idx = 0; idx < cuts_.size(); ++idx) {
    const std::string& name = cuts_[idx]->name();
    if (!cut_counter.count(name))
      cut_counter[name] = 0;
    std::stringstream realname;
    realname << name << "_" << cut_counter[name];
    names_to_index.emplace(realname.str(), idx);
    cut_counter[name]++;
  }
  return vid::CutFlowResult(name_, md5_string_, names_to_index, values_, bitmap_);
}

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
template <class T>
void VersionedSelector<T>::setConsumes(edm::ConsumesCollector cc) {
  for (size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i) {
    if (needs_event_content_[i]) {
      CutApplicatorWithEventContentBase* needsEvent = dynamic_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
      if (nullptr != needsEvent) {
        needsEvent->setConsumes(cc);
      } else {
        throw cms::Exception("InvalidCutConfiguration") << "Cut: " << ((CutApplicatorBase*)cuts_[i].get())->name()
                                                        << " configured to consume event products but does not "
                                                        << " inherit from CutApplicatorWithEventContenBase "
                                                        << " please correct either your python or C++!";
      }
    }
  }
}
#endif

#endif
