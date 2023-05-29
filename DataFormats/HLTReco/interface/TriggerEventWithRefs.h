#ifndef HLTReco_TriggerEventWithRefs_h
#define HLTReco_TriggerEventWithRefs_h

/** \class trigger::TriggerEventWithRefs
 *
 *  The single EDProduct to be saved for events (RAW case)
 *  describing the details of the (HLT) trigger table
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/traits.h"
#include <string>
#include <vector>

namespace trigger {

  /// The single EDProduct to be saved in addition for each event
  /// - but only in the "RAW" case: for a fraction of all events

  class TriggerEventWithRefs : public TriggerRefsCollections, public edm::DoNotRecordParents {
  public:
    /// Helper class: trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// encoded InputTag of filter product
      std::string filterTag_;
      /// 1-after-end (std C++) indices into linearised vector of Refs
      /// (-> first start index is always 0)
      size_type photons_;
      size_type electrons_;
      size_type muons_;
      size_type jets_;
      size_type composites_;
      size_type basemets_;
      size_type calomets_;
      size_type pixtracks_;
      size_type l1em_;
      size_type l1muon_;
      size_type l1jet_;
      size_type l1etmiss_;
      size_type l1hfrings_;
      size_type pfjets_;
      size_type pftaus_;
      size_type pfmets_;
      size_type l1tmuon_;
      size_type l1tmuonShower_;
      size_type l1tegamma_;
      size_type l1tjet_;
      size_type l1ttkmuon_;
      size_type l1ttkele_;
      size_type l1ttkem_;
      size_type l1tpfjet_;
      size_type l1tpftau_;
      size_type l1thpspftau_;
      size_type l1tpftrack_;
      size_type l1tp2etsum_;
      size_type l1ttau_;
      size_type l1tetsum_;

      /// constructor
      TriggerFilterObject()
          : filterTag_(),
            photons_(0),
            electrons_(0),
            muons_(0),
            jets_(0),
            composites_(0),
            basemets_(0),
            calomets_(0),
            pixtracks_(0),
            l1em_(0),
            l1muon_(0),
            l1jet_(0),
            l1etmiss_(0),
            l1hfrings_(0),
            pfjets_(0),
            pftaus_(0),
            pfmets_(0),
            l1tmuon_(0),
            l1tmuonShower_(0),
            l1tegamma_(0),
            l1tjet_(0),
            l1ttkmuon_(0),
            l1ttkele_(0),
            l1ttkem_(0),
            l1tpfjet_(0),
            l1tpftau_(0),
            l1thpspftau_(0),
            l1tpftrack_(0),
            l1tp2etsum_(0),
            l1ttau_(0),
            l1tetsum_(0) {
        filterTag_ = edm::InputTag().encode();
      }
      TriggerFilterObject(const edm::InputTag& filterTag,
                          size_type np,
                          size_type ne,
                          size_type nm,
                          size_type nj,
                          size_type nc,
                          size_type nB,
                          size_type nC,
                          size_type nt,
                          size_type l1em,
                          size_type l1muon,
                          size_type l1jet,
                          size_type l1etmiss,
                          size_type l1hfrings,
                          size_type pfjets,
                          size_type pftaus,
                          size_type pfmets,
                          size_type l1tmuon,
                          size_type l1tmuonShower,
                          size_type l1tegamma,
                          size_type l1tjet,
                          size_type l1ttkmuon,
                          size_type l1ttkele,
                          size_type l1ttkem,
                          size_type l1tpfjet,
                          size_type l1tpftau,
                          size_type l1thpspftau,
                          size_type l1tpftrack,
                          size_type l1tp2etsum,
                          size_type l1ttau,
                          size_type l1tetsum)
          : filterTag_(filterTag.encode()),
            photons_(np),
            electrons_(ne),
            muons_(nm),
            jets_(nj),
            composites_(nc),
            basemets_(nB),
            calomets_(nC),
            pixtracks_(nt),
            l1em_(l1em),
            l1muon_(l1muon),
            l1jet_(l1jet),
            l1etmiss_(l1etmiss),
            l1hfrings_(l1hfrings),
            pfjets_(pfjets),
            pftaus_(pftaus),
            pfmets_(pfmets),
            l1tmuon_(l1tmuon),
            l1tmuonShower_(l1tmuonShower),
            l1tegamma_(l1tegamma),
            l1tjet_(l1tjet),
            l1ttkmuon_(l1ttkmuon),
            l1ttkele_(l1ttkele),
            l1ttkem_(l1ttkem),
            l1tpfjet_(l1tpfjet),
            l1tpftau_(l1tpftau),
            l1thpspftau_(l1thpspftau),
            l1tpftrack_(l1tpftrack),
            l1tp2etsum_(l1tp2etsum),
            l1ttau_(l1ttau),
            l1tetsum_(l1tetsum) {}
    };

    /// data members
  private:
    /// processName used to select products packed up
    std::string usedProcessName_;
    /// the filters recorded here
    std::vector<TriggerFilterObject> filterObjects_;

    /// methods
  public:
    /// constructors
    TriggerEventWithRefs() : TriggerRefsCollections(), usedProcessName_(), filterObjects_() {}
    TriggerEventWithRefs(const std::string& usedProcessName, size_type n)
        : TriggerRefsCollections(), usedProcessName_(usedProcessName), filterObjects_() {
      filterObjects_.reserve(n);
    }

    /// setters - to build EDProduct
    void addFilterObject(const edm::InputTag& filterTag, const TriggerFilterObjectWithRefs& tfowr) {
      filterObjects_.push_back(

          TriggerFilterObject(filterTag,
                              addObjects(tfowr.photonIds(), tfowr.photonRefs()),
                              addObjects(tfowr.electronIds(), tfowr.electronRefs()),
                              addObjects(tfowr.muonIds(), tfowr.muonRefs()),
                              addObjects(tfowr.jetIds(), tfowr.jetRefs()),
                              addObjects(tfowr.compositeIds(), tfowr.compositeRefs()),
                              addObjects(tfowr.basemetIds(), tfowr.basemetRefs()),
                              addObjects(tfowr.calometIds(), tfowr.calometRefs()),
                              addObjects(tfowr.pixtrackIds(), tfowr.pixtrackRefs()),
                              addObjects(tfowr.l1emIds(), tfowr.l1emRefs()),
                              addObjects(tfowr.l1muonIds(), tfowr.l1muonRefs()),
                              addObjects(tfowr.l1jetIds(), tfowr.l1jetRefs()),
                              addObjects(tfowr.l1etmissIds(), tfowr.l1etmissRefs()),
                              addObjects(tfowr.l1hfringsIds(), tfowr.l1hfringsRefs()),
                              addObjects(tfowr.pfjetIds(), tfowr.pfjetRefs()),
                              addObjects(tfowr.pftauIds(), tfowr.pftauRefs()),
                              addObjects(tfowr.pfmetIds(), tfowr.pfmetRefs()),
                              addObjects(tfowr.l1tmuonIds(), tfowr.l1tmuonRefs()),
                              addObjects(tfowr.l1tmuonShowerIds(), tfowr.l1tmuonShowerRefs()),
                              addObjects(tfowr.l1tegammaIds(), tfowr.l1tegammaRefs()),
                              addObjects(tfowr.l1tjetIds(), tfowr.l1tjetRefs()),
                              addObjects(tfowr.l1ttkmuonIds(), tfowr.l1ttkmuonRefs()),
                              addObjects(tfowr.l1ttkeleIds(), tfowr.l1ttkeleRefs()),
                              addObjects(tfowr.l1ttkemIds(), tfowr.l1ttkemRefs()),
                              addObjects(tfowr.l1tpfjetIds(), tfowr.l1tpfjetRefs()),
                              addObjects(tfowr.l1tpftauIds(), tfowr.l1tpftauRefs()),
                              addObjects(tfowr.l1thpspftauIds(), tfowr.l1thpspftauRefs()),
                              addObjects(tfowr.l1tpftrackIds(), tfowr.l1tpftrackRefs()),
                              addObjects(tfowr.l1tp2etsumIds(), tfowr.l1tp2etsumRefs()),
                              addObjects(tfowr.l1ttauIds(), tfowr.l1ttauRefs()),
                              addObjects(tfowr.l1tetsumIds(), tfowr.l1tetsumRefs()))

      );
    }

    /// getters - for user access
    const std::string& usedProcessName() const { return usedProcessName_; }

    /// number of filters
    size_type size() const { return filterObjects_.size(); }

    /// tag from index
    const edm::InputTag filterTag(size_type filterIndex) const {
      return edm::InputTag(filterObjects_.at(filterIndex).filterTag_);
    }

    /// index from tag
    size_type filterIndex(const edm::InputTag& filterTag) const {
      const std::string encodedFilterTag(filterTag.encode());
      const size_type n(filterObjects_.size());
      for (size_type i = 0; i != n; ++i) {
        if (encodedFilterTag == filterObjects_[i].filterTag_) {
          return i;
        }
      }
      return n;
    }

    /// slices of objects for a specific filter: [begin,end[

    std::pair<size_type, size_type> photonSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).photons_);
      const size_type end(filterObjects_.at(filter).photons_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> electronSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).electrons_);
      const size_type end(filterObjects_.at(filter).electrons_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> muonSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).muons_);
      const size_type end(filterObjects_.at(filter).muons_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> jetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).jets_);
      const size_type end(filterObjects_.at(filter).jets_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> compositeSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).composites_);
      const size_type end(filterObjects_.at(filter).composites_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> basemetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).basemets_);
      const size_type end(filterObjects_.at(filter).basemets_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> calometSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).calomets_);
      const size_type end(filterObjects_.at(filter).calomets_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> pixtrackSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).pixtracks_);
      const size_type end(filterObjects_.at(filter).pixtracks_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1emSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1em_);
      const size_type end(filterObjects_.at(filter).l1em_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1muonSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1muon_);
      const size_type end(filterObjects_.at(filter).l1muon_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1jetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1jet_);
      const size_type end(filterObjects_.at(filter).l1jet_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1etmissSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1etmiss_);
      const size_type end(filterObjects_.at(filter).l1etmiss_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1hfringsSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1hfrings_);
      const size_type end(filterObjects_.at(filter).l1hfrings_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> pfjetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).pfjets_);
      const size_type end(filterObjects_.at(filter).pfjets_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> pftauSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).pftaus_);
      const size_type end(filterObjects_.at(filter).pftaus_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> pfmetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).pfmets_);
      const size_type end(filterObjects_.at(filter).pfmets_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1tmuonSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tmuon_);
      const size_type end(filterObjects_.at(filter).l1tmuon_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tmuonShowerSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tmuonShower_);
      const size_type end(filterObjects_.at(filter).l1tmuonShower_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tegammaSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tegamma_);
      const size_type end(filterObjects_.at(filter).l1tegamma_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tjetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tjet_);
      const size_type end(filterObjects_.at(filter).l1tjet_);
      return std::pair<size_type, size_type>(begin, end);
    }

    /* Phase-2 */
    std::pair<size_type, size_type> l1ttkmuonSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1ttkmuon_);
      const size_type end(filterObjects_.at(filter).l1ttkmuon_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1ttkeleSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1ttkele_);
      const size_type end(filterObjects_.at(filter).l1ttkele_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1ttkemSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1ttkem_);
      const size_type end(filterObjects_.at(filter).l1ttkem_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tpfjetSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tpfjet_);
      const size_type end(filterObjects_.at(filter).l1tpfjet_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tpftauSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tpftau_);
      const size_type end(filterObjects_.at(filter).l1tpftau_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1thpspftauSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1thpspftau_);
      const size_type end(filterObjects_.at(filter).l1thpspftau_);
      return std::pair<size_type, size_type>(begin, end);
    }
    std::pair<size_type, size_type> l1tpftrackSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tpftrack_);
      const size_type end(filterObjects_.at(filter).l1tpftrack_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1tp2etsumSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tp2etsum_);
      const size_type end(filterObjects_.at(filter).l1tp2etsum_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1ttauSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1ttau_);
      const size_type end(filterObjects_.at(filter).l1ttau_);
      return std::pair<size_type, size_type>(begin, end);
    }

    std::pair<size_type, size_type> l1tetsumSlice(size_type filter) const {
      const size_type begin(filter == 0 ? 0 : filterObjects_.at(filter - 1).l1tetsum_);
      const size_type end(filterObjects_.at(filter).l1tetsum_);
      return std::pair<size_type, size_type>(begin, end);
    }

    /// extract Ref<C>s for a specific filter and of specific physics type

    void getObjects(size_type filter, Vids& ids, VRphoton& photons) const {
      const size_type begin(photonSlice(filter).first);
      const size_type end(photonSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, photons, begin, end);
    }
    void getObjects(size_type filter, int id, VRphoton& photons) const {
      const size_type begin(photonSlice(filter).first);
      const size_type end(photonSlice(filter).second);
      TriggerRefsCollections::getObjects(id, photons, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRelectron& electrons) const {
      const size_type begin(electronSlice(filter).first);
      const size_type end(electronSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, electrons, begin, end);
    }
    void getObjects(size_type filter, int id, VRelectron& electrons) const {
      const size_type begin(electronSlice(filter).first);
      const size_type end(electronSlice(filter).second);
      TriggerRefsCollections::getObjects(id, electrons, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRmuon& muons) const {
      const size_type begin(muonSlice(filter).first);
      const size_type end(muonSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, muons, begin, end);
    }
    void getObjects(size_type filter, int id, VRmuon& muons) const {
      const size_type begin(muonSlice(filter).first);
      const size_type end(muonSlice(filter).second);
      TriggerRefsCollections::getObjects(id, muons, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRjet& jets) const {
      const size_type begin(jetSlice(filter).first);
      const size_type end(jetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, jets, begin, end);
    }
    void getObjects(size_type filter, int id, VRjet& jets) const {
      const size_type begin(jetSlice(filter).first);
      const size_type end(jetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, jets, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRcomposite& composites) const {
      const size_type begin(compositeSlice(filter).first);
      const size_type end(compositeSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, composites, begin, end);
    }
    void getObjects(size_type filter, int id, VRcomposite& composites) const {
      const size_type begin(compositeSlice(filter).first);
      const size_type end(compositeSlice(filter).second);
      TriggerRefsCollections::getObjects(id, composites, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRbasemet& basemets) const {
      const size_type begin(basemetSlice(filter).first);
      const size_type end(basemetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, basemets, begin, end);
    }
    void getObjects(size_type filter, int id, VRbasemet& basemets) const {
      const size_type begin(basemetSlice(filter).first);
      const size_type end(basemetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, basemets, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRcalomet& calomets) const {
      const size_type begin(calometSlice(filter).first);
      const size_type end(calometSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, calomets, begin, end);
    }
    void getObjects(size_type filter, int id, VRcalomet& calomets) const {
      const size_type begin(calometSlice(filter).first);
      const size_type end(calometSlice(filter).second);
      TriggerRefsCollections::getObjects(id, calomets, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRpixtrack& pixtracks) const {
      const size_type begin(pixtrackSlice(filter).first);
      const size_type end(pixtrackSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, pixtracks, begin, end);
    }
    void getObjects(size_type filter, int id, VRpixtrack& pixtracks) const {
      const size_type begin(pixtrackSlice(filter).first);
      const size_type end(pixtrackSlice(filter).second);
      TriggerRefsCollections::getObjects(id, pixtracks, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1em& l1em) const {
      const size_type begin(l1emSlice(filter).first);
      const size_type end(l1emSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1em, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1em& l1em) const {
      const size_type begin(l1emSlice(filter).first);
      const size_type end(l1emSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1em, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1muon& l1muon) const {
      const size_type begin(l1muonSlice(filter).first);
      const size_type end(l1muonSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1muon, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1muon& l1muon) const {
      const size_type begin(l1muonSlice(filter).first);
      const size_type end(l1muonSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1muon, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1jet& l1jet) const {
      const size_type begin(l1jetSlice(filter).first);
      const size_type end(l1jetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1jet, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1jet& l1jet) const {
      const size_type begin(l1jetSlice(filter).first);
      const size_type end(l1jetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1jet, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1etmiss& l1etmiss) const {
      const size_type begin(l1etmissSlice(filter).first);
      const size_type end(l1etmissSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1etmiss, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1etmiss& l1etmiss) const {
      const size_type begin(l1etmissSlice(filter).first);
      const size_type end(l1etmissSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1etmiss, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1hfrings& l1hfrings) const {
      const size_type begin(l1hfringsSlice(filter).first);
      const size_type end(l1hfringsSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1hfrings, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1hfrings& l1hfrings) const {
      const size_type begin(l1hfringsSlice(filter).first);
      const size_type end(l1hfringsSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1hfrings, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRpfjet& pfjets) const {
      const size_type begin(pfjetSlice(filter).first);
      const size_type end(pfjetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, pfjets, begin, end);
    }
    void getObjects(size_type filter, int id, VRpfjet& pfjets) const {
      const size_type begin(pfjetSlice(filter).first);
      const size_type end(pfjetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, pfjets, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRpftau& pftaus) const {
      const size_type begin(pftauSlice(filter).first);
      const size_type end(pftauSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, pftaus, begin, end);
    }
    void getObjects(size_type filter, int id, VRpftau& pftaus) const {
      const size_type begin(pftauSlice(filter).first);
      const size_type end(pftauSlice(filter).second);
      TriggerRefsCollections::getObjects(id, pftaus, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRpfmet& pfmets) const {
      const size_type begin(pfmetSlice(filter).first);
      const size_type end(pfmetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, pfmets, begin, end);
    }
    void getObjects(size_type filter, int id, VRpfmet& pfmets) const {
      const size_type begin(pfmetSlice(filter).first);
      const size_type end(pfmetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, pfmets, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tmuon& l1tmuon) const {
      const size_type begin(l1tmuonSlice(filter).first);
      const size_type end(l1tmuonSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tmuon, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tmuon& l1tmuon) const {
      const size_type begin(l1tmuonSlice(filter).first);
      const size_type end(l1tmuonSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tmuon, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tmuonShower& l1tmuonShower) const {
      const size_type begin(l1tmuonShowerSlice(filter).first);
      const size_type end(l1tmuonShowerSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tmuonShower, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tmuonShower& l1tmuonShower) const {
      const size_type begin(l1tmuonShowerSlice(filter).first);
      const size_type end(l1tmuonShowerSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tmuonShower, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tegamma& l1tegamma) const {
      const size_type begin(l1tegammaSlice(filter).first);
      const size_type end(l1tegammaSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tegamma, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tegamma& l1tegamma) const {
      const size_type begin(l1tegammaSlice(filter).first);
      const size_type end(l1tegammaSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tegamma, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tjet& l1tjet) const {
      const size_type begin(l1tjetSlice(filter).first);
      const size_type end(l1tjetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tjet, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tjet& l1tjet) const {
      const size_type begin(l1tjetSlice(filter).first);
      const size_type end(l1tjetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tjet, begin, end);
    }

    /* Phase-2 */
    void getObjects(size_type filter, Vids& ids, VRl1ttkmuon& l1ttkmuon) const {
      const size_type begin(l1ttkmuonSlice(filter).first);
      const size_type end(l1ttkmuonSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1ttkmuon, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1ttkmuon& l1ttkmuon) const {
      const size_type begin(l1ttkmuonSlice(filter).first);
      const size_type end(l1ttkmuonSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1ttkmuon, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1ttkele& l1ttkele) const {
      const size_type begin(l1ttkeleSlice(filter).first);
      const size_type end(l1ttkeleSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1ttkele, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1ttkele& l1ttkele) const {
      const size_type begin(l1ttkeleSlice(filter).first);
      const size_type end(l1ttkeleSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1ttkele, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1ttkem& l1ttkem) const {
      const size_type begin(l1ttkemSlice(filter).first);
      const size_type end(l1ttkemSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1ttkem, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1ttkem& l1ttkem) const {
      const size_type begin(l1ttkemSlice(filter).first);
      const size_type end(l1ttkemSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1ttkem, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tpfjet& l1tpfjet) const {
      const size_type begin(l1tpfjetSlice(filter).first);
      const size_type end(l1tpfjetSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tpfjet, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tpfjet& l1tpfjet) const {
      const size_type begin(l1tpfjetSlice(filter).first);
      const size_type end(l1tpfjetSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tpfjet, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tpftau& l1tpftau) const {
      const size_type begin(l1tpftauSlice(filter).first);
      const size_type end(l1tpftauSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tpftau, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tpftau& l1tpftau) const {
      const size_type begin(l1tpftauSlice(filter).first);
      const size_type end(l1tpftauSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tpftau, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1thpspftau& l1thpspftau) const {
      const size_type begin(l1thpspftauSlice(filter).first);
      const size_type end(l1thpspftauSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1thpspftau, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1thpspftau& l1thpspftau) const {
      const size_type begin(l1thpspftauSlice(filter).first);
      const size_type end(l1thpspftauSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1thpspftau, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tpftrack& l1tpftrack) const {
      const size_type begin(l1tpftrackSlice(filter).first);
      const size_type end(l1tpftrackSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tpftrack, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tpftrack& l1tpftrack) const {
      const size_type begin(l1tpftrackSlice(filter).first);
      const size_type end(l1tpftrackSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tpftrack, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tp2etsum& l1tp2etsum) const {
      const size_type begin(l1tp2etsumSlice(filter).first);
      const size_type end(l1tp2etsumSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tp2etsum, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tp2etsum& l1tp2etsum) const {
      const size_type begin(l1tp2etsumSlice(filter).first);
      const size_type end(l1tp2etsumSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tp2etsum, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1ttau& l1ttau) const {
      const size_type begin(l1ttauSlice(filter).first);
      const size_type end(l1ttauSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1ttau, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1ttau& l1ttau) const {
      const size_type begin(l1ttauSlice(filter).first);
      const size_type end(l1ttauSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1ttau, begin, end);
    }

    void getObjects(size_type filter, Vids& ids, VRl1tetsum& l1tetsum) const {
      const size_type begin(l1tetsumSlice(filter).first);
      const size_type end(l1tetsumSlice(filter).second);
      TriggerRefsCollections::getObjects(ids, l1tetsum, begin, end);
    }
    void getObjects(size_type filter, int id, VRl1tetsum& l1tetsum) const {
      const size_type begin(l1tetsumSlice(filter).first);
      const size_type end(l1tetsumSlice(filter).second);
      TriggerRefsCollections::getObjects(id, l1tetsum, begin, end);
    }
  };

}  // namespace trigger

#endif
