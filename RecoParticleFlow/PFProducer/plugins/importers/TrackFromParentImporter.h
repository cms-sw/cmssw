#ifndef __TrackFromParentImporter_H__
#define __TrackFromParentImporter_H__

#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

namespace pflow {
  namespace noop {
    // this adaptor class gets redefined later to match the
    // needs of the collection and importing cuts we are using
    template <class Collection>
    class ParentCollectionAdaptor {
    public:
      static bool check_importable(const typename Collection::value_type&) { return true; }
      static const std::vector<reco::PFRecTrackRef>& get_track_refs(const typename Collection::value_type&) {
        return empty_;
      }
      static void set_element_info(reco::PFBlockElement*, const typename edm::Ref<Collection>&) {}
      static const std::vector<reco::PFRecTrackRef> empty_;
    };
  }  // namespace noop
  namespace importers {
    template <class Collection, class Adaptor = noop::ParentCollectionAdaptor<Collection> >
    class TrackFromParentImporter : public BlockElementImporterBase {
    public:
      TrackFromParentImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
          : BlockElementImporterBase(conf, sumes),
            src_(sumes.consumes<Collection>(conf.getParameter<edm::InputTag>("source"))),
            vetoEndcap_(conf.getParameter<bool>("vetoEndcap")) {
        if (vetoEndcap_) {
          vetoEndcapSrc_ =
              sumes.consumes<reco::PFRecTrackCollection>(conf.getParameter<edm::InputTag>("vetoEndcapSource"));
          srcTag_ = conf.getParameter<edm::InputTag>("source");
        }
      }

      void importToBlock(const edm::Event&, ElementList&) const override;

    private:
      edm::EDGetTokenT<Collection> src_;
      edm::InputTag srcTag_;
      const bool vetoEndcap_;
      edm::EDGetTokenT<reco::PFRecTrackCollection> vetoEndcapSrc_;
    };

    template <class Collection, class Adaptor>
    void TrackFromParentImporter<Collection, Adaptor>::importToBlock(
        const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
      typedef BlockElementImporterBase::ElementList::value_type ElementType;
      auto pfparents = e.getHandle(src_);
      //
      // Store tracks to be vetoed

      std::unordered_set<unsigned> vetoed;
      edm::ProductID prodIdForVeto;
      if (vetoEndcap_) {
        auto pftrackForVetoesH = e.getHandle(vetoEndcapSrc_);
        const auto& vetoes = *pftrackForVetoesH;
        for (unsigned i = 0; i < vetoes.size(); ++i) {
          if (i == 0)
            prodIdForVeto = vetoes[i].trackRef().id();
          else if (vetoes[i].trackRef().id() != prodIdForVeto)
            throw cms::Exception("ProductIDMismatch") << "Multiple ProductID in the veto list. "
                                                      << vetoes[i].trackRef().id() << " " << prodIdForVeto << std::endl;
          vetoed.insert(vetoes[i].trackRef().key());
        }
      }
      //
      elems.reserve(elems.size() + 2 * pfparents->size());
      //
      // setup our elements so that all the tracks from this importer are grouped together
      auto TKs_end = std::partition(
          elems.begin(), elems.end(), [](const ElementType& a) { return a->type() == reco::PFBlockElement::TRACK; });
      // insert tracks into the element list, updating tracks that exist already
      auto bpar = pfparents->cbegin();
      auto epar = pfparents->cend();
      edm::Ref<Collection> parentRef;
      reco::PFBlockElement* trkElem = nullptr;
      for (auto pfparent = bpar; pfparent != epar; ++pfparent) {
        if (Adaptor::check_importable(*pfparent)) {
          parentRef = edm::Ref<Collection>(pfparents, std::distance(bpar, pfparent));
          const auto& pftracks = Adaptor::get_track_refs(*pfparent);
          for (const auto& pftrack : pftracks) {
            if (vetoEndcap_                                      // vetoEndcap
                && vetoed.count(pftrack->trackRef().key()) != 0  // found a track key in a veto list
                &&
                prodIdForVeto ==
                    pftrack->trackRef()
                        .id()  // check if the productID is the same as that for the veto list otherwise the key-based check won't work
            )
              continue;
            //
            // Now try to update an entry in pfblock or import
            auto tk_elem = std::find_if(
                elems.begin(), TKs_end, [&](const ElementType& a) { return (a->trackRef() == pftrack->trackRef()); });
            if (tk_elem != TKs_end) {  // if found flag the track, otherwise import
              Adaptor::set_element_info(tk_elem->get(), parentRef);
            } else {
              trkElem = new reco::PFBlockElementTrack(pftrack);
              Adaptor::set_element_info(trkElem, parentRef);
              TKs_end = elems.insert(TKs_end, ElementType(trkElem));
              ++TKs_end;
            }
          }  // daughter track loop ends
        }    // end of importable check
      }      // loop on tracking coming from common parent
      elems.shrink_to_fit();
    }  // end of importToBlock

  }  // namespace importers
}  // namespace pflow
#endif
