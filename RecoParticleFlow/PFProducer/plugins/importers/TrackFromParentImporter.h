#ifndef __TrackFromParentImporter_H__
#define __TrackFromParentImporter_H__

#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

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
    template <class Collection, class Adaptor = noop::ParentCollectionAdaptor<Collection>>
    class TrackFromParentImporter : public BlockElementImporterBase {
    public:
      TrackFromParentImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
          : BlockElementImporterBase(conf, sumes),
            src_(sumes.consumes<Collection>(conf.getParameter<edm::InputTag>("source"))),
            vetoEndcap_(conf.getParameter<bool>("vetoEndcap")) {
        if (vetoEndcap_) {
          vetoMode_ = conf.getParameter<unsigned>("vetoMode");
          switch (vetoMode_) {
            case 1:
              hgcalPFTracksSrc_ =
                  sumes.consumes<reco::PFRecTrackCollection>(conf.getParameter<edm::InputTag>("vetoSrc"));
              break;
            case 2:
              ticlSeedingSrc_ =
                  sumes.consumes<std::vector<TICLSeedingRegion>>(conf.getParameter<edm::InputTag>("vetoSrc"));
              tracksSrc_ = sumes.consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracksSrc"));
              break;
            case 3:
              vetoSrc_ = sumes.consumes<reco::PFCandidateCollection>(conf.getParameter<edm::InputTag>("vetoSrc"));
              break;
            default:;
          }  // switch
        }    // vetoEndcap_
      }

      void importToBlock(const edm::Event&, ElementList&) const override;

    private:
      edm::EDGetTokenT<Collection> src_;
      edm::InputTag srcTag_;
      const bool vetoEndcap_;
      unsigned int vetoMode_;
      edm::EDGetTokenT<reco::PFRecTrackCollection> hgcalPFTracksSrc_;
      edm::EDGetTokenT<std::vector<TICLSeedingRegion>> ticlSeedingSrc_;
      edm::EDGetTokenT<reco::TrackCollection> tracksSrc_;
      edm::EDGetTokenT<reco::PFCandidateCollection> vetoSrc_;
      typedef std::pair<edm::ProductID, unsigned> TrackProdIDKey;
    };

    template <class Collection, class Adaptor>
    void TrackFromParentImporter<Collection, Adaptor>::importToBlock(
        const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
      typedef BlockElementImporterBase::ElementList::value_type ElementType;
      auto pfparents = e.getHandle(src_);
      //
      // Store tracks to be vetoed
      std::vector<TrackProdIDKey> vetoed;
      edm::ProductID prodIdForVeto;
      if (vetoEndcap_) {
        if (vetoMode_ == 1) {
          auto vetoesH = e.getHandle(hgcalPFTracksSrc_);
          const auto& vetoes = *vetoesH;
          for (unsigned i = 0; i < vetoes.size(); ++i) {
            vetoed.emplace_back(vetoes[i].trackRef().id(), vetoes[i].trackRef().key());
          }
        } else if (vetoMode_ == 2) {
          auto vetoesH = e.getHandle(ticlSeedingSrc_);
          const auto& vetoes = *vetoesH;
          auto tracksH = e.getHandle(tracksSrc_);
          for (unsigned i = 0; i < vetoes.size(); ++i) {
            reco::TrackRef trkref = reco::TrackRef(tracksH, vetoes[i].index);
            vetoed.emplace_back(trkref.id(), trkref.key());
          }
        } else if (vetoMode_ == 3) {
          auto vetoesH = e.getHandle(vetoSrc_);
          const auto& vetoes = *vetoesH;
          for (unsigned i = 0; i < vetoes.size(); ++i) {
            if (!vetoes[i].trackRef().isNonnull())
              continue;
            vetoed.emplace_back(vetoes[i].trackRef().id(), vetoes[i].trackRef().key());
          }
        }
      }
      std::sort(vetoed.begin(), vetoed.end());
      //
      elems.reserve(elems.size() + 2 * pfparents->size());
      //
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
            if (vetoEndcap_) {  // vetoEndcap flag
              TrackProdIDKey trk = std::make_pair(pftrack->trackRef().id(), pftrack->trackRef().key());
              auto lower = std::lower_bound(vetoed.begin(), vetoed.end(), trk);
              bool inVetoList = (lower != vetoed.end() && *lower == trk);
              if (inVetoList)
                continue;  // found a track in a veto list
            }
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
  }    // namespace importers
}  // namespace pflow
#endif
