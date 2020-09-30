#ifndef RecoParticleFlow_PFProducer_PFMultiLinksIndex_h
#define RecoParticleFlow_PFProducer_PFMultiLinksIndex_h

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <set>

namespace reco {

  //For each element, store the indices of the linked elements
  using ElementLinks = std::vector<std::set<size_t>>;

  //stores the links between PFElements that are found by KDTrees.
  class PFMultiLinksIndex {
  public:
    PFMultiLinksIndex(size_t nelements)
        : _linkdata({{ElementLinks(nelements, std::set<size_t>()),
                      ElementLinks(nelements, std::set<size_t>()),
                      ElementLinks(nelements, std::set<size_t>()),
                      ElementLinks(nelements, std::set<size_t>()),
                      ElementLinks(nelements, std::set<size_t>()),
                      ElementLinks(nelements, std::set<size_t>())}}){

          };

    void addLink(size_t ielem1,
                 size_t ielem2,
                 reco::PFBlockElement::Type src_type,
                 reco::PFBlockElement::Type dst_type) {
      LogDebug("PFMultiLinksIndex") << "adding link src=" << src_type << " dst=" << dst_type << " ielem1=" << ielem1
                                    << " ielem2=" << ielem2;
      _linkdata[getLinkIndex(src_type, dst_type)][ielem1].insert(ielem2);
    }

    bool isValid(size_t ielem1,
                 size_t ielem2,
                 reco::PFBlockElement::Type src_type,
                 reco::PFBlockElement::Type dst_type) const {
      return getNumLinks(ielem1, src_type, dst_type) != 0;
    }

    size_t getNumLinks(size_t ielem1, reco::PFBlockElement::Type src_type, reco::PFBlockElement::Type dst_type) const {
      return _linkdata[getLinkIndex(src_type, dst_type)][ielem1].size();
    }

    bool isLinked(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type src_type,
                  reco::PFBlockElement::Type dst_type) const {
      const auto& ml = _linkdata[getLinkIndex(src_type, dst_type)][ielem1];
      return ml.find(ielem2) != ml.end();
    }

  private:
    //For each link type, store the per-element links (indices of linked elements)
    std::array<ElementLinks, 6> _linkdata;

    size_t getLinkIndex(reco::PFBlockElement::Type src_type, reco::PFBlockElement::Type dst_type) const {
      if (src_type == reco::PFBlockElement::ECAL && dst_type == reco::PFBlockElement::TRACK) {
        return 0;
      } else if (src_type == reco::PFBlockElement::PS2 && dst_type == reco::PFBlockElement::ECAL) {
        return 1;
      } else if (src_type == reco::PFBlockElement::PS1 && dst_type == reco::PFBlockElement::ECAL) {
        return 2;
      } else if (src_type == reco::PFBlockElement::TRACK && dst_type == reco::PFBlockElement::HCAL) {
        return 3;
      } else if (src_type == reco::PFBlockElement::TRACK && dst_type == reco::PFBlockElement::HFHAD) {
        return 4;
      } else if (src_type == reco::PFBlockElement::TRACK && dst_type == reco::PFBlockElement::HFEM) {
        return 5;
      }

      throw cms::Exception("LogicError") << "Unhandled types in PFMultiLinksIndex::getLinkIndex";
    }
  };

}  // namespace reco

#endif
