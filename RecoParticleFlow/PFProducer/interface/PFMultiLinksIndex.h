#ifndef MultiLinks_h
#define MultiLinks_h

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <set>

namespace reco {

  struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
    }
  };

  //stores the links between PFElements that are found by KDTree. This vector has size()==elements.size()
  //For the set at location N in the vector, we have the indices of the elements linked to element N.

  //we store the following links:
  //ECAL -> set of tracks (KDTreeLinkerTrackEcal)
  //HCAL, HFHAD, HFEM -> set of tracks (KDTreeLinkerTrackHcal)
  //ECAL -> set of PS1, PS2 (KDTreeLinkerPSEcal)
  class PFMultiLinksIndex {
  public:
    PFMultiLinksIndex(size_t nelements) {
      _linkdata[std::make_pair(reco::PFBlockElement::ECAL, reco::PFBlockElement::TRACK)].resize(nelements,
                                                                                                std::set<size_t>());
      _linkdata[std::make_pair(reco::PFBlockElement::PS2, reco::PFBlockElement::ECAL)].resize(nelements,
                                                                                              std::set<size_t>());
      _linkdata[std::make_pair(reco::PFBlockElement::PS1, reco::PFBlockElement::ECAL)].resize(nelements,
                                                                                              std::set<size_t>());
      _linkdata[std::make_pair(reco::PFBlockElement::TRACK, reco::PFBlockElement::HCAL)].resize(nelements,
                                                                                                std::set<size_t>());
      _linkdata[std::make_pair(reco::PFBlockElement::TRACK, reco::PFBlockElement::HFHAD)].resize(nelements,
                                                                                                 std::set<size_t>());
      _linkdata[std::make_pair(reco::PFBlockElement::TRACK, reco::PFBlockElement::HFEM)].resize(nelements,
                                                                                                std::set<size_t>());
    };

    void addLink(size_t ielem1,
                 size_t ielem2,
                 reco::PFBlockElement::Type src_type,
                 reco::PFBlockElement::Type dst_type) {
      LogDebug("PFMultiLinksIndex") << "adding link src=" << src_type << " dst=" << dst_type << " ielem1=" << ielem1
                                    << " ielem2=" << ielem2;
      _linkdata.at(std::make_pair(src_type, dst_type))[ielem1].insert(ielem2);
    }

    bool isValid(size_t ielem1, reco::PFBlockElement::Type src_type, reco::PFBlockElement::Type dst_type) const {
      return getNumLinks(ielem1, src_type, dst_type) != 0;
    }

    size_t getNumLinks(size_t ielem1, reco::PFBlockElement::Type src_type, reco::PFBlockElement::Type dst_type) const {
      return _linkdata.at(std::make_pair(src_type, dst_type))[ielem1].size();
    }

    bool isLinked(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type src_type,
                  reco::PFBlockElement::Type dst_type) const {
      const auto& ml = _linkdata.at(std::make_pair(src_type, dst_type))[ielem1];
      return ml.find(ielem2) != ml.end();
    }

  private:
    std::unordered_map<std::pair<reco::PFBlockElement::Type, reco::PFBlockElement::Type>,
                       std::vector<std::set<size_t>>,
                       pair_hash>
        _linkdata;
  };

}  // namespace reco

#endif
