#ifndef __PFBlockElementCluster__
#define __PFBlockElementCluster__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace reco {

  /// \brief Cluster Element.
  ///
  /// this class contains a reference to a PFCluster
  class PFBlockElementCluster final : public PFBlockElement {
  public:
    PFBlockElementCluster() {}

    /// \brief constructor.
    /// type must be equal to PS1, PS2, ECAL, HCAL.
    /// \todo add a protection against the other types...
    PFBlockElementCluster(const PFClusterRef& ref, PFBlockElement::Type type)
        : PFBlockElement(type), clusterRef_(ref) {}

    PFBlockElement* clone() const override { return new PFBlockElementCluster(*this); }

    /// \return reference to the corresponding cluster
    const PFClusterRef& clusterRef() const override { return clusterRef_; }
    const SuperClusterRef& superClusterRef() const { return superClusterRef_; }

    void setSuperClusterRef(const SuperClusterRef& ref) { superClusterRef_ = ref; }

    void Dump(std::ostream& out = std::cout, const char* tab = " ") const override;

  private:
    /// reference to the corresponding cluster
    PFClusterRef clusterRef_;
    SuperClusterRef superClusterRef_;
  };
}  // namespace reco

#endif
