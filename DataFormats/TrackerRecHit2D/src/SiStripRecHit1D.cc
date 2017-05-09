#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include <limits>

// ctor
SiStripRecHit1D::SiStripRecHit1D()
{
    sigmaPitch_.store(-1., std::memory_order_release);
}

// copy ctor
SiStripRecHit1D::SiStripRecHit1D(const SiStripRecHit1D& src)
{
    src.sigmaPitch_.exchange(
            sigmaPitch_.exchange(src.sigmaPitch_.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// copy assingment operator
SiStripRecHit1D& SiStripRecHit1D::operator=(const SiStripRecHit1D& rhs)
{
    SiStripRecHit1D temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void SiStripRecHit1D::swap(SiStripRecHit1D& other) {
    other.sigmaPitch_.exchange(
            sigmaPitch_.exchange(other.sigmaPitch_.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// move constructor
SiStripRecHit1D::SiStripRecHit1D(SiStripRecHit1D&& other)
    : SiStripRecHit1D() {
    other.swap(*this);
}

SiStripRecHit1D::SiStripRecHit1D(const SiStripRecHit2D* hit2D) :
TrackerSingleRecHit(hit2D->localPosition(),
		    LocalError(hit2D->localPositionError().xx(),0.f,std::numeric_limits<float>::max()),
		    hit2D->geographicalId(), hit2D->omniCluster()
		    )
{
  sigmaPitch_.store(-1, std::memory_order_release);
}

SiStripRecHit1D::SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
       const DetId& id,
       OmniClusterRef const&  clus) : TrackerSingleRecHit(p,e,id,clus)
{
  sigmaPitch_.store(-1., std::memory_order_release);
}

SiStripRecHit1D::SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
       const DetId& id,
       ClusterRef const&  clus) : TrackerSingleRecHit(p,e,id,clus)
{
  sigmaPitch_.store(-1., std::memory_order_release);
}

SiStripRecHit1D::SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
       const DetId& id,
       ClusterRegionalRef const& clus) : TrackerSingleRecHit(p,e,id,clus)
{
  sigmaPitch_.store(-1., std::memory_order_release);
}

double SiStripRecHit1D::sigmaPitch() const
{
    return sigmaPitch_.load(std::memory_order_acquire);
}
void SiStripRecHit1D::setSigmaPitch(double sigmap) const
{
    sigmaPitch_.store(sigmap, std::memory_order_release);
}
