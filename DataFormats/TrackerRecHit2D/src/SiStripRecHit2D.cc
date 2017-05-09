#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

// ctor
SiStripRecHit2D::SiStripRecHit2D()
{
    sigmaPitch_.store(-1., std::memory_order_release);
}

// copy ctor
SiStripRecHit2D::SiStripRecHit2D(const SiStripRecHit2D& src)
{
    src.sigmaPitch_.exchange(
            sigmaPitch_.exchange(src.sigmaPitch_.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// copy assingment operator
SiStripRecHit2D& SiStripRecHit2D::operator=(const SiStripRecHit2D& rhs)
{
    SiStripRecHit2D temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void SiStripRecHit2D::swap(SiStripRecHit2D& other) {
    other.sigmaPitch_.exchange(
            sigmaPitch_.exchange(other.sigmaPitch_.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// move constructor
SiStripRecHit2D::SiStripRecHit2D(SiStripRecHit2D&& other)
    : SiStripRecHit2D() {
    other.swap(*this);
}

SiStripRecHit2D::SiStripRecHit2D(const DetId& id,
      OmniClusterRef const& clus) : TrackerSingleRecHit(id, clus)
{
  sigmaPitch_.store(-1, std::memory_order_release);
}

SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
       const DetId& id,
       OmniClusterRef const& clus) : TrackerSingleRecHit(pos,err,id, clus)
{
  sigmaPitch_.store(-1, std::memory_order_release);
}

SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
       const DetId& id,
       ClusterRef const& clus) : TrackerSingleRecHit(pos,err,id, clus)
{
  sigmaPitch_.store(-1, std::memory_order_release);
}

SiStripRecHit2D::SiStripRecHit2D(const LocalPoint& pos, const LocalError& err,
      const DetId& id,
      ClusterRegionalRef const& clus) : TrackerSingleRecHit(pos,err,id, clus)
{
  sigmaPitch_.store(-1, std::memory_order_release);
}

double SiStripRecHit2D::sigmaPitch() const
{
    return sigmaPitch_.load(std::memory_order_acquire);
}
void SiStripRecHit2D::setSigmaPitch(double sigmap) const
{
    sigmaPitch_.store(sigmap, std::memory_order_release);
}
