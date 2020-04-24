#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"

EDM_REGISTER_PLUGINFACTORY(KDTreeLinkerFactory,"KDTreeLinkerFactory");

KDTreeLinkerBase::KDTreeLinkerBase()
  : cristalPhiEtaMaxSize_ (0.04),
    cristalXYMaxSize_ (3.),
    phiOffset_ (0.25),
    debug_ (false)
{
}

KDTreeLinkerBase::~KDTreeLinkerBase()
{
}

void
KDTreeLinkerBase::setCristalPhiEtaMaxSize(float size)
{
  cristalPhiEtaMaxSize_ = size;
}

void
KDTreeLinkerBase::setCristalXYMaxSize(float size)
{
  cristalXYMaxSize_ = size;
}

void
KDTreeLinkerBase::setPhiOffset(double phiOffset)
{
  phiOffset_ = phiOffset;
}

void
KDTreeLinkerBase::setDebug(bool debug)
{
  debug_ = debug;
}

float
KDTreeLinkerBase::getCristalPhiEtaMaxSize() const
{
  return cristalPhiEtaMaxSize_;
}

float
KDTreeLinkerBase::getCristalXYMaxSize() const
{
  return cristalXYMaxSize_;
}

float
KDTreeLinkerBase::getPhiOffset() const
{
  return phiOffset_;
}

void
KDTreeLinkerBase::process()
{
  buildTree();
  searchLinks();
  updatePFBlockEltWithLinks();
  clear();
}



