#include "Alignment/TrackerAlignment/interface/TrackerNameSpace.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


//______________________________________________________________________________
align::TrackerNameSpace
::TrackerNameSpace(const TrackerTopology* topology) :
  trackerTopology_(topology),
  tpb_(topology),
  tpe_(topology),
  tib_(topology),
  tob_(topology),
  tid_(topology),
  tec_(topology)
{
}


//______________________________________________________________________________
align::TrackerNameSpace::TPB
::TPB(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}


//______________________________________________________________________________
unsigned int align::TrackerNameSpace::TPB
::moduleNumber(align::ID id) const
{
  return trackerTopology_->pxbModule(id);
}


//______________________________________________________________________________
unsigned int align::TrackerNameSpace::TPB
::ladderNumber(align::ID id) const
{
  unsigned int l = trackerTopology_->pxbLadder(id); // increases with phi
  unsigned int c = trackerTopology_->pxbLayer(id) - 1;

  // Ladder in 1st quadrant: number = lpqc_ + 1 - l     (1 to lpqc_)
  // Ladder in 2nd quadrant: number = l - lpqc_         (1 to lpqc_)
  // Ladder in 3rd quadrant: number = l - lpqc_         (lpqc_ + 1 to 2 * lpqc_)
  // Ladder in 4th quadrant: number = 5 * lpqc_ + 1 - l (lpqc_ + 1 to 2 * lpqc_)

  return l > 3 * lpqc_[c] ? 5 * lpqc_[c] + 1 - l :  // ladder in 4th quadrant
           (l > lpqc_[c] ? l - lpqc_[c] :           // ladder not in 1st quadrant
             lpqc_[c] + 1 - l);
}


//______________________________________________________________________________
unsigned int align::TrackerNameSpace::TPB
::layerNumber(align::ID id) const
{
  return trackerTopology_->pxbLayer(id);
}


//______________________________________________________________________________
unsigned int align::TrackerNameSpace::TPB
::halfBarrelNumber(align::ID id) const
{
  unsigned int l = trackerTopology_->pxbLadder(id); // increases with phi
  unsigned int c = trackerTopology_->pxbLayer(id) - 1;

  return l > lpqc_[c] && l <= 3 * lpqc_[c] ? 1 : 2;
}


//______________________________________________________________________________
unsigned int align::TrackerNameSpace::TPB
::barrelNumber(align::ID) const
{
  return 1;
}


//______________________________________________________________________________
align::TrackerNameSpace::TPE
::TPE(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}


unsigned int align::TrackerNameSpace::TPE
::moduleNumber(align::ID id) const
{
  return trackerTopology_->pxfModule(id);
}

unsigned int align::TrackerNameSpace::TPE
::panelNumber(align::ID id) const
{
  return trackerTopology_->pxfPanel(id);
}

unsigned int align::TrackerNameSpace::TPE
::bladeNumber(align::ID id) const
{
  unsigned int b = trackerTopology_->pxfBlade(id); // 1 to 24 in increasing phi

  // Blade in 1st quadrant: number = bpqd_ + 1 - b     (1 to bpqd_)
  // Blade in 2nd quadrant: number = b - bpqd_         (1 to bpqd_)
  // Blade in 3rd quadrant: number = b - bpqd_         (bpqd_ + 1 to 2 * bpqd_)
  // Blade in 4th quadrant: number = 5 * bpqd_ + 1 - b (bpqd_ + 1 to 2 * bpqd_)

  return b > 3 * bpqd_ ? // blade in 4th quadrant
    5 * bpqd_ + 1 - b :
    (b > bpqd_ ? // blade not in 1st quadrant
     b - bpqd_ : bpqd_ + 1 - b);
}

unsigned int align::TrackerNameSpace::TPE
::halfDiskNumber(align::ID id) const
{
  return trackerTopology_->pxfDisk(id);
}

unsigned int align::TrackerNameSpace::TPE
::halfCylinderNumber(align::ID id) const
{
  unsigned int b = trackerTopology_->pxfBlade(id); // 1 to 24 in increasing phi

  return b > bpqd_ && b <= 3 * bpqd_ ? 1 : 2;
}

unsigned int align::TrackerNameSpace::TPE
::endcapNumber(align::ID id) const
{
  return trackerTopology_->pxfSide(id);
}


align::TrackerNameSpace::TIB
::TIB(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}

unsigned int align::TrackerNameSpace::TIB
::moduleNumber(align::ID id) const
{
  return trackerTopology_->tibModule(id);
}

unsigned int align::TrackerNameSpace::TIB
::stringNumber(align::ID id) const
{
  std::vector<unsigned int> s = trackerTopology_->tibStringInfo(id);
  // s[1]: surface lower = 1, upper = 2
  // s[2]: string no. increases with phi

  unsigned int l = 2 * (trackerTopology_->tibLayer(id) - 1) + s[1] - 1;

  // String on +y surface: number = s                (1 to sphs_)
  // String in -y surface: number = 2 * sphs_ + 1 - s (1 to sphs_)

  return s[2] > sphs_[l] ? 2 * sphs_[l] + 1 - s[2] : s[2];
}

unsigned int align::TrackerNameSpace::TIB
::surfaceNumber(align::ID id) const
{
  return trackerTopology_->tibStringInfo(id)[1];
}

unsigned int align::TrackerNameSpace::TIB
::halfShellNumber(align::ID id) const
{
  std::vector<unsigned int> s = trackerTopology_->tibStringInfo(id);
  // s[1]: surface lower = 1, upper = 2
  // s[2]: string no. increases with phi

  unsigned int l = 2 * (trackerTopology_->tibLayer(id) - 1) + s[1] - 1;

  return s[2] > sphs_[l] ? 1 : 2;
}

unsigned int align::TrackerNameSpace::TIB
::layerNumber(align::ID id) const
{
  return trackerTopology_->tibLayer(id);
}

unsigned int align::TrackerNameSpace::TIB
::halfBarrelNumber(align::ID id) const
{
  return trackerTopology_->tibStringInfo(id)[0];
}

unsigned int align::TrackerNameSpace::TIB
::barrelNumber(align::ID) const
{
  return 1;
}

align::TrackerNameSpace::TOB
::TOB(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}

unsigned int align::TrackerNameSpace::TOB
::moduleNumber(align::ID id) const
{
  return trackerTopology_->tobModule(id);
}

unsigned int align::TrackerNameSpace::TOB
::rodNumber(align::ID id) const
{
  return trackerTopology_->tobRodInfo(id)[1];
}

unsigned int align::TrackerNameSpace::TOB
::layerNumber(align::ID id) const
{
  return trackerTopology_->tobLayer(id);
}

unsigned int align::TrackerNameSpace::TOB
::halfBarrelNumber(align::ID id) const
{
  return trackerTopology_->tobRodInfo(id)[0];
}

unsigned int align::TrackerNameSpace::TOB
::barrelNumber(align::ID) const
{
  return 1;
}

align::TrackerNameSpace::TID
::TID(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}

unsigned int align::TrackerNameSpace::TID
::moduleNumber(align::ID id) const
{
  return trackerTopology_->tidModuleInfo(id)[1];
}

unsigned int align::TrackerNameSpace::TID
::sideNumber(align::ID id) const
{
  return trackerTopology_->tidModuleInfo(id)[0];
}

unsigned int align::TrackerNameSpace::TID
::ringNumber(align::ID id) const
{
  return trackerTopology_->tidRing(id);
}

unsigned int align::TrackerNameSpace::TID
::diskNumber(align::ID id) const
{
  return trackerTopology_->tidWheel(id);
}

unsigned int align::TrackerNameSpace::TID
::endcapNumber(align::ID id) const
{
  return trackerTopology_->tidSide(id);
}

align::TrackerNameSpace::TEC
::TEC(const TrackerTopology* topology) :
  trackerTopology_(topology)
{
}

unsigned int align::TrackerNameSpace::TEC
::moduleNumber(align::ID id) const
{
  return trackerTopology_->tecModule(id);
}

unsigned int align::TrackerNameSpace::TEC
::ringNumber(align::ID id) const
{
  return trackerTopology_->tecRing(id);
}

unsigned int align::TrackerNameSpace::TEC
::petalNumber(align::ID id) const
{
  return trackerTopology_->tecPetalInfo(id)[1];
}

unsigned int align::TrackerNameSpace::TEC
::sideNumber(align::ID id) const
{
  return trackerTopology_->tecPetalInfo(id)[0];
}

unsigned int align::TrackerNameSpace::TEC
::diskNumber(align::ID id) const
{
  return trackerTopology_->tecWheel(id);
}

unsigned int align::TrackerNameSpace::TEC
::endcapNumber(align::ID id) const
{
  return trackerTopology_->tecSide(id);
}
