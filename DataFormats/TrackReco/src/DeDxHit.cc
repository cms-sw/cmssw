#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
using namespace reco;

DeDxHit::DeDxHit(float ch,float dist,float len,DetId id): m_charge(ch),m_distance(dist),m_pathLength(len)
  {
        uint32_t subdet = id.subdetId();
        m_subDetId = (subdet&0x7)<<5;
        uint32_t layer = 0;

          if (subdet == PixelSubdetector::PixelBarrel)
              layer = PXBDetId(id).layer();
            else if (subdet == PixelSubdetector::PixelEndcap)
              layer = PXFDetId(id).disk();
            else if (subdet == StripSubdetector::TIB)
              layer = TIBDetId(id).layer();
            else if (subdet == StripSubdetector::TID)
              layer = (TIDDetId(id).wheel() & 0xF ) + ( (TIDDetId(id).side() -1 )<<4 );
            else if (subdet == StripSubdetector::TOB)
              layer = TOBDetId(id).layer();
            else if (subdet == StripSubdetector::TEC)
              layer = (TECDetId(id).wheel() & 0xF ) + ( (TECDetId(id).side() -1 )<<4 );
      m_subDetId += (layer & 0x1F) ; 
  }


