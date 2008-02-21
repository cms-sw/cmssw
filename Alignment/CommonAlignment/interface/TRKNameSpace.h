#ifndef Alignment_CommonAlignment_TRKNameSpace_H
#define Alignment_CommonAlignment_TRKNameSpace_H

/** \namespace trk
 *
 *  Namespace for numbering sub-detectors in Tracker.
 *
 *  Numbering starts from 1.
 *
 *  $Date: 2007/10/18 09:57:10 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

namespace align
{
  namespace trk
  {
    enum
    {
      TPB = PixelSubdetector::PixelBarrel,
      TPE = PixelSubdetector::PixelEndcap,
      TIB = StripSubdetector::TIB,
      TID = StripSubdetector::TID,
      TOB = StripSubdetector::TOB,
      TEC = StripSubdetector::TEC,
			MAX = TEC + 1
    };
	}
}

#endif
