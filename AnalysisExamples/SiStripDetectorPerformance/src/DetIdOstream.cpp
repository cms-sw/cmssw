// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 12/04/06
// Licence: GPL

#include <ostream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/DetIdOstream.h"

std::ostream &operator<<( std::ostream &roOut, const DetIdOstream &roDO) {
  switch( roDO.roDETID.subdetId()) {
    case StripSubdetector::TIB:
      {
        TIBDetId oTIBDetId( roDO.roDETID.rawId());
        roOut << "TIB  " << oTIBDetId.layer();
        break;
      }
    case StripSubdetector::TOB:
      {
        TOBDetId oTOBDetId( roDO.roDETID.rawId());
        roOut << "TOB  " << oTOBDetId.layer();
        break;
      }
    case StripSubdetector::TID:
      {
        TIDDetId oTIDDetId( roDO.roDETID.rawId());
        roOut << "TID  " << oTIDDetId.wheel();
        break;
      }
    case StripSubdetector::TEC:
      {
        TECDetId oTECDetId( roDO.roDETID.rawId());
        roOut << "TEC  " << oTECDetId.wheel();
      }
    default:
      break;
  }

  roOut << "  " << roDO.roDETID.rawId();

  return roOut;
}
