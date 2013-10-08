//---- Add the Class you need 
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"


#include "CondCore/Utilities/interface/CondCachedIter.h"


#include <iostream>

namespace {
  struct dictionary {

    CondCachedIter<Pedestals> dummy0;

    CondCachedIter<AlignmentErrors> dummy1;

    CondCachedIter<EcalPedestals> dummy2;

    CondCachedIter<EcalLaserAPDPNRatios> dummy3;

    CondCachedIter<SiPixelGainCalibration> dummy4;

    CondCachedIter<SiStripFedCabling> dummy5;

    CondCachedIter<DTReadOutMapping> dummy6;

    CondCachedIter<mySiStripNoises> dummy9;

    CondIter<Pedestals> dummy7;

    CondIter<EcalPedestals> dummy8;

  };
}

