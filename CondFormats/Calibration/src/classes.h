#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/BlobPedestals.h"
#include "CondFormats/Calibration/interface/BlobNoises.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/Calibration/interface/CalibHistograms.h"
#include<bitset>
#include "CondFormats/Calibration/interface/boostTypeObj.h"
#include "CondFormats/Calibration/interface/mypt.h"
#include "CondFormats/Calibration/interface/fakeMenu.h"
#include "CondFormats/Calibration/interface/strKeyMap.h"
#include "CondFormats/Calibration/interface/simpleInheritance.h"

#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/Calibration/interface/Conf.h"
#include "CondFormats/Calibration/interface/big.h"

namespace {
  struct dictionary {
    fixedArray<unsigned short,2097> d;
    std::map<std::string, Algo> e;
  };
}
