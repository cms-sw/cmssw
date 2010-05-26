#include "CondFormats/Common/interface/PayloadWrapper.h"

#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/BlobPedestals.h"
#include "CondFormats/Calibration/interface/BlobNoises.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/Calibration/interface/CalibHistograms.h"
#include<bitset>
#include "CondFormats/Calibration/interface/BitArray.h"
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
    BitArray<9> c;
    fixedArray<unsigned short,2097> d;
    std::map<std::string, Algo> e;
  };
  struct wrappers {
    pool::PolyPtr<mySiStripNoises> p0;
    cond::DataWrapper<mySiStripNoises> d0;
    pool::PolyPtr<Pedestals> p1;
    cond::DataWrapper<Pedestals> d1;
    pool::PolyPtr<BlobComplex> p2;
    cond::DataWrapper<BlobComplex> d2;
    pool::PolyPtr<condex::Efficiency> p3;
    cond::DataWrapper<condex::Efficiency> d3;
    pool::PolyPtr<BlobPedestals> p4;
    cond::DataWrapper<BlobPedestals> d4; 
    pool::PolyPtr<CalibHistograms> p5;
    cond::DataWrapper<CalibHistograms> d5; 
    pool::PolyPtr<BitArray<9> > p6;
    cond::DataWrapper<BitArray<9> > d6;
  };
}
