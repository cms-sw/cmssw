#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/BlobPedestals.h"
#include "CondFormats/Calibration/interface/BlobNoises.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/Calibration/interface/CalibHistograms.h"
#include<bitset>
#include "CondFormats/Calibration/interface/BitArray.h"
#include "CondFormats/Calibration/interface/boostTypeObj.h"
#include "CondFormats/Calibration/interface/fixedArray.h"
namespace {
  std::vector< Pedestals::Item >::iterator tmp0;
  std::vector< Pedestals::Item >::const_iterator tmp1;
  std::vector< std::vector<BlobNoises::DetRegistry> >::iterator tmp2;
  std::vector< std::vector<BlobNoises::DetRegistry> >::const_iterator tmp3; 
  std::vector< mySiStripNoises::DetRegistry >::iterator tmp6;
  std::vector< mySiStripNoises::DetRegistry >::const_iterator tmp7;   
  std::vector<unsigned int>::iterator tmp8;
  std::vector<unsigned int>::const_iterator tmp9;
  std::vector<short>::iterator tmp10;
  std::vector<short>::const_iterator tmp11;
  std::vector<float>::iterator tmp12;
  std::vector<float>::const_iterator tmp13;
  std::vector<CalibHistogram>::iterator tmp14;
  std::vector<CalibHistogram>::const_iterator tmp15;
  std::vector<BlobComplexData>::iterator tmp16;
  std::vector<BlobComplexData>::const_iterator tmp17;
  std::vector<BlobComplexContent>::iterator tmp18;
  std::vector<BlobComplexContent>::const_iterator tmp19;
  std::vector<BlobComplexObjects>::iterator tmp20;
  std::vector<BlobComplexObjects>::const_iterator tmp21;
  std::bitset<7> a;
  std::bitset<8> b;
  BitArray<9> c;
  fixedArray<unsigned short,2097> d;
}
