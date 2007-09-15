#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/BlobPedestals.h"
#include "CondFormats/Calibration/interface/BlobNoises.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/Calibration/interface/CalibHistograms.h"
/*#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
*/
namespace {
  std::vector< Pedestals::Item >::iterator tmp0;
  std::vector< Pedestals::Item >::const_iterator tmp1;
  std::vector< std::vector<BlobNoises::DetRegistry> >::iterator tmp2;
  std::vector< std::vector<BlobNoises::DetRegistry> >::const_iterator tmp3; 
  std::vector< mySiStripNoises::SiStripData >::iterator tmp4;
  std::vector< mySiStripNoises::SiStripData >::const_iterator tmp5;
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
}
