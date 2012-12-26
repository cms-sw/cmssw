#include "CondFormats/SiPixelObjects/interface/SiPixelDetSummary.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

using namespace std;

// ----------------------------------------------------------------------
SiPixelDetSummary::SiPixelDetSummary(int verbose): fComputeMean(true), fVerbose(verbose) {

  unsigned int layers[] = {3, 4};
  unsigned index = 0; 

  for (unsigned int idet = 0; idet < 2; ++idet) {
    for (unsigned int il = 0; il < layers[idet]; ++il) {
      index = (idet+1)*10000 + (il+1)*1000; 
      if (fVerbose) cout << "Adding index = " << index << endl;
      fCountMap[index] = 0; 
    }
  }
}


// ----------------------------------------------------------------------
void SiPixelDetSummary::add(const DetId & detid) {
  fComputeMean= false; 
  add(detid, 0.); 
}


// ----------------------------------------------------------------------
void SiPixelDetSummary::add(const DetId & detid, const float & value) {
  
  int detNum = -1;
  int idet(-1), il(-1); 
  string name;

  switch (detid.subdetId()) {
  case PixelSubdetector::PixelBarrel: {
    idet = 1;
    il   = PixelBarrelName(detid).layerName();
    name = PixelBarrelName(detid).name();
    break;
  }
  case PixelSubdetector::PixelEndcap: {
    idet = 2;
    PixelEndcapName::HalfCylinder hc = PixelEndcapName(detid).halfCylinder();
    name = PixelEndcapName(detid).name();
    if (hc == PixelEndcapName::pI || hc == PixelEndcapName::pO) {
      il = 3 - PixelEndcapName(detid).diskName();
    }
    if (hc == PixelEndcapName::mI || hc == PixelEndcapName::mO) {
      il = 2 + PixelEndcapName(detid).diskName();
    }
    break;
  }
  }

  detNum = idet*10000 + il*1000;

  if (fVerbose > 0)
    cout << "detNum: " << detNum 
	 << " detID: " << static_cast<int>(detid) 
	 << " " << name
	 << endl;

  fMeanMap[detNum] += value;
  fRmsMap[detNum] += value*value;
  fCountMap[detNum] += 1;
}

// ----------------------------------------------------------------------
void SiPixelDetSummary::print(std::stringstream & ss, const bool mean) const {
  std::map<int, int>::const_iterator countIt   = fCountMap.begin();
  std::map<int, double>::const_iterator meanIt = fMeanMap.begin();
  std::map<int, double>::const_iterator rmsIt  = fRmsMap.begin();
  
  ss << "subDet" << setw(15) << "layer" << setw(16);
  if (mean) ss << "mean +- rms" << endl;
  else ss << "count" << endl;

  std::string detector;
  std::string oldDetector;

  for (; countIt != fCountMap.end(); ++countIt, ++meanIt, ++rmsIt ) {
    int count = countIt->second;
    double mean = 0.;
    double rms = 0.;
    if (fComputeMean && count != 0) {
      mean = (meanIt->second)/count;
      rms = (rmsIt->second)/count - mean*mean;
      if (rms <= 0)
	rms = 0;
      else
	rms = sqrt(rms);
    }

    // -- Detector type
    switch ((countIt->first)/10000) {
    case 1:
      detector = "BPIX";
      break;
    case 2:
      detector = "FPIX";
      break;
    }
    if( detector != oldDetector ) {
      ss << std::endl << detector;
      oldDetector = detector;
    }
    else ss << "    ";

    // -- Layer number
    int layer = (countIt->first)/1000 - (countIt->first)/10000*10;
    
    ss << std::setw(15) << layer << std::setw(13) ;
    if (fComputeMean) ss << mean << " +- " << rms << std::endl;
    else ss << countIt->second << std::endl;
  }
}
