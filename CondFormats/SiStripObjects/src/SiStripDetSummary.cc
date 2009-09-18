#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

void SiStripDetSummary::add(const DetId & detid, const float & value)
{
  int layer = 0;
  int stereo = 0;
  int detNum = 0;

  // Using the operator[] if the element does not exist it is created with the default value. That is 0 for integral types.
  switch (detid.subdetId()) {
  case StripSubdetector::TIB:
    {
      TIBDetId theTIBDetId(detid.rawId());
      layer = theTIBDetId.layer();
      stereo = theTIBDetId.stereo();
      detNum = 1000;
      break;
    }
  case StripSubdetector::TOB:
    {
      TOBDetId theTOBDetId(detid.rawId());
      layer = theTOBDetId.layer();
      stereo = theTOBDetId.stereo();
      detNum = 2000;
      break;
    }
  case StripSubdetector::TEC:
    {
      TECDetId theTECDetId(detid.rawId());
      // is this module in TEC+ or TEC-?
      layer = theTECDetId.wheel();
      stereo = theTECDetId.stereo();
      detNum = 3000;
      break;
    }
  case StripSubdetector::TID:
    {
      TIDDetId theTIDDetId(detid.rawId());
      // is this module in TID+ or TID-?
      layer = theTIDDetId.wheel();
      stereo = theTIDDetId.stereo();
      detNum = 4000;
      break;
    }
  }
  detNum += layer*10 + stereo*1;
  // string name( detector + boost::lexical_cast<string>(layer) + boost::lexical_cast<string>(stereo) );
  meanMap_[detNum] += value;
  rmsMap_[detNum] += value*value;
  countMap_[detNum] += 1;
}

void SiStripDetSummary::print(stringstream & ss, const bool mean) const
{
  // Compute the mean for each detector and for each layer.
  // The maps have the same key and therefore are ordered in the same way.
  map<int, int>::const_iterator countIt = countMap_.begin();
  map<int, double>::const_iterator meanIt = meanMap_.begin();
  map<int, double>::const_iterator rmsIt = rmsMap_.begin();

  ss << "subDet" << setw(15) << "layer" << setw(16) << "mono/stereo" << setw(20);
  if( mean ) ss << "mean +- rms" << endl;
  else ss << "count" << endl;

  string detector;
  string oldDetector;

  for( ; countIt != countMap_.end(); ++countIt, ++meanIt, ++rmsIt ) {
    int count = countIt->second;
    double mean = 0.;
    double rms = 0.;
    if( count != 0 ) {
      mean = (meanIt->second)/count;
      // if ( (rmsIt->second)/count - pow(mean,2) < 0 ) cout << "Error: negative value, meanIt->second = " << meanIt->second << " count = " << count << " rmsIt->second/count = " << (rmsIt->second)/count << " mean*mean = " << mean*mean << " pow = " << pow(mean,2) << endl;
      rms = (rmsIt->second)/count - mean*mean;
      if (rms <= 0)
	rms = 0;
      else
	rms = sqrt(rms);
    }
    // Detector type
    switch ((countIt->first)/1000) {
    case 1:
      detector = "TIB ";
      break;
    case 2:
      detector = "TOB ";
      break;
    case 3:
      detector = "TEC ";
      break;
    case 4:
      detector = "TID ";
      break;
    }
    if( detector != oldDetector ) {
      ss << endl << detector;
      oldDetector = detector;
    }
    else ss << "    ";
    // Layer number
    int layer = (countIt->first)/10 - (countIt->first)/1000*100;
    int stereo = countIt->first - layer*10 -(countIt->first)/1000*1000;

    ss << setw(15) << layer << setw(13) << stereo << setw(18);
    if( mean ) ss << mean << " +- " << rms << endl;
    else ss << countIt->second << endl;
  }
}
