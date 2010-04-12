#ifndef EgammaTools_ConversionInfo_h
#define EgammaTools_ConversionInfo_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Point3D.h"

class ConversionInfo{
 public:
  ConversionInfo() {}
  ~ConversionInfo() {}
  ConversionInfo(double dist, double dcot, 
		 double radiusOfConversion, math::XYZPoint pointOfConversion,
		 reco::TrackRef conversionPartnerTk){
    dist_ = dist;
    dcot_ = dcot;
    radiusOfConversion = radiusOfConversion_;
    pointOfConversion_ = pointOfConversion_;
    conversionPartnerTk_ = conversionPartnerTk;
  }
  double dist() {return dist_;}
  double dcot() {return dcot_;}
  double radiusOfConversion() {return radiusOfConversion_;} 
  math::XYZPoint pointOfConversion() {return pointOfConversion_;}
  reco::TrackRef conversionPartnerTk() {return conversionPartnerTk_;}
 private:
  double dist_;
  double dcot_;
  double radiusOfConversion_;
  math::XYZPoint pointOfConversion_;
  reco::TrackRef conversionPartnerTk_;
};
  

#endif
