#ifndef EgammaTools_ConversionInfo_h
#define EgammaTools_ConversionInfo_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"


class ConversionInfo
 {
  public:

    ConversionInfo()
     {
      dist_ = -9999.;
      dcot_ = -9999.;
      radiusOfConversion_ = -9999.;
      pointOfConversion_ = math::XYZPoint(-9999.,-9999.,-9999);
      conversionPartnerCtfTk_ = reco::TrackRef();
      conversionPartnerGsfTk_ = reco::GsfTrackRef();
      deltaMissingHits_ = -9999;
      flag_ = -9999;
     }
    ~ConversionInfo() {}
    ConversionInfo
     ( double dist, double dcot,
       double radiusOfConversion, math::XYZPoint pointOfConversion,
       reco::TrackRef conversionPartnerCtfTk,
       reco::GsfTrackRef conversionPartnerGsfTk,
       int deltaMissingHits,
       int flag )
     {
      dist_ = dist;
      dcot_ = dcot;
      radiusOfConversion_ = radiusOfConversion;
      pointOfConversion_ = pointOfConversion;
      conversionPartnerCtfTk_ = conversionPartnerCtfTk;
      conversionPartnerGsfTk_ = conversionPartnerGsfTk;
      deltaMissingHits_ = deltaMissingHits;
      flag_ = flag;
     }
    double dist() const {return dist_ ; }
    double dcot() const {return dcot_ ; }
    double radiusOfConversion() const { return radiusOfConversion_ ; }
    math::XYZPoint pointOfConversion() const { return pointOfConversion_ ; }
    reco::TrackRef conversionPartnerCtfTk() const { return conversionPartnerCtfTk_ ; }

    //if the partner track is found in the  GSF track collection,
    //we return a ref to the GSF partner track
    reco::GsfTrackRef conversionPartnerGsfTk() const { return conversionPartnerGsfTk_ ; }

    //if the partner track is found in the  CTF track collection,
    //we return a ref to the CTF partner track
    int deltaMissingHits() const { return deltaMissingHits_ ; }
    /*
      if(flag == 0) //Partner track found in the CTF collection using the electron's CTF track
      if(flag == 1) //Partner track found in the CTF collection using the electron's GSF track
      if(flag == 2) //Partner track found in the GSF collection using the electron's CTF track
      if(flag == 3) //Partner track found in the GSF collection using the electron's GSF track
     */
    int flag() const { return flag_ ; }

    reco::TrackRef conversionPartnerTk() const {
      edm::LogWarning("ConversionInfo") << "The conversionPartnerTk() function is deprecated, but still returns the CTF partner track if found! \n"
                << "Please use either conversionPartnerCtfTk() and conversionPartnerGsfTk() instead. \n";
      return conversionPartnerCtfTk_;
    }

  private:

    double dist_;
    double dcot_;
    double radiusOfConversion_;
    math::XYZPoint pointOfConversion_;
    reco::TrackRef conversionPartnerCtfTk_;
    reco::GsfTrackRef conversionPartnerGsfTk_;
    int deltaMissingHits_;
    int flag_;

 } ;


#endif
