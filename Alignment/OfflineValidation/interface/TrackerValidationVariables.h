
#ifndef TrackerTrackerValidationVariables_h
#define TrackerTrackerValidationVariables_h
// system include files
#include <memory>
#include <vector>


// to be removed - but for now breaking plugins/TrackerOfflineValidation.cc
// that has some interfering development:
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class MagneticField;
class TrackerGeometry;

class TrackerValidationVariables  {
 public:  
  struct AVHitStruct{
    AVHitStruct() : resX(-999.), resY(-999.), resErrX(-999.), resErrY(-999.), resXprime(-999.), resXprimeErr(-999.), 
	 resYprime(-999.), resYprimeErr(-999.), phi(-999.), eta(-999.), rawDetId(0), overlapres(std::make_pair(0,-999.)) {}
    float resX;
    float resY;
    float resErrX;
    float resErrY;
    float resXprime;
    float resXprimeErr;
    float resYprime;
    float resYprimeErr;
    float phi;
    float eta;
    uint32_t rawDetId;
    std::pair<uint,float> overlapres;
  };
  struct AVTrackStruct{
  AVTrackStruct() : pt(0.), ptError(0.), px(0.), py(0.), pz(0.), eta(0.), phi(0.), kappa(0.),
		    chi2(0.), normchi2(0), d0(-999.), dz(-999.), charge(-999), numberOfValidHits(0),numberOfLostHits(0) {};
    float pt;
    float ptError;
    float px;
    float py;
    float pz;
    float eta;
    float phi;
    float kappa;
    float chi2;
    float chi2Prob;
    float normchi2;
    float d0;
    float dz;
    int charge;
    int numberOfValidHits;
    int numberOfLostHits;

  };
  TrackerValidationVariables();
  TrackerValidationVariables(const edm::EventSetup&, const edm::ParameterSet&);
  ~TrackerValidationVariables();
  void fillHitQuantities(const edm::Event&, std::vector<AVHitStruct> & v_avhitout );
  void fillTrackQuantities(const edm::Event&, std::vector<AVTrackStruct> & v_avtrackout );
 private:  
  const edm::ParameterSet conf_;
  edm::ESHandle<TrackerGeometry> tkGeom_;
  edm::ESHandle<MagneticField> magneticField_;
  //edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

};
#endif
