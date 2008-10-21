#ifndef TrackerTrackerValidationVariables_h
#define TrackerTrackerValidationVariables_h
// system include files
#include <memory>
#include <vector>


#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

class TrackerValidationVariables  {
 public:  
  struct AVHitStruct{
    AVHitStruct() : resX(-999.), resY(-999.), resErrX(-999.), resErrY(-999.), resXprime(-999.), 
	 phi(-999.), eta(-999.), rawDetId(0), overlapres(std::make_pair(0,-999.)) {}
    float resX;
    float resY;
    float resErrX;
    float resErrY;
    float resXprime;
    float phi;
    float eta;
    uint32_t rawDetId;
    std::pair<uint,float> overlapres;
  };
  struct AVTrackStruct{
    AVTrackStruct() : pt(0.), px(0.), py(0.), pz(0.), eta(0.), phi(0.), kappa(0.),
	 chi2(0.), normchi2(0), d0(-999.), dz(-999.), charge(-999) {};
    float pt;
    float px;
    float py;
    float pz;
    float eta;
    float phi;
    float kappa;
    float chi2;
    float normchi2;
    float d0;
    float dz;
    int charge;


  };
  TrackerValidationVariables();
  TrackerValidationVariables(const edm::EventSetup&, const edm::ParameterSet&);
  ~TrackerValidationVariables();
  void fillHitQuantities(const edm::Event&, std::vector<AVHitStruct> & v_avhitout );
  void fillTrackQuantities(const edm::Event&, std::vector<AVTrackStruct> & v_avtrackout );
 private:  
  const edm::ParameterSet conf_;
  edm::ESHandle<TrackerGeometry> tkgeom;
  //edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  double fBfield;
};
#endif
