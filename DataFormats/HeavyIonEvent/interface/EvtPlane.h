//
// $Id: EvtPlane.h,v 1.2 2008/07/11 10:05:00 sergeant Exp $
//

#ifndef DataFormats_EvtPlane_h
#define DataFormats_EvtPlane_h


namespace reco { class EvtPlane {
public:
  EvtPlane(double planeA=0);
  virtual ~EvtPlane();

  double    EvtPlaneAngle()          const { return EvtPlaneAngle_; } 

private:

  double    EvtPlaneAngle_  ;

};
}

#endif 


