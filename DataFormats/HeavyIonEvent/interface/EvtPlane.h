//
// $Id: EvtPlane.h,v 1.1 2008/07/20 19:18:24 yilmaz Exp $
//

#ifndef DataFormats_EvtPlane_h
#define DataFormats_EvtPlane_h

#include <vector>
#include <string>

namespace reco { class EvtPlane {
public:
   EvtPlane(double planeA=0, std::string label="");
  virtual ~EvtPlane();

  std::string label()   const { return label_; }
  double      angle()   const { return angle_; } 

private:

  std::string   label_;
  double        angle_  ;

};

 typedef std::vector<EvtPlane> EvtPlaneCollection;

}

#endif 


