
//
// $Id: EvtPlane.h,v 1.2 2009/08/17 18:08:14 yilmaz Exp $
//

#ifndef DataFormats_EvtPlane_h
#define DataFormats_EvtPlane_h

#include <vector>
#include <string>

namespace reco { class EvtPlane {
public:
   EvtPlane(double planeA=0,double sumSin=0, double sumCos=0,  std::string label="");
  virtual ~EvtPlane();

  std::string label()   const { return label_; }
  double      angle()   const { return angle_; }
  double      sumSin()  const { return sumSin_;}
  double      sumCos()  const { return sumCos_;}
 

private:

  std::string   label_;
  double        angle_  ;
  double        sumSin_;
  double        sumCos_;


};

 typedef std::vector<EvtPlane> EvtPlaneCollection;

}

#endif 






