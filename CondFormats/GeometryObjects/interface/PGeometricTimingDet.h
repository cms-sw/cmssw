#ifndef CondFormats_PGeometricTimingDet_h
#define CondFormats_PGeometricTimingDet_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>

class PGeometricTimingDet{

 public:
  PGeometricTimingDet() { };
  ~PGeometricTimingDet() { };

  struct Item{  
    std::string name_; // save only the name, not the namespace.
    std::string ns_; // save only the name, not the namespace.

    double x_;
    double y_;
    double z_;
    double phi_;
    double rho_;
    // fill as you will but intent is rotation matrix A where first number is row and second number is column
    double a11_, a12_, a13_, a21_, a22_, a23_, a31_, a32_, a33_;
    double params_0,params_1,params_2,params_3,params_4,params_5,params_6,params_7,params_8,params_9,params_10;
    double radLength_;
    double xi_;
    double pixROCRows_;
    double pixROCCols_;
    double pixROCx_;
    double pixROCy_;
    double siliconAPVNum_;
    
    int level_; // goes like 1, 2, 3, 4, 4, 4, 3, 4, 4, 3, 4, 4, 4, 1, 2, 3, etc.
    int shape_;
    //  nav_type _ddd; DO NOT SAVE!
    //  DDName _ddname; DO NOT SAVE!
    int type_;

    int numnt_;
    int nt0_, nt1_, nt2_, nt3_, nt4_, nt5_, nt6_, nt7_, nt8_, nt9_, nt10_;
    
    int geographicalID_; // to be converted to DetId
    bool stereo_;
  
  COND_SERIALIZABLE;
};

  std::vector<Item> pgeomdets_;


 COND_SERIALIZABLE;
};

#endif

