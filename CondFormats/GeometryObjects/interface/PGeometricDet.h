#ifndef CondFormats_PGeometricDet_h
#define CondFormats_PGeometricDet_h

#include <vector>
#include <string>

class PGeometricDet{

 public:
  PGeometricDet() { };
  ~PGeometricDet() { };

  struct Item{  
    std::string _name; // save only the name, not the namespace.
    std::string _ns; // save only the name, not the namespace.

    double _x;
    double _y;
    double _z;
    double _phi;
    double _rho;
    // fill as you will but intent is rotation matrix A where first number is row and second number is column
    double _a11, _a12, _a13, _a21, _a22, _a23, _a31, _a32, _a33;
    double _params0,_params1,_params2,_params3,_params4,_params5,_params6,_params7,_params8,_params9,_params10;
    double _radLength;
    double _xi;
    double _pixROCRows;
    double _pixROCCols;
    double _pixROCx;
    double _pixROCy;
    double _siliconAPVNum;
    
    int _level; // goes like 1, 2, 3, 4, 4, 4, 3, 4, 4, 3, 4, 4, 4, 1, 2, 3, etc.
    int _shape;
    //  nav_type _ddd; DO NOT SAVE!
    //  DDName _ddname; DO NOT SAVE!
    int _type;

    int _numnt;
    int _nt0, _nt1, _nt2, _nt3, _nt4, _nt5, _nt6 , _nt7, _nt8, _nt9, _nt10;
    
    int _geographicalID; // to be converted to DetId
    bool _stereo;
  };

  std::vector<Item> pgeomdets_;

};

#endif

