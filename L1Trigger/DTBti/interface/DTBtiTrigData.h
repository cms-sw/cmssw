//-------------------------------------------------
//
/**  \class DTBtiTrigData
 *
 *    DTBtiChip Trigger Data
 *
 *
 *   $Date: 2007/04/27 08:37:37 $
 *   $Revision: 1.2 $
 *
 *   \author C. Grandi, S. Vanini
 *
 *   Modifications: 
 *   SV 29/I/03 : insert trigger Strobe
 */
//
//--------------------------------------------------
#ifndef DT_BTI_TRIG_DATA_H
#define DT_BTI_TRIG_DATA_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/MuonDetId/interface/DTBtiId.h"

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTBtiTrigData : public DTTrigData {

 public:

  //! Constructor
  DTBtiTrigData() {}
  
  //! Destructor 
  ~DTBtiTrigData() {}

  //! Set the parent BTI identifier
  inline void setParent(DTBtiId btiid) {
    _btiid = btiid; 
  }  

  //! Set trigger step
  inline void setStep(int step) {
    _step = step;
  }

  //! Set trigger code
  inline void setCode(int code) {
    _code = code;
  }

  //! Set trigger K parameter
  inline void setK(int k) {
    _Kval = k; 
  }

  //! Set trigger X parameter
  inline void setX(int x) {
    _Xval = x;
  }

  //! Set triggering equation
  inline void setEq(int eq) {
    _eq = eq;
  }

  //! Set trigger strobe
  inline void setStrobe(int str) {
    _str = str;
  }

  //! Set trigger _Keq
  inline void setKeq(int num, float Keq) {
    _Keq[num] = Keq;
  }

  //! Clear
  void clear() {
    _step = 0;
    _eq = 0;
    _code = 0; 
    _Kval = 9999; 
    _Xval = 0;
    _str = -1;
    for(int j=0;j<6;j++)
	_Keq[j]=-1;
  }  

  //! Return chamber identifier
  DTChamberId ChamberId() const {
    return DTChamberId(_btiid.wheel(),_btiid.station(),_btiid.sector()); 
  }

  //! Print
  void print() const;

  //! Return parent BTI identifier
  inline DTBtiId parentId() const { 
    return _btiid; 
  }
  
  //! Return superlayer identifier
  inline DTSuperLayerId SLId() const {
    return _btiid.SLId(); 
  }

  //! Return parent BTI number
  inline int btiNumber() const { 
    return _btiid.bti(); 
  }
  
  //! Return parent BTI superlayer
  inline int btiSL() const { 
    return _btiid.superlayer(); 
  }
  
  //! Return trigger step
  inline int step() const { 
    return _step; 
  }

  //! Return trigger code
  inline int code() const { 
    return _code; 
  }
  
  //! Return trigger K parameter
  inline int K() const { 
    return _Kval; 
  }
  
  //! Return trigger X parameter
  inline int X() const { 
    return _Xval; 
  }

  //! Return triggering equation
  inline int eq() const { 
    return _eq; 
  }

  //! Return trigger strobe
  inline int Strobe() const{ 
    return _str; 
  }

  //! Return triggering K equations
  inline float Keq(int i) const { 
    return _Keq[i]; 
  }

 private:

  // Parent BTI identifier
  DTBtiId _btiid; // this is 5 bytes

  // output values
  myint8 _code;
  int _Kval;
  myint8 _Xval;

  myint8 _step;
  myint8 _eq;
  int _str;
  float _Keq[6];

};

#endif
