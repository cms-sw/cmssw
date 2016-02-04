#ifndef ALIGN_EVENT
#define ALIGN_EVENT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
// Description of the Alignment event classes for COCOA                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TObject.h"
#include "TClonesArray.h"
#include "TString.h"

//-----------------------------------------------------------------------------

class AliDaqEventHeader {

private:
  Int_t   fRunNum;        // Run number
  Int_t   fEvtNum;        // Event number
  Int_t   fTimeStamp;     // Event time stamp

public:
  AliDaqEventHeader() : fRunNum(0), fEvtNum(0), fTimeStamp(0) { }
  virtual ~AliDaqEventHeader() { }
  Int_t  GetRunNum()    const { return fRunNum; }
  Int_t  GetEvtNum()    const { return fEvtNum; }
  Int_t  GetTimeStamp() const { return fTimeStamp; }
  void   SetRunEvt(int run, int event) {fRunNum=run; fEvtNum=event;}
  void   DumpIt();

  ClassDef(AliDaqEventHeader,1)  
};

//-----------------------------------------------------------------------------

class AliDaqPosition2D : public TObject {

private:
  Float_t   fX;             // X position
  Float_t   fY;             // Y position
  Float_t   fX_error;       // uncertainty in X position
  Float_t   fY_error;       // uncertainty in Y position
  TString   fID;            // ID of sensor
  Int_t     fTimeStamp;     // position time stamp

public:
  AliDaqPosition2D() : fX(0), fY(0), fX_error(0), fY_error(0),
                 fID(""), fTimeStamp(0) { }
  virtual ~AliDaqPosition2D() { }
  Float_t  GetX()         const { return fX; }
  Float_t  GetY()         const { return fY; }
  Float_t  GetXerror()    const { return fX_error; }
  Float_t  GetYerror()    const { return fY_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     DumpIt(TString Name);

  ClassDef(AliDaqPosition2D,1)  
};

//-----------------------------------------------------------------------------

class AliDaqPositionCOPS : public TObject {

private:
  Float_t   fDCOPS_up;            // up position
  Float_t   fDCOPS_down;          // down position
  Float_t   fDCOPS_left;          // left position
  Float_t   fDCOPS_right;         // right position
  Float_t   fDCOPS_up_error;      // up position uncertainty
  Float_t   fDCOPS_down_error;    // down position uncertainty
  Float_t   fDCOPS_left_error;    // left position uncertainty
  Float_t   fDCOPS_right_error;   // right position uncertainty
  TString   fID;                  // ID of sensor
  Int_t     fTimeStamp;           // position time stamp

public:
  AliDaqPositionCOPS() : fDCOPS_up(0),         fDCOPS_down(0), 
                   fDCOPS_left(0),       fDCOPS_right(0),
                   fDCOPS_up_error(0),   fDCOPS_down_error(0),
                   fDCOPS_left_error(0), fDCOPS_right_error(0),
                   fID(""),              fTimeStamp(0) { }
  virtual ~AliDaqPositionCOPS() { }
  Float_t  GetUp()    const { return fDCOPS_up; }
  Float_t  GetDown()  const { return fDCOPS_down; }
  Float_t  GetLeft()  const { return fDCOPS_left; }
  Float_t  GetRight() const { return fDCOPS_right; }
  Float_t  GetUpError()    const { return fDCOPS_up_error; }
  Float_t  GetDownError()  const { return fDCOPS_down_error; }
  Float_t  GetLeftError()  const { return fDCOPS_left_error; }
  Float_t  GetRightError() const { return fDCOPS_right_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp()  const { return fTimeStamp; }
  void     DumpIt(TString Name);

  ClassDef(AliDaqPositionCOPS,1)  
};

//-----------------------------------------------------------------------------

class AliDaqTilt : public TObject {

private:
  Float_t fTilt;          // Tilt, or inclination
  Float_t fTilt_error;    // uncertainty in tilt
  TString fID;            // ID of sensor
  Int_t   fTimeStamp;

public:
  AliDaqTilt() : fTilt(0), fTilt_error(0), fID(""), fTimeStamp(0) { }
  virtual ~AliDaqTilt() { }
  Float_t  GetTilt()      const { return fTilt; }
  Float_t  GetTiltError() const { return fTilt_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     DumpIt(TString Name);

  ClassDef(AliDaqTilt,1)
};

//-----------------------------------------------------------------------------
class AliDaqDistance : public TObject {

private:
  Float_t fDistance;
  Float_t fDistance_error;
  TString fID;             // ID of sensor
  Int_t   fTimeStamp;

public:
  AliDaqDistance() : fDistance(0), fDistance_error(0), 
               fID(""), fTimeStamp(0) { }
  virtual ~AliDaqDistance() { }
  Float_t  GetDistance()      const { return fDistance; }
  Float_t  GetDistanceError() const { return fDistance_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     DumpIt(TString Name);

  ClassDef(AliDaqDistance,1)
};

//-----------------------------------------------------------------------------

class AliDaqTemperature : public TObject {

private:
  Float_t fTemperature;
  Float_t fTemperature_error;
  TString fID;                // ID of sensor
  Int_t   fTimeStamp;

public:
  AliDaqTemperature() : fTemperature(0), fTemperature_error(0),
                  fID(""), fTimeStamp(0) { }
  virtual ~AliDaqTemperature() { }
  Float_t  GetTemperature()      const { return fTemperature; }
  Float_t  GetTemperatureError() const { return fTemperature_error; }
  TString  GetID()  { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     DumpIt(TString Name);

  ClassDef(AliDaqTemperature,1)
};

//-----------------------------------------------------------------------------
class CocoaDaqRootEvent : public TObject {

private:
  AliDaqEventHeader  *Header;
  TClonesArray *Array_PositionCOPS;
  TClonesArray *Array_Position2D;
  TClonesArray *Array_Tilt;
  TClonesArray *Array_Dist;
  TClonesArray *Array_Temp;
  int            numPosCOPS;
  int            numPos2D;
  int            numTilt;
  int            numDist;
  int            numTemp;

public:
  AliDaqEventHeader  * GetHeader() const { return Header;}
  TClonesArray * GetArray_Position2D() const { return Array_Position2D;}
  TClonesArray * GetArray_PositionCOPS() const { return Array_PositionCOPS;}
  TClonesArray * GetArray_Tilt() const { return Array_Tilt;}
  TClonesArray * GetArray_Dist() const { return Array_Dist;}
  TClonesArray * GetArray_Temp() const { return Array_Temp;}
  int GetNumPos2D() const { return numPos2D;}
  int GetNumPosCOPS() const { return numPosCOPS;}
  int GetNumTilt() const { return numTilt;}
  int GetNumDist() const { return numDist;}
  int GetNumTemp() const { return numTemp;}

public:
  CocoaDaqRootEvent();
  virtual ~CocoaDaqRootEvent() { };
  void     SetHeader(int run, int evt) {Header->SetRunEvt(run,evt); }
  void     DumpIt();

  ClassDef(CocoaDaqRootEvent,1)  

 };


#endif
