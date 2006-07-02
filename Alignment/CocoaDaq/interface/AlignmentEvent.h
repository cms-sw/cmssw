#ifndef AlignmentEvent_H
#define AlignmentEvent_H

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

class EventHeader {

private:
  Int_t   fRunNum;        // Run number
  Int_t   fEvtNum;        // Event number
  Int_t   fTimeStamp;     // Event time stamp

public:
  EventHeader() : fRunNum(0), fEvtNum(0), fTimeStamp(0) { }
  virtual ~EventHeader() { }
  Int_t  GetRunNum()    const { return fRunNum; }
  Int_t  GetEvtNum()    const { return fEvtNum; }
  Int_t  GetTimeStamp() const { return fTimeStamp; }
  void   ReadTimeStampFromDB();
  void   SetRunEvt(int run, int event) {fRunNum=run; fEvtNum=event;}
  void   DumpIt();

  ClassDef(EventHeader,1)  
};

//-----------------------------------------------------------------------------

class Position2D : public TObject {

public:  //PAD
  Float_t   fX;             // X position
  Float_t   fY;             // Y position
  Float_t   fX_error;       // uncertainty in X position
  Float_t   fY_error;       // uncertainty in Y position
 public: //PAD
  TString   fID;            // ID of sensor
  Int_t     fTimeStamp;     // position time stamp

public:
  Position2D() : fX(0), fY(0), fX_error(0), fY_error(0),
                 fID(""), fTimeStamp(0) { }
  virtual ~Position2D() { }
  Float_t  GetX()         const { return fX; }
  Float_t  GetY()         const { return fY; }
  Float_t  GetXerror()    const { return fX_error; }
  Float_t  GetYerror()    const { return fY_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     ReadFromDB();
  void     DumpIt(TString Name);

  ClassDef(Position2D,1)  
};

//-----------------------------------------------------------------------------

class Position4x1D : public TObject {

public:  //PAD
  Float_t   fDCOPS_up;            // up position
  Float_t   fDCOPS_down;          // down position
  Float_t   fDCOPS_left;          // left position
  Float_t   fDCOPS_right;         // right position
  Float_t   fDCOPS_up_error;      // up position uncertainty
  Float_t   fDCOPS_down_error;    // down position uncertainty
  Float_t   fDCOPS_left_error;    // left position uncertainty
  Float_t   fDCOPS_right_error;   // right position uncertainty
 public: //PAD
  TString   fID;                  // ID of sensor
  Int_t     fTimeStamp;           // position time stamp

public:
  Position4x1D() : fDCOPS_up(0),         fDCOPS_down(0), 
                   fDCOPS_left(0),       fDCOPS_right(0),
                   fDCOPS_up_error(0),   fDCOPS_down_error(0),
                   fDCOPS_left_error(0), fDCOPS_right_error(0),
                   fID(""),              fTimeStamp(0) { }
  virtual ~Position4x1D() { }
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
  void     ReadFromDB();
  void     DumpIt(TString Name);

  ClassDef(Position4x1D,1)  
};

//-----------------------------------------------------------------------------

class Tilt1D : public TObject {

public:  //PAD
  Float_t fTilt;          // Tilt, or inclination
  Float_t fTilt_error;    // uncertainty in tilt
 public: //PAD
  TString fID;            // ID of sensor
  Int_t   fTimeStamp;

public:
  Tilt1D() : fTilt(0), fTilt_error(0), fID(""), fTimeStamp(0) { }
  virtual ~Tilt1D() { }
  Float_t  GetTilt()      const { return fTilt; }
  Float_t  GetTiltError() const { return fTilt_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     ReadFromDB();
  void     DumpIt(TString Name);

  ClassDef(Tilt1D,1)
};

//-----------------------------------------------------------------------------

class Distance : public TObject {

public:  //PAD
  Float_t fDistance;
  Float_t fDistance_error;
 public: //PAD
  TString fID;             // ID of sensor
  Int_t   fTimeStamp;

public:
  Distance() : fDistance(0), fDistance_error(0), 
               fID(""), fTimeStamp(0) { }
  virtual ~Distance() { }
  Float_t  GetDistance()      const { return fDistance; }
  Float_t  GetDistanceError() const { return fDistance_error; }
  TString  GetID() { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     ReadFromDB();
  void     DumpIt(TString Name);

  ClassDef(Distance,1)
};

//-----------------------------------------------------------------------------

class Temperature : public TObject {

public:  //PAD
  Float_t fTemperature;
  Float_t fTemperature_error;
 public: //PAD
  TString fID;                // ID of sensor
  Int_t   fTimeStamp;

public:
  Temperature() : fTemperature(0), fTemperature_error(0),
                  fID(""), fTimeStamp(0) { }
  virtual ~Temperature() { }
  Float_t  GetTemperature()      const { return fTemperature; }
  Float_t  GetTemperatureError() const { return fTemperature_error; }
  TString  GetID()  { return fID; }
  Int_t    GetTimeStamp() const { return fTimeStamp; }
  void     ReadFromDB();
  void     DumpIt(TString Name);

  ClassDef(Temperature,1)
};

//-----------------------------------------------------------------------------

class AlignmentEvent : public TObject {

private:
  EventHeader  *Header;
  TClonesArray *Array_Position2D;
  TClonesArray *Array_Position4x1D;
  TClonesArray *Array_Tilt1D;
  TClonesArray *Array_Dist;
  TClonesArray *Array_Temp;
  int           numPos2D;
  int           numPos4x1D;
  int           numTilt1D;
  int           numDist;
  int           numTemp;

public:
  EventHeader  * GetHeader() const { return Header;}
  TClonesArray * GetArray_Position2D() const { return Array_Position2D;}
  TClonesArray * GetArray_Position4x1D() const { return Array_Position4x1D;}
  TClonesArray * GetArray_Tilt1D() const { return Array_Tilt1D;}
  TClonesArray * GetArray_Dist() const { return Array_Dist;}
  TClonesArray * GetArray_Temp() const { return Array_Temp;}
  int GetNumPos2D() const { return numPos2D;}
  int GetNumPos4x1D() const { return numPos4x1D;}
  int GetNumTilt1D() const { return numTilt1D;}
  int GetNumDist() const { return numDist;}
  int GetNumTemp() const { return numTemp;}

public:
  AlignmentEvent();
  virtual ~AlignmentEvent() { };
  void     SetHeader(int run, int evt) {Header->SetRunEvt(run,evt); }
  void     ReadEventFromDB();
  void     addPos2D(TString ID);
  void     add4x1D(TString ID);
  void     addTilt1D(TString ID);
  void     addDist(TString ID);
  void     addTemp(TString ID);
  void     DumpIt();

  ClassDef(AlignmentEvent,1)  

 };

//-----------------------------------------------------------------------------

#endif
