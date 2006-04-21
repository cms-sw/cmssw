#ifndef ZTR_TRunInfo
#define ZTR_TRunInfo
#include "TObject.h"
#include "TStringLong.h"

enum run_types { data=1, data_scan, calibration, laser_ramp,
		 pedestal,  tempdark,
		 check_gains, check_LV, check_HV, PNsTPLinearity, playback };

class TRunInfo : public TObject {

private:
  Int_t fRunNum;              //Run number
  Int_t fRunType;             //Type of run
  Int_t fNTowersMax;          //number max of towers in the RO
  //Roses
  Int_t fNMod;                //number of ROSE modules
  Int_t fNChMod;              //number of channels per ROSE
  Int_t fROSEMode;            //daq=0, debug=1
  Int_t fFrameLength;         //number of samples
  //PNs
  Int_t fNPNs;                //number of PNs
  Int_t fFrameLengthPN;       //number of samples for PNs

  Int_t fSoftVersion;

  TStringLong fInfo1;          //run info
  TStringLong fInfo2;          //run info

  void   Init();

public:


  TRunInfo();
  virtual ~TRunInfo() {}

  Int_t  GetRunNum()      const { return fRunNum; }
  Int_t  GetRunType()     const { return fRunType; }
  Int_t  GetNTowersMax()  const { return fNTowersMax; }

  Int_t  GetNMod()        const { return fNMod; }
  Int_t  GetNChMod()      const { return fNChMod; }
  Int_t  GetROSEMode()    const { return fROSEMode; }
  Int_t  GetFrameLength() const { return fFrameLength; }

  Int_t  GetNPNs()        const { return fNPNs; }
  Int_t  GetFrameLengthPN() const { return fFrameLengthPN; }

  Int_t  GetSoftVersion() const { return fSoftVersion; }

  void   Set(Int_t iv[] );
  void   SetNTowMax(Int_t n );
  void   SetInfo1( Char_t* fi ) { fInfo1.Append( fi ); }
  const char*  GetInfo1() const {  return fInfo1.Data(); }
  void   SetInfo2( Char_t* fi ) { fInfo2.Append( fi ); }
  const char*  GetInfo2() const {  return fInfo2.Data(); }
  virtual void   Print(const char *opt=0)     const;
 
  ClassDef(TRunInfo,2)  //info run filled at SOR
};


// Do not forget to define the global variables so that rootcint
// produces the necessary stub
R__EXTERN const char *gRawRootVersion;

#endif
