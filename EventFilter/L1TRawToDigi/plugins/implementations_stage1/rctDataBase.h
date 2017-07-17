#ifndef _rctDATABASE_h
#define _rctDATABASE_h

#include <iostream>
#include <stdexcept>
#include <vector>
#include <stdint.h>

namespace l1t{
  class rctDataBase {
  public:
    enum rctObjectType {
      RCEt,
      RCTau,
      RCOf,
      HFEt,
      HFFg,
      IEEt,
      IEReg,
      IECard,
      NEEt,
      NEReg,
      NECard,
      RCHad,
      nObjects};

  private:

    int RCEt_start[7][2];   //card, region
    int RCTau_start[7][2];  //card, region
    int RCOf_start[7][2];   //card, region
    int HFEt_start[8];      //region
    int HFFg_start[8];      //region
    int IEEt_start[4];      //candidate
    int IEReg_start[4];     //candidate
    int IECard_start[4];     //candidate
    int NEEt_start[4];      //candidate
    int NEReg_start[4];     //candidate
    int NECard_start[4];     //candidate
    int RCHad_start[7][2];  //card, region

    int length[nObjects];

    int link[nObjects];
    int indexfromMP7toRCT[36];
    int indexfromoRSCtoMP7[36];
    

  public:
    rctDataBase();
    ~rctDataBase(){};

    int GetLength(rctObjectType t)
    {
      return length[t];
    }

    int GetLink(rctObjectType t)
    {
      return link[t];
    }

    void GetLinkRCT(int linkMP7,unsigned int &RCTcrate, bool &RCTeven){
        int oRSClink=indexfromMP7toRCT[linkMP7];
        RCTcrate=(int)(oRSClink/2);
        if (oRSClink%2==0) RCTeven=true;
        else RCTeven=false;
    }
    void GetLinkMP7(unsigned int RCTcrate, bool RCTeven, int &linkMP7){
        linkMP7=indexfromoRSCtoMP7[RCTcrate*2+(1-(int)RCTeven)];
    }

    int GetIndices(rctObjectType t, int firstindex, int secondindex = -1)
    {
      switch (t){
      case RCEt:
	return RCEt_start[firstindex][secondindex];
      case RCTau:
	return RCTau_start[firstindex][secondindex];
      case RCOf:
	return RCOf_start[firstindex][secondindex];
      case HFEt:
	return HFEt_start[firstindex];
      case HFFg:
	return HFFg_start[firstindex];
      case IEEt:
	return IEEt_start[firstindex];
      case IEReg:
	return IEReg_start[firstindex];
      case IECard:
	return IECard_start[firstindex];
      case NEEt:
	return NEEt_start[firstindex];
      case NEReg:
	return NEReg_start[firstindex];
      case NECard:
	return NECard_start[firstindex];
      case RCHad:
	return RCHad_start[firstindex][secondindex];
      default:
	return -1;
      }
    }
  };
}
#endif
