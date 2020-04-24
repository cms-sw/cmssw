/** \file
 *
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"



using namespace std;


bool FEDNumbering::init_ = true;
bool *FEDNumbering::in_ = new bool[MAXFEDID+1];

vector<string> FEDNumbering::from_(MAXFEDID+1,"");

int FEDNumbering::lastFEDId(){
  return MAXFEDID;
}

void FEDNumbering::init()
{
  int i = 0;
  for(i=0; i< lastFEDId(); i++)
    in_[i] = false;
  for(i=MINSiPixelFEDID; i<=MAXSiPixelFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "SiPixel";
    }
  for(i=MINSiStripFEDID; i<=MAXSiStripFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "SiStrip";
    }
  for(i=MINPreShowerFEDID; i<=MAXPreShowerFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "PreShower";
    }
  for(i=MINECALFEDID; i<=MAXECALFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Ecal";
    }
  for(i=MINCASTORFEDID; i<=MAXCASTORFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Castor";
    }
  for(i=MINHCALFEDID; i<=MAXHCALFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Hcal";
    }
  for(i=MINLUMISCALERSFEDID; i<=MAXLUMISCALERSFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "LumiScalers";
    }
  for(i=MINCSCFEDID; i<=MAXCSCFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "CSC";
    }
  for(i=MINCSCTFFEDID; i<=MAXCSCTFFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "CSCTF";
    }
  for(i=MINDTFEDID; i<=MAXDTFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DT";
    }
  for(i=MINDTTFFEDID; i<=MAXDTTFFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DTTF";
    }
  for(i=MINRPCFEDID; i<=MAXRPCFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "RPC";
    }
  for(i=MINTriggerGTPFEDID; i<=MAXTriggerGTPFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerGTP";
    }
  for(i=MINTriggerEGTPFEDID; i<=MAXTriggerEGTPFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerEGTP";
    }
  for(i=MINTriggerGCTFEDID; i<=MAXTriggerGCTFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerGCT";
    }
  for(i=MINTriggerLTCFEDID; i<=MAXTriggerLTCFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerLTC";
    }
  for(i=MINTriggerLTCmtccFEDID; i<=MAXTriggerLTCmtccFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerLTCmtcc";
    }
  for(i=MINCSCDDUFEDID; i<=MAXCSCDDUFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "CSCDDU";
    }
  for(i=MINCSCContingencyFEDID; i<=MAXCSCContingencyFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "CSCContingency";
    }
  for(i=MINCSCTFSPFEDID; i<=MAXCSCTFSPFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "CSCTFSP";
    }
  for(i=MINDAQeFEDFEDID; i<=MAXDAQeFEDFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DAQ";
    }
  for(i=MINDAQmFEDFEDID; i<=MAXDAQmFEDFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DAQ";
    }
  for(i=MINTCDSuTCAFEDID; i<=MAXTCDSuTCAFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "TCDS";
    }
  for(i=MINHCALuTCAFEDID; i<=MAXHCALuTCAFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Hcal";
    }
  for(i=MINSiPixeluTCAFEDID; i<=MAXSiPixeluTCAFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "SiPixel";
    }
  for(i=MINTriggerUpgradeFEDID; i<=MAXTriggerUpgradeFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "L1T";
    }


  init_ = false;
}

bool FEDNumbering::inRange(int i)
{
  if(init_) init();
  return in_[i];
}
bool FEDNumbering::inRangeNoGT(int i)
{
  if(init_) init();
  if((i>=MINTriggerGTPFEDID && i<=MAXTriggerGTPFEDID) || (i>=MINTriggerEGTPFEDID && i<=MAXTriggerEGTPFEDID)) return false;
  return in_[i];
}

string const &FEDNumbering::fromDet(int i)
{
  if(init_) init();
  return from_[i];
}
