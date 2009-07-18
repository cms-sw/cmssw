/** \file
 *
 *  $Date: 2009/02/10 14:24:18 $
 *  $Revision: 1.15 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"



using namespace std;


bool FEDNumbering::init_ = true;
bool *FEDNumbering::in_ = new bool[1024];

vector<string> FEDNumbering::from_(1024,"");

pair<int,int> FEDNumbering::getSiStripFEDIds(){

  return pair<int,int> (MINSiStripFEDID, MAXSiStripFEDID);

}


pair<int,int> FEDNumbering::getSiPixelFEDIds(){

  return pair<int,int> (MINSiPixelFEDID, MAXSiPixelFEDID);

}


pair<int,int> FEDNumbering::getDTFEDIds(){

  return pair<int,int> (MINDTFEDID, MAXDTFEDID);

}

pair<int,int> FEDNumbering::getCSCFEDIds(){

  return pair<int,int> (MINCSCFEDID, MAXCSCFEDID);

}
pair<int,int> FEDNumbering::getCSCTFFEDIds(){

  return pair<int,int> (MINCSCTFFEDID, MAXCSCTFFEDID);

}

pair<int,int> FEDNumbering::getRPCFEDIds(){

  return pair<int,int> (MINRPCFEDID, MAXRPCFEDID);

}

pair<int,int> FEDNumbering::getEcalFEDIds(){

  return pair<int,int> (MINECALFEDID, MAXECALFEDID);

}

pair<int,int> FEDNumbering::getHcalFEDIds(){

  return pair<int,int> (MINHCALFEDID, MAXHCALFEDID);

}

pair<int,int> FEDNumbering::getTriggerGTPFEDIds(){

  return pair<int,int> (MINTriggerGTPFEDID, MAXTriggerGTPFEDID);

}


pair<int,int> FEDNumbering::getTriggerEGTPFEDIds(){

  return pair<int,int> (MINTriggerEGTPFEDID, MAXTriggerEGTPFEDID);

}

int FEDNumbering::lastFEDId(){
  return MAXFEDID;
}

void FEDNumbering::init()
{
  int i = 0;
  for(i=0; i< lastFEDId(); i++)
    in_[i] = false;
  for(i=getSiPixelFEDIds().first; i<=getSiPixelFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "SiPixel";
    }
  for(i=getSiStripFEDIds().first; i<=getSiStripFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "SiStrip";
    }
  for(i=MINPreShowerFEDID; i<=MAXPreShowerFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "PreShower";
    }
  for(i=getEcalFEDIds().first; i<=getEcalFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "Ecal";
    }
  for(i=MINCASTORFEDID; i<=MAXCASTORFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Castor";
    }
  for(i=getHcalFEDIds().first; i<=getHcalFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "Hcal";
    }
  for(i=MINLUMISCALERSFEDID; i<=MAXLUMISCALERSFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "LumiScalers";
    }
  for(i=getCSCFEDIds().first; i<=getCSCFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "CSC";
    }
  for(i=getCSCTFFEDIds().first; i<=getCSCTFFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "CSCTF";
    }
  for(i=getDTFEDIds().first; i<=getDTFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "DT";
    }
  for(i=MINDTTFFEDID; i<=MAXDTTFFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DTTF";
    }
  for(i=getRPCFEDIds().first; i<=getRPCFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "RPC";
    }
  for(i=getTriggerGTPFEDIds().first; i<=getTriggerGTPFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerGTP";
    }
  for(i=getTriggerEGTPFEDIds().first; i<=getTriggerEGTPFEDIds().second; i++)
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
      from_[i] = "DAQeFED";
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
