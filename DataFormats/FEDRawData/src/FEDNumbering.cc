/** \file
 *
 *  $Date: 2010/03/16 21:05:18 $
 *  $Revision: 1.17 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"



using namespace std;


bool FEDNumbering::init_ = true;
bool *FEDNumbering::in_ = new bool[1024];

vector<string> FEDNumbering::from_(1024,"");

int FEDNumbering::lastFEDId(){
  return MAXFEDID;
}

void FEDNumbering::init()
{
  int i = 0;
  for(i=0; i< lastFEDId(); i++)
    in_[i] = false;
  for(i=MINPreShowerFEDID; i<=MAXPreShowerFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "PreShower";
    }
  for(i=MINCASTORFEDID; i<=MAXCASTORFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "Castor";
    }
  for(i=MINLUMISCALERSFEDID; i<=MAXLUMISCALERSFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "LumiScalers";
    }
  for(i=MINDTTFFEDID; i<=MAXDTTFFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DTTF";
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
  for(i=MINDAQmFEDFEDID; i<=MAXDAQmFEDFEDID; i++)
    {
      in_[i] = true;
      from_[i] = "DAQmFED";
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
