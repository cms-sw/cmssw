/** \file
 *
 *  $Date: 2006/05/16 10:33:53 $
 *  $Revision: 1.6 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


const int FEDNumbering::MAXFEDID = 1023; // 10 bits

const int FEDNumbering::MINSiPixelFEDID = 0;
const int FEDNumbering::MAXSiPixelFEDID = 39;

const int FEDNumbering::MINSiStripFEDID = 50;
const int FEDNumbering::MAXSiStripFEDID = 489;


const int FEDNumbering::MINCSCFEDID = 750;
const int FEDNumbering::MAXCSCFEDID = 757;  
const int FEDNumbering::MINCSCTFFEDID = 760;
const int FEDNumbering::MAXCSCTFFEDID = 760;  
  
const int FEDNumbering::MINDTFEDID = 770;
const int FEDNumbering::MAXDTFEDID = 775;
const int FEDNumbering::MINDTTFFEDID = 780;
const int FEDNumbering::MAXDTTFFEDID = 780;
  
const int FEDNumbering::MINRPCFEDID = 790;
const int FEDNumbering::MAXRPCFEDID = 795;


const int FEDNumbering::MINPreShowerFEDID = 550;
const int FEDNumbering::MAXPreShowerFEDID = 596;

const int FEDNumbering::MINECALFEDID = 600;
const int FEDNumbering::MAXECALFEDID = 670;
  
const int FEDNumbering::MINHCALFEDID = 700;
const int FEDNumbering::MAXHCALFEDID = 731;

  
const int FEDNumbering::MINTriggerGTPFEDID = 813;
const int FEDNumbering::MAXTriggerGTPFEDID = 813;
const int FEDNumbering::MINTriggerEGTPFEDID = 814;
const int FEDNumbering::MAXTriggerEGTPFEDID = 814;
const int FEDNumbering::MINTriggerLTCFEDID = 816;
const int FEDNumbering::MAXTriggerLTCFEDID = 823;
const int FEDNumbering::MINTriggerLTCmtccFEDID = 815;
const int FEDNumbering::MAXTriggerLTCmtccFEDID = 815;

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

pair<int,int> FEDNumbering::getDTTFFEDIds(){

  return pair<int,int> (MINDTTFFEDID, MAXDTTFFEDID);

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


pair<int,int> FEDNumbering::getPreShowerFEDIds(){

  return pair<int,int> (MINPreShowerFEDID, MAXPreShowerFEDID);

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

pair<int,int> FEDNumbering::getTriggerLTCmtccFEDIds(){

  return pair<int,int> (MINTriggerLTCmtccFEDID, MAXTriggerLTCmtccFEDID);


}
pair<int,int> FEDNumbering::getTriggerLTCFEDIds(){

  return pair<int,int> (MINTriggerLTCFEDID, MAXTriggerLTCFEDID);

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
  for(i=getPreShowerFEDIds().first; i<=getPreShowerFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "PreShower";
    }
  for(i=getEcalFEDIds().first; i<=getEcalFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "Ecal";
    }
  for(i=getHcalFEDIds().first; i<=getHcalFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "Hcal";
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
  for(i=getDTTFFEDIds().first; i<=getDTTFFEDIds().second; i++)
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
  for(i=getTriggerLTCFEDIds().first; i<=getTriggerLTCFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerLTC";
    }
  for(i=getTriggerLTCmtccFEDIds().first; i<=getTriggerLTCmtccFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerLTCmtcc";
    }
  init_ = false;
}

bool FEDNumbering::inRange(int i) 
{
  if(init_) init();
  return in_[i];
}

string const &FEDNumbering::fromDet(int i) 
{
  if(init_) init();
  return from_[i];
}  
