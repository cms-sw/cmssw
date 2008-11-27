/** \file
 *
 *  $Date: 2008/02/20 13:30:04 $
 *  $Revision: 1.12 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


const int FEDNumbering::MAXFEDID = 1023; // 10 bits

const int FEDNumbering::MINSiPixelFEDID = 0;
const int FEDNumbering::MAXSiPixelFEDID = 39;

const int FEDNumbering::MINSiStripFEDID = 50;
const int FEDNumbering::MAXSiStripFEDID = 489;

const int FEDNumbering::MINPreShowerFEDID = 520;
const int FEDNumbering::MAXPreShowerFEDID = 575;

const int FEDNumbering::MINECALFEDID = 600;
const int FEDNumbering::MAXECALFEDID = 670;
  
const int FEDNumbering::MINHCALFEDID = 700;
const int FEDNumbering::MAXHCALFEDID = 731;

const int FEDNumbering::MINLUMISCALERSFEDID = 735;
const int FEDNumbering::MAXLUMISCALERSFEDID = 735;

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
  
const int FEDNumbering::MINTriggerGTPFEDID = 812;
const int FEDNumbering::MAXTriggerGTPFEDID = 813;
const int FEDNumbering::MINTriggerEGTPFEDID = 814;
const int FEDNumbering::MAXTriggerEGTPFEDID = 815;
const int FEDNumbering::MINTriggerGCTFEDID = 745;
const int FEDNumbering::MAXTriggerGCTFEDID = 749;

const int FEDNumbering::MINTriggerLTCFEDID = 816;
const int FEDNumbering::MAXTriggerLTCFEDID = 824;
const int FEDNumbering::MINTriggerLTCmtccFEDID = 815;
const int FEDNumbering::MAXTriggerLTCmtccFEDID = 815;
const int FEDNumbering::MINTriggerLTCTriggerFEDID = 816;
const int FEDNumbering::MAXTriggerLTCTriggerFEDID = 816;
const int FEDNumbering::MINTriggerLTCHCALFEDID = 817;
const int FEDNumbering::MAXTriggerLTCHCALFEDID = 817;
const int FEDNumbering::MINTriggerLTCSiStripFEDID = 818;
const int FEDNumbering::MAXTriggerLTCSiStripFEDID = 818;
const int FEDNumbering::MINTriggerLTCECALFEDID = 819;
const int FEDNumbering::MAXTriggerLTCECALFEDID = 819;
const int FEDNumbering::MINTriggerLTCTotemCastorFEDID = 820;
const int FEDNumbering::MAXTriggerLTCTotemCastorFEDID = 820;
const int FEDNumbering::MINTriggerLTCRPCFEDID = 821;
const int FEDNumbering::MAXTriggerLTCRPCFEDID = 821;
const int FEDNumbering::MINTriggerLTCCSCFEDID = 822;
const int FEDNumbering::MAXTriggerLTCCSCFEDID = 822;
const int FEDNumbering::MINTriggerLTCDTFEDID = 823;
const int FEDNumbering::MAXTriggerLTCDTFEDID = 823;
const int FEDNumbering::MINTriggerLTCSiPixelFEDID = 824;
const int FEDNumbering::MAXTriggerLTCSiPixelFEDID = 824;

const int FEDNumbering::MINCSCDDUFEDID = 830;
const int FEDNumbering::MAXCSCDDUFEDID = 869;  
const int FEDNumbering::MINCSCContingencyFEDID = 880;
const int FEDNumbering::MAXCSCContingencyFEDID = 887;  
const int FEDNumbering::MINCSCTFSPFEDID = 890;
const int FEDNumbering::MAXCSCTFSPFEDID = 901;  

const int FEDNumbering::MINDAQeFEDFEDID = 902;
const int FEDNumbering::MAXDAQeFEDFEDID = 931;  


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

pair<int,int> FEDNumbering::getLumiScalersFEDIds(){

  return pair<int,int> (MINLUMISCALERSFEDID, MAXLUMISCALERSFEDID);

}

pair<int,int> FEDNumbering::getTriggerGTPFEDIds(){

  return pair<int,int> (MINTriggerGTPFEDID, MAXTriggerGTPFEDID);

}


pair<int,int> FEDNumbering::getTriggerEGTPFEDIds(){

  return pair<int,int> (MINTriggerEGTPFEDID, MAXTriggerEGTPFEDID);

}


pair<int,int> FEDNumbering::getTriggerGCTFEDIds(){

  return pair<int,int> (MINTriggerGCTFEDID, MAXTriggerGCTFEDID);

}


pair<int,int> FEDNumbering::getTriggerLTCmtccFEDIds(){

  return pair<int,int> (MINTriggerLTCmtccFEDID, MAXTriggerLTCmtccFEDID);


}
pair<int,int> FEDNumbering::getTriggerLTCFEDIds(){

  return pair<int,int> (MINTriggerLTCFEDID, MAXTriggerLTCFEDID);

}


pair<int, int> FEDNumbering::getTriggerLTCTriggerFEDID(){

return pair<int,int>(MINTriggerLTCTriggerFEDID, MAXTriggerLTCTriggerFEDID);

}

pair<int, int> FEDNumbering::getTriggerLTCHCALFEDID(){

  return pair<int,int>(MINTriggerLTCHCALFEDID, MAXTriggerLTCHCALFEDID);

}

pair<int, int> FEDNumbering::getTriggerLTCSiStripFEDID(){

  return pair<int,int>(MINTriggerLTCSiStripFEDID, MAXTriggerLTCSiStripFEDID);

}

pair<int, int> FEDNumbering::getTriggerLTCECALFEDID(){

return pair<int,int>(MINTriggerLTCECALFEDID, MAXTriggerLTCECALFEDID);

}

pair<int, int> FEDNumbering::getTriggerLTCTotemCastorFEDID(){

  return pair<int,int>(MINTriggerLTCTotemCastorFEDID, MAXTriggerLTCTotemCastorFEDID);

}

pair<int, int> FEDNumbering::getTriggerLTCRPCFEDID(){

return pair<int,int>(MINTriggerLTCRPCFEDID, MAXTriggerLTCRPCFEDID);

}
pair<int, int> FEDNumbering::getTriggerLTCCSCFEDID(){

return pair<int,int>(MINTriggerLTCCSCFEDID, MAXTriggerLTCCSCFEDID);

}
pair<int, int> FEDNumbering::getTriggerLTCDTFEDID(){

return pair<int,int>(MINTriggerLTCDTFEDID, MAXTriggerLTCDTFEDID);

}
pair<int, int> FEDNumbering::getTriggerLTCSiPixelFEDID(){

return pair<int,int>(MINTriggerLTCSiPixelFEDID, MAXTriggerLTCSiPixelFEDID);

}

pair<int, int> FEDNumbering::getCSCDDUFEDIds(){

return pair<int,int>(MINCSCDDUFEDID, MAXCSCDDUFEDID);  

}

pair<int, int> FEDNumbering::getCSCContingencyFEDIds(){

return pair<int,int>(MINCSCContingencyFEDID, MAXCSCContingencyFEDID);  

}

pair<int, int> FEDNumbering::getCSCTFSPFEDIds(){

return pair<int,int>(MINCSCTFSPFEDID, MAXCSCTFSPFEDID);  

}

pair<int, int> FEDNumbering::getDAQeFEDFEDIds(){

return pair<int,int>(MINDAQeFEDFEDID, MAXDAQeFEDFEDID);  

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
  for(i=getLumiScalersFEDIds().first; i<=getLumiScalersFEDIds().second; i++)
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
  for(i=getTriggerGCTFEDIds().first; i<=getTriggerGCTFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "TriggerGCT";
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
  for(i=getCSCDDUFEDIds().first; i<=getCSCDDUFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "CSCDDU";
    }
  for(i=getCSCContingencyFEDIds().first; i<=getCSCContingencyFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "CSCContingency";
    }
  for(i=getCSCTFSPFEDIds().first; i<=getCSCTFSPFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "CSCTFSP";
    }
  for(i=getDAQeFEDFEDIds().first; i<=getDAQeFEDFEDIds().second; i++)
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
