/** \file
 *
 *  $Date: 2009/01/23 17:00:31 $
 *  $Revision: 1.14 $
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


pair<int,int> FEDNumbering::getCastorFEDIds(){

  return pair<int,int> (MINCASTORFEDID, MAXCASTORFEDID);

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
  for(i=getCastorFEDIds().first; i<=getCastorFEDIds().second; i++)
    {
      in_[i] = true;
      from_[i] = "Castor";
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
