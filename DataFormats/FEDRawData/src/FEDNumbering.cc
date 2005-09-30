/**  
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2005/09/15 09:00:48 $
 *  $Revision: 1.2 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

using namespace std;

const int FEDNumbering::MAXFEDID=1023;


const int FEDNumbering::MINSiPixelFEDID=0;
const int FEDNumbering::MAXSiPixelFEDID=37;

const int FEDNumbering::MINSiStripFEDID=50;
const int FEDNumbering::MAXSiStripFEDID=489;


const int FEDNumbering::MINCSCFEDID=750;
const int FEDNumbering::MAXCSCFEDID=757;

const int FEDNumbering::MINDTFEDID=770;
const int FEDNumbering::MAXDTFEDID=775;

const int FEDNumbering::MINRPCFEDID=790;
const int FEDNumbering::MAXRPCFEDID=795;


const int FEDNumbering::MINPreShowerFEDID=550;
const int FEDNumbering::MAXPreShowerFEDID=596;

const int FEDNumbering::MINECALBarrelFEDID=620;
const int FEDNumbering::MAXECALBarrelFEDID=655;

const int FEDNumbering::MINECALEndcapFEDID=656;
const int FEDNumbering::MAXECALEndcapFEDID=673;


const int FEDNumbering::MINHCALFEDID=700;
const int FEDNumbering::MAXHCALFEDID=731;


const int FEDNumbering::MINTriggerFEDID=813;
const int FEDNumbering::MAXTriggerFEDID=815;


pair<int,int> FEDNumbering::getSiStripFEDIds(){

  return pair<int,int> (MINSiStripFEDID, MAXSiStripFEDID);

}


pair<int,int> FEDNumbering::getSiPixelFEDIds(){

  return pair<int,int> (MINSiPixelFEDID, MAXSiPixelFEDID);

}


pair<int,int> FEDNumbering::getMuBarFEDIds(){

  return pair<int,int> (MINDTFEDID, MAXDTFEDID);

}


pair<int,int> FEDNumbering::getMuEndFEDIds(){

  return pair<int,int> (MINCSCFEDID, MAXCSCFEDID);

}


pair<int,int> FEDNumbering::getMRpcFEDIds(){

  return pair<int,int> (MINRPCFEDID, MAXRPCFEDID);

}


pair<int,int> FEDNumbering::getPreShowerFEDIds(){

  return pair<int,int> (MINPreShowerFEDID, MAXPreShowerFEDID);

}


pair<int,int> FEDNumbering::getEcalBarrelFEDIds(){

  return pair<int,int> (MINECALBarrelFEDID, MAXECALBarrelFEDID);

}

pair<int,int> FEDNumbering::getEcalEndcapFEDIds(){

  return pair<int,int> (MINECALEndcapFEDID, MAXECALEndcapFEDID);

}

pair<int,int> FEDNumbering::getHcalFEDIds(){

  return pair<int,int> (MINHCALFEDID, MAXHCALFEDID);

}

pair<int,int> FEDNumbering::getTriggerFEDIds(){

  return pair<int,int> (MINTriggerFEDID, MAXTriggerFEDID);

}

int FEDNumbering::lastFEDId(){
  return MAXFEDID;
}

