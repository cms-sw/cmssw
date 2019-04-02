/** \file
 *
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <cassert>


using namespace std;

namespace {

  constexpr std::array<bool, FEDNumbering::FEDNumbering::MAXFEDID+1> initIn() {
    std::array<bool, FEDNumbering::MAXFEDID+1> in ={{false}};
    
    int i = 0;
    for(i=0; i< FEDNumbering::lastFEDId(); i++)
      in[i] = false;
    for(i=FEDNumbering::MINSiPixelFEDID; i<=FEDNumbering::MAXSiPixelFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINSiStripFEDID; i<=FEDNumbering::MAXSiStripFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINPreShowerFEDID; i<=FEDNumbering::MAXPreShowerFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINECALFEDID; i<=FEDNumbering::MAXECALFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCASTORFEDID; i<=FEDNumbering::MAXCASTORFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINLUMISCALERSFEDID; i<=FEDNumbering::MAXLUMISCALERSFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCSCFEDID; i<=FEDNumbering::MAXCSCFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCSCTFFEDID; i<=FEDNumbering::MAXCSCTFFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINDTFEDID; i<=FEDNumbering::MAXDTFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINDTTFFEDID; i<=FEDNumbering::MAXDTTFFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINRPCFEDID; i<=FEDNumbering::MAXRPCFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerGTPFEDID; i<=FEDNumbering::MAXTriggerGTPFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerEGTPFEDID; i<=FEDNumbering::MAXTriggerEGTPFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerGCTFEDID; i<=FEDNumbering::MAXTriggerGCTFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerLTCFEDID; i<=FEDNumbering::MAXTriggerLTCFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerLTCmtccFEDID; i<=FEDNumbering::MAXTriggerLTCmtccFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCSCDDUFEDID; i<=FEDNumbering::MAXCSCDDUFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCSCContingencyFEDID; i<=FEDNumbering::MAXCSCContingencyFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINCSCTFSPFEDID; i<=FEDNumbering::MAXCSCTFSPFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINDAQeFEDFEDID; i<=FEDNumbering::MAXDAQeFEDFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINDAQmFEDFEDID; i<=FEDNumbering::MAXDAQmFEDFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTCDSuTCAFEDID; i<=FEDNumbering::MAXTCDSuTCAFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINHCALuTCAFEDID; i<=FEDNumbering::MAXHCALuTCAFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINSiPixeluTCAFEDID; i<=FEDNumbering::MAXSiPixeluTCAFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINDTUROSFEDID; i<=FEDNumbering::MAXDTUROSFEDID; i++)
      {
        in[i] = true;
      }
    for(i=FEDNumbering::MINTriggerUpgradeFEDID; i<=FEDNumbering::MAXTriggerUpgradeFEDID; i++)
      {
        in[i] = true;
      }
    return in;
  }

  constexpr std::array<int, FEDNumbering::MAXFEDID+1> initFromIndex() {
    std::array<int, FEDNumbering::MAXFEDID+1> fromIndex= {{0}};
    int i = 0;
    int index = 0;
    for(i=0; i< FEDNumbering::lastFEDId(); i++)
      fromIndex[i] = index; //empty
    ++index;
    for(i=FEDNumbering::MINSiPixelFEDID; i<=FEDNumbering::MAXSiPixelFEDID; i++)
      {
        fromIndex[i] = index; //"SiPixel";
      }
    ++index;
    for(i=FEDNumbering::MINSiStripFEDID; i<=FEDNumbering::MAXSiStripFEDID; i++)
      {
        fromIndex[i] = index; //"SiStrip";
      }
    ++index;
    for(i=FEDNumbering::MINPreShowerFEDID; i<=FEDNumbering::MAXPreShowerFEDID; i++)
      {
        fromIndex[i] = index; //"PreShower";
      }
    ++index;
    for(i=FEDNumbering::MINECALFEDID; i<=FEDNumbering::MAXECALFEDID; i++)
      {
        fromIndex[i] = index; //"Ecal";
      }
    ++index;
    for(i=FEDNumbering::MINCASTORFEDID; i<=FEDNumbering::MAXCASTORFEDID; i++)
      {
        fromIndex[i] = index; //"Castor";
      }
    ++index;
    for(i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
      {
        fromIndex[i] = index; //"Hcal";
      }
    ++index;
    for(i=FEDNumbering::MINLUMISCALERSFEDID; i<=FEDNumbering::MAXLUMISCALERSFEDID; i++)
      {
        fromIndex[i] = index; //"LumiScalers";
      }
    ++index;
    for(i=FEDNumbering::MINCSCFEDID; i<=FEDNumbering::MAXCSCFEDID; i++)
      {
        fromIndex[i] = index; //"CSC";
      }
    ++index;
    for(i=FEDNumbering::MINCSCTFFEDID; i<=FEDNumbering::MAXCSCTFFEDID; i++)
      {
        fromIndex[i] = index; //"CSCTF";
      }
    ++index;
    for(i=FEDNumbering::MINDTFEDID; i<=FEDNumbering::MAXDTFEDID; i++)
      {
        fromIndex[i] = index; //"DT";
      }
    ++index;
    for(i=FEDNumbering::MINDTTFFEDID; i<=FEDNumbering::MAXDTTFFEDID; i++)
      {
        fromIndex[i] = index; //"DTTF";
      }
    ++index;
    for(i=FEDNumbering::MINRPCFEDID; i<=FEDNumbering::MAXRPCFEDID; i++)
      {
        fromIndex[i] = index; //"RPC";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerGTPFEDID; i<=FEDNumbering::MAXTriggerGTPFEDID; i++)
      {
        fromIndex[i] = index; //"TriggerGTP";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerEGTPFEDID; i<=FEDNumbering::MAXTriggerEGTPFEDID; i++)
      {
        fromIndex[i] = index; //"TriggerEGTP";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerGCTFEDID; i<=FEDNumbering::MAXTriggerGCTFEDID; i++)
      {
        fromIndex[i] = index; //"TriggerGCT";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerLTCFEDID; i<=FEDNumbering::MAXTriggerLTCFEDID; i++)
      {
        fromIndex[i] = index; //"TriggerLTC";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerLTCmtccFEDID; i<=FEDNumbering::MAXTriggerLTCmtccFEDID; i++)
      {
        fromIndex[i] = index; //"TriggerLTCmtcc";
      }
    ++index;
    for(i=FEDNumbering::MINCSCDDUFEDID; i<=FEDNumbering::MAXCSCDDUFEDID; i++)
      {
        fromIndex[i] = index; //"CSCDDU";
      }
    ++index;
    for(i=FEDNumbering::MINCSCContingencyFEDID; i<=FEDNumbering::MAXCSCContingencyFEDID; i++)
      {
        fromIndex[i] = index; //"CSCContingency";
      }
    ++index;
    for(i=FEDNumbering::MINCSCTFSPFEDID; i<=FEDNumbering::MAXCSCTFSPFEDID; i++)
      {
        fromIndex[i] = index; //"CSCTFSP";
      }
    ++index;
    for(i=FEDNumbering::MINDAQeFEDFEDID; i<=FEDNumbering::MAXDAQeFEDFEDID; i++)
      {
        fromIndex[i] = index; //"DAQ";
      }
    //++index; same name so no need for new index
    for(i=FEDNumbering::MINDAQmFEDFEDID; i<=FEDNumbering::MAXDAQmFEDFEDID; i++)
      {
        fromIndex[i] = index; //"DAQ";
      }
    ++index;
    for(i=FEDNumbering::MINTCDSuTCAFEDID; i<=FEDNumbering::MAXTCDSuTCAFEDID; i++)
      {
        fromIndex[i] = index; //"TCDS";
      }
    ++index;
    for(i=FEDNumbering::MINHCALuTCAFEDID; i<=FEDNumbering::MAXHCALuTCAFEDID; i++)
      {
        fromIndex[i] = index; //"Hcal";
      }
    ++index;
    for(i=FEDNumbering::MINSiPixeluTCAFEDID; i<=FEDNumbering::MAXSiPixeluTCAFEDID; i++)
      {
        fromIndex[i] = index; //"SiPixel";
      }
    ++index;
    for(i=FEDNumbering::MINDTUROSFEDID; i<=FEDNumbering::MAXDTUROSFEDID; i++)
      {
        fromIndex[i] = index; //"DTUROS";
      }
    ++index;
    for(i=FEDNumbering::MINTriggerUpgradeFEDID; i<=FEDNumbering::MAXTriggerUpgradeFEDID; i++)
      {
        fromIndex[i] = index; //"L1T";
      }
    return fromIndex;
  }


  std::array<std::string, 27> initFromString() {
    std::array<std::string, 27> fromString;
    int index = 0;
    fromString[index] = "";
    ++index;
    fromString[index] = "SiPixel";
    ++index;
    fromString[index] = "SiStrip";
    ++index;
    fromString[index] = "PreShower";
    ++index;
    fromString[index] = "Ecal";
    ++index;
    fromString[index] = "Castor";
    ++index;
    fromString[index] = "Hcal";
    ++index;
    fromString[index] = "LumiScalers";
    ++index;
    fromString[index] = "CSC";
    ++index;
    fromString[index] = "CSCTF";
    ++index;
    fromString[index] = "DT";
    ++index;
    fromString[index] = "DTTF";
    ++index;
    fromString[index] = "RPC";
    ++index;
    fromString[index] = "TriggerGTP";
    ++index;
    fromString[index] = "TriggerEGTP";
    ++index;
    fromString[index] = "TriggerGCT";
    ++index;
    fromString[index] = "TriggerLTC";
    ++index;
    fromString[index] = "TriggerLTCmtcc";
    ++index;
    fromString[index] = "CSCDDU";
    ++index;
    fromString[index] = "CSCContingency";
    ++index;
    fromString[index] = "CSCTFSP";
    ++index;
    fromString[index] = "DAQ";
    ++index;
    fromString[index] = "TCDS";
    ++index;
    fromString[index] = "Hcal";
    ++index;
    fromString[index] = "SiPixel";
    ++index;
    fromString[index] = "DTUROS";
    ++index;
    fromString[index] = "L1T";

    assert(index+1 == fromString.size());
    return fromString;
  }

  constexpr std::array<bool, FEDNumbering::MAXFEDID+1> in_ = initIn();

  constexpr std::array<int, FEDNumbering::MAXFEDID+1> fromIndex_ = initFromIndex();

  const std::array<string, 27> fromString_ = initFromString();



}

bool FEDNumbering::inRange(int i)
{
  return in_[i];
}
bool FEDNumbering::inRangeNoGT(int i)
{
  if((i>=MINTriggerGTPFEDID && i<=MAXTriggerGTPFEDID) || (i>=MINTriggerEGTPFEDID && i<=MAXTriggerEGTPFEDID)) return false;
  return in_[i];
}

string const &FEDNumbering::fromDet(int i)
{
  return fromString_[fromIndex_[i]];
}
