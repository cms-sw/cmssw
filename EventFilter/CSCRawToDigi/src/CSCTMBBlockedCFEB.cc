//_________________________________________________________
//
//  CSCTMBBlockedCFEB July 2010  Alexander Sakharov
//  Unpacks TMB Logic Blocked CFEB Analyzer and stores in CSCTMBBlockedCFEB.h
//_________________________________________________________
//
#include "EventFilter/CSCRawToDigi/interface/CSCTMBBlockedCFEB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

CSCTMBBlockedCFEB::CSCTMBBlockedCFEB(unsigned short *buf,int Line6BCB,int Line6ECB)
{

  size_ = UnpackBlockedCFEB(buf,Line6BCB,Line6ECB);

} ///CSCTMBMiniScope


int CSCTMBBlockedCFEB::UnpackBlockedCFEB(unsigned short *buf,int Line6BCB,int Line6ECB)
{

  if ((Line6ECB-Line6BCB) != 0)
    {

      for (int i=0; i<(Line6ECB-Line6BCB-1); i++)
        {
          BlockedCFEBdata.push_back(buf[Line6BCB+1+i]);
        }

    }

  //print();
  return (Line6ECB-Line6BCB + 1);

} ///UnpackBlockedCFEB

std::vector< std::vector<int> > CSCTMBBlockedCFEB::getSingleCFEBList(int CFEBn) const
{

  std::vector< std::vector<int> > CFEBnByLayers;
  CFEBnByLayers.clear();
  std::vector<int> CFEBnData;
  CFEBnData.clear();
  int idCFEB=-1;

  /// Get 4 words for a particular CFEB (CFEBn)
  for (int i=0; i<(int)getData().size(); ++i)
    {
      idCFEB = (getData()[i] >> 12) & 0x7;
      if (idCFEB==CFEBn)
        {
          CFEBnData.push_back(getData()[i] & 0xFFF);
        }
      idCFEB = -1;
    }

  std::vector<int> Layer0, Layer1, Layer2, Layer3, Layer4, Layer5;
  Layer0.clear();
  Layer1.clear();
  Layer2.clear();
  Layer3.clear();
  Layer4.clear();
  Layer5.clear();

  for (int k=0; k<(int)CFEBnData.size(); ++k)
    {
      for (int j=0; j<12; j++)
        {
          int DiStr=0;
          DiStr = (CFEBnData[k] >> j) & 0x1;
          if ( (DiStr !=0) && (j<8) && (k==0) )
            {
              Layer0.push_back(j);
            }
          if ((DiStr !=0) && (j>7) && (j<12) && (k==0))
            {
              Layer1.push_back(j);
            }
          if ((DiStr !=0) && (j<4) && (k==1))
            {
              Layer1.push_back(j);
            }
          if ((DiStr !=0) && (j>3) && (j<12) && (k==1))
            {
              Layer2.push_back(j);
            }
          if ( (DiStr !=0) && (j<8) && (k==2) )
            {
              Layer3.push_back(j);
            }
          if ((DiStr !=0) && (j>7) && (j<12) && (k==2))
            {
              Layer4.push_back(j);
            }
          if ((DiStr !=0) && (j<4) && (k==3))
            {
              Layer4.push_back(j);
            }
          if ((DiStr !=0) && (j>3) && (j<12) && (k==3))
            {
              Layer5.push_back(j);
            }
        }
    }

  CFEBnByLayers.push_back(Layer0);
  CFEBnByLayers.push_back(Layer1);
  CFEBnByLayers.push_back(Layer2);
  CFEBnByLayers.push_back(Layer3);
  CFEBnByLayers.push_back(Layer4);
  CFEBnByLayers.push_back(Layer5);

  return CFEBnByLayers;
}

void CSCTMBBlockedCFEB::print() const
{

  std::cout << " Blocked CFEB DiStrips List Content " << std::endl;
  for (int i=0; i<(int)getData().size(); ++i)
    {
      std::cout << " word " << i << " : " << std::hex << getData()[i] << std::dec << std::endl;
    }

  std::vector< std::vector<int> > anyCFEB;
  anyCFEB.clear();
  std::vector <int> anyLayer;
  anyLayer.clear();
  std::cout << std::endl;
  std::cout << " Blocked DiStrips by CFEB and Layers unpacked " << std::endl;
  for (int z=0; z<5; ++z)
    {
      anyCFEB = getSingleCFEBList(z);
      std::cout << " CFEB# " << z << std::endl;
      int LayerCnt=0;
      for (std::vector< std::vector<int> >::const_iterator layerIt=anyCFEB.begin(); layerIt !=anyCFEB.end(); layerIt++)
        {
          anyLayer=*layerIt;
          std::cout << " Layer: " << LayerCnt;
          if (anyLayer.size() !=0)
            {
              for (int i=0; i<(int)anyLayer.size(); i++)
                {
                  std::cout << " " << anyLayer[i];
                }
            }
          else
            std::cout << " No Blocked DiStrips on the Layer ";
          std::cout << std::endl;
          LayerCnt++;
          anyLayer.clear();
        }
      anyCFEB.clear();
    }

}

