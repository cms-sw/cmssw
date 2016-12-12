#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.h"
#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

#include "TFile.h"
#include "TH1F.h"

using namespace l1t;
using namespace std;


EmtfPtAssignment::EmtfPtAssignment(const char * tree_dir):
  allowedModes_({3,5,9,6,10,12,7,11,13,14,15}){
  
  for (unsigned i=0;i<allowedModes_.size();i++){
    int mode_inv = allowedModes_[i];
    stringstream ss;
    ss << tree_dir << "/" << mode_inv;
    forest_[mode_inv].loadForestFromXML(ss.str().c_str(),64);    
  }
}

//ModeVariables is a 2D arrary indexed by [TrackMode(13 Total Listed Below)][VariableNumber(20 Total Constructed Above)]
// Variable numbering
// 0 = dPhi12
// 1 = dPhi13
// 2 = dPhi14
// 3 = dPhi23
// 4 = dPhi24
// 5 = dPhi34
// 6 = dEta12
// 7 = dEta13
// 8 = dEta14
// 9 = dEta23
// 10 = dEta24
// 11 = dEta34
// 12 = CLCT1
// 13 = CLCT2
// 14 = CLCT3
// 15 = CLCT4
// 16 = CSCID1
// 17 = CSCID2
// 18 = CSCID3
// 19 = CSCID4
// 20 = FR1
// 21 = FR2
// 22 = FR3
// 23 = FR4

// Bobby's Scheme3 (or "SchemeC"), with 30 bit compression //
//3:TrackEta:dPhi12:dEta12:CLCT1:CLCT2:FR1
//4:Single Station Track Not Possible
//5:TrackEta:dPhi13:dEta13:CLCT1:CLCT3:FR1
//6:TrackEta:dPhi23:dEta23:CLCT2:CLCT3:FR2
//7:TrackEta:dPhi12:dPhi23:dEta13:CLCT1:FR1
//8:Single Station Track Not Possible
//9:TrackEta:dPhi14:dEta14:CLCT1:CLCT4:FR1
//10:TrackEta:dPhi24:dEta24:CLCT2:CLCT4:FR2 
//11:TrackEta:dPhi12:dPhi24:dEta14:CLCT1:FR1
//12:TrackEta:dPhi34:dEta34:CLCT3:CLCT4:FR3
//13:TrackEta:dPhi13:dPhi34:dEta14:CLCT1:FR1
//14:TrackEta:dPhi23:dPhi34:dEta24:CLCT2
//15:TrackEta:dPhi12:dPhi23:dPhi34:FR1
static const int ModeVariables_Scheme3[13][6] =
  {
    {0,6,12,13,20,-999},            // 3
    {-999,-999,-999,-999,-999,-999},  // 4
    {1,7,12,14,20,-999},            // 5  
    {3,9,13,14,21,-999},            // 6
    {0,3,7,12,20,-999},               // 7
    {-999,-999,-999,-999,-999,-999},  // 8
    {2,8,12,15,20,-999},            // 9
    {4,10,13,15,21,-999},           // 10
    {0,4,8,12,20,-999},               // 11
    {5,11,14,15,22,-999},           // 12
    {1,5,8,16,20,-999},               // 13
    {3,5,10,13,-999,-999},              // 14
    {0,3,5,20,-999,-999}            // 15
  };


// 256 max units----

static const int dPhiNLBMap_5bit_256Max[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136};

static const int dPhiNLBMap_6bit_256Max[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 45, 47, 49, 51, 53, 56, 58, 61, 65, 68, 73, 78, 83, 89, 97, 106, 116, 129, 145, 166, 193, 232};

static const int dPhiNLBMap_7bit_256Max[128] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 90, 91, 93, 94, 96, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 118, 120, 123, 125, 128, 131, 134, 138, 141, 145, 149, 153, 157, 161, 166, 171, 176, 182, 188, 194, 201, 209, 217, 225, 235, 245};


// 512 max units----
static const int dPhiNLBMap_7bit_512Max[256] =  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 86, 87, 89, 91, 92, 94, 96, 98, 100, 102, 105, 107, 110, 112, 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168, 174, 181, 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470};

static const int dPhiNLBMap_8bit_512Max[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 174, 175, 176, 178, 179, 180, 182, 183, 185, 186, 188, 190, 191, 193, 194, 196, 198, 200, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 228, 230, 232, 235, 237, 240, 242, 245, 248, 250, 253, 256, 259, 262, 265, 268, 272, 275, 278, 282, 285, 289, 293, 297, 300, 305, 309, 313, 317, 322, 327, 331, 336, 341, 347, 352, 358, 363, 369, 375, 382, 388, 395, 402, 410, 417, 425, 433, 442, 450, 460, 469, 479, 490, 500};


static const int dPhiNLBMap_5bit[32] =
  {  0       ,       1       ,       2       ,       4       ,       5       ,       7       ,       9       ,       11      ,       13      ,       15      ,       18      ,       21      ,       24      ,       28      ,       32      ,       37      ,       41      ,       47      ,       53      ,       60      ,       67      ,       75      ,       84      ,       94      ,       105     ,       117     ,       131     ,       145     ,       162     ,       180     ,       200     ,       222};
 
// Array that maps the 7-bit integer dPhi --> dPhi-units. It is assumed that this is used for dPhi12,
// which has a maximum value of 7.67 degrees (511 units) in the extrapolation units.
static const int dPhiNLBMap_7bit[128] =
  {     0       ,       1       ,       2       ,       3       ,       4       ,       5       ,       6       ,       8       ,       9       ,       10      ,       11      ,       12      ,       14      ,       15      ,       16      ,       17      ,       19      ,       20      ,       21      ,       23      ,       24      ,       26      ,       27      ,       29      ,       30      ,       32      ,       33      ,       35      ,       37      ,       38      ,       40      ,       42      ,       44      ,       45      ,       47      ,       49      ,       51      ,       53      ,       55      ,       57      ,       59      ,       61      ,       63      ,       65      ,       67      ,       70      ,       72      ,       74      ,       77      ,       79      ,       81      ,       84      ,       86      ,       89      ,       92      ,       94      ,       97      ,       100     ,       103     ,       105     ,       108     ,       111     ,       114     ,       117     ,       121     ,       124     ,       127     ,       130     ,       134     ,       137     ,       141     ,       144     ,       148     ,       151     ,       155     ,       159     ,       163     ,       167     ,       171     ,       175     ,       179     ,       183     ,       188     ,       192     ,       197     ,       201     ,       206     ,       210     ,       215     ,       220     ,       225     ,       230     ,       235     ,       241     ,       246     ,       251     ,       257     ,       263     ,       268     ,       274     ,       280     ,       286     ,       292     ,       299     ,       305     ,       312     ,       318     ,       325     ,       332     ,       339     ,       346     ,       353     ,       361     ,       368     ,       376     ,       383     ,       391     ,       399     ,       408     ,       416     ,       425     ,       433     ,       442     ,       451     ,       460     ,       469     ,       479     ,       489 };
 
// Array that maps the 8-bit integer dPhi --> dPhi-units. It is assumed that this is used for dPhi12,
// which has a maximum value of 7.67 degrees (511 units) in the extrapolation units.
static const int dPhiNLBMap_8bit[256] =
  {      0       ,       1       ,       2       ,       3       ,       4       ,       5       ,       6       ,       7       ,       8       ,       9       ,       10      ,       11      ,       12      ,       13      ,       14      ,       16      ,       17      ,       18      ,       19      ,       20      ,       21      ,       22      ,       23      ,       24      ,       25      ,       27      ,       28      ,       29      ,       30      ,       31      ,       32      ,       33      ,       35      ,       36      ,       37      ,       38      ,       39      ,       40      ,       42      ,       43      ,       44      ,       45      ,       46      ,       48      ,       49      ,       50      ,       51      ,       53      ,       54      ,       55      ,       56      ,       58      ,       59      ,       60      ,       61      ,       63      ,       64      ,       65      ,       67      ,       68      ,       69      ,       70      ,       72      ,       73      ,       74      ,       76      ,       77      ,       79      ,       80      ,       81      ,       83      ,       84      ,       85      ,       87      ,       88      ,       90      ,       91      ,       92      ,       94      ,       95      ,       97      ,       98      ,       100     ,       101     ,       103     ,       104     ,       105     ,       107     ,       108     ,       110     ,       111     ,       113     ,       115     ,       116     ,       118     ,       119     ,       121     ,       122     ,       124     ,       125     ,       127     ,       129     ,       130     ,       132     ,       133     ,       135     ,       137     ,       138     ,       140     ,       141     ,       143     ,       145     ,       146     ,       148     ,       150     ,       151     ,       153     ,       155     ,       157     ,       158     ,       160     ,       162     ,       163     ,       165     ,       167     ,       169     ,       171     ,       172     ,       174     ,       176     ,       178     ,       180     ,       181     ,       183     ,       185     ,       187     ,       189     ,       191     ,       192     ,       194     ,       196     ,       198     ,       200     ,       202     ,       204     ,       206     ,       208     ,       210     ,       212     ,       214     ,       216     ,       218     ,       220     ,       222     ,       224     ,       226     ,       228     ,       230     ,       232     ,       234     ,       236     ,       238     ,       240     ,       242     ,       244     ,       246     ,       249     ,       251     ,       253     ,       255     ,       257     ,       259     ,       261     ,       264     ,       266     ,       268     ,       270     ,       273     ,       275     ,       277     ,       279     ,       282     ,       284     ,       286     ,       289     ,       291     ,       293     ,       296     ,       298     ,       300     ,       303     ,       305     ,       307     ,       310     ,       312     ,       315     ,       317     ,       320     ,       322     ,       324     ,       327     ,       329     ,       332     ,       334     ,       337     ,       340     ,       342     ,       345     ,       347     ,       350     ,       352     ,       355     ,       358     ,       360     ,       363     ,       366     ,       368     ,       371     ,       374     ,       376     ,       379     ,       382     ,       385     ,       387     ,       390     ,       393     ,       396     ,       398     ,       401     ,       404     ,       407     ,       410     ,       413     ,       416     ,       419     ,       421     ,       424     ,       427     ,       430     ,       433     ,       436     ,       439     ,       442     ,       445     ,       448     ,       451     ,       454     ,       457     ,       461     ,       464     ,       467     ,       470     ,       473     ,       476     ,       479     ,       483     };



static float getNLBdPhi(float dPhi, int bits, int max=512)
{
  float dPhi_= max; 
  float sign_ = 1;
  if (dPhi<0)
    sign_ = -1;
  dPhi = fabs(dPhi);
  
  if (max==256)
    {
      if (bits == 5)
        {
          dPhi_ = dPhiNLBMap_5bit_256Max[(1<<bits)-1];
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_5bit_256Max[edge]<=dPhi && dPhiNLBMap_5bit_256Max[edge+1]>dPhi)
              {
                dPhi_ = dPhiNLBMap_5bit_256Max[edge];
                break;
              }
        }
      if (bits == 6)
        {
          dPhi_ = dPhiNLBMap_6bit_256Max[(1<<bits)-1];
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_6bit_256Max[edge]<=dPhi && dPhiNLBMap_6bit_256Max[edge+1]>dPhi)
              {
                dPhi_ = dPhiNLBMap_6bit_256Max[edge];
                break;
              }
        }
      if (bits == 7)
        {
          dPhi_ = dPhiNLBMap_7bit_256Max[(1<<bits)-1];
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_7bit_256Max[edge]<=dPhi && dPhiNLBMap_7bit_256Max[edge+1]>dPhi)
              {
                dPhi_ = dPhiNLBMap_7bit_256Max[edge];
                break;
              }
        }
    }
  if (max==512)
    {
      if (bits == 7)
        {
          dPhi_ = dPhiNLBMap_7bit_512Max[(1<<bits)-1];
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_7bit_512Max[edge]<=dPhi && dPhiNLBMap_7bit_512Max[edge+1]>dPhi)
              {
                dPhi_ = dPhiNLBMap_7bit_512Max[edge];
                break;
              }
        }
      if (bits == 8)
        {
          dPhi_ = dPhiNLBMap_8bit_512Max[(1<<bits)-1];
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_8bit_512Max[edge]<=dPhi && dPhiNLBMap_8bit_512Max[edge+1]>dPhi)
              {
                dPhi_ = dPhiNLBMap_8bit_512Max[edge];
                break;
              }
        }
    }
  
  if (dPhi>=max) dPhi_ = max;
  return (sign_ * dPhi_);  
}



static int getNLBdPhiBin(float dPhi, int bits, int max=512)
{
  int dPhiBin_= (1<<bits)-1; 
  
  // Unused variable
  /* float sign_ = 1; */
  /* if (dPhi<0) */
  /*   sign_ = -1; */

  dPhi = fabs(dPhi);
  
  if (max==256)
    {
      if (bits == 5)
        {
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_5bit_256Max[edge]<=dPhi && dPhiNLBMap_5bit_256Max[edge+1]>dPhi)
              {
                dPhiBin_ = edge;
                break;
              }
        }
      if (bits == 6)
        {
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_6bit_256Max[edge]<=dPhi && dPhiNLBMap_6bit_256Max[edge+1]>dPhi)
              {
                dPhiBin_ = edge;
                break;
              }
        }
      if (bits == 7)
        {
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_7bit_256Max[edge]<=dPhi && dPhiNLBMap_7bit_256Max[edge+1]>dPhi)
              {
                dPhiBin_ = edge;
                break;
              }
        }
    }
  if (max==512)
    {
      if (bits == 7)
        {
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_7bit_512Max[edge]<=dPhi && dPhiNLBMap_7bit_512Max[edge+1]>dPhi)
              {
                dPhiBin_ = edge;
                break;
              }
        }
      if (bits == 8)
        {
          for (int edge=0; edge<(1<<bits)-1; edge++)
            if (dPhiNLBMap_8bit_512Max[edge]<=dPhi && dPhiNLBMap_8bit_512Max[edge+1]>dPhi)
              {
                dPhiBin_ = edge;
                break;
              }
        }
    }
  
  return ( dPhiBin_);  
}


static float getdPhiFromBin(int dPhiBin, int bits, int max=512)
{
  int dPhi_= (1<<bits)-1; 
  
  if (dPhiBin>(1<<bits)-1)
    dPhiBin = (1<<bits)-1;
  
  if (max==256)
    {
      if (bits == 5)
        dPhi_ = dPhiNLBMap_5bit_256Max[dPhiBin];

      if (bits == 6)
        dPhi_ = dPhiNLBMap_6bit_256Max[dPhiBin];
        
      if (bits == 7)
        dPhi_ = dPhiNLBMap_7bit_256Max[dPhiBin];
    }
  if (max==512)
    {
      if (bits == 7)
        dPhi_ = dPhiNLBMap_7bit_512Max[dPhiBin];

      if (bits == 8)
        dPhi_ = dPhiNLBMap_8bit_512Max[dPhiBin];
    }
  
  return ( dPhi_);  
}

static float getCLCT(float clct)
{
  if ((int)clct==10)
    clct=0;
  else if ((int)clct==9)
    clct=1;
  else if ((int)clct==8)
    clct=-1;
  else if ((int)clct==7)
    clct=2;
  else if ((int)clct==6)
    clct=-2;
  else if ((int)clct==5)
    clct=3;
  else if ((int)clct==4)
    clct=-3;
  else if ((int)clct==3)
    clct=4;
  else if ((int)clct==2)
    clct=-4;
  else
    clct=-999;
    
  float clct_ = 0;
  float sign_ = 1;

  if (clct<0)
    sign_ = -1;
  
  clct = fabs(clct);

  if (clct<=0) // 0
    clct_ = 0;
  else if (clct<=1) //1,2
    clct_ = 1;
  else if (clct<=2)// 3,4
    clct_ = 2;
  else 
    clct_ = 3; // 5,6

  return (sign_ * clct_);
}

static float getdTheta(float dTheta)
{
  float dTheta_ = 0;
  // Unused variable
  /* float sign_ = 1; */
  /* if (dTheta<0) */
  /*   sign_ = -1; */
  
  if (dTheta<=-3)
    dTheta_ = 0;
  else if (dTheta<=-2)
    dTheta_ = 1;
  else if (dTheta<=-1)
    dTheta_ = 2;
  else if (dTheta<=0)
    dTheta_ = 3;
  else if (dTheta<=1)
    dTheta_ = 4;
  else if (dTheta<=2)
    dTheta_ = 5;
  else if (dTheta<=3)
    dTheta_ = 6;
  else 
    dTheta_ = 7;

  return ( dTheta_);
  
}


//static float getdEta(float deta)
//{
//  float deta_ = 0;
//  
//  if (deta<=-5)
//    deta_ = 0;
//  else if (deta<=-2)
//    deta_ = 1;
//  else if (deta<=-1)
//    deta_ = 2;
//  else if (deta<=0)
//    deta_ = 3;
//  else if (deta<=1)
//    deta_ = 4;
//  else if (deta<=3)
//    deta_ = 5;
//  else if (deta<=6)
//    deta_ = 6;
//  else 
//    deta_ = 7;
//
//  return ( deta_);
//}

static float getEta(float eta, int bits=5)
{
  float eta_ = 0;
  float sign_ = 1;
  if (eta<0)
    sign_ = -1;

  if (bits>5) bits = 5;
  int shift = 5 - bits;
  int etaInt = (fabs(eta) - 0.9)*(32.0/(1.6))-0.5;
  etaInt = (etaInt>>shift)<<shift;

  eta_ = 0.9 + (etaInt + 0.5)*(1.6/32.0);
  return (eta_*sign_);
}

static int getEtaInt(float eta, int bits=5)
{
  // Unused variables
  /* float eta_ = 0; */
  /* float sign_ = 1; */
  /* if (eta<0) */
  /*   sign_ = -1; */

  if (bits>5) bits = 5;
  int shift = 5 - bits;
  int etaInt = (fabs(eta) - 0.9)*(32.0/(1.6))-0.5;
  etaInt = (etaInt>>shift);
  if(etaInt > 31){etaInt = 31;}
  
  /* eta_ = 0.9 + (etaInt + 0.5)*(1.6/32.0); */
  return (etaInt);
}

static float getEtafromBin(int etaBin, int bits=5)
{
  if (etaBin>((1<<5)-1))
    etaBin = ((1<<5)-1);
  if (etaBin<0)
    etaBin = 0;
      
  if (bits>5) bits = 5;
  int shift = 5 - bits;
  int etaInt_ = etaBin << shift;
  float eta_ = 0.9 + (etaInt_ + 0.5)*(1.6/32.0);
  return (eta_);
}


float EmtfPtAssignment::calculatePt(unsigned long Address)
{
  bool verbose = false;
  int ModeVariables[13][6];

  //FIXME: use pointer to avoid a copy:
  for (int i=0;i<13;i++) {
    for (int j=0;j<6;j++) {
      ModeVariables[i][j] = ModeVariables_Scheme3[i][j];
    }
  }
  
  int dphi[6] = {-999,-999,-999,-999,-999,-999}, deta[6] = {-999,-999,-999,-999,-999,-999};
  int clct[4] = {-999,-999,-999,-999}, cscid[4] = {-999,-999,-999,-999};
  int phis[4] = {-999,-999,-999,-999}, etas[4] = {-999,-999,-999,-999}, mode_inv = 0;;
  int FR[4] = {-999,-999,-999,-999};
  float eta =0 ;
  
  float dPhi12 = dphi[0];
  float dPhi13 = dphi[1];
  float dPhi14 = dphi[2];
  float dPhi23 = dphi[3];
  float dPhi24 = dphi[4];
  float dPhi34 = dphi[5];
  float dTheta12 = deta[0];
  float dTheta13 = deta[1];
  float dTheta14 = deta[2];
  float dTheta23 = deta[3];
  float dTheta24 = deta[4];
  float dTheta34 = deta[5];
  float TrackEta = 0;
  float CLCT1 = clct[0];
  float CLCT2 = clct[1];
  float CLCT3 = clct[2];
  float CLCT4 = clct[3];
  float FR1 = FR[0]; 
  float FR2 = FR[1]; 
  float FR3 = FR[2]; 
  float FR4 = FR[3];

  mode_inv = (Address >> (30-4)) & ((1<<4)-1);
        
  // Decode the Pt LUT Address
  //unsigned long Address = 0x0;

  if (verbose) cout << "PtAssignment:getPt: decoding, inverted mode = " << mode_inv << endl;
  
  if (mode_inv == 3) // 1-2
    {

      int dPhi12_ =    (Address >> (0))   & ((1<<9)-1);
      float sign12 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta12 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT2  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT2Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR2 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi12 = dPhi12_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign12 == 0) dPhi12 = -1*dPhi12;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
      if (CLCT2Sign == 0) CLCT2 = -1*CLCT2;
      
      if (verbose) cout << "PtAssignment:getPt: decoding, Addess = 0x" << hex << Address << endl;
      if (verbose) cout << "PtAssignment:getPt: decoding, dPhi12 = " << dec << dPhi12_ << endl;

    }
  
  if (mode_inv == 5) // 1-3
    {

      int dPhi13_ =    (Address >> (0))   & ((1<<9)-1);
      float sign13 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta13 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT3  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT3Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR3 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi13 = dPhi13_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign13 == 0) dPhi13 = -1*dPhi13;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
      if (CLCT3Sign == 0) CLCT3 = -1*CLCT3;
    }
    
  if (mode_inv == 9) // 1-4
    {

      int dPhi14_ =    (Address >> (0))   & ((1<<9)-1);
      float sign14 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta14 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT4  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT4Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR4 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi14 = dPhi14_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign14 == 0) dPhi14 = -1*dPhi14;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
      if (CLCT4Sign == 0) CLCT4 = -1*CLCT4;
    }
   
  if (mode_inv == 6) // 2-3
    {

      int dPhi23_ =    (Address >> (0))   & ((1<<9)-1);
      float sign23 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta23 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT2  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT2Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT3  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT3Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR2 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR3 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi23 = dPhi23_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign23 == 0) dPhi23 = -1*dPhi23;
      if (CLCT2Sign == 0) CLCT2 = -1*CLCT2;
      if (CLCT3Sign == 0) CLCT3 = -1*CLCT3;
    }
  if (mode_inv == 10) // 2-4
    {

      int dPhi24_ =    (Address >> (0))   & ((1<<9)-1);
      float sign24 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta24 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT2  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT2Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT4  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT4Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR2 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR4 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi24 = dPhi24_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign24 == 0) dPhi24 = -1*dPhi24;
      if (CLCT2Sign == 0) CLCT2 = -1*CLCT2;
      if (CLCT4Sign == 0) CLCT4 = -1*CLCT4;
    }
  if (mode_inv == 12) // 3-4
    {

      int dPhi34_ =    (Address >> (0))   & ((1<<9)-1);
      float sign34 =   (Address >> (0+9)) & ((1<<1)-1);
      dTheta34 =       (Address >> (0+9+1)) & ((1<<3)-1);
      CLCT3  =         (Address >> (0+9+1+3)) & ((1<<2)-1);
      float CLCT3Sign= (Address >> (0+9+1+3+2)) & ((1<<1)-1);
      CLCT4  =         (Address >> (0+9+1+3+2+1)) & ((1<<2)-1);
      float CLCT4Sign= (Address >> (0+9+1+3+2+1+2)) & ((1<<1)-1);
      FR3 =            (Address >> (0+9+1+3+2+1+2+1)) & ((1<<1)-1);
      FR4 =            (Address >> (0+9+1+3+2+1+2+1+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+9+1+3+2+1+2+1+1+1)) & ((1<<5)-1);

      dPhi34 = dPhi34_;
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign34 == 0) dPhi34 = -1*dPhi34;
      if (CLCT3Sign == 0) CLCT3 = -1*CLCT3;
      if (CLCT4Sign == 0) CLCT4 = -1*CLCT4;
    }
  if (mode_inv == 7) //1-2-3
    {
      int dPhi12_ =    (Address >> (0))     & ((1<<7)-1);
      int dPhi23_ =    (Address >> (0+7))   & ((1<<5)-1);
      float sign12 =   (Address >> (0+7+5)) & ((1<<1)-1);
      float sign23 =   (Address >> (0+7+5+1)) & ((1<<1)-1);
      dTheta13 =       (Address >> (0+7+5+1+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+7+5+1+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+7+5+1+1+3+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+7+5+1+1+3+2+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+7+5+1+1+3+2+1+1)) & ((1<<5)-1);

      dPhi12 = getdPhiFromBin( dPhi12_, 7, 512 );
      dPhi23 = getdPhiFromBin( dPhi23_, 5, 256 );

      //cout << "getPt: dPhi12: " << dPhi12_ << " " << dPhi12 << endl;
      // cout << "getPt: dPhi23: " << dPhi23_ << " " << dPhi23 << endl;
      
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign12 == 0) dPhi12 = -1*dPhi12;
      if (sign23 == 0) dPhi23 = -1*dPhi23;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
    }
  if (mode_inv == 11) // 1-2-4
    {
      int dPhi12_ =    (Address >> (0))     & ((1<<7)-1);
      int dPhi24_ =    (Address >> (0+7))   & ((1<<5)-1);
      float sign12 =   (Address >> (0+7+5)) & ((1<<1)-1);
      float sign24 =   (Address >> (0+7+5+1)) & ((1<<1)-1);
      dTheta14 =       (Address >> (0+7+5+1+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+7+5+1+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+7+5+1+1+3+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+7+5+1+1+3+2+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+7+5+1+1+3+2+1+1)) & ((1<<5)-1);

      dPhi12 = getdPhiFromBin( dPhi12_, 7, 512 );
      dPhi24 = getdPhiFromBin( dPhi24_, 5, 256 );
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign12 == 0) dPhi12 = -1*dPhi12;
      if (sign24 == 0) dPhi24 = -1*dPhi24;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
    }
  if (mode_inv == 13) // 1-3-4
    {
      int dPhi13_ =    (Address >> (0))     & ((1<<7)-1);
      int dPhi34_ =    (Address >> (0+7))   & ((1<<5)-1);
      float sign13 =   (Address >> (0+7+5)) & ((1<<1)-1);
      float sign34 =   (Address >> (0+7+5+1)) & ((1<<1)-1);
      dTheta14 =       (Address >> (0+7+5+1+1)) & ((1<<3)-1);
      CLCT1  =         (Address >> (0+7+5+1+1+3)) & ((1<<2)-1);
      float CLCT1Sign= (Address >> (0+7+5+1+1+3+2)) & ((1<<1)-1);
      FR1 =            (Address >> (0+7+5+1+1+3+2+1)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+7+5+1+1+3+2+1+1)) & ((1<<5)-1);

      dPhi13 = getdPhiFromBin( dPhi13_, 7, 512 );
      dPhi34 = getdPhiFromBin( dPhi34_, 5, 256 );
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign13 == 0) dPhi13 = -1*dPhi13;
      if (sign34 == 0) dPhi34 = -1*dPhi34;
      if (CLCT1Sign == 0) CLCT1 = -1*CLCT1;
    }
  if (mode_inv == 14) // 2-3-4
    {
      int dPhi23_ =    (Address >> (0))     & ((1<<7)-1);
      int dPhi34_ =    (Address >> (0+7))   & ((1<<6)-1);
      float sign23 =   (Address >> (0+7+6)) & ((1<<1)-1);
      float sign34 =   (Address >> (0+7+6+1)) & ((1<<1)-1);
      dTheta24 =       (Address >> (0+7+6+1+1)) & ((1<<3)-1);
      CLCT2  =         (Address >> (0+7+5+1+1+3)) & ((1<<2)-1);
      float CLCT2Sign= (Address >> (0+7+6+1+1+3+2)) & ((1<<1)-1);
      int TrackEta_ =  (Address >> (0+7+6+1+1+3+2+1)) & ((1<<5)-1);

      dPhi23 = getdPhiFromBin( dPhi23_, 7, 512 );
      dPhi34 = getdPhiFromBin( dPhi34_, 6, 256 );
      TrackEta = getEtafromBin( TrackEta_, 5);
      
      if (sign23 == 0) dPhi23 = -1*dPhi23;
      if (sign34 == 0) dPhi34 = -1*dPhi34;
      if (CLCT2Sign == 0) CLCT2 = -1*CLCT2;
    }
  if (mode_inv == 15)
    {
      int dPhi12_ =         (Address >> (0))     & ((1<<7)-1);
      int dPhi23_ =         (Address >> (0+7))   & ((1<<5)-1);
      int dPhi34_ =         (Address >> (0+7+5)) & ((1<<6)-1);
      int sign23 =          (Address >> (0+7+5+6)) & ((1<<1)-1);
      int sign34 =          (Address >> (0+7+5+6+1)) & ((1<<1)-1);
      FR1 =                 (Address >> (0+7+5+6+1+1)) & ((1<<1)-1);
      int TrackEta_ =       (Address >> (0+7+5+6+1+1+1)) & ((1<<5)-1);
        
      dPhi12 = getdPhiFromBin( dPhi12_, 7, 512 );
      dPhi23 = getdPhiFromBin( dPhi23_, 5, 256 );
      dPhi34 = getdPhiFromBin( dPhi34_, 6, 256 );
      TrackEta = getEtafromBin( TrackEta_, 5 );
        
      if (sign23 == 0) dPhi23 = -1*dPhi23;
      if (sign34 == 0) dPhi34 = -1*dPhi34;
    }
  

  
  if(verbose){
    if (mode_inv == 15) // 1-2-3-4
      cout << "Inverted mode 15: " << hex << Address << " " << dec << dPhi12 << " " << dPhi23 <<  " " << dPhi34 << " " << " " << FR1 << " " << TrackEta << " " << dec << endl;
    if (mode_inv == 14) // 2-3-4
      cout << "Inverted mode 14: " << hex << Address << " " << dec << dPhi23 << " " << dPhi34  << " " << " " << " " << TrackEta << " " << dec << endl;
    if (mode_inv == 13) // 1-3-4
      cout << "Inverted mode 13: " << hex << Address << " " << dec << dPhi13 << " " << dPhi34  << " " << " " << FR1 << " " << TrackEta << " " << dec << endl;
    if (mode_inv == 11) // 1-2-4
      cout << "Inverted mode 11: " << hex << Address << " " << dec << dPhi12 << " " << dPhi24  << " " << " " << FR1 << " " << TrackEta << " " << dec << endl;
    if (mode_inv == 7)
      cout << "Inverted mode 7: " << hex << Address << " " << dec << dPhi12 << " " << dPhi23  << " " << " " << FR1 << " " << TrackEta << " " << dec << endl;
    if (mode_inv == 3)
      cout << "Inverted mode 3: " << hex << Address << " " << dec << dPhi12 << " " << " " << FR1 << " " << TrackEta << " " << dec << endl;
  }
  
  // now use rebinned values
  dphi[0] = dPhi12;
  dphi[1] = dPhi13;
  dphi[2] = dPhi14;
  dphi[3] = dPhi23;
  dphi[4] = dPhi24;
  dphi[5] = dPhi34;
  deta[0] = dTheta12;
  deta[1] = dTheta13;
  deta[2] = dTheta14;
  deta[3] = dTheta23;
  deta[4] = dTheta24;
  deta[5] = dTheta34;
  eta = TrackEta;
  clct[0] = CLCT1;
  clct[1] = CLCT2;
  clct[2] = CLCT3;
  clct[3] = CLCT4;
  FR[0] = FR1;
  FR[1] = FR2;
  FR[2] = FR3;
  FR[3] = FR4;
      
           
  if(verbose){
    for(int f=0;f<4;f++){
      cout<<"\nphis["<<f<<"] = "<<phis[f]<<" and etas = "<<etas[f]<<endl;
      cout<<"\nclct["<<f<<"] = "<<clct[f]<<" and cscid = "<<cscid[f]<<endl;
    }
	
    for(int u=0;u<6;u++)
      cout<<"\ndphi["<<u<<"] = "<<dphi[u]<<" and deta = "<<deta[u]<<endl;
  }
	
  float MpT = -1;//final pT to return
	
	
  ///////////////////////////////////////////////////////////////////////////////
  //// Variables is a array of all possible variables used in pT calculation ////
  ///////////////////////////////////////////////////////////////////////////////
	
  int size[13] = {5,0,5,5,5,0,5,5,5,5,5,4,4};
  int Variables[24] = {dphi[0], dphi[1], dphi[2], dphi[3], dphi[4], dphi[5], deta[0], deta[1], deta[2], deta[3], deta[4], deta[5],
                       clct[0], clct[1], clct[2], clct[3], cscid[0], cscid[1], cscid[2], cscid[3], FR[0], FR[1], FR[2], FR[3]};
	
	
		
  ////////////////////////
  //// pT Calculation ////
  ////////////////////////
  //float gpt = -1;

  int goodMode = false;
  int allowedModes[11] = {3,5,9,6,10,12,7,11,13,14,15};
  for (int i=0;i<11;i++)
    if (allowedModes[i] == mode_inv)
      {
        goodMode = true;
        break;
      }
  
  if (goodMode)
    for(int i=3;i<16;i++){
	
      if(i != mode_inv)
	continue;
			    
      vector<Double_t> Data;
      Data.push_back(1.0);
      Data.push_back(eta);
      for(int y=0;y<size[mode_inv-3];y++){
	Data.push_back(Variables[ModeVariables[mode_inv-3][y]]);
	if(verbose) cout<<"Generalized Variables "<<y<<" "<<Variables[ModeVariables[mode_inv-3][y]]<<"\n";
      }
		
      if(verbose){
	cout<<"Data.size() = "<<Data.size()<<"\n";
	for(int i=0;i<5;i++)  
	  cout<<"Data["<<i<<"] = "<<Data[i]<<"\n";
      }
		
      Event *event = new Event();
      event->data = Data;
		
      forest_[mode_inv].predictEvent(event,64);
      float OpT = event->predictedValue;
      MpT = 1/OpT;

      delete event;
    
      if (MpT<0.0) MpT = 1.0;
      if (MpT>200.0) MpT = 200.0;
                       
    }
  
  
  return MpT;
}



unsigned long EmtfPtAssignment::calculateAddress( L1TMuon::InternalTrack track, const edm::EventSetup& es, int mode) {
    edm::ESHandle<CSCGeometry> cscGeometry;
    es.get<MuonGeometryRecord>().get(cscGeometry);
  
    bool verbose = false;
    bool doComp = true;

    ///////////////////////
    /// Mode Variables ////
    ///////////////////////
  
    // Unused variables
    /* int ModeVariables[13][6]; */
    /* int ModeBits[13][6]; */
    /* for (int i=0;i<13;i++) { */
    /*   for (int j=0;j<6;j++) { */
    /*     if (whichScheme == 3) { */
    /* 	ModeVariables[i][j] = ModeVariables_Scheme3[i][j]; */
    /*     } */
    /*   } */
    /* } */

    // Unused variable
    /* const char *dir=""; */
    /* if (whichScheme == 3) */
    /*   dir = dirSchemeC; */

    int dphi[6] = {-999,-999,-999,-999,-999,-999}, deta[6] = {-999,-999,-999,-999,-999,-999};
    int clct[4] = {-999,-999,-999,-999}, cscid[4] = {-999,-999,-999,-999};
    int phis[4] = {-999,-999,-999,-999}, etas[4] = {-999,-999,-999,-999}, mode_inv = 0;;
    int FR[4] = {-999,-999,-999,-999};
	
    float theta_angle = ((track.theta)*0.2874016 + 8.5)*(3.14159265359/180);
    float eta = (-1)*log(tan(theta_angle/2));

    const TriggerPrimitiveStationMap stubs = track.getStubs();

    //track.print(cout);
		
    if(verbose) cout<<"Track eta = "<<eta<<" and has hits in stations ";//
	
    int x=0;             //12
    for(unsigned int s=8;s<12;s++){
      if((stubs.find(s)->second).size() == 1){
			
		
	if(verbose) cout<< "s= " << s << " " << endl; 
	//etas[s-8] = (fabs((stubs.find(s)->second)[0]->getCMSGlobalEta()) + 0.9)/(0.0125);
	etas[s-8] = track.thetas[x];
	if(verbose) cout<< "eta= " << etas[s-8] << " " << endl;
	phis[s-8] = track.phis[x];//(stubs.find(s)->second)[0]->getCMSGlobalPhi();//
	if(verbose) cout<< "phi= " << phis[s-8] << " " << endl;
	clct[s-8] = (stubs.find(s)->second)[0].getPattern();
	if(verbose) cout<< "clct= " << clct[s-8] << " " << endl;
	cscid[s-8] = (stubs.find(s)->second)[0].Id();
	if(verbose) cout<< "cscid= " << cscid[s-8] << " " << endl;

	if(verbose) cout<< s << " " << (stubs.find(s)->second)[0].detId<CSCDetId>().station()<<" " << phis[s-8] << endl;;

	const CSCChamber* layer = cscGeometry->chamber((stubs.find(s)->second)[0].detId<CSCDetId>());
	LocalPoint llc(0.,0.,0.);
	GlobalPoint glc = layer->toGlobal(llc);
            
			
	int FR_ = 0;
	int coord[5] = {586,686,815,924,1013};
	for(int i=0;i<5;i++){

	  if((fabs(glc.z()) < (coord[i] + 7)) && (fabs(glc.z()) > (coord[i] - 7)))
	    FR_ = 1;
                
	  FR[s-8] = FR_;
       
	}
        
	x++;
      }
    }
	
    mode_inv = 0;
    if(mode & 1)
      mode_inv |= 8;
    if(mode & 2)
      mode_inv |= 4;
    if(mode & 4)
      mode_inv |= 2;
    if(mode & 8)
      mode_inv |= 1;

    if(verbose) cout << "\n Mode = " << mode << ", inverted mode = " << mode_inv << endl; 
	
    //////////////////////////////////////////////////
    //// Calculate Delta Phi and Eta Combinations ////
    //////////////////////////////////////////////////
	
    if(phis[0] > 0 && phis[1] > 0){ // 1 - 2
      dphi[0] = phis[1] - phis[0];
      deta[0] = etas[1] - etas[0];
    }
    if(phis[0] > 0 && phis[2] > 0){ // 1 - 3
      dphi[1] = phis[2] - phis[0];
      deta[1] = etas[2] - etas[0];
    }
    if(phis[0] > 0 && phis[3] > 0){ // 1 - 4
      dphi[2] = phis[3] - phis[0];
      deta[2] = etas[3] - etas[0];
    }
    if(phis[1] > 0 && phis[2] > 0){ // 2 - 3
      dphi[3] = phis[2] - phis[1];
      deta[3] = etas[2] - etas[1];
    }
    if(phis[1] > 0 && phis[3] > 0){ // 2 - 4
      dphi[4] = phis[3] - phis[1];
      deta[4] = etas[3] - etas[1];
    }
    if(phis[2] > 0 && phis[3] > 0){ // 3 - 4
      dphi[5] = phis[3] - phis[2];
      deta[5] = etas[3] - etas[2];
    }


    if(verbose){
      if (mode_inv==3) // 1-2
	{
	  float dPhi12__ = fabs(dphi[0]);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  float clct2__ = getCLCT(clct[1]);

	  cout << endl;
	  cout << "Inverted mode 3 Track " << endl;
	  cout <<  setw(10) << "dPhi12: " << dphi[0] << setw(10) << dPhi12__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "CLCT2: " << clct[1] << setw(10) << clct2__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==5) // 1-3
	{
	  float dPhi13__ = fabs(dphi[1]);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  float clct3__ = getCLCT(clct[2]);

	  cout << endl;
	  cout << "Inverted mode 5 Track " << endl;
	  cout <<  setw(10) << "dPhi13: " << dphi[1] << setw(10) << dPhi13__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "CLCT3: " << clct[2] << setw(10) << clct3__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==9) // 1-4
	{
	  float dPhi14__ = fabs(dphi[2]);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  float clct4__ = getCLCT(clct[3]);

	  cout << endl;
	  cout << "Inverted mode 9 Track " << endl;
	  cout <<  setw(10) << "dPhi14: " << dphi[2] << setw(10) << dPhi14__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "CLCT4: " << clct[3] << setw(10) << clct4__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==6) // 2-3
	{
	  float dPhi23__ = fabs(dphi[3]);
	  float eta__ = getEta(eta, 5);
	  float clct2__ = getCLCT(clct[1]);
	  float clct3__ = getCLCT(clct[2]);

	  cout << endl;
	  cout << "Inverted mode 9 Track " << endl;
	  cout <<  setw(10) << "dPhi23: " << dphi[3] << setw(10) << dPhi23__ << endl;
	  cout <<  setw(10) << "CLCT2: " << clct[1] << setw(10) << clct2__ << endl;
	  cout <<  setw(10) << "CLCT3: " << clct[2] << setw(10) << clct3__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==10) // 2-4
	{
	  float dPhi24__ = fabs(dphi[4]);
	  float eta__ = getEta(eta, 5);
	  float clct2__ = getCLCT(clct[1]);
	  float clct4__ = getCLCT(clct[3]);

	  cout << endl;
	  cout << "Inverted mode 10 Track " << endl;
	  cout <<  setw(10) << "dPhi24: " << dphi[4] << setw(10) << dPhi24__ << endl;
	  cout <<  setw(10) << "CLCT2: " << clct[1] << setw(10) << clct2__ << endl;
	  cout <<  setw(10) << "CLCT4: " << clct[3] << setw(10) << clct4__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==12) // 3-4
	{
	  float dPhi34__ = fabs(dphi[5]);
	  float eta__ = getEta(eta, 5);
	  float clct3__ = getCLCT(clct[2]);
	  float clct4__ = getCLCT(clct[3]);

	  cout << endl;
	  cout << "Inverted mode 12 Track " << endl;
	  cout <<  setw(10) << "dPhi34: " << dphi[5] << setw(10) << dPhi34__ << endl;
	  cout <<  setw(10) << "CLCT3: " << clct[2] << setw(10) << clct3__ << endl;
	  cout <<  setw(10) << "CLCT4: " << clct[3] << setw(10) << clct4__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
  
      if (mode_inv==7) // 1-2-3
	{
	  float dPhi12__ = getNLBdPhi(dphi[0],7, 512);
	  float dPhi23__ = getNLBdPhi(dphi[3],5, 256);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  cout << endl;
	  cout << "Inverted mode 7 Track " << endl;
	  cout <<  setw(10) << "dPhi12: " << dphi[0] << setw(10) << dPhi12__ << endl;
	  cout <<  setw(10) << "dPhi23: " << dphi[3] << setw(10) << dPhi23__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==11) // 1-2-4
	{
	  float dPhi12__ = getNLBdPhi(dphi[0],7, 512);
	  float dPhi24__ = getNLBdPhi(dphi[4],5, 256);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  cout << endl;
	  cout << "Inverted mode 11 Track " << endl;
	  cout <<  setw(10) << "dPhi12: " << dphi[0] << setw(10) << dPhi12__ << endl;
	  cout <<  setw(10) << "dPhi24: " << dphi[4] << setw(10) << dPhi24__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==13) // 1-3-4
	{
	  float dPhi13__ = getNLBdPhi(dphi[1],7, 512);
	  float dPhi34__ = getNLBdPhi(dphi[5],5, 256);
	  float eta__ = getEta(eta, 5);
	  float clct1__ = getCLCT(clct[0]);
	  cout << endl;
	  cout << "Inverted mode 13 Track " << endl;
	  cout <<  setw(10) << "dPhi13: " << dphi[1] << setw(10) << dPhi13__ << endl;
	  cout <<  setw(10) << "dPhi34: " << dphi[5] << setw(10) << dPhi34__ << endl;
	  cout <<  setw(10) << "CLCT1: " << clct[0] << setw(10) << clct1__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==14) // 2-3-4
	{
	  float dPhi23__ = getNLBdPhi(dphi[3],7, 512);
	  float dPhi34__ = getNLBdPhi(dphi[5],5, 256);
	  float eta__ = getEta(eta, 5);
	  float clct2__ = getCLCT(clct[1]);
	  cout << endl;
	  cout << "Inverted mode 14 Track " << endl;
	  cout <<  setw(10) << "dPhi23: " << dphi[3] << setw(10) << dPhi23__ << endl;
	  cout <<  setw(10) << "dPhi34: " << dphi[5] << setw(10) << dPhi34__ << endl;
	  cout <<  setw(10) << "CLCT2: " << clct[1] << setw(10) << clct2__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
      if (mode_inv==15) //1-2-3-4
	{
	  float dPhi12__ = getNLBdPhi(dphi[0],7, 512);
	  float dPhi23__ = getNLBdPhi(dphi[3],5, 256);
	  float dPhi34__ = getNLBdPhi(dphi[5],6, 256);
	  float eta__ = getEta(eta, 5);

	  cout << endl;
	  cout << "Inverted mode 15 Track " << endl;
	  cout <<  setw(10) << "dPhi12: " << dphi[0] << setw(10) << dPhi12__ << endl;
	  cout <<  setw(10) << "dPhi23: " << dphi[3] << setw(10) << dPhi23__ << endl;
	  cout <<  setw(10) << "dPhi34: " << dphi[5] << setw(10) << dPhi34__ << endl;
	  cout <<  setw(10) << "eta   : " << eta << setw(10) << eta__ << endl;
	}
    }
  
    float dPhi12 = dphi[0];
    float dPhi13 = dphi[1];
    float dPhi14 = dphi[2];
    float dPhi23 = dphi[3];
    float dPhi24 = dphi[4];
    float dPhi34 = dphi[5];
    float dTheta12 = deta[0];
    float dTheta13 = deta[1];
    float dTheta14 = deta[2];
    float dTheta23 = deta[3];
    float dTheta24 = deta[4];
    float dTheta34 = deta[5];
    float TrackEta = eta;
    float CLCT1 = clct[0];
    float CLCT2 = clct[1];
    float CLCT3 = clct[2];
    float CLCT4 = clct[3];
    float FR1 = FR[0]; 
    float FR2 = FR[1]; 
    float FR3 = FR[2]; 
    float FR4 = FR[3];

    unsigned long Address = 0x0;
  
    if (doComp && mode_inv==3)
      {

	int dPhi12Sign = 1;
	if (dPhi12<0) dPhi12Sign = -1;
	// Unused variables
	/* int CLCT1Sign = 1; */
	/* int CLCT2Sign = 1;       */
	/* if (CLCT1<0) CLCT1Sign = -1; */
	/* if (CLCT2<0) CLCT2Sign = -1; */
      
	// Make Pt LUT Address
	int dPhi12_ = fabs(dPhi12);
	int sign12_ = dPhi12Sign > 0 ? 1 : 0;
	int dTheta12_ = getdTheta(dTheta12);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int CLCT2_ = getCLCT(CLCT2);
	int CLCT2Sign_ = CLCT2_ > 0 ? 1 : 0;
	CLCT2_ = abs(CLCT2_);
	int FR1_ = FR1;
	int FR2_ = FR2;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi12_ & ((1<<9)-1))    << (0);
	Address += ( sign12_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta12_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT2_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT2Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR2_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }

    if (doComp && mode_inv==5)
      {
	// signed full precision dPhi12
	int dPhi13Sign = 1;
	// Unused variables
	// int CLCT1Sign = 1;
	// int CLCT3Sign = 1;
      
	if (dPhi13<0) dPhi13Sign = -1;
	// if (CLCT1<0) CLCT1Sign = -1;
	// if (CLCT3<0) CLCT3Sign = -1;
      
	// Make Pt LUT Address
	int dPhi13_ = fabs(dPhi13);
	int sign13_ = dPhi13Sign > 0 ? 1 : 0;
	int dTheta13_ = getdTheta(dTheta13);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int CLCT3_ = getCLCT(CLCT3);
	int CLCT3Sign_ = CLCT3_ > 0 ? 1 : 0;
	CLCT3_ = abs(CLCT3_);   
	int FR1_ = FR1;
	int FR3_ = FR3;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi13_ & ((1<<9)-1))    << (0);
	Address += ( sign13_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta13_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT3_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT3Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR3_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }
    
    if (doComp && mode_inv==9)
      {
	// signed full precision dPhi12
	int dPhi14Sign = 1;
	// Unused variables
	// int dEta14Sign = 1;
	// int CLCT1Sign = 1;
	// int CLCT4Sign = 1;
      
	if (dPhi14<0) dPhi14Sign = -1;
	// if (CLCT1<0) CLCT1Sign = -1;
	// if (CLCT4<0) CLCT4Sign = -1;
      
	// Make Pt LUT Address
	int dPhi14_ = fabs(dPhi14);
	int sign14_ = dPhi14Sign > 0 ? 1 : 0;
	int dTheta14_ = getdTheta(dTheta14);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int CLCT4_ = getCLCT(CLCT4);
	int CLCT4Sign_ = CLCT4_ > 0 ? 1 : 0;
	CLCT4_ = abs(CLCT4_);
	int FR1_ = FR1;
	int FR4_ = FR4;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi14_ & ((1<<9)-1))    << (0);
	Address += ( sign14_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta14_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT4_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT4Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR4_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }
    if (doComp && mode_inv==6) // 2-3
      {
	// signed full precision dPhi12
	int dPhi23Sign = 1;
	// Unused variables
	// int CLCT2Sign = 1;
	// int CLCT3Sign = 1;
      
	if (dPhi23<0) dPhi23Sign = -1;
	// if (CLCT2<0) CLCT2Sign = -1;
	// if (CLCT3<0) CLCT3Sign = -1;
      
	// Make Pt LUT Address
	int dPhi23_ = fabs(dPhi23);
	int sign23_ = dPhi23Sign > 0 ? 1 : 0;
	int dTheta23_ = getdTheta(dTheta23);
	int CLCT2_ = getCLCT(CLCT2);
	int CLCT2Sign_ = CLCT2_ > 0 ? 1 : 0;
	CLCT2_ = abs(CLCT2_);
	int CLCT3_ = getCLCT(CLCT3);
	int CLCT3Sign_ = CLCT3_ > 0 ? 1 : 0;
	CLCT3_ = abs(CLCT3_);
	int FR2_ = FR2;
	int FR3_ = FR3;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi23_ & ((1<<9)-1))    << (0);
	Address += ( sign23_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta23_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT2_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT2Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT3_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT3Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR2_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR3_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }
    if (doComp && mode_inv==10) // 2-4
      {
	// signed full precision dPhi12
	int dPhi24Sign = 1;
	// Unused variables
	// int CLCT2Sign = 1;
	// int CLCT4Sign = 1;
      
	if (dPhi24<0) dPhi24Sign = -1;
	// if (CLCT2<0) CLCT2Sign = -1;
	// if (CLCT4<0) CLCT4Sign = -1;
      
	// Make Pt LUT Address
	int dPhi24_ = fabs(dPhi24);
	int sign24_ = dPhi24Sign > 0 ? 1 : 0;
	int dTheta24_ = getdTheta(dTheta24);
	int CLCT2_ = getCLCT(CLCT2);
	int CLCT2Sign_ = CLCT2_ > 0 ? 1 : 0;
	CLCT2_ = abs(CLCT2_);
	int CLCT4_ = getCLCT(CLCT4);
	int CLCT4Sign_ = CLCT4_ > 0 ? 1 : 0;
	CLCT4_ = abs(CLCT4_);
	int FR2_ = FR2;
	int FR4_ = FR4;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi24_ & ((1<<9)-1))    << (0);
	Address += ( sign24_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta24_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT2_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT2Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT4_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT4Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR2_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR4_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }
    if (doComp && mode_inv==12) // 3-4
      {
	int dPhi34Sign = 1;
	// Unused variables
	// int CLCT3Sign = 1;
	// int CLCT4Sign = 1;
      
	if (dPhi34<0) dPhi34Sign = -1;
	// if (CLCT3<0) CLCT3Sign = -1;
	// if (CLCT4<0) CLCT4Sign = -1;
      
	// Make Pt LUT Address
	int dPhi34_ = fabs(dPhi34);
	int sign34_ = dPhi34Sign > 0 ? 1 : 0;
	int dTheta34_ = getdTheta(dTheta34);
	int CLCT3_ = getCLCT(CLCT3);
	int CLCT3Sign_ = CLCT3_ > 0 ? 1 : 0;
	CLCT3_ = abs(CLCT3_);
	int CLCT4_ = getCLCT(CLCT4);
	int CLCT4Sign_ = CLCT4_ > 0 ? 1 : 0;
	CLCT4_ = abs(CLCT4_);
	int FR3_ = FR3;
	int FR4_ = FR4;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi34_ & ((1<<9)-1))    << (0);
	Address += ( sign34_ & ((1<<1)-1))    << (0+9);
	Address += ( dTheta34_ & ((1<<3)-1))  << (0+9+1);
	Address += ( CLCT3_  & ((1<<2)-1))    << (0+9+1+3);
	Address += ( CLCT3Sign_ & ((1<<1)-1)) << (0+9+1+3+2);
	Address += ( CLCT4_  & ((1<<2)-1))    << (0+9+1+3+2+1);
	Address += ( CLCT4Sign_ & ((1<<1)-1)) << (0+9+1+3+2+1+2);
	Address += ( FR3_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1);
	Address += ( FR4_  & ((1<<1)-1))      << (0+9+1+3+2+1+2+1+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+9+1+3+2+1+2+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+9+1+3+2+1+2+1+1+1+5);
      }
       
    if (doComp && mode_inv==7) // 1-2-3
      {
	int dPhi12Sign = 1;
	int dPhi23Sign = 1;
	// Unused variables
	// int dPhi34Sign = 1;
	// int CLCT1Sign = 1;
      
	if (dPhi12<0) dPhi12Sign = -1;
	if (dPhi23<0) dPhi23Sign = -1;
	// if (dPhi34<0) dPhi34Sign = -1;
	// if (CLCT1<0) CLCT1Sign = -1;
      
	// Make Pt LUT Address
	int dPhi12_ = getNLBdPhiBin(dPhi12, 7, 512);
	int dPhi23_ = getNLBdPhiBin(dPhi23, 5, 256);
	int sign12_ = dPhi12Sign > 0 ? 1 : 0;
	int sign23_ = dPhi23Sign > 0 ? 1 : 0;
	int dTheta13_ = getdTheta(dTheta13);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int FR1_ = FR1;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi12_ & ((1<<7)-1))    << (0);
	Address += ( dPhi23_ & ((1<<5)-1))    << (0+7);
	Address += ( sign12_  & ((1<<1)-1))   << (0+7+5);
	Address += ( sign23_  & ((1<<1)-1))   << (0+7+5+1);
	Address += ( dTheta13_ & ((1<<3)-1))  << (0+7+5+1+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+7+5+1+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+7+5+1+1+3+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+7+5+1+1+3+2+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+7+5+1+1+3+2+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+7+5+1+1+3+2+1+1+5);
      }
	    
    if (doComp && mode_inv==11)
      {
	int dPhi12Sign = 1;
	int dPhi24Sign = 1;
	// Unused variable
	// int CLCT1Sign = 1;
      
	if (dPhi12<0) dPhi12Sign = -1;
	if (dPhi24<0) dPhi24Sign = -1;
	// if (CLCT1<0) CLCT1Sign = -1;
      
	// Make Pt LUT Address
	int dPhi12_ = getNLBdPhiBin(dPhi12, 7, 512);
	int dPhi24_ = getNLBdPhiBin(dPhi24, 5, 256);
	int sign12_ = dPhi12Sign > 0 ? 1 : 0;
	int sign24_ = dPhi24Sign > 0 ? 1 : 0;
	int dTheta14_ = getdTheta(dTheta14);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int FR1_ = FR1;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi12_ & ((1<<7)-1))    << (0);
	Address += ( dPhi24_ & ((1<<5)-1))    << (0+7);
	Address += ( sign12_ & ((1<<1)-1))    << (0+7+5);
	Address += ( sign24_ & ((1<<1)-1))    << (0+7+5+1);
	Address += ( dTheta14_ & ((1<<3)-1))  << (0+7+5+1+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+7+5+1+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+7+5+1+1+3+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+7+5+1+1+3+2+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+7+5+1+1+3+2+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+7+5+1+1+3+2+1+1+5);
      }
    if (doComp && mode_inv==13) // 1-3-4
      {
	int dPhi13Sign = 1;
	int dPhi34Sign = 1;
	// Unused variable
	// int CLCT1Sign = 1;
      
	if (dPhi13<0) dPhi13Sign = -1;
	if (dPhi34<0) dPhi34Sign = -1;
	// if (CLCT1<0) CLCT1Sign = -1;
      
	// Make Pt LUT Address
	int dPhi13_ = getNLBdPhiBin(dPhi13, 7, 512);
	int dPhi34_ = getNLBdPhiBin(dPhi34, 5, 256);
	int sign13_ = dPhi13Sign > 0 ? 1 : 0;
	int sign34_ = dPhi34Sign > 0 ? 1 : 0;
	int dTheta14_ = getdTheta(dTheta14);
	int CLCT1_ = getCLCT(CLCT1);
	int CLCT1Sign_ = CLCT1_ > 0 ? 1 : 0;
	CLCT1_ = abs(CLCT1_);
	int FR1_ = FR1;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi13_ & ((1<<7)-1))    << (0);
	Address += ( dPhi34_ & ((1<<6)-1))    << (0+7);
	Address += ( sign13_  & ((1<<1)-1))   << (0+7+5);
	Address += ( sign34_  & ((1<<1)-1))   << (0+7+5+1);
	Address += ( dTheta14_ & ((1<<3)-1))  << (0+7+5+1+1);
	Address += ( CLCT1_  & ((1<<2)-1))    << (0+7+5+1+1+3);
	Address += ( CLCT1Sign_ & ((1<<1)-1)) << (0+7+5+1+1+3+2);
	Address += ( FR1_  & ((1<<1)-1))      << (0+7+5+1+1+3+2+1);
	Address += ( eta_  & ((1<<5)-1))      << (0+7+5+1+1+3+2+1+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+7+5+1+1+3+2+1+1+5);
      }                  

    if (doComp && mode_inv==14) // 2-3-4
      {
	int dPhi23Sign = 1;
	int dPhi34Sign = 1;
	// Unused variables
	// int dEta24Sign = 1;
	// int CLCT2Sign = 1;
      
	if (dPhi23<0) dPhi23Sign = -1;
	if (dPhi34<0) dPhi34Sign = -1;
	// if (CLCT2<0) CLCT2Sign = -1;
      
	// Make Pt LUT Address
	int dPhi23_ = getNLBdPhiBin(dPhi23, 7, 512);
	int dPhi34_ = getNLBdPhiBin(dPhi34, 6, 256);
	int sign23_ = dPhi23Sign > 0 ? 1 : 0;
	int sign34_ = dPhi34Sign > 0 ? 1 : 0;
	int dTheta24_ = getdTheta(dTheta24);
	int CLCT2_ = getCLCT(CLCT2);
	int CLCT2Sign_ = CLCT2_ > 0 ? 1 : 0;
	CLCT2_ = abs(CLCT2_);
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi23_ & ((1<<7)-1))    << (0);
	Address += ( dPhi34_ & ((1<<6)-1))    << (0+7);
	Address += ( sign23_ & ((1<<1)-1))    << (0+7+6);
	Address += ( sign34_ & ((1<<1)-1))    << (0+7+6+1);
	Address += ( dTheta24_ & ((1<<3)-1))  << (0+7+6+1+1);
	Address += ( CLCT2_  & ((1<<2)-1))    << (0+7+6+1+1+3);
	Address += ( CLCT2Sign_ & ((1<<1)-1)) << (0+7+6+1+1+3+2);
	Address += ( eta_  & ((1<<5)-1))      << (0+7+6+1+1+3+2+1);
	Address += ( Mode_ & ((1<<4)-1))      << (0+7+6+1+1+3+2+1+5);
      }

    if (doComp && mode_inv==15) // 1-2-3-4
      {
	int dPhi12Sign = 1;
	int dPhi23Sign = 1;
	int dPhi34Sign = 1;
      
	if (dPhi12<0) dPhi12Sign = -1;
	if (dPhi23<0) dPhi23Sign = -1;
	if (dPhi34<0) dPhi34Sign = -1;
      
	if (dPhi12Sign==-1 && dPhi23Sign==-1 && dPhi34Sign==-1)
	  { dPhi12Sign=1;dPhi23Sign=1;dPhi34Sign=1;}
	else if (dPhi12Sign==-1 && dPhi23Sign==1 && dPhi34Sign==1)
	  { dPhi12Sign=1;dPhi23Sign=-1;dPhi34Sign=-1;}
	else if (dPhi12Sign==-1 && dPhi23Sign==-1 && dPhi34Sign==1)
	  { dPhi12Sign=1;dPhi23Sign=1;dPhi34Sign=-1;}
	else if (dPhi12Sign==-1 && dPhi23Sign==1 && dPhi34Sign==-1)
	  { dPhi12Sign=1;dPhi23Sign=-1;dPhi34Sign=1;}
      
	// Make Pt LUT Address
	int dPhi12_ = getNLBdPhiBin(dPhi12, 7, 512);
	int dPhi23_ = getNLBdPhiBin(dPhi23, 5, 256);
	int dPhi34_ = getNLBdPhiBin(dPhi34, 6, 256);
	int sign23_ = dPhi23Sign > 0 ? 1 : 0;
	int sign34_ = dPhi34Sign > 0 ? 1 : 0;
	int FR1_ = FR1;
	int eta_ = getEtaInt(TrackEta, 5);
	int Mode_ = mode_inv;
      
	Address += ( dPhi12_ & ((1<<7)-1)) << 0;
	Address += ( dPhi23_ & ((1<<5)-1)) << (0+7);
	Address += ( dPhi34_ & ((1<<6)-1)) << (0+7+5);
	Address += ( sign23_ & ((1<<1)-1)) << (0+7+5+6);
	Address += ( sign34_ & ((1<<1)-1)) << (0+7+5+6+1);
	Address += ( FR1_ & ((1<<1)-1))    << (0+7+5+6+1+1);
	Address += ( eta_ & ((1<<5)-1))    << (0+7+5+6+1+1+1);
	Address += ( Mode_ & ((1<<4)-1))   << (0+7+5+6+1+1+1+5);
      }

    return Address;
}


