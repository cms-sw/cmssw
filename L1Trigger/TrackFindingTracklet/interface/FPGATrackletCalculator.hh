//This class implementes the tracklet engine
#ifndef FPGATRACKLETCALCULATOR_H
#define FPGATRACKLETCALCULATOR_H

#include "FPGAProcessBase.hh"
#include "FPGAInverseTable.hh"

using namespace std;

class FPGATrackletCalculator:public FPGAProcessBase{

public:

  FPGATrackletCalculator(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=two_pi/NSector;
    phimin_=iSector*dphi;
    phimax_=phimin_+dphi;
    if (phimin_>0.5*two_pi) phimin_-=two_pi;
    if (phimax_>0.5*two_pi) phimax_-=two_pi;
    if (phimin_>phimax_)  phimin_-=two_pi;
    phioffset_=phimin_-dphi/6.0;

   trackletproj_L1PHI1_=0;
   trackletproj_L1PHI2_=0;
   trackletproj_L1PHI3_=0;

   trackletproj_L2PHI1_=0;
   trackletproj_L2PHI2_=0;
   trackletproj_L2PHI3_=0;

   trackletproj_L3PHI1_=0;
   trackletproj_L3PHI2_=0;
   trackletproj_L3PHI3_=0;

   trackletproj_L4PHI1_=0;
   trackletproj_L4PHI2_=0;
   trackletproj_L4PHI3_=0;

   trackletproj_L5PHI1_=0;
   trackletproj_L5PHI2_=0;
   trackletproj_L5PHI3_=0;

   trackletproj_L6PHI1_=0;
   trackletproj_L6PHI2_=0;
   trackletproj_L6PHI3_=0;

   trackletproj_L1Plus_=0; 
   trackletproj_L1Minus_=0;
                         
   trackletproj_L2Plus_=0; 
   trackletproj_L2Minus_=0;
                         
   trackletproj_L3Plus_=0; 
   trackletproj_L3Minus_=0;
                         
   trackletproj_L4Plus_=0; 
   trackletproj_L4Minus_=0;
                         
   trackletproj_L5Plus_=0; 
   trackletproj_L5Minus_=0;
                         
   trackletproj_L6Plus_=0; 
   trackletproj_L6Minus_=0;

   trackletproj_F1PHI1_=0;
   trackletproj_F1PHI2_=0;
   trackletproj_F1PHI3_=0;

   trackletproj_F2PHI1_=0;
   trackletproj_F2PHI2_=0;
   trackletproj_F2PHI3_=0;

   trackletproj_F3PHI1_=0;
   trackletproj_F3PHI2_=0;
   trackletproj_F3PHI3_=0;

   trackletproj_F4PHI1_=0;
   trackletproj_F4PHI2_=0;
   trackletproj_F4PHI3_=0;

   trackletproj_F5PHI1_=0;
   trackletproj_F5PHI2_=0;
   trackletproj_F5PHI3_=0;


   trackletproj_B1PHI1_=0;
   trackletproj_B1PHI2_=0;
   trackletproj_B1PHI3_=0;

   trackletproj_B2PHI1_=0;
   trackletproj_B2PHI2_=0;
   trackletproj_B2PHI3_=0;

   trackletproj_B3PHI1_=0;
   trackletproj_B3PHI2_=0;
   trackletproj_B3PHI3_=0;

   trackletproj_B4PHI1_=0;
   trackletproj_B4PHI2_=0;
   trackletproj_B4PHI3_=0;

   trackletproj_B5PHI1_=0;
   trackletproj_B5PHI2_=0;
   trackletproj_B5PHI3_=0;

   trackletproj_F1Plus_=0; 
   trackletproj_F1Minus_=0;
                         
   trackletproj_F2Plus_=0; 
   trackletproj_F2Minus_=0;
                         
   trackletproj_F3Plus_=0; 
   trackletproj_F3Minus_=0;
                         
   trackletproj_F4Plus_=0; 
   trackletproj_F4Minus_=0;
                         
   trackletproj_F5Plus_=0; 
   trackletproj_F5Minus_=0;

  
   trackletproj_B1Plus_=0; 
   trackletproj_B1Minus_=0;
                         
   trackletproj_B2Plus_=0; 
   trackletproj_B2Minus_=0;
                         
   trackletproj_B3Plus_=0; 
   trackletproj_B3Minus_=0;
                         
   trackletproj_B4Plus_=0; 
   trackletproj_B4Minus_=0;
                         
   trackletproj_B5Plus_=0; 
   trackletproj_B5Minus_=0;


   layer_=0;
   disk_=0;

   if (name_[3]=='L') layer_=name_[4]-'0';    
   if (name_[3]=='D') disk_=name_[4]-'0';    
   if (name_[3]=='F') disk_=name_[4]-'0';    
   if (name_[3]=='B') disk_=-(name_[4]-'0');

   assert((layer_!=0)||(disk_!=0));

   if (name_=="TC_D1L2A"||name_=="TC_D1L2B") {
     invRTable_.initR(9,round_int((rmean[2]-rmean[1])/kr),idrinvbits,false);
     return;
   }
     
   if (layer_!=0){
     //layer seeded
     invRTable_.initR(9,round_int((rmean[layer_]-rmean[layer_-1])/kr),idrinvbits,false);
// for Ed's math need 25
//   invRTable_.initR(9,round_int((rmean[layer_]-rmean[layer_-1])/kr),25,false);
     int region = name_[10]-'0';
     invTTable_.initT(10,2,28,(region>=3));
     if (writeInvTable) {
       string fname="InvRTable_"+name+".dat";     
       invRTable_.write(fname);
       fname="InvTTable_"+name+".dat";     
       invTTable_.write(fname);
     }
   }
   else if(name_[5] == 'D' ||name_[5] == 'F' || name_[5] == 'B'){
     //disk seeded
     invRTable_.initR(9,0,23,true);
     //bool pos = ((name_[3]=='D')||(name_[3]=='F'));
     invTTable_.initT(10,3,28,true);
     invTTableNeg_.initT(10,3,28,false);
     if (writeInvTable) {
       string fname="InvRTable_"+name+".dat";     
       invRTable_.write(fname);
       fname="InvTTable_"+name+".dat";     
       invTTable_.write(fname);
     }
     
   }

  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="trackpar"){
      FPGATrackletParameters* tmp=dynamic_cast<FPGATrackletParameters*>(memory);
      assert(tmp!=0);
      trackletpars_=tmp;
      return;
    }

    if (output=="projoutL1PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1PHI1_=tmp;
      return;
    }

    if (output=="projoutL1PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1PHI2_=tmp;
      return;
    }

    if (output=="projoutL1PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1PHI3_=tmp;
      return;
    }

    if (output=="projoutL2PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2PHI1_=tmp;
      return;
    }

    if (output=="projoutL2PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2PHI2_=tmp;
      return;
    }

    if (output=="projoutL2PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2PHI3_=tmp;
      return;
    }

    if (output=="projoutL3PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3PHI1_=tmp;
      return;
    }

    if (output=="projoutL3PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3PHI2_=tmp;
      return;
    }

    if (output=="projoutL3PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3PHI3_=tmp;
      return;
    }

    if (output=="projoutL4PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4PHI1_=tmp;
      return;
    }

    if (output=="projoutL4PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4PHI2_=tmp;
      return;
    }

    if (output=="projoutL4PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4PHI3_=tmp;
      return;
    }

    if (output=="projoutL5PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5PHI1_=tmp;
      return;
    }

    if (output=="projoutL5PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5PHI2_=tmp;
      return;
    }

    if (output=="projoutL5PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5PHI3_=tmp;
      return;
    }

    if (output=="projoutL6PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6PHI1_=tmp;
      return;
    }

    if (output=="projoutL6PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6PHI2_=tmp;
      return;
    }

    if (output=="projoutL6PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6PHI3_=tmp;
      return;
    }



    if (output=="projoutF1PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI1_=tmp;
      return;
    }

    if (output=="projoutF1PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI2_=tmp;
      return;
    }

    if (output=="projoutF1PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI3_=tmp;
      return;
    }

    if (output=="projoutF2PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI1_=tmp;
      return;
    }

    if (output=="projoutF2PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI2_=tmp;
      return;
    }

    if (output=="projoutF2PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI3_=tmp;
      return;
    }

    if (output=="projoutF3PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI1_=tmp;
      return;
    }

    if (output=="projoutF3PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI2_=tmp;
      return;
    }

    if (output=="projoutF3PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI3_=tmp;
      return;
    }

    if (output=="projoutF4PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI1_=tmp;
      return;
    }

    if (output=="projoutF4PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI2_=tmp;
      return;
    }

    if (output=="projoutF4PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI3_=tmp;
      return;
    }

    if (output=="projoutF5PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI1_=tmp;
      return;
    }

    if (output=="projoutF5PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI2_=tmp;
      return;
    }

    if (output=="projoutF5PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI3_=tmp;
      return;
    }



    if (output=="projoutD1PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI1_=tmp;
      return;
    }

    if (output=="projoutD1PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI2_=tmp;
      return;
    }

    if (output=="projoutD1PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1PHI3_=tmp;
      return;
    }

    if (output=="projoutD2PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI1_=tmp;
      return;
    }

    if (output=="projoutD2PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI2_=tmp;
      return;
    }

    if (output=="projoutD2PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2PHI3_=tmp;
      return;
    }

    if (output=="projoutD3PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI1_=tmp;
      return;
    }

    if (output=="projoutD3PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI2_=tmp;
      return;
    }

    if (output=="projoutD3PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3PHI3_=tmp;
      return;
    }

    if (output=="projoutD4PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI1_=tmp;
      return;
    }

    if (output=="projoutD4PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI2_=tmp;
      return;
    }

    if (output=="projoutD4PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4PHI3_=tmp;
      return;
    }

    if (output=="projoutD5PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI1_=tmp;
      return;
    }

    if (output=="projoutD5PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI2_=tmp;
      return;
    }

    if (output=="projoutD5PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5PHI3_=tmp;
      return;
    }



    
    if (output=="projoutB1PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B1PHI1_=tmp;
      return;
    }

    if (output=="projoutB1PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B1PHI2_=tmp;
      return;
    }

    if (output=="projoutB1PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B1PHI3_=tmp;
      return;
    }

    if (output=="projoutB2PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B2PHI1_=tmp;
      return;
    }

    if (output=="projoutB2PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B2PHI2_=tmp;
      return;
    }

    if (output=="projoutB2PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B2PHI3_=tmp;
      return;
    }

    if (output=="projoutB3PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B3PHI1_=tmp;
      return;
    }

    if (output=="projoutB3PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B3PHI2_=tmp;
      return;
    }

    if (output=="projoutB3PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B3PHI3_=tmp;
      return;
    }

    if (output=="projoutB4PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B4PHI1_=tmp;
      return;
    }

    if (output=="projoutB4PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B4PHI2_=tmp;
      return;
    }

    if (output=="projoutB4PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B4PHI3_=tmp;
      return;
    }

    if (output=="projoutB5PHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B5PHI1_=tmp;
      return;
    }

    if (output=="projoutB5PHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B5PHI2_=tmp;
      return;
    }

    if (output=="projoutB5PHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B5PHI3_=tmp;
      return;
    }



    
    if (output=="projoutL1ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1Minus_=tmp;
      return;
    }

    if (output=="projoutL1ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1Plus_=tmp;
      return;
    }

    if (output=="projoutL2ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2Minus_=tmp;
      return;
    }

    if (output=="projoutL2ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2Plus_=tmp;
      return;
    }

    if (output=="projoutL3ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Minus_=tmp;
      return;
    }

    if (output=="projoutL3ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Plus_=tmp;
      return;
    }

    if (output=="projoutL4ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Minus_=tmp;
      return;
    }

    if (output=="projoutL4ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Plus_=tmp;
      return;
    }

    if (output=="projoutL5ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Minus_=tmp;
      return;
    }

    if (output=="projoutL5ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Plus_=tmp;
      return;
    }

    if (output=="projoutL6ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Minus_=tmp;
      return;
    }

    if (output=="projoutL6ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Plus_=tmp;
      return;
    }

    if (output=="projoutL3D4ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Minus_=tmp;
      return;
    }

    if (output=="projoutL3D4ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Plus_=tmp;
      return;
    }

    if (output=="projoutL4D3ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Minus_=tmp;
      return;
    }

    if (output=="projoutL4D3ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Plus_=tmp;
      return;
    }

    if (output=="projoutL5D2ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Minus_=tmp;
      return;
    }

    if (output=="projoutL5D2ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Plus_=tmp;
      return;
    }

    if (output=="projoutL6D1ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Minus_=tmp;
      return;
    }

    if (output=="projoutL6D1ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Plus_=tmp;
      return;
    }


    if (output=="projoutD1ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1Plus_=tmp;
      return;
    }

    if (output=="projoutD2ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2Plus_=tmp;
      return;
    }

    if (output=="projoutD3ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3Plus_=tmp;
      return;
    }

    if (output=="projoutD4ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4Plus_=tmp;
      return;
    }

    if (output=="projoutD5ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5Plus_=tmp;
      return;
    }    
    

    if (output=="projoutB1ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B1Plus_=tmp;
      return;
    }

    if (output=="projoutB2ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B2Plus_=tmp;
      return;
    }

    if (output=="projoutB3ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B3Plus_=tmp;
      return;
    }

    if (output=="projoutB4ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B4Plus_=tmp;
      return;
    }

    if (output=="projoutB5ToPlus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B5Plus_=tmp;
      return;
    }    
    

    if (output=="projoutD1ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F1Minus_=tmp;
      return;
    }

    if (output=="projoutD2ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F2Minus_=tmp;
      return;
    }

    if (output=="projoutD3ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F3Minus_=tmp;
      return;
    }

    if (output=="projoutD4ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F4Minus_=tmp;
      return;
    }

    if (output=="projoutD5ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_F5Minus_=tmp;
      return;
    }    
    

    if (output=="projoutB1ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B1Minus_=tmp;
      return;
    }

    if (output=="projoutB2ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B2Minus_=tmp;
      return;
    }

    if (output=="projoutB3ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B3Minus_=tmp;
      return;
    }

    if (output=="projoutB4ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B4Minus_=tmp;
      return;
    }

    if (output=="projoutB5ToMinus"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_B5Minus_=tmp;
      return;
    }    
    





    cout << "Could not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="innerallstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      innerallstubs_=tmp;
      return;
    }
    if (input=="outerallstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      outerallstubs_=tmp;
      return;
    }
    if (input=="stubpair1in"||
	input=="stubpair2in"||
	input=="stubpair3in"||
	input=="stubpair4in"||
	input=="stubpair5in"||
	input=="stubpair6in"||
	input=="stubpair7in"||
	input=="stubpair8in"||
	input=="stubpair9in"||
	input=="stubpair10in"||
	input=="stubpair11in"||
	input=="stubpair12in"||
	input=="stubpair13in"||
	input=="stubpair14in"||
	input=="stubpair15in"||
	input=="stubpair16in"||
	input=="stubpair17in"||
	input=="stubpair18in"||
	input=="stubpair19in"||
	input=="stubpair20in"||
	input=="stubpair21in"||
	input=="stubpair22in"||
	input=="stubpair23in"||
	input=="stubpair24in"||
	input=="stubpair25in"||
	input=="stubpair26in"||
	input=="stubpair27in"||
	input=="stubpair28in"||
	input=="stubpair29in"||
	input=="stubpair30in"||
	input=="stubpair31in"||
	input=="stubpair32in"||
	input=="stubpair33in"||
	input=="stubpair34in"||
	input=="stubpair35in"||
	input=="stubpair36in"||
	input=="stubpair37in"||
	input=="stubpair38in"||
	input=="stubpair39in"||
	input=="stubpair40in"||
	input=="stubpair41in"||
	input=="stubpair42in"){
      FPGAStubPairs* tmp=dynamic_cast<FPGAStubPairs*>(memory);
      assert(tmp!=0);
      stubpairs_.push_back(tmp);
      return;
    }
    assert(0);
  }


  void exacttracklet(double r1, double z1, double phi1,
		     double r2, double z2, double phi2, double sigmaz,
		     double& rinv, double& phi0,
		     double& t, double& z0,
		     double phiproj[4], double zproj[4], 
		     double phider[4], double zder[4],
		     double phiprojdisk[5], double rprojdisk[5], 
		     double phiderdisk[5], double rderdisk[5]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    assert(fabs(phi0)<0.5*two_pi);
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;

    for (int i=0;i<4;i++) {
      exactproj(rproj_[i],rinv,phi0,t,z0,
		phiproj[i],zproj[i],phider[i],zder[i]);
    }

    for (int i=0;i<5;i++) {
      int sign=1;
      if (t<0) sign=-1;
      exactprojdisk(sign*zmean[i],rinv,phi0,t,z0,
		phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
    }



  }


  void exacttrackletdisk(double r1, double z1, double phi1,
			 double r2, double z2, double phi2, double sigmaz,
			 double& rinv, double& phi0,
			 double& t, double& z0,
			 double phiprojLayer[3], double zprojLayer[3], 
			 double phiderLayer[3], double zderLayer[3],
			 double phiproj[3], double rproj[3], 
			 double phider[3], double rder[3]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    //cout << "phi1 phi2 phi1tmp : "<<phi1<<" "<<phi2<<" "<<phi1tmp<<endl;

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    if (!(fabs(phi0)<0.5*two_pi)) {
      cout << "phi1tmp r1 rinv phi0 deltaphi dist: "
	   <<phi1tmp<<" "<<r1<<" "<<rinv<<" "<<phi0
	   <<" "<<deltaphi<<" "<<dist<<endl;
      exit(1);
    }
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;


    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<3;i++) {
      exactprojdisk(zproj_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<3;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }


  void ed_approxtracklet(double r1, double z1, double phi1,
		      double r2, double z2, double phi2, double sigmaz,
		      double &rinv, double &phi0,
		      double &t, double &z0,
		      double phiproj[4], double zproj[4], 
		      double phider[4], double zder[4],
		      double phiprojdisk[5], double rprojdisk[5], 
		      double phiderdisk[5], double rderdisk[5]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }
    double deltaphi = 0;

    if (phi1<0.0) phi1+=two_pi;
    deltaphi=phi2-phi1;
    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);
    
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    assert(phi1tmp>-1e-10);
    double dr=r2-r1;
    double dz=z2-z1;
    double drinv=1.0/dr;
    double delta0 = deltaphi*drinv;
    double delta1 = r1 * delta0;
    double delta2 = r2 * delta0;
    double deltaZ = dz * drinv;
    double a2 = 2-delta1*delta2;
    rinv=-delta0*a2;
    phi0=phi1tmp-delta1*(1-delta0*delta2*(r2+r1)/6.);
    t= deltaZ*a2/2.;
    z0=z1-r1*deltaZ;

    // outapprox<<std::setprecision(4)<<outapprox_pt<<"\t"<<outapprox_eta<<"\t"<<outapprox_phi<<"\t"<<tmp_outapprox_phi<<"\t"<<outapprox_z0<<"\t"
    // 	     <<r1<<"\t"<<z1<<"\t"<<r2<<"\t"<<z2<<"\t"
    // 	     <<rinv<<"\t"<<phi0<<"\t"<<t<<"\t"<<z0<<"\t";

    //calculate projection
    for (int i=0;i<4;i++) {
      // double tmp1,tmp2, tmp3, tmp4;
      // exactproj(rproj_[i], outapprox_rinv, tmp_outapprox_phi, outapprox_t, outapprox_z0,
      // 		 tmp1,tmp2,tmp3,tmp4);
      ed_approxproj(rproj_[i],rinv,phi0,t,z0,
		 phiproj[i],zproj[i],phider[i],zder[i]);

      // outapprox<<setprecision(4)<<asin(sin(phiproj[i]-tmp1))<<"\t"<<zproj[i]-tmp2<<"\t"<<phider[i]-tmp3<<"\t"<<zder[i]-tmp4<<"\t" ;

      if (writeNeighborProj) {
	static ofstream out1("neighborproj.txt");
	if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	  double phipr=phiproj[i];
	  if (phipr>0.5*two_pi) phipr-=two_pi;
	  if (phipr<-0.5*two_pi) phipr+=two_pi;
	  if ((fabs(zproj[i])<270.0)&&(phipr<(phimax_-phimin_)/6)){
	    out1<<layer_<<" -1 "<<phipr<<endl;
	  } else  if ((fabs(zproj[i])<270.0)&&(phipr>7.0*(phimax_-phimin_)/6)){
	    out1<<layer_<<" +1 "<<phipr<<endl;
	  } else if (fabs(zproj[i])<270.0){
	    out1<<layer_<<" 0 "<<phipr<<endl;
	  }
	}
      }
    }
    

    for (int i=0;i<5;i++) {
      int sign=1;
      if (t<0) sign=-1;
      // double tmp1,tmp2, tmp3, tmp4;
      // exactprojdisk(sign*zmean[i], outapprox_rinv, tmp_outapprox_phi, outapprox_t, outapprox_z0,
      // 		     tmp1,tmp2,tmp3,tmp4);
      ed_approxprojdisk(sign*zmean[i],rinv,phi0,t,z0,
		     phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);

      // outapprox<<setprecision(4)<<asin(sin(phiprojdisk[i]-tmp1))<<"\t"<<rprojdisk[i]-tmp2<<"\t"<<phiderdisk[i]-tmp3<<"\t"<<rderdisk[i]-tmp4<<"\t" ;
      //cout << "DUMPDISKPROJ1: "<<i<<" "<<rprojdisk[i]
      //	   <<" t="<<t<<" z0="<<z0<<" zdisk="<<zmean[i]<<endl;

    }
    
    // outapprox<<"\n";

  }



  void approxtracklet(double r1, double z1, double phi1,
		      double r2, double z2, double phi2, double sigmaz,
		      double &rinv, double &phi0,
		      double &t, double &z0,
		      double phiproj[4], double zproj[4], 
		      double phider[4], double zder[4],
		      double phiprojdisk[5], double rprojdisk[5], 
		      double phiderdisk[5], double rderdisk[5]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
    assert(phi1tmp>-1e-10);

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;//*(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;//-2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;//*(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;//*(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;//+3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (layer_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS1:" 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" "<<r2-r1
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" | "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" "<<phi0
	     <<" * "<<t12
	     <<" "<<z0
	     <<endl;
      }
      /*

      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
	   <<" delta="<<delta
	   <<" t5="<<t5
           <<" deltainv="<<deltainv
	   <<" rinv="<<rinv
	   <<" t="<<t
	   <<" r1abs="<<r1
	   <<" t7="<<t7
	   <<" t9="<<t9
	   <<" phi1="<<phi1tmp
	   <<" ***phi0="<<phi0
	   <<" t12="<<t12
	   <<" z1="<<z1
	   <<" z0="<<z0
	   <<endl;

    */

    }
   

    //calculate projection

    for (int i=0;i<4;i++) {
      approxproj(rproj_[i],rinv,phi0,t,z0,
		 phiproj[i],zproj[i],phider[i],zder[i]);
      if (writeNeighborProj) {
	static ofstream out1("neighborproj.txt");
	if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	  double phipr=phiproj[i];
	  if (phipr>0.5*two_pi) phipr-=two_pi;
	  if (phipr<-0.5*two_pi) phipr+=two_pi;
	  if ((fabs(zproj[i])<270.0)&&(phipr<(phimax_-phimin_)/6)){
	    out1<<layer_<<" -1 "<<phipr<<endl;
	  } else  if ((fabs(zproj[i])<270.0)&&(phipr>7.0*(phimax_-phimin_)/6)){
	    out1<<layer_<<" +1 "<<phipr<<endl;
	  } else if (fabs(zproj[i])<270.0){
	    out1<<layer_<<" 0 "<<phipr<<endl;
	  }
	}
      }
    }
    

    for (int i=0;i<5;i++) {
      int sign=1;
      if (t<0) sign=-1;
      approxprojdisk(sign*zmean[i],rinv,phi0,t,z0,
		     phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
      //cout << "DUMPDISKPROJ1: "<<i<<" "<<rprojdisk[i]
      //	   <<" t="<<t<<" z0="<<z0<<" zdisk="<<zmean[i]<<endl;

    }

  }

  bool ed_binarytracklet(FPGAStub* innerFPGAStub, 
		      FPGAStub* outerFPGAStub,
		      double sigmaz,
		      int& irinv, int& iphi0,
		      int& it, int& iz0,
		      int iphiproj[4], int izproj[4],
		      int iphider[4], int izder[4],
		      bool minusNeighbor[4], bool plusNeighbor[4],
		      int iphiprojdisk[5], int irprojdisk[5],
		      int iphiderdisk[5], int irderdisk[5],
		      bool minusNeighborDisk[5], bool plusNeighborDisk[5]){

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //Simplify to work in common number of bits for all layers
    if (layer_<4) iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    if (layer_<3) iphi2<<=(nbitsphistubL456-nbitsphistubL123);

    if (layer_<4) ir1<<=(nbitsrL456-nbitsrL123);
    if (layer_<3) ir2<<=(nbitsrL456-nbitsrL123);

    if (layer_>3) iz1<<=(nbitszL123-nbitszL456);
    if (layer_>2) iz2<<=(nbitszL123-nbitszL456);

    int r1=ir1;
    int phi1=iphi1;
    int z1=iz1;

    int r2=ir2;
    int phi2=iphi2;
    int z2=iz2;

    //and now - the calculation

    // inv_rho math:
    long int dphi = phi2 - phi1;                 // unsigned=0, Nbits=17
    long int dr = r2 - r1;                       // unsigned=0, Nbits=14
    long int r1mean =  round_int(rmean[layer_-1]/kr);
    long int r2mean =  round_int(rmean[layer_]/kr);
    //long int dr12 = r2mean - r1mean;             // unsigned=0, Nbits=14
    //long int drabs = dr + dr12;                  // unsigned=0, Nbits=12
    //long int invdr = (1<<25)/drabs;              // unsigned=0, Nbits=17
    //cout << "will calc. invdr dr = " <<dr<< endl;
    long int invdr = invRTable_.lookup((dr&511));
    //cout << "done calc. invdr" << endl;

    long int delta_0 = (dphi * invdr)>>16;       // unsigned=0, Nbits=17
    long int r1abs = r1 + r1mean;                // unsigned=0, Nbits=14
    long int delta_1 = (delta_0 * r1abs)>>14;    // unsigned=0, Nbits=17
    long int r2abs = r2 + r2mean;                // unsigned=0, Nbits=14
    long int delta_2 = (delta_0 * r2abs)>>14;    // unsigned=0, Nbits=17
    long int a2a = (delta_1 * delta_2)>>17;      // unsigned=0, Nbits=17
    long int a2mA = (a2a*1466)>>12;              // unsigned=0, Nbits=17
    long int a2m = -1024 + a2mA;                 // unsigned=0, Nbits=12
    long int inv_rho = (delta_0 * a2m)>>11;      // unsigned=0, Nbits=14

    // phi0 math:
    long int R = r1 + r2;                        // unsigned=0, Nbits=14
    long int r12 = r2mean + r1mean;              // unsigned=0, Nbits=14
    long int Rabs = R + r12;                     // unsigned=0, Nbits=15
    long int R6 = (682 * Rabs)>>12;              // unsigned=0, Nbits=13
    long int x4 = (delta_0 * R6)>>13;            // unsigned=0, Nbits=17
    long int x6a = (delta_2 * x4)>>17;           // unsigned=0, Nbits=17
    long int x6mA = (x6a*1466)>>12;              // unsigned=0, Nbits=17
    long int x6m = -1024 + x6mA;                 // unsigned=0, Nbits=12
    long int phi0a = (delta_1 * x6m)>>5;         // unsigned=0, Nbits=17
    long int phi0s = phi1 + phi0a;               // unsigned=0, Nbits=18
    long int phi0  = phi0s>>1;                   // unsigned=0, Nbits=18

    // t math:
    long int dz = z2 - z1;                       // unsigned=0, Nbits=11
    long int delta_z = (dz * invdr)>>11;         // unsigned=0, Nbits=17
    long int x3 = (delta_z)>>1;                  // unsigned=0, Nbits=16
    long int a2 = -1 * a2m;                      // unsigned=0, Nbits=12
    long int t = (x3 * a2)>>12;                  // unsigned=0, Nbits=14

    // z0 math:
    long int z0a = (delta_z * r1abs)>>14;        // undigned=0, Nbits=17
    long int z0 = z1 - z0a;                      // undigned=0, Nbits=10
    
    //output tracklet parameters:
    irinv = inv_rho; //K = 1.31423e+06
    iphi0 = phi0;    //K = 219038
    it = t;          //K = 854.817 
    iz0 = z0;        //K = 17.8087

    //test
    if (fabs(irinv*krinvpars)>rinvcut) {
      //cout << "Failed tracklet pt cut"<<endl;
      return false;
    }
    if (fabs(iz0*kzpars)>z0cut) { 
        //cout << "Failed tracklet z0 cut"<<endl;
      return false;
    }

    //calculate projections
    for (int i=0;i<4;i++) {
      int rp = rproj_[i]/kr;
      ed_binaryproj(rp,phi0s,it,iz0,
		    iphiproj[i],izproj[i],iphider[i],izder[i],
		    minusNeighbor[i], plusNeighbor[i],
		    //extras
		    delta_0, a2m, a2);
    }


    long int x2 = (delta_0)>>1;  //to pass to disk proj
    for (int i=0;i<5;i++) {
      int sign=1;
      if (it<0) sign=-1;
      int zp = (sign*zmean[i])/kz;
      ed_binaryprojdisk(zp,phi0s,it,iz0,
			iphiprojdisk[i],irprojdisk[i],iphiderdisk[i],irderdisk[i],
			minusNeighborDisk[i], plusNeighborDisk[i],
			//extras
			x2, a2);
    }

    return true;
  }

  void ed_binaryproj(int rp, int iphi0, int it, int iz0,
		     int &iphiproj, int &izproj, int &iphider, int &izder,
		     bool &minusNeighbor, bool &plusNeighbor,
		     //plus some internal stuff from tracklet par calculations
		     long int delta_0, long int a2m, long int a2
		     ) 
  {

    //wire inputs
    int phi0s = iphi0;
    int t     = it;
    int z0    = iz0;

    // layer proj math:

    long int x2 = (delta_0)>>1;                  // unsigned=0, Nbits=16
    long int x1 = (x2 * rp)>>12;                 // unsigned=0, Nbits=17
    long int x8 = (x1 * a2m)>>8;                 // unsigned=0, Nbits=17
    long int x20 = (682 * x8)>>12;               // unsigned=0, Nbits=15
    long int x12 = (x8 * x8)>>24;                // unsigned=0, Nbits=10
    long int x10A = (x12*1466)>>12;              // unsigned=0, Nbits=10
    long int x10 = 1536 + x10A;                  // unsigned=0, Nbits=12
    long int x22 = (x20 * x10)>>6;               // unsigned=0, Nbits=17
    long int phiL = phi0s - x22;                 // unsigned=0, Nbits=19

    long int x11 = (rp * t)>>7;                  // unsigned=0, Nbits=17
    long int x21 = (682 * x11)>>12;              // unsigned=0, Nbits=15
    long int x23 = (x21 * x10)>>12;              // unsigned=0, Nbits=15
    long int zL = z0 + x23;                      // unsigned=0, Nbits=12

    long int der_phiL = (x2 * a2)>>18;           // unsigned=0, Nbits=6
    
    long int der_zL = t>>6;                      // unsigned=0, Nbits=7


    //wire outputs
    iphiproj = phiL;     //K = 438076         (if rp<60, 1/8);
    izproj   = zL;       //K = 17.8087        (if rp>60, 1/16);
    iphider  = der_phiL; // K =  10267.4
    izder    = der_zL;   // K = 13.3565


    //from the original
    minusNeighbor=false;
    plusNeighbor=false;

    if (iphiproj<(1<<nbitsphistubL456)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL456)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL456)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL456)/4;
    }

    // if(iphiproj>(1<<nbitsphistubL456)||iphiproj<0){
    //   cout<<iphiproj<<"\t"<<outapprox_pt<<"\t"<<outapprox_eta<<"\t"<<minusNeighbor<<" "<<plusNeighbor<<"\n";
    // }
    //assert(iphiproj>=0);
    //assert(iphiproj<(1<<nbitsphistubL456));

    float rproj = rp * kr;
    if (rproj<60.0) iphiproj>>=(nbitsphistubL456-nbitsphistubL123);

    if (izproj<-(1<<(nbitszprojL123-1))) izproj=-(1<<(nbitszprojL123-1));
    if (izproj>=(1<<(nbitszprojL123-1))) izproj=(1<<(nbitszprojL123-1))-1;

    if (rproj>60.) {
      izproj>>=(nbitszprojL123-nbitszprojL456);
    }

  }

  void ed_binaryprojdisk(int zp, int iphi0, int it, int iz0,
			 int &iphiproj, int &irproj, int &iphider, int &irder,
			 bool &minusNeighbor, bool &plusNeighbor,
			 //plus extra from tracklet pars calculation
			 long int x2, long int a2 ) 
  {

    //wire inputs
    int phi0s = iphi0;
    int t    = it;
    if(t==0) t = 1;
    int z0 = iz0;
    
    // disk proj math:

    long int x5 = zp - z0;                  // unsigned=0, Nbits=15
    long int x7 = (x2 * a2)>>7;             // unsigned=0, Nbits=17
    long int x13 = (x5 * x7)>>13;           // unsigned=0, Nbits=17
    //long int invt = (1<<28)/t;              // unsigned=0, Nbits=20
    int ntshift = layer_ > 0? 2 : 3;
    long int invt = 0;
    if (it>0) {
      invt=invTTable_.lookup((t>>ntshift)&1023);
    } else {
      invt=invTTableNeg_.lookup((t>>ntshift)&1023);
    }    
    long int x25 = (x13 * invt)>>18;        // unsigned=0, Nbits=17
    long int phiDs = phi0s + (x25<<3);      // unsigned=0, Nbits=19
    long int phiD = phiDs>>3;               // unsigned=0, Nbits=16

    long int x9 = (682 * x5)>>12;           // unsigned=0, Nbits=13
    long int x24 = (x9 * invt)>>16;         // unsigned=0, Nbits=17
    long int x26 = (x25 * x25)>>24;         // unsigned=0, Nbits=10
    long int x27A = (x26*1466)>>12;         // unsigned=0, Nbits=10
    long int x27 = -384 + x27A;             // unsigned=0, Nbits=11
    long int x27m = -1 * x27;               // unsigned=0, Nbits=11
    long int rD = (x24 * x27m)>>9;          // unsigned=0, Nbits=12    ...Anders changed this...

    long int der_phiD = (invt * x7)>>29;    // unsigned=0, Nbits=6
    long int der_rD = invt>>12;             // unsigned=0, Nbits=8 


    //wire outputs
    iphiproj = phiD; //K = 54759.5
    irproj   = rD;   //K =  10.6667   
    iphider  = der_phiD; // K = 12299.5
    irder    = der_rD;   // K = 76.6667


    //from original
    minusNeighbor=false;
    plusNeighbor=false;
    if (iphiproj<(1<<nbitsphistubL123)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL123)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL123)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL123)/4;
    }

    if (iphiproj<0) iphiproj=0;
    if (iphiproj>=(1<<nbitsphistubL123)) iphiproj=(1<<nbitsphistubL123)-1;
    if(iphider>=(1<<6)) iphider = (1<<6)-1;
    if(iphider<-(1<<6)) iphider = -(1<<6);
    if(irder>=(1<<7)) irder = (1<<7)-1;
    if(irder<-(1<<7)) irder = -(1<<7);


    if (irproj<=0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }

    assert(irproj>0);

    if (irproj*krprojshiftdisk>120.0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }
    
  }

  bool ed_binarytrackletdisk(FPGAStub* innerFPGAStub, 
			  FPGAStub* outerFPGAStub,
			  double sigmaz,
			  int& irinv, int& iphi0,
			  int& it, int& iz0,
			  int iphiprojLayer[6], int izprojLayer[6],
			  int iphiderLayer[6], int izderLayer[6],
			  bool minusNeighborLayer[6], bool plusNeighborLayer[6],
			  int iphiproj[4], int irproj[4],
			  int iphider[4], int irder[4],
			  bool minusNeighbor[4], bool plusNeighbor[4]){
    
    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //To get same precission as for layers.
    iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    iphi2<<=(nbitsphistubL456-nbitsphistubL123);

    //parameters
    int rmin  = round_int(rmindisk / kr);
    int sign= (disk_>0)? 1: -1;
    int z2mean = sign*zmean[abs(disk_)]/kzdisk;
    int z1mean = sign*zmean[abs(disk_)-1]/kzdisk;
    int dz12 = z2mean - z1mean;             // undigned=0, Nbits=15

    //wire inputs
    int phi2  = iphi2;
    int phi1  = iphi1;
    int r2    = ir2<<1;
    int r1    = ir1<<1;
    int z2rel = iz2;
    int z1rel = iz1;

    // inv_rho math:
    long int dphi = phi2 - phi1;                 // unsigned=0, Nbits=17
    long int dr = r2 - r1;                       // unsigned=0, Nbits=11
    //    long int invdr = (1<<23)/dr;                 // unsigned=0, Nbits=17
    //cout << "will calc. invdr 2" << endl;
    long int invdr = invRTable_.lookup((dr&511));
    //cout << "done calc. invdr 2" << endl;

    
    long int delta_0 = (dphi * invdr)>>14;       // unsigned=0, Nbits=17
    long int r1abs = r1 + rmin;                  // unsigned=0, Nbits=14
    long int delta_1 = (delta_0 * r1abs)>>14;    // unsigned=0, Nbits=17
    long int r2abs = r2 + rmin;                  // unsigned=0, Nbits=14
    long int delta_2 = (delta_0 * r2abs)>>14;    // unsigned=0, Nbits=17
    long int a2a = (delta_1 * delta_2)>>17;      // unsigned=0, Nbits=17
    long int a2mA = (a2a*1466)>>12;              // unsigned=0, Nbits=17
    long int a2m = -1024 + a2mA;                 // unsigned=0, Nbits=12
    long int inv_rho = (delta_0 * a2m)>>11;      // unsigned=0, Nbits=14

    // phi0 math:
    long int R = r1abs + r2abs;                  // unsigned=0, Nbits=15
    long int R6 = (682 * R)>>12;                 // unsigned=0, Nbits=13
    long int x4 = (delta_0 * R6)>>13;            // unsigned=0, Nbits=17
    long int x6a = (delta_2 * x4)>>17;           // unsigned=0, Nbits=17
    long int x6mA = (x6a*1466)>>12;              // unsigned=0, Nbits=17
    long int x6m = -1024 + x6mA;                 // unsigned=0, Nbits=12
    long int phi0a = (delta_1 * x6m)>>5;         // unsigned=0, Nbits=17
    long int phi0s = phi1 + phi0a;               // unsigned=0, Nbits=18
    long int phi0 = phi0s>>1;                    // unsigned=0, Nbits=18


    // t math:
    long int dzrel = z2rel - z1rel;              // unsigned=0, Nbits=15
    long int dz = dzrel + dz12;                  // unsigned=0, Nbits=11
    long int delta_z = (dz * invdr)>>10;         // unsigned=0, Nbits=17
    long int x3 = (delta_z)>>1;                  // unsigned=0, Nbits=16
    long int a2 = -1 * a2m;                      // unsigned=0, Nbits=12
    long int t = (x3 * a2)>>11;                  // unsigned=0, Nbits=14

    // z0 math:
    long int z1 = z1rel + z1mean;                // unsigned=0, Nbits=15
    long int z0a = (delta_z * r1abs)>>13;        // unsigned=0, Nbits=15
    long int z0 = z1 - z0a;                      // unsigned=0, Nbits=10

    //output tracklet parameters:
    irinv = inv_rho; //K = 1.31423e+06 
    iphi0 = phi0;    //K = 219038
    it = t;          //K = 854.817
    iz0 = z0;        //K = 17.8087

   
    if (fabs(iz0*kzdisk)>z0cut) {
      //cout << "DUMP iz0 too large: "<<iz0*kzdisk<<endl;
      return false;
    }
    //if (fabs(irinv*krinvparsdisk)>0.0057) {
    if (fabs(irinv*krinvparsdisk)>rinvcut) {
      //cout << "DUMP irinv too large: "<<irinv*krinvparsdisk<<endl;
      return false;
    }

    //calculate projections
    for (int i=0;i<3;i++) {
      int rp = rmean[i]/kr;
      ed_binaryproj(rp,phi0s,it,iz0,
		    iphiprojLayer[i],izprojLayer[i],iphiderLayer[i],izderLayer[i],
		    minusNeighborLayer[i], plusNeighborLayer[i],
		    //extras
		    delta_0, a2m, a2);
    }

    long int x2 = (delta_0)>>1;  //to pass to disk proj
    for (int i=0;i<3;i++) {
      int zp = zproj_[i]/kz;
      ed_binaryprojdisk(zp,phi0s,it,iz0,
			iphiproj[i],irproj[i],iphider[i],irder[i],
			minusNeighbor[i], plusNeighbor[i],
			//extras
			x2, a2);
    }

    //it/=2; //Big hack.... should be corrected earlier, but need to change ed_binaryprojdisk
    
    return true;

 }


  bool binarytracklet(FPGAStub* innerFPGAStub, 
		      FPGAStub* outerFPGAStub,
		      double sigmaz,
		      int& irinv, int& iphi0,
		      int& it, int& iz0,
		      int iphiproj[4], int izproj[4],
		      int iphider[4], int izder[4],
		      bool minusNeighbor[4], bool plusNeighbor[4],
		      int iphiprojdisk[5], int irprojdisk[5],
		      int iphiderdisk[5], int irderdisk[5],
		      bool minusNeighborDisk[5], bool plusNeighborDisk[5]){

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    //cout << "Layer : "<<layer_<<" "<<innerFPGAStub->str()
    //	 <<" "<<outerFPGAStub->str()<<endl;

    //double phi1tmp=innerFPGAStub.phitmp();
    //double phimintmp=innerFPGAStub.phimin();
 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    int layer=layer_;

    if (getName()=="TC_D1L2A"||getName()=="TC_D1L2B") {
      layer=2;
    }
    
    //Simplify to work in common number of bits for all layers
    if (layer<4) iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    if (layer<3) iphi2<<=(nbitsphistubL456-nbitsphistubL123);

    if (layer<4) ir1<<=(nbitsrL456-nbitsrL123);
    if (layer<3) ir2<<=(nbitsrL456-nbitsrL123);

    if (layer>3) iz1<<=(nbitszL123-nbitszL456);
    if (layer>2) iz2<<=(nbitszL123-nbitszL456);


    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1; 
    assert(abs(ideltaphi)<(1<<15));
    //step 2
    int idz=iz2-iz1;
    assert(abs(idz)<(1<<11));
    //step 3
    int idrrel=ir2-ir1;
    assert(abs(idrrel)<(1<<8));
    //step 4
    int ir1abs=round_int(rmean[layer-1]/kr)+ir1;
    assert(ir1abs<(1<<13));
    //step 5
    int ir2abs=round_int(rmean[layer]/kr)+ir2;
    assert(ir2abs<(1<<13));
    //step 6
    //    int idrinv=invTable_.lookup(idrrel&((1<<9)-1)); //Take top 9 bits
    //cout << "will calc. invdr 3" << endl;
    int idrinv=invRTable_.lookup(idrrel&((1<<9)-1));
    //cout << "will calc. invdr 3" << endl;

//Same result, but compatible with new LUTs
    // int nmore = 25-idrinvbits;
    // idrinv = idrinv>>nmore;

    //int idr=round_int((rmean[layer_]-rmean[layer_-1])/kr)+idrrel; 
    //int idrinvtmp=round_int((1<<idrinvbits)/(1.0*idr));  //not actually using idrinvbits since 
                                     //idr is largish. Implement in lookup 
                                     //table, just a fcn of ir1-ir2
                                     //for a given layer
    //cout << "idrrel idrinv= "<<idrrel<<" "<<(idrrel&((1<<9)-1))
    //	 <<" "<<idrinv<<" "<<idrinvtmp<<endl;
    //assert(idrinv==idrinvtmp);

    assert(idrinv<(1<<12));  //11
    //step 7
    int it1=(ir1abs*ir2abs)>>it1shift;
    assert(it1<(1<<2));
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shift;
    assert(abs(it2)<(1<<10));
    //step 9
    int it3=(it2*it2)>>it3shift;
    assert(it3<(1<<6));
    assert(it3>=0);
    //step 10
    int idelta=0.5*it1*it3;
    assert(idelta<(1<<6));
    assert(idelta>=0);
    //step 11
    int ideltatmp=round_int(idelta*(kdelta*(1<<it4bits)));
    assert(ideltatmp<(1<<6));
    //step 12
    int it4=(1<<it4bits)-3*(ideltatmp>>1);
    assert(it4<(1<<(it4bits+1)));
    //step 13    
    int it5=(1<<it5bits)-((ideltatmp*it4)>>(2*it4bits-it5bits));
    assert(it5<(1<<(it5bits+1)));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bits;  
    assert(fabs(iDeltainv)<(1<<12));  //11
    //step 15
    assert(rinvbitshift>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshift-1)); //-1 because of *2 
    assert(fabs(irinv)<(1<<14));
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    assert(fabs(it)<(1<<12));
    //step 17
    assert(it7shift+irinvshift-rinvbitshift>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shift+irinvshift-rinvbitshift);
    assert(fabs(it7)<(1<<17));
    //step 18
    int it7tmp=(it7*it7tmpfactor)>>it7tmpshift;
    assert(fabs(it7tmp)<(1<<7));
    //step 19
    int it9=(1<<it9bits)+((it7tmp*it7tmp)>>(2*it7tmpbits-2*it7tmpshift-2*it7shift-it9bits));
    assert(fabs(it9)<(1<<13));
    //step 20
    int shifttmp1=it9bits+idrinvbits-irinvshift-it7shift-it7shift2;
    iphi0=(iphi1+(((it7>>it7shift2)*it9)>>shifttmp1))>>phi0bitshift;
    assert(fabs(iphi0)<(1<<17));  
    //step 21
    assert(it12shift+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shift+itshift-tbitshift);
    assert(fabs(it12)<(1<<18)); 
    //step 22
    int shifttmp2=it9bits+idrinvbits-itshift-it12shift;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;
    if (fabs(iz0)>(1<<10)) cout << "FPGATrackletCalculator Warning"<<endl;
    assert(fabs(iz0)<(1<<10));


    if (layer_==1) {
      
      if (dumppars) {
	cout << "DUMPPARS2: " 
	     << ideltaphi*kphi1
	     <<" "<<idz*kz
	     <<" "<<idrinv*kdrinv
	     <<" "<<idelta*kdelta
	     <<" "<<it5*kt5
	     <<" "<<iDeltainv*kdrinv
	     <<" | "<<it*ktpars
	     <<" | "<<ir1abs*kr
	     <<" "<<it7*kt7
	     <<" | "<<((it7tmp*it7tmp)>>(2*it7tmpbits-2*it7tmpshift-2*it7shift-it9bits))*kt9
	     <<" "<<it9*kt9
	     <<" "<<iphi1*kphi1
	     <<" "<<iphi0*kphi0pars
	     <<" * "<<it12*kt12<<" ("<<it12<<")" 
	     <<" "<<iz0*kzpars
	     <<endl;
      }
	
    }

    if (writez0andrinv) {
      static ofstream out("z0_and_rinv.txt");
      out << layer <<" "<<iz0*kzpars<<" "<<irinv*krinvpars<<endl;
    }

    if (fabs(irinv*krinvpars)>rinvcut) {
      //cout << "Failed tracklet pt cut"<<endl;
      return false;
    }
    if (layer==1&&fabs(iz0*kzpars)>z0cut) { 
      //cout << "Failed tracklet z0 cut"<<endl;
      return false;
    }
    if (layer>=2&&fabs(iz0*kzpars)>1.5*z0cut) { 
      //cout << "Failed tracklet z0 cut "<<iz0*kzpars<<endl;
      return false;
    }



    assert(rinvbitshift>=0);
    assert(phi0bitshift>=0);
    assert(tbitshift>=0);
    assert(z0bitshift>=0);
    
    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));

    //calculate projections

    for (int i=0;i<4;i++) {
      binaryproj(rproj_[i],irinv,iphi0,it,iz0,
		 iphiproj[i],izproj[i],iphider[i],izder[i],
		 minusNeighbor[i], plusNeighbor[i]);
      //cout << "Calculating projection to r = "<<rproj_[i]<<" iphi = "<<iphiproj[i]<<endl;
    }


    for (int i=0;i<5;i++) {
      int sign=1;
      if (it<0) sign=-1;
      //cout << "Calculating projection to z = "<<sign*zmean[i]<<endl;
      binaryprojdisk(sign*zmean[i],irinv,iphi0,it,iz0,
		     iphiprojdisk[i],irprojdisk[i],iphiderdisk[i],irderdisk[i],
		     minusNeighborDisk[i], plusNeighborDisk[i]);
      //cout << "Calculating projection to z = "<<sign*zmean[i]<<" ir = "<<irprojdisk[i]<<endl;
      //cout << "DUMPDISKPROJ2: "<<i<<" "<<irprojdisk[i]*krprojshiftdisk<<endl;
    }



    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;

    return true;

  }

  void ed_approxproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    double x8  = rproj*rinv/2.;
    double x10 = 6 + x8*x8;

    phiproj = phi0 - x8*x10/6.;
    zproj   = z0 + rproj*t*x10/6.;
    phider  = -rinv/2.;
    zder    = t;

  }


  void approxproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    //This code was written when traveling across the
    //northpole A.R. 2014-04-09

    double s1=0.5*rproj*rinv;

    double s2=s1*s1;

    double s3=1.0+s2/6.0;
    
    double s4=s1*s3;

    phiproj=phi0-s4;

    double s5=t*rproj;

    double s6=s5*s3;

    zproj=z0+s6;

    phider=-0.5*rinv; //-0.25*rinv*s2;

    zder=t; //+0.5*t*s2;

   
    if (dumpproj) {
      if (fabs(rproj-50.0)<10.0) {
	cout << "DUMPPROJ: "
	     << rproj
	     << " "<<rinv
	     << " "<<s1
	     << " "<<s2
	     << " "<<s3
	     << " "<<s4
	     << " "<<phi0
	     << " "<<phiproj
	     << " * "<<t
	     << " "<<s5
	     << " "<<s6
	     << " # "<<zproj
	     << " "<<phider
	     << " "<<zder
	     << endl;
      }
    }

  }


  void exactproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    phiproj=phi0-asin(0.5*rproj*rinv);
    zproj=z0+(2*t/rinv)*asin(0.5*rproj*rinv);

    phider=-0.5*rinv/sqrt(1-pow(0.5*rproj*rinv,2));
    zder=t/sqrt(1-pow(0.5*rproj*rinv,2));

  }



  void binaryproj(double rproj,int irinv, int iphi0, int it, int iz0,
		  int &iphiproj, int &izproj, int &iphider, int &izder,
		  bool &minusNeighbor, bool &plusNeighbor) {


    int irproj=rproj/kr;  //fixed constant


    int is1=((irproj*irinv)>>1)>>is1shift;

    assert(abs(is1)<(1<<16));

    int is2=(is1*is1)>>is2shift;

    assert(is2<(1<<8));
   
    assert(is2>=0);

    int is3=(1<<is3bits)+is2*((ks2/6.0)*(1<<is3bits)); 
    //Above is exact, but (ks2/6.0)*(1<<is3bits) is 0.119
    //which is close to 1/8=0.125. So we approximate:
    //assert(fabs((ks2/6.0)*(1<<is3bits)-0.125)<0.01);
    //int is3=(1<<is3bits)+(is2>>3); 
    


    //cout << "is3 : "<<is3<<" "<<(1<<is3bits)
    //	 <<" "<<is2*((ks2/6.0)*(1<<is3bits))
    //	 <<" "<<is2
    //	 <<" "<<(ks2/6.0)*(1<<is3bits)<<endl;

    assert(is3<(1<<(is3bits+1)));

    int shifttmp=idrinvbits+phi0bitshift-is1shift+is3bits-rinvbitshift;
    
    int is4=(is1*is3)>>shifttmp;

    assert(abs(is4)<(1<<17));
 
    iphiproj=iphi0-is4; 
    
    int is5=(it*irproj)>>is5shift;

    assert(abs(is5)<(1<<15));

    int bitshift=is3bits+idrinvbits-tbitshift-is5shift;

    izproj=((iz0<<bitshift)+is5*is3)>>bitshift;

    //cout << "is1 ... is5 : "<<is1<<" "<<is2<<" "<<is3<<" "
    //	 <<is4<<" "<<is5<<endl;


    //cout << "bitshift = "<<bitshift<<endl;

    //izproj=iz0+((is6+(1<<(bitshift-1)))>>bitshift); //This fixes a
                                                      //bias in the
                                                      //calcualted t
    //izproj=iz0+(is6>>bitshift); //has a bias

    iphider=-0.5*irinv;

    izder=it;
    
    iphiproj<<=1; //correct for L456 to match with stubs. Not good... FIXME
    minusNeighbor=false;
    plusNeighbor=false;

    if (iphiproj<(1<<nbitsphistubL456)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL456)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL456)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL456)/4;
    }

    //cout << "iphiproj (1<<nbitsphistubL456) "
    //	 <<iphiproj<<" "<<(1<<nbitsphistubL456)<<endl;

    assert(iphiproj>=0);
    assert(iphiproj<(1<<nbitsphistubL456));

    if (rproj<60.0) iphiproj>>=(nbitsphistubL456-nbitsphistubL123);

    //cout << " izproj "<<izproj*kz<<endl;

    if (dumpproj) {
      double kphiproj=kphiproj123;
      if (rproj>60.0) kphiproj=kphiproj456;
      if (fabs(rproj-50.0)<10.0) {
	//cout << "kphi0pars kphiproj "<<kphi0pars<<" "<<kphiproj<<endl;
	cout << "DUMPPROJ2 :"<<irproj*kr
	     << " "<<irinv*krinvpars
	     << " "<<is1*ks1
	     << " "<<is2*ks2
	     << " "<<is3*ks3
	     << " "<<is4*ks4
	     << " "<<iphi0*kphi0pars
	     << " "<<iphiproj*kphiproj
	     << " * "<<it*ktpars 
	     << " "<<is5*ks5
	     << " # "<<izproj*kz
	     << " "<<iphider*krinvpars
	     << " "<<izder*ktpars
	     <<endl;
      }
    }


    if (izproj<-(1<<(nbitszprojL123-1))) izproj=-(1<<(nbitszprojL123-1));
    if (izproj>=(1<<(nbitszprojL123-1))) izproj=(1<<(nbitszprojL123-1))-1;

    if (rproj>60.) {
      izproj>>=(nbitszprojL123-nbitszprojL456);
    }
    iphider>>=phiderbitshift;
    izder>>=zderbitshift;

  }

  void ed_approxprojdisk(double zproj,double rinv,double phi0,
		      double t, double z0,
		      double &phiproj, double &rproj,
		      double &phider, double &rder) {

    double x5 = zproj - z0;
    double x7 = rinv / 2.;
    double x9 = x5/6.;
    double x24 = x9 / t;
    double x25 = x5*x7/t;
    double x27 = 6 - x25*x25;

    phiproj = phi0 - x25;
    rproj   = x24 * x27;
    phider = -x7/t; 
    rder = 1/t;

  }

  void approxprojdisk(double zproj,double rinv,double phi0,
		      double t, double z0,
		      double &phiproj, double &rproj,
		      double &phider, double &rder) {


    //double tmp=rinv*(zproj-z0)/(2.0*t);
    //double rprojexact=(2.0/rinv)*sin(tmp);
  
    double t1=zproj-z0;

    double t2=1.0/t;

    double t3=t1*t2;
    
    double t4=t3*rinv;

    phiproj=phi0-t4/2.0;

    double t5=t4*t4;

    double t6=1.0-t5/24.0;

    rproj=t3*t6;

    //cout << "rresid rprojexact rproj "<<rprojexact<<" "<<rproj<<endl;

    rder=t2; 

    phider=-0.5*t2*rinv; 

    //assert(fabs(phider)<0.1);

    //cout << "disk_ zproj "<<disk_<<" "<<zproj<<endl;

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK: "
	     << " "<<phi0
	     << " "<<zproj
	     << " "<<z0
	     << " "<<rinv
	     << " "<<t1
	     << " "<<t2
	     << " "<<t3
	     << " "<<t4
	     << " phi="<<phiproj
	     << " "<<t5
	     << " "<<t6
	     << " r="<<rproj
	     << " "<<phider
	     << " "<<rder
	     << endl;
      }
    }
  }



  void exactprojdisk(double zproj,double rinv,double phi0,
		     double t, double z0,
		     double &phiproj, double &rproj,
		     double &phider, double &rder) {

    double tmp=rinv*(zproj-z0)/(2.0*t);
    rproj=(2.0/rinv)*sin(tmp);
    phiproj=phi0-tmp;


    //if (fabs(1.0/rinv)>180.0&&fabs(z0)<15.0) {
    //  cout << "phiproj phi0 tmp zproj z0 t: "<<phiproj<<" "<<phi0
    //	   <<" "<<tmp<<" "<<zproj<<" "<<z0<<" "<<t<<endl;
    //}

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK1: "
	       << " phi="<<phiproj
	       << " r="<<rproj
	       << endl;
	}
      }


    phider=-rinv/(2*t);
    rder=cos(tmp)/t;

    //assert(fabs(phider)<0.1);

  }



  void binaryprojdisk(double zproj,int irinv, int iphi0, int it, int iz0,
		      int &iphiproj, int &irproj, int &iphider, int &irder,
		      bool &minusNeighbor, bool &plusNeighbor) {

    assert(fabs(zproj)>100.0);

    //cout << "it "<<it<<" "<<it*ktparsdisk<<" zproj = "<<zproj<<endl;

    //Check if track can hit disk
    if (fabs(it*ktparsdisk)<0.7) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;
    }

    int izproj=zproj/kzdisk;  //fixed constant

    int t1=izproj-iz0;

    int t2=(1<<t2bits)/it;

    int t3=(t1*t2)>>t3shift;
    
    int t4=(t3*irinv)>>t4shift;

    //cout << "kt4disk/kphi0parsdisk :"<<kt4disk/kphi0parsdisk<<" "
    //	 << t3shift+t4shift+rinvbitshiftdisk-t2bits-tbitshift-phi0bitshiftdisk
    //	 << endl;

    //The +1 is for division by 2
    int tmpshift=1+t2bits+tbitshift+phi0bitshiftdisk-t3shift-t4shift-rinvbitshiftdisk;

    iphiproj=iphi0-(t4>>tmpshift);

  
    int t5=(t4>>t4shift2)*(t4>>t4shift2);

    //cout << "t4 = "<<t4<<endl;

    assert(t5>=0);


    int t6=(1<<t6bits)-(t5/24.0)*kst5disk*(1<<t6bits);

    irproj=(t3*t6)>>t6bits;

    irder=t2; 
 
    iphider=-0.5*t2*irinv; 

    iphiproj<<=1; //bit that was shifted away...

    minusNeighbor=false;
    plusNeighbor=false;
    if (iphiproj<(1<<nbitsphistubL456)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL456)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL456)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL456)/4;
    }

    if (iphiproj<0) iphiproj=0;
    if (iphiproj>=(1<<nbitsphistubL456)) iphiproj=(1<<nbitsphistubL456)-1;


    assert(iphiproj<(1<<nbitsphistubL456));
   
    //cout << "irproj :"<<irproj<<" "<<irproj*krprojdisk<<endl;

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK2: "
	     << " "<<iphi0*kphi0parsdisk  
	     << " "<<izproj*kzdisk
	     << " "<<iz0*kzdisk
	     << " "<<irinv*krinvparsdisk
	     << " "<<t1*kzdisk
	     << " "<<t2*kt2disk
	     << " "<<t3*kt3disk
	     << " "<<t4*kt4disk
	     << " phi="<<iphiproj*kphi0parsdisk*0.5
	     << " "<<t5*kst5disk
	     << " "<<t6*kt6disk
	     << " r="<<irproj*krprojdisk
	     << " "<<iphider*kphiprojderdisk
	     << " "<<irder*krprojderdisk
	     << endl;
      }
    }

    iphiproj>>=(nbitsphistubL456-nbitsphistubL123);

    irproj>>=rprojdiskbitshift;
    iphider>>=phiderdiskbitshift;
    irder>>=rderdiskbitshift;

    if (irproj<=0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }

    assert(irproj>0);

    if (irproj*krprojshiftdisk>120.0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }
    


    //cout <<"iphiproj : "<<iphiproj<<endl;

    //assert(irproj*krprojshiftdisk<116);

    //cout << "FPGADisk projection: irproj="<<irproj<<" "<<irproj*krprojshiftdisk
    //	 <<"   iphiproj="<<iphiproj<<" "<<iphiproj*kphi0parsdisk<<endl;


  }



  bool binarytrackletdisk(FPGAStub* innerFPGAStub, 
			  FPGAStub* outerFPGAStub,
			  double sigmaz,
			  int& irinv, int& iphi0,
			  int& it, int& iz0,
			  int iphiprojLayer[6], int izprojLayer[6],
			  int iphiderLayer[6], int izderLayer[6],
			  bool minusNeighborLayer[6], bool plusNeighborLayer[6],
			  int iphiproj[4], int izproj[4],
			  int iphider[4], int izder[4],
			  bool minusNeighbor[4], bool plusNeighbor[4]){
    
    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //To get same precission as for layers.
    iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    iphi2<<=(nbitsphistubL456-nbitsphistubL123);


    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1;  //should make this positive and keep sign bit?
    //step 2
    int sign=1;
    if (disk_<0) {
      sign=-1;
    }

    int idz=sign*(zmean[abs(disk_)]-zmean[abs(disk_)-1])/kzdisk+iz2-iz1;
    //step 3
    int idr=ir2-ir1;  
    //step 4
    int ir1abs=rmindisk/krdisk+ir1;
    //step 5
    int ir2abs=rmindisk/krdisk+ir2;
    //step 6
    if (idr==0) idr=1;
    int idrinv=(1<<idrinvbits)/idr;  //not actually using 15 bits since idr is largish.
                             //implement in lookup table, just a fcn of ir1-ir2

    //step 7
    int it1=(ir1abs*ir2abs)>>it1shiftdisk;
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shiftdisk;
    //step 9
    int it3=(it2*it2)>>it3shiftdisk;
    //step 10
    int idelta=0.5*it1*it3;
    //step 11
    int ideltatmp=idelta*(kdeltadisk*(1<<it4bitsdisk));
    //step 12
    int it4=(1<<it4bitsdisk)-3*(ideltatmp>>1);
    //step 13    
    int it5=(1<<it5bitsdisk)-((ideltatmp*it4)>>(2*it4bitsdisk-it5bitsdisk));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bitsdisk;  
    //step 15
    assert(rinvbitshiftdisk>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshiftdisk-1)); //-1 because of *2 
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    //step 17
    assert(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk);
    //step 18
    int it7tmp=(it7*it7tmpfactordisk)>>it7tmpshiftdisk;
    //step 19
    int it9=(1<<it9bitsdisk)+((it7tmp*it7tmp)>>(2*it7tmpbitsdisk-2*it7tmpshiftdisk-2*it7shiftdisk-it9bitsdisk));
    //step 20
    int shifttmp1=it9bitsdisk+idrinvbits-irinvshiftdisk-it7shiftdisk-it7shift2disk;
    iphi0=(iphi1+(((it7>>it7shift2disk)*it9)>>shifttmp1))>>phi0bitshiftdisk;
    //step 21
    assert(it12shiftdisk+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shiftdisk+itshift-tbitshift);
    //step 23
    iz1+=sign*zmean[abs(disk_)-1]/kzdisk;
    //step 24
    int shifttmp2=it9bitsdisk+idrinvbits-itshift-it12shiftdisk;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;

 
    if (abs(disk_)==1) {
      
       if (dumppars) {
	cout << "DUMPPARS2:" 
	     << ideltaphi*kphi1
	     <<" "<<idz*kzdisk
	     <<" "<<idr*krdisk
	     <<" > "<<ir1abs*krdisk
	     <<" "<<ir2abs*krdisk
	     <<" "<<idrinv*kdrinvdisk
	     <<" "<<idelta*kdeltadisk
	     <<" "<<it5*kt5disk
	     <<" "<<iDeltainv*kdrinvdisk
	     <<" "<<irinv*krinvpars
	     <<" t= "<<it*ktpars
	     <<" | "<<ir1abs*krdisk
	     <<" "<<it7*kt7disk
	     <<" | "<<it9*kt9
	     <<" "<<iphi1*kphi1
	     <<" phi0= "<<iphi0*kphi0pars
	     <<" * "<<it12*kt12disk
	     <<" "<<iz1*kzdisk
	     <<" z0= "<<iz0*kzpars
	     <<endl;
      }
   

    }

    if (fabs(iz0*kzdisk)>z0cut) {
      //cout << "DUMP iz0 too large: "<<iz0*kzdisk<<endl;
      return false;
    }

    //if (fabs(irinv*krinvparsdisk)>0.0057) {
    if (fabs(irinv*krinvparsdisk)>rinvcut) {
      //cout << "DUMP irinv too large: "<<irinv*krinvparsdisk<<endl;
      return false;
    }


    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));


    //calculate projections

    //cout << "DUMP calling projectbinary disk_="<<disk_<<endl;

    for (int i=0;i<3;i++) {
      binaryprojdisk(zproj_[i],irinv,iphi0,it,iz0,
		     iphiproj[i],izproj[i],iphider[i],izder[i],
		     minusNeighbor[i], plusNeighbor[i]);
    }


    for (int i=0;i<3;i++) {
      binaryproj(rmean[i],irinv,iphi0,it,iz0,
		 iphiprojLayer[i],izprojLayer[i],
		 iphiderLayer[i],izderLayer[i],
		 minusNeighborLayer[i], plusNeighborLayer[i]);
      //cout << "iphiprojLayer : "<<iphiprojLayer[i]<<endl;
    }



    




    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;
    
    return true;

  }


  

  bool binarytrackletOverlap(FPGAStub* innerFPGAStub, 
			     FPGAStub* outerFPGAStub,
			     double sigmaz,
			     int& irinv, int& iphi0,
			     int& it, int& iz0,
			     int iphiprojLayer[6], int izprojLayer[6],
			     int iphiderLayer[6], int izderLayer[6],
			     bool minusNeighborLayer[6], 
			     bool plusNeighborLayer[6],
			     int iphiproj[4], int izproj[4],
			     int iphider[4], int izder[4],
			     bool minusNeighbor[4], bool plusNeighbor[4]){

    //cout << "In binarytrackletoverlap : "<<disk_<<endl;

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }
 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //cout << "iz2 iz1 disk_ : "<<iz2<<" "<<iz1<<" "<<disk_<<endl;
    
    
    //To get same precission as for disks.
    iphi1<<=3;
    iphi2<<=3;

    //Radial position in layers are less precise than disks.
    //to make them the same we need to add a bit. ???
    ir2<<=1;

    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1;  //should make this positive and keep sign bit?
    //step 2
    int sign=1;
    if (disk_<0) {
      sign=-1;
    }

    //cout << "disk_ = "<<disk_<<endl;
    assert(abs(disk_)==1);
    int idz=-sign*zmean[abs(disk_)-1]/kzdisk+iz2-iz1;
    //step 3
    int idr=rmean[outerFPGAStub->layer().value()]/kr+ir2-rmindisk/krdisk-ir1;  
    //step 4
    int ir1abs=rmindisk/krdisk+ir1;
    //step 5
    int ir2abs=rmean[outerFPGAStub->layer().value()]/kr+ir2;
    //step 6
    if (idr==0) idr=1;
    int idrinv=(1<<idrinvbits)/idr;  //not actually using 15 bits since idr is largish.
                             //implement in lookup table, just a fcn of ir1-ir2

    //step 7
    int it1=(ir1abs*ir2abs)>>it1shiftdisk;
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shiftdisk;
    //step 9
    int it3=(it2*it2)>>it3shiftdisk;
    //step 10
    int idelta=0.5*it1*it3;
    //step 11
    int ideltatmp=idelta*(kdeltadisk*(1<<it4bitsdisk));
    //step 12
    int it4=(1<<it4bitsdisk)-3*(ideltatmp>>1);
    //step 13    
    int it5=(1<<it5bitsdisk)-((ideltatmp*it4)>>(2*it4bitsdisk-it5bitsdisk));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bitsdisk;  
    //step 15
    assert(rinvbitshiftdisk>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshiftdisk-1)); //-1 because of *2 
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    //step 17
    assert(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk);
    //step 18
    int it7tmp=(it7*it7tmpfactordisk)>>it7tmpshiftdisk;
    //step 19
    int it9=(1<<it9bitsdisk)+((it7tmp*it7tmp)>>(2*it7tmpbitsdisk-2*it7tmpshiftdisk-2*it7shiftdisk-it9bitsdisk));
    //step 20
    int shifttmp1=it9bitsdisk+idrinvbits-irinvshiftdisk-it7shiftdisk-it7shift2disk;
    iphi0=(iphi1+(((it7>>it7shift2disk)*it9)>>shifttmp1))>>phi0bitshiftdisk;
    //step 22
    assert(it12shiftdisk+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shiftdisk+itshift-tbitshift);
    //step 23
    iz1+=sign*zmean[abs(disk_)-1]/kzdisk;
    //step 24
    int shifttmp2=it9bitsdisk+idrinvbits-itshift-it12shiftdisk;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;

    if (dumppars) {   //overlap
      cout << "DUMPPARS2:"
	   << ideltaphi*kphi1
	   <<" "<<idz*kzdisk
	   <<" "<<idr*krdisk
	   <<" > "<<ir1abs*krdisk
	   <<" "<<ir2abs*krdisk
	   <<" "<<idrinv*kdrinvdisk
	   <<" "<<idelta*kdeltadisk
	   <<" "<<it5*kt5disk
	   <<" "<<iDeltainv*kdrinvdisk
	   <<" "<<irinv*krinvpars
	   <<" t= "<<it*ktpars
	   <<" | "<<ir1abs*krdisk
	   <<" "<<it7*kt7disk
	   <<" | "<<it9*kt9disk
	   <<" "<<iphi1*kphi1
	   <<" phi0= "<<iphi0*kphi0pars
	   <<" * "<<it12*kt12disk
	   <<" "<<iz1*kzdisk
	   <<" z0= "<<iz0*kzpars
	   <<endl;
      
    }

    //cout << "overlap z0 and rinv it : "<<iz0*kzdisk<<" "<<irinv*krinvparsdisk<<" "<<it*ktparsdisk<<" "<<it<<endl;

    if (fabs(it*ktparsdisk)>10.0) {
      return false;
    }
    
    if (fabs(iz0*kzdisk)>z0cut) {
      //cout << "DUMP overlap iz0 too large: "<<iz0*kzdisk<<endl;
      return false;
    }

    if (fabs(irinv*krinvparsdisk)>rinvcut) {
      //cout << "DUMP overlap irinv too large: "<<irinv*krinvparsdisk<<endl;
      return false;
    }

    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));

    //calculate projections

    //cout << "DUMP calling projectbinary disk_="<<disk_<<endl;

    for (int i=0;i<4;i++) {
      binaryprojdisk(zprojoverlap_[i],irinv,iphi0,it,iz0,
		     iphiproj[i],izproj[i],iphider[i],izder[i],
		     minusNeighbor[i], plusNeighbor[i]);
      //cout << "zproj 1 der : "<<izder[i]<<" "<<it<<endl;
    }


    iphiprojLayer[0]=0;
    izprojLayer[0]=0;
    iphiderLayer[0]=0;
    izderLayer[0]=0;
    minusNeighborLayer[0]=false;
    plusNeighborLayer[0]=false;
  
    if (outerFPGAStub->layer().value()+1==2){
      for (int i=0;i<1;i++) {
	//cout << "Will call binaryproj r = "<<rmean[i]<<" "<<getName()<<" "<<layer_<<" "<<outerFPGAStub->layer().value()+1<<endl;
	//cout << "Inner fga stub : "<<innerFPGAStub->layer().value()+1<<endl;
	binaryproj(rmean[i],irinv,iphi0,it,iz0,
		   iphiprojLayer[i],izprojLayer[i],
		   iphiderLayer[i],izderLayer[i],
		   minusNeighborLayer[i], plusNeighborLayer[i]);
	//cout << "iphiprojLayer : "<<iphiprojLayer[i]<<endl;
	//cout << "zproj 2 der : "<<izderLayer[i]<<" "<<it<<endl;
      }
    } else {
      //cout << getName() << " Skipping calculating projections to r = "<<rmean[0]<<endl;
    }



    




    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;
    
    return true;

  }

  void ed_approxtrackletdisk(double r1, double z1, double phi1,
			  double r2, double z2, double phi2, double sigmaz,
			  double &rinv, double &phi0,
			  double &t, double &z0,
			  double phiprojLayer[4], double rprojLayer[4], 
			  double phiderLayer[4], double rderLayer[4],
			  double phiproj[4], double rproj[4], 
			  double phider[4], double rder[4]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }
    double deltaphi = 0;
    if (phi1<0.0) phi1+=two_pi;
    deltaphi=phi2-phi1;
    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
    assert(phi1tmp>-1e-10);
    double dr=r2-r1;
    double dz=z2-z1;
    double drinv=1.0/dr;
    double delta0 = deltaphi*drinv;
    double delta1 = r1 * delta0;
    double delta2 = r2 * delta0;
    double deltaZ = dz * drinv;
    double a2 = 2-delta1*delta2;
    rinv=-delta0*a2;
    phi0=phi1tmp-delta1*(1-delta0*delta2*(r2+r1)/6.);
    t= deltaZ/2.*a2;
    z0=z1-r1*deltaZ;

    //calculate projections
    for (int i=0;i<3;i++) {
      ed_approxproj(rmean[i],rinv,phi0,t,z0,
		 phiprojLayer[i],rprojLayer[i],
		 phiderLayer[i],rderLayer[i]);
    }

    for (int i=0;i<3;i++) {
      ed_approxprojdisk(zproj_[i],rinv,phi0,t,z0,
		     phiproj[i],rproj[i],phider[i],rder[i]);
    }



  }

  void approxtrackletdisk(double r1, double z1, double phi1,
			  double r2, double z2, double phi2, double sigmaz,
			  double &rinv, double &phi0,
			  double &t, double &z0,
			  double phiprojLayer[4], double rprojLayer[4], 
			  double phiderLayer[4], double rderLayer[4],
			  double phiproj[4], double rproj[4], 
			  double phider[4], double rder[4]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
    assert(phi1tmp>0.0);

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;// *(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;// -2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;// *(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;// *(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;// +3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (abs(disk_)==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "t7*t9 : "<<t7*t9<<endl;
	cout << "DUMPPARS1:" 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" > "<<r1
	     <<" "<<r2
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" "<<rinv
	     <<" t= "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" phi0= "<<phi0
	     <<" * "<<t12
	     <<" "<<z1
	     <<" z0= "<<z0
	     <<endl;
      }
      

//      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
//	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
//	   <<" delta="<<delta
//	   <<" t5="<<t5
//           <<" deltainv="<<deltainv
//	   <<" rinv="<<rinv
//	   <<" t="<<t
//	   <<" r1abs="<<r1
//	   <<" t7="<<t7
//	   <<" t9="<<t9
//	   <<" phi1="<<phi1tmp
//	   <<" ***phi0="<<phi0
//	   <<" t12="<<t12
//	   <<" z1="<<z1
//	   <<" z0="<<z0
//	   <<endl;

   

    }
   

    //calculate projection


    for (int i=0;i<3;i++) {
      approxproj(rmean[i],rinv,phi0,t,z0,
		 phiprojLayer[i],rprojLayer[i],
		 phiderLayer[i],rderLayer[i]);
    }


    //static ofstream out1("neighborproj.txt");

    for (int i=0;i<3;i++) {
      approxprojdisk(zproj_[i],rinv,phi0,t,z0,
		     phiproj[i],rproj[i],phider[i],rder[i]);
      /*
      if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	if ((fabs(rproj[i])<100.0)&&(phiproj[i]<(phimax_-phimin_)/6)){
	  out1<<disk_<<" -1 "<<phiproj[i]<<endl;
	} else  if ((fabs(rproj[i])<100.0)&&(phiproj[i]>7.0*(phimax_-phimin_)/6)){
	  out1<<disk_<<" +1 "<<phiproj[i]<<endl;
	} else if (fabs(rproj[i])<100.0){
	  out1<<disk_<<" 0 "<<phiproj[i]<<endl;
	}
      }
      */
    }

  }



 

  void execute() {

    //we are taking a shortcut here by using the information to
    //find the stubs from the stub pairs list...

    unsigned int countall=0;
    unsigned int countsel=0;

    //cout << "FPGATrackletCalculator execute "<<getName()<<" "<<stubpairs_.size()<<endl;
    
    for(unsigned int l=0;l<stubpairs_.size();l++){
      for(unsigned int i=0;i<stubpairs_[l]->nStubPairs();i++){

	countall++;

	L1TStub* innerStub=stubpairs_[l]->getL1TStub1(i);
	FPGAStub* innerFPGAStub=stubpairs_[l]->getFPGAStub1(i);

	L1TStub* outerStub=stubpairs_[l]->getL1TStub2(i);
	FPGAStub* outerFPGAStub=stubpairs_[l]->getFPGAStub2(i);

	if (debug1) {
	  cout << "FPGATrackletCalculator execute "<<getName()<<endl;
	}
	
	if (innerFPGAStub->isBarrel()&&(getName()!="TC_D1L2A"&&getName()!="TC_D1L2B")){

	  if (debug1) {
	    cout << "FPGATrackletCalculator "<<getName()<<" trying stub pair in layer (inner outer): "
		 <<innerFPGAStub->layer().value()<<" "<<outerFPGAStub->layer().value()<<endl;
	  }
	    
	  assert(outerFPGAStub->isBarrel());

	  //FIXME - should be set at initialization
	  assert(layer_==innerFPGAStub->layer().value()+1);
	    
	  assert(layer_==1||layer_==3||layer_==5);

	  int lproj[4];

	  if (layer_==1) {
	    rproj_[0]=rmeanL3;
	    rproj_[1]=rmeanL4;
	    rproj_[2]=rmeanL5;
	    rproj_[3]=rmeanL6;
	    lproj[0]=3;
	    lproj[1]=4;
	    lproj[2]=5;
	    lproj[3]=6;
	  }
      
	  if (layer_==3) {
	    rproj_[0]=rmeanL1;
	    rproj_[1]=rmeanL2;
	    rproj_[2]=rmeanL5;
	    rproj_[3]=rmeanL6;
	    lproj[0]=1;
	    lproj[1]=2;
	    lproj[2]=5;
	    lproj[3]=6;
	  }
	  
	  if (layer_==5) {
	    rproj_[0]=rmeanL1;
	    rproj_[1]=rmeanL2;
	    rproj_[2]=rmeanL3;
	    rproj_[3]=rmeanL4;
	    lproj[0]=1;
	    lproj[1]=2;
	    lproj[2]=3;
	    lproj[3]=4;
	  }
	  


	  //cout << "Calculating tracklet parameters in layer "<<layer_<<endl;
	
      	  
	  FPGAWord iphi1=innerFPGAStub->phi();
	  FPGAWord iz1=innerFPGAStub->z();
	  FPGAWord ir1=innerFPGAStub->r();

	  FPGAWord iphi2=outerFPGAStub->phi();
	  FPGAWord iz2=outerFPGAStub->z();
	  FPGAWord ir2=outerFPGAStub->r();
	  
	  
	  double r1=innerStub->r();
	  double z1=innerStub->z();
	  double phi1=innerStub->phi();
	  
	  double r2=outerStub->r();
	  double z2=outerStub->z();
	  double phi2=outerStub->phi();
	  
	  
	  double rinv,phi0,t,z0;
	  
	  double phiproj[4],zproj[4],phider[4],zder[4];
	  double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
	  
	  exacttracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			rinv,phi0,t,z0,
			phiproj,zproj,phider,zder,
			phiprojdisk,rprojdisk,phiderdisk,rderdisk);

	  if (fabs(rinv)>0.0065) {
	    //cout << "FPGATrackletCalculator warning will reject rinv "<<rinv<<endl;
	    continue;
	  }

	  
	  if (1) {
	    int lphi=1;
	    int lz=1;
	    int lr=2;
	    if (layer_>3) {
	      lphi=8;
	      lz=16;
	      lr=1;
	    }
	    
	    double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;

	    //cout << "dphi1 phi1 phimin_ phimax_ : "
	    //	 <<dphi1<<" "<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
	    
	    if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	    //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	    assert(fabs(dphi1)<1e-4);
	    phi1-=dphi1;
	    double dz=z1-iz1.value()*kz*lz;
	    assert(fabs(dz)<1.0*lz);
	    z1-=dz;
	    //cout <<r1<<" "<<lr*ir1.value()*kr<<" "<<rmean[layer_-1]<<endl;
	    double dr=r1-lr*ir1.value()*kr-rmean[layer_-1];
	    assert(fabs(dr)<0.1);
	    r1-=dr;
	  }
	  
	  


	  if (1) {
	    int lphi=1;
	    int lz=1;
	    int lr=2;
	    if (layer_>2) {
	      lphi=8;
	      lz=16;
	      lr=1;
	    }
	    double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	    if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	    //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	    //	 <<phi2<<" "<< iphi2.value()<<endl;
	    assert(fabs(dphi2)<1e-4);
	    phi2-=dphi2;
	    //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	    double dz=z2-iz2.value()*kz*lz;
	    assert(fabs(dz)<1.0*lz);
	    z2-=dz;
	    //cout <<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[layer_]<<endl;
	    double dr=r2-lr*ir2.value()*kr-rmean[layer_];
	    assert(fabs(dr)<0.1);
	    r2-=dr;
	  }
	  
	  double rinvapprox,phi0approx,tapprox,z0approx;
	  double phiprojapprox[4],zprojapprox[4],phiderapprox[4],zderapprox[4];
	  double phiprojdiskapprox[5],rprojdiskapprox[5];
	  double phiderdiskapprox[5],rderdiskapprox[5];
	  
	  approxtracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
	  //ed_approxtracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			 rinvapprox,phi0approx,tapprox,z0approx,
			 phiprojapprox,zprojapprox,phiderapprox,zderapprox,
			 phiprojdiskapprox,rprojdiskapprox,
			 phiderdiskapprox,rderdiskapprox);
	  
	  int irinv,iphi0,it,iz0;
	  int iphiproj[4],izproj[4],iphider[4],izder[4];
	  bool minusNeighbor[4],plusNeighbor[4];
	  int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
	  bool minusNeighborDisk[5],plusNeighborDisk[5];
	  
	  bool success=binarytracklet(innerFPGAStub,outerFPGAStub,
	  //bool success=ed_binarytracklet(innerFPGAStub,outerFPGAStub,
				      outerStub->sigmaz(),
				      irinv,iphi0,it,iz0,
				      iphiproj,izproj,iphider,izder,
				      minusNeighbor,plusNeighbor,
				      iphiprojdisk,irprojdisk,
				      iphiderdisk,irderdisk,
				      minusNeighborDisk,plusNeighborDisk);

	  //cout << "success : "<<success<<endl;
	  
	  if (!success) continue;
	  
	  for(unsigned int j=0;j<5;j++){
	    if (minusNeighborDisk[j]) {
	      phiprojdiskapprox[j]+=dphisector;
	      phiprojdisk[j]+=dphisector;
	    }
	    if (plusNeighborDisk[j]) {
	      phiprojdiskapprox[j]-=dphisector;
	      phiprojdisk[j]-=dphisector;
	    }
	  }
	  
	  for(unsigned int j=0;j<4;j++){
	    if (minusNeighbor[j]) {
	      phiprojapprox[j]+=dphisector;
	      phiproj[j]+=dphisector;
	    }
	    if (plusNeighbor[j]) {
	      phiprojapprox[j]-=dphisector;
	      phiproj[j]-=dphisector;
	    }	    
	  }
	  
	  
	  if ((layer_==1)&&(fabs(iz0*kzpars)>z0cut)) {
	    continue;
	  }
	  
	  if ((layer_>=3)&&(fabs(iz0*kzpars)>20.0)) {
	    continue;
	  }
	  
	  
	  if (writeTrackletPars) {
	    static ofstream out("trackletpars.txt");
	    out <<"Trackpars "<<layer_
		<<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvpars
		<<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0pars
		<<"   "<<t<<" "<<tapprox<<" "<<it*ktpars
		<<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzpars
		<<endl;
	  }	    

	  if (writeTrackProj) {
	    static ofstream out1("trackproj.txt");
	    for (int i=0;i<4;i++) {
	      double kphiproj=kphiproj123;
	      int lz=1;
	      if (rproj_[i]>60.0) {
		kphiproj=kphiproj456;
		lz=16;
	      }
	      out1 <<"Trackproj "<<layer_<<" "<<rproj_[i]
		   <<"   "<<phiproj[i]<<" "<<phiprojapprox[i]
		   <<" "<<iphiproj[i]*kphiproj
		   <<"   "<<zproj[i]<<" "<<zprojapprox[i]
		   <<" "<<izproj[i]*kzproj*lz
		   <<"   "<<phider[i]<<" "<<phiderapprox[i]
		   <<" "<<iphider[i]*kphider
		   <<"   "<<zder[i]<<" "<<zderapprox[i]
		   <<" "<<izder[i]*kzder
		   <<endl;
	    }
	    
	  }
	  


	  //Can also be done more accurately now
	  //counter for steps 4&5: number of tracklets to route and project into VM
	  //counter is per sector
	  //++Ntracklets_;
	  //if(Ntracklets_ < NMAXproj){
	  
	  FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						  innerFPGAStub,outerFPGAStub,
						  iSector_,
						  phioffset_,
						  rinv,phi0,z0,t,
						  rinvapprox,phi0approx,
						  z0approx,tapprox,
						  irinv,iphi0,iz0,it,
						  iphiproj,izproj,iphider,izder,
						  minusNeighbor,plusNeighbor,
						  phiproj,zproj,phider,zder,
						  phiprojapprox,zprojapprox,
						  phiderapprox,zderapprox,
						  iphiprojdisk,irprojdisk,
						  iphiderdisk,irderdisk,
						  minusNeighborDisk,
						  plusNeighborDisk,
						  phiprojdisk,rprojdisk,
						  phiderdisk,rderdisk,
						  phiprojdiskapprox,
						  rprojdiskapprox,
						  phiderdiskapprox,
						  rderdiskapprox,
						  false);

	  if (debug1) {
	    cout << "FPGATrackletCalculator "<<getName()<<" Found tracklet in layer = "<<layer_<<" "
		 <<iSector_<<" "<<tracklet<<endl;
	  }

	  countsel++;

	  tracklet->setTrackletIndex(trackletpars_->nTracklets());
	  tracklet->setTCIndex(0);

	  trackletpars_->addTracklet(tracklet);

	  //cout << "Will add disk projections"<<endl;

	  bool addL3=false;
	  bool addL4=false;
	  for(unsigned int j=0;j<4;j++){
	    //	    cout<<" LL to L "<<lproj[j]<<"\n";
	    bool added=false;
	    added=addLayerProj(tracklet,lproj[j]);
	    if (added&&lproj[j]==3) addL3=true;
	    if (added&&lproj[j]==4) addL4=true;
	  }

	  
	  for(unsigned int j=0;j<4;j++){ //no projections to 5th disk!!
	    int disk=j+1;
	    if (disk==4&&addL3) continue;
	    if (disk==3&&addL4) continue;
	    if (it<0) disk=-disk;
	    //	    cout<<" LL to disk "<<disk<<"\n";
	    addDiskProj(tracklet,disk);
	  }

	  //cout << "Done adding disk projection"<<endl;

	  
	  //cout << "Done adding layer projection"<<endl;


	  
	}  else {

	  if (outerFPGAStub->isDisk()) {

	    if (debug1) {
	      cout <<  "FPGATrackletCalculator::execute calculate disk seeds" << endl;
	    }
	      
	    //FIXME - should be set at initialization
	    disk_=innerFPGAStub->disk().value();
	    int dproj[3];

	    assert(abs(disk_)==1||abs(disk_)==3);
       
	    if (disk_==1) {
	      zproj_[0]=zmeanD3;
	      zproj_[1]=zmeanD4;
	      zproj_[2]=zmeanD5;
	      dproj[0]=3;
	      dproj[1]=4;
	      dproj[2]=5;
	    }
      
	    if (disk_==3) {
	      zproj_[0]=zmeanD1;
	      zproj_[1]=zmeanD2;
	      zproj_[2]=zmeanD5;
	      dproj[0]=1;
	      dproj[1]=2;
	      dproj[2]=5;
	    }
	    
	    if (disk_==-1) {
	      zproj_[0]=-zmeanD3;
	      zproj_[1]=-zmeanD4;
	      zproj_[2]=-zmeanD5;
	      dproj[0]=-3;
	      dproj[1]=-4;
	      dproj[2]=-5;
	    }
	    
	    if (disk_==-3) {
	      zproj_[0]=-zmeanD1;
	      zproj_[1]=-zmeanD2;
	      zproj_[2]=-zmeanD5;
	      dproj[0]=-1;
	      dproj[1]=-2;
	      dproj[2]=-5;
	    }
	    

	    //cout << "FPGATrackletCalculator inner and outer radius of stubs "<<innerStub->r()<<" "<<outerStub->r()<<endl;

	    if (innerStub->r()>60.0) continue;
	    if (outerStub->r()>60.0) continue;

	    
	    int istubpt1=innerFPGAStub->stubpt().value();
	    int iphivm1=innerFPGAStub->phivm().value();
	    FPGAWord iphi1=innerFPGAStub->phi();
	    FPGAWord iz1=innerFPGAStub->z();
	    FPGAWord ir1=innerFPGAStub->r();
	    int irvm1=innerFPGAStub->rvm().value();
	    //int izvm1=innerFPGAStub->zvm().value();
	    
	    int istubpt2=outerFPGAStub->stubpt().value();
	    int iphivm2=outerFPGAStub->phivm().value();
	    FPGAWord iphi2=outerFPGAStub->phi();
	    FPGAWord iz2=outerFPGAStub->z();
	    FPGAWord ir2=outerFPGAStub->r();
	    int irvm2=outerFPGAStub->rvm().value();
	    //int izvm2=outerFPGAStub->zvm().value();
	    
	  
	    
	    int ideltaphi=iphivm2-iphivm1;
	    int ideltar=irvm2-irvm1;

	    ideltar>>=2;
	    
	    if (ideltar<0) ideltar+=8;
	    assert(ideltar>=0);
	    if (ideltaphi<0) ideltaphi+=16;
	    assert(ideltaphi>=0);
	    
	    //cout << "istubpt1 istubpt2 : "<<istubpt1<<" "<<istubpt2<<endl;
	    assert(istubpt1>=0);
	    assert(istubpt2>=0);
	    
	    //int address=(istubpt1<<10)+(istubpt2<<7)+(ideltaphi<<3)+ideltar;
	    
	    //int i1=TEs_[i].first.first;
	    //int i2=TEs_[i].second.first;
	    //int j1=TEs_[i].first.second;
	    //int j2=TEs_[i].second.second;
	    
	    
	    //bool phimatch=(*TETables_)[i].phicheck(address,i1,i2,j1,j2);
	    
	    //bool zmatch=(*TETables_)[i].zcheck(izvm1,izvm2,irvm1,irvm2);
	    
	    
	    double r1=innerStub->r();
	    double z1=innerStub->z();
	    double phi1=innerStub->phi();
	    
	    double r2=outerStub->r();
	    double z2=outerStub->z();
	    double phi2=outerStub->phi();

	    //cout << "r1 r2 z1 z2 "<<r1<<" "<<r2<<" "<<z1<<" "<<z2<<endl;
    
	    if (r2<r1+2.0) continue; //Protection... Should be handled cleaner
	                             //to avoid problem with floating point 
	                             //calculation
	      

	    double rinv,phi0,t,z0;

	    double phiproj[3],zproj[3],phider[3],zder[3];
	    double phiprojdisk[3],rprojdisk[3],phiderdisk[3],rderdisk[3];
	    
	    exacttrackletdisk(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			      rinv,phi0,t,z0,
			      phiproj,zproj,phider,zder,
			      phiprojdisk,rprojdisk,phiderdisk,rderdisk);


	    //Truncates floating point positions to integer
	    //representation precision
	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      //if (layer_>3) {
	      //  lphi=8;
	      //  lz=16;
	      //  lr=1;
	      //}
	      double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;
	      if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	      //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	      assert(fabs(dphi1)<1e-4);
	      phi1-=dphi1;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z1-(iz1.value()*kzdisk*lz+sign*zmean[abs(disk_)-1]);
	      //cout << "z1 iz1.value() kz : "<<z1<<" "<<iz1.value()<<" "<<kz<<" "
	      //	 <<iz1.value()*kz<<" "<<sign*zmean[abs(disk_)-1]<<endl;
	      assert(fabs(dz)<1.0*lz);
	      z1-=dz;
	      //cout <<"r1 : "<<r1<<" "<<lr*ir1.value()*krdisk+rmindisk<<endl;
	      double dr=r1-lr*ir1.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.75);
	      r1-=dr;
	      //cout << "dz1 dr1 : "<<dz<<" "<<dr<<endl;
	    }
	    



	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      
	      double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	      if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	      //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	      //	 <<phi2<<" "<< iphi2.value()<<endl;
	      assert(fabs(dphi2)<1e-4);
	      phi2-=dphi2;
	      //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z2-iz2.value()*kzdisk*lz-sign*zmean[abs(disk_)];
	      assert(fabs(dz)<1.0*lz);
	      z2-=dz;
	      //cout <<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[layer_]<<endl;
	      double dr=r2-lr*ir2.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.1);
	      r2-=dr;
	      //cout << "dz2 dr2 : "<<dz<<" "<<dr<<endl;
	    }
	    
	    
	    
	    //FPGATrackletCand trackletcand(innerStub,outerStub);
	    //candlist_[i].addTrackletCand(trackletcand);
	    
	    
	    double rinvapprox,phi0approx,tapprox,z0approx;
	    double phiprojapprox[3],zprojapprox[3],phiderapprox[3],zderapprox[3];
	    double phiprojapproxdisk[3],rprojapproxdisk[3],
	      phiderapproxdisk[3],rderapproxdisk[3];
	  
	    ed_approxtrackletdisk(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			       rinvapprox,phi0approx,tapprox,z0approx,
			       phiprojapprox,zprojapprox,
			       phiderapprox,zderapprox,
			       phiprojapproxdisk,rprojapproxdisk,
			       phiderapproxdisk,rderapproxdisk);
	    
	    int irinv,iphi0,it,iz0;
	    int iphiproj[3],izproj[3],iphider[3],izder[3];
	    bool minusNeighbor[3],plusNeighbor[3];
	    
	    int iphiprojdisk[3],irprojdisk[3],iphiderdisk[3],irderdisk[3];
	    bool minusNeighbordisk[3],plusNeighbordisk[3];
	    
	    //bool success=binarytrackletdisk(innerFPGAStub,outerFPGAStub,
	    bool success=ed_binarytrackletdisk(innerFPGAStub,outerFPGAStub,
					    outerStub->sigmaz(),
					    irinv,iphi0,it,iz0,
					    iphiproj,izproj,
					    iphider,izder,
					    minusNeighbor,plusNeighbor,
					    iphiprojdisk,irprojdisk,
					    iphiderdisk,irderdisk,
					    minusNeighbordisk,plusNeighbordisk);

	    if (debug1) {
	      cout << "FPGATrackletCalculator success = "<<success<<endl;
	    }
	    
	    if (!success) continue;
	  
	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbordisk[j]) {
		phiprojapproxdisk[j]+=dphisector;
		phiprojdisk[j]+=dphisector;
	      }
	      if (plusNeighbordisk[j]) {
		phiprojapproxdisk[j]-=dphisector;
		phiprojdisk[j]-=dphisector;
	      }	    
	    }
	    
	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbor[j]) {
		phiprojapprox[j]+=dphisector;
		phiproj[j]+=dphisector;
	      }
	      if (plusNeighbor[j]) {
		phiprojapprox[j]-=dphisector;
		phiproj[j]-=dphisector;
	      }
	    }
	    
	    
	    /* 
	       if (!phimatch) {
	       //cout << "Rejected due to phimatch "<<istubpt1<<" "
	       //	 <<istubpt2<<" "<<rinv<<" "<<t<<endl;
	       continue;
	       }
	       
	       if (!zmatch) { 
	       //cout << "Rejected due to zmatch : "<<z0approx<<" "
	       //	 <<izvm1<<" "<<izvm2<<" "<<irvm1<<" "<<irvm2<<" "
	       //	 <<z1<<" "<<z2<<" "<<r1<<" "<<r2<<endl;
	       continue;
	       }
	    */
	    
	    /*
	    //limit on the number of processed matches in each TE
	    ++processed_matches;
	    if ( processed_matches >= NMAXTE) {
	    //	    cout<<i<<" "<<processed_matches<<"\n";
	    continue; 
	    }
	    */

	    if (writeTrackletParsDisk) {
	      static ofstream out("trackletparsdisk.txt");
	      out <<"Trackpars         "<<disk_
		  <<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvparsdisk
		  <<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0parsdisk
		  <<"   "<<t<<" "<<tapprox<<" "<<it*ktparsdisk
		  <<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzdisk
		  <<endl;
	    }

	    if (writeTrackProj) {
	      static ofstream out1("trackproj.txt");
	      for (int i=0;i<3;i++) {
		double kphiproj=kphiproj123;
		out1 <<"Trackproj Disk "<<disk_<<" "<<zproj_[i]
		     <<"   "<<phiprojdisk[i]<<" "<<phiprojapproxdisk[i]
		     <<" "<<iphiprojdisk[i]*kphiproj
		     <<"   "<<rprojdisk[i]<<" "<<rprojapproxdisk[i]
		     <<" "<<irprojdisk[i]*krprojshiftdisk
		     <<"   "<<phiderdisk[i]<<" "<<phiderapproxdisk[i]
		     <<" "<<iphiderdisk[i]*kphider
		     <<"   "<<rderdisk[i]<<" "<<rderapproxdisk[i]
		     <<" "<<irderdisk[i]*krprojderdiskshift
		     <<endl;
	      }	      
	    }
	    
	    //counter is per sector
	    //++Ntracklets_;
	    //if(Ntracklets_ < NMAXproj){

	    FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						    innerFPGAStub,outerFPGAStub,
						    iSector_,
						    phioffset_,
						    rinv,phi0,z0,t,
						    rinvapprox,phi0approx,
						    z0approx,tapprox,
						    irinv,iphi0,iz0,it,
						    iphiproj,izproj,iphider,izder,
						    minusNeighbor,plusNeighbor,	
						    phiproj,zproj,phider,zder,
						    phiprojapprox,zprojapprox,
						    phiderapprox,zderapprox,
						    iphiprojdisk,irprojdisk,
						    iphiderdisk,irderdisk,
						    minusNeighbordisk,
						    plusNeighbordisk,
						    phiprojdisk,rprojdisk,
						    phiderdisk,rderdisk,
						    phiprojapproxdisk,
						    rprojapproxdisk,
						    phiderapproxdisk,
						    rderapproxdisk,
						    true);
	    
	    if (debug1) {
	      cout << "Found tracklet in disk = "<<disk_<<" "<<tracklet
		   <<" "<<iSector_<<endl;
	    }


	    countsel++;

	    //cout << "Found tracklet "<<tracklet<<endl;

	    tracklet->setTrackletIndex(trackletpars_->nTracklets());
	    tracklet->setTCIndex(0);

	    trackletpars_->addTracklet(tracklet);

	    
	    //	    cout<<" DD to L 1\n";
	    addLayerProj(tracklet,1);
	    //	    cout<<" DD to L 2\n";
	    addLayerProj(tracklet,2);

	  
	    for(unsigned int j=0;j<3;j++){
	      //cout<<"FPGATrackletCalculator projection to disk "<<dproj[j]<<"\n";
	      addDiskProj(tracklet,dproj[j]);
	    }
	  

	  } else if (innerFPGAStub->isDisk()) {

	    //Deal with overlap stubs here
	    assert(outerFPGAStub->isBarrel());

	    assert(innerFPGAStub->isDisk());

	    disk_=innerFPGAStub->disk().value();

	    if (debug1) {
	      cout << "trying to make overlap tracklet disk_ = "<<disk_<<" "<<getName()<<endl;
	    }
	      
	    int sign=1;
	    if (disk_<0) sign=-1;
	    
	    zprojoverlap_[0]=sign*zmeanD2;
	    zprojoverlap_[1]=sign*zmeanD3;
	    zprojoverlap_[2]=sign*zmeanD4;
	    zprojoverlap_[3]=sign*zmeanD5;


	    int istubpt1=innerFPGAStub->stubpt().value();
	    FPGAWord iphi1=innerFPGAStub->phi();
	    FPGAWord iz1=innerFPGAStub->z();
	    FPGAWord ir1=innerFPGAStub->r();
	    
	    int istubpt2=outerFPGAStub->stubpt().value();
	    FPGAWord iphi2=outerFPGAStub->phi();
	    FPGAWord iz2=outerFPGAStub->z();
	    FPGAWord ir2=outerFPGAStub->r();
	  

	    //cout << "istubpt1 istubpt2 : "<<istubpt1<<" "<<istubpt2<<endl;
	    assert(istubpt1>=0);
	    assert(istubpt2>=0);
	    
	   
	    //int i1=TEOverlaps_[lindex][i].first.first;
	    //int i2=TEOverlaps_[lindex][i].second.first;
	    //int j1=TEOverlaps_[lindex][i].first.second;
	    //int j2=TEOverlaps_[lindex][i].second.second;
	    

	    //bool phimatch=(*TETablesOverlap_[lindex])[i].phicheck(address,i1,i2,j1,j2);

	    //bool zmatch=(*TETablesOverlap_[lindex])[i].zcheck(izvm1,izvm2,irvm1,irvm2*2); //the *2 is a hack since we need one more bit to match the irvm1 precision
       

	    double r1=innerStub->r();
	    double z1=innerStub->z();
	    double phi1=innerStub->phi();
	    
	    double r2=outerStub->r();
	    double z2=outerStub->z();
	    double phi2=outerStub->phi();
	    
	    //Protection... Should be handled cleaner
	    //to avoid problem with floating point 
	    //calculation and with overflows
	    //in the integer calculation
	    if (r1<r2+1.5) {
	      //cout << "in overlap tracklet: radii wrong"<<endl;
	      continue;
	    }


	    double rinv,phi0,t,z0;
	    
	    double phiproj[3],zproj[3],phider[3],zder[3];
	    double phiprojdisk[4],rprojdisk[4],phiderdisk[4],rderdisk[4];

	    exacttrackletOverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				 rinv,phi0,t,z0,
				 phiproj,zproj,phider,zder,
				 phiprojdisk,rprojdisk,phiderdisk,rderdisk);


	    //Truncates floating point positions to integer
	    //representation precision
	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      //if (layer_>3) {
	      //  lphi=8;
	      //  lz=16;
	      //  lr=1;
	      //}
	      double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;
	      if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	      //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	      assert(fabs(dphi1)<1e-4);
	      phi1-=dphi1;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z1-(iz1.value()*kzdisk*lz+sign*zmean[abs(disk_)-1]);
	      //cout << "z1 iz1.value() kz : "<<z1<<" "<<iz1.value()<<" "<<kz<<" "
	      //	 <<iz1.value()*kz<<" "<<sign*zmean[abs(disk_)-1]<<endl;
	      assert(fabs(dz)<1.0*lz);
	      z1-=dz;
	      //cout <<"r1 : "<<r1<<" "<<lr*ir1.value()*krdisk+rmindisk<<endl;
	      double dr=r1-lr*ir1.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.75);
	      r1-=dr;
	      //cout << "dz1 dr1 : "<<dz<<" "<<dr<<endl;
	    }



	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=2;
	      
	      double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	      if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	      //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	      //	 <<phi2<<" "<< iphi2.value()<<endl;
	      assert(fabs(dphi2)<1e-4);
	      phi2-=dphi2;
	      //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	      double dz=z2-iz2.value()*kz*lz;
	      assert(fabs(dz)<1.0*lz);
	      z2-=dz;
	      int lindex=outerFPGAStub->layer().value();
	      //cout <<lindex<<" "<<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[lindex]<<endl;
	      double dr=r2-lr*ir2.value()*kr-rmean[lindex]; 
	      assert(fabs(dr)<0.1);
	      r2-=dr;
	    }
	    

	    double rinvapprox,phi0approx,tapprox,z0approx;
	    double phiprojapprox[3],zprojapprox[3],phiderapprox[3],zderapprox[3];
	    double phiprojapproxdisk[4],rprojapproxdisk[4],
	      phiderapproxdisk[4],rderapproxdisk[4];
	  
	    approxtrackletoverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				  rinvapprox,phi0approx,tapprox,z0approx,
				  phiprojapprox,zprojapprox,
				  phiderapprox,zderapprox,
				  phiprojapproxdisk,rprojapproxdisk,
				  phiderapproxdisk,rderapproxdisk);
	    
	    int irinv,iphi0,it,iz0;
	    int iphiproj[3],izproj[3],iphider[3],izder[3];
	    bool minusNeighbor[3],plusNeighbor[3];
	    
	    int iphiprojdisk[4],irprojdisk[4],iphiderdisk[4],irderdisk[4];
	    bool minusNeighbordisk[4],plusNeighbordisk[4];

	    //cout << "Will call binarytrackletoverlap : "<<disk_<<endl;

	    bool success=binarytrackletOverlap(innerFPGAStub,outerFPGAStub,
					       outerStub->sigmaz(),
					       irinv,iphi0,it,iz0,
					       iphiproj,izproj,
					       iphider,izder,
					       minusNeighbor,plusNeighbor,
					       iphiprojdisk,irprojdisk,
					       iphiderdisk,irderdisk,
					       minusNeighbordisk,
					       plusNeighbordisk);
	    
	  
	    //cout << "Done calling binarytrackletoverlap : "<<disk_<<endl;

	    if (!success) {
	      //cout << "binarytrackletoverlap failed"<<endl;
	      continue;
	    }

	    //cout << "Trying stub pair 3" << endl;
	  

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbordisk[j]) {
		phiprojapproxdisk[j]+=dphisector;
		phiprojdisk[j]+=dphisector;
	      }
	      if (plusNeighbordisk[j]) {
		phiprojapproxdisk[j]-=dphisector;
		phiprojdisk[j]-=dphisector;
	      }	    
	    }

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbor[j]) {
		phiprojapprox[j]+=dphisector;
		phiproj[j]+=dphisector;
	      }
	      if (plusNeighbor[j]) {
		phiprojapprox[j]-=dphisector;
		phiproj[j]-=dphisector;
	      }
	    }
	    
	  
	    //if (fabs(z0)>15.0) continue;
	    //if (fabs(rinv)>0.0057) continue;
	    
	    //if (!pass) continue;
	    

	    //if (!phimatch) { //Have to fix the fact that L1 and D1 are not offset
	    //cout << "Rejected due to phimatch "<<istubpt1<<" "
	    //	 <<istubpt2<<" "<<rinv<<" "<<t<<endl;
	    //continue;
	    //}
	    
	    //if (!zmatch) { 
	      //cout << "Rejected due to zmatch : "<<z0approx<<" "
	      //	 <<izvm1<<" "<<izvm2<<" "<<irvm1<<" "<<irvm2<<" "
	      // 	 <<z1<<" "<<z2<<" "<<r1<<" "<<r2<<endl;
	    //  continue;
	    //}

	    //limit on the number of processed matches in each TE
	    //++processed_matches;
	    //if ( processed_matches >= NMAXTE) {
	    //	    cout<<i<<" "<<processed_matches<<"\n";
	    //  continue; 
	    //}


	    if (writeTrackletParsOverlap) {
	      static ofstream out("trackletparsoverlap.txt");
	      out <<"Trackpars "<<disk_
		  <<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvparsdisk
		  <<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0parsdisk
		  <<"   "<<t<<" "<<tapprox<<" "<<it*ktparsdisk
		  <<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzdisk
		  <<endl;
	    }

	    if (writeTrackProj) {
	      static ofstream out1("trackproj.txt");
	      for (int i=0;i<3;i++) {
		double kphiproj=kphiproj123;
		out1 <<"Trackproj Overlap "<<disk_<<" "<<zproj_[i]
		     <<"   "<<phiprojdisk[i]<<" "<<phiprojapproxdisk[i]
		     <<" "<<iphiprojdisk[i]*kphiproj
		     <<"   "<<rprojdisk[i]<<" "<<rprojapproxdisk[i]
		     <<" "<<irprojdisk[i]*krprojshiftdisk
		     <<"   "<<phiderdisk[i]<<" "<<phiderapproxdisk[i]
		     <<" "<<iphiderdisk[i]*kphider
		     <<"   "<<rderdisk[i]<<" "<<rderapproxdisk[i]
		     <<" "<<irderdisk[i]*krprojderdiskshift
		     <<endl;
	      }
	      
	    }

	    //cout << "disktracklet "<<iphi0*kphi0parsdisk<<endl;
	    
	    //cout << "Found new FPGATracklet : " 
	    //     <<iphider[0]<<" "<<phider[0]<<" "<<phiderapprox[0]
	    //     <<" <> "<<irinv<<" "<<rinv<<" "<<rinvapprox<<endl;
	    
	    //++NtrackletsOverlap_;
	    //if(NtrackletsOverlap_ < NMAXproj){
	      
	    FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						    innerFPGAStub,outerFPGAStub,
						    iSector_,
						    phioffset_,
						    rinv,phi0,z0,t,
						    rinvapprox,phi0approx,
						    z0approx,tapprox,
						    irinv,iphi0,iz0,it,
						    iphiproj,izproj,iphider,izder,
						    minusNeighbor,plusNeighbor,		
						    phiproj,zproj,phider,zder,
						    phiprojapprox,zprojapprox,
						    phiderapprox,zderapprox,
						    iphiprojdisk,irprojdisk,
						    iphiderdisk,irderdisk,
						    minusNeighbordisk,
						    plusNeighbordisk,
						    phiprojdisk,rprojdisk,
						    phiderdisk,rderdisk,
						    phiprojapproxdisk,
						    rprojapproxdisk,
						    phiderapproxdisk,
						    rderapproxdisk,
						    false,true);

	    if (debug1) {
	      cout << "Found tracklet in overlap = "<<layer_<<" "<<disk_
		   <<" "<<tracklet<<" "<<iSector_<<endl;
	    }

	    //cout << "Adding overlap tracklet" << endl;

	    countsel++;

	    tracklet->setTrackletIndex(trackletpars_->nTracklets());
	    tracklet->setTCIndex(0);

	    trackletpars_->addTracklet(tracklet);
	    //FIXME  need to stick projection in correct place

	    int layer=outerFPGAStub->layer().value()+1;

	    if (layer==2) {
	      //	      cout<<" DL to 1 \n";
	      addLayerProj(tracklet,1);
	    }

	  
	    for(unsigned int disk=2;disk<6;disk++){
	      if (layer==2 && disk==5 ) continue;
	      //	      cout<<" DL to D "<<disk<<"\n";
	      addDiskProj(tracklet,disk);
	    }


	  } else {
	    
	    assert(getName()=="TC_D1L2A"||getName()=="TC_D1L2B");
	    
	    //Deal with overlap layer layer stubs here
	    assert(outerFPGAStub->isBarrel());

	    //bit of a hack here...
	    rproj_[0]=rmean[0];
	    rproj_[1]=rmean[1];
	    rproj_[2]=rmean[2];
	    rproj_[3]=rmean[3];
	    
	    assert(innerFPGAStub->isBarrel());

	    if (debug1) {
	      cout << "trying to make overlap tracklet "<<getName()<<endl;
	    }
	      
	    int sign=1;
	    if (innerStub->z()<0.0) sign=-1;
	    
	    zprojoverlap_[0]=sign*zmeanD1;
	    zprojoverlap_[1]=sign*zmeanD2;
	    zprojoverlap_[2]=sign*zmeanD3;
	    zprojoverlap_[3]=sign*zmeanD4;


	    int istubpt1=innerFPGAStub->stubpt().value();
	    FPGAWord iphi1=innerFPGAStub->phi();
	    FPGAWord iz1=innerFPGAStub->z();
	    FPGAWord ir1=innerFPGAStub->r();
	    
	    int istubpt2=outerFPGAStub->stubpt().value();
	    FPGAWord iphi2=outerFPGAStub->phi();
	    FPGAWord iz2=outerFPGAStub->z();
	    FPGAWord ir2=outerFPGAStub->r();
	  

	    //cout << "istubpt1 istubpt2 : "<<istubpt1<<" "<<istubpt2<<endl;
	    assert(istubpt1>=0);
	    assert(istubpt2>=0);
	    
	   
	    //int i1=TEOverlaps_[lindex][i].first.first;
	    //int i2=TEOverlaps_[lindex][i].second.first;
	    //int j1=TEOverlaps_[lindex][i].first.second;
	    //int j2=TEOverlaps_[lindex][i].second.second;
	    

	    //bool phimatch=(*TETablesOverlap_[lindex])[i].phicheck(address,i1,i2,j1,j2);

	    //bool zmatch=(*TETablesOverlap_[lindex])[i].zcheck(izvm1,izvm2,irvm1,irvm2*2); //the *2 is a hack since we need one more bit to match the irvm1 precision
       

	    double r1=innerStub->r();
	    double z1=innerStub->z();
	    double phi1=innerStub->phi();
	    
	    double r2=outerStub->r();
	    double z2=outerStub->z();
	    double phi2=outerStub->phi();
	    
	    //Protection... Should be handled cleaner
	    //to avoid problem with floating point 
	    //calculation and with overflows
	    //in the integer calculation
	    if (r1<r2+1.5) {
	      cout << "in overlap tracklet: radii wrong"<<endl;
	      continue;
	    }


	    double rinv,phi0,t,z0;
	    
	    double phiproj[3],zproj[3],phider[3],zder[3];
	    double phiprojdisk[4],rprojdisk[4],phiderdisk[4],rderdisk[4];

	    exacttrackletOverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				 rinv,phi0,t,z0,
				 phiproj,zproj,phider,zder,
				 phiprojdisk,rprojdisk,phiderdisk,rderdisk);


	    //Truncates floating point positions to integer
	    //representation precision
	    if (0) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      //if (layer_>3) {
	      //  lphi=8;
	      //  lz=16;
	      //  lr=1;
	      //}
	      double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;
	      if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	      //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	      assert(fabs(dphi1)<1e-4);
	      phi1-=dphi1;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z1-(iz1.value()*kzdisk*lz+sign*zmean[abs(disk_)-1]);
	      //cout << "z1 iz1.value() kz : "<<z1<<" "<<iz1.value()<<" "<<kz<<" "
	      //	 <<iz1.value()*kz<<" "<<sign*zmean[abs(disk_)-1]<<endl;
	      assert(fabs(dz)<1.0*lz);
	      z1-=dz;
	      //cout <<"r1 : "<<r1<<" "<<lr*ir1.value()*krdisk+rmindisk<<endl;
	      double dr=r1-lr*ir1.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.75);
	      r1-=dr;
	      //cout << "dz1 dr1 : "<<dz<<" "<<dr<<endl;
	    }



	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=2;
	      
	      double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	      if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	      //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	      //	 <<phi2<<" "<< iphi2.value()<<endl;
	      assert(fabs(dphi2)<1e-4);
	      phi2-=dphi2;
	      //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	      double dz=z2-iz2.value()*kz*lz;
	      assert(fabs(dz)<1.0*lz);
	      z2-=dz;
	      int lindex=outerFPGAStub->layer().value();
	      //cout <<lindex<<" "<<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[lindex]<<endl;
	      double dr=r2-lr*ir2.value()*kr-rmean[lindex]; 
	      assert(fabs(dr)<0.1);
	      r2-=dr;
	    }
	    

	    double rinvapprox,phi0approx,tapprox,z0approx;
	    double phiprojapprox[3],zprojapprox[3],phiderapprox[3],zderapprox[3];
	    double phiprojapproxdisk[4],rprojapproxdisk[4],
	      phiderapproxdisk[4],rderapproxdisk[4];
	  
	    approxtrackletoverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				  rinvapprox,phi0approx,tapprox,z0approx,
				  phiprojapprox,zprojapprox,
				  phiderapprox,zderapprox,
				  phiprojapproxdisk,rprojapproxdisk,
				  phiderapproxdisk,rderapproxdisk);
	    
	    int irinv,iphi0,it,iz0;
	    int iphiproj[4],izproj[4],iphider[4],izder[4];
	    bool minusNeighbor[4],plusNeighbor[4];
	    
	    int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
	    bool minusNeighbordisk[5],plusNeighbordisk[5];

	    //cout << "Will call binarytrackletoverlap : "<<disk_<<endl;

	    //order backward here!!!
	    bool success=binarytracklet(outerFPGAStub,innerFPGAStub,
					outerStub->sigmaz(),
					irinv,iphi0,it,iz0,
					iphiproj,izproj,
					iphider,izder,
					minusNeighbor,plusNeighbor,
					iphiprojdisk,irprojdisk,
					iphiderdisk,irderdisk,
					minusNeighbordisk,
					plusNeighbordisk);
	    
	  
	    //cout << "Done calling binarytrackletoverlap : "<<disk_<<endl;

	    if (!success) {
	      //cout << "binarytrackletoverlap layer layer failed"<<endl;
	      continue;
	    }

	    //cout << "Trying stub pair 3" << endl;
	  

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbordisk[j]) {
		phiprojapproxdisk[j]+=dphisector;
		phiprojdisk[j]+=dphisector;
	      }
	      if (plusNeighbordisk[j]) {
		phiprojapproxdisk[j]-=dphisector;
		phiprojdisk[j]-=dphisector;
	      }	    
	    }

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbor[j]) {
		phiprojapprox[j]+=dphisector;
		phiproj[j]+=dphisector;
	      }
	      if (plusNeighbor[j]) {
		phiprojapprox[j]-=dphisector;
		phiproj[j]-=dphisector;
	      }
	    }
	    
	  


	    if (writeTrackletParsOverlap) {
	      static ofstream out("trackletparsoverlap.txt");
	      out <<"Trackpars "<<disk_
		  <<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvparsdisk
		  <<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0parsdisk
		  <<"   "<<t<<" "<<tapprox<<" "<<it*ktparsdisk
		  <<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzdisk
		  <<endl;
	    }

	    if (writeTrackProj) {
	      static ofstream out1("trackproj.txt");
	      for (int i=0;i<3;i++) {
		double kphiproj=kphiproj123;
		out1 <<"Trackproj Overlap "<<disk_<<" "<<zproj_[i]
		     <<"   "<<phiprojdisk[i]<<" "<<phiprojapproxdisk[i]
		     <<" "<<iphiprojdisk[i]*kphiproj
		     <<"   "<<rprojdisk[i]<<" "<<rprojapproxdisk[i]
		     <<" "<<irprojdisk[i]*krprojshiftdisk
		     <<"   "<<phiderdisk[i]<<" "<<phiderapproxdisk[i]
		     <<" "<<iphiderdisk[i]*kphider
		     <<"   "<<rderdisk[i]<<" "<<rderapproxdisk[i]
		     <<" "<<irderdisk[i]*krprojderdiskshift
		     <<endl;
	      }
	      
	    }

	      
	    FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						    innerFPGAStub,outerFPGAStub,
						    iSector_,
						    phioffset_,
						    rinv,phi0,z0,t,
						    rinvapprox,phi0approx,
						    z0approx,tapprox,
						    irinv,iphi0,iz0,it,
						    iphiproj,izproj,iphider,izder,
						    minusNeighbor,plusNeighbor,		
						    phiproj,zproj,phider,zder,
						    phiprojapprox,zprojapprox,
						    phiderapprox,zderapprox,
						    iphiprojdisk,irprojdisk,
						    iphiderdisk,irderdisk,
						    minusNeighbordisk,
						    plusNeighbordisk,
						    phiprojdisk,rprojdisk,
						    phiderdisk,rderdisk,
						    phiprojapproxdisk,
						    rprojapproxdisk,
						    phiderapproxdisk,
						    rderapproxdisk,
						    false,true);

	    if (debug1) { 
	      cout << "Found tracklet in overlap layer layer = "<<layer_<<" "<<disk_
		   <<" "<<tracklet<<" "<<iSector_<<endl;
	    }

	    //cout << "Adding overlap tracklet" << endl;

	    countsel++;

	    tracklet->setTrackletIndex(trackletpars_->nTracklets());
	    tracklet->setTCIndex(0);

	    trackletpars_->addTracklet(tracklet);
	    //FIXME  need to stick projection in correct place

	    addLayerProj(tracklet,1);
	    //addLayerProj(tracklet,4);

	  
	    for(unsigned int disk=1;disk<5;disk++){
	      //	      cout<<" DL to D "<<disk<<"\n";
	      addDiskProj(tracklet,disk);
	    }

	    //cout << "Done with overlap TC layer layer"<<endl;

	  }
	}
	if (countall>=MAXTC) break;
      }
      if (countall>=MAXTC) break;
    }

    if (writeTrackletCalculator) {
      static ofstream out("trackletcalculator.txt");
      out << getName()<<" "<<countall<<" "<<countsel<<endl;
    }


  }



  void exacttrackletOverlap(double r1, double z1, double phi1,
			    double r2, double z2, double phi2, double sigmaz,
			    double& rinv, double& phi0,
			    double& t, double& z0,
			    double phiprojLayer[3], double zprojLayer[3], 
			    double phiderLayer[3], double zderLayer[3],
			    double phiproj[3], double rproj[3], 
			    double phider[3], double rder[3]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    if (r1>r2) rinv=-rinv;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    //cout << "phi1 phi2 phi1tmp : "<<phi1<<" "<<phi2<<" "<<phi1tmp<<endl;

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    if (!(fabs(phi0)<0.5*two_pi)) {
      cout << "phi1tmp r1 rinv phi0 deltaphi dist: "
	   <<phi1tmp<<" "<<r1<<" "<<rinv<<" "<<phi0
	   <<" "<<deltaphi<<" "<<dist<<endl;
      exit(1);
    }
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;


    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<4;i++) {
      exactprojdisk(zprojoverlap_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<1;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }


  void approxtrackletoverlap(double r1, double z1, double phi1,
			     double r2, double z2, double phi2, double sigmaz,
			     double &rinv, double &phi0,
			     double &t, double &z0,
			     double phiprojLayer[4], double rprojLayer[4], 
			     double phiderLayer[4], double rderLayer[4],
			     double phiproj[4], double rproj[4], 
			     double phider[4], double rder[4]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    if (phi1tmp>two_pi) phi1-=two_pi;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_
    //	 <<" "<<phimax_<<" "<<phi1tmp<<endl;
    assert(phi1tmp>-1e-10);

    //cout << "DUMPPARS01 : "<<r1<<" "<<r2<<endl;

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;// *(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;// -2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;// *(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;// *(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;// +3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (abs(disk_)==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS1:"  //overlap 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" "<<r2-r1
	     <<" > "<<r1
	     <<" "<<r2
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" "<<rinv
	     <<" t= "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" phi0= "<<phi0
	     <<" * "<<t12
	     <<" "<<z1
	     <<" z0= "<<z0
	     <<endl;
      }
      

//      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
//	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
//	   <<" delta="<<delta
//	   <<" t5="<<t5
//           <<" deltainv="<<deltainv
//	   <<" rinv="<<rinv
//	   <<" t="<<t
//	   <<" r1abs="<<r1
//	   <<" t7="<<t7
//	   <<" t9="<<t9
//	   <<" phi1="<<phi1tmp
//	   <<" ***phi0="<<phi0
//	   <<" t12="<<t12
//	   <<" z1="<<z1
//	   <<" z0="<<z0
//	   <<endl;

   

    }
   

    //calculate projection



    for (int i=0;i<1;i++) {
      approxproj(rmean[i],rinv,phi0,t,z0,
		 phiprojLayer[i],rprojLayer[i],
		 phiderLayer[i],rderLayer[i]);
    }


    //static ofstream out1("neighborproj.txt");

    for (int i=0;i<4;i++) {
      approxprojdisk(zprojoverlap_[i],rinv,phi0,t,z0,
		     phiproj[i],rproj[i],phider[i],rder[i]);
      /*
      if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	if ((fabs(rproj[i])<100.0)&&(phiproj[i]<(phimax_-phimin_)/6)){
	  out1<<disk_<<" -1 "<<phiproj[i]<<endl;
	} else  if ((fabs(rproj[i])<100.0)&&(phiproj[i]>7.0*(phimax_-phimin_)/6)){
	  out1<<disk_<<" +1 "<<phiproj[i]<<endl;
	} else if (fabs(rproj[i])<100.0){
	  out1<<disk_<<" 0 "<<phiproj[i]<<endl;
	}
      }
      */
    }


  }



  void addDiskProj(FPGATracklet* tracklet, int disk){

    //cout << "addDiskProj projection to disk = "<<disk<<endl;

    
    
    FPGAWord fpgar=tracklet->fpgarprojdisk(disk);

    //cout << "r of projection : "<<fpgar.value()<<" disk = "<<disk<<endl;
    
    //if (fpgar.value()*krprojshiftdisk<rmindisk) return;
    //if (fpgar.value()*krprojshiftdisk>rmaxdisk) return;

    if (fpgar.value()*krprojshiftdisk<12.0) return;
    if (fpgar.value()*krprojshiftdisk>108.0) return;


    if (tracklet->plusNeighborDisk(disk)) {
      //cout << "addDiskProj projection to plus neighbor "<<endl;
      if (getName().find("L1L2")!=std::string::npos) {
	if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_L6Plus_,tracklet);
	if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_L5Plus_,tracklet);
	if (abs(disk)==3) addNeighborProjectionDisk(disk,trackletproj_L4Plus_,tracklet);
	if (abs(disk)==4) addNeighborProjectionDisk(disk,trackletproj_L3Plus_,tracklet);
	return;
      }
      if (getName().find("L3L4")!=std::string::npos) {
	if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_L6Plus_,tracklet);
	if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_L5Plus_,tracklet);
	return;
      }
      if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_F1Plus_,tracklet);
      if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_F2Plus_,tracklet);
      if (abs(disk)==3) addNeighborProjectionDisk(disk,trackletproj_F3Plus_,tracklet);
      if (abs(disk)==4) addNeighborProjectionDisk(disk,trackletproj_F4Plus_,tracklet);
      if (abs(disk)==5) addNeighborProjectionDisk(disk,trackletproj_F5Plus_,tracklet);
      //if (disk==-1) addNeighborProjectionDisk(disk,trackletproj_B1Plus_,tracklet);
      //if (disk==-2) addNeighborProjectionDisk(disk,trackletproj_B2Plus_,tracklet);
      //if (disk==-3) addNeighborProjectionDisk(disk,trackletproj_B3Plus_,tracklet);
      //if (disk==-4) addNeighborProjectionDisk(disk,trackletproj_B4Plus_,tracklet);
      //if (disk==-5) addNeighborProjectionDisk(disk,trackletproj_B5Plus_,tracklet);
      return;
    }
      
    if (tracklet->minusNeighborDisk(disk)) {
      //cout << "addDiskProj projection to minus neighbor "<<endl;
      if (getName().find("L1L2")!=std::string::npos) {
	if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_L6Minus_,tracklet);
	if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_L5Minus_,tracklet);
	if (abs(disk)==3) addNeighborProjectionDisk(disk,trackletproj_L4Minus_,tracklet);
	if (abs(disk)==4) addNeighborProjectionDisk(disk,trackletproj_L3Minus_,tracklet);
	return;
      }
      if (getName().find("L3L4")!=std::string::npos) {
	if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_L6Minus_,tracklet);
	if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_L5Minus_,tracklet);
	return;
      }
      if (abs(disk)==1) addNeighborProjectionDisk(disk,trackletproj_F1Minus_,tracklet);
      if (abs(disk)==2) addNeighborProjectionDisk(disk,trackletproj_F2Minus_,tracklet);
      if (abs(disk)==3) addNeighborProjectionDisk(disk,trackletproj_F3Minus_,tracklet);
      if (abs(disk)==4) addNeighborProjectionDisk(disk,trackletproj_F4Minus_,tracklet);
      if (abs(disk)==5) addNeighborProjectionDisk(disk,trackletproj_F5Minus_,tracklet);
      //if (disk==-1) addNeighborProjectionDisk(disk,trackletproj_B1Minus_,tracklet);
      //if (disk==-2) addNeighborProjectionDisk(disk,trackletproj_B2Minus_,tracklet);
      //if (disk==-3) addNeighborProjectionDisk(disk,trackletproj_B3Minus_,tracklet);
      //if (disk==-4) addNeighborProjectionDisk(disk,trackletproj_B4Minus_,tracklet);
      //if (disk==-5) addNeighborProjectionDisk(disk,trackletproj_B5Minus_,tracklet);
      return;
    }
      
    //cout << "addDiskProj projection to center "<<endl;

    FPGAWord fpgaphi=tracklet->fpgaphiprojdisk(disk);

    //cout << "FPGATrackletCalculator phi nbits : "<<fpgaphi.value()<<" "<<fpgaphi.nbits()<<endl;
    
    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);
    
    
    assert(iphivmRaw>=4);
    assert(iphivmRaw<=27);

    int iphi=(iphivmRaw-4)>>3;

    //cout << "disk fpgaphi iphivmRaw iphi : "<<disk<<" "<<fpgaphi.value()<<" "<<iphivmRaw<<" "<<iphi<<endl;

    
    assert(iphi>=0);
    assert(iphi<=2);

    if (abs(disk)==1) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_F1PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_F1PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_F1PHI3_,tracklet);
    }
    
    if (abs(disk)==2) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_F2PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_F2PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_F2PHI3_,tracklet);
    }

    if (abs(disk)==3) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_F3PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_F3PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_F3PHI3_,tracklet);
    }

    if (abs(disk)==4) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_F4PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_F4PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_F4PHI3_,tracklet);
    }

    if (abs(disk)==5) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_F5PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_F5PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_F5PHI3_,tracklet);
    }

  }


  bool addLayerProj(FPGATracklet* tracklet, int layer){

    //cout << "FPGATrackletCalculator:addLayerProj in "<<getName()<<" for layer "<<layer<<endl;
    
    
    assert(layer>0);

    FPGAWord fpgaz=tracklet->fpgazproj(layer);
    FPGAWord fpgaphi=tracklet->fpgaphiproj(layer);
    
    assert(!fpgaphi.atExtreme());
    
    if (fpgaz.atExtreme()) return false;

    if (fabs(fpgaz.value()*kz)>zlength) return false;

    if (tracklet->plusNeighbor(layer)) {
      if (layer==1) addNeighborProjection(layer,trackletproj_L1Plus_,tracklet);
      if (layer==2) addNeighborProjection(layer,trackletproj_L2Plus_,tracklet);
      if (layer==3) addNeighborProjection(layer,trackletproj_L3Plus_,tracklet);
      if (layer==4) addNeighborProjection(layer,trackletproj_L4Plus_,tracklet);
      if (layer==5) addNeighborProjection(layer,trackletproj_L5Plus_,tracklet);
      if (layer==6) addNeighborProjection(layer,trackletproj_L6Plus_,tracklet);
      return true;
    }
      
    if (tracklet->minusNeighbor(layer)) {
      if (layer==1) addNeighborProjection(layer,trackletproj_L1Minus_,tracklet);
      if (layer==2) addNeighborProjection(layer,trackletproj_L2Minus_,tracklet);
      if (layer==3) addNeighborProjection(layer,trackletproj_L3Minus_,tracklet);
      if (layer==4) addNeighborProjection(layer,trackletproj_L4Minus_,tracklet);
      if (layer==5) addNeighborProjection(layer,trackletproj_L5Minus_,tracklet);
      if (layer==6) addNeighborProjection(layer,trackletproj_L6Minus_,tracklet);
      return true;
    }
      


    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);
    
    
    assert(iphivmRaw>=4);
    assert(iphivmRaw<=27);

    int iphi=(iphivmRaw-4)>>3;

    //cout << "layer fpgaphi iphivmRaw iphi : "<<layer<<" "<<fpgaphi.value()<<" "<<iphivmRaw<<" "<<iphi<<endl;

    
    assert(iphi>=0);
    assert(iphi<=2);

    if (layer==1) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L1PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L1PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L1PHI3_,tracklet);
    }
    
    if (layer==2) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L2PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L2PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L2PHI3_,tracklet);
    }

    if (layer==3) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L3PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L3PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L3PHI3_,tracklet);
    }

    if (layer==4) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L4PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L4PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L4PHI3_,tracklet);
    }

    if (layer==5) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L5PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L5PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L5PHI3_,tracklet);
    }

    if (layer==6) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L6PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L6PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L6PHI3_,tracklet);
    }

    return true;

  }

  void addProjection(int layer,int iphi,FPGATrackletProjections* trackletprojs, FPGATracklet* tracklet){
    if (trackletprojs==0) {
      cout << "No projection memory exists in "<<getName()<<" for layer = "<<layer<<" iphi = "<<iphi+1<<endl;
      return;
    }
    assert(trackletprojs!=0);
    //cout << "FPGATrackletCalcultor::addProjection in "<<getName()
    //	 <<" adding projection to "<<trackletprojs->getName()<<endl;
    trackletprojs->addProj(tracklet);
  }

  void addProjectionDisk(int disk,int iphi,FPGATrackletProjections* trackletprojs, FPGATracklet* tracklet){
    if (trackletprojs==0) {
      if (layer_==3&&abs(disk)==3) return; //L3L4 projections to D3 are not used.
      cout << "No projection memory exists in "<<getName()<<" for disk = "<<abs(disk)<<" iphi = "<<iphi+1<<endl;
      return;
    }
    assert(trackletprojs!=0);
    //cout << "FPGATrackletCalcultor::addProjectionDisk in "<<getName()
    //	 <<" adding projection to "<<trackletprojs->getName()<<endl;
    trackletprojs->addProj(tracklet);
  }

  void addNeighborProjection(int layer, FPGATrackletProjections* trackletprojs,FPGATracklet* tracklet){
    if (trackletprojs==0) {
      string str="";
      if (tracklet->minusNeighbor(layer)){
	str="Minus";
      }
      if (tracklet->plusNeighbor(layer)){
	str="Plus";
      }
      assert(str!="");
      cout << "Error no projection memory exists in "<<getName()<<" for layer = "<<layer<<" to "
	   <<str<<" neighbor"<<endl;
      return;
    }
    assert(trackletprojs!=0);
    //cout << "FPGATrackletCalcultor::addProjection in "<<getName()
    //	 <<" adding projection to "<<trackletprojs->getName()<<endl;
    trackletprojs->addProj(tracklet);
  

  }

  
  void addNeighborProjectionDisk(int disk, FPGATrackletProjections* trackletprojs,FPGATracklet* tracklet){
    if (trackletprojs==0) {
      string str="";
      if (tracklet->minusNeighborDisk(disk)){
	str="Minus";
      }
      if (tracklet->plusNeighborDisk(disk)){
	str="Plus";
      }
      assert(str!="");
      cout << "Error no projection memory exists in "<<getName()<<" for disk = "<<abs(disk)<<" to "
	   <<str<<" neighbor"<<endl;
      return;
    }
    assert(trackletprojs!=0);
    //cout << "FPGATrackletCalcultor::addProjection in "<<getName()
    //	 <<" adding projection to "<<trackletprojs->getName()<<endl;
    trackletprojs->addProj(tracklet);
  

  }

    
  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }
 
    
private:

  int layer_;
  int disk_;
  double phimin_;
  double phimax_;
  double phioffset_;
  double rproj_[4];
  double zproj_[3];
  double zprojoverlap_[4];

  FPGAAllStubs* innerallstubs_;
  FPGAAllStubs* outerallstubs_;
  vector<FPGAStubPairs*> stubpairs_;

  FPGAInverseTable invRTable_;
  FPGAInverseTable invTTable_;
  FPGAInverseTable invTTableNeg_;

  FPGATrackletParameters* trackletpars_;

  FPGATrackletProjections* trackletproj_L1PHI1_;
  FPGATrackletProjections* trackletproj_L1PHI2_;
  FPGATrackletProjections* trackletproj_L1PHI3_;

  FPGATrackletProjections* trackletproj_L2PHI1_;
  FPGATrackletProjections* trackletproj_L2PHI2_;
  FPGATrackletProjections* trackletproj_L2PHI3_;

  FPGATrackletProjections* trackletproj_L3PHI1_;
  FPGATrackletProjections* trackletproj_L3PHI2_;
  FPGATrackletProjections* trackletproj_L3PHI3_;

  FPGATrackletProjections* trackletproj_L4PHI1_;
  FPGATrackletProjections* trackletproj_L4PHI2_;
  FPGATrackletProjections* trackletproj_L4PHI3_;

  FPGATrackletProjections* trackletproj_L5PHI1_;
  FPGATrackletProjections* trackletproj_L5PHI2_;
  FPGATrackletProjections* trackletproj_L5PHI3_;

  FPGATrackletProjections* trackletproj_L6PHI1_;
  FPGATrackletProjections* trackletproj_L6PHI2_;
  FPGATrackletProjections* trackletproj_L6PHI3_;

  FPGATrackletProjections* trackletproj_F1PHI1_;
  FPGATrackletProjections* trackletproj_F1PHI2_;
  FPGATrackletProjections* trackletproj_F1PHI3_;

  FPGATrackletProjections* trackletproj_F2PHI1_;
  FPGATrackletProjections* trackletproj_F2PHI2_;
  FPGATrackletProjections* trackletproj_F2PHI3_;

  FPGATrackletProjections* trackletproj_F3PHI1_;
  FPGATrackletProjections* trackletproj_F3PHI2_;
  FPGATrackletProjections* trackletproj_F3PHI3_;

  FPGATrackletProjections* trackletproj_F4PHI1_;
  FPGATrackletProjections* trackletproj_F4PHI2_;
  FPGATrackletProjections* trackletproj_F4PHI3_;

  FPGATrackletProjections* trackletproj_F5PHI1_;
  FPGATrackletProjections* trackletproj_F5PHI2_;
  FPGATrackletProjections* trackletproj_F5PHI3_;


  FPGATrackletProjections* trackletproj_B1PHI1_;
  FPGATrackletProjections* trackletproj_B1PHI2_;
  FPGATrackletProjections* trackletproj_B1PHI3_;

  FPGATrackletProjections* trackletproj_B2PHI1_;
  FPGATrackletProjections* trackletproj_B2PHI2_;
  FPGATrackletProjections* trackletproj_B2PHI3_;

  FPGATrackletProjections* trackletproj_B3PHI1_;
  FPGATrackletProjections* trackletproj_B3PHI2_;
  FPGATrackletProjections* trackletproj_B3PHI3_;

  FPGATrackletProjections* trackletproj_B4PHI1_;
  FPGATrackletProjections* trackletproj_B4PHI2_;
  FPGATrackletProjections* trackletproj_B4PHI3_;

  FPGATrackletProjections* trackletproj_B5PHI1_;
  FPGATrackletProjections* trackletproj_B5PHI2_;
  FPGATrackletProjections* trackletproj_B5PHI3_;

  

  
  FPGATrackletProjections* trackletproj_L1Plus_; 
  FPGATrackletProjections* trackletproj_L1Minus_;
			                         
  FPGATrackletProjections* trackletproj_L2Plus_; 
  FPGATrackletProjections* trackletproj_L2Minus_;
			                         
  FPGATrackletProjections* trackletproj_L3Plus_; 
  FPGATrackletProjections* trackletproj_L3Minus_;
			                         
  FPGATrackletProjections* trackletproj_L4Plus_; 
  FPGATrackletProjections* trackletproj_L4Minus_;
			                         
  FPGATrackletProjections* trackletproj_L5Plus_; 
  FPGATrackletProjections* trackletproj_L5Minus_;
			                         
  FPGATrackletProjections* trackletproj_L6Plus_; 
  FPGATrackletProjections* trackletproj_L6Minus_;


  FPGATrackletProjections* trackletproj_F1Plus_; 
  FPGATrackletProjections* trackletproj_F1Minus_;
			                         
  FPGATrackletProjections* trackletproj_F2Plus_; 
  FPGATrackletProjections* trackletproj_F2Minus_;
			                         
  FPGATrackletProjections* trackletproj_F3Plus_; 
  FPGATrackletProjections* trackletproj_F3Minus_;
			                         
  FPGATrackletProjections* trackletproj_F4Plus_; 
  FPGATrackletProjections* trackletproj_F4Minus_;
			                         
  FPGATrackletProjections* trackletproj_F5Plus_; 
  FPGATrackletProjections* trackletproj_F5Minus_;

  
  FPGATrackletProjections* trackletproj_B1Plus_; 
  FPGATrackletProjections* trackletproj_B1Minus_;
			                         
  FPGATrackletProjections* trackletproj_B2Plus_; 
  FPGATrackletProjections* trackletproj_B2Minus_;
			                         
  FPGATrackletProjections* trackletproj_B3Plus_; 
  FPGATrackletProjections* trackletproj_B3Minus_;
			                         
  FPGATrackletProjections* trackletproj_B4Plus_; 
  FPGATrackletProjections* trackletproj_B4Minus_;
			                         
  FPGATrackletProjections* trackletproj_B5Plus_; 
  FPGATrackletProjections* trackletproj_B5Minus_;
  
};

#endif
