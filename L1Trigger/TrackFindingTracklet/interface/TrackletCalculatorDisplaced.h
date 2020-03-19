//This class implementes the tracklet engine
#ifndef TRACKLETCALCULATORDISPLACED_H
#define TRACKLETCALCULATORDISPLACED_H

#include "IMATH_TrackletCalculator.h"
#include "IMATH_TrackletCalculatorDisk.h"
#include "IMATH_TrackletCalculatorOverlap.h"

#include "ProcessBase.h"
#include "TrackletProjectionsMemory.h"
#include "StubTripletsMemory.h"
#include "Constants.h"

using namespace std;

class TrackletCalculatorDisplaced:public ProcessBase{

public:

  TrackletCalculatorDisplaced(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    double dphi=2*M_PI/NSector;
    double dphiHG=0.5*dphisectorHG-M_PI/NSector;
    phimin_=iSector_*dphi-dphiHG;
    phimax_=phimin_+dphi+2*dphiHG;
    phimin_-=M_PI/NSector;
    phimax_-=M_PI/NSector;
    if (phimin_>M_PI) phimin_-=2*M_PI;
    if (phimax_>M_PI) phimax_-=2*M_PI;
    if (phimin_>phimax_)  phimin_-=2*M_PI;

    maxtracklet_=127;
    
    trackletproj_L1PHI1_=0;
    trackletproj_L1PHI2_=0;
    trackletproj_L1PHI3_=0;
    trackletproj_L1PHI4_=0;
    trackletproj_L1PHI5_=0;
    trackletproj_L1PHI6_=0;
    trackletproj_L1PHI7_=0;
    trackletproj_L1PHI8_=0;
   

   trackletproj_L2PHI1_=0;
   trackletproj_L2PHI2_=0;
   trackletproj_L2PHI3_=0;
   trackletproj_L2PHI4_=0;

   trackletproj_L3PHI1_=0;
   trackletproj_L3PHI2_=0;
   trackletproj_L3PHI3_=0;
   trackletproj_L3PHI4_=0;

   trackletproj_L4PHI1_=0;
   trackletproj_L4PHI2_=0;
   trackletproj_L4PHI3_=0;
   trackletproj_L4PHI4_=0;

   trackletproj_L5PHI1_=0;
   trackletproj_L5PHI2_=0;
   trackletproj_L5PHI3_=0;
   trackletproj_L5PHI4_=0;

   trackletproj_L6PHI1_=0;
   trackletproj_L6PHI2_=0;
   trackletproj_L6PHI3_=0;
   trackletproj_L6PHI4_=0;

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

   trackletproj_D1PHI1_=0;
   trackletproj_D1PHI2_=0;
   trackletproj_D1PHI3_=0;
   trackletproj_D1PHI4_=0;

   trackletproj_D2PHI1_=0;
   trackletproj_D2PHI2_=0;
   trackletproj_D2PHI3_=0;
   trackletproj_D2PHI4_=0;

   trackletproj_D3PHI1_=0;
   trackletproj_D3PHI2_=0;
   trackletproj_D3PHI3_=0;
   trackletproj_D3PHI4_=0;

   trackletproj_D4PHI1_=0;
   trackletproj_D4PHI2_=0;
   trackletproj_D4PHI3_=0;
   trackletproj_D4PHI4_=0;

   trackletproj_D5PHI1_=0;
   trackletproj_D5PHI2_=0;
   trackletproj_D5PHI3_=0;
   trackletproj_D5PHI4_=0;

   trackletproj_D1Plus_=0; 
   trackletproj_D1Minus_=0;
                         
   trackletproj_D2Plus_=0; 
   trackletproj_D2Minus_=0;
                         
   trackletproj_D3Plus_=0; 
   trackletproj_D3Minus_=0;
                         
   trackletproj_D4Plus_=0; 
   trackletproj_D4Minus_=0;
                         
   trackletproj_D5Plus_=0; 
   trackletproj_D5Minus_=0;

  
   layer_=0;
   disk_=0;

   string name1 = name.substr(1);//this is to correct for "TCD" having one more letter then "TC"
   if (name1[3]=='L') layer_=name1[4]-'0';    
   if (name1[3]=='D') disk_=name1[4]-'0';    


   // set TC index
   int iTC = -1;
   int iSeed = -1;
   
   if      (name1[9]=='A') iTC =0;
   else if (name1[9]=='B') iTC =1;
   else if (name1[9]=='C') iTC =2;
   else if (name1[9]=='D') iTC =3;
   else if (name1[9]=='E') iTC =4;
   else if (name1[9]=='F') iTC =5;
   else if (name1[9]=='G') iTC =6;
   else if (name1[9]=='H') iTC =7;
   else if (name1[9]=='I') iTC =8;
   else if (name1[9]=='J') iTC =9;
   else if (name1[9]=='K') iTC =10;
   else if (name1[9]=='L') iTC =11;
   else if (name1[9]=='M') iTC =12;
   else if (name1[9]=='N') iTC =13;
   else if (name1[9]=='O') iTC =14;

   assert(iTC!=-1);
   
   if (name1.substr(3,6)=="L3L4L2") iSeed = 8;
   else if (name1.substr(3,6)=="L5L6L4") iSeed = 9;
   else if (name1.substr(3,6)=="L2L3D1") iSeed = 10;
   else if (name1.substr(3,6)=="D1D2L2") iSeed = 11;

   assert(iSeed!=-1);

   TCIndex_ = (iSeed<<4) + iTC;
   assert(TCIndex_>=128 && TCIndex_<191);
   
   assert((layer_!=0)||(disk_!=0));

   toR_.clear();
   toZ_.clear();
   
   if (iSeed==8||iSeed==9) {
     if (layer_==3) {
       rproj_[0]=rmeanL1;
       rproj_[1]=rmeanL5;
       rproj_[2]=rmeanL6;
       lproj_[0]=1;
       lproj_[1]=5;
       lproj_[2]=6;

       dproj_[0]=1;
       dproj_[1]=2;
       dproj_[2]=0;
       toZ_.push_back(zmeanD1);
       toZ_.push_back(zmeanD2);
       
     }     
     if (layer_==5) {
       rproj_[0]=rmeanL1;
       rproj_[1]=rmeanL2;
       rproj_[2]=rmeanL3;
       lproj_[0]=1;
       lproj_[1]=2;
       lproj_[2]=3;
       
       dproj_[0]=0;
       dproj_[1]=0;
       dproj_[2]=0;
     }
     for(int i=0; i<3; ++i) toR_.push_back(rproj_[i]);
   }


   if (iSeed==10||iSeed==11) {
     if (layer_==2) {
       rproj_[0]=rmeanL1;
       lproj_[0]=1;
       lproj_[1]=-1;
       lproj_[2]=-1;

       zproj_[0]=zmeanD2;
       zproj_[1]=zmeanD3;
       zproj_[2]=zmeanD4;
       dproj_[0]=2;
       dproj_[1]=3;
       dproj_[2]=4;
     }
     if (disk_==1) {
       rproj_[0]=rmeanL1;
       lproj_[0]=1;
       lproj_[1]=-1;
       lproj_[2]=-1;

       zproj_[0]=zmeanD3;
       zproj_[1]=zmeanD4;
       zproj_[2]=zmeanD5;
       dproj_[0]=3;
       dproj_[1]=4;
       dproj_[2]=5;
     }
     toR_.push_back(rmeanL1);
     for(int i=0; i<3; ++i) toZ_.push_back(zproj_[i]);
   }
      
  }

  void addOutputProjection(TrackletProjectionsMemory* &outputProj, MemoryBase* memory){
      outputProj=dynamic_cast<TrackletProjectionsMemory*>(memory);
      assert(outputProj!=0);
  }
  
  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="trackpar"){
      TrackletParametersMemory* tmp=dynamic_cast<TrackletParametersMemory*>(memory);
      assert(tmp!=0);
      trackletpars_=tmp;
      return;
    }


    if (output=="projoutL1PHI1"||output=="projoutL1PHIA") {
      addOutputProjection(trackletproj_L1PHI1_,memory);
      return;
    }
    
    if (output=="projoutL1PHI2"||output=="projoutL1PHIB") {
      addOutputProjection(trackletproj_L1PHI2_,memory);
      return;
    }

    if (output=="projoutL1PHI3"||output=="projoutL1PHIC"){
      addOutputProjection(trackletproj_L1PHI3_,memory);
      return;
    }

    if (output=="projoutL1PHID"){
      addOutputProjection(trackletproj_L1PHI4_,memory);
      return;
    }

    if (output=="projoutL1PHIE"){
      addOutputProjection(trackletproj_L1PHI5_,memory);
      return;
    }

    if (output=="projoutL1PHIF"){
      addOutputProjection(trackletproj_L1PHI6_,memory);
      return;
    }

    if (output=="projoutL1PHIG"){
      addOutputProjection(trackletproj_L1PHI7_,memory);
      return;
    }

    if (output=="projoutL1PHIH"){
      addOutputProjection(trackletproj_L1PHI8_,memory);
      return;
    }

    if (output=="projoutL2PHI1"||output=="projoutL2PHIA"){
      addOutputProjection(trackletproj_L2PHI1_,memory);
      return;
    }

    if (output=="projoutL2PHI2"||output=="projoutL2PHIB"){
      addOutputProjection(trackletproj_L2PHI2_,memory);
      return;
    }

    if (output=="projoutL2PHI3"||output=="projoutL2PHIC"){
      addOutputProjection(trackletproj_L2PHI3_,memory);
      return;
    }

    if (output=="projoutL2PHI4"||output=="projoutL2PHID"){
      addOutputProjection(trackletproj_L2PHI4_,memory);
      return;
    }

    if (output=="projoutL3PHI1"||output=="projoutL3PHIA"){
      addOutputProjection(trackletproj_L3PHI1_,memory);
      return;
    }

    if (output=="projoutL3PHI2"||output=="projoutL3PHIB"){
      addOutputProjection(trackletproj_L3PHI2_,memory);
      return;
    }

    if (output=="projoutL3PHI3"||output=="projoutL3PHIC"){
      addOutputProjection(trackletproj_L3PHI3_,memory);
      return;
    }

    if (output=="projoutL3PHI4"||output=="projoutL3PHID"){
      addOutputProjection(trackletproj_L3PHI4_,memory);
      return;
    }

    if (output=="projoutL4PHI1"||output=="projoutL4PHIA"){
      addOutputProjection(trackletproj_L4PHI1_,memory);
      return;
    }

    if (output=="projoutL4PHI2"||output=="projoutL4PHIB"){
      addOutputProjection(trackletproj_L4PHI2_,memory);
      return;
    }

    if (output=="projoutL4PHI3"||output=="projoutL4PHIC"){
      addOutputProjection(trackletproj_L4PHI3_,memory);
      return;
    }

    if (output=="projoutL4PHI4"||output=="projoutL4PHID"){
      addOutputProjection(trackletproj_L4PHI4_,memory);
      return;
    }

    if (output=="projoutL5PHI1"||output=="projoutL5PHIA"){
      addOutputProjection(trackletproj_L5PHI1_,memory);
      return;
    }

    if (output=="projoutL5PHI2"||output=="projoutL5PHIB"){
      addOutputProjection(trackletproj_L5PHI2_,memory);
      return;
    }

    if (output=="projoutL5PHI3"||output=="projoutL5PHIC"){
      addOutputProjection(trackletproj_L5PHI3_,memory);
      return;
    }

    if (output=="projoutL5PHI4"||output=="projoutL5PHID"){
      addOutputProjection(trackletproj_L5PHI4_,memory);
      return;
    }

    if (output=="projoutL6PHI1"||output=="projoutL6PHIA"){
      addOutputProjection(trackletproj_L6PHI1_,memory);
      return;
    }

    if (output=="projoutL6PHI2"||output=="projoutL6PHIB"){
      addOutputProjection(trackletproj_L6PHI2_,memory);
      return;
    }

    if (output=="projoutL6PHI3"||output=="projoutL6PHIC"){
      addOutputProjection(trackletproj_L6PHI3_,memory);
      return;
    }

    if (output=="projoutL6PHI4"||output=="projoutL6PHID"){
      addOutputProjection(trackletproj_L6PHI4_,memory);
      return;
    }

    if (output=="projoutD1PHI1"||output=="projoutD1PHIA"){
      addOutputProjection(trackletproj_D1PHI1_,memory);
      return;
    }

    if (output=="projoutD1PHI2"||output=="projoutD1PHIB"){
      addOutputProjection(trackletproj_D1PHI2_,memory);
      return;
    }

    if (output=="projoutD1PHI3"||output=="projoutD1PHIC"){
      addOutputProjection(trackletproj_D1PHI3_,memory);
      return;
    }

    if (output=="projoutD1PHI4"||output=="projoutD1PHID"){
      addOutputProjection(trackletproj_D1PHI4_,memory);
      return;
    }

    if (output=="projoutD2PHI1"||output=="projoutD2PHIA"){
      addOutputProjection(trackletproj_D2PHI1_,memory);
      return;
    }

    if (output=="projoutD2PHI2"||output=="projoutD2PHIB"){
      addOutputProjection(trackletproj_D2PHI2_,memory);
      return;
    }

    if (output=="projoutD2PHI3"||output=="projoutD2PHIC"){
      addOutputProjection(trackletproj_D2PHI3_,memory);
      return;
    }

    if (output=="projoutD2PHI4"||output=="projoutD2PHID"){
      addOutputProjection(trackletproj_D2PHI4_,memory);
      return;
    }



    if (output=="projoutD3PHI1"||output=="projoutD3PHIA"){
      addOutputProjection(trackletproj_D3PHI1_,memory);
      return;
    }

    if (output=="projoutD3PHI2"||output=="projoutD3PHIB"){
      addOutputProjection(trackletproj_D3PHI2_,memory);
      return;
    }

    if (output=="projoutD3PHI3"||output=="projoutD3PHIC"){
      addOutputProjection(trackletproj_D3PHI3_,memory);
      return;
    }
    
    if (output=="projoutD3PHI4"||output=="projoutD3PHID"){
      addOutputProjection(trackletproj_D3PHI4_,memory);
      return;
    }


    if (output=="projoutD4PHI1"||output=="projoutD4PHIA"){
      addOutputProjection(trackletproj_D4PHI1_,memory);
      return;
    }

    if (output=="projoutD4PHI2"||output=="projoutD4PHIB"){
      addOutputProjection(trackletproj_D4PHI2_,memory);
      return;
    }

    if (output=="projoutD4PHI3"||output=="projoutD4PHIC"){
      addOutputProjection(trackletproj_D4PHI3_,memory);
      return;
    }

    if (output=="projoutD4PHI4"||output=="projoutD4PHID"){
      addOutputProjection(trackletproj_D4PHI4_,memory);
      return;
    }
    


    if (output=="projoutD5PHI1"||output=="projoutD5PHIA"){
      addOutputProjection(trackletproj_D5PHI1_,memory);
      return;
    }

    if (output=="projoutD5PHI2"||output=="projoutD5PHIB"){
      addOutputProjection(trackletproj_D5PHI2_,memory);
      return;
    }

    if (output=="projoutD5PHI3"||output=="projoutD5PHIC"){
      addOutputProjection(trackletproj_D5PHI3_,memory);
      return;
    }

    if (output=="projoutD5PHI4"||output=="projoutD5PHID"){
      addOutputProjection(trackletproj_D5PHI4_,memory);
      return;
    }


    
    if (output=="projoutL1ToMinus"){
      addOutputProjection(trackletproj_L1Minus_,memory);
      return;
    }

    if (output=="projoutL1ToPlus"){
      addOutputProjection(trackletproj_L1Plus_,memory);
      return;
    }

    if (output=="projoutL2ToMinus"){
      addOutputProjection(trackletproj_L2Minus_,memory);
      return;
    }

    if (output=="projoutL2ToPlus"){
      addOutputProjection(trackletproj_L2Plus_,memory);
      return;
    }

    if (output=="projoutL3ToMinus"){
      addOutputProjection(trackletproj_L3Minus_,memory);
      return;
    }

    if (output=="projoutL3ToPlus"){
      addOutputProjection(trackletproj_L3Plus_,memory);
      return;
    }

    if (output=="projoutL4ToMinus"){
      addOutputProjection(trackletproj_L4Minus_,memory);
      return;
    }

    if (output=="projoutL4ToPlus"){
      addOutputProjection(trackletproj_L4Plus_,memory);
      return;
    }

    if (output=="projoutL5ToMinus"){
      addOutputProjection(trackletproj_L5Minus_,memory);
      return;
    }

    if (output=="projoutL5ToPlus"){
      addOutputProjection(trackletproj_L5Plus_,memory);
      return;
    }

    if (output=="projoutL6ToMinus"){
      addOutputProjection(trackletproj_L6Minus_,memory);
      return;
    }

    if (output=="projoutL6ToPlus"){
      addOutputProjection(trackletproj_L6Plus_,memory);
      return;
    }

    if (output=="projoutL3D4ToMinus"){
      addOutputProjection(trackletproj_L3Minus_,memory);
      return;
    }

    if (output=="projoutL3D4ToPlus"){
      addOutputProjection(trackletproj_L3Plus_,memory);
      return;
    }

    if (output=="projoutL4D3ToMinus"){
      addOutputProjection(trackletproj_L4Minus_,memory);
      return;
    }

    if (output=="projoutL4D3ToPlus"){
      addOutputProjection(trackletproj_L4Plus_,memory);
      return;
    }

    if (output=="projoutL5D2ToMinus"){
      addOutputProjection(trackletproj_L5Minus_,memory);
      return;
    }

    if (output=="projoutL5D2ToPlus"){
      addOutputProjection(trackletproj_L5Plus_,memory);
      return;
    }

    if (output=="projoutL6D1ToMinus"){
      addOutputProjection(trackletproj_L6Minus_,memory);
      return;
    }

    if (output=="projoutL6D1ToPlus"){
      addOutputProjection(trackletproj_L6Plus_,memory);
      return;
    }


    if (output=="projoutD1ToPlus"){
      addOutputProjection(trackletproj_D1Plus_,memory);
      return;
    }

    if (output=="projoutD2ToPlus"){
      addOutputProjection(trackletproj_D2Plus_,memory);
      return;
    }

    if (output=="projoutD3ToPlus"){
      addOutputProjection(trackletproj_D3Plus_,memory);
      return;
    }

    if (output=="projoutD4ToPlus"){
      addOutputProjection(trackletproj_D4Plus_,memory);
      return;
    }

    if (output=="projoutD5ToPlus"){
      addOutputProjection(trackletproj_D5Plus_,memory);
      return;
    }    
    

    if (output=="projoutD1ToMinus"){
      addOutputProjection(trackletproj_D1Minus_,memory);
      return;
    }

    if (output=="projoutD2ToMinus"){
      addOutputProjection(trackletproj_D2Minus_,memory);
      return;
    }

    if (output=="projoutD3ToMinus"){
      addOutputProjection(trackletproj_D3Minus_,memory);
      return;
    }

    if (output=="projoutD4ToMinus"){
      addOutputProjection(trackletproj_D4Minus_,memory);
      return;
    }

    if (output=="projoutD5ToMinus"){
      addOutputProjection(trackletproj_D5Minus_,memory);
      return;
    }    
    

    cout << "Could not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="thirdallstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      innerallstubs_.push_back(tmp);
      return;
    }
    if (input=="firstallstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      middleallstubs_.push_back(tmp);
      return;
    }
    if (input=="secondallstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      outerallstubs_.push_back(tmp);
      return;
    }
    if (input.find("stubtriplet")==0){
      StubTripletsMemory* tmp=dynamic_cast<StubTripletsMemory*>(memory);
      assert(tmp!=0);
      stubtriplets_.push_back(tmp);
      return;
    }
    assert(0);
  }



  void execute() {

    unsigned int countall=0;
    unsigned int countsel=0;

    //cout << "TrackletCalculatorDisplaced execute "<<getName()<<" "<<stubtriplets_.size()<<endl;
    
    for(unsigned int l=0;l<stubtriplets_.size();l++){
      if (trackletpars_->nTracklets()>=maxtracklet_) {
	cout << "Will break on too many tracklets in "<<getName()<<endl;
	break;
      }
      for(unsigned int i=0;i<stubtriplets_[l]->nStubTriplets();i++){

	//if(stubtriplets_.size()>0)
	//  cout << "TrackletCalculatorDisplaced execute "<<getName()<<" "<<stubtriplets_[l]->getName()<<" "<<stubtriplets_[l]->nStubTriplets()<<" "<<layer_<<endl;
	
	countall++;

	L1TStub* innerStub=stubtriplets_[l]->getL1TStub1(i);
	Stub* innerFPGAStub=stubtriplets_[l]->getFPGAStub1(i);

	L1TStub* middleStub=stubtriplets_[l]->getL1TStub2(i);
	Stub* middleFPGAStub=stubtriplets_[l]->getFPGAStub2(i);

	L1TStub* outerStub=stubtriplets_[l]->getL1TStub3(i);
	Stub* outerFPGAStub=stubtriplets_[l]->getFPGAStub3(i);

	if (debug1) {
	  cout << "TrackletCalculatorDisplaced execute "<<getName()<<"["<<iSector_<<"]"<<endl;
	}
	
        if (innerFPGAStub->isBarrel()&&middleFPGAStub->isBarrel()&&outerFPGAStub->isBarrel()){
	    //barrel+barrel seeding	  
	    bool accept = LLLSeeding(innerFPGAStub,innerStub,middleFPGAStub,middleStub,outerFPGAStub,outerStub);
	    
	    if (accept) countsel++;
        }
        else if (innerFPGAStub->isDisk()&&middleFPGAStub->isDisk()&&outerFPGAStub->isDisk()){
            assert(0);
        }
        else{
	    //layer+disk seeding
	    
            if (innerFPGAStub->isBarrel() && middleFPGAStub->isDisk() && outerFPGAStub->isDisk()){ //D1D2L2
                bool accept = DDLSeeding(innerFPGAStub,innerStub,middleFPGAStub,middleStub,outerFPGAStub,outerStub);

                if (accept) countsel++;
            }
            else if (innerFPGAStub->isDisk() && middleFPGAStub->isBarrel() && outerFPGAStub->isBarrel()){ //L2L3D1
                bool accept = LLDSeeding(innerFPGAStub,innerStub,middleFPGAStub,middleStub,outerFPGAStub,outerStub);

                if (accept) countsel++;
            }
            else{
                assert(0);
            }
        }

	if (trackletpars_->nTracklets()>=maxtracklet_) {
	  cout << "Will break on number of tracklets in "<<getName()<<endl;
	  break;
	}
	
	if (countall>=MAXTC) {
	  if (debug1) cout << "Will break on MAXTC 1"<<endl;
	  break;
	}
	if (debug1) {
	  cout << "TrackletCalculatorDisplaced execute done"<<endl;
	}

      }
      if (countall>=MAXTC) {
	if (debug1) cout << "Will break on MAXTC 2"<<endl;
	break;
      }
    }

    if (writeTrackletCalculatorDisplaced) {
      static ofstream out("trackletcalculatordisplaced.txt");
      out << getName()<<" "<<countall<<" "<<countsel<<endl;
    }


  }


  void addDiskProj(Tracklet* tracklet, int disk){

    
    FPGAWord fpgar=tracklet->fpgarprojdisk(disk);

    if (fpgar.value()*krprojshiftdisk<12.0) return;
    if (fpgar.value()*krprojshiftdisk>112.0) return;

    
    FPGAWord fpgaphi=tracklet->fpgaphiprojdisk(disk);
    
    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);

    int iphi=iphivmRaw/(32/nallstubsdisks[abs(disk)-1]);
      
    
    if (abs(disk)==1) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_D1PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_D1PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_D1PHI3_,tracklet);
      if (iphi==3) addProjectionDisk(disk,iphi,trackletproj_D1PHI4_,tracklet);
    }
    
    if (abs(disk)==2) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_D2PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_D2PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_D2PHI3_,tracklet);
      if (iphi==3) addProjectionDisk(disk,iphi,trackletproj_D2PHI4_,tracklet);
    }

    if (abs(disk)==3) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_D3PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_D3PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_D3PHI3_,tracklet);
      if (iphi==3) addProjectionDisk(disk,iphi,trackletproj_D3PHI4_,tracklet);
    }

    if (abs(disk)==4) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_D4PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_D4PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_D4PHI3_,tracklet);
      if (iphi==3) addProjectionDisk(disk,iphi,trackletproj_D4PHI4_,tracklet);
    }

    if (abs(disk)==5) {
      if (iphi==0) addProjectionDisk(disk,iphi,trackletproj_D5PHI1_,tracklet);
      if (iphi==1) addProjectionDisk(disk,iphi,trackletproj_D5PHI2_,tracklet);
      if (iphi==2) addProjectionDisk(disk,iphi,trackletproj_D5PHI3_,tracklet);
      if (iphi==3) addProjectionDisk(disk,iphi,trackletproj_D5PHI4_,tracklet);
    }

    
  }


  bool addLayerProj(Tracklet* tracklet, int layer){

    
    assert(layer>0);

    FPGAWord fpgaz=tracklet->fpgazproj(layer);
    FPGAWord fpgaphi=tracklet->fpgaphiproj(layer);


    if(fpgaphi.atExtreme()) cout<<"at extreme! "<<fpgaphi.value()<<"\n";

    assert(!fpgaphi.atExtreme());
    
    if (fpgaz.atExtreme()) return false;

    if (fabs(fpgaz.value()*kz)>zlength) return false;
    
    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);

    int iphi=iphivmRaw/(32/nallstubslayers[layer-1]);
      
    //cout << "layer fpgaphi iphivmRaw iphi : "<<layer<<" "<<fpgaphi.value()<<" "<<iphivmRaw<<" "<<iphi<<endl;

    

    if (layer==1) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L1PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L1PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L1PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L1PHI4_,tracklet);
      if (iphi==4) addProjection(layer,iphi,trackletproj_L1PHI5_,tracklet);
      if (iphi==5) addProjection(layer,iphi,trackletproj_L1PHI6_,tracklet);
      if (iphi==6) addProjection(layer,iphi,trackletproj_L1PHI7_,tracklet);
      if (iphi==7) addProjection(layer,iphi,trackletproj_L1PHI8_,tracklet);
    }
    
    if (layer==2) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L2PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L2PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L2PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L2PHI4_,tracklet);
    }

    if (layer==3) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L3PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L3PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L3PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L3PHI4_,tracklet);
    }

    if (layer==4) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L4PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L4PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L4PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L4PHI4_,tracklet);
    }

    if (layer==5) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L5PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L5PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L5PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L5PHI4_,tracklet);
    }

    if (layer==6) {
      if (iphi==0) addProjection(layer,iphi,trackletproj_L6PHI1_,tracklet);
      if (iphi==1) addProjection(layer,iphi,trackletproj_L6PHI2_,tracklet);
      if (iphi==2) addProjection(layer,iphi,trackletproj_L6PHI3_,tracklet);
      if (iphi==3) addProjection(layer,iphi,trackletproj_L6PHI4_,tracklet);
    }

    return true;

  }

  void addProjection(int layer,int iphi,TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet){
    if (trackletprojs==0) {
      if (warnNoMem) {
	cout << "No projection memory exists in "<<getName()<<" for layer = "<<layer<<" iphi = "<<iphi+1<<endl;
      }
      return;
    }
    assert(trackletprojs!=0);
    trackletprojs->addProj(tracklet);
  }

  void addProjectionDisk(int disk,int iphi,TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet){
    if (trackletprojs==0) {
      if (layer_==3&&abs(disk)==3) return; //L3L4 projections to D3 are not used.
      if (warnNoMem) {       
	cout << "No projection memory exists in "<<getName()<<" for disk = "<<abs(disk)<<" iphi = "<<iphi+1<<endl;
      }
      return;
    }
    assert(trackletprojs!=0);
    trackletprojs->addProj(tracklet);
  }



  bool LLLSeeding(Stub* innerFPGAStub, L1TStub* innerStub, Stub* middleFPGAStub, L1TStub* middleStub, Stub* outerFPGAStub, L1TStub* outerStub){
	  
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" "<<layer_<<" trying stub triplet in layer (L L L): "
	   <<innerFPGAStub->layer().value()<<" "<<middleFPGAStub->layer().value()<<" "<<outerFPGAStub->layer().value()<<endl;
    }
	    
    assert(outerFPGAStub->isBarrel());
    
    //assert(layer_==innerFPGAStub->layer().value()+1);
    
    //assert(layer_==1||layer_==3||layer_==5);

    	  
    double r1=innerStub->r();
    double z1=innerStub->z();
    double phi1=innerStub->phi();
    
    double r2=middleStub->r();
    double z2=middleStub->z();
    double phi2=middleStub->phi();

    double r3=outerStub->r();
    double z3=outerStub->z();
    double phi3=outerStub->phi();

    int take3 = 0;
    if(layer_ == 5) take3 = 1;
    
    double rinv,phi0,d0,t,z0;

    LayerProjection layerprojs[4];
    DiskProjection diskprojs[5];    
    
    double phiproj[4],zproj[4],phider[4],zder[4];
    double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
    
    exacttracklet(r1,z1,phi1,r2,z2,phi2,r3,z3,phi3,
		  take3,
		  rinv,phi0,d0,t,z0,
		  phiproj,zproj,
		  phiprojdisk,rprojdisk,
		  phider,zder,
		  phiderdisk,rderdisk
		  );

    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();

      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }
    
    double rinvapprox,phi0approx,d0approx,tapprox,z0approx;
    double phiprojapprox[4],zprojapprox[4],phiderapprox[4],zderapprox[4];
    double phiprojdiskapprox[5],rprojdiskapprox[5];
    double phiderdiskapprox[5],rderdiskapprox[5];

    //FIXME: do the actual integer calculation
    
    phi0 -= 0.171;

    //store the approcximate results
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx   = d0;
    tapprox    = t;
    z0approx   = z0;
    
    for(unsigned int i=0; i<toR_.size(); ++i){
      phiproj[i] -= 0.171;
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i]   = zproj[i];
      phiderapprox[i]  = phider[i];
      zderapprox[i]    = zder[i];
    }

    for(unsigned int i=0; i<toZ_.size(); ++i){
      phiprojdisk[i] -= 0.171;
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i]   = rprojdisk[i];
      phiderdiskapprox[i]  = phiderdisk[i];
      rderdiskapprox[i]    = rderdisk[i];
    }

    //now binary
    double krinv = kphi1/kr*pow(2,rinv_shift),
           kphi0 = kphi1*pow(2,phi0_shift),
           kt = kz/kr*pow(2,t_shift),
           kz0 = kz*pow(2,z0_shift),
           kphiproj = kphi1*pow(2,SS_phiL_shift),
           kphider = kphi1/kr*pow(2,SS_phiderL_shift),
           kzproj = kz*pow(2,PS_zL_shift),
           kzder = kz/kr*pow(2,PS_zderL_shift),
           kphiprojdisk = kphi1*pow(2,SS_phiD_shift),
           kphiderdisk = kphi1/kr*pow(2,SS_phiderD_shift),
           krprojdisk = kr*pow(2,PS_rD_shift),
           krderdisk = kr/kz*pow(2,PS_rderD_shift);
    
    int irinv,iphi0,id0,it,iz0;
    int iphiproj[4],izproj[4],iphider[4],izder[4];
    int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
      
    //store the binary results
    irinv = rinvapprox / krinv;
    iphi0 = phi0approx / kphi0;
    id0 = d0approx / kd0;
    it    = tapprox / kt;
    iz0   = z0approx / kz0;

    bool success = true;
    if(fabs(rinvapprox)>rinvcut){
      if (debug1) 
	cout << "TrackletCalculator::LLL Seeding irinv too large: "
	     <<rinvapprox<<"("<<irinv<<")\n";
      success = false;
    }
    if (fabs(z0approx)>1.8*z0cut) { 
      if (debug1) cout << "Failed tracklet z0 cut "<<z0approx<<" in layer "<<layer_<<endl;
      success = false;
    }
    if (fabs(d0approx)>maxd0) {
      if (debug1) cout << "Failed tracklet d0 cut "<<d0approx<<endl;
      success = false;
    }
    
    if (!success) return false;

    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletCalculatorDisplaced::LLLSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }
    
    
    for(unsigned int i=0; i<toR_.size(); ++i){
      iphiproj[i] = phiprojapprox[i] / kphiproj;
      izproj[i]   = zprojapprox[i] / kzproj;

      iphider[i] = phiderapprox[i] / kphider;
      izder[i]   = zderapprox[i] / kzder;

      //check that z projection is in range
      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;

      //check that phi projection is in range
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;

      //adjust number of bits for phi and z projection
      if (rproj_[i]<60.0) {	
	//iphider[i]>>=3;  //Check me - added by aryd as in L123 iphider was out of range
	iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
	if (iphiproj[i]>=(1<<nbitsphistubL123)-1) iphiproj[i]=(1<<nbitsphistubL123)-2; //-2 not to hit atExtreme
      }
      else {
	izproj[i]>>=(nbitszprojL123-nbitszprojL456);
      }


      if (rproj_[i]<60.0) {
        if (iphider[i]<-(1<<(nbitsphiprojderL123-1))) {
	  iphider[i] = -(1<<(nbitsphiprojderL123-1));
	}
        if (iphider[i]>=(1<<(nbitsphiprojderL123-1))) {
	  iphider[i] = (1<<(nbitsphiprojderL123-1))-1;
	}
      }
      else {
        if (iphider[i]<-(1<<(nbitsphiprojderL456-1))) {
	  iphider[i] = -(1<<(nbitsphiprojderL456-1));
	}
        if (iphider[i]>=(1<<(nbitsphiprojderL456-1))) {
	  iphider[i] = (1<<(nbitsphiprojderL456-1))-1;
	}
      }

      layerprojs[i].init(lproj_[i],rproj_[i],
			 iphiproj[i],izproj[i],
			 iphider[i],izder[i],
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 phiderapprox[i],zderapprox[i]);
      
    }

    if(fabs(it * kt)>1.0) {
      for(unsigned int i=0; i<toZ_.size(); ++i){

        iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
        irprojdisk[i]   = rprojdiskapprox[i] / krprojdisk;

	iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
	irderdisk[i]   = rderdiskapprox[i] / krderdisk;

	//check phi projection in range
	if (iphiprojdisk[i]<=0) continue;
	if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;
	
	//check r projection in range
	if(rprojdiskapprox[i]< 20. || rprojdiskapprox[i] > 120.) continue;

	diskprojs[i].init(i+1,rproj_[i],
			  iphiprojdisk[i],irprojdisk[i],
			  iphiderdisk[i],irderdisk[i],
			  phiprojdisk[i],rprojdisk[i],
			  phiderdisk[i],rderdisk[i],
			  phiprojdiskapprox[i],rprojdiskapprox[i],
			  phiderdisk[i],rderdisk[i]);
      }
    }

    
    if (writeTrackletPars) {
      static ofstream out("trackletpars.txt");
      out <<"Trackpars "<<layer_
	  <<"   "<<rinv<<" "<<rinvapprox<<" "<<rinvapprox
	  <<"   "<<phi0<<" "<<phi0approx<<" "<<phi0approx
	  <<"   "<<t<<" "<<tapprox<<" "<<tapprox
	  <<"   "<<z0<<" "<<z0approx<<" "<<z0approx
	  <<endl;
    }	        
        
    Tracklet* tracklet=new Tracklet(innerStub,middleStub,outerStub,
				    innerFPGAStub,middleFPGAStub,outerFPGAStub,
				    rinv,phi0,d0,z0,t,
				    rinvapprox,phi0approx,d0approx,
				    z0approx,tapprox,
				    irinv,iphi0,id0,iz0,it,
				    layerprojs,
				    diskprojs,
				    false);
    
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" Found LLL tracklet in sector = "
	   <<iSector_<<" phi0 = "<<phi0<<endl;
    }
        

    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);
    
    bool addL5=false;
    bool addL6=false;
    for(unsigned int j=0;j<toR_.size();j++){
      bool added=false;
      if(debug1)
	cout<<"adding layer projection "<<j<<"/"<<toR_.size()<<" "<<lproj_[j]<<"\n";
      if (tracklet->validProj(lproj_[j])) {
	added=addLayerProj(tracklet,lproj_[j]);
	if (added&&lproj_[j]==5) addL5=true;
	if (added&&lproj_[j]==6) addL6=true;
      }
    }
    
    
    for(unsigned int j=0;j<toZ_.size();j++){ 
      int disk=dproj_[j];
      if (disk == 0) continue;
      if (disk==2&&addL5) continue;
      if (disk==1&&addL6) continue;
      if (it<0) disk=-disk;
      if(debug1)
	cout<<"adding disk projection "<<j<<"/"<<toZ_.size()<<" "<<disk<<"\n";
      if (tracklet->validProjDisk(abs(disk))) {
	addDiskProj(tracklet,disk);
      }
    }
    
    return true;

  }
    
  bool DDLSeeding(Stub* innerFPGAStub, L1TStub* innerStub, Stub* middleFPGAStub, L1TStub* middleStub, Stub* outerFPGAStub, L1TStub* outerStub){
	  
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" "<<layer_<<" trying stub triplet in  (L2 D1 D2): "
	   <<innerFPGAStub->layer().value()<<" "<<middleFPGAStub->disk().value()<<" "<<outerFPGAStub->disk().value()<<endl;
    }
	    
    int take3 = 1; //D1D2L2
    	  
    double r1=innerStub->r();
    double z1=innerStub->z();
    double phi1=innerStub->phi();
    
    double r2=middleStub->r();
    double z2=middleStub->z();
    double phi2=middleStub->phi();

    double r3=outerStub->r();
    double z3=outerStub->z();
    double phi3=outerStub->phi();
    
    double rinv,phi0,d0,t,z0;
    
    double phiproj[4],zproj[4],phider[4],zder[4];
    double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
    
    exacttracklet(r1,z1,phi1,r2,z2,phi2,r3,z3,phi3,
		  take3,
		  rinv,phi0,d0,t,z0,
		  phiproj,zproj,
		  phiprojdisk,rprojdisk,
		  phider,zder,
		  phiderdisk,rderdisk
		  );

    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();

      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }
    
    double rinvapprox,phi0approx,d0approx,tapprox,z0approx;
    double phiprojapprox[4],zprojapprox[4],phiderapprox[4],zderapprox[4];
    double phiprojdiskapprox[5],rprojdiskapprox[5];
    double phiderdiskapprox[5],rderdiskapprox[5];

    //FIXME: do the actual integer calculation
    
    phi0 -= 0.171;

    //store the approcximate results
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx   = d0;
    tapprox    = t;
    z0approx   = z0;
    
    for(unsigned int i=0; i<toR_.size(); ++i){
      phiproj[i] -= 0.171;
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i]   = zproj[i];
      phiderapprox[i]  = phider[i];
      zderapprox[i]    = zder[i];
    }

    for(unsigned int i=0; i<toZ_.size(); ++i){
      phiprojdisk[i] -= 0.171;
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i]   = rprojdisk[i];
      phiderdiskapprox[i]  = phiderdisk[i];
      rderdiskapprox[i]    = rderdisk[i];
    }

    //now binary
    double krinv = kphi1/kr*pow(2,rinv_shift),
           kphi0 = kphi1*pow(2,phi0_shift),
           kt = kz/kr*pow(2,t_shift),
           kz0 = kz*pow(2,z0_shift),
           kphiproj = kphi1*pow(2,SS_phiL_shift),
           kphider = kphi1/kr*pow(2,SS_phiderL_shift),
           kzproj = kz*pow(2,PS_zL_shift),
           kzder = kz/kr*pow(2,PS_zderL_shift),
           kphiprojdisk = kphi1*pow(2,SS_phiD_shift),
           kphiderdisk = kphi1/kr*pow(2,SS_phiderD_shift),
           krprojdisk = kr*pow(2,PS_rD_shift),
           krderdisk = kr/kz*pow(2,PS_rderD_shift);
    
    int irinv,iphi0,id0,it,iz0;
    int iphiproj[4],izproj[4],iphider[4],izder[4];
    int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
      
    //store the binary results
    irinv = rinvapprox / krinv;
    iphi0 = phi0approx / kphi0;
    id0 = d0approx / kd0;
    it    = tapprox / kt;
    iz0   = z0approx / kz0;

    bool success = true;
    if(fabs(rinvapprox)>rinvcut){
      if (debug1) 
	cout << "TrackletCalculator::DDL Seeding irinv too large: "
	     <<rinvapprox<<"("<<irinv<<")\n";
      success = false;
    }
    if (fabs(z0approx)>1.8*z0cut) {
      if (debug1) cout << "Failed tracklet z0 cut "<<z0approx<<endl;
      success = false;
    }
    if (fabs(d0approx)>maxd0) {
      if (debug1) cout << "Failed tracklet d0 cut "<<d0approx<<endl;
      success = false;
    }
    
    if (!success) return false;

    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletCalculatorDisplaced::DDLSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }

    LayerProjection layerprojs[4];
    DiskProjection diskprojs[5];
    
    for(unsigned int i=0; i<toR_.size(); ++i){
      iphiproj[i] = phiprojapprox[i] / kphiproj;
      izproj[i]   = zprojapprox[i] / kzproj;

      iphider[i] = phiderapprox[i] / kphider;
      izder[i]   = zderapprox[i] / kzder;

      //check that z projection in range
      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;
      
      //check that phi projection in range
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;
      
      if (rproj_[i]<60.0) {
	iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
      }
      else {
	izproj[i]>>=(nbitszprojL123-nbitszprojL456);
      }

      
      if (rproj_[i]<60.0) {
        if (iphider[i]<-(1<<(nbitsphiprojderL123-1))) iphider[i] = -(1<<(nbitsphiprojderL123-1));
        if (iphider[i]>=(1<<(nbitsphiprojderL123-1))) iphider[i] = (1<<(nbitsphiprojderL123-1))-1;
      }
      else {
        if (iphider[i]<-(1<<(nbitsphiprojderL456-1))) iphider[i] = -(1<<(nbitsphiprojderL456-1));
        if (iphider[i]>=(1<<(nbitsphiprojderL456-1))) iphider[i] = (1<<(nbitsphiprojderL456-1))-1;
      }

      layerprojs[i].init(lproj_[i],rproj_[i],
			 iphiproj[i],izproj[i],
			 iphider[i],izder[i],
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 phiderapprox[i],zderapprox[i]);

      
      
    }

    if(fabs(it * kt)>1.0) {
      for(unsigned int i=0; i<toZ_.size(); ++i){

        iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
        irprojdisk[i]   = rprojdiskapprox[i] / krprojdisk;

	iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
	irderdisk[i]   = rderdiskapprox[i] / krderdisk;
      
	if (iphiprojdisk[i]<=0) continue;	
	if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;
      
	if(irprojdisk[i]< 20. / krprojdisk ||
	   irprojdisk[i] > 120. / krprojdisk ) continue;

	diskprojs[i].init(i+1,rproj_[i],
			  iphiprojdisk[i],irprojdisk[i],
			  iphiderdisk[i],irderdisk[i],
			  phiprojdisk[i],rprojdisk[i],
			  phiderdisk[i],rderdisk[i],
			  phiprojdiskapprox[i],rprojdiskapprox[i],
			  phiderdisk[i],rderdisk[i]);

	
      }
    }

    
    if (writeTrackletPars) {
      static ofstream out("trackletpars.txt");
      out <<"Trackpars "<<layer_
	  <<"   "<<rinv<<" "<<rinvapprox<<" "<<rinvapprox
	  <<"   "<<phi0<<" "<<phi0approx<<" "<<phi0approx
	  <<"   "<<t<<" "<<tapprox<<" "<<tapprox
	  <<"   "<<z0<<" "<<z0approx<<" "<<z0approx
	  <<endl;
    }	        
        
    Tracklet* tracklet=new Tracklet(innerStub,middleStub,outerStub,
				    innerFPGAStub,middleFPGAStub,outerFPGAStub,
				    rinv,phi0,d0,z0,t,
				    rinvapprox,phi0approx,d0approx,
				    z0approx,tapprox,
				    irinv,iphi0,id0,iz0,it,
				    layerprojs,
				    diskprojs,
				    false);
    
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" Found DDL tracklet in sector = "
	   <<iSector_<<" phi0 = "<<phi0<<endl;
    }
        

    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);
    
    for(unsigned int j=0;j<toR_.size();j++){
      if(debug1)
	cout<<"adding layer projection "<<j<<"/"<<toR_.size()<<" "<<lproj_[j]<<" "<<tracklet->validProj(lproj_[j])<<"\n";      
      if (tracklet->validProj(lproj_[j])) {
	addLayerProj(tracklet,lproj_[j]);
      }
    }
    
    
    for(unsigned int j=0;j<toZ_.size();j++){ 
      int disk=dproj_[j];
      if(disk == 0) continue;
      if (it<0) disk=-disk;
      if(debug1)
	cout<<"adding disk projection "<<j<<"/"<<toZ_.size()<<" "<<disk<<" "<<tracklet->validProjDisk(abs(disk))<<"\n";
      if (tracklet->validProjDisk(abs(disk))) {
	addDiskProj(tracklet,disk);
      }
    }
    
    return true;

  }
    
  bool LLDSeeding(Stub* innerFPGAStub, L1TStub* innerStub, Stub* middleFPGAStub, L1TStub* middleStub, Stub* outerFPGAStub, L1TStub* outerStub){
	  
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" "<<layer_<<" trying stub triplet in  (L2L3D1): "
	   <<middleFPGAStub->layer().value()<<" "<<outerFPGAStub->layer().value()<<" "<<innerFPGAStub->disk().value()<<endl;
    }
	    
    int take3 = 0; //L2L3D1
    	  
    double r3=innerStub->r();
    double z3=innerStub->z();
    double phi3=innerStub->phi();
    
    double r1=middleStub->r();
    double z1=middleStub->z();
    double phi1=middleStub->phi();

    double r2=outerStub->r();
    double z2=outerStub->z();
    double phi2=outerStub->phi();

    
    double rinv,phi0,d0,t,z0;
    
    double phiproj[4],zproj[4],phider[4],zder[4];
    double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
    
    exacttracklet(r1,z1,phi1,r2,z2,phi2,r3,z3,phi3,
		  take3,
		  rinv,phi0,d0,t,z0,
		  phiproj,zproj,
		  phiprojdisk,rprojdisk,
		  phider,zder,
		  phiderdisk,rderdisk
		  );

    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();

      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }
    
    double rinvapprox,phi0approx,d0approx,tapprox,z0approx;
    double phiprojapprox[4],zprojapprox[4],phiderapprox[4],zderapprox[4];
    double phiprojdiskapprox[5],rprojdiskapprox[5];
    double phiderdiskapprox[5],rderdiskapprox[5];

    //FIXME: do the actual integer calculation
    
    phi0 -= 0.171;

    //store the approcximate results
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx   = d0;
    tapprox    = t;
    z0approx   = z0;
    
    for(unsigned int i=0; i<toR_.size(); ++i){
      phiproj[i] -= 0.171;
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i]   = zproj[i];
      phiderapprox[i]  = phider[i];
      zderapprox[i]    = zder[i];
    }

    for(unsigned int i=0; i<toZ_.size(); ++i){
      phiprojdisk[i] -= 0.171;
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i]   = rprojdisk[i];
      phiderdiskapprox[i]  = phiderdisk[i];
      rderdiskapprox[i]    = rderdisk[i];
    }

    //now binary
    double krinv = kphi1/kr*pow(2,rinv_shift),
           kphi0 = kphi1*pow(2,phi0_shift),
           kt = kz/kr*pow(2,t_shift),
           kz0 = kz*pow(2,z0_shift),
           kphiproj = kphi1*pow(2,SS_phiL_shift),
           kphider = kphi1/kr*pow(2,SS_phiderL_shift),
           kzproj = kz*pow(2,PS_zL_shift),
           kzder = kz/kr*pow(2,PS_zderL_shift),
           kphiprojdisk = kphi1*pow(2,SS_phiD_shift),
           kphiderdisk = kphi1/kr*pow(2,SS_phiderD_shift),
           krprojdisk = kr*pow(2,PS_rD_shift),
           krderdisk = kr/kz*pow(2,PS_rderD_shift);
    
    int irinv,iphi0,id0,it,iz0;
    int iphiproj[4],izproj[4],iphider[4],izder[4];
    int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
      
    //store the binary results
    irinv = rinvapprox / krinv;
    iphi0 = phi0approx / kphi0;
    id0 = d0approx / kd0;
    it    = tapprox / kt;
    iz0   = z0approx / kz0;

    bool success = true;
    if(fabs(rinvapprox)>rinvcut){
      if (debug1) 
	cout << "TrackletCalculator:: LLD Seeding irinv too large: "
	     <<rinvapprox<<"("<<irinv<<")\n";
      success = false;
    }
    if (fabs(z0approx)>1.8*z0cut) {
      if (debug1) cout << "Failed tracklet z0 cut "<<z0approx<<endl;
      success = false;
    }
    if (fabs(d0approx)>maxd0) {
      if (debug1) cout << "Failed tracklet d0 cut "<<d0approx<<endl;
      success = false;
    }
    
    if (!success) return false;

    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletCalculatorDisplaced::LLDSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }

    
    
    LayerProjection layerprojs[4];
    DiskProjection diskprojs[5];

    
    for(unsigned int i=0; i<toR_.size(); ++i){
      iphiproj[i] = phiprojapprox[i] / kphiproj;
      izproj[i]   = zprojapprox[i] / kzproj;

      iphider[i] = phiderapprox[i] / kphider;
      izder[i]   = zderapprox[i] / kzder;

      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;
      
      //this is left from the original....
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;
      
      if (rproj_[i]<60.0) {
	iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
      }
      else {
	izproj[i]>>=(nbitszprojL123-nbitszprojL456);
      }


      if (rproj_[i]<60.0) {
        if (iphider[i]<-(1<<(nbitsphiprojderL123-1))) iphider[i] = -(1<<(nbitsphiprojderL123-1));
        if (iphider[i]>=(1<<(nbitsphiprojderL123-1))) iphider[i] = (1<<(nbitsphiprojderL123-1))-1;
      }
      else {
        if (iphider[i]<-(1<<(nbitsphiprojderL456-1))) iphider[i] = -(1<<(nbitsphiprojderL456-1));
        if (iphider[i]>=(1<<(nbitsphiprojderL456-1))) iphider[i] = (1<<(nbitsphiprojderL456-1))-1;
      }

      layerprojs[i].init(lproj_[i],rproj_[i],
			 iphiproj[i],izproj[i],
			 iphider[i],izder[i],
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 phiderapprox[i],zderapprox[i]);
      
    }

    if(fabs(it * kt)>1.0) {
      for(unsigned int i=0; i<toZ_.size(); ++i){

        iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
        irprojdisk[i]   = rprojdiskapprox[i] / krprojdisk;

	iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
	irderdisk[i]   = rderdiskapprox[i] / krderdisk;

	//Check phi range of projection
	if (iphiprojdisk[i]<=0) continue;
	if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;
      
	//Check r range of projection
	if(irprojdisk[i]< 20. / krprojdisk ||
	   irprojdisk[i] > 120. / krprojdisk ) continue;

	diskprojs[i].init(i+1,rproj_[i],
			  iphiprojdisk[i],irprojdisk[i],
			  iphiderdisk[i],irderdisk[i],
			  phiprojdisk[i],rprojdisk[i],
			  phiderdisk[i],rderdisk[i],
			  phiprojdiskapprox[i],rprojdiskapprox[i],
			  phiderdisk[i],rderdisk[i]);

	
      }
    }

    
    if (writeTrackletPars) {
      static ofstream out("trackletpars.txt");
      out <<"Trackpars "<<layer_
	  <<"   "<<rinv<<" "<<rinvapprox<<" "<<rinvapprox
	  <<"   "<<phi0<<" "<<phi0approx<<" "<<phi0approx
	  <<"   "<<t<<" "<<tapprox<<" "<<tapprox
	  <<"   "<<z0<<" "<<z0approx<<" "<<z0approx
	  <<endl;
    }	        
        
    Tracklet* tracklet=new Tracklet(innerStub,middleStub,outerStub,
				    innerFPGAStub,middleFPGAStub,outerFPGAStub,
				    rinv,phi0,d0,z0,t,
				    rinvapprox,phi0approx,d0approx,
				    z0approx,tapprox,
				    irinv,iphi0,id0,iz0,it,
				    layerprojs,
				    diskprojs,
				    false);
    
    if (debug1) {
      cout << "TrackletCalculatorDisplaced "<<getName()<<" Found LLD tracklet in sector = "
	   <<iSector_<<" phi0 = "<<phi0<<endl;
    }
        

    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);
    
    for(unsigned int j=0;j<toR_.size();j++){
      if(debug1)
	cout<<"adding layer projection "<<j<<"/"<<toR_.size()<<" "<<lproj_[j]<<"\n";
      if (tracklet->validProj(lproj_[j])) {
	addLayerProj(tracklet,lproj_[j]);
      }
    }
    
    
    for(unsigned int j=0;j<toZ_.size();j++){ 
      int disk=dproj_[j];
      if(disk == 0) continue;
      if (it<0) disk=-disk;
      if(debug1)
	cout<<"adding disk projection "<<j<<"/"<<toZ_.size()<<" "<<disk<<"\n";      
      if (tracklet->validProjDisk(abs(disk))) {
	addDiskProj(tracklet,disk);
      }
    }
    
    return true;

  }
    
   
  
  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }

void exactproj(double rproj,double rinv, double phi0, double d0,
	       double t, double z0, double r0,
	       double &phiproj, double &zproj,
	       double &phider, double &zder) 
{  
  double rho = 1/rinv;
  if(rho<0) r0 = -r0;
  phiproj=phi0-asin((rproj*rproj+r0*r0-rho*rho)/(2*rproj*r0));
  double beta = acos((rho*rho+r0*r0-rproj*rproj)/(2*r0*rho));
  zproj=z0+ t*fabs(rho*beta);
  
  //not exact, but close
  phider=-0.5*rinv/sqrt(1-pow(0.5*rproj*rinv,2))-d0/(rproj*rproj);
  zder=t/sqrt(1-pow(0.5*rproj*rinv,2));

  if(debug1) cout <<"exact proj layer at "<<rproj<<" : "<< phiproj <<" "<<zproj<<"\n";
}

  
  void exactprojdisk(double zproj, double rinv, double, double,  //phi0 and d0 are not used.
		   double t, double z0,
		   double x0, double y0,
		   double &phiproj, double &rproj,
		   double &phider, double &rder)
{

  if(t<0) zproj = -zproj;
  double rho = fabs(1/rinv);
  double beta = (zproj-z0)/(t*rho);
  double phiV = atan2(-y0,-x0);
  double c = rinv>0? -1 : 1;

  double x = x0 + rho*cos(phiV+c*beta);
  double y = y0 + rho*sin(phiV+c*beta);

  phiproj = atan2(y,x);
  phiproj += -phimin_+(phimax_-phimin_)/6.0;
  if(phiproj >  2) phiproj -= 8*atan(1.);
  if(phiproj < -2) phiproj += 8*atan(1.);
  rproj   = sqrt(x*x+y*y);

  phider = c / t / (x*x+y*y) * (rho + x0*cos(phiV+c*beta) + y0*sin(phiV+c*beta));
  rder   = c / t / rproj * ( y0*cos(phiV+c*beta) - x0*sin(phiV+c*beta));

  if(debug1) cout <<"exact proj disk at"<<zproj<<" : "<< phiproj <<" "<<rproj<<"\n";

}

  
void exacttracklet(double r1, double z1, double phi1,
		   double r2, double z2, double phi2,
		   double r3, double z3, double phi3,
		   int take3,
		   double& rinv, double& phi0, double &d0,
		   double& t, double& z0,
		   double phiproj[5], double zproj[5], 
		   double phiprojdisk[5], double rprojdisk[5],
		   double phider[5], double zder[5],
		   double phiderdisk[5], double rderdisk[5]
		   )
{    
    //two lines perpendicular to the 1->2 and 2->3
    double x1 = r1*cos(phi1);
    double x2 = r2*cos(phi2);
    double x3 = r3*cos(phi3);
    
    double y1 = r1*sin(phi1);
    double y2 = r2*sin(phi2);
    double y3 = r3*sin(phi3);
    
    double k1 = - (x2-x1)/(y2-y1);
    double k2 = - (x3-x2)/(y3-y2);
    double b1 = 0.5*(y2+y1)-0.5*(x1+x2)*k1;
    double b2 = 0.5*(y3+y2)-0.5*(x2+x3)*k2;
    //their intersection gives the center of the circle
    double y0 = (b1*k2-b2*k1)/(k2-k1);
    double x0 = (b1-b2)/(k2-k1);
    //get the radius three ways:
    double R1 = sqrt(pow(x1-x0,2)+pow(y1-y0,2));
    double R2 = sqrt(pow(x2-x0,2)+pow(y2-y0,2));
    double R3 = sqrt(pow(x3-x0,2)+pow(y3-y0,2));
    //check if the same
    double eps1 = fabs(R1/R2-1);
    double eps2 = fabs(R3/R2-1);
    if(eps1>1e-10 || eps2>1e-10)
      cout<<"&&&&&&&&&&&& bad circle! "<<R1<<"\t"<<R2<<"\t"<<R3<<"\n";

    //results
    rinv = 1./R1;
    phi0 = atan(1.)*2 + atan2(y0,x0);

    phi0 += -phimin_+(phimax_-phimin_)/6.0; 
    d0 = -R1 + sqrt(x0*x0+y0*y0);
    //sign of rinv:
    double dphi = phi3 - atan2(y0,x0);
    if(dphi> 3.1415927) dphi -= 6.283185;
    if(dphi<-3.1415927) dphi += 6.283185;
    if(dphi<0) {
      rinv = -rinv;
      d0 = -d0;
      phi0 = phi0 + 4*atan(1);
    }
    if(phi0 >8*atan(1)) phi0 = phi0-8*atan(1);
    if(phi0 <0)         phi0 = phi0+8*atan(1);
      
    //now in RZ:
    //turning angle
    double beta1 = atan2(y1-y0,x1-x0)-atan2(-y0,-x0);
    double beta2 = atan2(y2-y0,x2-x0)-atan2(-y0,-x0);
    double beta3 = atan2(y3-y0,x3-x0)-atan2(-y0,-x0);

    if(beta1> 3.1415927) beta1 -= 6.283185;
    if(beta1<-3.1415927) beta1 += 6.283185;
    if(beta2> 3.1415927) beta2 -= 6.283185;
    if(beta2<-3.1415927) beta2 += 6.283185;
    if(beta3> 3.1415927) beta3 -= 6.283185;
    if(beta3<-3.1415927) beta3 += 6.283185;

    double t12 = (z2-z1)/fabs(beta2-beta1)/R1;
    double z12 = (z1*beta2-z2*beta1)/(beta2-beta1);
    double t13 = (z3-z1)/fabs(beta3-beta1)/R1;
    double z13 = (z1*beta3-z3*beta1)/(beta3-beta1);

    // cout<<"::::: "<<sigmaz<<" "<<beta1<<"\t"<<beta2<<"\t"<<beta3<<"\n";
    // cout<<"::::: "<<t12<<"\t"<<t13<<"\n";
    // cout<<"::::: "<<z12<<"\t"<<z13<<"\n";
    
    if(take3>0){
      //take 13 (large lever arm)
      t = t13;
      z0 = z13;
    }
    else{
      //take 12 (pixel layers)
      t = t12;
      z0 = z12;
    }
    
    if(debug1)
      cout<<"exact tracklet: "<<rinv<<" "<<phi0<<" "<<d0<<" "<<t<<" "<<z0<<"\n";

    for (unsigned int i=0;i<toR_.size();i++) {
      exactproj(toR_[i],rinv,phi0,d0,t,z0,sqrt(x0*x0+y0*y0),
		phiproj[i],zproj[i],phider[i],zder[i]);
    }    

    for (unsigned int i=0;i<toZ_.size();i++) {
      exactprojdisk(toZ_[i],rinv,phi0,d0,t,z0, x0, y0,
		phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
    }
}


  
    
private:

  int TCIndex_;
  int layer_;
  int disk_;
  double phimin_;
  double phimax_;
  double rproj_[4];
  int lproj_[4];
  double zproj_[3];
  int dproj_[3];

  vector<double> toR_;
  vector<double> toZ_;

  unsigned int maxtracklet_; //maximum numbor of tracklets that be stored
  
  vector<AllStubsMemory*> innerallstubs_;
  vector<AllStubsMemory*> middleallstubs_;
  vector<AllStubsMemory*> outerallstubs_;
  vector<StubTripletsMemory*> stubtriplets_;

  TrackletParametersMemory* trackletpars_;

  TrackletProjectionsMemory* trackletproj_L1PHI1_;
  TrackletProjectionsMemory* trackletproj_L1PHI2_;
  TrackletProjectionsMemory* trackletproj_L1PHI3_;
  TrackletProjectionsMemory* trackletproj_L1PHI4_;
  TrackletProjectionsMemory* trackletproj_L1PHI5_;
  TrackletProjectionsMemory* trackletproj_L1PHI6_;
  TrackletProjectionsMemory* trackletproj_L1PHI7_;
  TrackletProjectionsMemory* trackletproj_L1PHI8_;

  TrackletProjectionsMemory* trackletproj_L2PHI1_;
  TrackletProjectionsMemory* trackletproj_L2PHI2_;
  TrackletProjectionsMemory* trackletproj_L2PHI3_;
  TrackletProjectionsMemory* trackletproj_L2PHI4_;

  TrackletProjectionsMemory* trackletproj_L3PHI1_;
  TrackletProjectionsMemory* trackletproj_L3PHI2_;
  TrackletProjectionsMemory* trackletproj_L3PHI3_;
  TrackletProjectionsMemory* trackletproj_L3PHI4_;

  TrackletProjectionsMemory* trackletproj_L4PHI1_;
  TrackletProjectionsMemory* trackletproj_L4PHI2_;
  TrackletProjectionsMemory* trackletproj_L4PHI3_;
  TrackletProjectionsMemory* trackletproj_L4PHI4_;

  TrackletProjectionsMemory* trackletproj_L5PHI1_;
  TrackletProjectionsMemory* trackletproj_L5PHI2_;
  TrackletProjectionsMemory* trackletproj_L5PHI3_;
  TrackletProjectionsMemory* trackletproj_L5PHI4_;

  TrackletProjectionsMemory* trackletproj_L6PHI1_;
  TrackletProjectionsMemory* trackletproj_L6PHI2_;
  TrackletProjectionsMemory* trackletproj_L6PHI3_;
  TrackletProjectionsMemory* trackletproj_L6PHI4_;

  TrackletProjectionsMemory* trackletproj_D1PHI1_;
  TrackletProjectionsMemory* trackletproj_D1PHI2_;
  TrackletProjectionsMemory* trackletproj_D1PHI3_;
  TrackletProjectionsMemory* trackletproj_D1PHI4_;

  TrackletProjectionsMemory* trackletproj_D2PHI1_;
  TrackletProjectionsMemory* trackletproj_D2PHI2_;
  TrackletProjectionsMemory* trackletproj_D2PHI3_;
  TrackletProjectionsMemory* trackletproj_D2PHI4_;

  TrackletProjectionsMemory* trackletproj_D3PHI1_;
  TrackletProjectionsMemory* trackletproj_D3PHI2_;
  TrackletProjectionsMemory* trackletproj_D3PHI3_;
  TrackletProjectionsMemory* trackletproj_D3PHI4_;

  TrackletProjectionsMemory* trackletproj_D4PHI1_;
  TrackletProjectionsMemory* trackletproj_D4PHI2_;
  TrackletProjectionsMemory* trackletproj_D4PHI3_;
  TrackletProjectionsMemory* trackletproj_D4PHI4_;

  TrackletProjectionsMemory* trackletproj_D5PHI1_;
  TrackletProjectionsMemory* trackletproj_D5PHI2_;
  TrackletProjectionsMemory* trackletproj_D5PHI3_;
  TrackletProjectionsMemory* trackletproj_D5PHI4_;


  
  TrackletProjectionsMemory* trackletproj_L1Plus_; 
  TrackletProjectionsMemory* trackletproj_L1Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_L2Plus_; 
  TrackletProjectionsMemory* trackletproj_L2Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_L3Plus_; 
  TrackletProjectionsMemory* trackletproj_L3Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_L4Plus_; 
  TrackletProjectionsMemory* trackletproj_L4Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_L5Plus_; 
  TrackletProjectionsMemory* trackletproj_L5Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_L6Plus_; 
  TrackletProjectionsMemory* trackletproj_L6Minus_;


  TrackletProjectionsMemory* trackletproj_D1Plus_; 
  TrackletProjectionsMemory* trackletproj_D1Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_D2Plus_; 
  TrackletProjectionsMemory* trackletproj_D2Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_D3Plus_; 
  TrackletProjectionsMemory* trackletproj_D3Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_D4Plus_; 
  TrackletProjectionsMemory* trackletproj_D4Minus_;
			                         
  TrackletProjectionsMemory* trackletproj_D5Plus_; 
  TrackletProjectionsMemory* trackletproj_D5Minus_;
};

#endif
