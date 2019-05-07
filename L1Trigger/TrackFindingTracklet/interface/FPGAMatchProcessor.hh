//This class implementes the match processor
#ifndef FPGAMATCHPROCESSOR_H
#define FPGAMATCHPROCESSOR_H

#include "FPGAProcessBase.hh"
#include "FPGAUtil.hh"

using namespace std;

class FPGAMatchProcessor:public FPGAProcessBase{

public:

  FPGAMatchProcessor(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    
    fullmatchesToPlus_=0;
    fullmatchesToMinus_=0;
    double dphi=2*M_PI/NSector;
    double dphiHG=0.5*dphisectorHG-M_PI/NSector;
    phimin_=iSector_*dphi-dphiHG;
    phimax_=phimin_+dphi+2*dphiHG;
    phimin_=FPGAUtil::phiRange(phimin_);
    phimax_=FPGAUtil::phiRange(phimax_);
    phioffset_=phimin_;
    
    string subname=name.substr(3,2);
    fullmatchesToPlus_=0;
    fullmatchesToMinus_=0;
    layer_=0;
    disk_=0;
    if (subname=="L1") layer_=1;
    if (subname=="L2") layer_=2;
    if (subname=="L3") layer_=3;
    if (subname=="L4") layer_=4;
    if (subname=="L5") layer_=5;
    if (subname=="L6") layer_=6;
    if (subname=="D1") disk_=1;
    if (subname=="D2") disk_=2;
    if (subname=="D3") disk_=3;
    if (subname=="D4") disk_=4;
    if (subname=="D5") disk_=5;
    if (layer_==0 && disk_==0) {
      cout << "name subname "<<name<<" "<<subname<<endl;
      assert(0);
    }
    icorrshift_=5+idrinvbits+phi0bitshift-rinvbitshift-phiderbitshift;
    //icorzshift_=idrinvbits-zderbitshift-tbitshift;
    if (layer_<=3) {
      icorzshift_=-1-PS_zderL_shift;
    } else {
      icorzshift_=-1-SS_zderL_shift;      
    }
    phi0shift_=3;
    fact_=1;
    if (layer_>=4) {
      fact_=(1<<(nbitszprojL123-nbitszprojL456));
      icorrshift_-=(10-nbitsrL456);
      icorzshift_+=(nbitszprojL123-nbitszprojL456+nbitsrL456-nbitsrL123);
      phi0shift_=0;
    }

    //to adjust globaly the phi and rz matching cuts
    phifact_=1.0;
    rzfact_=1.0;

    for(unsigned int seedindex=0;seedindex<7;seedindex++){
      phimatchcut_[seedindex]=-1;
      zmatchcut_[seedindex]=-1;
    }
    
    if (layer_==1){
      phimatchcut_[1]=0.07/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=5.5/kz;
      phimatchcut_[2]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=15.0/kz;
      phimatchcut_[3]=0.07/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=1.5/kz;
      phimatchcut_[4]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[4]=2.0/kz;
      phimatchcut_[6]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[6]=1.5/kz;
    }
    if (layer_==2){
      phimatchcut_[1]=0.06/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=3.5/kz;
      phimatchcut_[2]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=15.0/kz;
      phimatchcut_[3]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=1.25/kz;
    }
    if (layer_==3){
      phimatchcut_[0]=0.1/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=0.7/kz;
      phimatchcut_[2]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=9.0/kz;
    }
    if (layer_==4){
      phimatchcut_[0]=0.19/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=3.0/kz;
      phimatchcut_[2]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=7.0/kz;
    }
    if (layer_==5){
      phimatchcut_[0]=0.4/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=3.0/kz;
      phimatchcut_[1]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=8.0/kz;
    }
    if (layer_==6){
      phimatchcut_[0]=0.5/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=4.0/kz;
      phimatchcut_[1]=0.19/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=9.5/kz;
    }

    for(int iseedindex=0;iseedindex<7;iseedindex++){
      rphicutPS_[iseedindex]=-1.0;
      rphicut2S_[iseedindex]=-1.0;
      rcutPS_[iseedindex]=-1.0;
      rcut2S_[iseedindex]=-1.0;
    }

    if (abs(disk_)==1){
      rphicutPS_[0]=0.2;
      rcutPS_[0]=0.5;
      rphicut2S_[0]=0.5;
      rcut2S_[0]=3.8;      

      rphicut2S_[1]=0.8;
      rcut2S_[1]=3.8;      

      rphicutPS_[4]=0.10;
      rcutPS_[4]=0.5;

    }

    if (abs(disk_)==2){
      rphicutPS_[0]=0.2;
      rcutPS_[0]=0.5;
      rphicut2S_[0]=0.5;
      rcut2S_[0]=3.8;      

      rphicut2S_[1]=0.8;
      rcut2S_[1]=3.8;      

      rphicutPS_[4]=0.10;
      rcutPS_[4]=0.5;

      rphicutPS_[5]=0.10;
      rcutPS_[5]=0.5;

      
      rphicut2S_[5]=0.5;
      rcut2S_[5]=3.8;      

      rphicutPS_[6]=0.10;
      rcutPS_[6]=0.5;
      rphicut2S_[6]=0.15;
      rcut2S_[6]=3.4;      
    }

    if (abs(disk_)==3){
      rphicutPS_[0]=0.25;
      rcutPS_[0]=0.5;
      rphicut2S_[0]=0.5;
      rcut2S_[0]=3.6;      


      rphicutPS_[3]=0.15;
      rcutPS_[3]=0.5;
      rphicut2S_[3]=0.15;
      rcut2S_[3]=3.6;      

      rphicutPS_[5]=0.2;
      rcutPS_[5]=0.6;
      rphicut2S_[5]=0.2;
      rcut2S_[5]=3.6;

      rphicutPS_[6]=0.15;
      rcutPS_[6]=0.8;
      rphicut2S_[6]=0.25;
      rcut2S_[6]=3.8;      
    }


    if (abs(disk_)==4){
      rphicutPS_[0]=0.5;
      rcutPS_[0]=0.5;      
      rphicut2S_[0]=0.5;
      rcut2S_[0]=3.6;      


      rphicutPS_[3]=0.2;
      rcutPS_[3]=0.8;
      rphicut2S_[3]=0.2;
      rcut2S_[3]=3.6;      

      rphicutPS_[5]=0.3;
      rcutPS_[5]=1.0;
      rphicut2S_[5]=0.25;
      rcut2S_[5]=3.5;      

      rphicutPS_[6]=0.5;
      rcutPS_[6]=1.0;      
      rphicut2S_[6]=0.5;
      rcut2S_[6]=3.8;      

    }



    if (abs(disk_)==5){
      rphicutPS_[3]=0.25;
      rcutPS_[3]=1.0;
      rphicut2S_[3]=0.4;
      rcut2S_[3]=3.6;      

      rphicutPS_[4]=0.10;
      rcutPS_[4]=0.5;
      rphicut2S_[4]=0.2;
      rcut2S_[4]=3.4;      

      rphicutPS_[5]=0.5;
      rcutPS_[5]=2.0;
      rphicut2S_[5]=0.4;
      rcut2S_[5]=3.7;      


    }

    
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout1"||
	output=="matchout2"||
	output=="matchout3"||
	output=="matchout4"||
	output=="matchout5"||
	output=="matchout6"||
	output=="matchout7"||
	output=="matchout8"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatches_.push_back(tmp);
      return;
    }
    if (output=="matchoutplus"){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToPlus_==0);
      fullmatchesToPlus_=tmp;
      return;
    }
    if (output=="matchoutminus"){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToMinus_==0);
      fullmatchesToMinus_=tmp;
      return;
    }
    cout << "Could not find output = "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="allstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      allstubs_=tmp;
      return;
    }
    if (input=="vmstubin"){
      FPGAVMStubsME* tmp=dynamic_cast<FPGAVMStubsME*>(memory);
      assert(tmp!=0);
      vmstubs_.push_back(tmp); //to allow more than one stub in?  vmstubs_=tmp;
      return;
    }
    if (input=="proj1in"||
	input=="proj2in"||
	input=="proj3in"||
	input=="proj4in"||
	input=="proj5in"||
	input=="proj6in"||
	input=="proj7in"||
	input=="proj8in"||
	input=="proj9in"||
	input=="proj10in"||
	input=="proj11in"||
	input=="proj12in"||
	input=="proj13in"||
	input=="proj14in"||
	input=="proj15in"||
	input=="proj16in"||
	input=="proj17in"||
	input=="proj18in"||
	input=="proj19in"||
	input=="proj20in"||
	input=="proj21in"
	){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojs_.push_back(tmp);
      return;
    }
    assert(0);
  }

  void execute() {
    
  }

    
private:

  int layer_;
  int disk_;
  int fact_;
  int icorrshift_;
  int icorzshift_;
  int phi0shift_;
  int phimatchcut_[7];
  int zmatchcut_[7];
  double phimin_;
  double phimax_;
  double phioffset_;

  double rphicutPS_[7];
  double rphicut2S_[7];
  double rcutPS_[7];
  double rcut2S_[7];

  double phifact_;
  double rzfact_;
  
  FPGAAllStubs* allstubs_;
  vector<FPGAVMStubsME*> vmstubs_;
  vector<FPGATrackletProjections*> inputprojs_;

  vector<FPGACandidateMatch*> matches_;

  vector<FPGAFullMatch*> fullmatches_;
  FPGAFullMatch* fullmatchesToPlus_;
  FPGAFullMatch* fullmatchesToMinus_;

};

#endif
