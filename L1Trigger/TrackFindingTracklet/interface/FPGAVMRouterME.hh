//This class implementes the VM router
#ifndef FPGAVMROUTERME_H
#define FPGAVMROUTERME_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAVMRouterME:public FPGAProcessBase{

public:

  FPGAVMRouterME(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="allstubout"||
	output=="allstuboutn1"||
	output=="allstuboutn2"||
	output=="allstuboutn3"||
	output=="allstuboutn4"||
	output=="allstuboutn5"||
	output=="allstuboutn6"
	){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      allstubs_.push_back(tmp);
      return;
    }

    bool print=getName()=="VMRME_L3PHI2";
    print=false;
    
    if (print) cout << "FPGAVMRouterME in "<<getName()<<endl;
    
    for (int i=0;i<12;i++) {
      std::ostringstream oss;
      oss<<(i+1);
      string s=oss.str();

      if (print) {
	cout << "FPGAVMRouterME strings to match "<<output<<" "<<"vmstuboutPHI"+s+"n1"<<endl;
      }
	
      if (output=="vmstuboutPHI"+s+"n1"||
	  output=="vmstuboutPHI"+s+"n2"||
	  output=="vmstuboutPHI"+s+"n3"
	  ){
	FPGAVMStubsME* tmp=dynamic_cast<FPGAVMStubsME*>(memory);
	assert(tmp!=0);
	if (print){
	  cout << "FPGAVMRouter found match "<<getName() << " " << i << endl;
	}
	vmstubsPHI_[i].push_back(tmp);
	return;
      }
    }

    cout << "Could not find : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="stubin"){
      FPGAInputLink* tmp1=dynamic_cast<FPGAInputLink*>(memory);
      assert(tmp1!=0);
      if (tmp1!=0){
	stubinputs_.push_back(tmp1);
      }
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute(){

    //only one of these can be filled!
    assert(stubinputsdisk_.size()*stubinputs_.size()==0);

    unsigned int count=0;

    if (stubinputs_.size()!=0){
      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  count++;
	  if (count>MAXVMROUTER) continue;
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  int iphiRaw=stub.first->iphivmRaw();

	  int iphiRawPlus=stub.first->iphivmRawPlus();
	  int iphiRawMinus=stub.first->iphivmRawMinus();

	  //cout << "FPGAVMRouterME : layer iphivmRaw "<<getName()<<" "<<stub.first->layer().value()
	  //     <<" "<<iphiRaw<<endl;
	  
	  bool insert=false;
	  
	  iphiRaw-=4;
	  iphiRaw=iphiRaw>>1;
	  assert(iphiRaw>=0);
	  assert(iphiRaw<=11);
	  
	  iphiRawPlus-=4;
	  iphiRawPlus=iphiRawPlus>>1;
	  if (iphiRawPlus<0) iphiRawPlus=0;
	  if (iphiRawPlus>11) iphiRawPlus=11;
	  
	  iphiRawMinus-=4;
	  iphiRawMinus=iphiRawMinus>>1;
	  if (iphiRawMinus<0) iphiRawMinus=0;
	  if (iphiRawMinus>11) iphiRawMinus=11;

	  //iphiRaw=(iphiRaw&3); because use 0-11 entries

	  //cout << "FPGAVMRouterME "<<getName()<<" iphiRaw size : "<<iphiRaw<<" "<<vmstubsPHI_[iphiRaw].size()<<endl;
	  for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
	    if (debug1) {
	      cout << "FPGAVMRouterME "<<getName()<<" add stub ( r = "<<stub.second->r()<<" ) in : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
	    }
	    vmstubsPHI_[iphiRaw][l]->addStub(stub);
	    insert=true;
	  }


	  
	  if (iphiRaw!=iphiRawPlus) {
	    for (unsigned int l=0;l<vmstubsPHI_[iphiRawPlus].size();l++){
	      //cout << "Add extra plus" << endl;
	      vmstubsPHI_[iphiRawPlus][l]->addStub(stub);
	    }
	  }
	  if (iphiRaw!=iphiRawMinus) {
	    for (unsigned int l=0;l<vmstubsPHI_[iphiRawMinus].size();l++){
	      //cout << "Add extra minus" << endl;
	      vmstubsPHI_[iphiRawMinus][l]->addStub(stub);
	    }
	  }
	 
	  
	  
	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());

	  for (unsigned int l=0;l<allstubs_.size();l++){
	    allstubs_[l]->addStub(stub);
	  }


	  assert(insert);

	}
      }
      
      if (writeVMOccupancyME) {
	static ofstream out("vmoccupancyme.txt");

	for (int i=0;i<24;i++) {
	  if (vmstubsPHI_[i].size()!=0) {
	    out<<vmstubsPHI_[i][0]->getName()<<" "<<vmstubsPHI_[i][0]->nStubs()<<endl;
	  }
	}

      }

    }
    if (stubinputsdisk_.size()>0) {
      assert(0); //should never get here
      //cout << "Routing stubs in disk" <<endl;
      for(unsigned int j=0;j<stubinputsdisk_.size();j++){
	for(unsigned int i=0;i<stubinputsdisk_[j]->nStubs();i++){
	  //cout << "Found stub in disk in "<<getName()<<endl;
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputsdisk_[j]->getStub(i);
	  //
          bool isSS = name_[7] == '6' || name_[7] == '8';
          if(isSS && stub.second->r() <60) {
            cout<<"in SS router "<<name_<<" but r = "<<stub.second->r()<<"\n";
            assert(0);
          }
          if(!isSS && stub.second->r() >60) {
            cout<<"in PS router "<<name_<<" but r = "<<stub.second->r()<<"\n";
            assert(0);
          }
	  //FIXME Next few lines should be member data in FPGAStub...
	  int irtmp=stub.first->r().value(); 
	  int ir;
	  if(stub.second->r()>60){//SS module, r encoded differently
	    ir=2+(irtmp>>3);
	    //cout<<"That's SS module! "<<stub.second->r()<<": "<<irtmp<<" with "<<stub.first->r().nbits()<<"\n";
	  }
	  else {
	    ir=irtmp>>(stub.first->r().nbits()-2);
	  }
	  int iphitmp=stub.first->phi().value();
	  int disk=stub.first->disk().value();
	  if ((disk%2)==0) iphitmp-=(1<<(stub.first->phi().nbits()-3));  
	  assert(iphitmp>=0);
	  int iphi=iphitmp>>(stub.first->phi().nbits()-2);

	  //cout << "iphi ir irtmp: "<<iphi<<" "<<ir<<"\t"<<irtmp<<"\t ("<<stub.second->r()<<")"<<endl;
	  assert(ir>=0);
	  assert(ir<=3);
	  ir=ir%2;
	  assert(iphi>=0);
	  assert(iphi<=3);

	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());
	  
	  for (unsigned int l=0;l<allstubs_.size();l++){
	    allstubs_[l]->addStub(stub);
	  }

	}
      }
    }


    if (writeAllStubs) {
      static ofstream out("allstubsme.txt");
      out<<allstubs_[0]->getName()<<" "<<allstubs_[0]->nStubs()<<endl;
    }
 


  }



private:

  vector<FPGAInputLink*> stubinputs_;
  vector<FPGAStubDisk*> stubinputsdisk_;
  vector<FPGAAllStubs*> allstubs_;

  vector<FPGAVMStubsME*> vmstubsPHI_[24];

  /*
  vector<FPGAVMStubsME*> vmstubsPHI1R1_;
  vector<FPGAVMStubsME*> vmstubsPHI1R2_;
  vector<FPGAVMStubsME*> vmstubsPHI2R1_;
  vector<FPGAVMStubsME*> vmstubsPHI2R2_;
  vector<FPGAVMStubsME*> vmstubsPHI3R1_;
  vector<FPGAVMStubsME*> vmstubsPHI3R2_;
  vector<FPGAVMStubsME*> vmstubsPHI4R1_;
  vector<FPGAVMStubsME*> vmstubsPHI4R2_;
  */

};

#endif

