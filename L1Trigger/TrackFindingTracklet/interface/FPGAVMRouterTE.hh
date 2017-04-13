//This class implementes the VM router
#ifndef FPGAVMROUTERTE_H
#define FPGAVMROUTERTE_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAVMRouterTE:public FPGAProcessBase{

public:

  FPGAVMRouterTE(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){

    layer_=0;
    disk_=0;
    
    if (name_[6]=='L') layer_=name_[7]-'0';    
    if (name_[6]=='D') disk_=name_[7]-'0';    
    if (name_[6]=='F') disk_=name_[7]-'0';    
    if (name_[6]=='B') disk_=-(name_[7]-'0');

    overlap_=false;
    
    overlap_=(name_[11]=='X'||name_[11]=='Y'||name_[11]=='W'||name_[11]=='Q');

    assert((layer_!=0)||(disk_!=0));

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
    for (int i=0;i<24;i++) {
      std::ostringstream oss;
      oss<<(i+1);
      string s=oss.str();
      if (output=="vmstuboutA"+s+"n1"||
	  output=="vmstuboutB"+s+"n1"||
	  output=="vmstuboutC"+s+"n1"||
	  output=="vmstuboutD"+s+"n1"||
	  output=="vmstuboutE"+s+"n1"||
	  output=="vmstuboutF"+s+"n1"||
	  output=="vmstuboutG"+s+"n1"||
	  output=="vmstuboutH"+s+"n1"||
	  output=="vmstuboutX"+s+"n1"||
	  output=="vmstuboutY"+s+"n1"||
	  output=="vmstuboutW"+s+"n1"||
	  output=="vmstuboutQ"+s+"n1"||
	  output=="vmstuboutA"+s+"n2"||
	  output=="vmstuboutB"+s+"n2"||
	  output=="vmstuboutC"+s+"n2"||
	  output=="vmstuboutD"+s+"n2"||
	  output=="vmstuboutE"+s+"n2"||
	  output=="vmstuboutF"+s+"n2"||
	  output=="vmstuboutG"+s+"n2"||
	  output=="vmstuboutH"+s+"n2"||
	  output=="vmstuboutX"+s+"n2"||
	  output=="vmstuboutY"+s+"n2"||
	  output=="vmstuboutW"+s+"n2"||
	  output=="vmstuboutQ"+s+"n2"||
	  output=="vmstuboutA"+s+"n3"||
	  output=="vmstuboutB"+s+"n3"||
	  output=="vmstuboutC"+s+"n3"||
	  output=="vmstuboutD"+s+"n3"||
	  output=="vmstuboutE"+s+"n3"||
	  output=="vmstuboutF"+s+"n3"||
	  output=="vmstuboutG"+s+"n3"||
	  output=="vmstuboutH"+s+"n3"||
	  output=="vmstuboutX"+s+"n3"||
	  output=="vmstuboutY"+s+"n3"||
	  output=="vmstuboutW"+s+"n3"||
	  output=="vmstuboutQ"+s+"n3"||
	  output=="vmstuboutA"+s+"n4"||
	  output=="vmstuboutB"+s+"n4"||
	  output=="vmstuboutC"+s+"n4"||
	  output=="vmstuboutD"+s+"n4"||
	  output=="vmstuboutE"+s+"n4"||
	  output=="vmstuboutF"+s+"n4"||
	  output=="vmstuboutG"+s+"n4"||
	  output=="vmstuboutH"+s+"n4"||
	  output=="vmstuboutA"+s+"n5"||
	  output=="vmstuboutB"+s+"n5"||
	  output=="vmstuboutC"+s+"n5"||
	  output=="vmstuboutD"+s+"n5"||
	  output=="vmstuboutE"+s+"n5"||
	  output=="vmstuboutF"+s+"n5"||
	  output=="vmstuboutG"+s+"n5"||
	  output=="vmstuboutH"+s+"n5"||
	  output=="vmstuboutA"+s+"n6"||
	  output=="vmstuboutB"+s+"n6"||
	  output=="vmstuboutC"+s+"n6"||
	  output=="vmstuboutD"+s+"n6"||
	  output=="vmstuboutE"+s+"n6"||
	  output=="vmstuboutF"+s+"n6"||
	  output=="vmstuboutG"+s+"n6"||
	  output=="vmstuboutH"+s+"n6"
	  ){
	FPGAVMStubsTE* tmp=dynamic_cast<FPGAVMStubsTE*>(memory);
	assert(tmp!=0);
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

    //if (debug1) {
    //  cout << "FPGAVMRouter::execute "<<getName()<<endl;
    //}
      
    assert(stubinputs_.size()!=0);

    unsigned int count=0;
 
    if (layer_!=0){
      for(unsigned int j=0;j<stubinputs_.size();j++){
	//cout << "FPGAVMRouterTE::execute : "<<stubinputs_[j]->getName() << " " << stubinputs_[j]->nStubs() << endl;
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  count++;
	  if (count>MAXVMROUTER) continue;
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  int iphiRaw=stub.first->iphivmRaw();

	  bool insert=false;

	  if (getName()=="VMRTE_L2PHIW"||getName()=="VMRTE_L2PHIQ") {
	    //special case where even even layer is treated as an odd (inner layer)
	    iphiRaw-=4;
	    assert(iphiRaw>=0);
	    assert(iphiRaw<24);
	    //cout << "iphiRaw = "<<iphiRaw<<" "<<getName()<<" "<<overlap_<<endl;
	    if (overlap_) iphiRaw>>=2;
	    //cout << "iphiRaw = "<<iphiRaw<<" "<<getName()<<" "<<overlap_<<endl;
	    //cout << "getName "<<getName()<<" "<<vmstubsPHI_[iphiRaw].size()<<endl;
	    for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
	      if (debug1) {
		cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      vmstubsPHI_[iphiRaw][l]->addStub(stub);
	      insert=true;
	    }
	  }  else {	    
	    if (stub.first->layer().value()%2==0) {
	      //cout << "Odd layer : "<<iphiRaw<<endl;
	      //odd layers here
	      iphiRaw-=4;
	      assert(iphiRaw>=0);
	      assert(iphiRaw<24);
	      //cout << "iphiRaw = "<<iphiRaw<<" "<<getName()<<" "<<overlap_<<endl;
	      if (overlap_) iphiRaw>>=2;
	      //cout << "iphiRaw = "<<iphiRaw<<" "<<getName()<<" "<<overlap_<<endl;
	      //cout << "getName "<<getName()<<" "<<vmstubsPHI_[iphiRaw].size()<<endl;
	      for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		//cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
		vmstubsPHI_[iphiRaw][l]->addStub(stub);
		insert=true;
	      }
	    }
	    else {
	      //cout << "Even layer : "<<iphiRaw<<endl;
	      //even layers here
	      iphiRaw/=2;
	      assert(iphiRaw>=0);
	      assert(iphiRaw<16);
	      for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		//cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
		vmstubsPHI_[iphiRaw][l]->addStub(stub);
		insert=true;
	      }
	    }
	  }
	    

	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());

	  for (unsigned int l=0;l<allstubs_.size();l++){
	    allstubs_[l]->addStub(stub);
	  }

	  //cout << "FPGAVMRouterTE "<<getName()<<" "<<stub.first->iphivmRaw()<<endl;
	  
	  assert(insert);
	}
      }

    }
    if (disk_!=0) {
      //cout << "FPGAVMRouterTE stubs in disk" <<endl;
      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputs_[j]->getStub(i);
	  //cout << "Found stub in disk in "<<getName()<<"  r = "<<stub.second->r()<<endl;
	  //

	  if (stub.second->r()>60.0) continue;
	  
	  int iphiRaw=stub.first->iphivmRaw();

	  bool insert=false;

	  if (getName()=="VMRTE_D1PHIW"||getName()=="VMRTE_D1PHIQ") {
	    //special case where odd disk is treated as outer disk
	    iphiRaw/=2;
	    assert(iphiRaw>=0);
	    assert(iphiRaw<16);
	    iphiRaw>>=1; //only 8 VMS in even disks
	    //cout << "FPGAVMRouterTE "<<getName()<<" 2 stubs in disk iphiRaw "<<iphiRaw <<endl;
	    for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
	      if (debug1) { 
		cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      vmstubsPHI_[iphiRaw][l]->addStub(stub);
	      insert=true;
	    }
	  } else {
	    if (getName()=="VMRTE_D1PHIA"||
		getName()=="VMRTE_D1PHIB"||
		getName()=="VMRTE_D1PHIC"||
		getName()=="VMRTE_D1PHID"||
		getName()=="VMRTE_D2PHIA"||
		getName()=="VMRTE_D2PHIB"||
		getName()=="VMRTE_D2PHIC"||
		getName()=="VMRTE_D2PHID"
		) {
	      if (abs(stub.first->disk().value())%2==1) {
		//odd disks here
		//cout << "odd disk"<<endl;
		if (overlap_) {
		  iphiRaw>>=1; //only 12 VMS in odd disks
		} else {
		  iphiRaw-=4;
		  assert(iphiRaw>=0);
		  assert(iphiRaw<24);
		  iphiRaw>>=1; //only 12 VMS in odd disks
		}
		
		//cout << "FPGAVMRouterTE "<<getName()<<" overlap_ = "<<overlap_<<" stubs in disk iphiRaw "<<iphiRaw<<endl;
		for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		  //cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<" "<<stub.second->r()<<endl;	     
		  vmstubsPHI_[iphiRaw][l]->addStub(stub);
		  insert=true;
		}
	      }
	      else {
		//even disks here
		iphiRaw/=2;
		assert(iphiRaw>=0);
		assert(iphiRaw<16);
		//cout << "FPGAVMRouterTE "<<getName()<<" 2 stubs in disk iphiRaw "<<iphiRaw <<endl;
		for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		  //cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
		  vmstubsPHI_[iphiRaw][l]->addStub(stub);
		  insert=true;
		}
	      }
	    } else {
	      if (abs(stub.first->disk().value())%2==1) {
		//odd disks here
		if (overlap_) {
		  iphiRaw>>=2; //only 6 VMS in odd disks
		} else {
		  iphiRaw-=4;
		  assert(iphiRaw>=0);
		  assert(iphiRaw<24);
		  iphiRaw>>=2; //only 6 VMS in odd disks
		}
		//cout << "FPGAVMRouterTE "<<getName()<<" overlap_ = "<<overlap_<<" stubs in disk iphiRaw "<<iphiRaw<<endl;
		for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		  //cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<" "<<stub.second->r()<<endl;	     
		  vmstubsPHI_[iphiRaw][l]->addStub(stub);
		  insert=true;
		}
	      }
	      else {
		//even disks here
		iphiRaw/=2;
		assert(iphiRaw>=0);
		assert(iphiRaw<16);
		iphiRaw>>=1; //only 8 VMS in even disks
		//cout << "FPGAVMRouterTE "<<getName()<<" 2 stubs in disk iphiRaw "<<iphiRaw <<endl;
		for (unsigned int l=0;l<vmstubsPHI_[iphiRaw].size();l++){
		  //cout << "FPGAVMRouterTE added stub to : "<<vmstubsPHI_[iphiRaw][l]->getName()<<endl;
		  vmstubsPHI_[iphiRaw][l]->addStub(stub);
		  insert=true;
		}
	      }

	    }
	  }


	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());
	  
	  for (unsigned int l=0;l<allstubs_.size();l++){
	    //cout << "FPGAVMRouterTE added stub to : "<<allstubs_[l]->getName()<<" "<<stub.second->r()<<endl;	     
	    allstubs_[l]->addStub(stub);
	  }

	  if (!insert) {
	    cout << getName() << " did not insert stub"<<endl;
	  }
	  assert(insert);

	}
      }
    }


    if (writeAllStubs) {
      static ofstream out("allstubste.txt");
      out<<allstubs_[0]->getName()<<" "<<allstubs_[0]->nStubs()<<endl;
    }


    if (writeVMOccupancyTE) {
      static ofstream out("vmoccupancyte.txt");
      
      for (int i=0;i<24;i++) {
	if (vmstubsPHI_[i].size()!=0) {
	  out<<vmstubsPHI_[i][0]->getName()<<" "<<vmstubsPHI_[i][0]->nStubs()<<endl;
	}
      }
    }

  }



private:

  int layer_;
  int disk_;

  bool overlap_;	     
	     
  vector<FPGAInputLink*> stubinputs_;
  vector<FPGAAllStubs*> allstubs_;

  vector<FPGAVMStubsTE*> vmstubsPHI_[24];


};

#endif

