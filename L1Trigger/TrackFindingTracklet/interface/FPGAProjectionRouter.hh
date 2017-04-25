//This class implementes the projection router
#ifndef FPGAPROJECTIONROUTER_H
#define FPGAPROJECTIONROUTER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAProjectionRouter:public FPGAProcessBase{

public:

  FPGAProjectionRouter(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    string subname=name.substr(8,2);
    //cout << "name subname : "<<name<<" "<<subname<<endl;
    layer_=0;
    disk_=0;
    vmprojPHI1_=0;
    vmprojPHI2_=0;
    vmprojPHI3_=0;
    vmprojPHI4_=0;
    
    
    if (subname=="L1") layer_=1;
    if (subname=="L2") layer_=2;
    if (subname=="L3") layer_=3;
    if (subname=="L4") layer_=4;
    if (subname=="L5") layer_=5;
    if (subname=="L6") layer_=6;
    if (subname=="F1") disk_=1;
    if (subname=="F2") disk_=2;
    if (subname=="F3") disk_=3;
    if (subname=="F4") disk_=4;
    if (subname=="F5") disk_=5;
    if (subname=="D1") disk_=1;
    if (subname=="D2") disk_=2;
    if (subname=="D3") disk_=3;
    if (subname=="D4") disk_=4;
    if (subname=="D5") disk_=5;
    if (subname=="B1") disk_=-1;
    if (subname=="B2") disk_=-2;
    if (subname=="B3") disk_=-3;
    if (subname=="B4") disk_=-4;
    if (subname=="B5") disk_=-5;
    assert(disk_!=0||layer_!=0);
    allproj_=0;
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="allprojout"){
      FPGAAllProjections* tmp=dynamic_cast<FPGAAllProjections*>(memory);
      assert(tmp!=0);
      allproj_=tmp;
      return;
    }
    if (output=="vmprojoutPHI1"||output=="vmprojoutPHI5"||output=="vmprojoutPHI9"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI2"||output=="vmprojoutPHI6"||output=="vmprojoutPHI10"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI3"||output=="vmprojoutPHI7"||output=="vmprojoutPHI11"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI3_=tmp;
      return;
    }
    if (output=="vmprojoutPHI4"||output=="vmprojoutPHI8"||output=="vmprojoutPHI12"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI4_=tmp;
      return;
    }

    cout << "Did not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="proj1in"||input=="proj2in"||
	input=="proj3in"||input=="proj4in"||
	input=="proj5in"||input=="proj6in"||
	input=="proj7in"||input=="proj8in"||
	input=="proj9in"||input=="proj10in"||
	input=="proj11in"||input=="proj12in"||
	input=="proj13in"||input=="proj14in"||
	input=="proj15in"||input=="proj16in"||
	input=="proj17in"||input=="proj18in"||
	input=="proj19in"||input=="proj20in"||
	input=="proj21in"||input=="proj22in"||
	input=="proj23in"||input=="proj24in"||
	input=="proj25in"||input=="proj26in"||
	input=="proj27in"||input=="proj28in"||
	input=="proj29in"||input=="proj30in"||
	input=="proj31in"||input=="proj32in"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputproj_.push_back(tmp);
      return;
    }
    if (input=="projplusin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputplusproj_=tmp;
      return;
    }
    if (input=="projminusin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputminusproj_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<" in "<<getName()<<endl;
    assert(0);
  }

  void execute() {

    //cout << "FPGAProjectionRouter::execute : "<<getName()<<" "<<inputproj_.size()<<endl;

    unsigned int count=0;

    if (layer_!=0) {
      for (unsigned int j=0;j<inputproj_.size();j++){
	//cout << "FPGAPRojectionRouter Inputproj : "<<inputproj_[j]->getName()<<" "
	//    <<inputproj_[j]->nTracklets()<<endl;
	for (unsigned int i=0;i<inputproj_[j]->nTracklets();i++){
	  //cout << "FPGAPRojectionRouter i : "<<i<<" "<<layer_<<endl;
	  count++;
	  if (count>MAXPROJROUTER) continue;
	  //cout << "Doing projection"<<endl;
	  
	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiproj(layer_);
	  FPGAWord fpgaz=inputproj_[j]->getFPGATracklet(i)->fpgazproj(layer_);

	  //cout << "FPGAPRojectionRouter got phi and z "<<endl;
	  
	  //if (inputproj_[j]->getFPGATracklet(i)->plusNeighbor(layer_)) {
	  //  cout << "Found plus neighbor in : "<<inputproj_[j]->getName()<<endl;
	  //} 
	  
	  //skip if projection is out of range!
	  if (fpgaz.atExtreme()) continue;
	  if (fpgaphi.atExtreme()) continue;
	  
	  int iphitmp=fpgaphi.value();
	  int iphi=iphitmp>>(fpgaphi.nbits()-5);
	  //cout << "iphitmp iphi "<<iphitmp<<" "<<iphi<<endl;
	  assert(iphi>=4);
	  assert(iphi<=27);
	  iphi-=4;
	  iphi=(iphi>>1);
	  iphi=iphi&3;
	  assert(iphi>=0);
	  assert(iphi<=3);
	  
	  assert(allproj_!=0);

	  unsigned int index=allproj_->nTracklets();
	  if (!(inputproj_[j]->getFPGATracklet(i)->minusNeighbor(layer_)||
		inputproj_[j]->getFPGATracklet(i)->plusNeighbor(layer_))){
	    //cout << "layer minusNeighbor plusNeighbor homeSector iSector :"
	    //	 <<layer_<<" "
	    //	 <<inputproj_[j]->getFPGATracklet(i)->minusNeighbor(layer_)<<" "
	    //	 <<inputproj_[j]->getFPGATracklet(i)->plusNeighbor(layer_)<<" "
	    // 	 <<inputproj_[j]->getFPGATracklet(i)->homeSector()<<" "
	    // 	 <<iSector_<<" "
	    //	 <<inputproj_[j]->getName()
	    //	 <<endl;
	    //assert(inputproj_[j]->getFPGATracklet(i)->homeSector()==iSector_);
	  }
	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));

	  //cout << "index iphi : "<<index<<" "<<iphi<<endl;
	  
	  if (iphi==0) {
	    assert(vmprojPHI1_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI1_->getName()<<endl;
	    }
	    vmprojPHI1_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==1) {
	    assert(vmprojPHI2_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI2_->getName()<<endl;
	    }
	    vmprojPHI2_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==2) {
	    assert(vmprojPHI3_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI3_->getName()<<endl;
	    }
	    vmprojPHI3_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==3) {
	    assert(vmprojPHI4_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI4_->getName()<<endl;
	    }
	    vmprojPHI4_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }
	  
	}
      }
    } else {
      for (unsigned int j=0;j<inputproj_.size();j++){
	for (unsigned int i=0;i<inputproj_[j]-> nTracklets();i++){
	  count++;
	  if (count>MAXPROJROUTER) continue;

	  int disk=disk_;
	  if (inputproj_[j]->getFPGATracklet(i)->t()<0.0) disk=-disk_;

	  //cout << "Here001"<<endl;
	  
	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiprojdisk(disk);
	  FPGAWord fpgar=inputproj_[j]->getFPGATracklet(i)->fpgarprojdisk(disk);

	  //cout << "Here002"<<endl;

	  //skip if projection is out of range!
	  if (fpgar.atExtreme()) continue;
	  if (fpgaphi.atExtreme()) continue;
	  int iphitmp=fpgaphi.value();
	  int iphi=iphitmp>>(fpgaphi.nbits()-5);
	  //cout << "FPGAProjectionRouter "<<getName()<<" "<<inputproj_[j]->getName()<<endl;
	  //cout << "iphitmp iphi "<<iphitmp<<" "<<iphi<<endl;
	  assert(iphi>=4);
	  assert(iphi<=27);
	  iphi-=4;
	  iphi=(iphi>>1);
	  iphi=iphi&3;
	  assert(iphi>=0);
	  assert(iphi<=3);
	  
	  assert(allproj_!=0);

	  unsigned int index=allproj_->nTracklets();
	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));

	  //cout << "index iphi : "<<index<<" "<<iphi<<endl;
	  
	  if (iphi==0) {
	    assert(vmprojPHI1_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI1_->getName()<<endl;
	    }
	    vmprojPHI1_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==1) {
	    assert(vmprojPHI2_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI2_->getName()<<endl;
	    }
	    vmprojPHI2_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==2) {
	    assert(vmprojPHI3_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI3_->getName()<<endl;
	    }
	    vmprojPHI3_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }

	  if (iphi==3) {
	    assert(vmprojPHI4_!=0);
	    if (debug1){
	      cout << "FPGAProjectionRouter "<<getName()<<" add projection to : "<<vmprojPHI4_->getName()<<endl;
	    }
	    vmprojPHI4_->addTracklet(inputproj_[j]->getFPGATracklet(i),index);
	  }	  
	}
      }
    }


    if (writeAllProjections) {
      static ofstream out("allprojections.txt"); 
      out << getName() << " " << allproj_->nTracklets() << endl;
    } 
   

    if (writeVMProjections) {
      static ofstream out("vmprojections.txt"); 
      out << vmprojPHI1_->getName() << " " << vmprojPHI1_->nTracklets() << endl;      out << vmprojPHI2_->getName() << " " << vmprojPHI2_->nTracklets() << endl;      out << vmprojPHI3_->getName() << " " << vmprojPHI3_->nTracklets() << endl;      out << vmprojPHI4_->getName() << " " << vmprojPHI4_->nTracklets() << endl;
    }
  }
  
  
private:

  int layer_; 
  int disk_; 

  vector<FPGATrackletProjections*> inputproj_;
  FPGATrackletProjections* inputplusproj_;
  FPGATrackletProjections* inputminusproj_;

  FPGAAllProjections* allproj_;
  FPGAVMProjections* vmprojPHI1_;
  FPGAVMProjections* vmprojPHI2_;
  FPGAVMProjections* vmprojPHI3_;
  FPGAVMProjections* vmprojPHI4_;


};

#endif
