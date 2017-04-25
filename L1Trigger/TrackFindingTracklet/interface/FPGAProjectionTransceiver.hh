//This class implementes the projection tranceiver
#ifndef FPGAPROJECTIONTRANSCEIVER_H
#define FPGAPROJECTIONTRANSCEIVER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAProjectionTransceiver:public FPGAProcessBase{

public:

  FPGAProjectionTransceiver(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){

    outputprojLPHI1=0;
    outputprojLPHI2=0;
    outputprojLPHI3=0;

    outputprojDPHI1=0;
    outputprojDPHI2=0;
    outputprojDPHI3=0;

    
    layer_=0;
    disk_=0;

    string subname=name.substr(3,4);
    //cout << "FPGAProjectionTransceiver name subname : "<<name<<" "<<subname<<endl;

    if (subname=="L3D4") {
      layer_=3;
      disk_=4;
      return;
    }

    if (subname=="L4D3") {
      layer_=4;
      disk_=3;
      return;
    }

    if (subname=="L5D2") {
      layer_=5;
      disk_=2;
      return;
    }

    if (subname=="L6D1") {
      layer_=6;
      disk_=1;
      return;
    }

    
    subname=name.substr(3,2);
    //cout << "FPGAProjectionTransceiver name subname : "<<name<<" "<<subname<<endl;
    
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

    if (subname=="F1") disk_=1;
    if (subname=="F2") disk_=2;
    if (subname=="F3") disk_=3;
    if (subname=="F4") disk_=4;
    if (subname=="F5") disk_=5;

    if (subname=="B1") disk_=-1;
    if (subname=="B2") disk_=-2;
    if (subname=="B3") disk_=-3;
    if (subname=="B4") disk_=-4;
    if (subname=="B5") disk_=-5;

    
    assert(layer_!=0||disk_!=0);
    
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    string subname=memory->getName().substr(11,1);
    assert(subname=="L"||subname=="D");
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()<<" subname "<<subname
	   << " to output "<<output<<endl;
    }
    if (output=="projoutPHI1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      if (subname=="L") {
	outputprojLPHI1=tmp;
      }
      if (subname=="D") {
	outputprojDPHI1=tmp;
      }
      return;
    }
    if (output=="projoutPHI2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      if (subname=="L") {
	outputprojLPHI2=tmp;
      }
      if (subname=="D") {
	outputprojDPHI2=tmp;
      }
      return;
    }
    if (output=="projoutPHI3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      if (subname=="L") {
	outputprojLPHI3=tmp;
      }
      if (subname=="D") {
	outputprojDPHI3=tmp;
      }
      return;
    }

    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }

    if (input=="projin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojections_.push_back(tmp);
      return;
    }

    assert(0);
  }

  //Copy otherSector->inputprojections_ to this->outputprojections_ 
  void execute(FPGAProjectionTransceiver* otherSector){

    //cout << "FPGAProjectionTransceiver::execute "<<getName()<<endl;

    if (!doProjections) return;
    
    unsigned int count=0;
    //cout << "in FPGAProjectionTransceiver "<<otherSector->inputprojections_.size()<<endl;
    for(unsigned int i=0;i<otherSector->inputprojections_.size();i++){
      FPGATrackletProjections* otherProj=otherSector->inputprojections_[i];
      for (unsigned int l=0;l<otherProj->nTracklets();l++){
	count++;
	FPGATracklet* tracklet=otherProj->getFPGATracklet(l);
	FPGAWord fpgaphi;
	bool layer=false;
	bool disk=false;
	//cout << "FPGAProjectionTransceiver otherProj->name() = "<<otherProj->getName()
	//     <<" layer "<<layer_<<" disk "<<disk_<<" tracklet layer disk : "<<tracklet->layer()<<" "<<tracklet->disk()<<endl;
	int disk1=disk_;
	if (tracklet->t()<0.0) disk1=-disk_;
	//Handle PT that handles both disk and layer
	if (layer_!=0&&disk1!=0) {
	  if (abs(tracklet->disk())){
	    if (layer_<3){
	      fpgaphi=tracklet->fpgaphiproj(layer_);
	      assert(tracklet->minusNeighbor(layer_)||tracklet->plusNeighbor(layer_));
	      layer=true;
	    }
	    else {
	      fpgaphi=tracklet->fpgaphiprojdisk(disk1);
	      assert(tracklet->minusNeighborDisk(disk1)||tracklet->plusNeighborDisk(disk1));
	      disk=true;
	    }
	  } else {
	    //cout << "FPGAProjectionTransceiver "<<tracklet<<" layer_ = "<<layer_<<endl;
	    if (!tracklet->fpgazproj(layer_).atExtreme()){
	      fpgaphi=tracklet->fpgaphiproj(layer_);
	      if (!(tracklet->minusNeighbor(layer_)||tracklet->plusNeighbor(layer_))) continue;
	      layer=true;
	    } else {
	      fpgaphi=tracklet->fpgaphiprojdisk(disk1);
	      if (!tracklet->fpgarprojdisk(disk1).atExtreme()){
		assert(tracklet->minusNeighborDisk(disk1)||tracklet->plusNeighborDisk(disk1));
		disk=true;
	      }
	    }
	  }
	  if (fpgaphi.atExtreme()) {
	    cout << "FPGAProjectionTransceiver: Warning skipping projection"<<endl;
	    continue;
	  }
	}
	//Handle PT to only a layer
	if (layer_!=0&&disk1==0) {
	  fpgaphi=tracklet->fpgaphiproj(layer_);
	  layer=true;
	}
	//Handle PT to only a disk
	if (disk1!=0&&layer_==0) {
	  fpgaphi=tracklet->fpgaphiprojdisk(disk1);
	  disk=true;
	}
	
	assert(disk1!=0||layer_!=0);
      

	assert(disk||layer);

	//cout << "(bool) layer disk : "<<layer<<" "<<disk<<endl;
	
	int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);
    
	//cout << "FPGAProjectionTransceiver iphivmRaw "<<iphivmRaw << " "
	//     <<((fpgaphi.value()+1)>>(fpgaphi.nbits()-5))<<endl;
	if (iphivmRaw<4||iphivmRaw>27) {
	  cout << "FPGAProjectionTransceiver "<<getName()<<" will skip projection"<<endl;
	  continue;
	}
	assert(iphivmRaw>=4);
	assert(iphivmRaw<=27);

	int iphi=(iphivmRaw-4)>>3;

	//cout << "FPGAProjectionTranceiver "<<getName()<<" layer fpgaphi iphivmRaw iphi : "<<layer_<<" "<<fpgaphi.value()<<" "<<iphivmRaw<<" "<<iphi<<endl;

    
	assert(iphi>=0);
	assert(iphi<=2);

	if (iphi==0) {
	  if (layer) {
	    //cout << "In getName = "<<getName()<<endl;
	    if (outputprojLPHI1==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojLPHI1 is zero"<<endl;
	    }
	    assert(outputprojLPHI1!=0);
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojLPHI1->getName()<<endl;
	    outputprojLPHI1->addTracklet(otherProj->getFPGATracklet(l));
	  }
	  if (disk) {
	    if (outputprojDPHI1==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojDPHI1 is zero"<<endl;
	    }
	    //cout << "In getName = "<<getName()<<endl;
	    assert(outputprojDPHI1!=0);
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojDPHI1->getName()<<endl;
	    //cout << "FPGAProjectionTransceiver add projection to : "<<outputprojDPHI1->getName()<<endl;
	    outputprojDPHI1->addTracklet(otherProj->getFPGATracklet(l));
	  }
	}

	if (iphi==1) {
	  if (layer) {
	    if (outputprojLPHI2==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojLPHI2 is zero"<<endl;
	    }
	    assert(outputprojLPHI2!=0);
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojLPHI2->getName()<<endl;
	    outputprojLPHI2->addTracklet(otherProj->getFPGATracklet(l));
	  }
	  if (disk) {
	    if (outputprojDPHI2==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojDPHI2 is zero"<<endl;
	    }
	    assert(outputprojDPHI2!=0);
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojDPHI2->getName()<<endl;
	    outputprojDPHI2->addTracklet(otherProj->getFPGATracklet(l));
	  }
	}
	
	if (iphi==2) {
	  if (layer) {
	    //cout << "In getName = "<<getName()<<endl;
	    if (outputprojLPHI3==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojLPHI3 is zero"<<endl;
	    }
	    assert(outputprojLPHI3!=0);
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojLPHI3->getName()<<endl;
	    outputprojLPHI3->addTracklet(otherProj->getFPGATracklet(l));
	  }
	  if (disk) {
	    if (outputprojDPHI3==0) {
	      cout << "FPGAProjectionTransceiver in : "<<getName()<< " outputprojDPHI3 is zero"<<endl;
	    }
	    assert(outputprojDPHI3!=0);
	    //cout << "FPGAProjectionTransceiver add projection to : "<<outputprojDPHI3->getName()<<endl;
	    //cout << "Adding tracklet "<<otherProj->getFPGATracklet(l)<<" to "<<outputprojDPHI3->getName()<<endl;
	    outputprojDPHI3->addTracklet(otherProj->getFPGATracklet(l));
	  }
	}

      }	

    }

    if (writeProjectionTransceiver) {
      static ofstream out("projectiontransceiver.txt");
      out << getName() << " " 
	  << count << endl;
    }

  }


private:

  int layer_;
  int disk_;

  FPGATrackletProjections*     outputprojLPHI1;
  FPGATrackletProjections*     outputprojLPHI2;
  FPGATrackletProjections*     outputprojLPHI3;

  FPGATrackletProjections*     outputprojDPHI1;
  FPGATrackletProjections*     outputprojDPHI2;
  FPGATrackletProjections*     outputprojDPHI3;


  vector<FPGATrackletProjections*> inputprojections_;


};

#endif
