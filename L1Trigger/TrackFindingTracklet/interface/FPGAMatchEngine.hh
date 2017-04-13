//This class implementes the tracklet engine
#ifndef FPGAMATCHENGINE_H
#define FPGAMATCHENGINE_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAMatchEngine:public FPGAProcessBase{

public:

  FPGAMatchEngine(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    layer_=0;
    disk_=0;
    string subname=name.substr(8,2);
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
    if (layer_==0&&disk_==0) {
      cout << name<<" subname = "<<subname<<" "<<layer_<<" "<<disk_<<endl;
    }
    assert((layer_!=0)||(disk_!=0));

  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout") {
      FPGACandidateMatch* tmp=dynamic_cast<FPGACandidateMatch*>(memory);
      assert(tmp!=0);
      candmatches_=tmp;
      return;
    }
    assert(0);

  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="vmstubin") {
      FPGAVMStubsME* tmp=dynamic_cast<FPGAVMStubsME*>(memory);
      assert(tmp!=0);
      vmstubs_=tmp;
      return;
    }
    if (input=="vmprojin") {
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojs_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    unsigned int countall=0;
    unsigned int countpass=0;

    //FIXME - order should be changed. Loop over tracklets in outer loop!
    for(unsigned int i=0;i<vmstubs_->nStubs();i++){
      if (debug1) {
	cout << "Found stub in "<<getName()<<endl;
      }
      std::pair<FPGAStub*,L1TStub*> stub=vmstubs_->getStub(i);
      if (layer_>0){
	for(unsigned int j=0;j<vmprojs_->nTracklets();j++){
	  FPGATracklet* proj=vmprojs_->getFPGATracklet(j);
	  //cout << "FPGAMatchEngine zproj = "<<proj->fpgazproj(layer_).value()<<" "<<stub.second->z()<<" layer_ = "<<layer_<<endl;
	  if (fabs(proj->zproj(layer_)-stub.second->z())>20.0) continue;
	  if (debug1) {
	    cout << "Adding match in "<<getName()<<endl;
	  }
	  countall++;
	  if (layer_>0) {
	    //cout << "  Proj: "<<proj->phiprojvm(layer_)
	    //     <<" "<<proj->zprojvm(layer_)<<" "<<proj->zproj(layer_)<<endl;
	    if (doMEMatch){
	      double zcut=10.0;
	      if (layer_==1&&proj->layer()==5) zcut=20;
	      if (layer_==1&&abs(proj->disk())==3) zcut=20;
	      if (fabs(proj->zproj(layer_)-stub.second->z())>zcut) continue;
	      double dphi=proj->phiproj(layer_)-stub.second->phi();
	      double deltaphi=two_pi/NSector;
	      dphi-=deltaphi/6.0;
	      do {
		if (dphi>0.5*deltaphi) dphi-=deltaphi;
		if (dphi<-0.5*deltaphi) dphi+=deltaphi;
	      }while (abs(dphi)>=0.5*deltaphi);
	      //cout << "layer_ dphi r*dphi "<<layer_<<" "<<dphi<<" "
	      //	   << dphi*stub.second->r() << endl;
	      if (layer_==1&&abs(dphi*stub.second->r())>0.12) continue;
	      if (layer_==2&&abs(dphi*stub.second->r())>0.15) continue;
	      if (layer_==3&&abs(dphi*stub.second->r())>0.25) continue;
	      if (abs(stub.first->phivm().value()-
		      stub.first->phivm().value())>1) continue;
	    }
	  }
	  countpass++;
	  candmatches_->addMatch(proj,stub);
	  if (countall>=MAXME) break;
	}
	if (countall>=MAXME) break;
      } else if (disk_!=0) {
	for(unsigned int j=0;j<vmprojs_->nTracklets();j++){
	  FPGATracklet* proj=vmprojs_->getFPGATracklet(j);
	  int disk=disk_;
	  if (proj->t()<0.0) disk=-disk_;
	  if (debug1) {
	    cout << " Found projection in "<<getName()<<" "
		 << proj->rprojdisk(disk)<<" "<<stub.second->r()<<endl;
	    
	  }
	  //cout << "FPGAMatchEngine "<<getName()<<" disk = "<<disk<<" rproj = "<<proj->rprojdisk(disk)<<" "<<proj->fpgarprojdisk(disk).value()<<" stub r = "<<stub.second->r()<<endl;
	  double rbin=10.0;
	  if (proj->rprojdisk(disk)<40.0) rbin=5.0;
	  if (fabs(proj->rprojdisk(disk)-stub.second->r())>rbin) continue;
	  countall++;
	  if (debug1) {
	    cout << "Adding match in "<<getName()<<endl;
	  }
	  if (disk_!=0) {
	    double rcut=5.0;
	    if (proj->rprojdisk(disk)<60.0) rcut=2.0;
	    if (fabs(proj->rprojdisk(disk)-stub.second->r())>rcut) continue;
     
	    //cout << "  Proj: "<<proj->phiprojvm(layer_)
	    //     <<" "<<proj->zprojvm(layer_)<<" "<<proj->zproj(layer_)<<endl;
	    if (doMEMatch){
	      if (abs(stub.first->phivm().value()-
		      stub.first->phivm().value())>1) {
		cout << "Rejecting match: "<<stub.second->z()<<endl;
		continue;
	      }
	    }
	  }
	  //cout << "FPGAMatchEngine "<<getName()<<" adding match to "<<candmatches_->getName()<<endl;
	  countpass++;
	  candmatches_->addMatch(proj,stub);
	  if (countall>=MAXME) break;
	}
	if (countall>=MAXME) break;
      } else {
	assert(0);
      }
      
    }

    if (writeME) {
      static ofstream out("matchengine.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }

  }


private:

  FPGAVMStubsME* vmstubs_;
  FPGAVMProjections* vmprojs_;

  FPGACandidateMatch* candmatches_;

  int layer_;
  int disk_;
 
};

#endif
