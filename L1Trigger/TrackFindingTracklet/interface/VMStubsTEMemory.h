//This class holds the reduced VM stubs
#ifndef VMSTUBSTEMEMORY_H
#define VMSTUBSTEMEMORY_H

#include "L1TStub.h"
#include "Stub.h"
#include "VMStubTE.h"
#include "MemoryBase.h"

using namespace std;

class VMStubsTEMemory:public MemoryBase{

public:
  
 VMStubsTEMemory(string name, unsigned int iSector, 
		 double phimin, double phimax):
  MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;

    //set the layer or disk that the memory is in
    initLayerDisk(6,layer_,disk_);

    //Pointer to other VMStub memory for creating stub pairs
    other_=0;    

    //What type of seeding is this memory used for
    initSpecialSeeding(11,overlap_,extra_,extended_);
    
    string subname=name.substr(12,2);
    phibin_=subname[0]-'0';
    if (subname[1]!='n') {
      phibin_=10*phibin_+(subname[1]-'0');
    }

    //set the bins used in the bend tabele
    unsigned int nbins=8;
    if (layer_>=4) nbins=16;
    if (disk_==1 && extended_ && overlap_) nbins = 16;
    for (unsigned int i=0;i<nbins;i++){
      vmbendtable_.push_back(true);
    }

    isinner_ = (layer_%2==1 or disk_%2==1);
    // special cases with overlap seeding
    if (overlap_ and layer_==2) isinner_ = true;
    if (overlap_ and layer_==3) isinner_ = false;
    if (overlap_ and disk_==1) isinner_ = false; 
    
    if (extra_ and layer_==2) isinner_ = true;
    if (extra_ and layer_==3) isinner_ = false;
    // more special cases for triplets
    if (!overlap_ and extended_ and layer_==2) isinner_ = true;
    if (!overlap_ and extended_ and layer_==3) isinner_ = false;
    if ( overlap_ and extended_ and layer_==2) isinner_ = false;
    if ( overlap_ and extended_ and disk_==1)  isinner_ = false;
    
  }
  
  bool addVMStub(VMStubTE vmstub) {

    FPGAWord binlookup=vmstub.vmbits();
    
    assert(binlookup.value()>=0);
    int bin=(binlookup.value()/8);

    //If the pt of the stub is consistent with the allowed pt of tracklets
    //in that can be formed in this VM and the other VM used in the TE.
    bool pass=passbend(vmstub.bend().value());

    if (!pass) {
      if (debug1) cout << getName() << " Stub failed bend cut. bend = "<<Stub::benddecode(vmstub.bend().value(),vmstub.isPSmodule())<<endl;
      return false;
    }

    bool negdisk=vmstub.stub().first->disk().value()<0.0;

    if(!extended_){
      if (overlap_) {
	if (disk_==1) {
	  //bool negdisk=vmstub.stub().first->disk().value()<0.0;
	  assert(bin<4);
	  if (negdisk) bin+=4;
	  stubsbinnedvm_[bin].push_back(vmstub);
	  if (debug1) cout << getName()<<" Stub with lookup = "<<binlookup.value()
			   <<" in disk = "<<disk_<<"  in bin = "<<bin<<endl;
	}
      } else {
        if (vmstub.stub().first->isBarrel()){
          if (!isinner_) {
	    stubsbinnedvm_[bin].push_back(vmstub);
          }
	
	} else {

	  //bool negdisk=vmstub.stub().first->disk().value()<0.0;

	  if (disk_%2==0) {
	    assert(bin<4);
	    if (negdisk) bin+=4;
	    stubsbinnedvm_[bin].push_back(vmstub);
	  }
		  
	}
      }
    }
    else {  //extended
      if(!isinner_){
	if(layer_>0){
	  stubsbinnedvm_[bin].push_back(vmstub);
	}
	else{
	  if(overlap_){
	    assert(disk_==1); // D1 from L2L3D1

	    //bin 0 is PS, 1 through 3 is 2S
	    if(vmstub.stub().first->isPSmodule()) {
	      bin = 0;
	    } else {
	      bin = vmstub.stub().first->ir(); // 0 to 9 //FIXME
	      bin = bin >> 2; // 0 to 2
	      bin += 1;
	    }
	    
	  }
	  //bool negdisk=vmstub.stub().first->disk().value()<0.0;
	  assert(bin<4);
	  if (negdisk) bin+=4;
	  stubsbinnedvm_[bin].push_back(vmstub);	  
	}
      }
    }

    if (debug1) cout << "Adding stubs to "<<getName()<<endl;
    stubsvm_.push_back(vmstub);
    return true;
  }
    
  unsigned int nVMStubs() const {return stubsvm_.size();}

  unsigned int nVMStubsBinned(unsigned int bin) const {return stubsbinnedvm_[bin].size();}

  VMStubTE getVMStubTE(unsigned int i) const {return stubsvm_[i];}

  VMStubTE getVMStubTEBinned(unsigned int bin, unsigned int i) const {return stubsbinnedvm_[bin][i];}


  void clean() {
    stubsvm_.clear();
    for (unsigned int i=0;i<NLONGVMBINS;i++){
      stubsbinnedvm_[i].clear();
    }
  }

  void writeStubs(bool first) {


    openFile(first,"../data/MemPrints/VMStubsTE/VMStubs_");
    
    if (isinner_) { // inner VM for TE purpose
      for (unsigned int j=0;j<stubsvm_.size();j++){
	out_<<"0x";
	if (j<16) out_ <<"0";
	out_ << hex << j << dec ;
	string stub=stubsvm_[j].str();
	out_ <<" "<<stub<<" "<<hexFormat(stub)<<endl;
      }     
    }
    else { // outer VM for TE purpose
      for (unsigned int i=0;i<NLONGVMBINS;i++) {
	for (unsigned int j=0;j<stubsbinnedvm_[i].size();j++){
	  string stub=stubsbinnedvm_[i][j].str();
	  out_ << hex << i << " " << j << dec << " "<<stub<<" "<<hexFormat(stub)<<endl;
	}
      }
    }
    
    out_.close();

  }

  int phibin() const {
    return phibin_;
  }

  void getPhiRange(double &phimin, double &phimax) {


    int nvm=-1;
    if (overlap_) {
      if (layer_>0) {
	nvm=nallstubsoverlaplayers[layer_-1]*nvmteoverlaplayers[layer_-1];
      }
      if (disk_>0) {
	nvm=nallstubsoverlapdisks[disk_-1]*nvmteoverlapdisks[disk_-1];
      }
    } else {
      if (layer_>0) {
	nvm=nallstubslayers[layer_-1]*nvmtelayers[layer_-1];
	if (extra_) {
	  nvm=nallstubslayers[layer_-1]*nvmteextralayers[layer_-1];
	}
      }
      if (disk_>0) {
	nvm=nallstubsdisks[disk_-1]*nvmtedisks[disk_-1];
      }
    }
    assert(nvm>0);
    assert(nvm<=32);
    double dphi=dphisectorHG/nvm;
    phimax=phibin()*dphi;
    phimin=phimax-dphi;

    return;
    
  }
  
  void setother(VMStubsTEMemory* other){
    other_=other;
  }
  
  VMStubsTEMemory* other() const {
    return other_;
  }

  void setbendtable(std::vector<bool> vmbendtable){
    assert(vmbendtable_.size()==vmbendtable.size());
    for (unsigned int i=0;i<vmbendtable.size();i++){
      vmbendtable_[i]=vmbendtable[i];
    }

    if (iSector_==0&&writeTETables) writeVMBendTable();
  }

  bool passbend(unsigned int ibend) const {
    assert(ibend<vmbendtable_.size());
    return vmbendtable_[ibend];
  }

  void writeVMBendTable() {
    
    ofstream outvmbendcut;
    outvmbendcut.open(getName()+"_vmbendcut.txt");
    unsigned int vmbendtableSize = vmbendtable_.size();
    assert(vmbendtableSize==16||vmbendtableSize==8);
    for (unsigned int i=0;i<vmbendtableSize;i++){
      outvmbendcut << vmbendtable_[i] << endl;
    }
    outvmbendcut.close();
  }
  

private:

  int layer_;
  int disk_;
  int phibin_;
  VMStubsTEMemory* other_;
  bool overlap_;
  bool extra_;
  bool extended_; // for the L2L3->D1 and D1D2->L2
  bool isinner_;  // is inner layer/disk for TE purpose
  double phimin_;
  double phimax_;
  std::vector<bool> vmbendtable_;

  std::vector<VMStubTE> stubsvm_;
  std::vector<VMStubTE> stubsbinnedvm_[NLONGVMBINS];
  
};

#endif
