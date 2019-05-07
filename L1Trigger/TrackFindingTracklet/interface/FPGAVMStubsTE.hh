//This class holds the reduced VM stubs
#ifndef FPGAVMSTUBSTE_H
#define FPGAVMSTUBSTE_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGAVMStubsTE:public FPGAMemoryBase{

public:

  FPGAVMStubsTE(string name, unsigned int iSector, 
	      double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    string subname=name.substr(6,2);
    layer_ = 0;
    disk_  = 0;
    phibin_=-1;
    other_=0;
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
    if (layer_==0&&disk_==0) {
      cout << name<<" subname = "<<subname<<" "<<layer_<<" "<<disk_<<endl;
    }
    assert((layer_!=0)||(disk_!=0));
    overlap_=false;
    extra_=false;
    subname=name.substr(11,1);
    if (subname=="X") overlap_=true;
    if (subname=="Y") overlap_=true;
    if (subname=="W") overlap_=true;
    if (subname=="Q") overlap_=true;
    if (subname=="R") overlap_=true;
    if (subname=="S") overlap_=true;
    if (subname=="T") overlap_=true;
    if (subname=="Z") overlap_=true;
    if (subname=="x") overlap_=true;
    if (subname=="y") overlap_=true;
    if (subname=="z") overlap_=true;
    if (subname=="w") overlap_=true;
    if (subname=="q") overlap_=true;
    if (subname=="r") overlap_=true;
    if (subname=="s") overlap_=true;
    if (subname=="t") overlap_=true;
    if (subname=="I") extra_=true;
    if (subname=="J") extra_=true;
    if (subname=="K") extra_=true;
    if (subname=="L") extra_=true;
    

    extended_ = false;
    if (subname=="a") extended_=true;
    if (subname=="b") extended_=true;
    if (subname=="c") extended_=true;
    if (subname=="d") extended_=true;
    if (subname=="e") extended_=true;
    if (subname=="f") extended_=true;
    if (subname=="g") extended_=true;
    if (subname=="h") extended_=true;
    if (subname=="x") extended_=true;
    if (subname=="y") extended_=true;
    if (subname=="z") extended_=true;
    if (subname=="w") extended_=true;
    if (subname=="q") extended_=true;
    if (subname=="r") extended_=true;
    if (subname=="s") extended_=true;
    if (subname=="t") extended_=true;
        
    subname=name.substr(12,2);
    phibin_=subname[0]-'0';
    if (subname[1]!='n') {
      phibin_=10*phibin_+(subname[1]-'0');
    }

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
  
  bool addStub(std::pair<FPGAStub*,L1TStub*> stub) {
    int mask=-1;
    if (layer_==1 || layer_==3 || layer_==5 || (extended_ && layer_==2))
      mask=1023;

    int binlookup= -1;
    if(!extended_){
      binlookup=stub.first->getVMBits().value()&mask;
      if (overlap_) {
	binlookup=stub.first->getVMBitsOverlap().value();
      }
    } else{
      binlookup=stub.first->getVMBitsExtended().value()&mask;
      if (overlap_) {
	binlookup=stub.first->getVMBitsOverlapExtended().value()&mask;
      }
    }
    if (extra_) {
      binlookup=stub.first->getVMBitsExtra().value();
    }
    if (binlookup<0) {
      cout << getName() << " binlookup = "<<binlookup<<endl;
    }
    assert(binlookup>=0);
    int bin=(binlookup/8);

    bool pass=passbend(stub.first->bend().value());

    if (!pass) {
      if (debug1) cout << getName() << " Stub failed bend cut. bend = "<<FPGAStub::benddecode(stub.first->bend().value(),stub.first->isPSmodule())<<endl;
      return false;
    }

    if(!extended_){
      if (overlap_) {
	if (disk_==1) {
	  bool negdisk=stub.first->disk().value()<0.0;
	  assert(bin<4);
	  if (negdisk) bin+=4;
	  stubsbinned_[bin].push_back(stub);
	  if (debug1) cout << getName()<<" Stub with lookup = "<<binlookup
			   <<" in disk = "<<disk_<<"  in bin = "<<bin<<endl;
	}
      } else {
        if (stub.first->isBarrel()){
          if (!isinner_) {
	    stubsbinned_[bin].push_back(stub);
          }
	
	} else {

	  bool negdisk=stub.first->disk().value()<0.0;

	  if (disk_%2==0) {
	    assert(bin<4);
	    if (negdisk) bin+=4;
	    stubsbinned_[bin].push_back(stub);
	  }
		  
	}
      }
    }
    else {
      if(!isinner_){
	if(layer_>0){
	  stubsbinned_[bin].push_back(stub);
	}
	else{
	  if(overlap_){
	    assert(disk_==1); // D1 from L2L3D1

	    //bin 0 is PS, 1 through 3 is 2S
	    
	    bin = stub.first->ir(); // 0 to 9
	    bin = bin >> 2; // 0 to 2
	    bin += 1;
	    if(stub.first->isPSmodule())
	      bin = 0;
	  }
	  assert(bin<4);
	  bool negdisk=stub.first->disk().value()<0.0;
	  if (negdisk) bin+=4;
	  stubsbinned_[bin].push_back(stub);	  
	}
      }
    }
    if (debug1) cout << "Adding stubs to "<<getName()<<endl;
    stubs_.push_back(stub);
    return true;
  }

  unsigned int nStubs() const {return stubs_.size();}
  unsigned int nStubsBinned(unsigned int bin) const {return stubsbinned_[bin].size();}

  FPGAStub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  FPGAStub* getFPGAStubBinned(unsigned int bin, unsigned int i) const {return stubsbinned_[bin][i].first;}
  L1TStub* getL1TStubBinned(unsigned int bin, unsigned int i) const {return stubsbinned_[bin][i].second;}
  std::pair<FPGAStub*,L1TStub*> getStubBinned(unsigned int bin,unsigned int i) const {return stubsbinned_[bin][i];}


  
  void clean() {
    stubs_.clear();
    for (unsigned int i=0;i<NLONGVMBINS;i++){
      stubsbinned_[i].clear();
    }
  }

  void writeStubs(bool first) {

    std::string fname="../data/MemPrints/VMStubsTE/VMStubs_";
    fname+=getName();

    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_ = 0;
      event_ = 1;
      out_.open(fname.c_str());
    } else {
      out_.open(fname.c_str(),std::ofstream::app);
    }
      
    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    //if (layer_!=0) { // barrel
    if (layer_!=0 or disk_!=0) { // same format for barrel and disk?
      if (isinner_) { // inner VM for TE purpose
        for (unsigned int j=0;j<stubs_.size();j++){
          string stub=stubs_[j].first->stubindex().str();
          stub+="|";
          stub+=stubs_[j].first->bend().str();
          stub+="|";	  
          FPGAWord iphifinebins;
	  if (overlap_) {
	    iphifinebins.set(stubs_[j].first->iphivmFineBins(5,nfinephioverlapinner),
			     nfinephioverlapinner,true,__LINE__,__FILE__);
	  } else if (layer_>0) {
	    iphifinebins.set(stubs_[j].first->iphivmFineBins(5,nfinephibarrelinner),
			     nfinephibarrelinner,true,__LINE__,__FILE__);
	  } else if (disk_>0) {
	    iphifinebins.set(stubs_[j].first->iphivmFineBins(5,nfinephidiskinner),
			     nfinephidiskinner,true,__LINE__,__FILE__);
	  } else {
	    assert(0);
	  }

          stub+=iphifinebins.str();
          stub+="|";
          FPGAWord tmp;
	  if (overlap_) {
	    assert(stubs_[j].first->getVMBitsOverlap().nbits()!=-1);
	    stub+=stubs_[j].first->getVMBitsOverlap().str();
	  } else {
	    assert(stubs_[j].first->getVMBits().nbits()!=-1);
	    stub+=stubs_[j].first->getVMBits().str();
	  }
	  out_<<"0x";
          if (j<16) out_ <<"0";
          out_ << hex << j << dec ;
          out_ <<" "<<stub<<" "<<hexFormat(stub)<<endl;
        }     
      }
      else { // outer VM for TE purpose
        for (unsigned int i=0;i<NLONGVMBINS;i++) {
      for (unsigned int j=0;j<stubsbinned_[i].size();j++){
        string stub=stubsbinned_[i][j].first->stubindex().str();
        stub+="|";
        stub+=stubsbinned_[i][j].first->bend().str();
        stub+="|";
        FPGAWord iphifinebins;

	if (overlap_) { 
	  iphifinebins.set(stubsbinned_[i][j].first->iphivmFineBins(4,nfinephioverlapouter),
			   nfinephioverlapouter,true,__LINE__,__FILE__);
	} else if (layer_>0) {
	  iphifinebins.set(stubsbinned_[i][j].first->iphivmFineBins(5,nfinephibarrelouter),
			   nfinephibarrelouter,true,__LINE__,__FILE__);
	} else if (disk_>0) {
	  iphifinebins.set(stubsbinned_[i][j].first->iphivmFineBins(4,nfinephidiskouter),
			   nfinephidiskouter,true,__LINE__,__FILE__);
	} else {
	  assert(0);
	}
        stub+=iphifinebins.str();
        stub+="|";
	int ifinezbin=-1;
	if (overlap_) {
	  ifinezbin=stubsbinned_[i][j].first->getVMBitsOverlap().value();
	} else {
	  ifinezbin=stubsbinned_[i][j].first->getVMBits().value();
	}
	assert(ifinezbin!=-1);
        FPGAWord finezbin;
	finezbin.set(ifinezbin&7,3,true,__LINE__,__FILE__);
	stub+=finezbin.str();
        out_ << hex << i << " " << j << dec << " "<<stub<<" "<<hexFormat(stub)<<endl;
      }
        }
      }
    }
    /*  uncomment and modify below in case disk VMStubsTE have different format
    else if (disk_!=0) { // disk
      if (isinner_) { // inner disk VM for TE purpose
        for (unsigned int j=0;j<stubs_.size();j++){
          string stub=stubs_[j].first->stubindex().str();
          stub+="|";
          FPGAWord tmp;
          tmp.set(stubs_[j].first->getVMBits().value(),10,true,__LINE__,__FILE__);
          stub+=tmp.str();  
          if (j<16) out_ <<"0";
          out_ << hex << j << dec ;
          out_ <<" "<<stub<< endl;
        }
      }
      else { // outer disk VM for TE purpose
        for (unsigned int i=0;i<NLONGVMBINS;i++) {
      for (unsigned int j=0;j<stubsbinned_[i].size();j++){
	    string stub=stubsbinned_[i][j].first->stubindex().str();
	    out_ << hex << i << " " << j << dec << " "<<stub<<endl;
	  }
        }
      }
    }
    */

    
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

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
    //if (iSector_==1) cout << "phiRange: "<<getName()<<" "<<phibin()<<" overlap "<<overlap_<<" phimin, max "<<phimin<<" "<<phimax<<endl;
    return;
    
  }
  
  void setother(FPGAVMStubsTE* other){
    other_=other;
  }
  
  FPGAVMStubsTE* other() const {
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
      //outvmbendcut << i << " " << vmbendtable_[i] << endl;
      outvmbendcut << vmbendtable_[i] << endl;
    }
    outvmbendcut.close();
  }
  

private:

  int layer_;
  int disk_;
  int phibin_;
  FPGAVMStubsTE* other_;
  bool overlap_;
  bool extra_;
  bool extended_; // for the L2L3->D1 and D1D2->L2
  bool isinner_;  // is inner layer/disk for TE purpose
  double phimin_;
  double phimax_;
  std::vector<bool> vmbendtable_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubsbinned_[NLONGVMBINS];

};

#endif
