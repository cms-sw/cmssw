// This class holds a list of stubs for an input link.
//This modules 'owns' the pointers to the stubs. All
//subsequent modules that handles stubs uses a pointer
//to the original stored here

 
#ifndef INPUTLINKMEMORY_H
#define INPUTLINKMEMORY_H

#include "L1TStub.h"
#include "Stub.h"
#include "MemoryBase.h"
#include "VMRouterPhiCorrTable.h"
#include <math.h>
#include <sstream>
#include <ctype.h>

using namespace std;

class InputLinkMemory:public MemoryBase{

public:

 InputLinkMemory(string name, unsigned int iSector, 
		 double phimin, double phimax):
  MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    
    string subname=name.substr(5,7);

    phiregion_=subname[3]-'A';
    phiregionoverlap_=-1;

    if (phiregion_>8) {
      if (subname[3]=='X') phiregionoverlap_=0;
      if (subname[3]=='Y') phiregionoverlap_=1;
      if (subname[3]=='Z') phiregionoverlap_=2;
      if (subname[3]=='W') phiregionoverlap_=3;
      if (subname[3]=='Q') phiregionoverlap_=4;
      if (subname[3]=='R') phiregionoverlap_=5;
      if (subname[3]=='S') phiregionoverlap_=6;
      if (subname[3]=='T') phiregionoverlap_=7;
      if (phiregionoverlap_!=-1) phiregion_=-1;
    }

    string subnamelayer=getName().substr(3,2);
    layerdisk_=subnamelayer[1]-'0';
    
    isLayer_=subnamelayer[0]=='L';
    isDisk_=subnamelayer[0]=='D';

    
  }

  bool addStub(L1TStub& al1stub, Stub& stub, string dtc="") {

    static bool first=true;
    static VMRouterPhiCorrTable phiCorrLayers[6];

    if (first) {
      for (int l=0;l<6;l++){
	int nbits=3;
	if (l>=3) nbits=4;
        phiCorrLayers[l].init(l+1,nbits,3);
      }
      first=false;
    }
    
    
       
    if (stub.layer().value()==-1 && isLayer() ) return false; 
    if (stub.layer().value()!=-1 && isDisk() ) return false;
    if (stub.layer().value()!=-1){
      if (stub.layer().value()+1!=layerdisk()) return false;
    } else {
      if (abs(stub.disk().value())!=layerdisk()) return false;
    }


    if (stub.layer().value()!=-1) {

      FPGAWord r=stub.r();
      FPGAWord bend=stub.bend();
      int bendbin=bend.value();
      int rbin=(r.value()+(1<<(r.nbits()-1)))>>(r.nbits()-3);
      
      int iphicorr=phiCorrLayers[stub.layer().value()].getphiCorrValue(bendbin,rbin);

      stub.setPhiCorr(iphicorr);

    }
    
    int iphivmRaw=stub.iphivmRaw();

    
    int phibin=-1;
    if (stub.layer().value()!=-1) {
      if (phiregionoverlap()==-1) {
	phibin=iphivmRaw/(32/nallstubslayers[layerdisk()-1]);
      } else {
	phibin=iphivmRaw/(32/nallstubsoverlaplayers[layerdisk()-1]);
      }
    } else {
      phibin=iphivmRaw/(32/nallstubsdisks[layerdisk()-1]); 
    }
    
    if (phibin!=phiregion() && phibin!=phiregionoverlap()) return false;
    
    if (getName().substr(10,dtc.size())!=dtc) return false;
    
    string half=getName().substr(getName().size()-3,3);
    if (half[1]!='n') {
      half=getName().substr(getName().size()-1,1);
    }

    assert(half[0]=='A' || half[0]=='B');

    if (half[0]=='B' && iphivmRaw<=15) return false;
    if (half[0]=='A' && iphivmRaw>15) return false;
    
    
    if (debug1) {
      cout << "Will add stub in "<<getName()<<" phimin_ phimax_ "<<phimin_<<" "<<phimax_<<" "<<"iphiwmRaw = "<<iphivmRaw<<" phi="<<al1stub.phi()<<" z="<<al1stub.z()<<" r="<<al1stub.r()<<endl;
    }
    if (stubs_.size()<MAXSTUBSLINK) {
      L1TStub* l1stub=new L1TStub(al1stub);
      Stub* stubptr=new Stub(stub);

      std::pair<Stub*,L1TStub*> tmp(stubptr,l1stub);
      stubs_.push_back(tmp);
    }
    return true;
  }

  unsigned int nStubs() const {return stubs_.size();}

  Stub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<Stub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  void writeStubs(bool first)
  {

    openFile(first,"../data/MemPrints/InputStubs/InputStubs_");
    
    for (unsigned int j=0;j<stubs_.size();j++){
      string stub = stubs_[j].first->str();
      if (j<16) out_ <<"0";
      out_ << hex << j << dec;
      out_ << " " << stub <<" "<<hexFormat(stub)<< endl;
    }
    out_.close();

  }
	
  void writeInputStubs(bool first, bool w2, bool padded) {
    
    //Barrel
    std::string fname="../data/MemPrints/InputStubs/InputStubs_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    int nlay[6];
    for (unsigned int i=0;i<6;i++) {
      nlay[i]=0;
    }

    unsigned int maxstubslink=0;
    if (TMUX==4) maxstubslink=21;
    else if (TMUX==6) maxstubslink=33;
    else if (TMUX==8) maxstubslink=45;
    else {
      cout << "ERROR! Only TMUX=4/6/8 are supported! Exiting..." << endl;
      return;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isBarrel()){
	int lay=stubs_[i].first->layer().value();
	assert(lay>=0);
	assert(lay<6);
	nlay[lay]++;
      }
    }
    unsigned long int nlay2[6];
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<6; ++i) {
      nlay2[i] = nlay2[i-1]+nlay[i];
      if (nlay2[i]>maxstubslink) {
	nlay2[i]=maxstubslink;
      }
      assert(nlay2[i]<=maxstubslink);
    }

    string marker="";

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in layer order
      string leftover="";

      
      for (unsigned int i=0;i<6;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isBarrel()){
	    unsigned int lay=stubs_[j].first->layer().value();
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      int scount = 0;
      for (unsigned int i=0;i<6;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isBarrel()){
	    unsigned int lay=stubs_[j].first->layer().value();
	    if (lay==i) {
	      scount++;
	      if (scount>(int)maxstubslink) continue;
	      string tmp=stubs_[j].first->strbare();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }
      if (padded) {
    for (int k=0;k<(int)maxstubslink-scount;k++) {
	  xline = (xline+1)&15;
	  out_<< std::hex << xline<<"0000000 "<< std::hex << xline<<"0000001";
	  out_<<" 51000003 51000007\n";
	}
      }
      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
    }    
  
    out_.close();
      
    //Now the disks

    //forward

    fname="../data/MemPrints/InputStubs/InputStubsDiskF_";
    fname+=getName();
    fname+="_";
    ostringstream ossF;
    ossF << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=ossF.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    for (unsigned int i=0;i<5;i++) {
      nlay[i]=0;
      nlay2[i]=0;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isDisk()&&stubs_[i].first->disk().value()>0){
	int lay=stubs_[i].first->disk().value()-1;
	assert(lay>=0);
	assert(lay<5);
	nlay[lay]++;
      }
    }
    nlay[5] = 0;
    nlay2[5] = 0;
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<5; ++i)
      nlay2[i] = nlay2[i-1]+nlay[i];

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in disk order
      string leftover="";
      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()>0){
	    unsigned int lay=stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()>0){
	    unsigned int lay=stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }

      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";

    }    
  
    out_.close();

    //back

    fname="../data/MemPrints/InputStubs/InputStubsDiskB_";
    fname+=getName();
    fname+="_";
    ostringstream ossB;
    ossB << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=ossB.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    for (unsigned int i=0;i<5;i++) {
      nlay[i]=0;
      nlay2[i]=0;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isDisk()&&stubs_[i].first->disk().value()<0){
	int lay=-stubs_[i].first->disk().value()-1;
	assert(lay>=0);
	assert(lay<5);
	nlay[lay]++;
      }
    }
    nlay[5] = 0;
    nlay2[5] = 0;
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<5; ++i)
      nlay2[i] = nlay2[i-1]+nlay[i];

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in disk order
      string leftover="";
      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()<0){
	    unsigned int lay=-stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()<0){
	    unsigned int lay=-stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }

      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";

    }    
  
    out_.close();

    //increment BX
    bx_++;
    if (bx_>7) bx_=0;

  }
  
  

  void clean() {
    for(unsigned int i=0;i<stubs_.size();i++){
      delete stubs_[i].first;
      delete stubs_[i].second;
    }
    stubs_.clear();
  }

  double phimin() const {return phimin_;}
  double phimax() const {return phimax_;}

  int phiregion() const {return phiregion_;}
  int phiregionoverlap() const {return phiregionoverlap_;}

  int layerdisk() const {return layerdisk_;}

  bool isLayer() const {return isLayer_;}
  bool isDisk() const {return isDisk_;}
  
  
private:

  double phimin_;
  double phimax_;
  vector<std::pair<Stub*,L1TStub*> > stubs_;

  int phiregion_;
  int phiregionoverlap_;
  
  int layerdisk_;

  int isLayer_;
  int isDisk_;

  
};

#endif
