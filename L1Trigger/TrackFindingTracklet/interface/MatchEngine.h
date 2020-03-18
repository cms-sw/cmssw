//This class implementes the tracklet engine
#ifndef MATCHENGINE_H
#define MATCHENGINE_H

#include "ProcessBase.h"
#include "TrackletCalculator.h"

using namespace std;

class MatchEngine:public ProcessBase{

public:

  MatchEngine(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    layer_=0;
    disk_=0;
    string subname=name.substr(3,2);
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

    if (layer_>0) {

      unsigned int nbits=3;
      if (layer_>=4) nbits=4;
      
      for(unsigned int irinv=0;irinv<32;irinv++){
	double rinv=(irinv-15.5)*(1<<(nbitsrinv-5))*krinvpars;
	double projbend=bend(rmean[layer_-1],rinv);
	for(unsigned int ibend=0;ibend<(unsigned int)(1<<nbits);ibend++){
	  double stubbend=Stub::benddecode(ibend,layer_<=3);
	  bool pass=fabs(stubbend-projbend)<mecut;
	  table_.push_back(pass);
	}
      }

      if (writeMETables){
	ofstream out;
	char layer='0'+layer_;
	string fname="METable_L";
	fname+=layer;
	fname+=".dat";
	out.open(fname.c_str());
	out << "{" <<endl;
	for(unsigned int i=0;i<table_.size();i++){
	  if (i!=0) {
	    out <<","<<endl;
	  }
	  out << table_[i] ;
	}
	out << "};"<<endl;
	out.close();
      }
      
    }

    if (disk_>0) {

      for(unsigned int iprojbend=0;iprojbend<32;iprojbend++){
	double projbend=0.5*(iprojbend-15.0);
	for(unsigned int ibend=0;ibend<8;ibend++){
	  double stubbend=Stub::benddecode(ibend,true);
	  bool pass=fabs(stubbend-projbend)<mecutdisk;
	  tablePS_.push_back(pass);
	}
	for(unsigned int ibend=0;ibend<16;ibend++){
	  double stubbend=Stub::benddecode(ibend,false);
	  bool pass=fabs(stubbend-projbend)<mecutdisk;
	  table2S_.push_back(pass);
	}
      }
      
    }

    
  }

  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout") {
      CandidateMatchMemory* tmp=dynamic_cast<CandidateMatchMemory*>(memory);
      assert(tmp!=0);
      candmatches_=tmp;
      return;
    }
    assert(0);

  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="vmstubin") {
      VMStubsMEMemory* tmp=dynamic_cast<VMStubsMEMemory*>(memory);
      assert(tmp!=0);
      vmstubs_=tmp;
      return;
    }
    if (input=="vmprojin") {
      VMProjectionsMemory* tmp=dynamic_cast<VMProjectionsMemory*>(memory);
      assert(tmp!=0);
      vmprojs_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    bool barrel=layer_>0;

    unsigned int countall=0;
    unsigned int countpass=0;
    
    //bool print=getName()=="ME_L4PHIB12"&&iSector_==3;
    //print=false;

    constexpr unsigned int kNBitsBuffer=3;  

    int writeindex=0;
    int readindex=0;
    std::pair<int,int> projbuffer[1<<kNBitsBuffer]; //iproj zbin

    //The next projection to read, the number of projections and flag if we have
    //more projections to read
    int iproj=0;
    int nproj=vmprojs_->nTracklets();
    bool moreproj=iproj<nproj;

    //Projection that is read from the buffer and compared to stubs  
    int rzbin=0;
    int projfinerz=0;
    int projfinerzadj=0;

    
    int projindex;
    int projrinv=0;
    bool isPSseed=false;
    
    //Number of stubs for current zbin and the stub being processed on this clock
    int nstubs=0;
    int istub=0;

    //Main processing loops starts here  
    for (unsigned int istep=0;istep<MAXME;istep++) {

      countall++;
      
      int writeindexplus=(writeindex+1)%(1<<kNBitsBuffer);
      int writeindexplusplus=(writeindex+2)%(1<<kNBitsBuffer);

      //Determine if buffere is full - or near full as a projection
      //can point to two z bins we might fill two slots in the buffer
      bool bufferfull=(writeindexplus==readindex)||(writeindexplusplus==readindex);

      //Determin if buffere is empty
      bool buffernotempty=(writeindex!=readindex);

      //If we have more projections and the buffer is not full we read
      //next projection and put in buffer if there are stubs in the 
      //memory the projection points to

      if ((!moreproj)&&(!buffernotempty)) break;
		
      if (moreproj&&(!bufferfull)){
	Tracklet* proj=vmprojs_->getFPGATracklet(iproj);

	int iprojtmp=iproj;
	
	iproj++;
	moreproj=iproj<nproj;

	unsigned int rzfirst = barrel?proj->zbin1projvm(layer_):proj->rbin1projvm(disk_);
	unsigned int rzlast = rzfirst;
	bool second=(barrel?proj->zbin2projvm(layer_):proj->rbin2projvm(disk_))==1;
	if (second) rzlast += 1;

	//Check if there are stubs in the memory
	int nstubfirst=vmstubs_->nStubsBin(rzfirst);
	int nstublast=vmstubs_->nStubsBin(rzlast);
	bool savefirst=nstubfirst!=0;
	bool savelast=second&&(nstublast!=0);

	int writeindextmp=writeindex;
	int writeindextmpplus=(writeindex+1)%(1<<kNBitsBuffer);
	
	if (savefirst&&savelast) {
	  writeindex=writeindexplusplus;
	} else if (savefirst||savelast) {
	  writeindex=writeindexplus;
	}

	if (savefirst) { //FIXME code needs to be cleaner
	  std::pair<int,int> tmp(iprojtmp,rzfirst);
	  projbuffer[writeindextmp]=tmp;
	}
	if (savelast) {
	  std::pair<int,int> tmp(iprojtmp,rzlast+100); //hack to flag that this is second bin
	  if (savefirst) {
	    projbuffer[writeindextmpplus]=tmp;
	  } else {
	    projbuffer[writeindextmp]=tmp;
	  }
	}
      }
      

      //If the buffer is not empty we have a projection that we need to 
      //process.

      if (buffernotempty) {

	int istubtmp=istub;

	//New projection
	if (istub==0) {

	  projindex=projbuffer[readindex].first;
	  rzbin=projbuffer[readindex].second;
	  bool second=false;
	  if (rzbin>=100) {
	    rzbin-=100;
	    second=true;
	  }

	  Tracklet* proj=vmprojs_->getFPGATracklet(projindex);

	  nstubs=vmstubs_->nStubsBin(rzbin);

	  projfinerz = barrel?proj->finezvm(layer_):proj->finervm(disk_);

	  projrinv=barrel?(16+(proj->fpgarinv().value()>>(proj->fpgarinv().nbits()-5))):proj->getBendIndex(disk_).value();
	  assert(projrinv>=0);
	  assert(projrinv<32);
	  
	  isPSseed=proj->PSseed()==1;
	  
	  //Calculate fine z position
	  if (second) {
	    projfinerzadj=projfinerz-8;
	  } else {
	    projfinerzadj=projfinerz;
	  }
	  if (nstubs==1) {
	    istub=0;
	    readindex=(readindex+1)%(1<<kNBitsBuffer);
	  } else {
	    istub++;
	  }
	} else {
	  //Check if last stub, if so, go to next buffer entry 
	  if (istub+1>=nstubs){
	    istub=0;
	    readindex=(readindex+1)%(1<<kNBitsBuffer);
	  } else {
	    istub++;
	  }
	}

	//Read stub memory and extract data fields
	std::pair<Stub*,L1TStub*> stub=vmstubs_->getStubBin(rzbin,istubtmp);

	bool isPSmodule=stub.first->isPSmodule();
	
	int stubfinerz=barrel?stub.first->finez().value():stub.first->finer().value();
	
	int nbits=isPSmodule?3:4;

	unsigned int index=(projrinv<<nbits)+stub.first->bend().value();

	//Check if stub z position consistent
	int idrz=stubfinerz-projfinerzadj;
	bool pass;
	
	if (barrel) {
	  if (isPSseed) {
	    pass=idrz>=-2&&idrz<=2;
	  } else {
	    pass=idrz>=-5&&idrz<=5;
	  }
	} else {
	  if (isPSmodule) {
	    pass=idrz>=-1&&idrz<=1;
	  } else {
	    pass=idrz>=-5&&idrz<=5;
	  }
	}

	//Check if stub bend and proj rinv consistent
	if (pass){
	  if (barrel?table_[index]:(isPSmodule?tablePS_[index]:table2S_[index])) {
	    Tracklet* proj=vmprojs_->getFPGATracklet(projindex);
	    std::pair<Tracklet*,int> tmp(proj,vmprojs_->getAllProjIndex(projindex));
            if (writeSeeds) {
              ofstream fout("seeds.txt", ofstream::app);
              fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << proj->getISeed() << endl;
              fout.close();
            }
	    candmatches_->addMatch(tmp,stub);
	    countpass++;
	  }
	}
      }
      
    }

    if (writeME) {
      static ofstream out("matchengine.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }

    
  }

 
  double bend(double r, double rinv) {

    double dr=0.18;
    
    double delta=r*dr*0.5*rinv;

    double bend=-delta/0.009;
    if (r<55.0) bend=-delta/0.01;

    return bend;
    
  }

  
private:

  VMStubsMEMemory* vmstubs_;
  VMProjectionsMemory* vmprojs_;

  CandidateMatchMemory* candmatches_;

  int layer_;
  int disk_;

  //used in the layers
  vector<bool> table_;

  //used in the disks
  vector<bool> tablePS_;
  vector<bool> table2S_;

};

#endif
