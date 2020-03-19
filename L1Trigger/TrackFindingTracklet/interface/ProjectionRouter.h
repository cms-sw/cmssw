//This class implementes the projection router
#ifndef PROJECTIONROUTER_H
#define PROJECTIONROUTER_H

#include "ProcessBase.h"

using namespace std;

class ProjectionRouter:public ProcessBase{

public:

  ProjectionRouter(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    string subname=name.substr(3,2);
    layer_=0;
    disk_=0;

    for (unsigned int i=0;i<8;i++){
      vmprojs_.push_back(0);
    }
    
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
    assert(disk_!=0||layer_!=0);
    allproj_=0;

    nrbits_=5;
    nphiderbits_=6;

    
  }

  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="allprojout"){
      AllProjectionsMemory* tmp=dynamic_cast<AllProjectionsMemory*>(memory);
      assert(tmp!=0);
      allproj_=tmp;
      return;
    }
    
    int nproj=-1;
    int nprojvm=-1;
    if (layer_>0) {
      nproj=nallprojlayers[layer_-1];
      nprojvm=nvmmelayers[layer_-1];
    }
    if (disk_>0) {
      nproj=nallprojdisks[disk_-1];
      nprojvm=nvmmedisks[disk_-1];
    }
    assert(nproj!=-1);
    
    for (int iproj=0;iproj<nproj;iproj++) {
      for (int iprojvm=0;iprojvm<nprojvm;iprojvm++) {
	ostringstream oss;
	oss << "vmprojoutPHI"<<char(iproj+'A')<<iproj*nprojvm+iprojvm+1;
	string name=oss.str();
	if (output==name) {
	  VMProjectionsMemory* tmp=dynamic_cast<VMProjectionsMemory*>(memory);
	  assert(tmp!=0);
	  vmprojs_[iprojvm]=tmp;
	  return;
	}
      }
    }
    
    cout << "Did not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
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
	input=="proj31in"||input=="proj32in"||
	input=="proj33in"||input=="proj34in"||
	input=="proj35in"||input=="proj36in"||
	input=="proj37in"||input=="proj38in"||
	input=="proj39in"||input=="proj40in"||
	input=="proj41in"||input=="proj42in"||
	input=="proj43in"||input=="proj44in"||
	input=="proj45in"||input=="proj46in"||
	input=="proj47in"||input=="proj48in"){
      TrackletProjectionsMemory* tmp=dynamic_cast<TrackletProjectionsMemory*>(memory);
      assert(tmp!=0);
      inputproj_.push_back(tmp);
      return;
    }
    cout << "Could not find input : "<<input<<" in "<<getName()<<endl;
    assert(0);
  }

  void execute() {

    //cout << "ProjectionRouter::execute : "<<getName()<<" "<<inputproj_.size()<<endl;

    unsigned int count=0;

    //These are just here to test that the order is correct. Does not affect
    //the actual execution
    int lastTCID=-1;
    
    if (layer_!=0) {
      for (unsigned int j=0;j<inputproj_.size();j++){

	for (unsigned int i=0;i<inputproj_[j]->nTracklets();i++){

	  count++;
	  if (count>MAXPROJROUTER) continue;

	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiproj(layer_);
	  FPGAWord fpgaz=inputproj_[j]->getFPGATracklet(i)->fpgazproj(layer_);

	  //Check that all projections are valid
	  assert(!fpgaz.atExtreme());
	  assert(!fpgaphi.atExtreme());

	  int iphitmp=fpgaphi.value();
	  int iphi=iphitmp>>(fpgaphi.nbits()-5);

	  int nvm=-1;
	  int nbins=-1;
	  nvm=nvmmelayers[layer_-1]*nallstubslayers[layer_-1];
	  nbins=nvmmelayers[layer_-1];
	  assert(nvm>0);
	  iphi=(iphi/(32/nvm))&(nbins-1);	    
	  
	  assert(allproj_!=0);

	  unsigned int index=allproj_->nTracklets();

	  Tracklet* tracklet=inputproj_[j]->getFPGATracklet(i);

	  //This block of code just checks that the configuration is consistent
	  if (lastTCID>=tracklet->TCID()) {
	    cout << "Wrong TCID ordering for projections in "<<getName()<<endl;
	  } else {
	    lastTCID=tracklet->TCID();
	  }

	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));

	  addVMProj(vmprojs_[iphi],inputproj_[j]->getFPGATracklet(i),index);
	  
	}
      }
    } else {  //do the disk now
      for (unsigned int j=0;j<inputproj_.size();j++){

	for (unsigned int i=0;i<inputproj_[j]-> nTracklets();i++){
	  count++;
	  if (count>MAXPROJROUTER) continue;

	  int disk=disk_;
	  if (inputproj_[j]->getFPGATracklet(i)->t()<0.0) disk=-disk_;

	  
	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiprojdisk(disk);
	  FPGAWord fpgar=inputproj_[j]->getFPGATracklet(i)->fpgarprojdisk(disk);

	  //skip if projection is out of range!
	  assert(!fpgar.atExtreme());
	  assert(!fpgaphi.atExtreme());

	  int iphitmp=fpgaphi.value();
	  int iphi=iphitmp>>(fpgaphi.nbits()-5);

	  int nvm=-1;
	  int nbins=-1;
	  nvm=nvmmedisks[disk_-1]*nallstubsdisks[disk_-1];
	  nbins=nvmmedisks[disk_-1];
	  assert(nvm>0);
	  iphi=(iphi/(32/nvm))&(nbins-1);

	  assert(allproj_!=0);

	  unsigned int index=allproj_->nTracklets();
          if (writeSeeds) {
            ofstream fout("seeds.txt", ofstream::app);
            fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << inputproj_[j]->getFPGATracklet(i)->getISeed() << endl;
            fout.close();
          }
	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));

	  
	  Tracklet* tracklet=inputproj_[j]->getFPGATracklet(i);


	  //The next lines looks up the predicted bend based on:
	  // 1 - r projections
	  // 2 - phi derivative
	  // 3 - the sign - i.e. if track is forward or backward
	  int rindex=(tracklet->fpgarprojdisk(disk_).value()>>(tracklet->fpgarprojdisk(disk_).nbits()-nrbits_))&((1<<nrbits_)-1);

	  int phiderindex=(tracklet->fpgaphiprojderdisk(disk_).value()>>(tracklet->fpgaphiprojderdisk(disk_).nbits()-nphiderbits_))&((1<<nphiderbits_)-1);

	  int signindex=(tracklet->fpgarprojderdisk(disk_).value()<0);

	  int bendindex=(signindex<<(nphiderbits_+nrbits_))+
	    (rindex<<(nphiderbits_))+
	    phiderindex;
	  
	  int ibendproj=bendTable(abs(disk_)-1,bendindex);

	  tracklet->setBendIndex(ibendproj,disk_);

	  addVMProj(vmprojs_[iphi],tracklet,index);
	  
	}
      }
    }

    //cout << "Done in "<<getName()<<endl;
    
    if (writeAllProjections) {
      static ofstream out("allprojections.txt"); 
      out << getName() << " " << allproj_->nTracklets() << endl;
    } 
   

    if (writeVMProjections) {
      static ofstream out("vmprojections.txt");
      for (unsigned int i=0;i<8;i++) {
	if (vmprojs_[i]!=0) {
	  out << vmprojs_[i]->getName() << " " << vmprojs_[i]->nTracklets() << endl;
	}
      }
    }
  }

  double bend(double r, double rinv) {
    
    double dr=0.18;
    
    double delta=r*dr*0.5*rinv;
    
    double bend=-delta/0.009;
    if (r<55.0) bend=-delta/0.01;
    
    return bend;
    
  }
  
  int bendTable(int diskindex,int bendindex) {

    static vector<int> bendtable[5];

    static bool first=true;

    if (first) {
      first=false;
    
      for (unsigned int idisk=0;idisk<5;idisk++) {

	unsigned int nsignbins=2;
	unsigned int nrbins=1<<(nrbits_);
	unsigned int nphiderbins=1<<(nphiderbits_);
      
	for(unsigned int isignbin=0;isignbin<nsignbins;isignbin++) {
	  for(unsigned int irbin=0;irbin<nrbins;irbin++) {
	    int ir=irbin;
	    if (ir>(1<<(nrbits_-1))) ir-=(1<<nrbits_);
	    ir=ir<<(nrbitsprojdisk-nrbits_);
	    for(unsigned int iphiderbin=0;iphiderbin<nphiderbins;iphiderbin++) {
	      int iphider=iphiderbin;
	      if (iphider>(1<<(nphiderbits_-1))) iphider-=(1<<nphiderbits_);
	      iphider=iphider<<(nbitsphiprojderL123-nphiderbits_);
	      
	      double rproj=ir*krprojshiftdisk;
	      double phider=iphider*TrackletCalculator::ITC_L1L2.der_phiD_final.get_K();
	      double t=zmean[idisk]/rproj;
	      
	      if (isignbin) t=-t;
	  
	      double rinv=-phider*(2.0*t);

	      double bendproj=0.5*bend(rproj,rinv);

	    
	      int ibendproj=2.0*bendproj+15.5;
	      if (ibendproj<0) ibendproj=0;
	      if (ibendproj>31) ibendproj=31;
	      
	      bendtable[idisk].push_back(ibendproj);

	    }
	  }
	}
      }
    }

    

    return bendtable[diskindex][bendindex];

  }
  
  void addVMProj(VMProjectionsMemory* vmproj,Tracklet* tracklet, unsigned int index){

    assert(vmproj!=0);
    if (debug1){
      cout << "ProjectionRouter "<<getName()<<" add projection to : "<<vmproj->getName()<<endl;
    }
    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    vmproj->addTracklet(tracklet,index);
  }

  
private:

  int layer_; 
  int disk_; 

  int nrbits_;
  int nphiderbits_;

  
  vector<TrackletProjectionsMemory*> inputproj_;

  AllProjectionsMemory* allproj_;
  std::vector<VMProjectionsMemory*> vmprojs_;


};

#endif
