//This class implementes the VM router
#ifndef VMROUTER_H
#define VMROUTER_H

#include "ProcessBase.h"
#include "VMStubTE.h"
#include "TETableOuter.h"
#include "TETableInner.h"
#include "TETableOuterDisk.h"
#include "TETableInnerDisk.h"
#include "TETableInnerOverlap.h"

using namespace std;

class VMRouter:public ProcessBase{

public:

  VMRouter(string name, unsigned int iSector):
    ProcessBase(name,iSector){

    layer_=0;
    disk_=0;
    
    if (name_[4]=='L') layer_=name_[5]-'0';    
    if (name_[4]=='D') disk_=name_[5]-'0';    

    assert((layer_!=0)||(disk_!=0));

    if (layer_!=0) {
      nbitsfinebintable_=8;
      unsigned int nbins=1<<nbitsfinebintable_;
      
      for(unsigned int i=0;i<nbins;i++) {
	finebintable_.push_back(-1);
      }
      
      for(unsigned int i=0;i<nbins;i++) {

	
	int ibin=(i>>(nbitsfinebintable_-3));
	
	int zfine=(i>>(nbitsfinebintable_-6))-(ibin<<3);
	
	//awkward bit manipulations since the index is from a signed number...
	int index=i+(1<<(nbitsfinebintable_-1));
	
	if (index>=(1<<nbitsfinebintable_)){
	  index-=(1<<nbitsfinebintable_);
	}
	
	finebintable_[index]=zfine;
	
      }


      if (iSector_==0&&writeVMRTables) {

	ofstream outfinebin;
	outfinebin.open(getName()+"_finebin.txt");
	outfinebin << "{"<<endl;
	for(unsigned int i=0;i<nbins;i++) {
	  if (i!=0) outfinebin<<","<<endl;
	  outfinebin << finebintable_[i];
	}
	outfinebin <<endl<<"};"<<endl;
	outfinebin.close();
	
      }

    }

    if (disk_!=0) {

      nbitsfinebintable_=8;
      unsigned int nbins=1<<nbitsfinebintable_;
      
      for(unsigned int i=0;i<nbins;i++) {

	double rstub=0.0;
	
	if (i<10) {
	  if (disk_<=2) {
	    rstub=rDSSinner[i];
	  } else {
	    rstub=rDSSouter[i];
	  }
	} else {
	  rstub=kr*(i<<(nrbitsdisk-nbitsfinebintable_));
	}

	if (rstub<rmindiskvm) {
	  finebintable_.push_back(-1);
	} else {	
	  int bin=8.0*(rstub-rmindiskvm)/(rmaxdisk-rmindiskvm);
	  assert(bin>=0);
	  assert(bin<MEBinsDisks);	  
	  int rfine=64*((rstub-rmindiskvm)-bin*(rmaxdisk-rmindiskvm)/8.0)/(rmaxdisk-rmindiskvm);
	  finebintable_.push_back(rfine);
	}
      }
    }
  }
   
  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output.substr(0,10)=="allstubout"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      allstubs_.push_back(tmp);
      return;
    }
    for (int i=0;i<32;i++) {
      std::ostringstream oss;
      oss<<(i+1);
      string s=oss.str();
      for (int n=0;n<20;n++) {
	std::ostringstream oss1;
	oss1<<(n+1);
	string ns=oss1.str();	
	if (
	    output=="vmstuboutPHIA"+s+"n"+ns||
	    output=="vmstuboutPHIB"+s+"n"+ns||
	    output=="vmstuboutPHIC"+s+"n"+ns||
	    output=="vmstuboutPHID"+s+"n"+ns||
	    output=="vmstuboutPHIE"+s+"n"+ns||
	    output=="vmstuboutPHIF"+s+"n"+ns||
	    output=="vmstuboutPHIG"+s+"n"+ns||
	    output=="vmstuboutPHIH"+s+"n"+ns||
	    output=="vmstuboutPHII"+s+"n"+ns||
	    output=="vmstuboutPHIJ"+s+"n"+ns||
	    output=="vmstuboutPHIK"+s+"n"+ns||
	    output=="vmstuboutPHIL"+s+"n"+ns||
	    output=="vmstuboutPHIX"+s+"n"+ns||
	    output=="vmstuboutPHIY"+s+"n"+ns||
	    output=="vmstuboutPHIZ"+s+"n"+ns||
	    output=="vmstuboutPHIW"+s+"n"+ns||
	    output=="vmstuboutPHIQ"+s+"n"+ns||
	    output=="vmstuboutPHIR"+s+"n"+ns||
	    output=="vmstuboutPHIS"+s+"n"+ns||
	    output=="vmstuboutPHIT"+s+"n"+ns||
	    output=="vmstuboutPHIa"+s+"n"+ns||
	    output=="vmstuboutPHIb"+s+"n"+ns||
	    output=="vmstuboutPHIc"+s+"n"+ns||
	    output=="vmstuboutPHId"+s+"n"+ns||
	    output=="vmstuboutPHIe"+s+"n"+ns||
	    output=="vmstuboutPHIf"+s+"n"+ns||
	    output=="vmstuboutPHIg"+s+"n"+ns||
	    output=="vmstuboutPHIh"+s+"n"+ns||
	    output=="vmstuboutPHIx"+s+"n"+ns||
	    output=="vmstuboutPHIy"+s+"n"+ns||
	    output=="vmstuboutPHIz"+s+"n"+ns||
	    output=="vmstuboutPHIw"+s+"n"+ns||
	    output=="vmstuboutPHIq"+s+"n"+ns||
	    output=="vmstuboutPHIr"+s+"n"+ns||
	    output=="vmstuboutPHIs"+s+"n"+ns||
	    output=="vmstuboutPHIt"+s+"n"+ns
	  ){

	  if (memory->getName().substr(3,2)=="TE") {
	    VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
	    assert(tmp!=0);
	    if (memory->getName().substr(11,1)[0]<'I') {
	      vmstubsTEPHI_[i].push_back(tmp);
	    } else if (memory->getName().substr(11,1)[0]<'M') {
	      vmstubsTEExtraPHI_[i].push_back(tmp);
	    } else if (memory->getName().substr(11,1)[0]<='Z') {
	      vmstubsTEOverlapPHI_[i].push_back(tmp);
	    } else if (memory->getName().substr(11,1)[0]<'o' && memory->getName().substr(11,1)[0]>='a') {
	      vmstubsTEExtendedPHI_[i].push_back(tmp);
	    } else if (memory->getName().substr(11,1)[0]>'o' && memory->getName().substr(11,1)[0]<='z'){
	      vmstubsTEOverlapExtendedPHI_[i].push_back(tmp);
	    } else {
	      assert(0);
	    } 
	  } else if (memory->getName().substr(3,2)=="ME") {
	    VMStubsMEMemory* tmp=dynamic_cast<VMStubsMEMemory*>(memory);
	    assert(tmp!=0);
	    vmstubsMEPHI_[i].push_back(tmp);
	  } else {
	    assert(0);
	  }
	  
	  return;
	}
      }
    }
    cout << "Could not find : "<<output<<endl;
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="stubin"){
      InputLinkMemory* tmp1=dynamic_cast<InputLinkMemory*>(memory);
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

    assert(allstubs_.size()!=0);

    
    unsigned int count=0;
    for(unsigned int j=0;j<stubinputs_.size();j++){
      for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	if (count>MAXVMROUTER) continue;
	std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);
		
	stub.first->setAllStubIndex(count);
	stub.second->setAllStubIndex(count);

	for (unsigned int l=0;l<allstubs_.size();l++){
	  allstubs_[l]->addStub(stub);
	}
	
	count++;
      }
    }	


    if (writeAllStubs) {
      static ofstream out("allstubs.txt");
      out<<allstubs_[0]->getName()<<" "<<allstubs_[0]->nStubs()<<endl;
    }

    
    if (disk_!=5) {
      executeTE(false);
    }
    if (layer_==1||layer_==2||disk_==1) {
      executeTE(true);
    }

    if (extended_) {
      if (layer_==2||layer_==3) //needed for L2L3D1 since for it L2 is inner and L3 is outer
        executeTEextended(false);
      if (layer_==2||disk_==1)   //needed for D1 in L2L3D1 and L2 in D1D2L2 (possibility to have a different VM granularity) 
        executeTEextended(true);
    }
    
    executeME();
  }


  
  
  void executeTE(bool overlap){

    
    //cout << "In FPGAVMRouterTE"<<endl;
    
    assert(stubinputs_.size()!=0);

    unsigned int count=0;

 
    if (layer_!=0){  //First handle layer stubs

      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  count++;
	  if (count>MAXVMROUTER) continue;
	  std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  int iphiRaw=stub.first->iphivmRaw();

	  bool insert=false;


	  FPGAWord binlookup;
	  FPGAWord binlookupextra;
	  if (overlap) {
	    assert(layer_==1||layer_==2);
	    binlookup=lookupInnerOverlapLayer(stub.first);
	  } else {
	    switch (layer_) {
	    case 2 : binlookup=lookupOuterLayer(stub.first);
	      binlookupextra=lookupInnerLayer(stub.first);
	      break;
	    case 4 : binlookup=lookupOuterLayer(stub.first);
	      break;
	    case 6 : binlookup=lookupOuterLayer(stub.first);
	      break;
	    case 1 : binlookup=lookupInnerLayer(stub.first);
	      break;
	    case 3 : binlookup=lookupInnerLayer(stub.first);
	      binlookupextra=lookupOuterLayer(stub.first);
	      break;
	    case 5 : binlookup=lookupInnerLayer(stub.first);
	      break;
	    default : assert(0);
	    }
	  }
	  unsigned int layer=stub.first->layer().value();
	  if ((layer==1 || layer== 2) && binlookupextra.value()!=-1 ) {
	    int iphiRawTmp=iphiRaw/(32/(nallstubslayers[layer]*nvmteextralayers[layer]));
	    for (unsigned int l=0;l<vmstubsTEExtraPHI_[iphiRawTmp].size();l++){
	      if (debug1) {
		cout << getName()<<" try adding extra stub to "<<vmstubsTEExtraPHI_[iphiRawTmp][l]->getName()<<endl;
	      }
	      int nfinephi=nfinephibarrelinner;
	      int nphireg=4;
	      if (layer==2) {
		nfinephi=nfinephibarrelouter;
	      }
	      VMStubTE tmpstub(stub,
			       stub.first->iphivmFineBins(nphireg,nfinephi),
			       stub.first->finer(),
			       stub.first->bend(),
			       binlookupextra,
			       stub.first->stubindex());
	      vmstubsTEExtraPHI_[iphiRawTmp][l]->addVMStub(tmpstub); 
	      insert=true;
	    }
	  }
	  
	  if (binlookup.value()!=-1) {
	    if (overlap) {
	      int iphiRawTmp=iphiRaw/(32/(nallstubsoverlaplayers[layer]*nvmteoverlaplayers[layer]));
	      for (unsigned int l=0;l<vmstubsTEOverlapPHI_[iphiRawTmp].size();l++){
		if (debug1) {
		  cout << getName()<<" try adding overlap stub to "<<vmstubsTEOverlapPHI_[iphiRawTmp][l]->getName()<<endl;
		}
		int nphireg=4;
		if (layer==1) nphireg=3; //layer 2
		VMStubTE tmpstub(stub,
				 stub.first->iphivmFineBins(nphireg,nfinephioverlapinner),
				 stub.first->finez(),
				 stub.first->bend(),
				 binlookup,
				 stub.first->stubindex());
		vmstubsTEOverlapPHI_[iphiRawTmp][l]->addVMStub(tmpstub); 
		insert=true;
	      }
	    } else {
	      int iphiRawTmp=iphiRaw/(32/(nallstubslayers[layer]*nvmtelayers[layer]));
	      for (unsigned int l=0;l<vmstubsTEPHI_[iphiRawTmp].size();l++){
		if (debug1) {
		  cout << getName()<<" try adding stub to "<<vmstubsTEPHI_[iphiRawTmp][l]->getName()<<endl;
		}
		int nfinephi=nfinephibarrelinner;
		if (layer_==2||layer_==4||layer_==6) {
		  nfinephi=nfinephibarrelouter;
		}
		int nphireg=5;
		if (layer_==3) nphireg=4;
		if (layer_==5) nphireg=4;
		VMStubTE tmpstub(stub,
				 stub.first->iphivmFineBins(nphireg,nfinephi),
				 stub.first->finer(),
				 stub.first->bend(),
				 //(binlookup&1023),
				 binlookup,
				 stub.first->stubindex());
		vmstubsTEPHI_[iphiRawTmp][l]->addVMStub(tmpstub); 

		insert=true;
	      }
	    }
	  }
	  
	  if (false) {
	    if (!insert) {
	      cout << getName()<<" did not insert stub"<<endl;
	    }
	    assert(insert);
	  }
	}
      }

    }

    if (disk_!=0) {
      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  if (!stub.second->isPSmodule()) {
	    if (debug1) {
	      cout << getName() <<" stub at r = "<<stub.second->r()<<" is 2S module"<<endl;
	    }
	    continue;
	  }
	  
	  int iphiRaw=stub.first->iphivmRaw();

	  bool insert=false;


	  if (overlap) {
	    
	    FPGAWord binlookup=lookupOuterOverlapD1(stub.first);
	    assert(binlookup.value()>=0);

	    iphiRaw=iphiRaw/(32/(nallstubsoverlapdisks[0]*nvmteoverlapdisks[0]));

	    for (unsigned int l=0;l<vmstubsTEOverlapPHI_[iphiRaw].size();l++){
	      if (debug1) {
		cout << getName()<<" added stub to : "<<vmstubsTEOverlapPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      VMStubTE tmpstub(stub,
			       stub.first->iphivmFineBins(4,nfinephioverlapouter),
			       stub.first->finer(),
			       stub.first->bend(),
			       binlookup,
			       stub.first->stubindex());
	      vmstubsTEOverlapPHI_[iphiRaw][l]->addVMStub(tmpstub); 
	      insert=true;
	    }
	      
	  } else {
	    // disk-disk seeding
	    
	    FPGAWord binlookup;

	    if (!overlap) {
	      switch (disk_) {
	      case 2 : binlookup=lookupOuterDisk(stub.first);
		break;
	      case 4 : binlookup=lookupOuterDisk(stub.first);
		break;
	      case 1 : binlookup=lookupInnerDisk(stub.first,extended_);
		break;
	      case 3 : binlookup=lookupInnerDisk(stub.first);
		break;
	      default : assert(0);  
	      }
	    } else {
	      assert(disk_==1);
	      binlookup=lookupOuterOverlapD1(stub.first);
	    }
            
	    if (binlookup.value()==-1) continue;

              
	    unsigned int disk=abs(stub.first->disk().value());
	    iphiRaw=iphiRaw/(32/(nallstubsdisks[disk-1]*nvmtedisks[disk-1]));

	    for (unsigned int l=0;l<vmstubsTEPHI_[iphiRaw].size();l++){
	      if (debug1) {
		cout << getName()<<" added stub to : "<<vmstubsTEPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      int nfinephi=nfinephidiskinner;
	      if (disk_==2||disk_==4) {
		nfinephi=nfinephidiskouter;
	      }
	      int nphireg=4;
	      VMStubTE tmpstub(stub,
			       stub.first->iphivmFineBins(nphireg,nfinephi),
			       stub.first->finer(),
			       stub.first->bend(),
			       //(binlookup&1023),
			       binlookup,
			       stub.first->stubindex());
	      vmstubsTEPHI_[iphiRaw][l]->addVMStub(tmpstub); 
	      insert=true;
	    }
	    
	  }

	  if (!insert) {
	    cout << getName() << " did not insert stub"<<endl;
	  }
	  assert(insert);

	}
      }
    }


    if (writeVMOccupancyTE) {
      static ofstream out("vmoccupancyte.txt");
      
      for (int i=0;i<24;i++) {
	if (vmstubsTEPHI_[i].size()!=0) {
	  //for(int j=0;j<vmstubsTEPHI_[i].size();j++){
	  //  out<<vmstubsTEPHI_[i][j]->getName()<<" "<<vmstubsTEPHI_[i][j]->nStubs()<<endl;
	  //}
	  out<<vmstubsTEPHI_[i][0]->getName()<<" "<<vmstubsTEPHI_[i][0]->nVMStubs()<<endl;
	}
      }
    }
    
    //cout << "Done in FPGAVMRouterTE"<<endl;
    
  }
  
  void executeTEextended(bool overlap){  
    //cout << "In FPGAVMRouterTEextended "<<overlap<<endl;
    
    assert(stubinputs_.size()!=0);
    unsigned int count=0;

    if (layer_!=0){  //First handle layer stubs
      for(unsigned int j=0;j<stubinputs_.size();j++){
       	//cout<<"TEextended: layer "<<layer_<<" "<<j<<" "<<stubinputs_[j]->getName()<<"\t"<<stubinputs_[j]->nStubs()<<"\n";
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  count++;
	  if (count>MAXVMROUTER) continue;
	  std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  int iphiRaw=stub.first->iphivmRaw();
	  bool insert=false;

	  FPGAWord binlookup;
	  if (overlap) {
	    assert(layer_==2);//D1D2L2
	    binlookup=lookupOuterLayer(stub.first);//no special selection, just an outer stub with overlap VM segmentation.
	  } else {
	    switch (layer_) {
	    case 3 : binlookup=lookupOuterLayer(stub.first);
	      break;
	    case 2 : binlookup=lookupInnerLayer(stub.first,true);
	      break;
	    default : assert(0);
	    }
	  }

	  if (binlookup.value()==-1) continue;
	  
	  unsigned int layer=stub.first->layer().value();
	  if (overlap) {
	    iphiRaw=iphiRaw/(32/(nallstubsoverlaplayers[layer]*nvmteoverlaplayers[layer]));
	    for (unsigned int l=0;l<vmstubsTEOverlapExtendedPHI_[iphiRaw].size();l++){
	      int nfinephi=nfinephibarrelinner;
	      if (layer_==2) {
		nfinephi=nfinephibarrelouter;
	      }
	      int nphireg=4;
	      VMStubTE tmpstub(stub,
			       stub.first->iphivmFineBins(nphireg,nfinephi),
			       stub.first->finer(),
			       stub.first->bend(),
			       //(binlookup&1023),
			       binlookup,
			       stub.first->stubindex());
	      vmstubsTEOverlapExtendedPHI_[iphiRaw][l]->addVMStub(tmpstub); 

	      if (debug1) {
		cout << getName()<<" adding stub to "<<vmstubsTEOverlapExtendedPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      insert=true;
	    }
	  } else {
	    iphiRaw=iphiRaw/(32/(nallstubslayers[layer]*nvmtelayers[layer]));
	    for (unsigned int l=0;l<vmstubsTEExtendedPHI_[iphiRaw].size();l++){
	      int nfinephi=3;
	      int nphireg=4;
	      if (layer_==2) {
		nfinephi=2;
		nphireg=5;
	      }
	      VMStubTE tmpstub(stub,
			       stub.first->iphivmFineBins(nphireg,nfinephi),
			       stub.first->finer(),
			       stub.first->bend(),
			       //(binlookup&1023),
			       binlookup,
			       stub.first->stubindex());
	      vmstubsTEExtendedPHI_[iphiRaw][l]->addVMStub(tmpstub); 
	      if (debug1) {
		cout << getName()<<" adding stub to "<<vmstubsTEExtendedPHI_[iphiRaw][l]->getName()<<endl;
	      }
	      insert=true;
	    }
	  }

	  //cout<<"inserted\n";
	  
	  if (!insert) {
	    cout << getName()<<" did not insert stub"<<endl;
	  }
	  assert(insert);
	}
      }

    }

    if (disk_!=0) {
      //only get here for overlap L2L3D1
      assert(overlap);
      assert(disk_==1);
      
      for(unsigned int j=0;j<stubinputs_.size();j++){
       	//cout<<"TEextended: disk "<<disk_<<" "<<j<<" "<<stubinputs_[j]->getName()<<"\t"<<stubinputs_[j]->nStubs()<<"\n";
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  if(stub.second->r() < rmindiskl3overlapvm) continue;
	  
	  int iphiRaw=stub.first->iphivmRaw();
	  bool insert=false;

	  //
	  // a hack instead of a LUT. fine R is module index+1 for 2S and 0 for PS
	  // fine for now as it's only used as a third stub in a triplet
	  int binlookup=stub.first->ir();
	  assert(binlookup>=0);
	  binlookup = binlookup+1;
	  if(stub.first->isPSmodule())
	    binlookup = 0;
	  
	  iphiRaw=iphiRaw/(32/(nallstubsoverlapdisks[0]*nvmteoverlapdisks[0]));
	  
	  for (unsigned int l=0;l<vmstubsTEOverlapExtendedPHI_[iphiRaw].size();l++){
	    if (debug1) {
	      cout << getName()<<" added stub to : "<<vmstubsTEOverlapExtendedPHI_[iphiRaw][l]->getName()<<endl;
	    }
	    int nfinephi=nfinephioverlapouter;
	    int nphireg=4;
	    VMStubTE tmpstub(stub,
			     stub.first->iphivmFineBins(nphireg,nfinephi),
			     stub.first->finer(),
			     stub.first->bend(),
			     //(binlookup&1023),
			     FPGAWord(binlookup,4,true,__LINE__,__FILE__),
			     stub.first->stubindex());
	    vmstubsTEOverlapExtendedPHI_[iphiRaw][l]->addVMStub(tmpstub); 

	    insert=true;
	  }

	  if (!insert) {
	    cout << getName() << " did not insert stub"<<endl;
	  }
	  assert(insert);

	}
      }
    }


    if (writeVMOccupancyTE) {
      static ofstream out("vmoccupancyteextended.txt");
      
      for (int i=0;i<24;i++) {
	if (vmstubsTEExtendedPHI_[i].size()!=0) {
	  out<<vmstubsTEExtendedPHI_[i][0]->getName()<<" "<<vmstubsTEExtendedPHI_[i][0]->nVMStubs()<<endl;
	}
      }
    }
    
    //cout << "Done in FPGAVMRouterTEextended"<<endl;    
  }


  void executeME(){

    //cout << "In FPGAVMRouterME "<<getName()<<" "<<stubinputs_.size()<<endl;

    
    unsigned int count=0;

    if (stubinputs_.size()!=0){
      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  count++;
	  if (count>MAXVMROUTER) continue;
	  std::pair<Stub*,L1TStub*> stub=stubinputs_[j]->getStub(i);

	  int iphiRaw=stub.first->iphivmRaw();
	  int iphiRawPlus=stub.first->iphivmRawPlus();
	  int iphiRawMinus=stub.first->iphivmRawMinus();

	  int iphistub=iphiRaw;

	  int layer=stub.first->layer().value();
	  int disk=abs(stub.first->disk().value());

	  //cout << getName()<<" layer,disk : "<<layer<<" "<<disk<<endl;
	    
	  int nvm=-1;
	  if (layer!=-1) {
	    nvm=nallstubslayers[layer]*nvmmelayers[layer];
	  }
	  if (disk!=0){
	    nvm=nallstubsdisks[disk-1]*nvmmedisks[disk-1];
	  }
	  assert(nvm>0&&nvm<=32);
	  iphiRaw=iphiRaw/(32/nvm);
	  iphiRawPlus=iphiRawPlus/(32/nvm);
	  iphiRawMinus=iphiRawMinus/(32/nvm);
	  if (iphiRawPlus<0) iphiRawPlus=0;
	  if (iphiRawPlus>=nvm) iphiRawPlus=nvm-1;
	  if (iphiRawMinus<0) iphiRawMinus=0;
	  if (iphiRawMinus>=nvm) iphiRawMinus=nvm-1;
	    
	  if (disk_!=0) {

	    int index=stub.first->r().value();
	    if (stub.first->isPSmodule()){
	      index=stub.first->r().value()>>(stub.first->r().nbits()-nbitsfinebintable_);
	    }

	    int rfine=finebintable_[index];

	    assert(rfine>=0);
	    
	    stub.first->setfiner(rfine);

	  }

	  if (layer_!=0) {

	    //Take the top nbitsfinebintable_ bits of the z coordinate. The & is to handle the negative
	    //z values.
	    int index=(stub.first->z().value()>>(stub.first->z().nbits()-nbitsfinebintable_))&((1<<nbitsfinebintable_)-1);
	    
	    int zfine=finebintable_[index];

	    stub.first->setfinez(zfine);

	  }
	  
	  bool insert=false;
	  
	  
	  for (unsigned int l=0;l<vmstubsMEPHI_[iphiRaw].size();l++){
	    if (debug1) {
	      cout << "FPGAVMRouterME "<<getName()<<" add stub ( r = "<<stub.second->r()<<" phi = "<<stub.second->phi()<<" ) in : "<<vmstubsMEPHI_[iphiRaw][l]->getName()<<" iphistub = " << iphistub << " iphivmRaw Minus Plus "<<stub.first->iphivmRaw()<<" "<<stub.first->iphivmRawMinus()<<" "<<stub.first->iphivmRawPlus()<<" bins "
		   <<iphiRawMinus<<" "<<iphiRawPlus<<endl;
	    }
	    vmstubsMEPHI_[iphiRaw][l]->addStub(stub);
	    insert=true;
	  }

	  if (iphiRaw!=iphiRawPlus) {
	    for (unsigned int l=0;l<vmstubsMEPHI_[iphiRawPlus].size();l++){
	      vmstubsMEPHI_[iphiRawPlus][l]->addStub(stub);
	    }
	  }
	  if (iphiRaw!=iphiRawMinus) {
	    for (unsigned int l=0;l<vmstubsMEPHI_[iphiRawMinus].size();l++){
	      vmstubsMEPHI_[iphiRawMinus][l]->addStub(stub);
	    }
	  }
	 

	  if (!insert){
	    cout << "In "<<getName()<<" did not insert stub from input "<<stubinputs_[j]->getName()<<endl;
	  }
	  assert(insert);

	}
      }
      
      if (writeVMOccupancyME) {
	static ofstream out("vmoccupancyme.txt");

	for (int i=0;i<24;i++) {
	  if (vmstubsMEPHI_[i].size()!=0) {
	    out<<vmstubsMEPHI_[i][0]->getName()<<" "<<vmstubsMEPHI_[i][0]->nStubs();
	    for (unsigned int ibin=0;ibin<MEBinsDisks*2;ibin++){
	      out <<" "<<vmstubsMEPHI_[i][0]->nStubsBin(ibin);
	    }
	    out<<endl;
	  }
	}

      }

    }


  }

  FPGAWord lookupOuterOverlapD1(Stub* stub){

    assert(disk_==1);
    
    static TETableOuterDisk outerTableOverlapD1;
    static bool first=true;

    if (first) {
      outerTableOverlapD1.init(1,7,3);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int rbin=(r.value())>>(r.nbits()-7);
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-3);
    bool negdisk=stub->disk().value()<0;
    if (negdisk) zbin=7-zbin; //Should this be separate table?
    return FPGAWord(outerTableOverlapD1.lookup(rbin,zbin),10,true,__LINE__,__FILE__);

  }


  FPGAWord lookupOuterDisk(Stub* stub){

    assert(disk_==2||disk_==4);
    
    static TETableOuterDisk outerTableD2;
    static TETableOuterDisk outerTableD4;
    static bool first=true;

    if (first) {
      outerTableD2.init(2,7,3);
      outerTableD4.init(4,7,3);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int rbin=(r.value())>>(r.nbits()-7);
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-3);
    bool negdisk=stub->disk().value()<0;
    if (negdisk) zbin=7-zbin; //Should this be separate table?
    switch (disk_){
    case 2: return FPGAWord(outerTableD2.lookup(rbin,zbin),10,true,__LINE__,__FILE__);
      break;
    case 4: return FPGAWord(outerTableD4.lookup(rbin,zbin),10,true,__LINE__,__FILE__);
      break;
    }
    assert(0);
  }


  FPGAWord lookupInnerDisk(Stub* stub, const bool extended = false){

    assert(disk_==1||disk_==3);
    
    static TETableInnerDisk innerTableD1;
    static TETableInnerDisk innerTableD1_extended;
    static TETableInnerDisk innerTableD3;
    static bool first=true;

    int nrbits=8;
    
    if (first) {
      innerTableD1.init(1,2,-1,nrbits,3);
      innerTableD1_extended.init(1,2,2,nrbits,3);
      innerTableD3.init(3,4,-1,nrbits,3);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int rbin=(r.value())>>(r.nbits()-nrbits);
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-3);
    bool negdisk=stub->disk().value()<0;
    if (negdisk) zbin=7-zbin; //Should this be separate table?
    switch (disk_){
    case 1:
      if (!extended)
        return FPGAWord(innerTableD1.lookup(rbin,zbin),21,false,__LINE__,__FILE__);
      else
        return FPGAWord(innerTableD1_extended.lookup(rbin,zbin),21,false,__LINE__,__FILE__);
      break;
    case 3: return FPGAWord(innerTableD3.lookup(rbin,zbin),21,false,__LINE__,__FILE__);
      break;
    }
    assert(0);
  }

  FPGAWord lookupOuterLayer(Stub* stub){

    assert(layer_==2||layer_==3||layer_==4||layer_==6);
    
    static TETableOuter outerTableL2;
    static TETableOuter outerTableL3;
    static TETableOuter outerTableL4;
    static TETableOuter outerTableL6;
    static bool first=true;

    if (first) {
      outerTableL2.init(2,7,4);
      outerTableL3.init(3,7,4);
      outerTableL4.init(4,7,4);
      outerTableL6.init(6,7,4);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-7);
    int rbin=(r.value()+(1<<(r.nbits()-1)))>>(r.nbits()-4);
    switch (layer_){
    case 2: return FPGAWord(outerTableL2.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    case 3: return FPGAWord(outerTableL3.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    case 4: return FPGAWord(outerTableL4.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    case 6: return FPGAWord(outerTableL6.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    }
    assert(0);
  }


  FPGAWord lookupInnerLayer(Stub* stub, const bool extended = false){

    assert(layer_==1||layer_==2||layer_==3||layer_==5);
    
    static TETableInner innerTableL1;
    static TETableInner innerTableL2;
    static TETableInner innerTableL2_extended;
    static TETableInner innerTableL3;
    static TETableInner innerTableL5;
    static bool first=true;

    if (first) {
      innerTableL1.init(1,2,-1,7,4);
      innerTableL2.init(2,3,-1,7,4);
      innerTableL2_extended.init(2,3,1,7,4,true);
      innerTableL3.init(3,4,2,7,4);
      innerTableL5.init(5,6,4,7,4);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-7);
    int rbin=(r.value()+(1<<(r.nbits()-1)))>>(r.nbits()-4);
    switch (layer_){
    case 1: return FPGAWord(innerTableL1.lookup(zbin,rbin),21,false,__LINE__,__FILE__);
      break;
    case 2:
      if (!extended)
        return FPGAWord(innerTableL2.lookup(zbin,rbin),21,false,__LINE__,__FILE__);
      else
        return FPGAWord(innerTableL2_extended.lookup(zbin,rbin),21,false,__LINE__,__FILE__);
      break;
    case 3: return FPGAWord(innerTableL3.lookup(zbin,rbin),21,false,__LINE__,__FILE__);
      break;
    case 5: return FPGAWord(innerTableL5.lookup(zbin,rbin),21,false,__LINE__,__FILE__);
      break;
    }
    assert(0);
  }


  FPGAWord lookupInnerOverlapLayer(Stub* stub){

    assert(layer_==1||layer_==2);
    
    static TETableInnerOverlap innerTableOverlapL1;
    static TETableInnerOverlap innerTableOverlapL2;
    static bool first=true;

    if (first) {
      innerTableOverlapL1.init(1,1,7,3);
      innerTableOverlapL2.init(2,1,7,3);
      first=false;
    }
    
    FPGAWord r=stub->r();
    FPGAWord z=stub->z();
    int zbin=(z.value()+(1<<(z.nbits()-1)))>>(z.nbits()-7);
    int rbin=(r.value()+(1<<(r.nbits()-1)))>>(r.nbits()-3);
    switch (layer_){
    case 1: return FPGAWord(innerTableOverlapL1.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    case 2: return FPGAWord(innerTableOverlapL2.lookup(zbin,rbin),11,false,__LINE__,__FILE__);
      break;
    }
    assert(0);
  }


  

private:

  int layer_;
  int disk_;

  int nbitsfinebintable_;
  vector<int> finebintable_;

  vector<InputLinkMemory*> stubinputs_;
  vector<AllStubsMemory*> allstubs_;

  vector<VMStubsTEMemory*> vmstubsTEPHI_[32];
  vector<VMStubsTEMemory*> vmstubsTEExtraPHI_[32];
  vector<VMStubsTEMemory*> vmstubsTEOverlapPHI_[32];
  vector<VMStubsTEMemory*> vmstubsTEExtendedPHI_[32];
  vector<VMStubsTEMemory*> vmstubsTEOverlapExtendedPHI_[32];
  vector<VMStubsMEMemory*> vmstubsMEPHI_[32];


};

#endif

