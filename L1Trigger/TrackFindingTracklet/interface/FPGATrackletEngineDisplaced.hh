//This class implementes the tracklet engine
#ifndef FPGATRACKLETENGINEDISPLACED_H
#define FPGATRACKLETENGINEDISPLACED_H

#include "FPGAProcessBase.hh"


using namespace std;

class FPGATrackletEngineDisplaced:public FPGAProcessBase{

public:

  FPGATrackletEngineDisplaced(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=2*M_PI/NSector;
    phimin_=iSector*dphi;
    phimax_=phimin_+dphi;
    if (phimin_>M_PI) phimin_-=2*M_PI;
    if (phimax_>M_PI) phimax_-=2*M_PI;
    if (phimin_>phimax_)  phimin_-=2*M_PI;
    assert(phimax_>phimin_);
    stubpairs_.clear();
    firstvmstubs_.clear();
    secondvmstubs_=0;
    layer1_=0;
    layer2_=0;
    disk1_=0;
    disk2_=0;
    string name1 = name.substr(1);//this is to correct for "TED" having one more letter then "TE"
    if (name1[3]=='L') {
      layer1_=name1[4]-'0';
    }
    if (name1[3]=='D') {
      disk1_=name1[4]-'0';
    }
    if (name1[11]=='L') {
      layer2_=name1[12]-'0';
    }
    if (name1[11]=='D') {
      disk2_=name1[12]-'0';
    }
    if (name1[12]=='L') {
      layer2_=name1[13]-'0';
    }
    if (name1[12]=='D') {
      disk2_=name1[13]-'0';
    }

    iSeed_ = -1;
    if (layer1_ == 3 && layer2_ == 4) iSeed_ = 8;
    if (layer1_ == 5 && layer2_ == 6) iSeed_ = 9;
    if (layer1_ == 2 && layer2_ == 3) iSeed_ = 10;
    if (disk1_ == 1 && disk2_ == 2) iSeed_ = 11;
    
    firstphibits_=-1;
    secondphibits_=-1;
    
    if ((layer1_==3 && layer2_==4)||
	(layer1_==5 && layer2_==6)||
	(layer1_==2 && layer2_==3)){
      firstphibits_=nfinephibarrelinner;
      secondphibits_=nfinephibarrelouter;
    }
    if (disk1_==1 && disk2_==2){
      firstphibits_=nfinephidiskinner;
      secondphibits_=nfinephidiskouter;
    }

    readTables();
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="stubpairout") {
      FPGAStubPairs* tmp=dynamic_cast<FPGAStubPairs*>(memory);
      assert(tmp!=0);
      stubpairs_.push_back(tmp);
      return;
    }
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="firstvmstubin") {
      FPGAVMStubsTE* tmp=dynamic_cast<FPGAVMStubsTE*>(memory);
      assert(tmp!=0);
      firstvmstubs_.push_back(tmp);
      return;
    }
    if (input=="secondvmstubin") {
      FPGAVMStubsTE* tmp=dynamic_cast<FPGAVMStubsTE*>(memory);
      assert(tmp!=0);
      secondvmstubs_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    
    if (!((doL2L3D1&&(layer1_==2)&&(layer2_==3))||
	  (doL3L4L2&&(layer1_==3)&&(layer2_==4))||
	  (doL5L6L4&&(layer1_==5)&&(layer2_==6))||
	  (doD1D2L2&&(disk1_==1)&&(disk2_==2)))){
      return;
    }


    unsigned int countall=0;
    unsigned int countpass=0;
    unsigned int nInnerStubs=0;

    for(unsigned int iInnerMem=0;iInnerMem<firstvmstubs_.size();nInnerStubs+=firstvmstubs_.at(iInnerMem)->nStubs(),iInnerMem++);

    assert(!firstvmstubs_.empty());
    assert(secondvmstubs_!=0);

    for(unsigned int iInnerMem=0;iInnerMem<firstvmstubs_.size();iInnerMem++){
      for(unsigned int i=0;i<firstvmstubs_.at(iInnerMem)->nStubs();i++){
	std::pair<FPGAStub*,L1TStub*> firststub=firstvmstubs_.at(iInnerMem)->getStub(i);
	if (debug1) {
	  cout << "In "<<getName()<<" have first stub"<<endl;
	}
	
	if ((layer1_==3 && layer2_==4)||
	    (layer1_==5 && layer2_==6)) {
	  

	  int lookupbits=firststub.first->getVMBits().value()&1023;
	  int zdiffmax=(lookupbits>>7);	
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	  
	  int zbinfirst=newbin&7;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);
	  if (debug1) {
	    cout << "Will look in zbins "<<start<<" to "<<last<<endl;
	  }
	  for(int ibin=start;ibin<=last;ibin++) {
	    for(unsigned int j=0;j<secondvmstubs_->nStubsBinned(ibin);j++){
	      if (debug1) {
		cout << "In "<<getName()<<" have second stub"<<endl;
	      }

	      if (countall>=MAXTE) break;
	      countall++;
	      std::pair<FPGAStub*,L1TStub*> secondstub=secondvmstubs_->getStubBinned(ibin,j);
              
	      int zbin=(secondstub.first->getVMBits().value()&7);
	      if (start!=ibin) zbin+=8;
	      if (zbin<zbinfirst||zbin-zbinfirst>zdiffmax) {
		if (debug1) {
		  cout << "Stubpair rejected because of wrong zbin"<<endl;
		}
		continue;
	      }

	      //For debugging
	      //double trinv=rinv(firststub.second->phi(), secondstub.second->phi(),
	      //		       firststub.second->r(), secondstub.second->r());
	      
	      assert(firstphibits_!=-1);
	      assert(secondphibits_!=-1);
	      
	      unsigned int nvmfirst=nallstubslayers[layer1_-1]*nvmtelayers[layer1_-1];
	      unsigned int nvmsecond=nallstubslayers[layer1_]*nvmtelayers[layer1_];
	      unsigned int nvmbitsfirst=nbits(nvmfirst);
	      unsigned int nvmbitssecond=nbits(nvmsecond);
	      int iphifirstbin=firststub.first->iphivmFineBins(nvmbitsfirst,firstphibits_);
	      int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);
	      
	      unsigned int index = (iphifirstbin<<secondphibits_)+iphisecondbin;

	      //assert(index<(int)phitable_.size());		

	      //cout << "Stubpair layer rinv/rinvmax : "<<layer1_<<" "<<trinv/0.0057<<" "<<phitable_[index]<<endl;
	      
	      /*if (!phitable_[index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }*/
		
              FPGAWord firstbend=firststub.first->bend();
              FPGAWord secondbend=secondstub.first->bend();
              
              index=(index<<firstbend.nbits())+firstbend.value();
              index=(index<<secondbend.nbits())+secondbend.value();

	      //cout <<"bendfirst "<<bend(rmean[layer1_-1],trinv)<<" "<<0.5*(firstbend.value()-15.0)
	      //	   <<" "<<pttablefirst_[ptfirstindex]
	      //   <<"     bendsecond "<<bend(rmean[layer1_],trinv)<<" "<<0.5*(secondbend.value()-15.0)
	      //   <<" "<<pttablesecond_[ptsecondindex]<<endl;

              if (index >= table_.size())
                table_.resize(index+1);
	      
	      if (table_.at(index).empty()) {
		if (debug1) {
		  cout << "Stub pair rejected because of stub pt cut bends : "
		       <<FPGAStub::benddecode(firststub.first->bend().value(),firststub.first->isPSmodule())
		       <<" "
		       <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
		       <<endl;
		}		
                if (!writeTripletTables)
                  continue;
	      }
	      		
	      if (debug1) cout << "Adding layer-layer pair in " <<getName()<<endl;
              for(unsigned int isp=0; isp<stubpairs_.size(); ++isp){
                if (writeTripletTables || table_.at(index).count(stubpairs_.at(isp)->getName())) {
                  if (writeSeeds) {
                    ofstream fout("seeds.txt", ofstream::app);
                    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                    fout.close();
                  }
                  stubpairs_.at(isp)->addStubPair(firststub,secondstub,index,getName());
                }
              }

	      countpass++;
	    }
	  }

        } else if (layer1_==2 && layer2_==3) {
	  

	  int lookupbits=firststub.first->getVMBitsExtended().value()&1023;
	  int zdiffmax=(lookupbits>>7);	
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	  
	  int zbinfirst=newbin&7;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);
	  if (debug1) {
	    cout << "Will look in zbins "<<start<<" to "<<last<<endl;
	  }
	  for(int ibin=start;ibin<=last;ibin++) {
	    for(unsigned int j=0;j<secondvmstubs_->nStubsBinned(ibin);j++){
	      if (debug1) {
		cout << "In "<<getName()<<" have second stub"<<endl;
	      }

	      if (countall>=MAXTE) break;
	      countall++;
	      std::pair<FPGAStub*,L1TStub*> secondstub=secondvmstubs_->getStubBinned(ibin,j);
              
	      int zbin=(secondstub.first->getVMBitsExtended().value()&7);
	      if (start!=ibin) zbin+=8;
	      if (zbin<zbinfirst||zbin-zbinfirst>zdiffmax) {
		if (debug1) {
		  cout << "Stubpair rejected because of wrong zbin"<<endl;
		}
		continue;
	      }

	      //For debugging
	      //double trinv=rinv(firststub.second->phi(), secondstub.second->phi(),
	      //		       firststub.second->r(), secondstub.second->r());
	      
	      assert(firstphibits_!=-1);
	      assert(secondphibits_!=-1);
	      
	      unsigned int nvmfirst=nallstubslayers[layer1_-1]*nvmtelayers[layer1_-1];
	      unsigned int nvmsecond=nallstubslayers[layer1_]*nvmtelayers[layer1_];
	      unsigned int nvmbitsfirst=nbits(nvmfirst);
	      unsigned int nvmbitssecond=nbits(nvmsecond);
	      int iphifirstbin=firststub.first->iphivmFineBins(nvmbitsfirst,firstphibits_);
	      int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);
	      
	      unsigned int index = (iphifirstbin<<secondphibits_)+iphisecondbin;

	      //assert(index<(int)phitable_.size());		

	      //cout << "Stubpair layer rinv/rinvmax : "<<layer1_<<" "<<trinv/0.0057<<" "<<phitable_[index]<<endl;
	      
	      /*if (!phitable_[index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }*/
		
              FPGAWord firstbend=firststub.first->bend();
              FPGAWord secondbend=secondstub.first->bend();
              
              index=(index<<firstbend.nbits())+firstbend.value();
              index=(index<<secondbend.nbits())+secondbend.value();

	      //cout <<"bendfirst "<<bend(rmean[layer1_-1],trinv)<<" "<<0.5*(firstbend.value()-15.0)
	      //	   <<" "<<pttablefirst_[ptfirstindex]
	      //   <<"     bendsecond "<<bend(rmean[layer1_],trinv)<<" "<<0.5*(secondbend.value()-15.0)
	      //   <<" "<<pttablesecond_[ptsecondindex]<<endl;

              if (index >= table_.size())
                table_.resize(index+1);
	      
	      if (table_.at(index).empty()) {
		if (debug1) {
		  cout << "Stub pair rejected because of stub pt cut bends : "
		       <<FPGAStub::benddecode(firststub.first->bend().value(),firststub.first->isPSmodule())
		       <<" "
		       <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
		       <<endl;
		}		
                if (!writeTripletTables)
                  continue;
	      }
	      		
	      if (debug1) cout << "Adding layer-layer pair in " <<getName()<<endl;
              for(unsigned int isp=0; isp<stubpairs_.size(); ++isp){
                if (writeTripletTables || table_.at(index).count(stubpairs_.at(isp)->getName())) {
                  if (writeSeeds) {
                    ofstream fout("seeds.txt", ofstream::app);
                    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                    fout.close();
                  }
                  stubpairs_.at(isp)->addStubPair(firststub,secondstub,index,getName());
                }
              }

	      countpass++;
	    }
	  }
	
	} else if (disk1_==1 && disk2_==2){
	  
	  if (debug1) cout << getName()<<"["<<iSector_<<"] Disk-disk pair" <<endl;
	  
	  int lookupbits=firststub.first->getVMBits().value()&511;
	  bool negdisk=firststub.first->disk().value()<0;
	  int rdiffmax=(lookupbits>>6);	
	  int newbin=(lookupbits&63);
	  int bin=newbin/8;
	  
	  int rbinfirst=newbin&7;
	  
	  int start=(bin>>1);
	  if (negdisk) start+=4;
	  int last=start+(bin&1);
	  for(int ibin=start;ibin<=last;ibin++) {
	    if (debug1) cout << getName() << " looking for matching stub in bin "<<ibin
			     <<" with "<<secondvmstubs_->nStubsBinned(ibin)<<" stubs"<<endl;
	    for(unsigned int j=0;j<secondvmstubs_->nStubsBinned(ibin);j++){
	      if (countall>=MAXTE) break;
	      countall++;
	      std::pair<FPGAStub*,L1TStub*> secondstub=secondvmstubs_->getStubBinned(ibin,j);
	      int rbin=(secondstub.first->getVMBits().value()&7);
	      if (start!=ibin) rbin+=8;
	      if (rbin<rbinfirst) continue;
	      if (rbin-rbinfirst>rdiffmax) continue;

	      
	      
	      unsigned int irsecondbin=secondstub.first->getVMBits().value()>>2;

	      unsigned int nvmfirst=nallstubsdisks[disk1_-1]*nvmtedisks[disk1_-1];
	      unsigned int nvmsecond=nallstubsdisks[disk1_]*nvmtedisks[disk1_];
	      unsigned int nvmbitsfirst=nbits(nvmfirst);
	      unsigned int nvmbitssecond=nbits(nvmsecond);
	      int iphifirstbin=firststub.first->iphivmFineBins(nvmbitsfirst,firstphibits_);
	      int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);

	      
	      unsigned int index = (irsecondbin<<(secondphibits_+firstphibits_))+(iphifirstbin<<secondphibits_)+iphisecondbin;

	      //cout << "secondphibits_ firstphibits_"<<secondphibits_<<" "<<firstphibits_<<endl;
	      //cout << "index size : "<<index<<" "<<phitable_.size()<<" "<<irsecondbin<<" "<<iphifirstbin<<" "<<iphisecondbin<<endl;
	      
	      //assert(index<phitable_.size());		
	      /*if (!phitable_[index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }*/
		
              FPGAWord firstbend=firststub.first->bend();
              FPGAWord secondbend=secondstub.first->bend();
              
              index=(index<<firstbend.nbits())+firstbend.value();
              index=(index<<secondbend.nbits())+secondbend.value();

	      //assert(ptfirstindex<pttablefirst_.size());
	      //assert(ptsecondindex<pttablesecond_.size());
              if (index >= table_.size())
                table_.resize(index+1);
	      
	      if (table_.at(index).empty()) {
		if (debug1) {
		  cout << "Stub pair rejected because of stub pt cut bends : "
		       <<FPGAStub::benddecode(firststub.first->bend().value(),firststub.first->isPSmodule())
		       <<" "
		       <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
		       <<" FP bend: "<<firststub.second->bend()<<" "<<secondstub.second->bend()
		       //<<" pass : "<<pttablefirst_[ptfirstindex]<<" "<<pttablesecond_[ptsecondindex]
		       <<endl;
		}
                if (!writeTripletTables)
                  continue;
	      }

	      if (debug1) cout << "Adding disk-disk pair in " <<getName()<<endl;
	      
              for(unsigned int isp=0; isp<stubpairs_.size(); ++isp){
                if (writeTripletTables || table_.at(index).count(stubpairs_.at(isp)->getName())) {
                  if (writeSeeds) {
                    ofstream fout("seeds.txt", ofstream::app);
                    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                    fout.close();
                  }
                  stubpairs_.at(isp)->addStubPair(firststub,secondstub,index,getName());
                }
              }
	      countpass++;
	
	    }
	  }
	}
      }
    }
    if (countall>5000) {
      cout << "In FPGATrackletEngineDisplaced::execute : "<<getName()
	   <<" "<<nInnerStubs
	   <<" "<<secondvmstubs_->nStubs()
	   <<" "<<countall<<" "<<countpass
	   <<endl;
      for(unsigned int iInnerMem=0;iInnerMem<firstvmstubs_.size();iInnerMem++){
        for(unsigned int i=0;i<firstvmstubs_.at(iInnerMem)->nStubs();i++){
          std::pair<FPGAStub*,L1TStub*> firststub=firstvmstubs_.at(iInnerMem)->getStub(i);
          cout << "In FPGATrackletEngineDisplaced::execute first stub : "
               << firststub.second->r()<<" "
               << firststub.second->phi()<<" "
               << firststub.second->r()*firststub.second->phi()<<" "
               << firststub.second->z()<<endl;
        }
      }
      for(unsigned int i=0;i<secondvmstubs_->nStubs();i++){
	std::pair<FPGAStub*,L1TStub*> secondstub=secondvmstubs_->getStub(i);
	cout << "In FPGATrackletEngineDisplaced::execute second stub : "
	     << secondstub.second->r()<<" "
	     << secondstub.second->phi()<<" "
	     << secondstub.second->r()*secondstub.second->phi()<<" "
	     << secondstub.second->z()<<endl;
      }
      
    }
      
    if (writeTED) {
      static ofstream out("trackletenginedisplaced.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }

    
  }

  void readTables() {
    ifstream fin;
    string tableName, line, word;

    tableName = "../data/table_TED/table_" + name_ + ".txt";

    fin.open (tableName, ifstream::in);
    while (getline (fin, line)){
      istringstream iss (line);
      table_.resize (table_.size () + 1);

      while (iss >> word)
        table_[table_.size () - 1].insert (word);
    }
    fin.close ();
  }
  
private:

  double phimax_;
  double phimin_;
  
  int layer1_;
  int layer2_;
  int disk1_;
  int disk2_;
  
  vector<FPGAVMStubsTE*> firstvmstubs_;
  FPGAVMStubsTE* secondvmstubs_;
  
  vector<FPGAStubPairs*> stubpairs_;
  
  vector<set<string> > table_;
  
  int firstphibits_;
  int secondphibits_;
  
  int iSeed_;
  
};

#endif
