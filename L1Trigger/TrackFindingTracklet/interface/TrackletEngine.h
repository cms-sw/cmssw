//This class implementes the tracklet engine
#ifndef TRACKLETENGINE_H
#define TRACKLETENGINE_H

#include "ProcessBase.h"
#include "Util.h"


using namespace std;

class TrackletEngine:public ProcessBase{

public:

  TrackletEngine(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    double dphi=2*M_PI/NSector;
    phimin_=Util::phiRange(iSector*dphi);
    phimax_=Util::phiRange(phimin_+dphi);
    if (phimin_>phimax_)  phimin_-=2*M_PI;
    assert(phimax_>phimin_);
    stubpairs_=0;
    innervmstubs_=0;
    outervmstubs_=0;
    layer1_=0;
    layer2_=0;
    disk1_=0;
    disk2_=0;
    dct1_=0;
    dct2_=0;
    phi1_=0;
    phi2_=0;
    z1_=0;
    z2_=0;
    r1_=0;
    r2_=0;
    if (name[3]=='L') {
      layer1_=name[4]-'0';
      //z1_=name[12]-'0';
    }
    if (name[3]=='D') {
      disk1_=name[4]-'0';
    }
    if (name[11]=='L') {
      layer2_=name[12]-'0';
    }
    if (name[11]=='D') {
      disk2_=name[12]-'0';
    }
    if (name[12]=='L') {
      layer2_=name[13]-'0';
    }
    if (name[12]=='D') {
      disk2_=name[13]-'0';
    }
    
    innerphibits_=-1;
    outerphibits_=-1;

    extra_=(layer1_==2&&layer2_==3);

    iSeed_ = -1;
    if (layer1_ == 1 && layer2_ == 2) iSeed_ = 0;
    else if (layer1_ == 3 && layer2_ == 4) iSeed_ = 2;
    else if (layer1_ == 5 && layer2_ == 6) iSeed_ = 3;
    else if (disk1_ == 1 && disk2_ == 2) iSeed_ = 4;
    else if (disk1_ == 3 && disk2_ == 4) iSeed_ = 5;
    else if (disk1_ == 1 && layer2_ == 1) iSeed_ = 6;
    else if (disk1_ == 1 && layer2_ == 2) iSeed_ = 7;
    else if (layer1_ == 1 && disk2_ == 1) iSeed_ = 6;
    else if (layer1_ == 2 && disk2_ == 1) iSeed_ = 7;
    else if (layer1_ == 2 && layer2_ == 3) iSeed_ = 1;
  }

  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="stubpairout") {
      StubPairsMemory* tmp=dynamic_cast<StubPairsMemory*>(memory);
      assert(tmp!=0);
      stubpairs_=tmp;
      return;
    }
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="innervmstubin") {
      VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp!=0);
      innervmstubs_=tmp;
      setVMPhiBin();
      return;
    }
    if (input=="outervmstubin") {
      VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp!=0);
      outervmstubs_=tmp;
      setVMPhiBin();
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    if (!((doL1L2&&(layer1_==1)&&(layer2_==2))||
	  (doL2L3&&(layer1_==2)&&(layer2_==3))||
	  (doL3L4&&(layer1_==3)&&(layer2_==4))||
	  (doL5L6&&(layer1_==5)&&(layer2_==6))||
	  (doD1D2&&(disk1_==1)&&(disk2_==2))||
	  (doD3D4&&(disk1_==3)&&(disk2_==4))||
	  (doL1D1&&(disk1_==1)&&(layer2_==1))||
	  (doL2D1&&(disk1_==1)&&(layer2_==2))||
	  (doL1D1&&(disk2_==1)&&(layer1_==1))||
	  (doL2D1&&(disk2_==1)&&(layer1_==2)))) return;


    bool print=getName()=="TE_L1PHIE18_L2PHIC17" && iSector_==3;
    print=false;
    
    unsigned int countall=0;
    unsigned int countpass=0;

    assert(innervmstubs_!=0);
    assert(outervmstubs_!=0);

    //overlap seeding
    if (disk2_==1 && (layer1_==1 || layer1_==2) ) {
      for(unsigned int i=0;i<innervmstubs_->nVMStubs();i++){

	VMStubTE innervmstub=innervmstubs_->getVMStubTE(i);

	int lookupbits=(int)innervmstub.vmbits().value();
        int rdiffmax=(lookupbits>>7);	
	int newbin=(lookupbits&127);
	int bin=newbin/8;

	int rbinfirst=newbin&7;

	int start=(bin>>1);
	int last=start+(bin&1);

	for(int ibin=start;ibin<=last;ibin++) {
	  if (debug1) cout << getName() << " looking for matching stub in bin "<<ibin
			   <<" with "<<outervmstubs_->nVMStubsBinned(ibin)<<" stubs"<<endl;
	  for(unsigned int j=0;j<outervmstubs_->nVMStubsBinned(ibin);j++){
	    if (countall>=MAXTE) break;
	    countall++;
	    VMStubTE outervmstub=outervmstubs_->getVMStubTEBinned(ibin,j);

	    int rbin=(outervmstub.vmbits().value()&7);
	    if (start!=ibin) rbin+=8;
	    if ((rbin<rbinfirst)||(rbin-rbinfirst>rdiffmax)) {
	      if (debug1) {
		cout << getName() << " layer-disk stub pair rejected because rbin cut : "
		     <<rbin<<" "<<rbinfirst<<" "<<rdiffmax<<endl;
	      }
	      continue;
	    }

	    int ir=((start&3)<<3)+rbinfirst;
	    //cout << "start rbinfirst ir : "<<start<<" "<<rbinfirst<<" "<<ir<<" "<<innerstub.second->z()/innerstub.second->r()<<endl;
	    
	    assert(innerphibits_!=-1);
	    assert(outerphibits_!=-1);
	    
	    FPGAWord iphiinnerbin=innervmstub.finephi();
	    FPGAWord iphiouterbin=outervmstub.finephi();
	    
	    unsigned int index = (((iphiinnerbin.value()<<outerphibits_)+iphiouterbin.value())<<5)+ir;
	    
	    assert(index<phitable_.size());
	      
	    if (!phitable_[index]) {
	      if (debug1) {
		cout << "Stub pair rejected because of tracklet pt cut"<<endl;
	      }
	      continue;
	    }
		
	    FPGAWord innerbend=innervmstub.bend();
	    FPGAWord outerbend=outervmstub.bend();
            
	    int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
	    int ptouterindex=(index<<outerbend.nbits())+outerbend.value();
	    
	    if (!(pttableinner_[ptinnerindex]&&pttableouter_[ptouterindex])) {
	      if (debug1) {
		cout << "Stub pair rejected because of stub pt cut bends : "
		     <<Stub::benddecode(innervmstub.bend().value(),innervmstub.isPSmodule())
		     <<" "
		     <<Stub::benddecode(outervmstub.bend().value(),outervmstub.isPSmodule())
		     <<endl;
	      }		
	      continue;
	    }
	    
	    if (debug1) cout << "Adding layer-disk pair in " <<getName()<<endl;
            if (writeSeeds) {
              ofstream fout("seeds.txt", ofstream::app);
              fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
              fout.close();
            }
	    stubpairs_->addStubPair(innervmstub,outervmstub);
	    countpass++;
	  }
	}
      }

    } else {


      for(unsigned int i=0;i<innervmstubs_->nVMStubs();i++){

	VMStubTE innervmstub=innervmstubs_->getVMStubTE(i);
	if (debug1) {
	  cout << "In "<<getName()<<" have inner stub"<<endl;
	}
	
	if ((layer1_==1 && layer2_==2)||
	    (layer1_==2 && layer2_==3)||
	    (layer1_==3 && layer2_==4)||
	    (layer1_==5 && layer2_==6)) {	  

	  int lookupbits=(int)innervmstub.vmbits().value();
	  int zdiffmax=(lookupbits>>7);	
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;

	  int zbinfirst=newbin&7;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);

	  if (print) {
	    cout << "start last : "<<start<<" "<<last<<endl;
	  }
	  
	  if (debug1) {
	    cout << "Will look in zbins "<<start<<" to "<<last<<endl;
	  }
	  for(int ibin=start;ibin<=last;ibin++) {
	    for(unsigned int j=0;j<outervmstubs_->nVMStubsBinned(ibin);j++){
	      if (debug1) {
		cout << "In "<<getName()<<" have outer stub"<<endl;
	      }
	      
	      if (countall>=MAXTE) break;
	      countall++;

	      VMStubTE outervmstub=outervmstubs_->getVMStubTEBinned(ibin,j);
	      
	      int zbin=(outervmstub.vmbits().value()&7);
	      if (start!=ibin) zbin+=8;

	      if (zbin<zbinfirst||zbin-zbinfirst>zdiffmax) {
		if (debug1) {
		  cout << "Stubpair rejected because of wrong fine z"<<endl;
		}
		continue;
	      }

	      if (print) {
		cout << "ibin j "<<ibin<<" "<<j<<endl;
	      }
	      
	      //For debugging
	      //double trinv=rinv(innerstub.second->phi(), outerstub.second->phi(),
	      //		       innerstub.second->r(), outerstub.second->r());
	      
	      assert(innerphibits_!=-1);
	      assert(outerphibits_!=-1);
	      
	      FPGAWord iphiinnerbin=innervmstub.finephi();
	      FPGAWord iphiouterbin=outervmstub.finephi();
	      
	      int index = (iphiinnerbin.value()<<outerphibits_)+iphiouterbin.value();


	      assert(index<(int)phitable_.size());		

	      
	      if (!phitable_[index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }
		
	      FPGAWord innerbend=innervmstub.bend();
	      FPGAWord outerbend=outervmstub.bend();
	      
              int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
              int ptouterindex=(index<<outerbend.nbits())+outerbend.value();

	      //cout <<"bendinner "<<bend(rmean[layer1_-1],trinv)<<" "<<0.5*(innerbend.value()-15.0)
	      //	   <<" "<<pttableinner_[ptinnerindex]
	      //   <<"     bendouter "<<bend(rmean[layer1_],trinv)<<" "<<0.5*(outerbend.value()-15.0)
	      //   <<" "<<pttableouter_[ptouterindex]<<endl;

	      if (!(pttableinner_[ptinnerindex]&&pttableouter_[ptouterindex])) {
		if (debug1) {
		  cout << "Stub pair rejected because of stub pt cut bends : "
		       <<Stub::benddecode(innervmstub.bend().value(),innervmstub.isPSmodule())
		       <<" "
		       <<Stub::benddecode(outervmstub.bend().value(),outervmstub.isPSmodule())
		       <<endl;
		}		
		continue;
	      }
	      
	      if (debug1) cout << "Adding layer-layer pair in " <<getName()<<endl;
              if (writeSeeds) {
                ofstream fout("seeds.txt", ofstream::app);
                fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                fout.close();
              }
	      stubpairs_->addStubPair(innervmstub,outervmstub);
	      countpass++;
	    }
	  }
	
	} else if ((disk1_==1 && disk2_==2)||
		   (disk1_==3 && disk2_==4)) {
	  
	  if (debug1) cout << getName()<<"["<<iSector_<<"] Disk-disk pair" <<endl;
	  
	  int lookupbits=(int)innervmstub.vmbits().value();
	  bool negdisk=innervmstub.stub().first->disk().value()<0; //FIXME
	  int rdiffmax=(lookupbits>>6);	
	  int newbin=(lookupbits&63);
	  int bin=newbin/8;
	  
	  int rbinfirst=newbin&7;

	  //cout << "rbinfirst+rdiffmax next "<<rdiffmax<<" "<<rbinfirst+rdiffmax<<" "<<(bin&1)<<endl;
	  
	  int start=(bin>>1);
	  if (negdisk) start+=4;
	  int last=start+(bin&1);
	  for(int ibin=start;ibin<=last;ibin++) {
	    if (debug1) cout << getName() << " looking for matching stub in bin "<<ibin
			     <<" with "<<outervmstubs_->nVMStubsBinned(ibin)<<" stubs"<<endl;
	    for(unsigned int j=0;j<outervmstubs_->nVMStubsBinned(ibin);j++){
	      if (countall>=MAXTE) break;
	      countall++;
	      VMStubTE outervmstub=outervmstubs_->getVMStubTEBinned(ibin,j);
	      int rbin=(outervmstub.vmbits().value()&7);
	      if (start!=ibin) rbin+=8;
	      if (rbin<rbinfirst) continue;
	      if (rbin-rbinfirst>rdiffmax) continue;

	      
	      unsigned int irouterbin=outervmstub.vmbits().value()>>2;


	      FPGAWord iphiinnerbin=innervmstub.finephi();
	      FPGAWord iphiouterbin=outervmstub.finephi();
	      
	      unsigned int index = (irouterbin<<(outerphibits_+innerphibits_))+(iphiinnerbin.value()<<outerphibits_)+iphiouterbin.value();

	      assert(index<phitable_.size());		
	      if (!phitable_[index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }
		
	      FPGAWord innerbend=innervmstub.bend();
	      FPGAWord outerbend=outervmstub.bend();
	      
              unsigned int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
              unsigned int ptouterindex=(index<<outerbend.nbits())+outerbend.value();

	      assert(ptinnerindex<pttableinner_.size());
	      assert(ptouterindex<pttableouter_.size());
	      
	      if (!(pttableinner_[ptinnerindex]&&pttableouter_[ptouterindex])) {
		if (debug1) {
		  cout << "Stub pair rejected because of stub pt cut bends : "
		       <<Stub::benddecode(innervmstub.bend().value(),innervmstub.isPSmodule())
		       <<" "
		       <<Stub::benddecode(outervmstub.bend().value(),outervmstub.isPSmodule())
		    // <<" FP bend: "<<innerstub.second->bend()<<" "<<outerstub.second->bend()
		       <<" pass : "<<pttableinner_[ptinnerindex]<<" "<<pttableouter_[ptouterindex]
		       <<endl;
		}
		continue;
	      }

	      if (debug1) cout << "Adding disk-disk pair in " <<getName()<<endl;
	      
              if (writeSeeds) {
                ofstream fout("seeds.txt", ofstream::app);
                fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                fout.close();
              }
	      stubpairs_->addStubPair(innervmstub,outervmstub);
	      countpass++;
	
	    }
	  }
	}
      }
      
    }
  
    if (countall>5000) {
      cout << "In TrackletEngine::execute : "<<getName()
	   <<" "<<innervmstubs_->nVMStubs()
	   <<" "<<outervmstubs_->nVMStubs()
	   <<" "<<countall<<" "<<countpass
	   <<endl;
      for(unsigned int i=0;i<innervmstubs_->nVMStubs();i++){
	std::pair<Stub*,L1TStub*> innerstub=innervmstubs_->getVMStubTE(i).stub();
	cout << "In TrackletEngine::execute inner stub : "
	     << innerstub.second->r()<<" "
	     << innerstub.second->phi()<<" "
	     << innerstub.second->r()*innerstub.second->phi()<<" "
	     << innerstub.second->z()<<endl;
      }
      for(unsigned int i=0;i<outervmstubs_->nVMStubs();i++){
	std::pair<Stub*,L1TStub*> outerstub=outervmstubs_->getVMStubTE(i).stub();
	cout << "In TrackletEngine::execute outer stub : "
	     << outerstub.second->r()<<" "
	     << outerstub.second->phi()<<" "
	     << outerstub.second->r()*outerstub.second->phi()<<" "
	     << outerstub.second->z()<<endl;
      }
      
    }
      
    if (writeTE) {
      static ofstream out("trackletengine.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }

    
  }

  void setVMPhiBin() {
    if (innervmstubs_==0 || outervmstubs_==0 ) return;

    innervmstubs_->setother(outervmstubs_);
    outervmstubs_->setother(innervmstubs_);

    
    if ((layer1_==1 && layer2_==2)||
	(layer1_==2 && layer2_==3)||
	(layer1_==3 && layer2_==4)||
	(layer1_==5 && layer2_==6)){
      
      innerphibits_=nfinephibarrelinner;
      outerphibits_=nfinephibarrelouter;

      int innerphibins=(1<<innerphibits_);
      int outerphibins=(1<<outerphibits_);

      double innerphimin, innerphimax;
      innervmstubs_->getPhiRange(innerphimin,innerphimax);
      double rinner=rmean[layer1_-1];
      
      double outerphimin, outerphimax;
      outervmstubs_->getPhiRange(outerphimin,outerphimax);
      double router=rmean[layer2_-1];

      double phiinner[2];
      double phiouter[2];

      std::vector<bool> vmbendinner;
      std::vector<bool> vmbendouter;
      unsigned int nbins1=8;
      if (layer1_>=4) nbins1=16;
      for (unsigned int i=0;i<nbins1;i++) {
	vmbendinner.push_back(false);
      }

      unsigned int nbins2=8;
      if (layer2_>=4) nbins2=16;
      for (unsigned int i=0;i<nbins2;i++) {
	vmbendouter.push_back(false);
      }

      for (int iphiinnerbin=0;iphiinnerbin<innerphibins;iphiinnerbin++){
	phiinner[0]=innerphimin+iphiinnerbin*(innerphimax-innerphimin)/innerphibins;
	phiinner[1]=innerphimin+(iphiinnerbin+1)*(innerphimax-innerphimin)/innerphibins;
	for (int iphiouterbin=0;iphiouterbin<outerphibins;iphiouterbin++){
	  phiouter[0]=outerphimin+iphiouterbin*(outerphimax-outerphimin)/outerphibins;
	  phiouter[1]=outerphimin+(iphiouterbin+1)*(outerphimax-outerphimin)/outerphibins;

          double bendinnermin=20.0;
          double bendinnermax=-20.0;
          double bendoutermin=20.0;
          double bendoutermax=-20.0;
          double rinvmin=1.0; 
          for(int i1=0;i1<2;i1++) {
            for(int i2=0;i2<2;i2++) {
              double rinv1=rinv(phiinner[i1],phiouter[i2],rinner,router);
              double abendinner=bend(rinner,rinv1); 
              double abendouter=bend(router,rinv1);
              if (abendinner<bendinnermin) bendinnermin=abendinner;
              if (abendinner>bendinnermax) bendinnermax=abendinner;
              if (abendouter<bendoutermin) bendoutermin=abendouter;
              if (abendouter>bendoutermax) bendoutermax=abendouter;
              if (fabs(rinv1)<rinvmin) {
                rinvmin=fabs(rinv1);
              }
		      
            }
          }

          phitable_.push_back(rinvmin<rinvcutte);

	  int nbins1=8;
	  if (layer1_>=4) nbins1=16;
	  for(int ibend=0;ibend<nbins1;ibend++) {
	    double bend=Stub::benddecode(ibend,layer1_<=3); 
	    
	    bool passinner=bend-bendinnermin>-bendcut&&bend-bendinnermax<bendcut;	    
	    if (passinner) vmbendinner[ibend]=true;
	    pttableinner_.push_back(passinner);
	    
	  }
	  
	  int nbins2=8;
	  if (layer2_>=4) nbins2=16;
	  for(int ibend=0;ibend<nbins2;ibend++) {
	    double bend=Stub::benddecode(ibend,layer2_<=3); 
	    
	    bool passouter=bend-bendoutermin>-bendcut&&bend-bendoutermax<bendcut;
	    if (passouter) vmbendouter[ibend]=true;
	    pttableouter_.push_back(passouter);
	    
	  }

        }
      }

      innervmstubs_->setbendtable(vmbendinner);
      outervmstubs_->setbendtable(vmbendouter);
      
      if (iSector_==0&&writeTETables) writeTETable();
      
    }

    if ((disk1_==1 && disk2_==2)||
	(disk1_==3 && disk2_==4)){

      innerphibits_=nfinephidiskinner;
      outerphibits_=nfinephidiskouter;

      
      int outerrbits=3;

      int outerrbins=(1<<outerrbits);
      int innerphibins=(1<<innerphibits_);
      int outerphibins=(1<<outerphibits_);

      double innerphimin, innerphimax;
      innervmstubs_->getPhiRange(innerphimin,innerphimax);

      double outerphimin, outerphimax;
      outervmstubs_->getPhiRange(outerphimin,outerphimax);


      double phiinner[2];
      double phiouter[2];
      double router[2];

      std::vector<bool> vmbendinner;
      std::vector<bool> vmbendouter;

      for (unsigned int i=0;i<8;i++) {
	vmbendinner.push_back(false);
	vmbendouter.push_back(false);
      }


      for (int irouterbin=0;irouterbin<outerrbins;irouterbin++){
	router[0]=rmindiskvm+irouterbin*(rmaxdiskvm-rmindiskvm)/outerrbins;
	router[1]=rmindiskvm+(irouterbin+1)*(rmaxdiskvm-rmindiskvm)/outerrbins;
	for (int iphiinnerbin=0;iphiinnerbin<innerphibins;iphiinnerbin++){
	  phiinner[0]=innerphimin+iphiinnerbin*(innerphimax-innerphimin)/innerphibins;
	  phiinner[1]=innerphimin+(iphiinnerbin+1)*(innerphimax-innerphimin)/innerphibins;
	  for (int iphiouterbin=0;iphiouterbin<outerphibins;iphiouterbin++){
	    phiouter[0]=outerphimin+iphiouterbin*(outerphimax-outerphimin)/outerphibins;
	    phiouter[1]=outerphimin+(iphiouterbin+1)*(outerphimax-outerphimin)/outerphibins;

	    double bendinnermin=20.0;
	    double bendinnermax=-20.0;
	    double bendoutermin=20.0;
	    double bendoutermax=-20.0;
	    double rinvmin=1.0; 
	    double rinvmax=-1.0; 
	    for(int i1=0;i1<2;i1++) {
	      for(int i2=0;i2<2;i2++) {
		for(int i3=0;i3<2;i3++) {
		  double rinner=router[i3]*zmean[disk1_-1]/zmean[disk2_-1];
		  double rinv1=rinv(phiinner[i1],phiouter[i2],rinner,router[i3]);
		  double abendinner=bend(rinner,rinv1);
		  double abendouter=bend(router[i3],rinv1);
		  if (abendinner<bendinnermin) bendinnermin=abendinner;
		  if (abendinner>bendinnermax) bendinnermax=abendinner;
		  if (abendouter<bendoutermin) bendoutermin=abendouter;
		  if (abendouter>bendoutermax) bendoutermax=abendouter;
		  if (fabs(rinv1)<rinvmin) {
		    rinvmin=fabs(rinv1);
		  }
		  if (fabs(rinv1)>rinvmax) {
		    rinvmax=fabs(rinv1);
		  }
		}
	      }
	    }

	    //if (disk1_==1 && rinvmax>0.013 && rinvmin<0.0057){
	    //  cout << "router rinvmax rinvmin :"<<router[0]<<" "<<rinvmax<<" "<<rinvmin<<endl;
	    //}
	    
	    phitable_.push_back(rinvmin<rinvcutte);


	    for(int ibend=0;ibend<8;ibend++) {
	      double bend=Stub::benddecode(ibend,true); 
	      
	      bool passinner=bend-bendinnermin>-bendcutdisk&&bend-bendinnermax<bendcutdisk;	    
	      if (passinner) vmbendinner[ibend]=true;
	      pttableinner_.push_back(passinner);
	      
	    }
	    
	    for(int ibend=0;ibend<8;ibend++) {
	      double bend=Stub::benddecode(ibend,true); 
	      
	      bool passouter=bend-bendoutermin>-bendcutdisk&&bend-bendoutermax<bendcutdisk;
	      if (passouter) vmbendouter[ibend]=true;
	      pttableouter_.push_back(passouter);
	    
	    }
	    
	  }
	}
      }

      innervmstubs_->setbendtable(vmbendinner);
      outervmstubs_->setbendtable(vmbendouter);
      
      if (iSector_==0&&writeTETables) writeTETable();
      
    } else if (disk2_==1 && (layer1_==1 || layer1_==2)) {

      innerphibits_=nfinephioverlapinner;
      outerphibits_=nfinephioverlapouter;
      unsigned int nrbits=5;
      
      int innerphibins=(1<<innerphibits_);
      int outerphibins=(1<<outerphibits_);

      double innerphimin, innerphimax;
      innervmstubs_->getPhiRange(innerphimin,innerphimax);

      double outerphimin, outerphimax;
      outervmstubs_->getPhiRange(outerphimin,outerphimax);

      double phiinner[2];
      double phiouter[2];
      double router[2];


      std::vector<bool> vmbendinner;
      std::vector<bool> vmbendouter;

      for (unsigned int i=0;i<8;i++) {
	vmbendinner.push_back(false);
	vmbendouter.push_back(false);
      }
      
      double dr=(rmaxdiskvm-rmindiskvm)/(1<<nrbits);

      for (int iphiinnerbin=0;iphiinnerbin<innerphibins;iphiinnerbin++){
	phiinner[0]=innerphimin+iphiinnerbin*(innerphimax-innerphimin)/innerphibins;
	phiinner[1]=innerphimin+(iphiinnerbin+1)*(innerphimax-innerphimin)/innerphibins;
	for (int iphiouterbin=0;iphiouterbin<outerphibins;iphiouterbin++){
	  phiouter[0]=outerphimin+iphiouterbin*(outerphimax-outerphimin)/outerphibins;
	  phiouter[1]=outerphimin+(iphiouterbin+1)*(outerphimax-outerphimin)/outerphibins;
	  for (int irbin=0;irbin<(1<<nrbits);irbin++){
	    router[0]=rmindiskvm+dr*irbin;
	    router[1]=router[0]+dr; 
	    double bendinnermin=20.0;
	    double bendinnermax=-20.0;
	    double bendoutermin=20.0;
	    double bendoutermax=-20.0;
	    double rinvmin=1.0; 
	    for(int i1=0;i1<2;i1++) {
	      for(int i2=0;i2<2;i2++) {
		for(int i3=0;i3<2;i3++) {
		  double rinner=rmean[layer1_-1];
		  double rinv1=rinv(phiinner[i1],phiouter[i2],rinner,router[i3]);
		  double abendinner=bend(rinner,rinv1);
		  double abendouter=bend(router[i3],rinv1);
		  if (abendinner<bendinnermin) bendinnermin=abendinner;
		  if (abendinner>bendinnermax) bendinnermax=abendinner;
		  if (abendouter<bendoutermin) bendoutermin=abendouter;
		  if (abendouter>bendoutermax) bendoutermax=abendouter;
		  if (fabs(rinv1)<rinvmin) {
		    rinvmin=fabs(rinv1);
		  }
		}
	      }
	    }
	    
	    phitable_.push_back(rinvmin<rinvcutte);

	  
	    for(int ibend=0;ibend<8;ibend++) {
	      double bend=Stub::benddecode(ibend,true); 
	    
	      bool passinner=bend-bendinnermin>-bendcut&&bend-bendinnermax<bendcut;	    
	      if (passinner) vmbendinner[ibend]=true;
	      pttableinner_.push_back(passinner);
	      
	    }

	    for(int ibend=0;ibend<8;ibend++) {
	      double bend=Stub::benddecode(ibend,true); 
	      
	      bool passouter=bend-bendoutermin>-bendcut&&bend-bendoutermax<bendcut;
	      if (passouter) vmbendouter[ibend]=true;
	      pttableouter_.push_back(passouter);
	    
	    }
	  }
	}
      }
    
    
      innervmstubs_->setbendtable(vmbendinner);
      outervmstubs_->setbendtable(vmbendouter);
      
      if (iSector_==0&&writeTETables) writeTETable();
      
      
    }

  }

  double rinv(double phi1, double phi2,double r1, double r2){

    if (r2<r1) { //can not form tracklet
      return 20.0; 
    }
    
    assert(r2>r1);

    double dphi=phi2-phi1;
    double dr=r2-r1;
    
    return 2.0*sin(dphi)/dr/sqrt(1.0+2*r1*r2*(1.0-cos(dphi))/(dr*dr));
    
  }

  double bend(double r, double rinv) {

    double dr=0.18;
    
    double delta=r*dr*0.5*rinv;

    double bend=delta/0.009;
    if (r<55.0) bend=delta/0.01;

    return bend;
    
  }

  void writeTETable() {

    ofstream outptcut;
    outptcut.open(getName()+"_ptcut.txt");
    outptcut << "{"<<endl;
    for(unsigned int i=0;i<phitable_.size();i++){
      if (i!=0) outptcut<<","<<endl;
      outptcut << phitable_[i];
    }
    outptcut <<endl<<"};"<<endl;
    outptcut.close();

    ofstream outstubptinnercut;
    outstubptinnercut.open(getName()+"_stubptinnercut.txt");
    outstubptinnercut << "{"<<endl;
    for(unsigned int i=0;i<pttableinner_.size();i++){
      if (i!=0) outstubptinnercut<<","<<endl;
      outstubptinnercut << pttableinner_[i];
    }
    outstubptinnercut <<endl<<"};"<<endl;
    outstubptinnercut.close();
    
    ofstream outstubptoutercut;
    outstubptoutercut.open(getName()+"_stubptoutercut.txt");
    outstubptoutercut << "{"<<endl;
    for(unsigned int i=0;i<pttableouter_.size();i++){
      if (i!=0) outstubptoutercut<<","<<endl;
      outstubptoutercut << pttableouter_[i];
    }
    outstubptoutercut <<endl<<"};"<<endl;
    outstubptoutercut.close();

    
  }
  
private:

  double phimax_;
  double phimin_;
  
  int layer1_;
  int layer2_;
  int disk1_;
  int disk2_;
  int dct1_;
  int dct2_;
  int phi1_;
  int phi2_;
  int z1_;
  int z2_;
  int r1_;
  int r2_;
  
  VMStubsTEMemory* innervmstubs_;
  VMStubsTEMemory* outervmstubs_;
  
  StubPairsMemory* stubpairs_;

  bool extra_;
  
  vector<bool> phitable_;
  vector<bool> pttableinner_;
  vector<bool> pttableouter_;
  
  int innerphibits_;
  int outerphibits_;
  
  int iSeed_;
};

#endif
