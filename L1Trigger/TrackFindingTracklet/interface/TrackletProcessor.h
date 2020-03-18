//This class implementes the tracklet processor
#ifndef TRACKLETPROCESSOR_H
#define TRACKLETPROCESSOR_H

#include "IMATH_TrackletCalculator.h"
#include "IMATH_TrackletCalculatorDisk.h"
#include "IMATH_TrackletCalculatorOverlap.h"

#include "ProcessBase.h"
#include "VMStubsTEMemory.h"
#include "StubPairsMemory.h"
#include "TrackletProjectionsMemory.h"
#include "GlobalHistTruth.h"
#include "Util.h"


using namespace std;

class TrackletProcessor:public ProcessBase{

public:

  TrackletProcessor(string name, unsigned int iSector):
    ProcessBase(name,iSector){
   double dphi=2*M_PI/NSector;
   double dphiHG=0.5*dphisectorHG-M_PI/NSector;
    phimin_=iSector_*dphi-dphiHG;
    phimax_=phimin_+dphi+2*dphiHG;
    phimin_-=M_PI/NSector;
    phimax_-=M_PI/NSector;
    phimin_=Util::phiRange(phimin_);
    phimax_=Util::phiRange(phimax_);
    if (phimin_>phimax_)  phimin_-=2*M_PI;
    phioffset_=phimin_;
    
    maxtracklet_=127;

    for(unsigned int ilayer=0;ilayer<6;ilayer++){
      vector<TrackletProjectionsMemory*> tmp(nallstubslayers[ilayer],0);
      trackletprojlayers_.push_back(tmp);
    }

    for(unsigned int idisk=0;idisk<5;idisk++){
      vector<TrackletProjectionsMemory*> tmp(nallstubsdisks[idisk],0);
      trackletprojdisks_.push_back(tmp);
    }
          
    
    layer_=0;
    disk_=0;

    if (name_[3]=='L') {
      layer_=name_[4]-'0';
      if (name_[5]=='D') disk_=name_[6]-'0';    
    }
    if (name_[3]=='D') disk_=name_[4]-'0';    

    extra_=(layer_==2&&disk_==0);


    // set TC index
    if      (name_[7]=='A') iTC_ =0;
    else if (name_[7]=='B') iTC_ =1;
    else if (name_[7]=='C') iTC_ =2;
    else if (name_[7]=='D') iTC_ =3;
    else if (name_[7]=='E') iTC_ =4;
    else if (name_[7]=='F') iTC_ =5;
    else if (name_[7]=='G') iTC_ =6;
    else if (name_[7]=='H') iTC_ =7;
    else if (name_[7]=='I') iTC_ =8;
    else if (name_[7]=='J') iTC_ =9;
    else if (name_[7]=='K') iTC_ =10;
    else if (name_[7]=='L') iTC_ =11;
    else if (name_[7]=='M') iTC_ =12;
    else if (name_[7]=='N') iTC_ =13;
    else if (name_[7]=='O') iTC_ =14;
    
    assert(iTC_!=-1);
    
    if (name_.substr(3,4)=="L1L2") iSeed_ = 0;
    else if (name_.substr(3,4)=="L3L4") iSeed_ = 2;
    else if (name_.substr(3,4)=="L5L6") iSeed_ = 3;
    else if (name_.substr(3,4)=="D1D2") iSeed_ = 4;
    else if (name_.substr(3,4)=="D3D4") iSeed_ = 5;
    else if (name_.substr(3,4)=="D1L1") iSeed_ = 6;
    else if (name_.substr(3,4)=="D1L2") iSeed_ = 7;
    else if (name_.substr(3,4)=="L1D1") iSeed_ = 6;
    else if (name_.substr(3,4)=="L2D1") iSeed_ = 7;
    else if (name_.substr(3,4)=="L2L3") iSeed_ = 1;
    
    assert(iSeed_!=-1);

    TCIndex_ = (iSeed_<<4) + iTC_;
    assert(TCIndex_>=0 && TCIndex_<128);
   
    assert((layer_!=0)||(disk_!=0));

   
    if (iSeed_==0||iSeed_==1||iSeed_==2||iSeed_==3) {
      if (layer_==1) {
	rproj_[0]=rmeanL3;
	rproj_[1]=rmeanL4;
	rproj_[2]=rmeanL5;
	rproj_[3]=rmeanL6;
	lproj_[0]=3;
	lproj_[1]=4;
	lproj_[2]=5;
	lproj_[3]=6;
      }
      
      if (layer_==2) {
	rproj_[0]=rmeanL1;
	rproj_[1]=rmeanL4;
	rproj_[2]=rmeanL5;
	rproj_[3]=rmeanL6;
	lproj_[0]=1;
	lproj_[1]=4;
	lproj_[2]=5;
	lproj_[3]=6;
      }
      
      if (layer_==3) {
	rproj_[0]=rmeanL1;
	rproj_[1]=rmeanL2;
	rproj_[2]=rmeanL5;
	rproj_[3]=rmeanL6;
	lproj_[0]=1;
	lproj_[1]=2;
	lproj_[2]=5;
	lproj_[3]=6;
      }
      
      if (layer_==5) {
	rproj_[0]=rmeanL1;
	rproj_[1]=rmeanL2;
	rproj_[2]=rmeanL3;
	rproj_[3]=rmeanL4;
	lproj_[0]=1;
	lproj_[1]=2;
	lproj_[2]=3;
	lproj_[3]=4;
      }
    }
    
    if (iSeed_==4||iSeed_==5) {
      if (disk_==1) {
	zproj_[0]=zmeanD3;
	zproj_[1]=zmeanD4;
	zproj_[2]=zmeanD5;
	dproj_[0]=3;
	dproj_[1]=4;
	dproj_[2]=5;
      }
      
      if (disk_==3) {
	zproj_[0]=zmeanD1;
	zproj_[1]=zmeanD2;
	zproj_[2]=zmeanD5;
	dproj_[0]=1;
	dproj_[1]=2;
	dproj_[2]=5;
      }
    }
    
    
    if (iSeed_==6||iSeed_==7) {
      zprojoverlap_[0]=zmeanD2;
      zprojoverlap_[1]=zmeanD3;
      zprojoverlap_[2]=zmeanD4;
      zprojoverlap_[3]=zmeanD5;
    }
    
    if (usephicritapprox) {
      double phicritFactor = 0.5 * rcrit * TrackletProcessor::ITC_L1L2.rinv_final.get_K() / TrackletProcessor::ITC_L1L2.phi0_final.get_K();
      if (fabs(phicritFactor - 2.) > 0.25)
        cout << "TrackletProcessor::TrackletProcessor phicrit approximation may be invalid! Please check." << endl;
    }
  }
  
  void addOutputProjection(TrackletProjectionsMemory* &outputProj, MemoryBase* memory){
      outputProj=dynamic_cast<TrackletProjectionsMemory*>(memory);
      assert(outputProj!=0);
  }
  
  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="trackpar"){
      TrackletParametersMemory* tmp=dynamic_cast<TrackletParametersMemory*>(memory);
      assert(tmp!=0);
      trackletpars_=tmp;
      return;
    }

    if (output.substr(0,7)=="projout") {
      //output is on the form 'projoutL2PHIC' or 'projoutD3PHIB'
      TrackletProjectionsMemory* tmp=dynamic_cast<TrackletProjectionsMemory*>(memory);
      assert(tmp!=0);

      unsigned int layerdisk=output[8]-'1'; //layer or disk counting from 0
      unsigned int phiregion=output[12]-'A'; //phiregion counting from 0

      if (output[7]=='L') {
	assert(layerdisk<6);
	assert(phiregion<trackletprojlayers_[layerdisk].size());
	//check that phiregion not already initialized
	assert(trackletprojlayers_[layerdisk][phiregion]==0);
	trackletprojlayers_[layerdisk][phiregion]=tmp;
	return;
      }

      if (output[7]=='D') {
	assert(layerdisk<5);
	assert(phiregion<trackletprojdisks_[layerdisk].size());
	//check that phiregion not already initialized
	assert(trackletprojdisks_[layerdisk][phiregion]==0);
	trackletprojdisks_[layerdisk][phiregion]=tmp;
	return;
      }

    }

    cout << "Could not find output : "<<output<<endl;
    assert(0);


  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }

    if (input=="innervmstubin"){
      VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp!=0);
      innervmstubs_.push_back(tmp);
      setVMPhiBin();
      return;
    }
    if (input=="outervmstubin"){
      VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp!=0);
      outervmstubs_.push_back(tmp);
      setVMPhiBin();
      return;
    }
    if (input=="innerallstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      innerallstubs_.push_back(tmp);
      return;
    }
    if (input=="outerallstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      outerallstubs_.push_back(tmp);
      return;
    }
    assert(0);
  }


  void exacttracklet(double r1, double z1, double phi1,
		     double r2, double z2, double phi2, double,
		     double& rinv, double& phi0,
		     double& t, double& z0,
		     double phiproj[4], double zproj[4], 
		     double phider[4], double zder[4],
		     double phiprojdisk[5], double rprojdisk[5], 
		     double phiderdisk[5], double rderdisk[5]) {

    double deltaphi=Util::phiRange(phi1-phi2);
    
    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_;    
     
    phi0=Util::phiRange(phi1tmp+asin(0.5*r1*rinv));
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;

    for (int i=0;i<4;i++) {
      exactproj(rproj_[i],rinv,phi0,t,z0,
		phiproj[i],zproj[i],phider[i],zder[i]);
    }

    for (int i=0;i<5;i++) {
      //int sign=1;
      //if (t<0) sign=-1;
      exactprojdisk(zmean[i],rinv,phi0,t,z0,
		phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
    }



  }


  void exacttrackletdisk(double r1, double z1, double phi1,
			 double r2, double z2, double phi2, double,
			 double& rinv, double& phi0,
			 double& t, double& z0,
			 double phiprojLayer[3], double zprojLayer[3], 
			 double phiderLayer[3], double zderLayer[3],
			 double phiproj[3], double rproj[3], 
			 double phider[3], double rder[3]) {

    double deltaphi=Util::phiRange(phi1-phi2);

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_;    

    phi0=Util::phiRange(phi1tmp+asin(0.5*r1*rinv));
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;

    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<3;i++) {
      exactprojdisk(zproj_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<3;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }

  void exacttrackletOverlap(double r1, double z1, double phi1,
			    double r2, double z2, double phi2, double,
			    double& rinv, double& phi0,
			    double& t, double& z0,
			    double phiprojLayer[3], double zprojLayer[3], 
			    double phiderLayer[3], double zderLayer[3],
			    double phiproj[3], double rproj[3], 
			    double phider[3], double rder[3]) {

    double deltaphi=Util::phiRange(phi1-phi2);

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    if (r1>r2) rinv=-rinv;

    double phi1tmp=phi1-phimin_;    

    
    //cout << "phi1 phi2 phi1tmp : "<<phi1<<" "<<phi2<<" "<<phi1tmp<<endl;

    phi0=Util::phiRange(phi1tmp+asin(0.5*r1*rinv));
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;


    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<4;i++) {
      exactprojdisk(zprojoverlap_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<1;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }


  void exactproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    phiproj=phi0-asin(0.5*rproj*rinv);
    zproj=z0+(2*t/rinv)*asin(0.5*rproj*rinv);

    phider=-0.5*rinv/sqrt(1-pow(0.5*rproj*rinv,2));
    zder=t/sqrt(1-pow(0.5*rproj*rinv,2));

  }


  void exactprojdisk(double zproj,double rinv,double phi0,
		     double t, double z0,
		     double &phiproj, double &rproj,
		     double &phider, double &rder) {

    if (t<0) zproj=-zproj;
    
    double tmp=rinv*(zproj-z0)/(2.0*t);
    rproj=(2.0/rinv)*sin(tmp);
    phiproj=phi0-tmp;


    //if (fabs(1.0/rinv)>180.0&&fabs(z0)<15.0) {
    //  cout << "phiproj phi0 tmp zproj z0 t: "<<phiproj<<" "<<phi0
    //	   <<" "<<tmp<<" "<<zproj<<" "<<z0<<" "<<t<<endl;
    //}

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK1: "
	       << " phi="<<phiproj
	       << " r="<<rproj
	       << endl;
	}
      }


    phider=-rinv/(2*t);
    rder=cos(tmp)/t;

    //assert(fabs(phider)<0.1);

  } 

  void execute() {

    unsigned int countall=0;
    unsigned int countsel=0;

    StubPairsMemory stubpairs("tmp",iSector_,0.0,1.0); //dummy arguments for now

    bool print=false;
    
    assert(innervmstubs_.size()==outervmstubs_.size());
    
    if (!((doL1L2&&(layer_==1)&&(disk_==0))||
	  (doL2L3&&(layer_==2)&&(disk_==0))||
	  (doL3L4&&(layer_==3)&&(disk_==0))||
	  (doL5L6&&(layer_==5)&&(disk_==0))||
	  (doD1D2&&(disk_==1)&&(layer_==0))||
	  (doD3D4&&(disk_==3)&&(layer_==0))||
	  (doL1D1&&(layer_==1)&&(disk_==1))||
	  (doL2D1&&(layer_==2)&&(disk_==1)))) return;
    


    for (unsigned int ivmmem=0;ivmmem<innervmstubs_.size();ivmmem++) {

      unsigned int innerphibin=innervmstubs_[ivmmem]->phibin();
      unsigned int outerphibin=outervmstubs_[ivmmem]->phibin();

      unsigned phiindex=32*innerphibin+outerphibin;
      
      
      //overlap seeding
      if (disk_==1 && (layer_==1 || layer_==2) ) {

	for(unsigned int i=0;i<innervmstubs_[ivmmem]->nVMStubs();i++){

	  VMStubTE innervmstub=innervmstubs_[ivmmem]->getVMStubTE(i); 
	
	  int lookupbits=innervmstub.vmbits().value();

	  int rdiffmax=(lookupbits>>7);	
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	  
	  int rbinfirst=newbin&7;

	  int start=(bin>>1);
	  int last=start+(bin&1);
	  
	  for(int ibin=start;ibin<=last;ibin++) {
	    if (debug1) cout << getName() << " looking for matching stub in bin "<<ibin
			     <<" with "<<outervmstubs_[ivmmem]->nVMStubsBinned(ibin)<<" stubs"<<endl;
	    for(unsigned int j=0;j<outervmstubs_[ivmmem]->nVMStubsBinned(ibin);j++){
	      //if (countall>=MAXTE) break;
	      countall++;
	      
	      VMStubTE outervmstub=outervmstubs_[ivmmem]->getVMStubTEBinned(ibin,j);
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
	    
	      assert(innerphibits_!=-1);
	      assert(outerphibits_!=-1);
	      
	      FPGAWord iphiinnerbin=innervmstub.finephi();
	      FPGAWord iphiouterbin=outervmstub.finephi();

	      assert(iphiouterbin==outervmstub.finephi());
	      
	      unsigned int index = (((iphiinnerbin.value()<<outerphibits_)+iphiouterbin.value())<<5)+ir;
	      
	      assert(index<phitable_[phiindex].size());
	      
	      if (!phitable_[phiindex][index]) {
		if (debug1) {
		  cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		}
		continue;
	      }
	      
	      FPGAWord innerbend=innervmstub.bend();
	      FPGAWord outerbend=outervmstub.bend();

	      int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
	      int ptouterindex=(index<<outerbend.nbits())+outerbend.value();
	      
	      if (!(pttableinner_[phiindex][ptinnerindex]&&pttableouter_[phiindex][ptouterindex])) {
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
	      stubpairs.addStubPair(innervmstub,outervmstub);
	      countall++;
	    }
	  }
	}

      } else {



	for(unsigned int i=0;i<innervmstubs_[ivmmem]->nVMStubs();i++){
	  if (debug1) {
	    cout << "In "<<getName()<<" have inner stub"<<endl;
	  }

	  
	  if ((layer_==1 && disk_==0)||
	      (layer_==2 && disk_==0)||
	      (layer_==3 && disk_==0)||
	      (layer_==5 && disk_==0)) {	  

	    

	    VMStubTE innervmstub=innervmstubs_[ivmmem]->getVMStubTE(i);

	    
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
	      for(unsigned int j=0;j<outervmstubs_[ivmmem]->nVMStubsBinned(ibin);j++){
		if (debug1) {
		  cout << "In "<<getName()<<" have outer stub"<<endl;
		}
		
		//if (countall>=MAXTE) break;
		countall++;

		VMStubTE outervmstub=outervmstubs_[ivmmem]->getVMStubTEBinned(ibin,j);

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
		
		assert(index<(int)phitable_[phiindex].size());		
		
		
		if (!phitable_[phiindex][index]) {
		  if (debug1) {
		    cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		  }
		  continue;
		}
		
		FPGAWord innerbend=innervmstub.bend();
		FPGAWord outerbend=outervmstub.bend();

		
		int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
		int ptouterindex=(index<<outerbend.nbits())+outerbend.value();
		
		
		if (!(pttableinner_[phiindex][ptinnerindex]&&pttableouter_[phiindex][ptouterindex])) {
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
		stubpairs.addStubPair(innervmstub,outervmstub);
		
		countall++;
	      }
	    }
	    
	  } else if ((disk_==1 && layer_==0)||
		     (disk_==3 && layer_==0)) {
	    
	    if (debug1) cout << getName()<<"["<<iSector_<<"] Disk-disk pair" <<endl;

	    VMStubTE innervmstub=innervmstubs_[ivmmem]->getVMStubTE(i);

	    
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
			       <<" with "<<outervmstubs_[ivmmem]->nVMStubsBinned(ibin)<<" stubs"<<endl;
	      for(unsigned int j=0;j<outervmstubs_[ivmmem]->nVMStubsBinned(ibin);j++){
		//if (countall>=MAXTE) break;
		countall++;
		
		VMStubTE outervmstub=outervmstubs_[ivmmem]->getVMStubTEBinned(ibin,j);

		int vmbits=(int)outervmstub.vmbits().value();
		int rbin=(vmbits&7);
		if (start!=ibin) rbin+=8;
		if (rbin<rbinfirst) continue;
		if (rbin-rbinfirst>rdiffmax) continue;
		
	      
		unsigned int irouterbin=vmbits>>2;
		
		FPGAWord iphiinnerbin=innervmstub.finephi();
		FPGAWord iphiouterbin=outervmstub.finephi();
	      	      
		unsigned int index = (irouterbin<<(outerphibits_+innerphibits_))+(iphiinnerbin.value()<<outerphibits_)+iphiouterbin.value();
		
		assert(index<phitable_[phiindex].size());		
		if (!phitable_[phiindex][index]) {
		  if (debug1) {
		    cout << "Stub pair rejected because of tracklet pt cut"<<endl;
		  }
		  continue;
		}
		
		FPGAWord innerbend=innervmstub.bend();
		FPGAWord outerbend=outervmstub.bend();
		
		unsigned int ptinnerindex=(index<<innerbend.nbits())+innerbend.value();
		unsigned int ptouterindex=(index<<outerbend.nbits())+outerbend.value();
	      
		assert(ptinnerindex<pttableinner_[phiindex].size());
		assert(ptouterindex<pttableouter_[phiindex].size());
	      
		if (!(pttableinner_[phiindex][ptinnerindex]&&pttableouter_[phiindex][ptouterindex])) {
		  if (debug1) {
		    cout << "Stub pair rejected because of stub pt cut bends : "
			 <<Stub::benddecode(innervmstub.bend().value(),innervmstub.isPSmodule())
			 <<" "
			 <<Stub::benddecode(outervmstub.bend().value(),outervmstub.isPSmodule())
		      //<<" FP bend: "<<innerstub.second->bend()<<" "<<outerstub.second->bend()
		      <<" pass : "<<pttableinner_[phiindex][ptinnerindex]<<" "<<pttableouter_[phiindex][ptouterindex]
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
		stubpairs.addStubPair(innervmstub,outervmstub);
		countall++;
	
	      }
	    }
	  }
	}
      
      }
    }
    
        
    for(unsigned int i=0;i<stubpairs.nStubPairs();i++){

      if (trackletpars_->nTracklets()>=maxtracklet_) {
	cout << "Will break on too many tracklets in "<<getName()<<endl;
	break;
      }
      countall++;
      L1TStub* innerStub=stubpairs.getL1TStub1(i);
      Stub* innerFPGAStub=stubpairs.getFPGAStub1(i);
      
      L1TStub* outerStub=stubpairs.getL1TStub2(i);
      Stub* outerFPGAStub=stubpairs.getFPGAStub2(i);
      
      if (debug1) {
	cout << "TrackletProcessor execute "<<getName()<<"["<<iSector_<<"]"<<endl;
      }
      
      if (innerFPGAStub->isBarrel()&&(getName()!="TC_D1L2A"&&getName()!="TC_D1L2B")){
	
	
	if (outerFPGAStub->isDisk()) {
	  
	  //overlap seeding                                              
	  bool accept = overlapSeeding(outerFPGAStub,outerStub,innerFPGAStub,innerStub);
	  if (accept) countsel++;
	  
	} else {
	  
	  //barrel+barrel seeding	  
	    bool accept = barrelSeeding(innerFPGAStub,innerStub,outerFPGAStub,outerStub);
	    
	    if (accept) countsel++;
	    
	}
	
	}  else {
	
	if (outerFPGAStub->isDisk()) {
	  
	  //disk+disk seeding
	  
	  bool accept = diskSeeding(innerFPGAStub,innerStub,outerFPGAStub,outerStub);
	  
	  if (accept) countsel++;
	  

	} else if (innerFPGAStub->isDisk()) {
	  
	  
	  //layer+disk seeding
	  
	  bool accept = overlapSeeding(innerFPGAStub,innerStub,outerFPGAStub,outerStub);
	  
	  if (accept) countsel++;
	  
	  
	} else {
	  
	  assert(0);
	  
	}
      }
      
      if (trackletpars_->nTracklets()>=maxtracklet_) {
	cout << "Will break on number of tracklets in "<<getName()<<endl;
	break;
      }
      
      if (countall>=MAXTC) {
	if (debug1) cout << "Will break on MAXTC 1"<<endl;
	break;
      }
      if (debug1) {
	cout << "TrackletProcessor execute done"<<endl;
      }
      
    }
    if (countall>=MAXTC) {
      if (debug1) cout << "Will break on MAXTC 2"<<endl;
      //break;
    }
  
    
    if (writeTrackletProcessor) {
      static ofstream out("trackletcalculator.txt");
      out << getName()<<" "<<countall<<" "<<countsel<<endl;
    }
    
    
  }
  

  void addDiskProj(Tracklet* tracklet, int disk){

    FPGAWord fpgar=tracklet->fpgarprojdisk(disk);

    if (fpgar.value()*krprojshiftdisk<12.0) return;
    if (fpgar.value()*krprojshiftdisk>112.0) return;


    
    FPGAWord fpgaphi=tracklet->fpgaphiprojdisk(disk);
    
    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);

    int iphi=iphivmRaw/(32/nallstubsdisks[abs(disk)-1]);

    addProjectionDisk(disk,iphi,trackletprojdisks_[abs(disk)-1][iphi],tracklet);

  }


  bool addLayerProj(Tracklet* tracklet, int layer){

    assert(layer>0);

    FPGAWord fpgaz=tracklet->fpgazproj(layer);
    FPGAWord fpgaphi=tracklet->fpgaphiproj(layer);


    if(fpgaphi.atExtreme()) cout<<"at extreme! "<<fpgaphi.value()<<"\n";

    assert(!fpgaphi.atExtreme());
    
    if (fpgaz.atExtreme()) return false;

    if (fabs(fpgaz.value()*kz)>zlength) return false;

    int iphivmRaw=fpgaphi.value()>>(fpgaphi.nbits()-5);

    int iphi=iphivmRaw/(32/nallstubslayers[layer-1]);

    addProjection(layer,iphi,trackletprojlayers_[layer-1][iphi],tracklet);
    
    return true;

  }

  void addProjection(int layer,int iphi,TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet){
    if (trackletprojs==0) {
      if (warnNoMem) {
	cout << "No projection memory exists in "<<getName()<<" for layer = "<<layer<<" iphi = "<<iphi+1<<endl;
      }
      return;
    }
    assert(trackletprojs!=0);
    trackletprojs->addProj(tracklet);
  }

  void addProjectionDisk(int disk,int iphi,TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet){
    if (layer_==2&&abs(disk)==4) return; //L3L4 projections to D3 are not used. Should be in configuration
    if (trackletprojs==0) {
      if (layer_==3&&abs(disk)==3) return; //L3L4 projections to D3 are not used.
      if (warnNoMem) {       
	cout << "No projection memory exists in "<<getName()<<" for disk = "<<abs(disk)<<" iphi = "<<iphi+1<<endl;
      }
      return;
    }
    assert(trackletprojs!=0);
    trackletprojs->addProj(tracklet);
  }

  bool barrelSeeding(Stub* innerFPGAStub, L1TStub* innerStub, Stub* outerFPGAStub, L1TStub* outerStub){

    if (debug1) {
      cout << "TrackletProcessor "<<getName()<<" "<<layer_<<" trying stub pair in layer (inner outer): "
	   <<innerFPGAStub->layer().value()<<" "<<outerFPGAStub->layer().value()<<endl;
    }
	    
    assert(outerFPGAStub->isBarrel());
    
    assert(layer_==innerFPGAStub->layer().value()+1);
    
    assert(layer_==1||layer_==2||layer_==3||layer_==5);

    	  
    double r1=innerStub->r();
    double z1=innerStub->z();
    double phi1=innerStub->phi();
    
    double r2=outerStub->r();
    double z2=outerStub->z();
    double phi2=outerStub->phi();

    LayerProjection layerprojs[4];
    DiskProjection diskprojs[5];
    
    double rinv,phi0,t,z0;
    
    double phiproj[4],zproj[4],phider[4],zder[4];
    double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
    
    exacttracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
		  rinv,phi0,t,z0,
		  phiproj,zproj,phider,zder,
		  phiprojdisk,rprojdisk,phiderdisk,rderdisk);

    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();

      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }
    
    double rinvapprox,phi0approx,tapprox,z0approx;
    double phiprojapprox[4],zprojapprox[4];
    double phiprojdiskapprox[5],rprojdiskapprox[5];
    
    IMATH_TrackletCalculator *ITC;
    if(layer_==1)      ITC = &ITC_L1L2;
    else if(layer_==2) ITC = &ITC_L2L3;
    else if(layer_==3) ITC = &ITC_L3L4;
    else               ITC = &ITC_L5L6;
    
    ITC->r1.set_fval(r1-rmean[layer_-1]);
    ITC->r2.set_fval(r2-rmean[layer_]);
    ITC->z1.set_fval(z1);
    ITC->z2.set_fval(z2);
    double sphi1 = phi1 - phioffset_;
    if(sphi1<0) sphi1 += 8*atan(1.);
    if(sphi1>8*atan(1.)) sphi1 -= 8*atan(1.);
    //cout << "sphi1: "<<phi2<<endl;
    double sphi2 = phi2 - phioffset_;
    if(sphi2<0) sphi2 += 8*atan(1.);
    if(sphi2>8*atan(1.)) sphi2 -= 8*atan(1.);
    ITC->phi1.set_fval(sphi1);
    ITC->phi2.set_fval(sphi2);

    ITC->rproj0.set_fval(rproj_[0]);
    ITC->rproj1.set_fval(rproj_[1]);
    ITC->rproj2.set_fval(rproj_[2]);
    ITC->rproj3.set_fval(rproj_[3]);

    ITC->zproj0.set_fval(t>0? zmean[0] : -zmean[0]);
    ITC->zproj1.set_fval(t>0? zmean[1] : -zmean[1]);
    ITC->zproj2.set_fval(t>0? zmean[2] : -zmean[2]);
    ITC->zproj3.set_fval(t>0? zmean[3] : -zmean[3]);
    ITC->zproj4.set_fval(t>0? zmean[4] : -zmean[4]);

    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();
    ITC->phiL_3_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();
    ITC->zL_3_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();
    ITC->phiD_3_final.calculate();
    ITC->phiD_4_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();
    ITC->rD_3_final.calculate();
    ITC->rD_4_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the approcximate results
    rinvapprox = ITC->rinv_final.get_fval();
    phi0approx = ITC->phi0_final.get_fval();
    tapprox    = ITC->t_final.get_fval();
    z0approx   = ITC->z0_final.get_fval();
    
    /*
    cout << "texact:"<<t<<" tapprox:"<<tapprox<<endl;
    cout << "z0exact:"<<z0<<" z0approx:"<<z0approx<<" "
	 <<" zeroth order:"<<z1-tapprox*r1
	 <<" first order:"
	 <<z1-tapprox*r1*(1+r1*r1*rinvapprox*rinvapprox/24.0)
	 <<" wrong first order:"
	 <<z1-tapprox*r1*(1+r1*r1*rinvapprox*rinvapprox/6.0)
	 <<endl;
    */
    
    phiprojapprox[0] = ITC->phiL_0_final.get_fval();
    phiprojapprox[1] = ITC->phiL_1_final.get_fval();
    phiprojapprox[2] = ITC->phiL_2_final.get_fval();
    phiprojapprox[3] = ITC->phiL_3_final.get_fval();

    zprojapprox[0]   = ITC->zL_0_final.get_fval();
    zprojapprox[1]   = ITC->zL_1_final.get_fval();
    zprojapprox[2]   = ITC->zL_2_final.get_fval();
    zprojapprox[3]   = ITC->zL_3_final.get_fval();


    phiprojdiskapprox[0] = ITC->phiD_0_final.get_fval();
    phiprojdiskapprox[1] = ITC->phiD_1_final.get_fval();
    phiprojdiskapprox[2] = ITC->phiD_2_final.get_fval();
    phiprojdiskapprox[3] = ITC->phiD_3_final.get_fval();
    phiprojdiskapprox[4] = ITC->phiD_4_final.get_fval();

    rprojdiskapprox[0] = ITC->rD_0_final.get_fval();
    rprojdiskapprox[1] = ITC->rD_1_final.get_fval();
    rprojdiskapprox[2] = ITC->rD_2_final.get_fval();
    rprojdiskapprox[3] = ITC->rD_3_final.get_fval();
    rprojdiskapprox[4] = ITC->rD_4_final.get_fval();

    //now binary
    
    int irinv,iphi0,it,iz0;
    int iphiproj[4],izproj[4];
    int iphiprojdisk[5],irprojdisk[5];
    
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();
      
    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();
    
    if (layer_<4) iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    if (layer_<3) iphi2<<=(nbitsphistubL456-nbitsphistubL123);
    if (layer_<4) {
      ir1<<=(8-nbitsrL123);
    } else {
      ir1<<=(8-nbitsrL456);
    }
    if (layer_<3) {
      ir2<<=(8-nbitsrL123);
    } else {
      ir2<<=(8-nbitsrL456);
    }
    if (layer_>3) iz1<<=(nbitszL123-nbitszL456);
    if (layer_>2) iz2<<=(nbitszL123-nbitszL456);  
      
    ITC->r1.set_ival(ir1);
    ITC->r2.set_ival(ir2);
    ITC->z1.set_ival(iz1);
    ITC->z2.set_ival(iz2);
    ITC->phi1.set_ival(iphi1);
    ITC->phi2.set_ival(iphi2);
    
    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();
    ITC->phiL_3_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();
    ITC->zL_3_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();
    ITC->phiD_3_final.calculate();
    ITC->phiD_4_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();
    ITC->rD_3_final.calculate();
    ITC->rD_4_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the binary results
    irinv = ITC->rinv_final.get_ival();
    iphi0 = ITC->phi0_final.get_ival();
    it    = ITC->t_final.get_ival();
    iz0   = ITC->z0_final.get_ival();

    iphiproj[0] = ITC->phiL_0_final.get_ival();
    iphiproj[1] = ITC->phiL_1_final.get_ival();
    iphiproj[2] = ITC->phiL_2_final.get_ival();
    iphiproj[3] = ITC->phiL_3_final.get_ival();
    
    izproj[0]   = ITC->zL_0_final.get_ival();
    izproj[1]   = ITC->zL_1_final.get_ival();
    izproj[2]   = ITC->zL_2_final.get_ival();
    izproj[3]   = ITC->zL_3_final.get_ival();


    bool success = true;
    if(!ITC->rinv_final.local_passes()){
      if (debug1) 
	cout << "TrackletProcessor::BarrelSeeding irinv too large: "
	     <<ITC->rinv_final.get_fval()<<"("<<ITC->rinv_final.get_ival()<<")\n";
      success = false;
    }
    if (!ITC->z0_final.local_passes()){
      if (debug1) cout << "Failed tracklet z0 cut "<<ITC->z0_final.get_fval()<<" in layer "<<layer_<<endl;
      success = false;
    }
    success = success && ITC->valid_trackpar.passes();

    if (!success) return false;

    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletProcessor::barrelSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }
    
    

    
    if (writeTC) {
      cout << "TC "<<layer_<<" "<<innerFPGAStub->iphivmRaw()<<" "<<outerFPGAStub->iphivmRaw();
    }
    
    for(int i=0; i<4; ++i){

      //reject projection if z is out of range
      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;

      //reject projection if phi is out of range
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;
      
      //Adjust bits for r and z projection depending on layer
      if (lproj_[i]<=3) {
	iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
      }
      else {
	izproj[i]>>=(nbitszprojL123-nbitszprojL456);
      }

      layerprojs[i].init(lproj_[i],rproj_[i],
			 iphiproj[i],izproj[i],
			 ITC->der_phiL_final.get_ival(),ITC->der_zL_final.get_ival(),
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 ITC->der_phiL_final.get_fval(),ITC->der_zL_final.get_fval());
      
    }

    if (writeTC) {
      cout << endl;
    }
      
    iphiprojdisk[0] = ITC->phiD_0_final.get_ival();
    iphiprojdisk[1] = ITC->phiD_1_final.get_ival();
    iphiprojdisk[2] = ITC->phiD_2_final.get_ival();
    iphiprojdisk[3] = ITC->phiD_3_final.get_ival();
    iphiprojdisk[4] = ITC->phiD_4_final.get_ival();

    irprojdisk[0]   = ITC->rD_0_final.get_ival();
    irprojdisk[1]   = ITC->rD_1_final.get_ival();
    irprojdisk[2]   = ITC->rD_2_final.get_ival();
    irprojdisk[3]   = ITC->rD_3_final.get_ival();
    irprojdisk[4]   = ITC->rD_4_final.get_ival();

    if(fabs(it * ITC->t_final.get_K())>1.0) {
      for(int i=0; i<5; ++i){

	if (iphiprojdisk[i]<=0) continue;
	if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;
	
	if(irprojdisk[i]< 20. / ITC->rD_0_final.get_K() ||
	   irprojdisk[i] > 120. / ITC->rD_0_final.get_K() ) continue;

	diskprojs[i].init(i+1,rproj_[i],
			   iphiprojdisk[i],irprojdisk[i],
			   ITC->der_phiD_final.get_ival(),ITC->der_rD_final.get_ival(),
			   phiprojdisk[i],rprojdisk[i],
			   phiderdisk[i],rderdisk[i],
			   phiprojdiskapprox[i],rprojdiskapprox[i],
			   ITC->der_phiD_final.get_fval(),ITC->der_rD_final.get_fval());
	
      }
    }

    
    
    if (writeTrackletPars) {
      static ofstream out("trackletpars.txt");
      out <<"Trackpars "<<layer_
	  <<"   "<<rinv<<" "<<rinvapprox<<" "<<ITC->rinv_final.get_fval()
	  <<"   "<<phi0<<" "<<phi0approx<<" "<<ITC->phi0_final.get_fval()
	  <<"   "<<t<<" "<<tapprox<<" "<<ITC->t_final.get_fval()
	  <<"   "<<z0<<" "<<z0approx<<" "<<ITC->z0_final.get_fval()
	  <<endl;
    }	        

    
    Tracklet* tracklet=new Tracklet(innerStub,NULL,outerStub,
				    innerFPGAStub,NULL,outerFPGAStub,
				    rinv,phi0,0.0,z0,t,
				    rinvapprox,phi0approx,0.0,
				    z0approx,tapprox,
				    irinv,iphi0,0,iz0,it,
				    layerprojs,
				    diskprojs,
				    false);
    
    if (debug1) {
      cout << "TrackletProcessor "<<getName()<<" Found tracklet in layer = "<<layer_<<" "
	   <<iSector_<<" phi0 = "<<phi0<<endl;
    }
        

    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);

    HistBase* hists=GlobalHistTruth::histograms();
    int tp=tracklet->tpseed();
    hists->fillTrackletParams(iSeed_,iSector_,
			      rinvapprox,irinv*ITC->rinv_final.get_K(),
			      phi0approx,iphi0*ITC->phi0_final.get_K(),
			      asinh(tapprox),asinh(it*ITC->t_final.get_K()),
			      z0approx,iz0*ITC->z0_final.get_K(),
			      tp);

    
    bool addL3=false;
    bool addL4=false;
    bool addL5=false;
    bool addL6=false;
    for(unsigned int j=0;j<4;j++){
      //	    cout<<" LL to L "<<lproj[j]<<"\n";
      bool added=false;
      if (tracklet->validProj(lproj_[j])) {
	added=addLayerProj(tracklet,lproj_[j]);
	//cout << "Add tracklet proj for layer "<<lproj_[j]<<": "<<phiproj[j]<<" "<<iphiproj[j]<<" added = "
	//     <<added<<endl;
	if (added&&lproj_[j]==3) addL3=true;
	if (added&&lproj_[j]==4) addL4=true;
	if (added&&lproj_[j]==5) addL5=true;
	if (added&&lproj_[j]==6) addL6=true;
      }
    }
    
    
    for(unsigned int j=0;j<4;j++){ //no projections to 5th disk!!
      int disk=j+1;
      if (disk==4&&addL3) continue;
      if (disk==3&&addL4) continue;
      if (disk==2&&addL5) continue;
      if (disk==1&&addL6) continue;
      if (it<0) disk=-disk;
      //	    cout<<" LL to disk "<<disk<<"\n";
      if (tracklet->validProjDisk(abs(disk))) {
	//cout << "Add tracklet "<<tracklet<<" for disk "<<disk<<endl;
	addDiskProj(tracklet,disk);
      }
    }
    
    return true;

  }
    

  bool diskSeeding(Stub* innerFPGAStub,L1TStub* innerStub,Stub* outerFPGAStub,L1TStub* outerStub){

	    
    if (debug1) {
      cout <<  "TrackletProcessor::execute calculate disk seeds" << endl;
    }
	      
    int sign=1;
    if (innerFPGAStub->disk().value()<0) sign=-1;
    
    int disk=innerFPGAStub->disk().value();
    assert(abs(disk)==1||abs(disk)==3);
    
    
    assert(innerStub->isPSmodule());
    assert(outerStub->isPSmodule());
	    
    double r1=innerStub->r();
    double z1=innerStub->z();
    double phi1=innerStub->phi();
    
    double r2=outerStub->r();
    double z2=outerStub->z();
    double phi2=outerStub->phi();
	    
    
    if (r2<r1+2.0) {
      //assert(0);
      return false; //Protection... Should be handled cleaner
      //to avoid problem with floating point 
      //calculation
    }

    LayerProjection layerprojs[3];
    DiskProjection diskprojs[3];
    
    double rinv,phi0,t,z0;
    
    double phiproj[3],zproj[3],phider[3],zder[3];
    double phiprojdisk[3],rprojdisk[3],phiderdisk[3],rderdisk[3];
    
    exacttrackletdisk(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
		      rinv,phi0,t,z0,
		      phiproj,zproj,phider,zder,
		      phiprojdisk,rprojdisk,phiderdisk,rderdisk);


    //Truncates floating point positions to integer
    //representation precision
    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();
      
      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }
    
    double rinvapprox,phi0approx,tapprox,z0approx;
    double phiprojapprox[3],zprojapprox[3];
    double phiprojdiskapprox[3],rprojdiskapprox[3];
	    
    IMATH_TrackletCalculatorDisk *ITC;
    if(disk==1)       ITC = &ITC_F1F2;
    else if(disk==3)  ITC = &ITC_F3F4;
    else if(disk==-1) ITC = &ITC_B1B2;
    else               ITC = &ITC_B3B4;
    
    ITC->r1.set_fval(r1);
    ITC->r2.set_fval(r2);
    int signt = t>0? 1 : -1;
    ITC->z1.set_fval(z1-signt*zmean[abs(disk_)-1]);
    ITC->z2.set_fval(z2-signt*zmean[abs(disk_)]);
    double sphi1 = phi1 - phioffset_;
    if(sphi1<0) sphi1 += 8*atan(1.);
    if(sphi1>8*atan(1.)) sphi1 -= 8*atan(1.);
    double sphi2 = phi2 - phioffset_;
    if(sphi2<0) sphi2 += 8*atan(1.);
    if(sphi2>8*atan(1.)) sphi2 -= 8*atan(1.);
    ITC->phi1.set_fval(sphi1);
    ITC->phi2.set_fval(sphi2);
	    
    ITC->rproj0.set_fval(rmean[0]);
    ITC->rproj1.set_fval(rmean[1]);
    ITC->rproj2.set_fval(rmean[2]);

    ITC->zproj0.set_fval(t>0? zproj_[0] : -zproj_[0]);
    ITC->zproj1.set_fval(t>0? zproj_[1] : -zproj_[1]);
    ITC->zproj2.set_fval(t>0? zproj_[2] : -zproj_[2]);

    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the approximate results
    rinvapprox = ITC->rinv_final.get_fval();
    phi0approx = ITC->phi0_final.get_fval();
    tapprox    = ITC->t_final.get_fval();
    z0approx   = ITC->z0_final.get_fval();

    /*
    cout << "texact:"<<t<<" tapprox:"<<tapprox<<endl;
    cout << "z0exact:"<<z0<<" z0approx:"<<z0approx<<" "
	 <<" zeroth order:"<<z1-tapprox*r1
	 <<" first order:"
	 <<z1-tapprox*r1*(1+r1*r1*rinvapprox*rinvapprox/24.0)
	 <<" wrong first order:"
	 <<z1-tapprox*r1*(1+r1*r1*rinvapprox*rinvapprox/6.0)
	 <<endl;
    */
    
    
    phiprojapprox[0] = ITC->phiL_0_final.get_fval();
    phiprojapprox[1] = ITC->phiL_1_final.get_fval();
    phiprojapprox[2] = ITC->phiL_2_final.get_fval();

    zprojapprox[0]   = ITC->zL_0_final.get_fval();
    zprojapprox[1]   = ITC->zL_1_final.get_fval();
    zprojapprox[2]   = ITC->zL_2_final.get_fval();

    phiprojdiskapprox[0] = ITC->phiD_0_final.get_fval();
    phiprojdiskapprox[1] = ITC->phiD_1_final.get_fval();
    phiprojdiskapprox[2] = ITC->phiD_2_final.get_fval();

    rprojdiskapprox[0] = ITC->rD_0_final.get_fval();
    rprojdiskapprox[1] = ITC->rD_1_final.get_fval();
    rprojdiskapprox[2] = ITC->rD_2_final.get_fval();

    //now binary
    
    int irinv,iphi0,it,iz0;
    int iphiproj[3],izproj[3];
    
    int iphiprojdisk[3],irprojdisk[3];

    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();
    
    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();
    
    //To get same precission as for layers.
    iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    iphi2<<=(nbitsphistubL456-nbitsphistubL123);
    
    ITC->r1.set_ival(ir1);
    ITC->r2.set_ival(ir2);
    ITC->z1.set_ival(iz1);
    ITC->z2.set_ival(iz2);
    ITC->phi1.set_ival(iphi1);
    ITC->phi2.set_ival(iphi2);

    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the binary results
    irinv = ITC->rinv_final.get_ival();
    iphi0 = ITC->phi0_final.get_ival();
    it    = ITC->t_final.get_ival();
    iz0   = ITC->z0_final.get_ival();

    iphiproj[0] = ITC->phiL_0_final.get_ival();
    iphiproj[1] = ITC->phiL_1_final.get_ival();
    iphiproj[2] = ITC->phiL_2_final.get_ival();
    
    izproj[0]   = ITC->zL_0_final.get_ival();
    izproj[1]   = ITC->zL_1_final.get_ival();
    izproj[2]   = ITC->zL_2_final.get_ival();


    bool success = true;
    if(!ITC->rinv_final.local_passes()){
     if (debug1) 
	cout << "TrackletProcessor::DiskSeeding irinv too large: "<<ITC->rinv_final.get_fval()
	     << " disk = "<<disk_<<" r1="<<r1<<endl;
      success = false;
    }
    if (!ITC->z0_final.local_passes()) {
      if (debug1)
	cout << "Failed tracklet z0 cut "<<ITC->z0_final.get_fval()<<" in disk "<<disk_<<endl;
      success = false;
    }
    success = success && ITC->valid_trackpar.passes();
   
    if (!success) return false;

    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletProcessor::diskSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }

    
    
    
    for(int i=0; i<3; ++i){

      //Check is outside z range
      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;

      //Check if outside phi range
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;

      //shift bits - allways in PS modules for disk seeding
      iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
      
      layerprojs[i].init(i+1,rmean[i],
			 iphiproj[i],izproj[i],
			 ITC->der_phiL_final.get_ival(),ITC->der_zL_final.get_ival(),
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 ITC->der_phiL_final.get_fval(),ITC->der_zL_final.get_fval());
    }

    iphiprojdisk[0] = ITC->phiD_0_final.get_ival();
    iphiprojdisk[1] = ITC->phiD_1_final.get_ival();
    iphiprojdisk[2] = ITC->phiD_2_final.get_ival();

    irprojdisk[0]   = ITC->rD_0_final.get_ival();
    irprojdisk[1]   = ITC->rD_1_final.get_ival();
    irprojdisk[2]   = ITC->rD_2_final.get_ival();

    for(int i=0; i<3; ++i){

      //check that phi projection in range
      if (iphiprojdisk[i]<=0) continue;
      if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;

      //check that r projection in range
      if(irprojdisk[i]<=0 ||
	 irprojdisk[i] > 120. / ITC->rD_0_final.get_K() ) continue;

      diskprojs[i].init(i+1,rproj_[i],
			iphiprojdisk[i],irprojdisk[i],
			ITC->der_phiD_final.get_ival(),ITC->der_rD_final.get_ival(),
			phiprojdisk[i],rprojdisk[i],
			phiderdisk[i],rderdisk[i],
			phiprojdiskapprox[i],rprojdiskapprox[i],
			ITC->der_phiD_final.get_fval(),ITC->der_rD_final.get_fval());

      
    }

    
    if (writeTrackletParsDisk) {
      static ofstream out("trackletparsdisk.txt");
      out <<"Trackpars         "<<disk_
	  <<"   "<<rinv<<" "<<rinvapprox<<" "<<ITC->rinv_final.get_fval()
	  <<"   "<<phi0<<" "<<phi0approx<<" "<<ITC->phi0_final.get_fval()
	  <<"   "<<t<<" "<<tapprox<<" "<<ITC->t_final.get_fval()
	  <<"   "<<z0<<" "<<z0approx<<" "<<ITC->z0_final.get_fval()
	  <<endl;
    }
	    
    Tracklet* tracklet=new Tracklet(innerStub,NULL,outerStub,
				    innerFPGAStub,NULL,outerFPGAStub,
				    rinv,phi0,0.0,z0,t,
				    rinvapprox,phi0approx,0.0,
				    z0approx,tapprox,
				    irinv,iphi0,0,iz0,it,
				    layerprojs,
				    diskprojs,
				    true);
    
    if (debug1) {
      cout << "Found tracklet in disk = "<<disk_<<" "<<tracklet
	   <<" "<<iSector_<<endl;
    }
        
    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);
    
    if (tracklet->validProj(1)) {
      addLayerProj(tracklet,1);
    }
    
    if (tracklet->validProj(2)) {
      addLayerProj(tracklet,2);
    }
    
    for(unsigned int j=0;j<3;j++){
      if (tracklet->validProjDisk(sign*dproj_[j])) {
	addDiskProj(tracklet,sign*dproj_[j]);
      }
    }

    return true;
    
  }
  

  bool overlapSeeding(Stub* innerFPGAStub, L1TStub* innerStub, Stub* outerFPGAStub, L1TStub* outerStub){
    
    //Deal with overlap stubs here
    assert(outerFPGAStub->isBarrel());
    
    assert(innerFPGAStub->isDisk());
    
    int disk=innerFPGAStub->disk().value();
    
    if (debug1) {
      cout << "trying to make overlap tracklet disk_ = "<<disk_<<" "<<getName()<<endl;
    }
    
    //int sign=1;
    //if (disk_<0) sign=-1;
    
    double r1=innerStub->r();
    double z1=innerStub->z();
    double phi1=innerStub->phi();
    
    double r2=outerStub->r();
    double z2=outerStub->z();
    double phi2=outerStub->phi();
    
    //Protection... Should be handled cleaner
    //to avoid problem with floating point 
    //calculation and with overflows
    //in the integer calculation
    if (r1<r2+1.5) {
      //cout << "in overlap tracklet: radii wrong"<<endl;
      return false;
    }
    

    double rinv,phi0,t,z0;
	    
    double phiproj[3],zproj[3],phider[3],zder[3];
    double phiprojdisk[4],rprojdisk[4],phiderdisk[4],rderdisk[4];
    
    exacttrackletOverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			 rinv,phi0,t,z0,
			 phiproj,zproj,phider,zder,
			 phiprojdisk,rprojdisk,phiderdisk,rderdisk);
    
    
    //Truncates floating point positions to integer
    //representation precision
    if (useapprox) {
      phi1=innerFPGAStub->phiapprox(phimin_,phimax_);
      z1=innerFPGAStub->zapprox();
      r1=innerFPGAStub->rapprox();
	      
      phi2=outerFPGAStub->phiapprox(phimin_,phimax_);
      z2=outerFPGAStub->zapprox();
      r2=outerFPGAStub->rapprox();
    }

    LayerProjection layerprojs[3];
    DiskProjection diskprojs[4];

    double rinvapprox,phi0approx,tapprox,z0approx;
    double phiprojapprox[3],zprojapprox[3];
    double phiprojdiskapprox[4],rprojdiskapprox[4];

    IMATH_TrackletCalculatorOverlap *ITC;
    int ll = outerFPGAStub->layer().value()+1;
    if     (ll==1 && disk==1)  ITC = &ITC_L1F1;
    else if(ll==2 && disk==1)  ITC = &ITC_L2F1;
    else if(ll==1 && disk==-1) ITC = &ITC_L1B1;
    else if(ll==2 && disk==-1) ITC = &ITC_L2B1;
    else assert(0);
    
    ITC->r1.set_fval(r2-rmean[ll-1]);
    ITC->r2.set_fval(r1);
    int signt = t>0? 1 : -1;
    ITC->z1.set_fval(z2);
    ITC->z2.set_fval(z1-signt*zmean[abs(disk_)-1]);
    double sphi1 = phi1 - phioffset_;
    if(sphi1<0) sphi1 += 8*atan(1.);
    if(sphi1>8*atan(1.)) sphi1 -= 8*atan(1.);
    double sphi2 = phi2 - phioffset_;
    if(sphi2<0) sphi2 += 8*atan(1.);
    if(sphi2>8*atan(1.)) sphi2 -= 8*atan(1.);
    ITC->phi1.set_fval(sphi2);
    ITC->phi2.set_fval(sphi1);

    ITC->rproj0.set_fval(rmean[0]);
    ITC->rproj1.set_fval(rmean[1]);
    ITC->rproj2.set_fval(rmean[2]);
    
    ITC->zproj0.set_fval(t>0? zprojoverlap_[0] : -zprojoverlap_[0]);
    ITC->zproj1.set_fval(t>0? zprojoverlap_[1] : -zprojoverlap_[1]);
    ITC->zproj2.set_fval(t>0? zprojoverlap_[2] : -zprojoverlap_[2]);
    ITC->zproj3.set_fval(t>0? zprojoverlap_[3] : -zprojoverlap_[3]);
    
    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();
    ITC->phiD_3_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();
    ITC->rD_3_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the approximate results
    rinvapprox = ITC->rinv_final.get_fval();
    phi0approx = ITC->phi0_final.get_fval();
    tapprox    = ITC->t_final.get_fval();
    z0approx   = ITC->z0_final.get_fval();

    phiprojapprox[0] = ITC->phiL_0_final.get_fval();
    phiprojapprox[1] = ITC->phiL_1_final.get_fval();
    phiprojapprox[2] = ITC->phiL_2_final.get_fval();

    zprojapprox[0]   = ITC->zL_0_final.get_fval();
    zprojapprox[1]   = ITC->zL_1_final.get_fval();
    zprojapprox[2]   = ITC->zL_2_final.get_fval();

    phiprojdiskapprox[0] = ITC->phiD_0_final.get_fval();
    phiprojdiskapprox[1] = ITC->phiD_1_final.get_fval();
    phiprojdiskapprox[2] = ITC->phiD_2_final.get_fval();
    phiprojdiskapprox[3] = ITC->phiD_3_final.get_fval();

    rprojdiskapprox[0] = ITC->rD_0_final.get_fval();
    rprojdiskapprox[1] = ITC->rD_1_final.get_fval();
    rprojdiskapprox[2] = ITC->rD_2_final.get_fval();
    rprojdiskapprox[3] = ITC->rD_3_final.get_fval();

    //now binary

    int irinv,iphi0,it,iz0;
    int iphiproj[3],izproj[3];
    
    int iphiprojdisk[4],irprojdisk[4];
    
    int ir2=innerFPGAStub->ir();
    int iphi2=innerFPGAStub->iphi();
    int iz2=innerFPGAStub->iz();
      
    int ir1=outerFPGAStub->ir();
    int iphi1=outerFPGAStub->iphi();
    int iz1=outerFPGAStub->iz();
      
    //To get global precission
    ir1<<=(8-nbitsrL123);
    iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    iphi2<<=(nbitsphistubL456-nbitsphistubL123);

    ITC->r1.set_ival(ir1);
    ITC->r2.set_ival(ir2);
    ITC->z1.set_ival(iz1);
    ITC->z2.set_ival(iz2);
    ITC->phi1.set_ival(iphi1);
    ITC->phi2.set_ival(iphi2);
      
    ITC->rinv_final.calculate();
    ITC->phi0_final.calculate();
    ITC->t_final.calculate();
    ITC->z0_final.calculate();

    ITC->phiL_0_final.calculate();
    ITC->phiL_1_final.calculate();
    ITC->phiL_2_final.calculate();

    ITC->zL_0_final.calculate();
    ITC->zL_1_final.calculate();
    ITC->zL_2_final.calculate();

    ITC->phiD_0_final.calculate();
    ITC->phiD_1_final.calculate();
    ITC->phiD_2_final.calculate();
    ITC->phiD_3_final.calculate();

    ITC->rD_0_final.calculate();
    ITC->rD_1_final.calculate();
    ITC->rD_2_final.calculate();
    ITC->rD_3_final.calculate();

    ITC->der_phiL_final.calculate();
    ITC->der_zL_final.calculate();
    ITC->der_phiD_final.calculate();
    ITC->der_rD_final.calculate();

    //store the binary results
    irinv = ITC->rinv_final.get_ival();
    iphi0 = ITC->phi0_final.get_ival();
    it    = ITC->t_final.get_ival();
    iz0   = ITC->z0_final.get_ival();

    //"protection" from the original, reinterpreted
    if (iz0>= 1<<(ITC->z0_final.get_nbits()-1)) iz0=(1<<(ITC->z0_final.get_nbits()-1))-1;
    if (iz0<=-(1<<(ITC->z0_final.get_nbits()-1))) iz0=1-(1<<(ITC->z0_final.get_nbits()-1))-1; 
    if (irinv>= (1<<(ITC->rinv_final.get_nbits()-1))) irinv=(1<<(ITC->rinv_final.get_nbits()-1))-1;
    if (irinv<=-(1<<(ITC->rinv_final.get_nbits()-1))) irinv=1-(1<<(ITC->rinv_final.get_nbits()-1))-1; 

    iphiproj[0] = ITC->phiL_0_final.get_ival();
    iphiproj[1] = ITC->phiL_1_final.get_ival();
    iphiproj[2] = ITC->phiL_2_final.get_ival();
    
    izproj[0]   = ITC->zL_0_final.get_ival();
    izproj[1]   = ITC->zL_1_final.get_ival();
    izproj[2]   = ITC->zL_2_final.get_ival();

    
    bool success = true;
    if(!ITC->t_final.local_passes()) {
      if (debug1) {
	cout << "TrackletProcessor::OverlapSeeding t too large: "<<ITC->t_final.get_fval()<<endl;
      }
      success = false;
    }
    if(!ITC->rinv_final.local_passes()){
      if (debug1) {
	cout << "TrackletProcessor::OverlapSeeding irinv too large: "<<ITC->rinv_final.get_fval()<<endl;
       }
      success = false;
    }
    if (!ITC->z0_final.local_passes()) {
      if (debug1) {
	cout << "TrackletProcessor::OverlapSeeding Failed tracklet z0 cut "<<ITC->z0_final.get_fval()<<" r1="<<r1<<" "<<layer_<<endl;
      }
      success = false;
    }

    success = success && ITC->valid_trackpar.passes();

    if (!success) {
      if (debug1) {
	
	cout << "TrackletProcessor::OverlapSeeding rejected no success: "
	     <<ITC->valid_trackpar.passes()<<" rinv="
	     <<ITC->rinv_final.get_ival()*ITC->rinv_final.get_K()<<" eta="
	     <<asinh(ITC->t_final.get_ival()*ITC->t_final.get_K())<<" z0="
	     <<ITC->z0_final.get_ival()*ITC->z0_final.get_K()<<" phi0="
	     <<ITC->phi0_final.get_ival()*ITC->phi0_final.get_K()<<endl;
      }
      return false;
    }


    double phicrit=phi0approx-asin(0.5*rcrit*rinvapprox);
    int phicritapprox=iphi0-2*irinv;
    bool keep=(phicrit>phicritminmc)&&(phicrit<phicritmaxmc),
         keepapprox=(phicritapprox>phicritapproxminmc)&&(phicritapprox<phicritapproxmaxmc);
    if (debug1)
      if (keep && !keepapprox)
        cout << "TrackletProcessor::overlapSeeding tracklet kept with exact phicrit cut but not approximate, phicritapprox: " << phicritapprox << endl;
    if (!usephicritapprox) {
      if (!keep) return false;
    }
    else {
      if (!keepapprox) return false;
    }

    
    
    for(int i=0; i<3; ++i){

      //check that zproj is in range
      if (izproj[i]<-(1<<(nbitszprojL123-1))) continue;
      if (izproj[i]>=(1<<(nbitszprojL123-1))) continue;

      //check that phiproj is in range
      if (iphiproj[i]>=(1<<nbitsphistubL456)-1) continue;
      if (iphiproj[i]<=0) continue;

      //adjust bits for PS modules (no 2S modules in overlap seeds)
      iphiproj[i]>>=(nbitsphistubL456-nbitsphistubL123);
      
      layerprojs[i].init(i+1,rmean[i],
			 iphiproj[i],izproj[i],
			 ITC->der_phiL_final.get_ival(),ITC->der_zL_final.get_ival(),
			 phiproj[i],zproj[i],
			 phider[i],zder[i],
			 phiprojapprox[i],zprojapprox[i],
			 ITC->der_phiL_final.get_fval(),ITC->der_zL_final.get_fval());
      
    }

    iphiprojdisk[0] = ITC->phiD_0_final.get_ival();
    iphiprojdisk[1] = ITC->phiD_1_final.get_ival();
    iphiprojdisk[2] = ITC->phiD_2_final.get_ival();
    iphiprojdisk[3] = ITC->phiD_3_final.get_ival();

    irprojdisk[0]   = ITC->rD_0_final.get_ival();
    irprojdisk[1]   = ITC->rD_1_final.get_ival();
    irprojdisk[2]   = ITC->rD_2_final.get_ival();
    irprojdisk[3]   = ITC->rD_3_final.get_ival();

    for(int i=0; i<4; ++i){

      //check that phi projetion in range
      if (iphiprojdisk[i]<=0) continue;
      if (iphiprojdisk[i]>=(1<<nbitsphistubL123)-1) continue;

      //check that r projetion in range
      if(irprojdisk[i]<=0
	 || irprojdisk[i] > 120. / ITC->rD_0_final.get_K() ) continue;

      
      diskprojs[i].init(i+1,rproj_[i],
			iphiprojdisk[i],irprojdisk[i],
			ITC->der_phiD_final.get_ival(),ITC->der_rD_final.get_ival(),
			phiprojdisk[i],rprojdisk[i],
			phiderdisk[i],rderdisk[i],
			phiprojdiskapprox[i],rprojdiskapprox[i],
			ITC->der_phiD_final.get_fval(),ITC->der_rD_final.get_fval());
      
      
    }

    

    if (writeTrackletParsOverlap) {
      static ofstream out("trackletparsoverlap.txt");
      out <<"Trackpars "<<disk_
	  <<"   "<<rinv<<" "<<irinv<<" "<<ITC->rinv_final.get_fval()
	  <<"   "<<phi0<<" "<<iphi0<<" "<<ITC->phi0_final.get_fval()
	  <<"   "<<t<<" "<<it<<" "<<ITC->t_final.get_fval()
	  <<"   "<<z0<<" "<<iz0<<" "<<ITC->z0_final.get_fval()
	  <<endl;
    }
	      
    Tracklet* tracklet=new Tracklet(innerStub,NULL,outerStub,
				    innerFPGAStub,NULL,outerFPGAStub,
				    rinv,phi0,0.0,z0,t,
				    rinvapprox,phi0approx,0.0,
				    z0approx,tapprox,
				    irinv,iphi0,0,iz0,it,
				    layerprojs,
				    diskprojs,
				    false,true);
    
    if (debug1) {
      cout << "Found tracklet in overlap = "<<layer_<<" "<<disk_
	   <<" "<<tracklet<<" "<<iSector_<<endl;
    }
    
        
    tracklet->setTrackletIndex(trackletpars_->nTracklets());
    tracklet->setTCIndex(TCIndex_);
    
    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
      fout.close();
    }
    trackletpars_->addTracklet(tracklet);
    
    int layer=outerFPGAStub->layer().value()+1;
    
    if (layer==2) {
      if (tracklet->validProj(1)) {
	addLayerProj(tracklet,1);
      }
    }
    
    
    for(unsigned int disk=2;disk<6;disk++){
      if (layer==2 && disk==5 ) continue;
      if (tracklet->validProjDisk(disk)) {
	addDiskProj(tracklet,disk);
      }
    }

    return true;
    
  }
  
  
  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }

  void setVMPhiBin() {

    //cout << "setVMPhiBin"<<innervmstubs_.size()<<" "<<outervmstubs_.size()<<endl;
    
    if (innervmstubs_.size()!=outervmstubs_.size() ) return;

    for(unsigned int ivmmem=0;ivmmem<innervmstubs_.size();ivmmem++){
    
      unsigned int innerphibin=innervmstubs_[ivmmem]->phibin();
      unsigned int outerphibin=outervmstubs_[ivmmem]->phibin();
      
      unsigned phiindex=32*innerphibin+outerphibin;


      if (phitable_.find(phiindex)!=phitable_.end()) continue;
      
      innervmstubs_[ivmmem]->setother(outervmstubs_[ivmmem]);
      outervmstubs_[ivmmem]->setother(innervmstubs_[ivmmem]);


    
      if ((layer_==1 && disk_==0)||
	  (layer_==2 && disk_==0)||
	  (layer_==3 && disk_==0)||
	  (layer_==5 && disk_==0)){
      
	innerphibits_=nfinephibarrelinner;
	outerphibits_=nfinephibarrelouter;
	
	int innerphibins=(1<<innerphibits_);
	int outerphibins=(1<<outerphibits_);
	
	double innerphimin, innerphimax;
	innervmstubs_[ivmmem]->getPhiRange(innerphimin,innerphimax);
	double rinner=rmean[layer_-1];
	
	double outerphimin, outerphimax;
	outervmstubs_[ivmmem]->getPhiRange(outerphimin,outerphimax);
	double router=rmean[layer_];
	
	double phiinner[2];
	double phiouter[2];
	
	std::vector<bool> vmbendinner;
	std::vector<bool> vmbendouter;
	unsigned int nbins1=8;
	if (layer_>=4) nbins1=16;
	for (unsigned int i=0;i<nbins1;i++) {
	  vmbendinner.push_back(false);
	}
	
	unsigned int nbins2=8;
	if (layer_>=3) nbins2=16;
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
	    
	    //cout << "Fill: "<<getName()<<" "<<phiindex<<endl;
	    phitable_[phiindex].push_back(rinvmin<rinvcutte);
	    
	    int nbins1=8;
	    if (layer_>=4) nbins1=16;
	    for(int ibend=0;ibend<nbins1;ibend++) {
	      double bend=Stub::benddecode(ibend,layer_<=3); 
	      
	      bool passinner=bend-bendinnermin>-bendcut&&bend-bendinnermax<bendcut;	    
	      if (passinner) vmbendinner[ibend]=true;
	      pttableinner_[phiindex].push_back(passinner);
	      
	    }
	    
	    int nbins2=8;
	    if (layer_>=3) nbins2=16;
	    for(int ibend=0;ibend<nbins2;ibend++) {
	      double bend=Stub::benddecode(ibend,layer_<=2); 
	      
	      bool passouter=bend-bendoutermin>-bendcut&&bend-bendoutermax<bendcut;
	      if (passouter) vmbendouter[ibend]=true;
	      pttableouter_[phiindex].push_back(passouter);
	      
	    }   
	  }
	}

	innervmstubs_[ivmmem]->setbendtable(vmbendinner);
	outervmstubs_[ivmmem]->setbendtable(vmbendouter);
      
	if (iSector_==0&&writeTETables) writeTETable();
      
      }

      if ((disk_==1 && layer_==0)||
	  (disk_==3 && layer_==0)){
	
	innerphibits_=nfinephidiskinner;
	outerphibits_=nfinephidiskouter;
	
	
	int outerrbits=3;
	
	int outerrbins=(1<<outerrbits);
	int innerphibins=(1<<innerphibits_);
	int outerphibins=(1<<outerphibits_);
	
	double innerphimin, innerphimax;
	innervmstubs_[ivmmem]->getPhiRange(innerphimin,innerphimax);
	
	double outerphimin, outerphimax;
	outervmstubs_[ivmmem]->getPhiRange(outerphimin,outerphimax);
	
	
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
		    double rinner=router[i3]*zmean[disk_-1]/zmean[disk_];
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
	      
	      phitable_[phiindex].push_back(rinvmin<rinvcutte);


	      for(int ibend=0;ibend<8;ibend++) {
		double bend=Stub::benddecode(ibend,true); 
		
		bool passinner=bend-bendinnermin>-bendcutdisk&&bend-bendinnermax<bendcutdisk;	    
		if (passinner) vmbendinner[ibend]=true;
		pttableinner_[phiindex].push_back(passinner);
		
	      }
	      
	      for(int ibend=0;ibend<8;ibend++) {
		double bend=Stub::benddecode(ibend,true); 
		
		bool passouter=bend-bendoutermin>-bendcutdisk&&bend-bendoutermax<bendcutdisk;
		if (passouter) vmbendouter[ibend]=true;
		pttableouter_[phiindex].push_back(passouter);
		
	      }
	    
	    }
	  }
	}
	
	innervmstubs_[ivmmem]->setbendtable(vmbendinner);
	outervmstubs_[ivmmem]->setbendtable(vmbendouter);
	
	if (iSector_==0&&writeTETables) writeTETable();
	
      } else if (disk_==1 && (layer_==1 || layer_==2)) {
	
	innerphibits_=nfinephioverlapinner;
	outerphibits_=nfinephioverlapouter;
	unsigned int nrbits=5;
	
	int innerphibins=(1<<innerphibits_);
	int outerphibins=(1<<outerphibits_);
	
	double innerphimin, innerphimax;
	innervmstubs_[ivmmem]->getPhiRange(innerphimin,innerphimax);
	
	double outerphimin, outerphimax;
	outervmstubs_[ivmmem]->getPhiRange(outerphimin,outerphimax);
	
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
		    double rinner=rmean[layer_-1];
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
	    
	      phitable_[phiindex].push_back(rinvmin<rinvcutte);
	      
	      
	      for(int ibend=0;ibend<8;ibend++) {
		double bend=Stub::benddecode(ibend,true); 
		
		bool passinner=bend-bendinnermin>-bendcut&&bend-bendinnermax<bendcut;	    
		if (passinner) vmbendinner[ibend]=true;
		pttableinner_[phiindex].push_back(passinner);
		
	      }
	      
	      for(int ibend=0;ibend<8;ibend++) {
		double bend=Stub::benddecode(ibend,true); 
		
		bool passouter=bend-bendoutermin>-bendcut&&bend-bendoutermax<bendcut;
		if (passouter) vmbendouter[ibend]=true;
		pttableouter_[phiindex].push_back(passouter);
		
	      }
	    }
	  }
	}
	
    
	innervmstubs_[ivmmem]->setbendtable(vmbendinner);
	outervmstubs_[ivmmem]->setbendtable(vmbendouter);
	
	if (iSector_==0&&writeTETables) writeTETable();
	
	
      }
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
    //for(unsigned int i=0;i<phitable_.size();i++){
    //  if (i!=0) outptcut<<","<<endl;
    //  outptcut << phitable_[i];
    //}
    outptcut <<endl<<"};"<<endl;
    outptcut.close();

    ofstream outstubptinnercut;
    outstubptinnercut.open(getName()+"_stubptinnercut.txt");
    outstubptinnercut << "{"<<endl;
    //for(unsigned int i=0;i<pttableinner_.size();i++){
    //  if (i!=0) outstubptinnercut<<","<<endl;
    //  outstubptinnercut << pttableinner_[i];
    //}
    outstubptinnercut <<endl<<"};"<<endl;
    outstubptinnercut.close();
    
    ofstream outstubptoutercut;
    outstubptoutercut.open(getName()+"_stubptoutercut.txt");
    outstubptoutercut << "{"<<endl;
    //for(unsigned int i=0;i<pttableouter_.size();i++){
    //  if (i!=0) outstubptoutercut<<","<<endl;
    //  outstubptoutercut << pttableouter_[i];
    //}
    outstubptoutercut <<endl<<"};"<<endl;
    outstubptoutercut.close();

    
  }
  

  
    
private:
  
  int iTC_;
  int iSeed_;
  int TCIndex_;
  int layer_;
  int disk_;
  double phimin_;
  double phimax_;
  double phioffset_;
  double rproj_[4];
  int lproj_[4];
  double zproj_[3];
  int dproj_[3];

  unsigned int maxtracklet_; //maximum numbor of tracklets that be stored
  
  double zprojoverlap_[4];

  vector<VMStubsTEMemory*> innervmstubs_;
  vector<VMStubsTEMemory*> outervmstubs_;
  
  vector<AllStubsMemory*> innerallstubs_;
  vector<AllStubsMemory*> outerallstubs_;

  TrackletParametersMemory* trackletpars_;

  //First index is layer/disk second is phi region
  vector<vector<TrackletProjectionsMemory*> > trackletprojlayers_;
  vector<vector<TrackletProjectionsMemory*> > trackletprojdisks_;

  bool extra_;

  
  map<unsigned int, vector<bool> > phitable_;
  map<unsigned int, vector<bool> > pttableinner_;
  map<unsigned int, vector<bool> > pttableouter_;
  
  int innerphibits_;
  int outerphibits_;


public:
  static IMATH_TrackletCalculator ITC_L1L2;
  static IMATH_TrackletCalculator ITC_L2L3;
  static IMATH_TrackletCalculator ITC_L3L4;
  static IMATH_TrackletCalculator ITC_L5L6;
  
  static IMATH_TrackletCalculatorDisk ITC_F1F2;
  static IMATH_TrackletCalculatorDisk ITC_F3F4;
  static IMATH_TrackletCalculatorDisk ITC_B1B2;
  static IMATH_TrackletCalculatorDisk ITC_B3B4;

  static IMATH_TrackletCalculatorOverlap ITC_L1F1;
  static IMATH_TrackletCalculatorOverlap ITC_L2F1;
  static IMATH_TrackletCalculatorOverlap ITC_L1B1;
  static IMATH_TrackletCalculatorOverlap ITC_L2B1;
    
};

IMATH_TrackletCalculator TrackletProcessor::ITC_L1L2{1,2};
IMATH_TrackletCalculator TrackletProcessor::ITC_L2L3{2,3};
IMATH_TrackletCalculator TrackletProcessor::ITC_L3L4{3,4};
IMATH_TrackletCalculator TrackletProcessor::ITC_L5L6{5,6};

IMATH_TrackletCalculatorDisk TrackletProcessor::ITC_F1F2{1,2};
IMATH_TrackletCalculatorDisk TrackletProcessor::ITC_F3F4{3,4};
IMATH_TrackletCalculatorDisk TrackletProcessor::ITC_B1B2{-1,-2};
IMATH_TrackletCalculatorDisk TrackletProcessor::ITC_B3B4{-3,-4};

IMATH_TrackletCalculatorOverlap TrackletProcessor::ITC_L1F1{1,1};
IMATH_TrackletCalculatorOverlap TrackletProcessor::ITC_L2F1{2,1};
IMATH_TrackletCalculatorOverlap TrackletProcessor::ITC_L1B1{1,-1};
IMATH_TrackletCalculatorOverlap TrackletProcessor::ITC_L2B1{2,-1};

#endif
