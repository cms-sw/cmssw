#ifndef TRACKDERTABLE_H
#define TRACKDERTABLE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include "TrackDer.h"

using namespace std;

class TrackDerTable{

public:

  TrackDerTable() {

    Nlay_=6;
    Ndisk_=5;

    LayerMemBits_=6;
    DiskMemBits_=7;
    
    LayerDiskMemBits_=18;

    alphaBits_=alphaBitsTable;

    nextLayerValue_=0;
    nextDiskValue_=0;
    nextLayerDiskValue_=0;
    lastMultiplicity_=(1<<(3*alphaBits_));


    for(int i=0;i<(1<<Nlay_);i++){
      LayerMem_.push_back(-1);
    }

    for(int i=0;i<(1<<(2*Ndisk_));i++){
      DiskMem_.push_back(-1);
    }

    for(int i=0;i<(1<<(LayerMemBits_+DiskMemBits_));i++){
      LayerDiskMem_.push_back(-1);
    }
   
  }


  ~TrackDerTable() {

  }

  TrackDer* getDerivatives(int index){
    return &derivatives_[index];
  }

  TrackDer* getDerivatives(unsigned int layermask, 
			       unsigned int diskmask,
			       unsigned int alphaindex,
			       unsigned int rinvindex){
    int index=getIndex(layermask,diskmask);
    //if (index<0||index!=17984||alphaindex!=20) {
    if (index<0) {
      return 0;
    }
    //cout << "getDerivatives index alphaindex "<<index<<" "<<alphaindex<<" "<<rinvindex<<endl;
    return &derivatives_[index+alphaindex*(1<<nrinvBitsTable)+rinvindex];
  }


  int getIndex(unsigned int layermask,unsigned int diskmask) {

    assert(layermask<LayerMem_.size());

    assert(diskmask<DiskMem_.size());

    int layercode=LayerMem_[layermask];
    int diskcode=DiskMem_[diskmask];

    if (diskcode<0||layercode<0) {
      cout << "layermask diskmask : "<<layermask<<" "<<diskmask<<endl;
      return -1;
    }

    assert(layercode>=0);
    assert(layercode<(1<<LayerMemBits_));
    assert(diskcode>=0);
    assert(diskcode<(1<<DiskMemBits_));

    int layerdiskaddress=layercode+(diskcode<<LayerMemBits_);

    assert(layerdiskaddress>=0);
    assert(layerdiskaddress<(1<<(LayerMemBits_+DiskMemBits_)));

    int address=LayerDiskMem_[layerdiskaddress];

    if (address<0) {
      cout << "layermask diskmask : "<<layermask<<" "<<diskmask<<endl;
      return -1;
    }

    assert(address>=0);
    //cout << "address LayerDiskMemBits_ : "<<address<<" "<<LayerDiskMemBits_<<endl;
    assert(address<(1<<LayerDiskMemBits_));

    return address;

  }

  void addEntry(unsigned int layermask, unsigned int diskmask, int multiplicity, int nrinv){

    //cout << "layermask diskmatch "<<layermask<<" "<<diskmask<<endl;
    
    assert(multiplicity<=(1<<(3*alphaBits_)));

    assert(layermask<(unsigned int)(1<<Nlay_));

    assert(diskmask<(unsigned int)(1<<(2*Ndisk_)));

    if (LayerMem_[layermask]==-1) {
      LayerMem_[layermask]=nextLayerValue_++;
    }
    if (DiskMem_[diskmask]==-1) {
      DiskMem_[diskmask]=nextDiskValue_++;
    }

    int layercode=LayerMem_[layermask];
    int diskcode=DiskMem_[diskmask];

    assert(layercode>=0);
    assert(layercode<(1<<LayerMemBits_));
    assert(diskcode>=0);
    assert(diskcode<(1<<DiskMemBits_));

    int layerdiskaddress=layercode+(diskcode<<LayerMemBits_);

    assert(layerdiskaddress>=0);
    assert(layerdiskaddress<(1<<(LayerMemBits_+DiskMemBits_)));

    int address=LayerDiskMem_[layerdiskaddress];

    if (address!=-1) {
      cout << "Duplicate entry:  layermask="
	   <<layermask<<" diskmaks="<<diskmask<<endl;
    }

    assert(address==-1);  //Should not already have this one!

    LayerDiskMem_[layerdiskaddress]=nextLayerDiskValue_;

    nextLayerDiskValue_+=multiplicity*nrinv;

    lastMultiplicity_=multiplicity*nrinv;

    for(int i=0;i<multiplicity;i++) {
      for (int irinv=0;irinv<nrinv;irinv++) {
	TrackDer tmp;
	tmp.setIndex(layermask,diskmask,i,irinv);
	derivatives_.push_back(tmp);
      }
    }

  }

  void readPatternFile(std::string fileName){

    ifstream in(fileName.c_str());
    cout<<"reading fit pattern file "<<fileName<<"\n";
    cout<<"  flags (good/eof/fail/bad): "<<in.good()<<" "<<in.eof()<<" "<<in.fail()<<" "<<in.bad()<<"\n"; 

    while (in.good()) {

      std::string layerstr,diskstr;
      int multiplicity;

      in >>layerstr>>diskstr>>multiplicity;

      //correct multiplicity if you dont want 3 bits of alpha.
      if (alphaBits_==2) {
	if (multiplicity==8) multiplicity=4;
	if (multiplicity==64) multiplicity=16;
	if (multiplicity==512) multiplicity=64;
      }

      if (alphaBits_==1) {
	if (multiplicity==8) multiplicity=2;
	if (multiplicity==64) multiplicity=4;
	if (multiplicity==512) multiplicity=8;
      }
	  
      if (!in.good()) continue;
      
      char** tmpptr=0;

      int layers=strtol(layerstr.c_str(), tmpptr, 2); 
      int disks=strtol(diskstr.c_str(), tmpptr, 2); 

      //cout << "adding: "<<layers<<" "<<disks<<" "<<multiplicity<<endl;
   
      addEntry(layers,disks,multiplicity,(1<<nrinvBitsTable));

    }

  }

  int getEntries() const {
    return nextLayerDiskValue_;
  }

  void fillTable() {

    int nentries=getEntries();

    
    
    for (int i=0;i<nentries;i++){
      TrackDer& der=derivatives_[i];
      int layermask=der.getLayerMask();
      int diskmask=der.getDiskMask();
      int alphamask=der.getAlphaMask();
      int irinv=der.getirinv();

      double rinv=(irinv-((1<<(nrinvBitsTable-1))-0.5))*0.0057/(1<<(nrinvBitsTable-1));
      
      bool print=false;
      //bool print=getIndex(layermask,diskmask)==300 && alphamask==1;
      //print=false;

      if (print) {
	cout << "PRINT i "<<i<<" "<<layermask<<" "<<diskmask<<" "
	     <<alphamask<<" "<<print<<endl;
      }

      int nlayers=0;
      //int layers[6];
      double r[6];

      for (unsigned l=0;l<6;l++){
	if (layermask&(1<<(5-l))) {
	  //layers[nlayers]=l+1;
	  r[nlayers]=rmean[l];
	  //cout << "Hit in layer "<<layers[nlayers]<<" "<<r[nlayers]<<endl;
	  nlayers++;  
	}
      }

      int ndisks=0;
      //int disks[5];
      double z[5];
      double alpha[5];

      double t=gett(diskmask,layermask);
      //double rinv=0.00000001;
      
      for (unsigned d=0;d<5;d++){
	if (diskmask&(3<<(2*(4-d)))) {
	  //disks[ndisks]=d+1;
	  z[ndisks]=zmean[d];
	  alpha[ndisks]=0.0;
	  double r=zmean[d]/t;
	  double r2=r*r;
	  if (diskmask&(1<<(2*(4-d)))) {
	    if (alphaBits_==3) {
	      int ialpha=alphamask&7;
	      alphamask=alphamask>>3;
	      //double r=zmean[d]/t;
	      alpha[ndisks]=4.57*(ialpha-3.5)/4.0/r2;
	      //alpha[ndisks]=480*0.009*(ialpha-3.5)/(4.0*r*r);
	      if (print) cout << "PRINT 3 alpha ialpha : "<<alpha[ndisks]<<" "<<ialpha<<endl;
	    }
	    if (alphaBits_==2) {
	      int ialpha=alphamask&3;
	      alphamask=alphamask>>2;
	      //double r=zmean[d]/t;
	      alpha[ndisks]=4.57*(ialpha-1.5)/2.0/r2;
	      //alpha[ndisks]=480*0.009*(ialpha-1.5)/(4.0*r*r);
	    }
	    if (alphaBits_==1) {
	      int ialpha=alphamask&1;
	      alphamask=alphamask>>1;
	      //double r=zmean[d]/t;
	      alpha[ndisks]=4.57*(ialpha-0.5)/r2;
	      //alpha[ndisks]=480*0.009*(ialpha-0.5)/(4.0*r*r);
	      if (print) cout << "PRINT 1 alpha ialpha : "<<alpha[ndisks]<<" "<<ialpha<<endl;
	    }
	  }
	  ndisks++;  
	}
      }


      double D[4][12];
      int iD[4][12];
      double MinvDt[4][12];
      double MinvDtDelta[4][12];
      int iMinvDt[4][12];
      double sigma[12];
      double kfactor[12];


      if (print) {
	cout << "PRINT ndisks alpha[0] z[0] t: "<<ndisks<<" "<<alpha[0]<<" "<<z[0]<<" "<<t<<endl;
	for(int iii=0;iii<nlayers;iii++) {
	  cout << "PRINT iii r: "<<iii<<" "<<r[iii]<<endl;
	}
      }
      
      calculateDerivatives(nlayers,r,ndisks,z,alpha,t,rinv,D,iD,MinvDt,iMinvDt,sigma,kfactor);

      double delta=0.1;

      //should be merged in calculateDerivatves???
      for (int i=0;i<nlayers;i++){
	if (r[i]>60.0) continue;

	r[i]+=delta;

	calculateDerivatives(nlayers,r,ndisks,z,alpha,t,
			     rinv,D,iD,MinvDtDelta,iMinvDt,sigma,kfactor);
	    
	for (int ii=0;ii<nlayers;ii++){
	  if (r[ii]>60.0) continue;
	  double tder=(MinvDtDelta[2][2*ii+1]-MinvDt[2][2*ii+1])/delta;
	  int itder=(1<<(fittbitshift+rcorrbits))*tder*kr*kzproj/ktpars;
	  double zder=(MinvDtDelta[3][2*ii+1]-MinvDt[3][2*ii+1])/delta;
	  int izder=(1<<(fitz0bitshift+rcorrbits))*zder*kr*kzproj/kzpars;
	  der.settdzcorr(i,ii,tder);
	  der.setz0dzcorr(i,ii,zder);
	  der.setitdzcorr(i,ii,itder);
	  der.setiz0dzcorr(i,ii,izder);
	}
	      
	r[i]-=delta;
      }
	    

      
      if (print) {
	cout << "iMinvDt table build : "<<iMinvDt[0][10]<<" "<<iMinvDt[1][10]<<" "
	     <<iMinvDt[2][10]<<" "<<iMinvDt[3][10]<<" "<<t<<" "<<nlayers<<" "<<ndisks<<endl;
	cout << "alpha :";
	for (int iii=0;iii<ndisks;iii++) cout <<" "<<alpha[iii];
	cout << endl;
	cout << "z :";
	for (int iii=0;iii<ndisks;iii++) cout <<" "<<z[iii];
	cout << endl;

      }

      if (print) {
	cout << "PRINT nlayers ndisks : "<<nlayers<<" "<<ndisks<<endl;
      }
    
      for(int j=0;j<nlayers+ndisks;j++){
	/*
	if (print) {
	  cout << "Table "<<endl;
	  cout << MinvDt[0][2*j] <<" "
	       << MinvDt[1][2*j] <<" "
	       << MinvDt[2][2*j] <<" "
	       << MinvDt[3][2*j] <<" "
	       <<endl;
	  cout << MinvDt[0][2*j+1] <<" "
	       << MinvDt[1][2*j+1] <<" "
	       << MinvDt[2][2*j+1] <<" "
	       << MinvDt[3][2*j+1] <<" "
	       <<endl;
	}
	*/	

	der.sett(t);
	
	//integer
	assert(fabs(iMinvDt[0][2*j])<(1<<23));
	assert(fabs(iMinvDt[0][2*j+1])<(1<<23));
	assert(fabs(iMinvDt[1][2*j])<(1<<23));
	assert(fabs(iMinvDt[1][2*j+1])<(1<<23));
	assert(fabs(iMinvDt[2][2*j])<(1<<19));
	assert(fabs(iMinvDt[2][2*j+1])<(1<<19));
	assert(fabs(iMinvDt[3][2*j])<(1<<19));
	assert(fabs(iMinvDt[3][2*j+1])<(1<<19));

	if (print) {
	  cout << "PRINT i "<<i<<" "<<j<<" "<<iMinvDt[1][2*j]<<" " 
	       <<fabs(iMinvDt[1][2*j])<<endl;
	}

	
	der.setirinvdphi(j,iMinvDt[0][2*j]); 
	der.setirinvdzordr(j,iMinvDt[0][2*j+1]); 
	der.setiphi0dphi(j,iMinvDt[1][2*j]); 
	der.setiphi0dzordr(j,iMinvDt[1][2*j+1]); 
	der.setitdphi(j,iMinvDt[2][2*j]); 
	der.setitdzordr(j,iMinvDt[2][2*j+1]); 
	der.setiz0dphi(j,iMinvDt[3][2*j]); 
	der.setiz0dzordr(j,iMinvDt[3][2*j+1]); 
	//floating point
	der.setrinvdphi(j,MinvDt[0][2*j]); 
	der.setrinvdzordr(j,MinvDt[0][2*j+1]); 
	der.setphi0dphi(j,MinvDt[1][2*j]); 
	der.setphi0dzordr(j,MinvDt[1][2*j+1]); 
	der.settdphi(j,MinvDt[2][2*j]); 
	der.settdzordr(j,MinvDt[2][2*j+1]); 
	der.setz0dphi(j,MinvDt[3][2*j]); 
	der.setz0dzordr(j,MinvDt[3][2*j+1]); 
      }



    }

    
    if (writeFitDerTable) {
	  /*
      for (unsigned int seedlayer=1;seedlayer<=5;seedlayer+=2) {
	for(unsigned int j=0;j<8;j++) {
	  std::ostringstream oss;
	  oss << "FitDerTable_L"<<seedlayer<<"_"<<j+1<<".txt";
	  std::string fname=oss.str();
	  ofstream out(fname.c_str());
	  for(unsigned int imask=0;imask<16;imask++) {
	    int nmatches=0;
	    for (unsigned int i=0;i<4;i++) {
	      if (((1<<i)&imask)!=0) nmatches++;
	    }
	    unsigned int ifullmask=0;
	    if (seedlayer==5) ifullmask=(imask<<2)+3;
	    if (seedlayer==3) ifullmask=(imask&12)*4+12+(imask&3);
	    if (seedlayer==1) ifullmask=imask+48;
	    
	    unsigned int layer=j/2;
	    if (seedlayer==1) layer+=2;
	    if (seedlayer==3&&layer>1) layer+=2;
	    assert(layer<6);

	    bool hitlayer=((1<<(5-layer))&ifullmask)!=0;

	    TrackDer* der=0;
	    if (hitlayer&&(nmatches>1)) {
	      der=getDerivatives(ifullmask,0,0); 
	    }
	    

	   
	    if (der!=0) {

	      unsigned int index=0;
	      for (unsigned int i=0;i<layer;i++) {
		if (((1<<(5-i))&ifullmask)!=0) index++;
		//cout << "layer index : "<<layer<<" "<<index<<endl;
	      }
	      
	      //cout << "seedlayer j imask ifullmask layer index "<<seedlayer<<" "
	      //		   <<j<<" "<<imask<<" "<<ifullmask<<" "<<layer<<" "
	      //   <<index<<endl;


	      //for (unsigned int i=0;i<6;i++) {
	      //	cout <<der->getirinvdphi(i)<<" ";
	      //}
	      //cout << endl;

	      assert(index<6);



	      FPGAWord tmp1,tmp2,tmp3,tmp4;
	      if (j%2==0) {
		tmp1.set(der->getirinvdphi(index),15,false); 
		tmp2.set(der->getiphi0dphi(index),15,false); 
		tmp3.set(der->getitdphi(index),15,false); 
		tmp4.set(der->getiz0dphi(index),15,false); 
	      } else {
		tmp1.set(der->getirinvdzordr(index),15,false); 
		tmp2.set(der->getiphi0dzordr(index),15,false); 
		tmp3.set(der->getitdzordr(index),15,false); 
		tmp4.set(der->getiz0dzordr(index),15,false); 
	      }
	      //FPGAWord tmp;
	      //tmp.set(imask,4,true);
	      out //<< tmp.str()<<" "<<index<<" "
		  <<tmp1.str()<<tmp2.str()<<tmp3.str()<<tmp4.str()<<endl;
	   
	    } else {
	      //FPGAWord tmp;
	      //tmp.set(ifullmask,6,true);
	      out //<<tmp.str()<<" "<<hitlayer
		<<"000000000000000000000000000000000000000000000000000000000000"<<endl;
	    }
	  }
	}
      }
	  */

	  // New table format with disks
	  ofstream outL("FitDerTableNew_LayerMem.txt");
	  for (unsigned int i=0;i<LayerMem_.size();i++){
	    FPGAWord tmp;
		int tmp1=LayerMem_[i];
		if (tmp1<0) tmp1=(1<<6)-1;
		cout << "i LayerMem_ : "<<i<<" "<<tmp1<<endl;
		tmp.set(tmp1,6,true,__LINE__,__FILE__);
		outL << tmp.str() <<endl;
	  }
	  outL.close();

	  ofstream outD("FitDerTableNew_DiskMem.txt");
	  for (unsigned int i=0;i<DiskMem_.size();i++){
	    int tmp1=DiskMem_[i];
		if (tmp1<0) tmp1=(1<<7)-1;
		FPGAWord tmp;
		tmp.set(tmp1,7,true,__LINE__,__FILE__);
		outD << tmp.str() <<endl;
	  }
	  outD.close();

	  ofstream outLD("FitDerTableNew_LayerDiskMem.txt");
	  for (unsigned int i=0;i<LayerDiskMem_.size();i++){
	    int tmp1=LayerDiskMem_[i];
		if (tmp1<0) tmp1=(1<<10)-1;
		FPGAWord tmp;
		tmp.set(tmp1,10,true,__LINE__,__FILE__);
		outLD << tmp.str() <<endl;
	  }
	  outLD.close();

	  unsigned int nderivatives = derivatives_.size();
	  cout << "nderivatives = " << nderivatives << endl;

	  const string seedings[]={"L1L2","L3L4","L5L6","D1D2","D3D4","D1L1","D1L2"};
	  const string prefix = "FitDerTableNew_";

	  // open files for derivative tables
	  ofstream outrinvdphi[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Rinvdphi_" + seedings[i] + ".txt";
	    outrinvdphi[i].open(fname.c_str());
	  }
	  
	  ofstream outrinvdzordr[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Rinvdzordr_" + seedings[i] + ".txt";
	    outrinvdzordr[i].open(fname.c_str());
	  }
	  
	  ofstream outphi0dphi[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Phi0dphi_" + seedings[i] + ".txt";
	    outphi0dphi[i].open(fname.c_str());
	  }
	  
	  ofstream outphi0dzordr[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Phi0dzordr_" + seedings[i] + ".txt";
	    outphi0dzordr[i].open(fname.c_str());
	  }
	  
	  ofstream outtdphi[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Tdphi_" + seedings[i] + ".txt";
	    outtdphi[i].open(fname.c_str());
	  }
	  
	  ofstream outtdzordr[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Tdzordr_" + seedings[i] + ".txt";
	    outtdzordr[i].open(fname.c_str());
	  }
	  
	  ofstream outz0dphi[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    const string fname = prefix + "Z0dphi_" + seedings[i] + ".txt";
	    outz0dphi[i].open(fname.c_str());
	  }
	  
	  ofstream outz0dzordr[7];
	  for (unsigned int i = 0; i < 7; ++i) {
	    string fname = prefix + "Z0dzordr_" + seedings[i] + ".txt";
	    outz0dzordr[i].open(fname.c_str());
	  }

	  for (auto & der : derivatives_) {
	    unsigned int layerhits = der.getLayerMask(); // 6 bits layer hit pattern
	    unsigned int diskmask = der.getDiskMask(); // 10 bits disk hit pattern
		unsigned int diskhits = 0;
		if (diskmask&(3<<8)) diskhits += 16;
		if (diskmask&(3<<6)) diskhits += 8;
		if (diskmask&(3<<4)) diskhits += 4;
		if (diskmask&(3<<2)) diskhits += 2;
		if (diskmask&(3<<0)) diskhits += 1;
		assert(diskhits < 32);  // 5 bits
		unsigned int hits = (layerhits<<5) + diskhits; // 11 bits hit pattern
		assert(hits < 4096);

		// loop over all seedings
		int i = 0;  // seeding index
		for (const string & seed : seedings) {
			
		  unsigned int iseed1 = 0;
		  unsigned int iseed2 = 0;
		  // check if the seeding is good for the current hit pattern
		  if (seed == "L1L2") {iseed1=1; iseed2=2;}
		  if (seed == "L3L4") {iseed1=3; iseed2=4;}
		  if (seed == "L5L6") {iseed1=5; iseed2=6;}
		  if (seed == "D1D2") {iseed1=7; iseed2=8;}
		  if (seed == "D3D4") {iseed1=9; iseed2=10;}
		  if (seed == "D1L1") {iseed1=7; iseed2=1;}
		  if (seed == "D1L2") {iseed1=7; iseed2=2;}

		  bool goodseed = (hits&(1<<(11-iseed1))) and (hits&(1<<(11-iseed2)));

		  int itmprinvdphi[4] = {9999999,9999999,9999999,9999999};
		  int itmprinvdzordr[4] = {9999999,9999999,9999999,9999999};
		  int itmpphi0dphi[4] = {9999999,9999999,9999999,9999999};
		  int itmpphi0dzordr[4] = {9999999,9999999,9999999,9999999};
		  int itmptdphi[4] = {9999999,9999999,9999999,9999999};
		  int itmptdzordr[4] = {9999999,9999999,9999999,9999999};
		  int itmpz0dphi[4] = {9999999,9999999,9999999,9999999};
		  int itmpz0dzordr[4] = {9999999,9999999,9999999,9999999};
		  
		  // loop over bits in hit pattern
		  int ider = 0;
		  if (goodseed) {
		    for (unsigned int ihit = 1; ihit < 12; ++ihit) {
			  
		      // skip seeding layers
		      if (ihit == iseed1 or ihit == iseed2) {
			    ider++;
			    continue;
			  }
			  // skip if no hit
			  if (not (hits&(1<<(11-ihit))) ) continue;

			  int inputI = -1;
			  if (seed == "L1L2") {
			    if (ihit == 3 or ihit == 10) inputI = 0;  // L3 or D4
			    if (ihit == 4 or ihit == 9)  inputI = 1;  // L4 or D3
			    if (ihit == 5 or ihit == 8)  inputI = 2;  // L5 or D2
			    if (ihit == 6 or ihit == 7)  inputI = 3;  // L6 or D1
			  }
			  else if (seed == "L3L4") {
			    if (ihit == 1) inputI = 0;  // L1
			    if (ihit == 2) inputI = 1;  // L2
			    if (ihit == 5 or ihit == 8) inputI = 2;  // L5 or D2
			    if (ihit == 6 or ihit == 7) inputI = 3;  // L6 or D1
			  }
			  else if (seed == "L5L6") {
			    if (ihit == 1) inputI = 0;  // L1 
			    if (ihit == 2) inputI = 1;  // L2
			    if (ihit == 3) inputI = 2;  // L3 
			    if (ihit == 4) inputI = 3;  // L4 
			  }
			  else if (seed == "D1D2") {
			    if (ihit == 1)  inputI = 0;  // L1 
			    if (ihit == 9)  inputI = 1;  // D3
			    if (ihit == 10) inputI = 2;  // D4
			    if (ihit == 2 or ihit == 11) inputI = 3;  // L2 or D5
			  }
			  else if (seed == "D3D4") {
			    if (ihit == 1) inputI = 0;  // L1 
			    if (ihit == 7) inputI = 1;  // D1
			    if (ihit == 8) inputI = 2;  // D2
			    if (ihit == 2 or ihit == 11) inputI = 3;  // L2 or D5
			  }
			  else if (seed == "D1L1" or "D1L2") {
			    if (ihit == 8)  inputI = 0;  // D2
			    if (ihit == 9)  inputI = 1;  // D3
			    if (ihit == 10) inputI = 2;  // D4
			    if (ihit == 11) inputI = 3;  // D5
			  }
			  if (inputI>=0 and inputI<4) {
			    itmprinvdphi[inputI] = der.getirinvdphi(ider);
			    itmprinvdzordr[inputI] = der.getirinvdzordr(ider);
			    itmpphi0dphi[inputI] = der.getiphi0dphi(ider);
			    itmpphi0dzordr[inputI] = der.getiphi0dzordr(ider);
			    itmptdphi[inputI] = der.getitdphi(ider);
			    itmptdzordr[inputI] = der.getitdzordr(ider);
			    itmpz0dphi[inputI] = der.getiz0dphi(ider);
			    itmpz0dzordr[inputI] = der.getiz0dzordr(ider);
			  }
			
			  ider++;
			
		    } // for (unsigned int ihit = 1; ihit < 12; ++ihit)
		  } // if (goodseed)

		  FPGAWord tmprinvdphi[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmprinvdphi[j]>(1<<13)) itmprinvdphi[j]=(1<<13)-1;
			tmprinvdphi[j].set(itmprinvdphi[j],14,false,__LINE__,__FILE__);
		  }
		  outrinvdphi[i] << tmprinvdphi[0].str() << tmprinvdphi[1].str()
						 << tmprinvdphi[2].str() << tmprinvdphi[3].str()
						 << endl;
		  
		  FPGAWord tmprinvdzordr[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmprinvdzordr[j]>(1<<15)) itmprinvdzordr[j]=(1<<15)-1;
			tmprinvdzordr[j].set(itmprinvdzordr[j],16,false,__LINE__,__FILE__);
		  }
		  outrinvdzordr[i] << tmprinvdzordr[0].str() << tmprinvdzordr[1].str()
						   << tmprinvdzordr[2].str() << tmprinvdzordr[3].str()
						   << endl;
		  
		  FPGAWord tmpphi0dphi[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmpphi0dphi[j]>(1<<13)) itmpphi0dphi[j]=(1<<13)-1;
			tmpphi0dphi[j].set(itmpphi0dphi[j],14,false,__LINE__,__FILE__);
		  }
		  outphi0dphi[i] << tmpphi0dphi[0].str() << tmpphi0dphi[1].str()
						 << tmpphi0dphi[2].str() << tmpphi0dphi[3].str()
						 << endl;
		  
		  FPGAWord tmpphi0dzordr[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmpphi0dzordr[j]>(1<<15)) itmpphi0dzordr[j]=(1<<15)-1;
			tmpphi0dzordr[j].set(itmpphi0dzordr[j],16,false,__LINE__,__FILE__);
		  }
		  outphi0dzordr[i] << tmpphi0dzordr[0].str() << tmpphi0dzordr[1].str()
						   << tmpphi0dzordr[2].str() << tmpphi0dzordr[3].str()
						   << endl;
		  
		  FPGAWord tmptdphi[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmptdphi[j]>(1<<13)) itmptdphi[j]=(1<<13)-1;
			tmptdphi[j].set(itmptdphi[j],14,false,__LINE__,__FILE__);
		  }
		  outtdphi[i] << tmptdphi[0].str() << tmptdphi[1].str()
					  << tmptdphi[2].str() << tmptdphi[3].str()
					  << endl;
		  
		  FPGAWord tmptdzordr[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmptdzordr[j]>(1<<15)) itmptdzordr[j]=(1<<15)-1;
			tmptdzordr[j].set(itmptdzordr[j],16,false,__LINE__,__FILE__);
		  }
		  outtdzordr[i] << tmptdzordr[0].str() << tmptdzordr[1].str()
						<< tmptdzordr[2].str() << tmptdzordr[3].str()
						<< endl;
		  
		  FPGAWord tmpz0dphi[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmpz0dphi[j]>(1<<13)) itmpz0dphi[j]=(1<<13)-1;
			tmpz0dphi[j].set(itmpz0dphi[j],14,false,__LINE__,__FILE__);
		  }
		  outz0dphi[i] << tmpz0dphi[0].str() << tmpz0dphi[1].str()
					   << tmpz0dphi[2].str() << tmpz0dphi[3].str()
					   << endl;
		  
		  FPGAWord tmpz0dzordr[4];
		  for (unsigned int j = 0; j < 4; ++j) {
		    if (itmpz0dzordr[j]>(1<<15)) itmpz0dzordr[j]=(1<<15)-1;
			tmpz0dzordr[j].set(itmpz0dzordr[j],16,false,__LINE__,__FILE__);
		  }
		  outz0dzordr[i] << tmpz0dzordr[0].str() << tmpz0dzordr[1].str()
						 << tmpz0dzordr[2].str() << tmpz0dzordr[3].str()
						 << endl;	  

		  i++;
		} // for (const string & seed : seedings)
		
	  } // for (auto & der : derivatives_)

	  // close files
	  for (unsigned int i = 0; i < 6; ++i) {
	    outrinvdphi[i].close();
		outrinvdzordr[i].close();
		outphi0dphi[i].close();
		outphi0dzordr[i].close();
		outtdphi[i].close();
		outtdzordr[i].close();
		outz0dphi[i].close();
		outz0dzordr[i].close();
	  }
	  
    } // if (writeFitDerTable)
	
	
  }



  static void invert(double M[4][8],unsigned int n){

    assert(n<=4);

    unsigned int i,j,k;
    double ratio,a;

    for(i = 0; i < n; i++){
      for(j = n; j < 2*n; j++){
	if(i==(j-n))
	  M[i][j] = 1.0;
	else
	  M[i][j] = 0.0;
      }
    }

    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
	if(i!=j){
	  ratio = M[j][i]/M[i][i];
	  for(k = 0; k < 2*n; k++){
	    M[j][k] -= ratio * M[i][k];
	  }
	}
      }
    }

    for(i = 0; i < n; i++){
      a = M[i][i];
      for(j = 0; j < 2*n; j++){
	M[i][j] /= a;
      }
    }
  }



  static void invert(std::vector<std::vector<double> >& M,unsigned int n){

    assert(M.size()==n);
    assert(M[0].size()==2*n);
   
    unsigned int i,j,k;
    double ratio,a;

    for(i = 0; i < n; i++){
      for(j = n; j < 2*n; j++){
	if(i==(j-n))
	  M[i][j] = 1.0;
	else
	  M[i][j] = 0.0;
      }
    }

    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
	if(i!=j){
	  ratio = M[j][i]/M[i][i];
	  for(k = 0; k < 2*n; k++){
	    M[j][k] -= ratio * M[i][k];
	  }
	}
      }
    }

    for(i = 0; i < n; i++){
      a = M[i][i];
      for(j = 0; j < 2*n; j++){
	M[i][j] /= a;
      }
    }
  }


  static void getVarianceMatrix(bool layer[6],bool disk[5], int ptbin,
				std::vector<std::vector< double > >& V) {

    
    static bool first=true;

    static std::map<string, int> layerdiskmap;
    
    static double Vfull[11][11][4][1000];

    double sigmaz=0.15/sqrt(12.0);
    double sigmaz2=5.0/sqrt(12.0);
    
    if (first) {

      first=false;
      
      for(unsigned int i=0;i<11;i++) {
	for(unsigned int j=0;j<11;j++) {
	  for(unsigned int k=0;k<4;k++) {	  
	    for(unsigned int l=0;l<1000;l++) {	  
	      Vfull[i][j][k][l]=0.0;
	    }
	  }
	}
      }
	  
      ifstream in("variance.dat");

      unsigned int ptbin;

      int indexcount=0;

      int index=0;//determined later, but initialize here
      
      std::string type;
      
      in >> type;
      
      while (in.good()) {

	assert(type=="V" || type=="E");

	if (type=="V") {
	  string layer,disk;

	  in >> ptbin >> layer >> disk;

	  string layerdisk=layer+disk;

	  if (layerdiskmap.find(layerdisk)==layerdiskmap.end()) {
	    layerdiskmap[layerdisk]=indexcount;
	    indexcount++;
	  }

	  index=layerdiskmap[layerdisk];

	}

	if (type=="E") {
	  int i,j,entries;
	  double vij;
	  in >> i >> j >> vij >> entries;
	  assert(ptbin<4);
	  assert(i<11);
	  assert(j<11);
	  Vfull[i][j][ptbin][index]=vij;
	  Vfull[j][i][ptbin][index]=vij;
	}

	in >> type;
	  
      }
	
    }

    unsigned int index[11];
    std::string layerdisk="0000000000000000";
    
    unsigned int N=0;
    for(unsigned int i=0;i<6;i++) {
      if (layer[i]) {
	layerdisk[i]='1';
	index[N]=i;
	N++;
      }
    }
    for(unsigned int i=0;i<5;i++) {
      if (disk[i]) {
	index[N]=i+6;
	N++;
      }
    }

    V.clear();

    if (layerdiskmap.find(layerdisk)==layerdiskmap.end()) {
      cout << "Could not find an entry for layerdisk : "<<layerdisk<<endl;
      assert(0);
    }

    int mapindex=layerdiskmap[layerdisk];
    
    for(unsigned int i=0;i<2*N;i++) {
      std::vector<double> tmp;
      for(unsigned int j=0;j<4*N;j++) {
	tmp.push_back(0.0);
      }
      V.push_back(tmp);
    }

    for(unsigned int i=0;i<2*N;i++) {
      for(unsigned int j=0;j<2*N;j++) {
	if (i%2==0 && j%2==0) {
	  //if (i==j) cout << "i j index V : "<<i<<" "<<j<<" "<<index[j/2]
	  //		 <<" "<<Vfull[index[i/2]][index[j/2]][ptbin][mapindex]<<endl;
	  int indexi=index[i/2];
	  int indexj=index[j/2];
	  //if (indexi<6) indexi=5-indexi;
	  //if (indexj<6) indexj=5-indexj;
	  V[i][j]=Vfull[indexi][indexj][ptbin][mapindex];
	}
	if (i%2==1 && i==j){
	  if (index[i/2]<3) {
	    V[i][j]=sigmaz*sigmaz;
	    continue;
	  }
	  if (index[i/2]<6) {
	    V[i][j]=sigmaz2*sigmaz2;
	    continue;
	  }
	  V[i][j]=sigmaz*sigmaz;   //Fixme have to tell is disk 2S modules...
	}
      }
    }	
    
  }


  static void calculateDerivatives(unsigned int nlayers,
				   double r[6],
				   unsigned int ndisks,
				   double z[5],
				   double alpha[5],
				   double t,
				   double rinv,
				   double D[4][12],
				   int iD[4][12],
				   double MinvDt[4][12],
				   int iMinvDt[4][12],
				   double sigma[12],
				   double kfactor[12]){



    double sigmax=0.01/sqrt(12.0);
    double sigmaz=0.15/sqrt(12.0);
    double sigmaz2=5.0/sqrt(12.0);

    double sigmazpsbarrel=sigmaz;   //This is a bit of a hack - these weights should be properly determined
    if (fabs(t)>2.0) sigmazpsbarrel=sigmaz*fabs(t)/2.0;
    if (fabs(t)>3.8) sigmazpsbarrel=sigmaz*fabs(t);


    double sigmax2sdisk=0.009/sqrt(12.0);
    double sigmaz2sdisk=5.0/sqrt(12.0);

    double sigmaxpsdisk=0.01/sqrt(12.0);
    double sigmazpsdisk=0.15/sqrt(12.0);

    
    unsigned int n=nlayers+ndisks;
    
    assert(n<=6);

    double rnew[6];

    int j=0;


    //here we handle a barrel hit
    for(unsigned int i=0;i<nlayers;i++) {

      double ri=r[i];

      rnew[i]=ri;

      //first we have the phi position
      D[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv*rinv)/sigmax;
      D[1][j]=ri/sigmax;
      D[2][j]=0.0;
      D[3][j]=0.0;
      sigma[j]=sigmax;
      kfactor[j]=kphi1;
      j++;
      //second the z position
      D[0][j]=0.0;
      D[1][j]=0.0;
      if (ri<60.0) {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv)/sigmazpsbarrel;
	D[3][j]=1.0/sigmazpsbarrel;
        sigma[j]=sigmazpsbarrel;
        kfactor[j]=kz;
      } else {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv)/sigmaz2;
	D[3][j]=1.0/sigmaz2;
        sigma[j]=sigmaz2;
        kfactor[j]=kz;
      }

      j++;

    }


    for(unsigned int i=0;i<ndisks;i++) {

      double zi=z[i];

      double z0=0.0;
 
      double rmultiplier=alpha[i]*zi/t;

      double phimultiplier=zi/t;
      
      double drdrinv=-2.0*sin(0.5*rinv*(zi-z0)/t)/(rinv*rinv)
      +(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(rinv*t);
      double drdphi0=0;
      double drdt=-(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(t*t);
      double drdz0=-cos(0.5*rinv*(zi-z0)/t)/t;


      double dphidrinv=-0.5*(zi-z0)/t;
      double dphidphi0=1.0;
      double dphidt=0.5*rinv*(zi-z0)/(t*t);
      double dphidz0=0.5*rinv/t;

      double r=(zi-z0)/t;

      rnew[i+nlayers]=r;

      sigma[j]=sigmax2sdisk;
      if (fabs(alpha[i])<1e-10) {
	sigma[j]=sigmaxpsdisk;
      }
      
      D[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigma[j];
      D[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigma[j];
      D[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigma[j];
      D[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigma[j];
      kfactor[j]=kphiproj123;

      j++;

      if (fabs(alpha[i])<1e-10) {
	D[0][j]=drdrinv/sigmazpsdisk;
	D[1][j]=drdphi0/sigmazpsdisk;
	D[2][j]=drdt/sigmazpsdisk;
	D[3][j]=drdz0/sigmazpsdisk;
        sigma[j]=sigmazpsdisk;
        kfactor[j]=kr;
      }
      else {
	D[0][j]=drdrinv/sigmaz2sdisk;
	D[1][j]=drdphi0/sigmaz2sdisk;
	D[2][j]=drdt/sigmaz2sdisk;
	D[3][j]=drdz0/sigmaz2sdisk;
        sigma[j]=sigmaz2sdisk;
        kfactor[j]=kr;
      }


      j++;
      

    }

    double M[4][8];

    for(unsigned int i1=0;i1<4;i1++){
      for(unsigned int i2=0;i2<4;i2++){
	M[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  M[i1][i2]+=D[i1][j]*D[i2][j];	  
	}
      }
    }

    invert(M,4);

    for(unsigned int j=0;j<12;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	MinvDt[i1][j]=0.0;
	iMinvDt[i1][j]=0;
      }
    }  

    for(unsigned int j=0;j<2*n;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	for(unsigned int i2=0;i2<4;i2++) {
	  MinvDt[i1][j]+=M[i1][i2+4]*D[i2][j];
	}
      }
    }


    for (unsigned int i=0;i<n;i++) {

      iD[0][2*i]=D[0][2*i]*(1<<chisqphifactbits)*krinvpars/(1<<fitrinvbitshift);
      iD[1][2*i]=D[1][2*i]*(1<<chisqphifactbits)*kphi0pars/(1<<fitphi0bitshift);
      iD[2][2*i]=D[2][2*i]*(1<<chisqphifactbits)*ktpars/(1<<fittbitshift);
      iD[3][2*i]=D[3][2*i]*(1<<chisqphifactbits)*kzpars/(1<<fitz0bitshift);

      
      iD[0][2*i+1]=D[0][2*i+1]*(1<<chisqzfactbits)*krinvpars/(1<<fitrinvbitshift);
      iD[1][2*i+1]=D[1][2*i+1]*(1<<chisqzfactbits)*kphi0pars/(1<<fitphi0bitshift);
      iD[2][2*i+1]=D[2][2*i+1]*(1<<chisqzfactbits)*ktpars/(1<<fittbitshift);
      iD[3][2*i+1]=D[3][2*i+1]*(1<<chisqzfactbits)*kzpars/(1<<fitz0bitshift);
	
      


      //First the barrel
      if (i<nlayers) {

	MinvDt[0][2*i]*=rnew[i]/sigmax;
	MinvDt[1][2*i]*=rnew[i]/sigmax;
	MinvDt[2][2*i]*=rnew[i]/sigmax;
	MinvDt[3][2*i]*=rnew[i]/sigmax;

	
	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphi1/krinvpars;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphi1/kphi0pars;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphi1/ktpars;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphi1/kzpars;

	if (rnew[i]<60.0) {
	  MinvDt[0][2*i+1]/=sigmazpsbarrel;
	  MinvDt[1][2*i+1]/=sigmazpsbarrel;
	  MinvDt[2][2*i+1]/=sigmazpsbarrel;
	  MinvDt[3][2*i+1]/=sigmazpsbarrel;

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*kzproj/kzpars;
	} else {
	  MinvDt[0][2*i+1]/=sigmaz2;
	  MinvDt[1][2*i+1]/=sigmaz2;
	  MinvDt[2][2*i+1]/=sigmaz2;
	  MinvDt[3][2*i+1]/=sigmaz2;

	  int fact=(1<<(nbitszprojL123-nbitszprojL456));

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*fact*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*fact*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*fact*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*fact*kzproj/kzpars;
	}
      }

      //Secondly the disks
      else {

	if (fabs(alpha[i-nlayers])<1e-10) {
	  MinvDt[0][2*i]*=(rnew[i]/sigmaxpsdisk);
	  MinvDt[1][2*i]*=(rnew[i]/sigmaxpsdisk);
	  MinvDt[2][2*i]*=(rnew[i]/sigmaxpsdisk);
	  MinvDt[3][2*i]*=(rnew[i]/sigmaxpsdisk);
	} else {
	  MinvDt[0][2*i]*=(rnew[i]/sigmax2sdisk);
	  MinvDt[1][2*i]*=(rnew[i]/sigmax2sdisk);
	  MinvDt[2][2*i]*=(rnew[i]/sigmax2sdisk);
	  MinvDt[3][2*i]*=(rnew[i]/sigmax2sdisk);
	}      

	assert(MinvDt[0][2*i]==MinvDt[0][2*i]);

	//iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphiprojdisk/krinvparsdisk;
	//iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphiprojdisk/kphi0parsdisk;
	//iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphiprojdisk/ktparsdisk;
	//iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphiprojdisk/kzdisk;

	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphiproj123/krinvpars;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphiproj123/kphi0pars;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphiproj123/ktpars;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphiproj123/kz;

	if (fabs(alpha[i-nlayers])<1e-10) {
	  MinvDt[0][2*i+1]/=sigmazpsdisk;
	  MinvDt[1][2*i+1]/=sigmazpsdisk;
	  MinvDt[2][2*i+1]/=sigmazpsdisk;
	  MinvDt[3][2*i+1]/=sigmazpsdisk;
	} else {
	  MinvDt[0][2*i+1]/=sigmaz2sdisk;
	  MinvDt[1][2*i+1]/=sigmaz2sdisk;
	  MinvDt[2][2*i+1]/=sigmaz2sdisk;
	  MinvDt[3][2*i+1]/=sigmaz2sdisk;
	}

	iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*krprojshiftdisk/krinvpars;
	iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*krprojshiftdisk/kphi0pars;
	iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*krprojshiftdisk/ktpars;
	iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*krprojshiftdisk/kz;
      
      }

    }
    

  }
  

  
  static void calculateDerivativesMS(unsigned int nlayers,
				     double r[6],
				     unsigned int ndisks,
				     double z[5],
				     double alpha[5],
				     double t,
				     double rinv,
				     double D[4][12],
				     int iD[4][12],
				     double MinvDt[4][12],
				     int iMinvDt[4][12],
				     double sigma[12],
				     double kfactor[12],
				     int ptbin){





    double sigmax=0.01/sqrt(12.0);
    double sigmaz=0.15/sqrt(12.0);
    double sigmaz2=5.0/sqrt(12.0);

    unsigned int n=nlayers+ndisks;
    
    assert(n<=6);

    double rnew[6];

    int j=0;

    bool layer[6],disk[5];
    for(unsigned int i=0;i<6;i++) layer[i]=false;
    for(unsigned int i=0;i<5;i++) disk[i]=false;
    for(unsigned int i=0;i<nlayers;i++){
      for(unsigned int j=0;j<6;j++) {
	if (fabs(r[i]-rmean[j])<drmax) layer[j]=true;
      }
    }
    for(unsigned int i=0;i<ndisks;i++){
      for(unsigned int j=0;j<5;j++) {
	if (fabs(fabs(z[i])-zmean[j])<dzmax) {
	  disk[j]=true;
	  cout << "z zmean ndisks"<<z[i]<<" "<<zmean[j]<<" "<<ndisks<<" "<<nlayers<<endl;
	}
      }
    }

    std::vector<std::vector< double > > V;

    cout << "layer : "<<nlayers<<" "<<layer[0]<<layer[1]<<layer[2]<<layer[3]<<layer[4]<<layer[5]<<endl;
    cout << "disk  : "<<ndisks<<" "<<disk[0]<<disk[1]<<disk[2]<<disk[3]<<disk[4]<<endl;
    
    getVarianceMatrix(layer,disk,ptbin,V);

    cout <<"V: "<<ptbin<<endl;
    for(unsigned int ii=0;ii<2*n;ii+=2){
      for(unsigned int jj=0;jj<2*n;jj+=2){
	cout<<V[ii][jj]<<" ";
      }
      cout<<endl;
    }
    cout <<"Vcorr:"<<endl;
    for(unsigned int ii=0;ii<2*n;ii+=2){
      for(unsigned int jj=0;jj<2*n;jj+=2){
	cout<<V[ii][jj]/sqrt(V[ii][ii]*V[jj][jj])<<" ";
      }
      cout<<endl;
    }
    
    invert(V,2*n);

    cout <<"Vinv:"<<endl;
    for(unsigned int ii=0;ii<2*n;ii+=2){
      for(unsigned int jj=0;jj<2*n;jj+=2){
	cout<<V[ii][jj+2*n]<<" ";
      }
      cout<<endl;
    }


    //here we handle a barrel hit
    for(unsigned int i=0;i<nlayers;i++) {

      double ri=r[i];

      rnew[i]=ri;

      //first we have the phi position
      D[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv*rinv);
      D[1][j]=ri;
      D[2][j]=0.0;
      D[3][j]=0.0;
      sigma[j]=sigmax;
      kfactor[j]=kphi1;
      j++;
      //second the z position
      D[0][j]=0.0;
      D[1][j]=0.0;
      if (ri<60.0) {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv);
	D[3][j]=1.0;
        sigma[j]=sigmaz;
        kfactor[j]=kz;
      } else {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv);
	D[3][j]=1.0;
        sigma[j]=sigmaz2;
        kfactor[j]=kz;
      }

      j++;

    }


    for(unsigned int i=0;i<ndisks;i++) {

      double zi=z[i];

      double z0=0.0;
 
      double rmultiplier=alpha[i]*zi/t;

      double phimultiplier=zi/t;
      
      double drdrinv=-2.0*sin(0.5*rinv*(zi-z0)/t)/(rinv*rinv)
      +(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(rinv*t);
      double drdphi0=0;
      double drdt=-(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(t*t);
      double drdz0=-cos(0.5*rinv*(zi-z0)/t)/t;


      double dphidrinv=-0.5*(zi-z0)/t;
      double dphidphi0=1.0;
      double dphidt=0.5*rinv*(zi-z0)/(t*t);
      double dphidz0=0.5*rinv/t;

      double r=(zi-z0)/t;

      rnew[i+nlayers]=r;

      D[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv);
      D[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0);
      D[2][j]=(phimultiplier*dphidt+rmultiplier*drdt);
      D[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0);
      sigma[j]=sigmax;
      kfactor[j]=kphiproj123;

      j++;

      if (fabs(alpha[i])<1e-10) {
	D[0][j]=drdrinv;
	D[1][j]=drdphi0;
	D[2][j]=drdt;
	D[3][j]=drdz0;
        sigma[j]=sigmaz;
        kfactor[j]=kr;
      }
      else {
	D[0][j]=drdrinv;
	D[1][j]=drdphi0;
	D[2][j]=drdt;
	D[3][j]=drdz0;
        sigma[j]=sigmaz2;
        kfactor[j]=kr;
      }


      j++;
      

    }

    double M[4][8];

    for(unsigned int i1=0;i1<4;i1++){
      for(unsigned int i2=0;i2<4;i2++){
	M[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  for(unsigned int i=0;i<2*n;i++){
	    M[i1][i2]+=D[i1][i]*V[i][j+2*n]*D[i2][j];	  
	  }
	}
      }
    }

    invert(M,4);

    for(unsigned int j=0;j<12;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	MinvDt[i1][j]=0.0;
	iMinvDt[i1][j]=0;
      }
    }  

    for(unsigned int i1=0;i1<4;i1++) {
      for(unsigned int i2=0;i2<2*n;i2++) {
	for(unsigned int i=0;i<4;i++) {
	  for(unsigned int j=0;j<2*n;j++) {
	    MinvDt[i1][i2]+=M[i1][i+4]*D[i][j]*V[j][i2+2*n];
	  }
	}
      }
    }


    for (unsigned int i=0;i<n;i++) {

      iD[0][2*i]=D[0][2*i]*(1<<chisqphifactbits)*krinvpars/(1<<fitrinvbitshift);
      iD[1][2*i]=D[1][2*i]*(1<<chisqphifactbits)*kphi0pars/(1<<fitphi0bitshift);
      iD[2][2*i]=D[2][2*i]*(1<<chisqphifactbits)*ktpars/(1<<fittbitshift);
      iD[3][2*i]=D[3][2*i]*(1<<chisqphifactbits)*kzpars/(1<<fitz0bitshift);

      
      iD[0][2*i+1]=D[0][2*i+1]*(1<<chisqzfactbits)*krinvpars/(1<<fitrinvbitshift);
      iD[1][2*i+1]=D[1][2*i+1]*(1<<chisqzfactbits)*kphi0pars/(1<<fitphi0bitshift);
      iD[2][2*i+1]=D[2][2*i+1]*(1<<chisqzfactbits)*ktpars/(1<<fittbitshift);
      iD[3][2*i+1]=D[3][2*i+1]*(1<<chisqzfactbits)*kzpars/(1<<fitz0bitshift);
	
      


      //First the barrel
      if (i<nlayers) {

	MinvDt[0][2*i]*=rnew[i];
	MinvDt[1][2*i]*=rnew[i];
	MinvDt[2][2*i]*=rnew[i];
	MinvDt[3][2*i]*=rnew[i];

	
	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphi1/krinvpars;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphi1/kphi0pars;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphi1/ktpars;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphi1/kzpars;

	if (rnew[i]<60.0) {
	  //MinvDt[0][2*i+1]/=sigmaz;
	  //MinvDt[1][2*i+1]/=sigmaz;
	  //MinvDt[2][2*i+1]/=sigmaz;
	  //MinvDt[3][2*i+1]/=sigmaz;

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*kzproj/kzpars;
	} else {
	  //MinvDt[0][2*i+1]/=sigmaz2;
	  //MinvDt[1][2*i+1]/=sigmaz2;
	  //MinvDt[2][2*i+1]/=sigmaz2;
	  //MinvDt[3][2*i+1]/=sigmaz2;

	  int fact=(1<<(nbitszprojL123-nbitszprojL456));

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*fact*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*fact*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*fact*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*fact*kzproj/kzpars;
	}
      }

      //Secondly the disks
      else {

	if (fabs(alpha[i-nlayers])<1e-10) {
	  MinvDt[0][2*i]*=(rnew[i]);
	  MinvDt[1][2*i]*=(rnew[i]);
	  MinvDt[2][2*i]*=(rnew[i]);
	  MinvDt[3][2*i]*=(rnew[i]);
	} else {
	  MinvDt[0][2*i]*=(rnew[i]);
	  MinvDt[1][2*i]*=(rnew[i]);
	  MinvDt[2][2*i]*=(rnew[i]);
	  MinvDt[3][2*i]*=(rnew[i]);
	}      

	assert(MinvDt[0][2*i]==MinvDt[0][2*i]);

	//iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphiprojdisk/krinvparsdisk;
	//iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphiprojdisk/kphi0parsdisk;
	//iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphiprojdisk/ktparsdisk;
	//iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphiprojdisk/kzdisk;

	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphiproj123/krinvpars;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphiproj123/kphi0pars;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphiproj123/ktpars;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphiproj123/kz;

	/*
	if (fabs(alpha[i])<1e-10) {
	  MinvDt[0][2*i+1]/=sigmaz;
	  MinvDt[1][2*i+1]/=sigmaz;
	  MinvDt[2][2*i+1]/=sigmaz;
	  MinvDt[3][2*i+1]/=sigmaz;
	} else {
	  MinvDt[0][2*i+1]/=sigmaz2;
	  MinvDt[1][2*i+1]/=sigmaz2;
	  MinvDt[2][2*i+1]/=sigmaz2;
	  MinvDt[3][2*i+1]/=sigmaz2;
	}
	*/

	iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*krprojshiftdisk/krinvpars;
	iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*krprojshiftdisk/kphi0pars;
	iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*krprojshiftdisk/ktpars;
	iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*krprojshiftdisk/kz;
      
      }

    }
    

  }
  

  


  static double gett(int diskmask, int layermask) {

    if (diskmask==0) return 0.0;

    double tmax=1000.0;
    double tmin=0.0;

    for(int d=1;d<=5;d++) {

      if (diskmask&(1<<(2*(5-d)+1))) { //PS hit
	double dmax=zmean[d-1]/22.0;
	if (dmax>sinh(2.4)) dmax=sinh(2.4);
	double dmin=zmean[d-1]/65.0;
	if (dmax<tmax) tmax=dmax;
	if (dmin>tmin) tmin=dmin;
      } 

      if (diskmask&(1<<(2*(5-d)))) { //2S hit
	double dmax=zmean[d-1]/65.0;
	double dmin=zmean[d-1]/105.0;
	if (dmax<tmax) tmax=dmax;
	if (dmin>tmin) tmin=dmin;	
      } 

    }

    for (int l=1;l<=6;l++) {

      if (layermask&(1<<(6-l))) {
	double lmax=zlength/rmean[l-1];
	if (lmax<tmax) tmax=lmax;	
      }
      
    }
    
    //cout << "diskmask tmin tmax : "<<diskmask<<" "<<tmin<<" "<<tmax<<endl;
    
    return 0.5*(tmax+tmin)*1.07;

  }


private:


  vector<int> LayerMem_;
  vector<int> DiskMem_;

  vector<int> LayerDiskMem_;

  unsigned int LayerMemBits_;
  unsigned int DiskMemBits_;
  unsigned int LayerDiskMemBits_;
  unsigned int alphaBits_;

  unsigned int Nlay_;
  unsigned int Ndisk_;

  vector<TrackDer> derivatives_;
  
  int nextLayerValue_;
  int nextDiskValue_;
  int nextLayerDiskValue_;
  int lastMultiplicity_;

};



#endif



