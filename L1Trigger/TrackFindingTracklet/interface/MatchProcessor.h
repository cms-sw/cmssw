//This class implementes the match processor
#ifndef MATCHPROCESSOR_H
#define MATCHPROCESSOR_H

#include "ProcessBase.h"
#include "Util.h"

using namespace std;

class MatchProcessor:public ProcessBase{

public:

  MatchProcessor(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    
    fullmatchesToPlus_=0;
    fullmatchesToMinus_=0;
    double dphi=2*M_PI/NSector;
    double dphiHG=0.5*dphisectorHG-M_PI/NSector;
    phimin_=iSector_*dphi-dphiHG;
    phimax_=phimin_+dphi+2*dphiHG;

    phimin_-=M_PI/NSector;
    phimax_-=M_PI/NSector;
    if (phimin_>M_PI) {
      phimin_-=2*M_PI;
      phimax_-=2*M_PI;
    }

    phioffset_=phimin_;
    
    string subname=name.substr(3,2);
    fullmatchesToPlus_=0;
    fullmatchesToMinus_=0;
    layer_=0;
    disk_=0;
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
    if (layer_==0 && disk_==0) {
      cout << "name subname "<<name<<" "<<subname<<endl;
      assert(0);
    }
    //FIXME should sort out constants here
    icorrshift_=7+idrinvbits+phi0bitshift-rinvbitshift-phiderbitshift;
    if (layer_<=3) {
      icorzshift_=-1-PS_zderL_shift;
    } else {
      icorzshift_=-1-SS_zderL_shift;      
    }
    phi0shift_=3;
    fact_=1;
    if (layer_>=4) {
      fact_=(1<<(nbitszprojL123-nbitszprojL456));
      icorrshift_-=(10-nbitsrL456);
      icorzshift_+=(nbitszprojL123-nbitszprojL456+nbitsrL456-nbitsrL123);
      phi0shift_=0;
    }


    nrbits_=5;
    nphiderbits_=6;

    
    //to adjust globaly the phi and rz matching cuts
    phifact_=1.0;
    rzfact_=1.0;

   for(unsigned int seedindex=0;seedindex<12;seedindex++){
      phimatchcut_[seedindex]=-1;
      zmatchcut_[seedindex]=-1;
    }
    
    if (layer_==1){
      phimatchcut_[2]=0.07/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=5.5/kz;
      phimatchcut_[3]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=15.0/kz;
      phimatchcut_[4]=0.07/(kphi1*rmean[layer_-1]);
      zmatchcut_[4]=1.5/kz;
      phimatchcut_[5]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[5]=2.0/kz;
      phimatchcut_[7]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[7]=1.5/kz;
      phimatchcut_[1]=0.1/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=0.7/kz;

      phimatchcut_[8]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[8]=1.0/kz;
      phimatchcut_[9]=0.15/(kphi1*rmean[layer_-1]);
      zmatchcut_[9]=8.0/kz;
      phimatchcut_[10]=0.125/(kphi1*rmean[layer_-1]);
      zmatchcut_[10]=1.0/kz;
      phimatchcut_[11]=0.15/(kphi1*rmean[layer_-1]);
      zmatchcut_[11]=1.5/kz;
    }
    if (layer_==2){
      phimatchcut_[2]=0.06/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=3.5/kz;
      phimatchcut_[3]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=15.0/kz;
      phimatchcut_[4]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[4]=1.25/kz;

      phimatchcut_[9]=0.1/(kphi1*rmean[layer_-1]);
      zmatchcut_[9]=7.0/kz;
    }
    if (layer_==3){
      phimatchcut_[0]=0.1/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=0.7/kz;
      phimatchcut_[3]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=9.0/kz;

      phimatchcut_[9]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[9]=5.0/kz;
      phimatchcut_[11]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[11]=0.0/kz;
    }
    if (layer_==4){
      phimatchcut_[0]=0.19/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=3.0/kz;
      phimatchcut_[3]=0.05/(kphi1*rmean[layer_-1]);
      zmatchcut_[3]=7.0/kz;
      phimatchcut_[1]=0.19/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=3.0/kz;

      phimatchcut_[10]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[10]=0.0/kz;
      phimatchcut_[11]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[11]=0.0/kz;
    }
    if (layer_==5){
      phimatchcut_[0]=0.4/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=3.0/kz;
      phimatchcut_[2]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=8.0/kz;
      phimatchcut_[1]=0.4/(kphi1*rmean[layer_-1]);
      zmatchcut_[1]=3.0/kz;

      phimatchcut_[8]=0.08/(kphi1*rmean[layer_-1]);
      zmatchcut_[8]=4.5/kz;
      phimatchcut_[10]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[10]=0.0/kz;
      phimatchcut_[11]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[11]=0.0/kz;
    }
    if (layer_==6){
      phimatchcut_[0]=0.5/(kphi1*rmean[layer_-1]);
      zmatchcut_[0]=4.0/kz;
      phimatchcut_[2]=0.19/(kphi1*rmean[layer_-1]);
      zmatchcut_[2]=9.5/kz;

      phimatchcut_[8]=0.2/(kphi1*rmean[layer_-1]);
      zmatchcut_[8]=4.5/kz;
      phimatchcut_[10]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[10]=0.0/kz;
      phimatchcut_[11]=0.0/(kphi1*rmean[layer_-1]);
      zmatchcut_[11]=0.0/kz;
    }

    if (iSector_==0&&layer_>0&&writeMCcuts) {

      ofstream outphicut;
      outphicut.open(getName()+"_phicut.txt");
      outphicut << "{"<<endl;
      for(unsigned int seedindex=0;seedindex<12;seedindex++){
	if (seedindex!=0) outphicut<<","<<endl;
	outphicut << phimatchcut_[seedindex];
      }
      outphicut <<endl<<"};"<<endl;
      outphicut.close();

      ofstream outzcut;
      outzcut.open(getName()+"_zcut.txt");
      outzcut << "{"<<endl;
      for(unsigned int seedindex=0;seedindex<12;seedindex++){
	if (seedindex!=0) outzcut<<","<<endl;
	outzcut << zmatchcut_[seedindex];
      }
      outzcut <<endl<<"};"<<endl;
      outzcut.close();
    }

    for(int iseedindex=0;iseedindex<12;iseedindex++){
      rphicutPS_[iseedindex]=-1.0;
      rphicut2S_[iseedindex]=-1.0;
      rcutPS_[iseedindex]=-1.0;
      rcut2S_[iseedindex]=-1.0;
    }
    
    if (abs(disk_)==1){
      rphicutPS_[0]=0.2/(kphiproj123*kr);
      rcutPS_[0]=0.5/krprojshiftdisk;
      rphicut2S_[0]=0.5/(kphiproj123*kr);
      rcut2S_[0]=3.8/krprojshiftdisk;      
      
      rphicut2S_[2]=0.8/(kphiproj123*kr);
      rcut2S_[2]=3.8/krprojshiftdisk;   
      rphicutPS_[2]=999.0;
      rcutPS_[2]=999.0;      
      
      rphicutPS_[5]=0.10/(kphiproj123*kr);
      rcutPS_[5]=0.5/krprojshiftdisk;
      
      rphicutPS_[1]=0.2/(kphiproj123*kr);
      rcutPS_[1]=0.5/krprojshiftdisk;
      rphicut2S_[1]=0.5/(kphiproj123*kr);
      rcut2S_[1]=3.8/krprojshiftdisk;
      
      rphicutPS_[8]=0.0;
      rcutPS_[8]=0.0;
      rphicut2S_[8]=0.20/(kphiproj123*kr);
      rcut2S_[8]=3.0/krprojshiftdisk;      
      rphicutPS_[9]=0.0;
      rcutPS_[9]=0.0;
      rphicut2S_[9]=0.0;
      rcut2S_[9]=0.0;      
    }

    if (abs(disk_)==2){
      rphicutPS_[0]=0.2/(kphiproj123*kr);
      rcutPS_[0]=0.5/krprojshiftdisk;
      rphicut2S_[0]=0.5/(kphiproj123*kr);
      rcut2S_[0]=3.8/krprojshiftdisk;      

      rphicut2S_[2]=0.8/(kphiproj123*kr);
      rcut2S_[2]=3.8/krprojshiftdisk;      

      rphicutPS_[5]=0.10/(kphiproj123*kr);
      rcutPS_[5]=0.5/krprojshiftdisk;

      rphicutPS_[6]=0.10/(kphiproj123*kr);
      rcutPS_[6]=0.5/krprojshiftdisk;

      
      rphicut2S_[6]=0.5/(kphiproj123*kr);
      rcut2S_[6]=3.8/krprojshiftdisk;      

      rphicutPS_[7]=0.10/(kphiproj123*kr);
      rcutPS_[7]=0.5/krprojshiftdisk;
      rphicut2S_[7]=0.15/(kphiproj123*kr);
      rcut2S_[7]=3.4/krprojshiftdisk;      

      rphicutPS_[1]=0.2/(kphiproj123*kr);
      rcutPS_[1]=0.5/krprojshiftdisk;
      rphicut2S_[1]=0.5/(kphiproj123*kr);
      rcut2S_[1]=3.8/krprojshiftdisk;       
           
      rphicutPS_[8]=0.0;
      rcutPS_[8]=0.0;
      rphicut2S_[8]=0.30/(kphiproj123*kr);
      rcut2S_[8]=3.0/krprojshiftdisk;      
      rphicutPS_[9]=0.0;
      rcutPS_[9]=0.0;
      rphicut2S_[9]=0.0;
      rcut2S_[9]=0.0;      
      rphicutPS_[10]=0.15/(kphiproj123*kr);
      rcutPS_[10]=0.5/krprojshiftdisk;
      rphicut2S_[10]=0.68/(kphiproj123*kr);
      rcut2S_[10]=3.0/krprojshiftdisk;      
    }

    if (abs(disk_)==3){
      rphicutPS_[0]=0.25/(kphiproj123*kr);
      rcutPS_[0]=0.5/krprojshiftdisk;
      rphicut2S_[0]=0.5/(kphiproj123*kr);
      rcut2S_[0]=3.6/krprojshiftdisk;      


      rphicutPS_[4]=0.15/(kphiproj123*kr);
      rcutPS_[4]=0.5/krprojshiftdisk;
      rphicut2S_[4]=0.15/(kphiproj123*kr);
      rcut2S_[4]=3.6/krprojshiftdisk;      

      rphicutPS_[6]=0.2/(kphiproj123*kr);
      rcutPS_[6]=0.6/krprojshiftdisk;
      rphicut2S_[6]=0.2/(kphiproj123*kr);
      rcut2S_[6]=3.6/krprojshiftdisk;

      rphicutPS_[7]=0.15/(kphiproj123*kr);
      rcutPS_[7]=0.8/krprojshiftdisk;
      rphicut2S_[7]=0.25/(kphiproj123*kr);
      rcut2S_[7]=3.8/krprojshiftdisk;      

      rphicutPS_[1]=0.2/(kphiproj123*kr);
      rcutPS_[1]=0.5/krprojshiftdisk;
      rphicut2S_[1]=0.5/(kphiproj123*kr);
      rcut2S_[1]=3.8/krprojshiftdisk;            
      
      rphicutPS_[8]=0.0;
      rcutPS_[8]=0.0;
      rphicut2S_[8]=0.0;
      rcut2S_[8]=0.0;      
      rphicutPS_[9]=0.0;
      rcutPS_[9]=0.0;
      rphicut2S_[9]=0.0;
      rcut2S_[9]=0.0;      
      rphicutPS_[10]=0.0;
      rcutPS_[10]=0.0;
      rphicut2S_[10]=0.8/(kphiproj123*kr);
      rcut2S_[10]=3.8/krprojshiftdisk;      
      rphicutPS_[11]=0.2/(kphiproj123*kr);
      rcutPS_[11]=0.4/krprojshiftdisk;
      rphicut2S_[11]=0.1/(kphiproj123*kr);
      rcut2S_[11]=3.0/krprojshiftdisk;      
    }


    if (abs(disk_)==4){
      rphicutPS_[0]=0.5/(kphiproj123*kr);
      rcutPS_[0]=0.5/krprojshiftdisk;      
      rphicut2S_[0]=0.5/(kphiproj123*kr);
      rcut2S_[0]=3.6/krprojshiftdisk;      


      rphicutPS_[4]=0.2/(kphiproj123*kr);
      rcutPS_[4]=0.8/krprojshiftdisk;
      rphicut2S_[4]=0.2/(kphiproj123*kr);
      rcut2S_[4]=3.6/krprojshiftdisk;      

      rphicutPS_[6]=0.3/(kphiproj123*kr);
      rcutPS_[6]=1.0/krprojshiftdisk;
      rphicut2S_[6]=0.25/(kphiproj123*kr);
      rcut2S_[6]=3.5/krprojshiftdisk;      

      rphicutPS_[7]=0.5/(kphiproj123*kr);
      rcutPS_[7]=1.0/krprojshiftdisk;      
      rphicut2S_[7]=0.5/(kphiproj123*kr);
      rcut2S_[7]=3.8/krprojshiftdisk;      

      rphicutPS_[1]=0.2/(kphiproj123*kr);
      rcutPS_[1]=0.5/krprojshiftdisk;
      rphicut2S_[1]=0.5/(kphiproj123*kr);
      rcut2S_[1]=3.8/krprojshiftdisk;
      
      rphicutPS_[8]=0.0;
      rcutPS_[8]=0.0;
      rphicut2S_[8]=0.0;
      rcut2S_[8]=0.0;      
      rphicutPS_[9]=0.0;
      rcutPS_[9]=0.0;
      rphicut2S_[9]=0.0;
      rcut2S_[9]=0.0;      
      rphicutPS_[10]=0.0;
      rcutPS_[10]=0.0;
      rphicut2S_[10]=0.6/(kphiproj123*kr);
      rcut2S_[10]=3.0/krprojshiftdisk;      
      rphicutPS_[11]=0.0;
      rcutPS_[11]=0.0;
      rphicut2S_[11]=0.4/(kphiproj123*kr);
      rcut2S_[11]=3.0/krprojshiftdisk;      
    }



    if (abs(disk_)==5){
      rphicutPS_[4]=0.25/(kphiproj123*kr);
      rcutPS_[4]=1.0/krprojshiftdisk;
      rphicut2S_[4]=0.4/(kphiproj123*kr);
      rcut2S_[4]=3.6/krprojshiftdisk;      

      rphicutPS_[5]=0.10/(kphiproj123*kr);
      rcutPS_[5]=0.5/krprojshiftdisk;
      rphicut2S_[5]=0.2/(kphiproj123*kr);
      rcut2S_[5]=3.4/krprojshiftdisk;      

      rphicutPS_[6]=0.5/(kphiproj123*kr);
      rcutPS_[6]=2.0/krprojshiftdisk;
      rphicut2S_[6]=0.4/(kphiproj123*kr);
      rcut2S_[6]=3.7/krprojshiftdisk;      


      rphicutPS_[8]=0.0;
      rcutPS_[8]=0.0;
      rphicut2S_[8]=0.0;
      rcut2S_[8]=0.0;      
      rphicutPS_[9]=0.0;
      rcutPS_[9]=0.0;
      rphicut2S_[9]=0.0;
      rcut2S_[9]=0.0;      
      rphicutPS_[10]=0.0;
      rcutPS_[10]=0.0;
      rphicut2S_[10]=0.0;
      rcut2S_[10]=0.0;      
      rphicutPS_[11]=0.0;
      rcutPS_[11]=0.0;
      rphicut2S_[11]=0.8/(kphiproj123*kr);
      rcut2S_[11]=3.0/krprojshiftdisk;       
    }

    

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


    for (unsigned int i=0; i<10; i++) {
      ialphafactinner_[i]= 
	(1<<alphashift)*krprojshiftdisk*half2SmoduleWidth/(1<<(nbitsalpha-1))/(rDSSinner[i]*rDSSinner[i])/kphiproj123;
      ialphafactouter_[i]= 
	(1<<alphashift)*krprojshiftdisk*half2SmoduleWidth/(1<<(nbitsalpha-1))/(rDSSouter[i]*rDSSouter[i])/kphiproj123;
    }

    
    
  }

  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout1"||
	output=="matchout2"||
	output=="matchout3"||
	output=="matchout4"||
	output=="matchout5"||
	output=="matchout6"||
	output=="matchout7"||
	output=="matchout8"
	){
      FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
      assert(tmp!=0);
      fullmatches_.push_back(tmp);
      return;
    }
    if (output=="matchoutplus"){
      FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToPlus_==0);
      fullmatchesToPlus_=tmp;
      return;
    }
    if (output=="matchoutminus"){
      FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToMinus_==0);
      fullmatchesToMinus_=tmp;
      return;
    }
    cout << "Could not find output = "<<output<<endl;
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="allstubin"){
      AllStubsMemory* tmp=dynamic_cast<AllStubsMemory*>(memory);
      assert(tmp!=0);
      allstubs_=tmp;
      return;
    }
    if (input=="vmstubin"){
      VMStubsMEMemory* tmp=dynamic_cast<VMStubsMEMemory*>(memory);
      assert(tmp!=0);
      vmstubs_.push_back(tmp); //to allow more than one stub in?  vmstubs_=tmp;
      return;
    }
    if (input=="projin"||
	input=="proj1in"||
	input=="proj2in"||
	input=="proj3in"||
	input=="proj4in"||
	input=="proj5in"||
	input=="proj6in"||
	input=="proj7in"||
	input=="proj8in"||
	input=="proj9in"||
	input=="proj10in"||
	input=="proj11in"||
	input=="proj12in"||
	input=="proj13in"||
	input=="proj14in"||
	input=="proj15in"||
	input=="proj16in"||
	input=="proj17in"||
	input=="proj18in"||
	input=="proj19in"||
	input=="proj20in"||
	input=="proj21in"
	){
      TrackletProjectionsMemory* tmp=dynamic_cast<TrackletProjectionsMemory*>(memory);
      assert(tmp!=0);
      inputprojs_.push_back(tmp);
      return;
    }
    cout << "MatchProcessor input = "<<input<<endl;
    assert(0);
  }

  void execute() {

    bool barrel=layer_>0;

    int countpass=0;

    vector<std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > > candmatches;
		
    //cout << "MatchProcessor : "<<getName()<<" "<<layer_<<" "<<disk_<<endl;
    //cout << "inputprojs_.size() : "<<inputprojs_.size()<<endl;
    for (unsigned int iprojmem=0;iprojmem<inputprojs_.size();iprojmem++){
      TrackletProjectionsMemory* projMem=inputprojs_[iprojmem];
      //cout << "Projection Memory:"<<projMem->getName()<<" "<<projMem->nTracklets()<<endl;
      for (unsigned int iproj=0;iproj<projMem->nTracklets();iproj++) {
	Tracklet* proj=projMem->getFPGATracklet(iproj);
	FPGAWord fpgaphi=barrel?proj->fpgaphiproj(layer_):proj->fpgaphiprojdisk(disk_);
	FPGAWord fpgarz=barrel?proj->fpgazproj(layer_):proj->fpgarprojdisk(disk_);

	//int disk=disk_;
	//if (proj->t()<0.0) disk=-disk_;
	
	assert(!fpgaphi.atExtreme());

	int iphitmp=fpgaphi.value();
	int iphi=iphitmp>>(fpgaphi.nbits()-5);

	int nvm=-1;
	int nbins=-1;
	nvm=barrel?nvmmelayers[layer_-1]*nallstubslayers[layer_-1]:nvmmedisks[disk_-1]*nallstubsdisks[disk_-1];
	nbins=barrel?nvmmelayers[layer_-1]:nvmmedisks[disk_-1];
	assert(nvm>0);
	iphi=(iphi/(32/nvm))&(nbins-1);
	//cout << "iproj iphi "<<iproj<<" "<<iphi<<endl;
	//cout << "vmstubs_.size() : "<<vmstubs_.size()<<endl;
	assert((unsigned int)iphi<vmstubs_.size());

	VMStubsMEMemory* stubmem=vmstubs_[iphi];
	//cout << "VMStubMEMemory: "<<stubmem->getName()<<endl;

	unsigned int rzfirst = barrel?proj->zbin1projvm(layer_):proj->rbin1projvm(disk_);
	unsigned int rzlast = rzfirst;
	bool second=(barrel?proj->zbin2projvm(layer_):proj->rbin2projvm(disk_))==1;
	if (second) rzlast += 1;
	
	int projfinerz = barrel?proj->finezvm(layer_):proj->finervm(disk_);

	int projfinerzadj = 0;

	if (!barrel) {
	  //The next lines looks up the predicted bend based on:
	  // 1 - r projections
	  // 2 - phi derivative
	  // 3 - the sign - i.e. if track is forward or backward
	  int rindex=(proj->fpgarprojdisk(disk_).value()>>(proj->fpgarprojdisk(disk_).nbits()-nrbits_))&((1<<nrbits_)-1);
	  
	  int phiderindex=(proj->fpgaphiprojderdisk(disk_).value()>>(proj->fpgaphiprojderdisk(disk_).nbits()-nphiderbits_))&((1<<nphiderbits_)-1);
	  
	  int signindex=(proj->fpgarprojderdisk(disk_).value()<0);
	  
	  int bendindex=(signindex<<(nphiderbits_+nrbits_))+
	    (rindex<<(nphiderbits_))+
	    phiderindex;
	  
	  int ibendproj=bendTable(abs(disk_)-1,bendindex);
	  
	  proj->setBendIndex(ibendproj,disk_);
	  
	}
	
	int projrinv=barrel?(16+(proj->fpgarinv().value()>>(proj->fpgarinv().nbits()-5))):proj->getBendIndex(disk_).value();
	//if (!barrel) cout << "barrel projrinv disk_ : "<<barrel<<" "<<projrinv<<" "<<disk_<<" "<<proj->getBendIndex(disk_).value()<<endl;
	if (!barrel) assert(proj->validProjDisk(disk_));
	assert(projrinv>=0);
	assert(projrinv<32);
	  
	bool isPSseed=proj->PSseed()==1;
	  
	//cout << "rzfirst rzlast : "<<rzfirst<<" "<<rzlast<<endl;
	for (unsigned int irzbin=rzfirst;irzbin<=rzlast;irzbin++) {
	  unsigned int nstub=stubmem->nStubsBin(irzbin);
	  //cout << "irzbin nstub : "<<irzbin<<" "<<nstub<<endl;

	  //Calculate fine z position
	  if (irzbin!=rzfirst) {
	    projfinerzadj=projfinerz-8;
	  } else {
	    projfinerzadj=projfinerz;
	  }


	  
	  for (unsigned int istub=0;istub<nstub; istub++) {
	  
	    //Read stub memory and extract data fields
	    std::pair<Stub*,L1TStub*> stub=stubmem->getStubBin(irzbin,istub);

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
		//Tracklet* proj=vmprojs_->getFPGATracklet(projindex);
		std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > tmp(proj,stub); 
		if (writeSeeds) {
		  ofstream fout("seeds.txt", ofstream::app);
		  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << proj->getISeed() << endl;
		  fout.close();
		}
		//cout << "Have rough match"<<endl;
		candmatches.push_back(tmp);
		countpass++;
	      }
	    }
	  }
	}
	
      }
    }
    
    //cout << "candmatches.size() = "<<candmatches.size()<<endl;

    Tracklet* oldTracklet=0;
    
    unsigned int countall=0;
    unsigned int countsel=0;
    
    for(unsigned int j=0;j<candmatches.size();j++){
	
      if (debug1&&j==0) {
        cout << getName() <<" has "<<candmatches.size()<<" candidate matches"<<endl;
      }
      
      countall++;
	
      L1TStub* stub=candmatches[j].second.second;
      Stub* fpgastub=candmatches[j].second.first;
      Tracklet* tracklet=candmatches[j].first;

      if (oldTracklet!=0) {
	//allow equal here since we can have more than one cadidate match per tracklet projection
	assert(oldTracklet->TCID()<=tracklet->TCID());
      }
      oldTracklet=tracklet;
      
      if (layer_!=0) {
	  
	//Integer calculation
	
	int ir=fpgastub->r().value();
       	int iphi=tracklet->fpgaphiproj(layer_).value();
	int icorr=(ir*tracklet->fpgaphiprojder(layer_).value())>>icorrshift_;	
	iphi+=icorr;
	
	int iz=tracklet->fpgazproj(layer_).value();
	int izcor=(ir*tracklet->fpgazprojder(layer_).value()+(1<<(icorzshift_-1)))>>icorzshift_;
	iz+=izcor;	

	int ideltaz=fpgastub->z().value()-iz;
	int ideltaphi=(fpgastub->phi().value()<<phi0shift_)-(iphi<<(phi0bitshift-1+phi0shift_)); 


	//Floating point calculations

	double phi=stub->phi();
	double r=stub->r();
	double z=stub->z();

	
	if (useapprox) {
	  double dphi=Util::phiRange(phi-fpgastub->phiapprox(phimin_,phimax_));
	  //cout << "l1stub->phi - fpgastub->phiapprox "<<phi<<" "<<fpgastub->phiapprox(phimin_,phimax_)<<endl;
	  assert(fabs(dphi)<0.001);
	  phi=fpgastub->phiapprox(phimin_,phimax_);
	  z=fpgastub->zapprox();
	  r=fpgastub->rapprox();
	}
	
	if (phi<0) phi+=2*M_PI;
	phi-=phioffset_;
	
	double dr=r-tracklet->rproj(layer_);
	assert(fabs(dr)<drmax);

	double dphi=Util::phiRange(phi-(tracklet->phiproj(layer_)+dr*tracklet->phiprojder(layer_)));
	
       	double dz=z-(tracklet->zproj(layer_)+dr*tracklet->zprojder(layer_));
	
	double dphiapprox=Util::phiRange(phi-(tracklet->phiprojapprox(layer_)+
						  dr*tracklet->phiprojderapprox(layer_)));
	    
	double dzapprox=z-(tracklet->zprojapprox(layer_)+
				   dr*tracklet->zprojderapprox(layer_));
	
	int seedindex=tracklet->getISeed();

	assert(phimatchcut_[seedindex]>0);
	assert(zmatchcut_[seedindex]>0);

	bool truthmatch=tracklet->stubtruthmatch(stub);

	HistBase* hists=GlobalHistTruth::histograms();
	hists->FillLayerResidual(layer_, seedindex,
				 dphiapprox*rmean[layer_-1],
				 ideltaphi*kphi1*rmean[layer_-1],
				 ideltaz*fact_*kz, dz,truthmatch);
    
   

	
	if (writeResiduals) {
	  static ofstream out("layerresiduals.txt");
	  
	  double pt=0.003*3.8/fabs(tracklet->rinv());
	  
	  out << layer_<<" "<<seedindex<<" "<<pt<<" "<<ideltaphi*kphi1*rmean[layer_-1]
	      <<" "<<dphiapprox*rmean[layer_-1]
	      <<" "<<phimatchcut_[seedindex]*kphi1*rmean[layer_-1]
	      <<"   "<<ideltaz*fact_*kz<<" "<<dz<<" "<<zmatchcut_[seedindex]*kz<<endl;	  
	}


	//cout << "Layer "<<layer_<<" phimatch "<<(fabs(ideltaphi)<=phimatchcut_[seedindex])<<" "<<dphi*rmean[layer_-1]*10
	//     <<"mm zmatch "<<(fabs(ideltaz*fact_)<=zmatchcut_[seedindex])<<endl;
	
	bool imatch=(fabs(ideltaphi)<=phifact_*phimatchcut_[seedindex])&&(fabs(ideltaz*fact_)<=rzfact_*zmatchcut_[seedindex]);

	//if (!imatch) {
	//  cout << "Match fail in layer "<<layer_<<" dphi dz : "<< (fabs(ideltaphi)<=phimatchcut_[seedindex])<<" "<<(fabs(ideltaz*fact_)<=zmatchcut_[seedindex])<<" "<<(fabs(ideltaphi)/(phimatchcut_[seedindex]))<<" "<<fabs(ideltaz*fact_)/(zmatchcut_[seedindex])<<endl;
	//}
	
	if (debug1) {
	  cout << getName()<<" imatch = "<<imatch<<" ideltaphi cut "<<ideltaphi
	       <<" "<<phimatchcut_[seedindex]
	       <<" ideltaz*fact cut "<<ideltaz*fact_<<" "<<zmatchcut_[seedindex]<<endl;
	}

	if (fabs(dphi)>0.2 || fabs(dphiapprox)>0.2 ) {
	  cout << "WARNING dphi and/or dphiapprox too large : "
	       <<dphi<<" "<<dphiapprox<<endl;
	}
	
	assert(fabs(dphi)<0.2);
	assert(fabs(dphiapprox)<0.2);
	
	if (imatch) {
	  
	  std::pair<Stub*,L1TStub*> tmp(fpgastub,stub);
	  
	  countsel++;
	  
	  tracklet->addMatch(layer_,ideltaphi,ideltaz,
			     dphi,dz,dphiapprox,dzapprox,
			     (fpgastub->phiregion().value()<<7)+fpgastub->stubindex().value(),
			     stub->r(),tmp);
	  

	  if (debug1) {
	    cout << "Accepted full match in layer " <<getName()
		 << " "<<tracklet
		 << " "<<iSector_<<endl;	   
	  }
	      
	  for (unsigned int l=0;l<fullmatches_.size();l++){
	    if (debug1) {
	      cout << getName()<< " Trying to add match to: "<<fullmatches_[l]->getName()<<" "
		   <<tracklet->layer()<<" "<<tracklet->disk()<<" "<<fullmatches_[l]->getName().substr(3,4)
		   <<endl;
	    }
	    int layer=tracklet->layer();
	    int disk=abs(tracklet->disk());
	    if (!extended_) {
	      if ((layer==1&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L1L2")||
		  (layer==2&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L2L3")||
		  (layer==3&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L3L4")||
		  (layer==5&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L5L6")||
		  (layer==0&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="D1D2")||
		  (layer==0&&disk==3&&fullmatches_[l]->getName().substr(3,4)=="D3D4")||
		  (layer==1&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="L1D1")||
		  (layer==2&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="L2D1")){
		if (debug1) {
		  cout << getName()<<" adding match to "<<fullmatches_[l]->getName()<<endl;
		}
		if (writeSeeds) {
		  ofstream fout("seeds.txt", ofstream::app);
		  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
		  fout.close();
		}
		fullmatches_[l]->addMatch(tracklet,tmp);
	      } 
	    }
	    else {
	      int iSeed = tracklet->getISeed ();
	      if ((iSeed==0&&fullmatches_[l]->getName().substr(3,6)=="L1L2XX")||
		  (iSeed==1&&fullmatches_[l]->getName().substr(3,6)=="L2L3XX")||
		  (iSeed==2&&fullmatches_[l]->getName().substr(3,6)=="L3L4XX")||
		  (iSeed==3&&fullmatches_[l]->getName().substr(3,6)=="L5L6XX")||
		  (iSeed==4&&fullmatches_[l]->getName().substr(3,6)=="D1D2XX")||
		  (iSeed==5&&fullmatches_[l]->getName().substr(3,6)=="D3D4XX")||
		  (iSeed==6&&fullmatches_[l]->getName().substr(3,6)=="L1D1XX")||
		  (iSeed==7&&fullmatches_[l]->getName().substr(3,6)=="L2D1XX")||
		  (iSeed==8&&fullmatches_[l]->getName().substr(3,6)=="L3L4L2")||
		  (iSeed==9&&fullmatches_[l]->getName().substr(3,6)=="L5L6L4")||
		  (iSeed==10&&fullmatches_[l]->getName().substr(3,6)=="L2L3D1")||
		  (iSeed==11&&fullmatches_[l]->getName().substr(3,6)=="D1D2L2")){
		if (debug1) {
		  cout << getName()<<" adding match to "<<fullmatches_[l]->getName()<<endl;
		}
		if (writeSeeds) {
		  ofstream fout("seeds.txt", ofstream::app);
		  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
		  fout.close();
		  }
		fullmatches_[l]->addMatch(tracklet,tmp);
	      } 
	    }
	  }	  
	}
      } else {  //disk matches
	
	
	//check that stubs and projections in same half of detector
	assert(stub->z()*tracklet->t()>0.0);

	int sign=(tracklet->t()>0.0)?1:-1;
	int disk=sign*disk_;
	assert(disk!=0);
	  
	//Perform integer calculations here

	int iz=fpgastub->z().value();
	int iphi=tracklet->fpgaphiprojdisk(disk).value();

	int shifttmp=t2bits+tbitshift+phi0bitshift+2-rinvbitshiftdisk-phiderdiskbitshift-PS_phiderD_shift;
	assert(shifttmp>=0);
	int iphicorr=(iz*tracklet->fpgaphiprojderdisk(disk).value())>>shifttmp;
	
	iphi+=iphicorr;

	int ir=tracklet->fpgarprojdisk(disk).value();
	
	  
	int shifttmp2=rprojdiskbitshift+t3shift-rderdiskbitshift;
	
	assert(shifttmp2>=0);
	int ircorr=(iz*tracklet->fpgarprojderdisk(disk).value())>>shifttmp2;
										
	ir+=ircorr;

	int ideltaphi=fpgastub->phi().value()*kphi/kphiproj123-iphi; 


	int irstub = fpgastub->r().value();
	int ialphafact=0;
	if(!stub->isPSmodule()){
	  assert(irstub<10);
	  if (disk_<=2) {
	    ialphafact = ialphafactinner_[irstub];
	    irstub = rDSSinner[irstub]/kr;
	  } else {
	    ialphafact = ialphafactouter_[irstub];
	    irstub = rDSSouter[irstub]/kr;
	  }
	}
	
	int ideltar=(irstub*krdisk)/krprojshiftdisk-ir;

	if (!stub->isPSmodule()) {	  
	  int ialphanew=fpgastub->alphanew().value();
	  int iphialphacor=((ideltar*ialphanew*ialphafact)>>alphashift);
	  ideltaphi+=iphialphacor;
	}



	
	//Perform floating point calculations here
	
	double phi=stub->phi();
	double z=stub->z();
	double r=stub->r();

		
	if (useapprox) {
	  double dphi=Util::phiRange(phi-fpgastub->phiapprox(phimin_,phimax_));
	  assert(fabs(dphi)<0.001);
	  phi=fpgastub->phiapprox(phimin_,phimax_);
	  z=fpgastub->zapprox();
	  r=fpgastub->rapprox();
	}

	if (phi<0) phi+=2*M_PI;
	phi-=phioffset_;

	double dz=z-sign*zmean[disk_-1];
	
	if(fabs(dz) > dzmax){
	  cout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
	  cout << "stub "<<stub->z() <<" disk "<<disk<<" "<<dz<<endl;
	  assert(fabs(dz)<dzmax);
	}	
		  

	double phiproj=tracklet->phiprojdisk(disk)+dz*tracklet->phiprojderdisk(disk);

	double rproj=tracklet->rprojdisk(disk)+dz*tracklet->rprojderdisk(disk);
		
	double deltar=r-rproj;
	  
	
	double dr=stub->r()-rproj;
	
	double dphi=Util::phiRange(phi-phiproj);

	double dphiapprox=Util::phiRange(phi-(tracklet->phiprojapproxdisk(disk)+
						  dz*tracklet->phiprojderapproxdisk(disk)));

	double drapprox=stub->r()-(tracklet->rprojapproxdisk(disk)+
				   dz*tracklet->rprojderapproxdisk(disk));
	
	double drphi=dphi*stub->r();
	double drphiapprox=dphiapprox*stub->r();


	
	if (!stub->isPSmodule()) {
	  double alphanew=stub->alphanew();
	  dphi+=dr*alphanew*half2SmoduleWidth/stub->r2();;
	  dphiapprox+=drapprox*alphanew*half2SmoduleWidth/stub->r2();
	  
	  drphi+=dr*alphanew*half2SmoduleWidth/stub->r();
	  drphiapprox+=dr*alphanew*half2SmoduleWidth/stub->r();
	}


	int seedindex=tracklet->getISeed();
	
	int idrphicut=rphicutPS_[seedindex];
	int idrcut=rcutPS_[seedindex]; 
	if (!stub->isPSmodule()) {
	  idrphicut=rphicut2S_[seedindex];
	  idrcut=rcut2S_[seedindex]; 
	}

	double drphicut=idrphicut*kphiproj123*kr;
	double drcut=idrcut*krprojshiftdisk;

	
	if (writeResiduals) {
	  static ofstream out("diskresiduals.txt");
	  
	  double pt=0.003*3.8/fabs(tracklet->rinv());
	  
	  out << disk_<<" "<<stub->isPSmodule()<<" "<<tracklet->layer()<<" "
	      <<abs(tracklet->disk())<<" "<<pt<<" "
	      <<ideltaphi*kphiproj123*stub->r()<<" "<<drphiapprox<<" "
	      <<drphicut<<" "
	      <<ideltar*krprojshiftdisk<<" "<<deltar<<" "
	      <<drcut<<" "
	      <<endl;	  
	}

	
	bool match=(fabs(drphi)<drphicut)&&(fabs(deltar)<drcut);
	
	bool imatch=(fabs(ideltaphi*irstub)<idrphicut)&&(fabs(ideltar)<idrcut);


	if (debug1) {
	  cout << "imatch match disk: "<<imatch<<" "<<match<<" "
	       <<fabs(ideltaphi)<<" "<<drphicut/(kphiproj123*stub->r())<<" "
	       <<fabs(ideltar)<<" "<<drcut/krprojshiftdisk<<" r = "<<stub->r()<<endl;
	}
		
	  
	if (imatch) {

	  std::pair<Stub*,L1TStub*> tmp(fpgastub,stub);
	    
	  countsel++;
	  
	  if (debug1) {
	    cout << "MatchCalculator found match in disk "<<getName()<<endl;
	  }

          if(fabs(dphi)>=0.25){
            cout<<"dphi "<<dphi<<"\n";
            cout<<"ISeed "<<tracklet->getISeed()<<"\n";
          }
          assert(fabs(dphi)<0.25);
          assert(fabs(dphiapprox)<0.25);

	  tracklet->addMatchDisk(disk,ideltaphi,ideltar,
				 drphi/stub->r(),dr,drphiapprox/stub->r(),drapprox,
				 stub->alpha(),
				 (fpgastub->phiregion().value()<<7)+fpgastub->stubindex().value(),
				 stub->z(),tmp);
	  if (debug1) {
	    cout << "Accepted full match in disk " <<getName()
		 << " "<<tracklet
		 << " "<<iSector_<<endl;	   
	  }
	  
	  bool foundmem=false;
	  int layer=tracklet->layer();
	  int disk=abs(tracklet->disk());
	  for (unsigned int l=0;l<fullmatches_.size();l++){
	    if (!extended_) {
	      if ((layer==1&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L1L2")||
		  (layer==3&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L3L4")||
		  (layer==5&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L5L6")||
		  (layer==0&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="D1D2")||
		  (layer==0&&disk==3&&fullmatches_[l]->getName().substr(3,4)=="D3D4")||
		  (layer==1&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="L1D1")||
		  (layer==2&&disk==1&&fullmatches_[l]->getName().substr(3,4)=="L2D1")||
		  (layer==2&&disk==0&&fullmatches_[l]->getName().substr(3,4)=="L2L3")){
		if (debug1) {
		  cout << getName()<<" adding match to "<<fullmatches_[l]->getName()<<endl;
		}
		if (writeSeeds) {
		  ofstream fout("seeds.txt", ofstream::app);
		  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
		  fout.close();
		}
		foundmem=true;
		fullmatches_[l]->addMatch(tracklet,tmp);
	      } 
	    }
	    else {
	      int iSeed = tracklet->getISeed ();
	      if ((iSeed==0&&fullmatches_[l]->getName().substr(3,6)=="L1L2XX")||
		  (iSeed==1&&fullmatches_[l]->getName().substr(3,6)=="L2L3XX")||
		  (iSeed==2&&fullmatches_[l]->getName().substr(3,6)=="L3L4XX")||
		  (iSeed==3&&fullmatches_[l]->getName().substr(3,6)=="L5L6XX")||
		  (iSeed==4&&fullmatches_[l]->getName().substr(3,6)=="D1D2XX")||
		  (iSeed==5&&fullmatches_[l]->getName().substr(3,6)=="D3D4XX")||
		  (iSeed==6&&fullmatches_[l]->getName().substr(3,6)=="L1D1XX")||
		  (iSeed==7&&fullmatches_[l]->getName().substr(3,6)=="L2D1XX")||
		  (iSeed==8&&fullmatches_[l]->getName().substr(3,6)=="L3L4L2")||
		  (iSeed==9&&fullmatches_[l]->getName().substr(3,6)=="L5L6L4")||
		  (iSeed==10&&fullmatches_[l]->getName().substr(3,6)=="L2L3D1")||
		  (iSeed==11&&fullmatches_[l]->getName().substr(3,6)=="D1D2L2")){
		if (debug1) {
		  cout << getName()<<" adding match to "<<fullmatches_[l]->getName()<<endl;
		}
		if (writeSeeds) {
		  ofstream fout("seeds.txt", ofstream::app);
		  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
		  fout.close();
		}
		foundmem=true;
		fullmatches_[l]->addMatch(tracklet,tmp);
	      }
	    }
	  }
	  if (!foundmem) {
	    cout << getName()<<" layer:"<<layer<<" disk="<<disk<<endl;
	  }
	  assert(foundmem);
	}
      }
      //if (countall>=MAXMC) break;
    }


    if (writeMatchCalculator) {
      static ofstream out("matchcalculator.txt");
      out << getName()<<" "<<countall<<" "<<countsel<<endl;
    }


    
    
    //assert(0);
    
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

  
    
private:

  int layer_;
  int disk_;
  int fact_;
  int icorrshift_;
  int icorzshift_;
  int phi0shift_;
  int phimatchcut_[12];
  int zmatchcut_[12];
  double phimin_;
  double phimax_;
  double phioffset_;

  double rphicutPS_[12];
  double rphicut2S_[12];
  double rcutPS_[12];
  double rcut2S_[12];

  double phifact_;
  double rzfact_;

  int nrbits_;
  int nphiderbits_;
  
  AllStubsMemory* allstubs_;
  vector<VMStubsMEMemory*> vmstubs_;
  vector<TrackletProjectionsMemory*> inputprojs_;

  int ialphafactinner_[10];
  int ialphafactouter_[10];

  
  vector<CandidateMatchMemory*> matches_;

  vector<FullMatchMemory*> fullmatches_;
  FullMatchMemory* fullmatchesToPlus_;
  FullMatchMemory* fullmatchesToMinus_;

  //used in the layers
  vector<bool> table_;

  //used in the disks
  vector<bool> tablePS_;
  vector<bool> table2S_;
  
};

#endif
