#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include "CommonTools/TrackerMap/interface/TmCcu.h"
#include "CommonTools/TrackerMap/interface/TmPsu.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include "TCanvas.h"
#include "TPolyLine.h"
#include "TStyle.h"
#include "TColor.h"
#include "TROOT.h"
#include "TGaxis.h"
#include "TLatex.h"
#include "TArrow.h"


/**********************************************************
Allocate all the modules in a map of TmModule
The filling of the values for each module is done later
when the user starts to fill it.
**********************************************************/

TrackerMap::TrackerMap(const edm::ParameterSet & tkmapPset,const SiStripFedCabling* tkFed) {

 psetAvailable=true;
  xsize=340;ysize=200;

  title=" ";
  jsPath="";
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  saveAsSingleLayer=false;
  tkMapLog = false;
  //  if(tkmapPset.exists("trackermaptxtPath")){
  jsPath=tkmapPset.getUntrackedParameter<std::string>("trackermaptxtPath","CommonTools/TrackerMap/data/");
  jsfilename=jsPath+"trackermap.txt";
  std::cout << jsfilename << std::endl;
  infilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","CommonTools/TrackerMap/data/")+"tracker.dat";
  std::cout << infilename << std::endl;
  saveWebInterface=tkmapPset.getUntrackedParameter<bool>("saveWebInterface",false);
  saveGeoTrackerMap=tkmapPset.getUntrackedParameter<bool>("saveGeoTrackerMap",true);
  ncrates=0;
  firstcrate=0;
  enableFedProcessing=tkmapPset.getUntrackedParameter<bool>("loadFedCabling",false);
  if(tkFed==0 && enableFedProcessing){enableFedProcessing=false;std::cout << "ERROR:fed trackermap requested but no valid fedCabling is available!!!"<<std::endl;}
  nfeccrates=0;
  enableFecProcessing=tkmapPset.getUntrackedParameter<bool>("loadFecCabling",false);
  if(tkFed==0 && enableFecProcessing){enableFecProcessing=false;std::cout << "ERROR:fec trackermap requested but no valid fedCabling is available!!!"<<std::endl;} 
 // std::cout << "loadFecCabling " << enableFecProcessing << std::endl;
  npsuracks=0;
  enableLVProcessing=tkmapPset.getUntrackedParameter<bool>("loadLVCabling",false);
 // std::cout << "loadLVCabling " << enableLVProcessing << std::endl;
  enableHVProcessing=tkmapPset.getUntrackedParameter<bool>("loadHVCabling",false);
 // std::cout << "loadHVCabling " << enableHVProcessing << std::endl;
  tkMapLog = tkmapPset.getUntrackedParameter<bool>("logScale",false);
  //  } else std::cout << "no parameters found" << std::endl;

 init();
// Now load fed cabling information
 if(enableFedProcessing){
 const std::vector<unsigned short> feds = tkFed->feds();
  std::cout<<"SiStripFedCabling has "<< feds.size()<<" active FEDS"<<std::endl;
  //    int num_board=0;
    //    int num_crate=0;
  for(std::vector<unsigned short>::const_iterator ifed = feds.begin();ifed<feds.end();ifed++){
    const std::vector<FedChannelConnection> theconn = tkFed->connections( *ifed );
    int num_conn=0;
    for(std::vector<FedChannelConnection>::const_iterator iconn = theconn.begin();iconn<theconn.end();iconn++){

      if( iconn->fedId()== sistrip::invalid_    ||  
	  iconn->detId() == sistrip::invalid_   ||  
	  iconn->detId() == sistrip::invalid32_ ||  
	  iconn->apvPairNumber() == sistrip::invalid_  ||
	  iconn->nApvPairs() == sistrip::invalid_ ) {
	continue;
      }
	
      TmModule *imod = imoduleMap[iconn->detId()];
      int key = iconn->fedId()*1000+iconn->fedCh();
      TmApvPair* apvpair = apvMap[key];
      if(apvpair!=0)std::cout << "Fed "<< iconn->fedId() << " channel " << iconn->fedCh() << " seem to be already loaded!"<<std::endl;
      else
	{
	  num_conn++;
	  if(num_conn==1){
	    //	    if(fedMap[iconn->fedId()]==0){num_crate=num_board/18+1;fedMap[iconn->fedId()]=num_crate;num_board++;}
	    if(fedMap[iconn->fedId()]==0){fedMap[iconn->fedId()]=iconn->fedCrate();}
	    if(slotMap[iconn->fedId()]==0){slotMap[iconn->fedId()]=iconn->fedSlot();}
	    if(ncrates==0 || ncrates < iconn->fedCrate()) ncrates = iconn->fedCrate();
	    if(firstcrate==0 || firstcrate > iconn->fedCrate()) firstcrate = iconn->fedCrate();
          }

	  //	  apvpair = new TmApvPair(key,num_crate);
	  apvpair = new TmApvPair(key,iconn->fedCrate());
	  apvpair->mod=imod;
          apvpair->mpos=iconn->apvPairNumber();
	  apvMap[key] = apvpair;	
          apvModuleMap.insert(std::make_pair(iconn->detId(),apvpair));
	  std::stringstream s;
          iconn->print(s);  
          apvpair->text=s.str();
	}
    }
  }
  //  ncrates=num_crate;
  std::cout << "from " << firstcrate << " to " << ncrates << " crates used "<< std::endl;
//Now add APv information to module name
    std::map<int , TmModule *>::iterator i_mod;
    for( i_mod=imoduleMap.begin();i_mod !=imoduleMap.end(); i_mod++){
      TmModule *  mod= i_mod->second;
      if(mod!=0) {
	std::ostringstream outs,outs1;
       outs << " connected to ";
       outs1 << "(";

      int idmod=mod->idex;
       int nchan=0;
       std::multimap<const int, TmApvPair*>::iterator pos;
       for (pos = apvModuleMap.lower_bound(idmod);
         pos != apvModuleMap.upper_bound(idmod); ++pos) {
       TmApvPair* apvpair = pos->second;
       if(apvpair!=0){
       outs << apvpair->mpos << " " <<apvpair->getFedId() << "/"<<apvpair->getFedCh()<<" ";
       outs1 << apvpair->idex+apvpair->crate*1000000<<",";
      nchan++;
    }

  }
       outs<< "("<<nchan<<")";
      mod->name=mod->name + outs.str(); 
      std::string s = outs1.str(); s.erase(s.end()-1,s.end());
      mod->capvids=s+")";
  }
  }
}
// Now load fec cabling information
 if(enableFecProcessing){
   int nfec=0; int nccu; int nmod;
   int crate,slot,ring,addr,pos;
   SiStripFecCabling* fecCabling_;
   fecCabling_ = new SiStripFecCabling( *tkFed );
   std::string Ccufilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","")+"cculist.txt";
   ifstream Ccufile(edm::FileInPath(Ccufilename).fullPath().c_str(),std::ios::in);
   std::string dummys;
   while(!Ccufile.eof()) {
     Ccufile >> crate >> slot >> ring >> addr >> pos;
     getline(Ccufile,dummys);
     int key =crate*10000000+slot*100000+ring*1000+addr;
     TmCcu * ccu = ccuMap[key];
     if(ccu==0){
       ccu = new TmCcu(crate,slot,ring,addr);
       ccu->mpos=pos,
         ccuMap[key]=ccu;
     }
   }

   for ( std::vector<SiStripFecCrate>::const_iterator icrate = fecCabling_->crates().begin(); icrate != fecCabling_->crates().end(); icrate++ ) {
     for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
       for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
         nccu=0;nfec++;
         for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
           nccu++; nmod=0;
           int key = icrate->fecCrate()*10000000+ifec->fecSlot()*100000+iring->fecRing()*1000+iccu->ccuAddr();
           int layer=0;
           TmCcu * ccu = ccuMap[key];
           for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
             nmod++;
             TmModule *imod1 = imoduleMap[imod->detId()];
             layer=imod1->layer;
             fecModuleMap.insert(std::make_pair(ccu,imod1));
             if(imod1!=0)imod1->CcuId=key;//imod1->ccuId=key+Crate*1000000
           }
           if(ccu==0)std::cout <<key<< " This ccu seems to have not been stored! " << std::endl; else{ ccu->nmod=nmod;ccu->layer=layer;}
           //std::cout <<nfec<<" "<< nccu << " " << nmod << std::endl;

         }
       }
     }
   }

   std::map<int , TmCcu *>::iterator i_ccu;
   std::multimap<TmCcu*, TmModule*>::iterator it;
   std::pair<std::multimap<TmCcu*, TmModule*>::iterator,std::multimap<TmCcu*, TmModule*>::iterator> ret;
   nccu=0;
   for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
     TmCcu *  ccu= i_ccu->second;
     nccu++;
     if(ccu!=0){
       std::ostringstream outs;
       std::ostringstream outs1;
       outs << "CCU "<<ccu->idex <<" connected to fec,ring " << ccu->getCcuSlot() <<","<<ccu->getCcuRing()<< " in crate " <<ccu->getCcuCrate()<<" at position "<< ccu->mpos << " with  " << ccu->nmod << " modules: ";
        outs1<<"(";
        ret = fecModuleMap.equal_range(ccu);
        for (it = ret.first; it != ret.second; ++it)
          {
            outs << (*it).second->idex << " ";
            outs1 << (*it).second->getKey() <<",";
          }
        outs1 << ")";
        ccu->text=outs.str();
        ccu->cmodid=outs1.str();
        //std::cout << ccu->text << std::endl;
     }

   }
   nfeccrates=4;
   std::cout << nccu << " ccu stored in " <<nfeccrates<< " crates"<< std::endl;

   delete fecCabling_ ;

 }
//load Psu cabling info
 //load Psu cabling info 
 if(enableLVProcessing || enableHVProcessing){

   SiStripDetCabling* detCabling = 0;
   if(enableFedProcessing) detCabling = new SiStripDetCabling( *tkFed );


   int npsu=0; int nmod,nmodHV2,nmodHV3;
   int modId1, dcuId; // ,modId2;
   int dcs,branch,crate,board;
   int rack=0;
   std::string channelstr1;
   short int channel;
   std::string psinfo;
   std::string psIdinfo;
   int rack_order[54]={0,1,0,2,0,3,0,
                         4,0,5,6,0,7,
                         8,0,9,10,0,11,
                         12,0,13,14,0,15,
                         0,0,0,0,0,0,
                         16,0,17,18,0,19,
                         20,0,21,0,22,0,
                         23,0,24,25,0,26,
                         27,0,28,0,29};
 //  ifstream *LVfile;
 //  ifstream *HVfile;
   
  
 
  std::string LVfilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","CommonTools/TrackerMap/data/")+"psdcumap.dat";
  //std::string HVfilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","")+"hvmap.dat";
  
  ifstream LVfile(edm::FileInPath(LVfilename).fullPath().c_str(),std::ios::in);
  
  std::cout<<LVfilename<<std::endl;
  
 /* 
   if(enableHVProcessing){
	    ifstream HVfile(edm::FileInPath(HVfilename).fullPath().c_str(),std::ios::in);
	    while(!HVfile.eof()) {
	    HVfile >> modId2 >> channelstr1;
	    std::string channelstr2 = channelstr1.substr(9,1);
            channel= atoi(channelstr2.c_str());
	    TmModule *imod = imoduleMap[modId2];
	   // if(modId1==modId2){
            imod->HVchannel=channel;      
           
	      } 
           	
	    }
*/
	
  
   while(!LVfile.eof()) {
      LVfile >> modId1  >> dcuId >> psIdinfo >> psinfo;
      
      if(detCabling && detCabling->getConnections(modId1).size()==0) continue;

      //      int length=psinfo.length();
      std::string dcsinfo = psinfo.substr(39,1);
      std::string branchinfo = psinfo.substr(57,2);
      std::string crateinfo= psinfo.substr(69,1);
      std::string boardinfo = psinfo.substr(80,2);
      std::string channelinfo = psinfo.substr(90,3);
    
      dcs= atoi(dcsinfo.c_str());
      branch= atoi(branchinfo.c_str());
      crate= atoi(crateinfo.c_str())+1;
      board= atoi(boardinfo.c_str())+1;
      rack = (branch+1)+(dcs-1)*6; 
      rack = rack_order[rack]; 
      channel = atoi(channelinfo.c_str());
      //      std::cout << dcs << " " << branch<< " " <<crate<< " " << board<<" " << rack << std::endl;
      int key = rack*1000+crate*100+board;
      
      TmPsu *psu = psuMap[key];
      TmModule *imod = imoduleMap[modId1];
      if(psu==0){
        psu = new TmPsu(dcs,branch,rack,crate,board);
        psuMap[key]=psu;
        psu->psId=psIdinfo;
      }
   
      psuModuleMap.insert(std::make_pair(psu,imod));
      if(imod!=0){imod->PsuId=psIdinfo;imod->psuIdex=psu->idex;imod->HVchannel=channel;}
       
   }
      
   
 //  int nmax=0; 
   std::map<int , TmPsu *>::iterator ipsu;
   std::multimap<TmPsu*, TmModule*>::iterator it;
   std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
   npsu=0;
  
   for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
     TmPsu *  psu= ipsu->second;
     npsu++;
    
    if(psu!=0){
	
      std::ostringstream outs;
      std::ostringstream outs1;
	
      std::ostringstream outs3;
      std::ostringstream outs4;
	
      std::ostringstream outs5;
      std::ostringstream outs6;
	
	outs <<"PSU "<<psu->psId<<" connected to Mainframe "<<psu->getPsuDcs()<<" BranchController "<<psu->getPsuBranch()<<" (Rack "<<psu->getPsuRack()<<"), crate "<<psu->getPsuCrate()<<" in position "<< psu->getPsuBoard()<< " with modules: ";
        outs1<<"(";
	
	if(enableHVProcessing){
	  outs3 <<"PSU "<<psu->psId<<" connected to Mainframe "<<psu->getPsuDcs()<<" BranchController "<<psu->getPsuBranch()<<" (Rack "<<psu->getPsuRack()<<"),crate "<<psu->getPsuCrate()<<" in position "<< psu->getPsuBoard()<<" and HV channel 002 with modules: ";
          outs4<<"(";
	
	  outs5 <<"PSU "<<psu->psId<<" connected to Mainframe "<<psu->getPsuDcs()<<" BranchController "<<psu->getPsuBranch()<<" (Rack "<<psu->getPsuRack()<<"), crate "<<psu->getPsuCrate()<<" in position "<< psu->getPsuBoard()<<" and HV channel 002 with modules: ";
          outs6<<"(";}
	
	
	ret = psuModuleMap.equal_range(psu);
        nmod=0;
	nmodHV2=0;
	nmodHV3=0;
	for (it = ret.first; it != ret.second; ++it)
	  {
	    nmod++; 
	    outs << (*it).second->idex << ", ";
	    outs1 << (*it).second->getKey() <<",";
	
	    if(enableHVProcessing){
	      if((*it).second->HVchannel==2){
	      nmodHV2++;
	      outs3 << (*it).second->idex << ", ";
	      outs4 << (*it).second->getKey() <<",";}
	      else if((*it).second->HVchannel==3){
	      nmodHV3++;
	      outs5 << (*it).second->idex << ", ";
	      outs6 << (*it).second->getKey() <<",";}
	
	      }
	  }

	outs1 << ")";
	psu->nmod=nmod;
	outs << "(" << psu->nmod << ")";
	psu->text=outs.str();
        psu->cmodid_LV=outs1.str();
        if(enableHVProcessing){
        outs4 << ")";
	outs6 << ")";
	psu->nmodHV2=nmodHV2;
	psu->nmodHV3=nmodHV3;
	outs3 << "(" << psu->nmodHV2 << ")";
	outs5 << "(" << psu->nmodHV3 << ")";
	psu->textHV2=outs3.str();
	psu->textHV3=outs5.str();
	psu->cmodid_HV2=outs4.str();
        psu->cmodid_HV3=outs6.str();
        }
    }
   }
  
  
   npsuracks=29;
   std::cout << npsu << " psu stored in " <<npsuracks<<" racks"<<std::endl;
  }
}

TrackerMap::TrackerMap(const edm::ParameterSet & tkmapPset) {
 psetAvailable=true;
  xsize=340;ysize=200;
  title=" ";
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  enableFedProcessing=true;ncrates=0;firstcrate=0;
  saveAsSingleLayer=false;
  tkMapLog = tkmapPset.getUntrackedParameter<bool>("logScale",false);
  saveWebInterface=tkmapPset.getUntrackedParameter<bool>("saveWebInterface",false);
  saveGeoTrackerMap=tkmapPset.getUntrackedParameter<bool>("saveGeoTrackerMap",true);
  //  if(tkmapPset.exists("trackermaptxtPath")){
  jsfilename=tkmapPset.getUntrackedParameter<std::string>("trackermaptxtPath","CommonTools/TrackerMap/data/")+"trackermap.txt";
  std::cout << jsfilename << std::endl;
  infilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","CommonTools/TrackerMap/data/")+"tracker.dat";
  std::cout << infilename << std::endl;
  //  } else std::cout << "no parameters found" << std::endl;
 init();
}

TrackerMap::TrackerMap(std::string s,int xsize1,int ysize1) {
 psetAvailable=false;
  xsize=xsize1;ysize=ysize1;
  title=s;
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  saveWebInterface=false;
  saveGeoTrackerMap=true;
  tkMapLog=false;
  jsPath="CommonTools/TrackerMap/data/";
  enableFedProcessing=false; 
  enableFecProcessing=false; 
  enableLVProcessing=false; 
  enableHVProcessing=false; 
  saveAsSingleLayer=false;
 init();

}

void TrackerMap::reset() {
std::map<int , TmModule *>::iterator i_mod;
    for( i_mod=imoduleMap.begin();i_mod !=imoduleMap.end(); i_mod++){
      TmModule *  mod= i_mod->second;
      mod->count=0;mod->value=0;mod->red=-1;
      }
}

void TrackerMap::init() {
  
  int ntotmod=0;
  ix=0;iy=0; //used to compute the place of each layer in the tracker map
  firstcall = true;
  minvalue=0.; maxvalue=minvalue;
  posrel=true;
  palette = 1;
  printflag=true;
  addPixelFlag=false;
  temporary_file=false;
  gminvalue=0.; gmaxvalue=0.;//default global range for online rendering

  ndet = 3; // number of detectors: pixel, inner silicon, outer silicon
  npart = 3; // number of detector parts: endcap -z, barrel, endcap +z

  //allocate module map
  for (int subdet=1; subdet < ndet+1; subdet++){//loop on subdetectors
    for (int detpart=1; detpart < npart+1; detpart++){//loop on subdetectors parts
      int nlayers = getlayerCount(subdet,detpart); // compute number of layers
      for(int layer=1; layer < nlayers+1; layer++){//loop on layers
	int nrings = getringCount(subdet,detpart,layer);// compute number of rings
	//fill arrays used to do the loop on the rings	
        int layer_g = nlayer(subdet,detpart,layer);
	ntotRing[layer_g-1]=nrings;
	firstRing[layer_g-1]=1;
	if(subdet==3 && detpart!=2)  firstRing[layer_g-1]= 8-nrings; //special numbering for TEC 
	for (int ring=firstRing[layer_g-1]; ring < ntotRing[layer_g-1]+firstRing[layer_g-1];ring++){//loop on rings
	  int nmodules = getmoduleCount(subdet,detpart,layer,ring);// compute number of modules
	  int key=0;
	  TmModule *smodule; 
          for(int module=1; module < nmodules+1; module++){//loop on modules
            smodule = new TmModule(module,ring,layer_g);
	    key=layer_g*100000+ring*1000+module;//key identifying module
	    smoduleMap[key]=smodule;
	    ntotmod++;
	  }
	  if(isRingStereo(key))for(int module=1; module < nmodules+1; module++){//loop on stereo modules
            smodule = new TmModule(module+100,ring,layer_g);
	    int key=layer_g*100000+ring*1000+module+100;
	    smoduleMap[key]=smodule;
	    ntotmod++;
	  }
	}
      }
    }
  }
 build();
}

TrackerMap::~TrackerMap() {
 
for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 ) delete mod;
      }
    }
  }

//std::map<int , TmModule *>::iterator i_mod;
//   for( i_mod=imoduleMap.begin();i_mod !=imoduleMap.end(); i_mod++){
//      TmModule *  mod= i_mod->second;
//      delete mod;
//      }
std::map<int , TmApvPair *>::iterator i_apv;
  for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      delete apvPair;
      }


std::map<int , TmCcu *>::iterator i_ccu;
   for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
     TmCcu *  ccu= i_ccu->second;
     delete ccu;
     }

std::map<int , TmPsu *>::iterator ipsu;
    for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
     TmPsu *  psu= ipsu->second;
     delete psu;
     }

gROOT->Reset();


//for(std::vector<TColor*>::iterator col1=vc.begin();col1!=vc.end();col1++){
//     std::cout<<(*col1)<<std::endl;}
}





void TrackerMap::drawModule(TmModule * mod, int key,int mlay, bool print_total, std::ofstream * svgfile){
  //int x,y;
  nlay = mlay;
  double phi,r,dx,dy, dy1;
  double xp[4],yp[4],xp1,yp1;
  double vhbot,vhtop,vhapo;
  double rmedio[]={0.041,0.0701,0.0988,0.255,0.340,0.430,0.520,0.610,0.696,0.782,0.868,0.965,1.080};
  double xt1,yt1,xs1=0.,ys1=0.,xt2,yt2,xs2,ys2,pv1,pv2;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  //int numrec=0;
  int numod=0;
  phi = phival(mod->posx,mod->posy);
  r = sqrt(mod->posx*mod->posx+mod->posy*mod->posy);
  vhbot = mod->width;
  vhtop=mod->width;
  vhapo=mod->length;
  if(mlay < 31){ //endcap
    vhbot = mod->widthAtHalfLength/2.-(mod->width/2.-mod->widthAtHalfLength/2.);
    vhtop=mod->width/2.;
    vhapo=mod->length/2.;
    if(mlay >12 && mlay <19){
      if(posrel)r = r+r;
      xp[0]=r-vhtop;yp[0]=-vhapo;
      xp[1]=r+vhtop;yp[1]=-vhapo;
      xp[2]=r+vhtop;yp[2]=vhapo;
      xp[3]=r-vhtop;yp[3]=vhapo;
    }else{
      if(posrel)r = r + r/3.;
      xp[0]=r-vhapo;yp[0]=-vhbot;
      xp[1]=r+vhapo;yp[1]=-vhtop;
      xp[2]=r+vhapo;yp[2]=vhtop;
      xp[3]=r-vhapo;yp[3]=vhbot;
    }
    for(int j=0;j<4;j++){
      xp1 = xp[j]*cos(phi)-yp[j]*sin(phi);
      yp1 = xp[j]*sin(phi)+yp[j]*cos(phi);
      xp[j] = xp1;yp[j]=yp1;
    }
  } else { //barrel
    numod=mod->idModule;if(numod>100)numod=numod-100;
    int vane = mod->ring;
    if(posrel){
      dx = vhapo;
      phi=M_PI;
      xt1=rmedio[mlay-31]; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=rmedio[mlay-31]; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      dy=phival(xs2,ys2)-phival(xs1,ys1);
	 dy1 = dy;
      if(mlay==31)dy1=0.39;
      if(mlay==32)dy1=0.23;
      if(mlay==33)dy1=0.16;
      xp[0]=vane*(dx+dx/8.);yp[0]=numod*(dy1);
      xp[1]=vane*(dx+dx/8.)+dx;yp[1]=numod*(dy1);
      xp[2]=vane*(dx+dx/8.)+dx;yp[2]=numod*(dy1)+dy;
      xp[3]=vane*(dx+dx/8.);yp[3]=numod*(dy1)+dy;
    }else{
      xt1=r; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=r; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      pv1=phival(xs1,ys1);
      pv2=phival(xs2,ys2);
      if(fabs(pv1-pv2)>M_PI && numod==1)pv1=pv1-2.*M_PI;
      if(fabs(pv1-pv2)>M_PI && numod!=1)pv2=pv2+2.*M_PI;
      xp[0]=mod->posz-vhapo/2.;yp[0]=4.2*pv1;
      xp[1]=mod->posz+vhapo/2.;yp[1]=4.2*pv1;
      xp[2]=mod->posz+vhapo/2. ;yp[2]=4.2*pv2;
          xp[3]=mod->posz-vhapo/2.;yp[3]=4.2*pv2;
    }
  }
  if(isRingStereo(key))
        {
	  np = 3;
	  if(mod->idModule>100 ){for(int j=0;j<3;j++){
	      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
	    }
	  }else {
	    xd[0]=xdpixel(xp[2]);yd[0]=ydpixel(yp[2]);
	    xd[1]=xdpixel(xp[3]);yd[1]=ydpixel(yp[3]);
	    xd[2]=xdpixel(xp[0]);yd[2]=ydpixel(yp[0]);
	  }
        } else {
    for(int j=0;j<4;j++){
      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
    }
  }
  char buffer [20];
   sprintf(buffer,"%X",mod->idex);

 if(mod->red < 0){ //use count to compute color
 int color = getcolor(mod->value,palette);
     red=(color>>16)&0xFF;
     green=(color>>8)&0xFF;
     blue=(color)&0xFF;
  
if(!print_total)mod->value=mod->value*mod->count;//restore mod->value
  
  if(mod->count > 0)
    if(temporary_file) *svgfile << red << " " << green << " " << blue << " "; else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" lv=\""<<mod->psuIdex<<"\" hv=\""<<mod->psuIdex*10 + mod->HVchannel<<"\" fec=\""<<mod->CcuId<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
  else
    if(temporary_file) *svgfile << 255 << " " << 255 << " " << 255 << " "; else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" lv=\""<<mod->psuIdex<<"\" hv=\""<<mod->psuIdex*10 + mod->HVchannel<<"\" fec=\""<<mod->CcuId<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"white\" points=\"";
  if(temporary_file) *svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; else
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << std::endl; else *svgfile <<"\" />" <<std::endl;
 } else {//color defined with fillc
  if(mod->red>255)mod->red=255;
  if(mod->green>255)mod->green=255;
  if(mod->blue>255)mod->blue=255;
    if(temporary_file) *svgfile << mod->red << " " << mod->green << " " << mod->blue << " "; else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" lv=\""<<mod->psuIdex<<"\" hv=\""<<mod->psuIdex*10 + mod->HVchannel<<"\" fec=\""<<mod->CcuId<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"rgb("<<mod->red<<","<<mod->green<<","<<mod->blue<<")\" points=\"";
  if(temporary_file) *svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; else
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << std::endl; else *svgfile <<"\" />" <<std::endl;
 }
  
}
void TrackerMap::setRange(float min,float max){gminvalue=min;gmaxvalue=max;
if(tkMapLog) {gminvalue=pow(10.,min);gmaxvalue=pow(10.,max);}
}

std::pair<float,float> TrackerMap::getAutomaticRange(){
  float minval,maxval;
  minval=9999999.;
  maxval=-9999999.;
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()  && mod->count>0){
          if (minval > mod->value)minval=mod->value;
          if (maxval < mod->value)maxval=mod->value;
        }
      }
    }
  }
if(tkMapLog) {minval=log(minval)/log(10);maxval=log(maxval)/log(10);}
 return std::make_pair(minval,maxval);

}

//export  tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::save(bool print_total,float minval, float maxval,std::string s,int width, int height){

  printflag=true;
  bool rangefound = true; 
  if(saveGeoTrackerMap){ 
  std::string filetype=s,outputfilename=s;
  std::vector<TPolyLine*> vp;
  TGaxis *axis = 0 ;
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found);
  //outputfilename.erase(outputfilename.begin()+outputfilename.find("."),outputfilename.end());
  temporary_file=true;
  if(filetype=="svg")temporary_file=false;
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  if(!print_total){
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()){
	    mod->value = mod->value / mod->count;
	  }
	}
      }
    }
  }
  
  if(minvalue>=maxvalue){
    minvalue=9999999.;
    maxvalue=-9999999.;
    rangefound=false;
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()  && mod->count>0){
	    rangefound=true;
	    if (minvalue > mod->value)minvalue=mod->value;
	    if (maxvalue < mod->value)maxvalue=mod->value;
	  }
	}
      }
    }
  }
  if ((title==" Tracker Map from  QTestAlarm") || (maxvalue == minvalue)||!rangefound) printflag = false;
  if(!temporary_file){
    *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<std::endl;
    *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<std::endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<std::endl;
    *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3100 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<std::endl;
    *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3100\" height=\"1600\" /> "<<std::endl; 
    *savefile << "<svg:g id=\"tracker\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<std::endl;
  }
  for (int layer=1; layer < 44; layer++){
    //    nlay=layer;
    defwindow(layer);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,savefile);
        }
      }
    }
  }
  
  if(!temporary_file){
    *savefile << "</svg:g>"<<std::endl;
    *savefile << " <svg:text id=\"Title\" class=\"normalText\"  x=\"300\" y=\"0\">"<<title<<"</svg:text>"<<std::endl;
  }
  
  if(printflag)drawPalette(savefile);
  if(!temporary_file){
    *savefile << "</svg:svg>"<<std::endl;
    *savefile << "</svg>"<<std::endl;
  }
  savefile->close(); delete savefile;
  
  const char * command1;
  std::string tempfilename = outputfilename + ".coor";
  if(filetype=="svg"){
    std::string command = "mv "+tempfilename +" " +outputfilename + ".svg";
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
  }
  

  if (temporary_file){ // create root trackermap image
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),std::ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);
    
    if(addPixelFlag)gPad->Range(0,0,3800,1600);else gPad->Range(800,0,3800,1600);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col, *c;
    std::cout<<"tempfilename "<<tempfilename<<std::endl;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ colorList[colindex]=ncolor+100; col =gROOT->GetColor(ncolor+100);
	if(col) col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); else c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));vc.push_back(c); ncolor++;}
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    
    tempfile.clear();
    tempfile.seekg(0,std::ios::beg);
    std::cout << "created palette with " << ncolor << " colors" << std::endl;
    
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
        TPolyLine*  pline = new TPolyLine(npoints,y,x);
        vp.push_back(pline);
        pline->SetFillColor(colorList[colindex]);
        pline->SetLineWidth(0);
        pline->Draw("f");
      }
    }
    if (printflag) {
      float lminvalue=minvalue; float lmaxvalue=maxvalue;
      if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10);}
      axis = new TGaxis(3660,36,3660,1530,lminvalue,lmaxvalue,510,"+L");
      axis->SetLabelSize(0.02);
      axis->Draw();
    }
    TLatex l;
    l.SetTextSize(0.03);
    l.DrawLatex(950,1330,"TID");
    l.DrawLatex(2300,1330,"TEC");
    l.DrawLatex(300,1330,"FPIX");
    l.DrawLatex(20,560,"BPIX L1");
    l.DrawLatex(500,385,"BPIX L2");
    l.DrawLatex(500,945,"BPIX L3");
    l.SetTextSize(0.04);
    std::string fulltitle = title;
    if(tkMapLog && (fulltitle.find("Log10 scale") == std::string::npos)) fulltitle += ": Log10 scale";
    l.DrawLatex(850,1500,fulltitle.c_str());
    l.DrawLatex(1730,40,"-z");
    l.DrawLatex(1730,1360,"+z");
    l.DrawLatex(1085,330,"TIB L1");
    l.DrawLatex(1085,1000,"TIB L2");
    l.DrawLatex(1585,330,"TIB L3");
    l.DrawLatex(1585,1000,"TIB L4");
    l.DrawLatex(2085,330,"TOB L1");
    l.DrawLatex(2085,1000,"TOB L2");
    l.DrawLatex(2585,330,"TOB L3");
    l.DrawLatex(2585,1000,"TOB L4");
    l.DrawLatex(3085,330,"TOB L5");
    l.DrawLatex(3085,1000,"TOB L6");
    TArrow arx(3448,1190,3448,1350,0.01,"|>");
    l.DrawLatex(3460,1350,"x");
    TArrow ary(3448,1190,3312,1190,0.01,"|>");
    l.DrawLatex(3312,1210,"y");
    TArrow arz(3485,373,3485,676,0.01,"|>");
    l.DrawLatex(3510,667,"z");
    TArrow arphi(3485,511,3037,511,0.01,"|>");
    l.DrawLatex(3023,520,"#Phi");
    arx.SetLineWidth(3);
    ary.SetLineWidth(3);
    arz.SetLineWidth(3);
    arphi.SetLineWidth(3);
    arx.Draw();
    ary.Draw();
    arz.Draw();
    arphi.Draw();
    MyC->Update();
    if(filetype=="png"){
      
      std::string filename = outputfilename + ".png";
      std::cout << "printing " <<filename<< std::endl;
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      std::string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      std::string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    std::string command = "rm "+tempfilename ;
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
    MyC->Clear();
    delete MyC;
    if (printflag)delete axis;
    for(std::vector<TPolyLine*>::iterator pos1=vp.begin();pos1!=vp.end();pos1++){
      delete (*pos1);}
    
  }
  }
  return;
}
void TrackerMap::drawApvPair(int crate, int numfed_incrate, bool print_total, TmApvPair* apvPair,std::ofstream * svgfile,bool useApvPairValue)
{
  double xp[4],yp[4];
  int color;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=0., boxinity=0.; 
  double dx=.9,dy=.9;
  /*
  int numfedch_incolumn = 12;
  int numfedch_inrow = 8;
  int numfed_incolumn = 6;
  int numfed_inrow = 4;
  */
  boxinitx=boxinitx+(NUMFED_INCOLUMN-1-(numfed_incrate-1)/NUMFED_INROW)*(NUMFEDCH_INCOLUMN+2);
  boxinity=boxinity+(NUMFED_INROW-1-(numfed_incrate-1)%NUMFED_INROW)*(NUMFEDCH_INROW+1);
  boxinity=boxinity+NUMFEDCH_INROW-(apvPair->getFedCh()/NUMFEDCH_INCOLUMN);
  boxinitx = boxinitx+NUMFEDCH_INCOLUMN-(int)(apvPair->getFedCh()%NUMFEDCH_INCOLUMN);
  //  std::cout << crate << " " << numfed_incrate << " " << apvPair->getFedCh()<<" "<<boxinitx<< " " << boxinity << std::endl; ;
  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
  for(int j=0;j<4;j++){
    xd[j]=xdpixelc(xp[j]);yd[j]=ydpixelc(yp[j]);
    //std::cout << boxinity << " "<< ymax << " "<< yp[j] << std::endl;
  }
  
  char buffer [20];
  sprintf(buffer,"%X",apvPair->mod->idex);
  std::string s = apvPair->mod->name;
  s.erase(s.begin()+s.find("connected"),s.end());

  if(useApvPairValue){ 
    if(apvPair->red < 0){ //use count to compute color
      if(apvPair->count > 0) {
	color = getcolor(apvPair->value,palette);
	red=(color>>16)&0xFF;
	green=(color>>8)&0xFF;
	blue=(color)&0xFF;
	if(!print_total)apvPair->value=apvPair->value*apvPair->count;//restore mod->value
	if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
           else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
      } else {
        if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\"  cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
      }
    } else {//color defined with fillc
      if(apvPair->red>255)apvPair->red=255;
      if(apvPair->green>255)apvPair->green=255;
      if(apvPair->blue>255)apvPair->blue=255;
      if(temporary_file)*svgfile << apvPair->red << " " << apvPair->green << " " << apvPair->blue << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<apvPair->red<<","<<apvPair->green<<","<<apvPair->blue<<")\" points=\"";
    }
  }else{
    if(apvPair->mod->red < 0){ //use count to compute color
      if(apvPair->mod->count > 0) {
	color = getcolor(apvPair->mod->value,palette);
	red=(color>>16)&0xFF;
	green=(color>>8)&0xFF;
	blue=(color)&0xFF;
	if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
           else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
      } else {
        if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\"  cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
      }
    } else {//color defined with fillc
      if(apvPair->mod->red>255)apvPair->mod->red=255;
      if(apvPair->mod->green>255)apvPair->mod->green=255;
      if(apvPair->mod->blue>255)apvPair->mod->blue=255;
      if(temporary_file)*svgfile << apvPair->mod->red << " " << apvPair->mod->green << " " << apvPair->mod->blue << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<apvPair->mod->red<<","<<apvPair->mod->green<<","<<apvPair->mod->blue<<")\" points=\"";
    }
  }
  if(temporary_file)*svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; 
      else *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << std::endl;
     else *svgfile <<"\" />" <<std::endl;
}
void TrackerMap::drawCcu(int crate, int numfec_incrate, bool print_total, TmCcu* ccu,std::ofstream * svgfile,bool useCcuValue)
{
  double xp[4],yp[4];
  int color;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=0., boxinity=0.;
  double dx=.9,dy=.9;
  int numccu_incolumn = 8;
  int numccu_inrow = 15;
  int numfec_incolumn = 5;
  int numfec_inrow = 4;
  boxinitx=boxinitx+(numfec_incolumn-(numfec_incrate-1)/numfec_inrow)*14.;
  boxinity=boxinity+(numfec_inrow-(numfec_incrate-1)%numfec_inrow)*16.;
  boxinity=boxinity+numccu_inrow-ccu->mpos;
  boxinitx = boxinitx+numccu_incolumn-(int)(ccu->getCcuRing()%numccu_incolumn);
  //std::cout << crate << " " << numfec_incrate << " " << ccu->getCcuRing()<<" "<<ccu->mpos<<" "<<boxinitx<< " " << boxinity << std::endl; ;
  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
  for(int j=0;j<4;j++){
    xd[j]=xdpixelfec(xp[j]);yd[j]=ydpixelfec(yp[j]);
    //std::cout << boxinity << " "<< ymax << " "<< yp[j] << std::endl;
  }

  char buffer [20];
  sprintf(buffer,"%X",ccu->idex);
  //sprintf(buffer,"%X",ccu->mod->idex);
  //std::string s = ccu->mod->name;
  std::string s = ccu->text;
  s.erase(s.begin()+s.find("connected"),s.end());

  if(ccu->red < 0){ //use count to compute color
    if(ccu->count > 0) {
      color = getcolor(ccu->value,palette);
      red=(color>>16)&0xFF;
      green=(color>>8)&0xFF;
      blue=(color)&0xFF;
      if(!print_total)ccu->value=ccu->value*ccu->count;//restore mod->value
      if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
      else *svgfile <<"<svg:polygon detid=\""<<ccu->idex<<"\" count=\""<<ccu->count <<"\" value=\""<<ccu->value<<"\" id=\""<<ccu->idex+crate*1000000<<"\" cmodid=\""<<ccu->cmodid<<"\" layer=\""<<ccu->layer<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Slot/Ring"<<ccu->getCcuSlot()<<"/"<<ccu->getCcuRing()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
    } else {
      if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<ccu->idex<<"\" count=\""<<ccu->count <<"\" value=\""<<ccu->value<<"\" id=\""<<ccu->idex+crate*1000000<<"\"  cmodid=\""<<ccu->cmodid<<"\" layer=\""<<ccu->layer<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Slot/Ring "<<ccu->getCcuSlot()<<"/"<<ccu->getCcuRing()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
    }
  } else {//color defined with fillc
    if(ccu->red>255)ccu->red=255;
    if(ccu->green>255)ccu->green=255;
    if(ccu->blue>255)ccu->blue=255;
    if(temporary_file)*svgfile << ccu->red << " " << ccu->green << " " << ccu->blue << " ";
    else *svgfile <<"<svg:polygon detid=\""<<ccu->idex<<"\" count=\""<<ccu->count <<"\" value=\""<<ccu->value<<"\" id=\""<<ccu->idex+crate*1000000<<"\" cmodid=\""<<ccu->cmodid<<"\" layer=\""<<ccu->layer<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Slot/Ring "<<ccu->getCcuSlot()<<"/"<<ccu->getCcuRing()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<ccu->red<<","<<ccu->green<<","<<ccu->blue<<")\" points=\"";
  }
if(temporary_file)*svgfile << np << " ";
for(int k=0;k<np;k++){
  if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ;
  else *svgfile << xd[k] << "," << yd[k] << " " ;
}
if(temporary_file)*svgfile << std::endl;
else *svgfile <<"\" />" <<std::endl;

}
void TrackerMap::drawPsu(int rack,int numcrate_inrack , bool print_total, TmPsu* psu,std::ofstream * svgfile,bool usePsuValue)
{
  double xp[4],yp[4];
  int color;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=0., boxinity=0.; 
  double dx=.9,dy=.9;

  boxinitx=boxinitx+(NUMPSUCRATE_INCOLUMN-psu->getPsuCrate())*1.5;
  boxinity=boxinity+(NUMPSUCH_INROW-psu->getPsuBoard());

  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
 
 
  for(int j=0;j<4;j++){
    xd[j]=xdpixelpsu(xp[j]);yd[j]=ydpixelpsu(yp[j]);
    //std::cout << boxinity << " "<< ymax << " "<< yp[j] << std::endl;
  }

  // lines needed to prepare the clickable maps: understand why I get twice the full list of channels (HV1 and HV0?)
  /*
  double scalex=2695./2700.;
  double scaley=1520./1550.;
  std::cout << "<area shape=\"rect\" coords=\" " 
	    << int(scalex*yd[2]) << "," << int(1520-scaley*xd[2]) 
	    << "," << int(scalex*yd[0]) << "," << int(1520-scaley*xd[0]) 
	    << "\" title=\" " << psu->psId << "\" /> " << std::endl;
  */
  //
  
  char buffer [20];
  sprintf(buffer,"%X",psu->idex);
  std::string s = psu->text;
  s.erase(s.begin()+s.find("connected"),s.end());
   
  if(psu->red < 0){ //use count to compute color
    if(psu->count > 0){
      color = getcolor(psu->value,palette);
      red=(color>>16)&0xFF;
      green=(color>>8)&0xFF;
      blue=(color)&0xFF;
      if(!print_total)psu->value=psu->value*psu->count;//restore mod->value
      if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->count <<"\" value=\""<<psu->value<<"\" id=\""<< psu->idex <<"\" cmodid=\""<<psu->cmodid_LV<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
      } 
      else{
     
      if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->count <<"\" value=\""<<psu->value<<"\" id=\""<< psu->idex <<"\"  cmodid=\""<<psu->cmodid_LV<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"white\" points=\"";
      }
    } 
   
    else {//color defined with fillc
    if(psu->red>255)psu->red=255;
    if(psu->green>255)psu->green=255;
    if(psu->blue>255)psu->blue=255;
    if(temporary_file)*svgfile << psu->red << " " << psu->green << " " << psu->blue << " ";
    else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->count <<"\" value=\""<<psu->value<<"\" id=\""<< psu->idex <<"\" cmodid=\""<<psu->cmodid_LV<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<psu->red<<","<<psu->green<<","<<psu->blue<<")\" points=\"";
  }

if(temporary_file)*svgfile << np << " ";
for(int k=0;k<np;k++){
  if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; 
  else *svgfile << xd[k] << "," << yd[k] << " " ;
}
if(temporary_file)*svgfile << std::endl;
else *svgfile <<"\" />" <<std::endl;

}

void TrackerMap::drawHV2(int rack,int numcrate_inrack , bool print_total, TmPsu* psu,std::ofstream * svgfile,bool usePsuValue)
{
  double xp[4],yp[4];
  int color;
  int greenHV2 = 0;
  int redHV2 = 0;
  int blueHV2 = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=35, boxinity=12; 
  double dx=1.1,dy=1.3;
  
  boxinitx= boxinitx+(5 - psu->getPsuCrate())*5;
  boxinity= boxinity+(18 - psu->getPsuBoard())*1.75;

  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
 
 
  for(int j=0;j<4;j++){
    xd[j]=xdpixelpsu(xp[j]);yd[j]=ydpixelpsu(yp[j]);
    //std::cout << boxinity << " "<< ymax << " "<< yp[j] << std::endl;
  }
  
  char buffer [20];
  sprintf(buffer,"%X",psu->idex);
  std::string s = psu->textHV2;
  s.erase(s.begin()+s.find("connected"),s.end());
   
  if(psu->redHV2 < 0){ //use count to compute color
    
    if(psu->valueHV2 > 0){
      color = getcolor(psu->valueHV2,palette);
      redHV2=(color>>16)&0xFF;
      greenHV2=(color>>8)&0xFF;
      blueHV2=(color)&0xFF;
      if(!print_total)psu->valueHV2=psu->valueHV2*psu->countHV2;//restore mod->value
      if(temporary_file)*svgfile << redHV2 << " " << greenHV2 << " " << blueHV2 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV2 <<"\" value=\""<<psu->valueHV2<<"\" id=\""<< psu->idex*10+2 <<"\" cmodid=\""<<psu->cmodid_HV2<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<redHV2<<","<<greenHV2<<","<<blueHV2<<")\" points=\"";
      } 
      else{
      if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV2 <<"\" value=\""<<psu->valueHV2<<"\" id=\""<< psu->idex*10+2 <<"\"  cmodid=\""<<psu->cmodid_HV2<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"white\" points=\"";
      }
    } 
   
    else {//color defined with fillc
    if(psu->redHV2>255)psu->redHV2=255;
    if(psu->greenHV2>255)psu->greenHV2=255;
    if(psu->blueHV2>255)psu->blueHV2=255;
    if(temporary_file)*svgfile << psu->redHV2 << " " << psu->greenHV2 << " " << psu->blueHV2 << " ";
    else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV2 <<"\" value=\""<<psu->valueHV2<<"\" id=\""<< psu->idex*10+2 <<"\" cmodid=\""<<psu->cmodid_HV2<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<psu->redHV2<<","<<psu->greenHV2<<","<<psu->blueHV2<<")\" points=\"";
  }

if(temporary_file)*svgfile << np << " ";
for(int k=0;k<np;k++){
  if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; 
  else *svgfile << xd[k] << "," << yd[k] << " " ;
}
if(temporary_file)*svgfile << std::endl;
else *svgfile <<"\" />" <<std::endl;

}


void TrackerMap::drawHV3(int rack,int numcrate_inrack , bool print_total, TmPsu* psu,std::ofstream * svgfile,bool usePsuValue)
{
  double xp[4],yp[4];
  int color;
  int greenHV3 = 0;
  int redHV3 = 0;
  int blueHV3 = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=36.5, boxinity=12; 
  double dx=1.1,dy=1.3;
  
  boxinitx= boxinitx+(5 - psu->getPsuCrate())*5;
  boxinity= boxinity+(18 - psu->getPsuBoard())*1.75;

  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
 
 
  for(int j=0;j<4;j++){
    xd[j]=xdpixelpsu(xp[j]);yd[j]=ydpixelpsu(yp[j]);
    //std::cout << boxinity << " "<< ymax << " "<< yp[j] << std::endl;
  }
  
  char buffer [20];
  sprintf(buffer,"%X",psu->idex);
  std::string s = psu->textHV3;
  s.erase(s.begin()+s.find("connected"),s.end());
   
  if(psu->redHV3 < 0){ //use count to compute color
    if(psu->valueHV3 > 0){
      color = getcolor(psu->valueHV3,palette);
      redHV3=(color>>16)&0xFF;
      greenHV3=(color>>8)&0xFF;
      blueHV3=(color)&0xFF;
      if(!print_total)psu->valueHV3=psu->valueHV3*psu->countHV3;//restore mod->value
      if(temporary_file)*svgfile << redHV3 << " " << greenHV3 << " " << blueHV3 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV3 <<"\" value=\""<<psu->valueHV3<<"\" id=\""<< psu->idex*10+3 <<"\" cmodid=\""<<psu->cmodid_HV3<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard"<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<redHV3<<","<<greenHV3<<","<<blueHV3<<")\" points=\"";
      } 
      else{
      if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
      else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV3 <<"\" value=\""<<psu->valueHV3<<"\" id=\""<< psu->idex*10+3 <<"\"  cmodid=\""<<psu->cmodid_HV3<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"white\" points=\"";
      }
    } 
   
    else {//color defined with fillc
    if(psu->redHV3>255)psu->redHV3=255;
    if(psu->greenHV3>255)psu->greenHV3=255;
    if(psu->blueHV3>255)psu->blueHV3=255;
    if(temporary_file)*svgfile << psu->redHV3 << " " << psu->greenHV3 << " " << psu->blueHV3 << " ";
    else *svgfile <<"<svg:polygon detid=\""<<psu->idex<<"\" count=\""<<psu->countHV3 <<"\" value=\""<<psu->valueHV3<<"\" id=\""<< psu->idex*10+3 <<"\" cmodid=\""<<psu->cmodid_HV3<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"easyCrate/easyBoard "<<psu->getPsuCrate()<<"/"<<psu->getPsuBoard()<<" connected to "<<s<<" \" fill=\"rgb("<<psu->redHV3<<","<<psu->greenHV3<<","<<psu->blueHV3<<")\" points=\"";
  }

if(temporary_file)*svgfile << np << " ";
for(int k=0;k<np;k++){
  if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; 
  else *svgfile << xd[k] << "," << yd[k] << " " ;
}
if(temporary_file)*svgfile << std::endl;
else *svgfile <<"\" />" <<std::endl;

}

void TrackerMap::save_as_fectrackermap(bool print_total,float minval, float maxval,std::string s,int width, int height){

 if(enableFecProcessing){
  std::string filetype=s,outputfilename=s;
  std::vector<TPolyLine*> vp;
  TGaxis *axis = 0 ;
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found); 
  temporary_file=true;
  if(filetype=="xml"||filetype=="svg")temporary_file=false;
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  if(temporary_file)savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
   std::map<int , TmCcu *>::iterator i_ccu;
   std::multimap<TmCcu*, TmModule*>::iterator it;
   std::pair<std::multimap<TmCcu*, TmModule*>::iterator,std::multimap<TmCcu*, TmModule*>::iterator> ret;
  //Decide if we must use Module or Ccu value
  bool useCcuValue=false;
  
  
  for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
    TmCcu *  ccu= i_ccu->second;
    if(ccu!=0) {
        if(ccu->count > 0 || ccu->red!=-1) { useCcuValue=true; break;}
      }
    }
 
  
  if(!useCcuValue)//store mean of connected modules value{
    for(  i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
    TmCcu *  ccu= i_ccu->second;
      if(ccu!=0) {
            ret = fecModuleMap.equal_range(ccu);
        for (it = ret.first; it != ret.second; ++it)
          {
           if( (*it).second->count>0){ccu->value=ccu->value+(*it).second->value;ccu->count++;}
          }
          if(ccu->count>0)ccu->value=ccu->value/ccu->count;
          if(ccu->nmod==0)  { ccu->red=0;ccu->green=0;ccu->blue=0;}
          }
          }
  
 
  if(title==" Tracker Map from  QTestAlarm"){
      for(  i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
          TmCcu *  ccu= i_ccu->second;
          if(ccu!=0) {
	    ret = fecModuleMap.equal_range(ccu);
	    ccu->red=0;ccu->green=255;ccu->blue=0;
	    for (it = ret.first; it != ret.second; ++it) {
              if( !( (*it).second->red==0 && (*it).second->green==255 && (*it).second->blue==0 ) && !( (*it).second->red==255 && (*it).second->green==255 && (*it).second->blue==255 ) ){
		ccu->red=255;ccu->green=0;ccu->blue=0;
		}
	      }
	   }
      }
   }
  

  
  if(!print_total){
    for(  i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
    TmCcu *  ccu= i_ccu->second;
      if(ccu!=0) {
          if(useCcuValue) ccu->value = ccu->value / ccu->count;

        }
    }
  }
 
  if(minvalue>=maxvalue){

    minvalue=9999999.;
    maxvalue=-9999999.;
    for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
       TmCcu *  ccu= i_ccu->second;
       if(ccu!=0 && ccu->count>0) {
              if (minvalue > ccu->value)minvalue=ccu->value;
              if (maxvalue < ccu->value)maxvalue=ccu->value;
        }
    }
  }


  
 if(filetype=="svg"){
      saveAsSingleLayer=false;
      std::ostringstream outs;
    outs << outputfilename<<".svg";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<std::endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<std::endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<std::endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<std::endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<std::endl;
  *savefile << "<svg:g id=\"fedtrackermap\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<std::endl;
     }
  for (int crate=1; crate < (nfeccrates+1); crate++){
    if(filetype=="xml"){
      saveAsSingleLayer=true;
      std::ostringstream outs;
    outs << outputfilename<<"feccrate" <<crate<< ".xml";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *savefile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<std::endl;
    *savefile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<std::endl;
    *savefile << "<script type=\"text/ecmascript\" xlink:href=\"feccrate.js\" />"<<std::endl;
    *savefile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerCrate.init()\">"<<std::endl;
    *savefile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<std::endl;
    *savefile << "<g id=\"crate\" transform=\" translate(280,580) rotate(270) scale(.7,.8)\"  > "<<std::endl;
         }
    //    ncrate=crate;
    deffecwindow(crate);
 
    for ( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
      TmCcu *  ccu= i_ccu->second;
      if(ccu!=0){
        if(ccu->getCcuCrate() == crate){
              
	      drawCcu(crate,ccu->getCcuSlot()-2,print_total,ccu,savefile,useCcuValue);
        }
      }
    }
 
   if(!temporary_file){
    if(filetype=="xml"){
    *savefile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> " << std::endl;
    *savefile << "<tspan id=\"line1\" x=\"40\" y=\"30\"> </tspan> " << std::endl;
    *savefile << "<tspan id=\"line2\" x=\"40\" y=\"60\"> </tspan> " << std::endl;
    *savefile << " </text> </svg>" << std::endl;
    savefile->close();
     saveAsSingleLayer=false;
      }
      }
    }
    if(filetype=="svg"){
    *savefile << "</g> </svg> </svg> " << std::endl;
    savefile->close();
      }
  if(!print_total && !useCcuValue){
//Restore ccu value
    for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
       TmCcu *  ccu= i_ccu->second;
       if(ccu!=0) {
          ccu->value = ccu->value * ccu->count;
    }
}
}
 if(temporary_file){
 if(printflag&&!saveWebInterface)drawPalette(savefile);
  savefile->close();

  const char * command1;
  std::string tempfilename = outputfilename + ".coor";
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),std::ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);

    if(saveWebInterface)gPad->Range(0,0,3700,1600); else gPad->Range(0,0,3800,1600);

    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col,*c;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints;
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex);
      if(pos == colorList.end()){
        colorList[colindex]=ncolor+100;
        col =gROOT->GetColor(ncolor+100);
        if(col)
          col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
        else
          c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
          vc.push_back(c);
	ncolor++;
      }
      for (int i=0;i<npoints;i++){
        tempfile >> x[i] >> y[i];
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,std::ios::beg);
    std::cout << "created palette with " << ncolor << " colors" << std::endl;
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints;
      for (int i=0;i<npoints;i++){
        tempfile >> x[i] >> y[i];
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex);
      if(pos != colorList.end()){
        TPolyLine*  pline = new TPolyLine(npoints,y,x);
        vp.push_back(pline);
        pline->SetFillColor(colorList[colindex]);
        pline->SetLineWidth(0);
        pline->Draw("f");
      }
    }
        if (printflag&&!saveWebInterface) {
      float lminvalue=minvalue; float lmaxvalue=maxvalue;
      if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10);}
      axis = new TGaxis(3660,36,3660,1530,lminvalue,lmaxvalue,510,"+L");
      axis->SetLabelSize(0.02);
      axis->Draw();
    }

    if(!saveWebInterface){
    TLatex l;
    l.SetTextSize(0.05);
    std::string fulltitle = title;
    if(tkMapLog && (fulltitle.find("Log10 scale") == std::string::npos)) fulltitle += ": Log10 scale";
    l.DrawLatex(50,1530,fulltitle.c_str());
       }
    MyC->Update();
    std::cout << "Filetype " << filetype << std::endl;
    if(filetype=="png"){
      std::string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      std::string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      std::string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    std::string command = "rm "+tempfilename ;
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
    MyC->Clear();
    delete MyC;
    if (printflag&&!saveWebInterface)delete axis;
    for(std::vector<TPolyLine*>::iterator pos1=vp.begin();pos1!=vp.end();pos1++){
         delete (*pos1);}
   



}//if(temporary_file)
}//if(enabledFecProcessing)
}
void TrackerMap::save_as_HVtrackermap(bool print_total,float minval, float maxval,std::string s,int width, int height){
  
 if(enableHVProcessing){
  std::string filetype=s,outputfilename=s;
  std::vector<TPolyLine*> vp;
  TGaxis *axis = 0 ;
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found);

  temporary_file=true;
  

  if(filetype=="xml"||filetype=="svg")temporary_file=false;
  
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  if(temporary_file)savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
 
  
  bool usePsuValue=false;
  
  for( ipsu=psuMap.begin();ipsu!=psuMap.end(); ipsu++){
    TmPsu*  psu= ipsu->second;
    if(psu!=0) {
      if(psu->countHV2 > 0 || psu->redHV2!=-1 || psu->countHV3 > 0 || psu->redHV3!=-1) { usePsuValue=true; break;}
      }
    }
   
  if(!usePsuValue){//store mean of connected modules value{
    
    for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
       TmPsu *  psu= ipsu->second;
       if(psu!=0) {
       ret = psuModuleMap.equal_range(psu);
         int nconn1=0;int nconn2=0;
         for(it = ret.first; it != ret.second; ++it){
           if((*it).second->HVchannel==2&&(*it).second->count>0){ nconn1++;psu->valueHV2=psu->valueHV2+(*it).second->value;}
           if((*it).second->HVchannel==3&&(*it).second->count>0){ nconn2++;psu->valueHV3=psu->valueHV3+(*it).second->value;} 
	    }
         if(psu->nmodHV2!=0 &&nconn1>0){psu->valueHV2=psu->valueHV2/psu->nmodHV2; psu->countHV2=1; }
         if(psu->nmodHV3!=0 &&nconn2>0){psu->valueHV3=psu->valueHV3/psu->nmodHV3; psu->countHV3=1; }
	  
	   }
	 
	 }
       }
   
   if(title==" Tracker Map from  QTestAlarm"){
      for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
          TmPsu *  psu= ipsu->second;
          if(psu!=0) {
	    ret = psuModuleMap.equal_range(psu);
	    psu->redHV2=0;psu->greenHV2=255;psu->blueHV2=0;
	    psu->redHV3=0;psu->greenHV3=255;psu->blueHV3=0;
	    for (it = ret.first; it != ret.second; ++it) {
              if((*it).second->HVchannel==2){
	        if( !( (*it).second->red==0 && (*it).second->green==255 && (*it).second->blue==0 ) && !( (*it).second->red==255 && (*it).second->green==255 && (*it).second->blue==255 ) ){
		   psu->redHV2=255;psu->greenHV2=0;psu->blueHV2=0;
		   }
		}
	      if((*it).second->HVchannel==3){
	        if( !( (*it).second->red==0 && (*it).second->green==255 && (*it).second->blue==0 ) && !( (*it).second->red==255 && (*it).second->green==255 && (*it).second->blue==255 ) ){
		   psu->redHV3=255;psu->greenHV3=0;psu->blueHV3=0;
		   }
		}
	     }
	   }
      }
   } 
 
  if(!print_total){
    for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
    TmPsu *  psu= ipsu->second;
      if(psu!=0) {
	  if(usePsuValue){ 
	    psu->valueHV2 = psu->valueHV2 / psu->countHV2;
            psu->valueHV3 = psu->valueHV3 / psu->countHV3;
	}
    }
  }
 } 
  
  if(minvalue>=maxvalue){
    
    minvalue=9999999.;
    maxvalue=-9999999.;
    
    for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
       TmPsu *  psu= ipsu->second;
       if(psu!=0 && psu->countHV2>0 && psu->countHV3 >0) {

	      if (minvalue > psu->valueHV2 || minvalue > psu->valueHV3)minvalue=std::min(psu->valueHV2,psu->valueHV3);
	      if (maxvalue < psu->valueHV2 || maxvalue < psu->valueHV3)maxvalue=std::max(psu->valueHV2,psu->valueHV3);
	      
	
	}
    }
  }
  
     if(filetype=="svg"){
      saveAsSingleLayer=false;
      std::ostringstream outs;
    outs << outputfilename<<".svg";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<std::endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<std::endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<std::endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<std::endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<std::endl; 
  *savefile << "<svg:g id=\"HVtrackermap\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<std::endl;
     }
   
  for (int irack=1; irack < (npsuracks+1); irack++){
    if(filetype=="xml"){
      saveAsSingleLayer=true;
      std::ostringstream outs;
    outs << outputfilename<<"HVrack" <<irack<< ".xml";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *savefile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<std::endl;
    *savefile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<std::endl;
    *savefile << "<script type=\"text/ecmascript\" xlink:href=\"rackhv.js\" />"<<std::endl;
    *savefile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerRackhv.init()\">"<<std::endl;
    *savefile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<std::endl;
    *savefile << "<g id=\"rackhv\" transform=\" translate(150,500) rotate(270) scale(1.,1.)\"  > "<<std::endl;
         }
    
    // nrack=irack;
    defpsuwindow(irack);
    for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu->getPsuRack() == irack){
	      drawHV2(irack,psu->getPsuCrate(),print_total,psu,savefile,usePsuValue);
              drawHV3(irack,psu->getPsuCrate(),print_total,psu,savefile,usePsuValue);
      }
    }
   
  
   if(!temporary_file){
    if(filetype=="xml"){
    *savefile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> " << std::endl;
    *savefile << "<tspan id=\"line1\" x=\"40\" y=\"30\"> </tspan> " << std::endl;
    *savefile << "<tspan id=\"line2\" x=\"40\" y=\"60\"> </tspan> " << std::endl;
    *savefile << " </text> </svg>" << std::endl;
    savefile->close();
     saveAsSingleLayer=false;
      }
      }
    }
    if(filetype=="svg"){
    *savefile << "</g> </svg> </svg> " << std::endl;
    savefile->close();
      }
 
    //Restore psu value
    if(!print_total && !usePsuValue){
     for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
       TmPsu *psu = ipsu->second;
       if(psu!=0) {
	  psu->valueHV2 = psu->valueHV2 * psu->countHV2;
          psu->valueHV3 = psu->valueHV3 * psu->countHV3;
	  }
       }
     }
  

  if(temporary_file){
  if(printflag&&!saveWebInterface)drawPalette(savefile);
    savefile->close(); 

  const char * command1;
  std::string tempfilename = outputfilename + ".coor";
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),std::ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);
    
    if(saveWebInterface)gPad->Range(0,0,3700,1600); else gPad->Range(0,0,3800,1600);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col,*c;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ 
	colorList[colindex]=ncolor+100; 
	col =gROOT->GetColor(ncolor+100);
	if(col) 
	  col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); 
	else 
	  c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
	  vc.push_back(c);
	ncolor++;
      }
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,std::ios::beg);
    std::cout << "created palette with " << ncolor << " colors" << std::endl;
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
        TPolyLine*  pline = new TPolyLine(npoints,y,x);
        vp.push_back(pline);
        pline->SetFillColor(colorList[colindex]);
        pline->SetLineWidth(0);
        pline->Draw("f");
      }
    }
        if (printflag&&!saveWebInterface) {
      float lminvalue=minvalue; float lmaxvalue=maxvalue;
      if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10);}
      axis = new TGaxis(3660,36,3660,1530,lminvalue,lmaxvalue,510,"+L");
      axis->SetLabelSize(0.02);
      axis->Draw();
    }


    if(!saveWebInterface){
    TLatex l;
    l.SetTextSize(0.05);
    std::string fulltitle = title;
    if(tkMapLog && (fulltitle.find("Log10 scale") == std::string::npos)) fulltitle += ": Log10 scale";
    l.DrawLatex(50,1530,fulltitle.c_str());
       }
    MyC->Update();
    std::cout << "Filetype " << filetype << std::endl;
    if(filetype=="png"){
      std::string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      std::string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      std::string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    std::string command = "rm "+tempfilename ;
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
    MyC->Clear();
    delete MyC;
     if (printflag&&!saveWebInterface)delete axis;
    for(std::vector<TPolyLine*>::iterator pos1=vp.begin();pos1!=vp.end();pos1++){
         delete (*pos1);}
    
	 
	 }//if(temporary_file)
}//if(enabledHVProcessing)
}


void TrackerMap::save_as_psutrackermap(bool print_total,float minval, float maxval,std::string s,int width, int height){

 if(enableLVProcessing){
  
  printflag=true;
  bool rangefound=true;
  std::string filetype=s,outputfilename=s;
  std::vector<TPolyLine*> vp;
  TGaxis *axis = 0 ;
  
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found);
  
  temporary_file=true;
  

  
  if(filetype=="xml"||filetype=="svg")temporary_file=false;
   
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  if(temporary_file)savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
 
  //Decide if we must use Module or Power Psupply value
  bool usePsuValue=false;
  
  for( ipsu=psuMap.begin();ipsu!=psuMap.end(); ipsu++){
    TmPsu*  psu= ipsu->second;
    if(psu!=0) {
      if(psu->count > 0 || psu->red!=-1) { usePsuValue=true; break;}
      }
    }
  
  if(!usePsuValue){//store mean of connected modules value{
    for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0) {
	ret = psuModuleMap.equal_range(psu);
	int nconn=0;
	for(it = ret.first; it != ret.second; ++it){
	  if((*it).second->count>0){nconn++;psu->value=psu->value+(*it).second->value;}
          
	}
	if(nconn>0){ psu->value=psu->value/psu->nmod; psu->count=1;}
	
      }
    }
  }
  
  if(title==" Tracker Map from  QTestAlarm"){
    for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0) {
	ret = psuModuleMap.equal_range(psu);
	//	psu->red=255;psu->green=255;psu->blue=255;
	psu->red=-1;
	int nconn=0;
	for (it = ret.first; it != ret.second; ++it) {
	  if( !( (*it).second->red==0 && (*it).second->green==255 && (*it).second->blue==0 ) && !( (*it).second->red==255 && (*it).second->green==255 && (*it).second->blue==255 ) ){
	    nconn++;psu->value++;
	  }
	}
	if(nconn>0){ psu->value=psu->value/psu->nmod; psu->count=1;}
      }
    }
  }
  
  
  
  if(!print_total){
    for(  ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
    TmPsu *  psu= ipsu->second;
      if(psu!=0) {
	  if(usePsuValue) psu->value = psu->value / psu->count;

	}
    }
  }
  
  if(minvalue>=maxvalue){
    
    minvalue=9999999.;
    maxvalue=-9999999.;
    rangefound=false;
    for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
       TmPsu *  psu= ipsu->second;
       if(psu!=0 && psu->count>0) {
	 rangefound = true;
	      if (minvalue > psu->value)minvalue=psu->value;
	      if (maxvalue < psu->value)maxvalue=psu->value;
	}
    }
  }
  if ((maxvalue == minvalue)||!rangefound) printflag = false;

  
     if(filetype=="svg"){
      saveAsSingleLayer=false;
      std::ostringstream outs;
    outs << outputfilename<<".svg";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<std::endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<std::endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<std::endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<std::endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<std::endl; 
  *savefile << "<svg:g id=\"psutrackermap\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<std::endl;
     }
  
  for (int irack=1; irack < (npsuracks+1); irack++){
    if(filetype=="xml"){
      saveAsSingleLayer=true;
      std::ostringstream outs;
    outs << outputfilename<<"psurack" <<irack<< ".xml";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *savefile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<std::endl;
    *savefile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<std::endl;
    *savefile << "<script type=\"text/ecmascript\" xlink:href=\"rack.js\" />"<<std::endl;
    *savefile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerCrate.init()\">"<<std::endl;
    *savefile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<std::endl;
    *savefile << "<g id=\"rack\" transform=\" translate(150,500) rotate(270) scale(1.,1.)\"  > "<<std::endl;
         }
   
    
    // nrack=irack;
    defpsuwindow(irack);
    for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu->getPsuRack() == irack){
	      
	      drawPsu(irack,psu->getPsuCrate(),print_total,psu,savefile,usePsuValue);
      }
    }
   
    
   if(!temporary_file){
    if(filetype=="xml"){
    *savefile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> " << std::endl;
    *savefile << "<tspan id=\"line1\" x=\"40\" y=\"30\"> </tspan> " << std::endl;
    *savefile << "<tspan id=\"line2\" x=\"40\" y=\"60\"> </tspan> " << std::endl;
    *savefile << " </text> </svg>" << std::endl;
    savefile->close();
     saveAsSingleLayer=false;
      }
      }
    }
    if(filetype=="svg"){
    *savefile << "</g> </svg> </svg> " << std::endl;
    savefile->close();
      }
 
    //Restore psu value
    if(!print_total && !usePsuValue){
     for( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
       TmPsu *psu = ipsu->second;
       if(psu!=0) {
	  psu->value = psu->value * psu->count;
          }
       }
     }

    int rangex=YPSUOFFSET+(YPSURSIZE+YPSUOFFSET)*NUMPSURACK_INROW+300; 
    int rangey=XPSUOFFSET+(XPSURSIZE+XPSUOFFSET)*NUMPSURACK_INCOLUMN+300;
  
  
  if(temporary_file){
    if(printflag&&!saveWebInterface)drawPalette(savefile,rangex-140,rangey-100);
    savefile->close(); 

  const char * command1;
  std::string tempfilename = outputfilename + ".coor";
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),std::ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);
    
    //    if(saveWebInterface)gPad->Range(0,0,3700,1600); else gPad->Range(0,0,3800,1600);
    std::cout << " range x " <<  rangex << std::endl;
    std::cout << " range y " <<  rangey << std::endl;
    gPad->Range(0,0,rangex,rangey);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col,*c;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ 
	colorList[colindex]=ncolor+100; 
	col =gROOT->GetColor(ncolor+100);
	if(col) 
	  col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); 
	else 
	  c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
	vc.push_back(c);
	ncolor++;
      }
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,std::ios::beg);
    std::cout << "created palette with " << ncolor << " colors" << std::endl;
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
        TPolyLine*  pline = new TPolyLine(npoints,y,x);
        vp.push_back(pline);
        pline->SetFillColor(colorList[colindex]);
        pline->SetLineWidth(0);
        pline->Draw("f");
      }
    }
        if (printflag&&!saveWebInterface) {
      float lminvalue=minvalue; float lmaxvalue=maxvalue;
      if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10);}
      axis = new TGaxis(rangex-140,34,rangex-140,rangey-106,lminvalue,lmaxvalue,510,"+L");
      axis->SetLabelSize(0.02);
      axis->Draw();
    }

    if(!saveWebInterface){
    TLatex l;
    l.SetTextSize(0.05);
    std::string fulltitle = title;
    if(tkMapLog && (fulltitle.find("Log10 scale") == std::string::npos)) fulltitle += ": Log10 scale";
    l.DrawLatex(50,rangey-200,fulltitle.c_str());
       }
    MyC->Update();
    std::cout << "Filetype " << filetype << std::endl;
    if(filetype=="png"){
      std::string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      std::string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      std::string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    std::string command = "rm "+tempfilename ;
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
    MyC->Clear();
    delete MyC;
     if (printflag&&!saveWebInterface)delete axis;
    for(std::vector<TPolyLine*>::iterator pos1=vp.begin();pos1!=vp.end();pos1++){
         delete (*pos1);}
   
}//if(temporary_file)
}//if(enabledFedProcessing)
}

void TrackerMap::save_as_fedtrackermap(bool print_total,float minval, float maxval,std::string s,int width, int height){
 if(enableFedProcessing){
  printflag=true;
   bool rangefound = true; 
  std::string filetype=s,outputfilename=s;
  std::vector<TPolyLine*> vp;
  TGaxis *axis = 0 ;
  
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found);
  
  temporary_file=true;
  if(filetype=="xml"||filetype=="svg")temporary_file=false;
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  if(temporary_file)savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  std::map<int , TmApvPair *>::iterator i_apv;
  std::map<int , int>::iterator i_fed;
  //Decide if we must use Module or ApvPair value
  bool useApvPairValue=false;
  for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
    TmApvPair *  apvPair= i_apv->second;
    if(apvPair!=0) {
      TmModule * apv_mod = apvPair->mod;
      if(apv_mod !=0 && !apv_mod->notInUse()){
        if(apvPair->count > 0 || apvPair->red!=-1) { useApvPairValue=true; break;}
      }
    }
  }
  if(!print_total){
    for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      if(apvPair!=0) {
	TmModule * apv_mod = apvPair->mod;
	if(apv_mod !=0 && !apv_mod->notInUse() ){
	  if(useApvPairValue) apvPair->value = apvPair->value / apvPair->count;
	  else if(apvPair->mpos==0 && apv_mod->count>0)apv_mod->value = apv_mod->value / apv_mod->count; 
	}
      }
    }
  }
  if(minvalue>=maxvalue){
    
    minvalue=9999999.;
    maxvalue=-9999999.;
    rangefound=false;
    for(i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
	TmApvPair *  apvPair= i_apv->second;
	if(apvPair!=0 ) {
	  TmModule * apv_mod = apvPair->mod;
	  if( apv_mod !=0 && !apv_mod->notInUse() ){
	    if(useApvPairValue){
	      rangefound=true;
	      if (minvalue > apvPair->value)minvalue=apvPair->value;
	      if (maxvalue < apvPair->value)maxvalue=apvPair->value;
	    } else {
              if(apv_mod->count>0){
	      rangefound=true;
	      if (minvalue > apv_mod->value)minvalue=apv_mod->value;
	      if (maxvalue < apv_mod->value)maxvalue=apv_mod->value;}
	    }
	  }
	}
    }
  }
  if ((title==" Tracker Map from  QTestAlarm") || (maxvalue == minvalue)||!rangefound) printflag = false;

     if(filetype=="svg"){
      saveAsSingleLayer=false;
      std::ostringstream outs;
    outs << outputfilename<<".svg";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<std::endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<std::endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<std::endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<std::endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<std::endl; 
  *savefile << "<svg:g id=\"fedtrackermap\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<std::endl;
     }
  for (int crate=firstcrate; crate < (ncrates+1); crate++){
    if(filetype=="xml"){
      saveAsSingleLayer=true;
      std::ostringstream outs;
    outs << outputfilename<<"crate" <<crate<< ".xml";
    savefile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *savefile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<std::endl;
    *savefile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<std::endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<std::endl;
    *savefile << "<script type=\"text/ecmascript\" xlink:href=\"crate.js\" />"<<std::endl;
    *savefile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerCrate.init()\">"<<std::endl;
    *savefile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<std::endl;
    *savefile << "<g id=\"crate\" transform=\" translate(150,500) rotate(270) scale(1.,1.)\"  > "<<std::endl;
        }
    //    ncrate=crate;
    defcwindow(crate);
    int numfed_incrate=0;
    for (i_fed=fedMap.begin();i_fed != fedMap.end(); i_fed++){
      if(i_fed->second == crate){
	int fedId = i_fed->first;
	//	numfed_incrate++;
	numfed_incrate = slotMap[fedId];
	// the following piece of code is used to prepare the HTML clickable map
	/*	
	double scalex=6285./6290.;
	double scaley=3510./3540.;
	double boxinitix=(NUMFED_INCOLUMN-1-(numfed_incrate-1)/NUMFED_INROW)*(NUMFEDCH_INCOLUMN+2)+NUMFEDCH_INCOLUMN+0.9;
	double boxinitiy=(NUMFED_INROW-1-(numfed_incrate-1)%NUMFED_INROW)*(NUMFEDCH_INROW+1)+NUMFEDCH_INROW+0.9;
	double boxendix=boxinitix-(NUMFEDCH_INCOLUMN-1)-0.9;
	double boxendiy=boxinitiy-(NUMFEDCH_INROW-1)-0.9;

	std::cout << "<area shape=\"rect\" coords=\" " 
		  << int(scalex*ydpixelc(boxinitiy)) << "," << int(3510-scaley*xdpixelc(boxinitix)) 
		  << "," << int(scalex*ydpixelc(boxendiy)) << "," << int(3510-scaley*xdpixelc(boxendix)) 
		  << "\" href=\"\" title=\"crate " << crate << " slot " << numfed_incrate << " FED " << fedId << "\" /> " << std::endl;
	*/
	//
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= apvMap[key];
	  if(apvPair !=0){
	    TmModule * apv_mod = apvPair->mod;
	    if(apv_mod !=0 && !apv_mod->notInUse()){
	      drawApvPair(crate,numfed_incrate,print_total,apvPair,savefile,useApvPairValue);
	    }
	  } 
	}
      }
    }
   if(!temporary_file){
    if(filetype=="xml"){
    *savefile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> " << std::endl;
    *savefile << "<tspan id=\"line1\" x=\"40\" y=\"30\"> </tspan> " << std::endl;
    *savefile << "<tspan id=\"line2\" x=\"40\" y=\"60\"> </tspan> " << std::endl;
    *savefile << " </text> </svg>" << std::endl;
    savefile->close();delete savefile;
     saveAsSingleLayer=false;
      }
      }
  }
    if(filetype=="svg"){
    *savefile << "</g> </svg> </svg> " << std::endl;
    savefile->close();delete savefile;
      }
  if(!print_total && !useApvPairValue){
//Restore module value
    for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      if(apvPair!=0) {
	TmModule * apv_mod = apvPair->mod;
	if(apv_mod !=0 && apvPair->mpos==0 && !apv_mod->notInUse()){
	  apv_mod->value = apv_mod->value * apv_mod->count;
	}
      }
    }
}
  
  int rangex = YFEDOFFSET+(YFEDCSIZE+YFEDOFFSET)*NUMFEDCRATE_INROW+300;
  int rangey = XFEDOFFSET+(XFEDCSIZE+XFEDOFFSET)*NUMFEDCRATE_INCOLUMN+300;
    
    if(temporary_file){
      if(printflag&&!saveWebInterface)drawPalette(savefile,rangex-140,rangey-100);
  savefile->close(); delete savefile;

  const char * command1;
  std::string tempfilename = outputfilename + ".coor";
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),std::ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);

    //    if(saveWebInterface)gPad->Range(0,0,3750,1600); else gPad->Range(0,0,3800,1600);
    std::cout << " range x " <<  rangex << std::endl;
    std::cout << " range y " <<  rangey << std::endl;
    gPad->Range(0,0,rangex,rangey);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col,*c;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ 
	colorList[colindex]=ncolor+100; 
	col =gROOT->GetColor(ncolor+100);
	if(col) 
	  col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); 
	else 
	  c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
	vc.push_back(c);
	ncolor++;
      }
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,std::ios::beg);
    std::cout << "created palette with " << ncolor << " colors" << std::endl;
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
        TPolyLine*  pline = new TPolyLine(npoints,y,x);
        vp.push_back(pline);
        pline->SetFillColor(colorList[colindex]);
        pline->SetLineWidth(0);
        pline->Draw("f");
      }
    }
        if (printflag&&!saveWebInterface) {
      float lminvalue=minvalue; float lmaxvalue=maxvalue;
      if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10);}
      axis = new TGaxis(rangex-140,34,rangex-140,rangey-106,lminvalue,lmaxvalue,510,"+L");
      axis->SetLabelSize(0.02);
      axis->Draw();
    }

    if(!saveWebInterface){
    TLatex l;
    l.SetTextSize(0.05);
    std::string fulltitle = title;
    if(tkMapLog && (fulltitle.find("Log10 scale") == std::string::npos)) fulltitle += ": Log10 scale";
    l.DrawLatex(50,rangey-200,fulltitle.c_str());
       }
    MyC->Update();
    std::cout << "Filetype " << filetype << std::endl;
    if(filetype=="png"){
      std::string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      std::string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      std::string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    std::string command = "rm "+tempfilename ;
    command1=command.c_str();
    std::cout << "Executing " << command1 << std::endl;
    system(command1);
    MyC->Clear();
    delete MyC;
     if (printflag&&!saveWebInterface)delete axis;
    for(std::vector<TPolyLine*>::iterator pos1=vp.begin();pos1!=vp.end();pos1++){
         delete (*pos1);}
   
  
}//if(temporary_file)
}//if(enabledFedProcessing)
}

void TrackerMap::load(std::string inputfilename){
  inputfile = new ifstream(inputfilename.c_str(),std::ios::in);
  std::string line,value;
  int ipos,ipos1,ipos2,id=0,val=0;
  int nline=0;
  while (getline( *inputfile, line ))
        {
        ipos1 = line.find("value=\"");
        if(ipos1 > 0)      {
             value = line.substr(ipos1+7,10);
             ipos = value.find("\"");
             value = value.substr(0,ipos); 
             val=atoi(value.c_str());
             }
        ipos2 = line.find("detid=\"");
        if(ipos2 > 0)      {
             value = line.substr(ipos2+7,10);
             ipos = value.find("\"");
             value = value.substr(0,ipos); 
             id = atoi(value.c_str());
             }
        if(ipos1>0 && ipos2>0 && val>0)this->fill(id,val);
        if(ipos1>0 && ipos2>0)nline++;
        //if(ipos1>0 && ipos2>0)std::cout << nline << " " << id << " " << val << std::endl; 

        }
       std::cout << nline << " modules found in this svg file " << std::endl;
       inputfile->close();delete inputfile;
 }



//print in svg format tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::print(bool print_total, float minval, float maxval, std::string outputfilename){
  temporary_file=false;
  std::ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".xml";
  svgfile = new std::ofstream(outs.str().c_str(),std::ios::out);
  jsfile = new ifstream(edm::FileInPath(jsfilename).fullPath().c_str(),std::ios::in);

  //copy javascript interface from trackermap.txt file
  std::string line;
  while (getline( *jsfile, line ))
        {
            *svgfile << line << std::endl;
        }
  jsfile->close();delete jsfile;
  //
 if(!print_total){
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          mod->value = mod->value / mod->count;
        }
      }
    }
  }
  }
  if(minvalue>=maxvalue){
  minvalue=9999999.;
  maxvalue=-9999999.;
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          if (minvalue > mod->value)minvalue=mod->value;
          if (maxvalue < mod->value)maxvalue=mod->value;
        }
      }
    }
  }
}
  for (int layer=1; layer < 44; layer++){
    //    nlay=layer;
    defwindow(layer);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,svgfile);
        }
      }
    }
  }
  *svgfile << "</svg:g></svg:svg>"<<std::endl;
  *svgfile << " <svg:text id=\"Title\" class=\"normalText\"  x=\"300\" y=\"0\">"<<title<<"</svg:text>"<<std::endl;
  if(printflag)drawPalette(svgfile);
  *svgfile << "</svg:svg>"<<std::endl;
  *svgfile << "</body></html>"<<std::endl;
   svgfile->close();delete svgfile;

}

void TrackerMap::drawPalette(std::ofstream * svgfile,int xoffset, int yoffset){
  std::cout << "preparing the palette" << std::endl;
  int color,red, green, blue;
  float val=minvalue;
  int paletteLength = 250;
  int width=50*(yoffset-40)/1500;
  float dval = (maxvalue-minvalue)/(float)paletteLength;
  bool rtkMapLog = tkMapLog; if (tkMapLog)tkMapLog=false;
  float step = float(yoffset-40)/float(paletteLength);
  for(int i=1;i<paletteLength+1;i++){
  color = getcolor(val,palette);
     red=(color>>16)&0xFF;
     green=(color>>8)&0xFF;
     blue=(color)&0xFF;
 //   if(!temporary_file)*svgfile <<"<svg:rect  x=\"3010\" y=\""<<(1550-6*i)<<"\" width=\"50\" height=\"6\" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" />\n"; 
  //  else *svgfile << red << " " << green << " " << blue << " 4 " << (6*i)+40 << " 3010. " <<//
   //           (6*i)+40 << " 3060. " <<//
    //          (6*(i-1))+40 << " 3060. " <<//
     //         (6*(i-1))+40 <<" 3010. " << std::endl; //

   // if(i%50 == 0){
    //  if(!temporary_file)*svgfile <<"<svg:rect  x=\"3010\" y=\""<<(1550-6*i)<<"\" width=\"50\" height=\"1\" fill=\"black\" />\n";
     // if(i%50==0&&!temporary_file)*svgfile << " <svg:text  class=\"normalText\"  x=\"3060\" y=\""<<(1560-6*i)<<"\">" <<val<<"</svg:text>"<<std::endl;

    if(!temporary_file)*svgfile <<"<svg:rect  x=\"3610\" y=\""<<(1550-6*i)<<"\" width=\"50\" height=\"6\" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" />\n"; 
    else *svgfile << red << " " << green << " " << blue << " 4 " << int(step*i)+34 << " " << xoffset-width << ". " <<//
      int(step*i)+34 << " " << xoffset << ". " <<//
      int(step*(i-1))+34 << " " << xoffset << ". " <<//
      int(step*(i-1))+34 << " " << xoffset-width << ". " << std::endl; //

    if(i%50 == 0){
     if(!temporary_file)*svgfile <<"<svg:rect  x=\"3610\" y=\""<<(1550-6*i)<<"\" width=\"50\" height=\"1\" fill=\"black\" />\n";
      if(i%50==0&&!temporary_file)*svgfile << " <svg:text  class=\"normalText\"  x=\"3660\" y=\""<<(1560-6*i)<<"\">" <<val<<"</svg:text>"<<std::endl;
       }
    val = val + dval;
   }
  tkMapLog=rtkMapLog;
} 
void TrackerMap::fillc_fed_channel(int fedId,int fedCh, int red, int green, int blue  )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  
  if(apvpair!=0){
    apvpair->red=red; apvpair->green=green; apvpair->blue=blue;
    return;
  }
  std::cout << "*** error in FedTrackerMap fillc method ***";
}

void TrackerMap::fill_fed_channel(int idmod, float qty  )
{
  std::multimap<const int, TmApvPair*>::iterator pos;
  for (pos = apvModuleMap.lower_bound(idmod);
         pos != apvModuleMap.upper_bound(idmod); ++pos) {
  TmApvPair* apvpair = pos->second;
  if(apvpair!=0){
    apvpair->value=apvpair->value+qty;
    apvpair->count++;
  }
  }
    return;
  std::cout << "*** error in FedTrackerMap fill by module method ***";
  }

void TrackerMap::fill_current_val_fed_channel(int fedId, int fedCh, float current_val )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  
  if(apvpair!=0)  {apvpair->value=current_val; apvpair->count=1; apvpair->red=-1;}
  else 
    std::cout << "*** error in FedTrackerMap fill_current_val method ***";
}


void TrackerMap::fillc_fec_channel(int crate,int slot, int ring, int addr, int red, int green, int blue  )
 {
 int key =crate*10000000+slot*100000+ring*1000+addr;

 TmCcu *ccu = ccuMap[key];
 
 if(ccu!=0){
    ccu->red=red; ccu->green=green; ccu->blue=blue;
    return;
  }
  std::cout << "*** error in FecTrackerMap fillc method ***";
}

void TrackerMap::fill_fec_channel(int crate,int slot, int ring, int addr, float qty  )
{
 int key =crate*10000000+slot*100000+ring*1000+addr;
 TmCcu *ccu = ccuMap[key];
  if(ccu!=0){
    ccu->count++; ccu->value=ccu->value+qty;
    return;
 
  }
  
  std::cout << "*** error in FecTrackerMap fill by module method ***";
  }

 

void TrackerMap::fillc_lv_channel(int rack,int crate, int board, int red, int green, int blue  )
{
 
 int key = rack*1000+crate*100+board;
 
 TmPsu *psu = psuMap[key];
  
  if(psu!=0){
    psu->red=red; psu->green=green; psu->blue=blue;
    return;
  }
  std::cout << "*** error in LVTrackerMap fillc method ***";
}

void TrackerMap::fill_lv_channel(int rack,int crate, int board, float qty  )
{
 int key = rack*1000+crate*100+board;
 TmPsu *psu = psuMap[key];
  if(psu!=0){
    psu->count++; psu->value=psu->value+qty;
    return;
 
  }
  
  std::cout << "*** error in LVTrackerMap fill by module method ***";
  }

void TrackerMap::fillc_hv_channel2(int rack,int crate, int board, int red, int green, int blue  )
{
 
 int key = rack*1000+crate*100+board;
 
 TmPsu *psu = psuMap[key];
  
  if(psu!=0){
    psu->redHV2=red; psu->greenHV2=green; psu->blueHV2=blue;
    return;
  }
  std::cout << "*** error in HVTrackerMap (channel 2) fillc method ***";
}
void TrackerMap::fillc_hv_channel3(int rack,int crate, int board, int red, int green, int blue  )
{
 
 int key = rack*1000+crate*100+board;
 
 TmPsu *psu = psuMap[key];
  
  if(psu!=0){
    psu->redHV3=red; psu->greenHV3=green; psu->blueHV3=blue;
    return;
  }
  std::cout << "*** error in HVTrackerMap (channel 3) fillc method ***";
}


void TrackerMap::fill_hv_channel2(int rack,int crate, int board, float qty  )
{
 int key = rack*1000+crate*100+board;
 TmPsu *psu = psuMap[key];
  if(psu!=0){
    psu->countHV2++; psu->valueHV2=psu->valueHV2+qty;
    return;
 
  }
  
  std::cout << "*** error in HVTrackerMap fill by module method ***";
  }
void TrackerMap::fill_hv_channel3(int rack,int crate, int board, float qty  )
{
 int key = rack*1000+crate*100+board;
 TmPsu *psu = psuMap[key];
  if(psu!=0){
    psu->countHV3++; psu->valueHV3=psu->valueHV3+qty;
    return;
 
  }
  
  std::cout << "*** error in HVTrackerMap fill by module method ***";
  }





int TrackerMap::module(int fedId, int fedCh)
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  if(apvpair!=0){
    return(apvpair->mod->idex);
  }
  return(0);
  std::cout << "*** error in FedTrackerMap module method ***";
}
void TrackerMap::fill_fed_channel(int fedId, int fedCh, float qty )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  if(apvpair!=0){
    apvpair->value=apvpair->value+qty;
    apvpair->count++;
    return;
  }
  std::cout << "*** error inFedTrackerMap fill method ***";
}


void TrackerMap::fillc(int idmod, int red, int green, int blue  ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
     mod->red=red; mod->green=green; mod->blue=blue;
     return;
  }
  std::cout << "**************************error in fill method **************module "<<idmod<<std::endl;
}
void TrackerMap::fillc(int layer, int ring, int nmod, int red, int green, int blue  ){
  
  int key = layer*10000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];

  if(mod!=0){
     mod->red=red; mod->green=green; mod->blue=blue;
    return;
  }
  std::cout << "**************************error in fill method **************"<< std::endl;
}

void TrackerMap::fillc_all_blank(){

  std::map<const int  , TmModule *>::iterator imod;
   for( imod=imoduleMap.begin();imod !=imoduleMap.end(); imod++){
   fillc(imod->first,255,255,255); 
   }
}

void TrackerMap::fill_all_blank(){

  std::map<const int  , TmModule *>::iterator imod;
   for( imod=imoduleMap.begin();imod !=imoduleMap.end(); imod++){
   fill_current_val(imod->first,0); 
   }
}



void TrackerMap::fill_current_val(int idmod, float current_val ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0)  {mod->value=current_val; mod->count=1;  mod->red=-1;}
  else std::cout << "**error in fill_current_val method ***module "<<idmod<<std::endl;
}

void TrackerMap::fill(int idmod, float qty ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
    mod->value=mod->value+qty;
    mod->count++;
    return;
  }else{
   TmModule * mod1 = imoduleMap[idmod+1];
   TmModule * mod2 = imoduleMap[idmod+2];
   if(mod1!=0 && mod2!=0){
    mod1->value=mod1->value+qty;
    mod1->count++;
    mod2->value=mod2->value+qty;
    mod2->count++;
    return;
   }}
  std::cout << "**************************error in fill method **************module "<<idmod<<std::endl;
}

void TrackerMap::fill(int layer, int ring, int nmod,  float qty){

  int key = layer*100000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];
  if(mod!=0){
     mod->value=mod->value+qty;
     mod->count++;
  }
  else std::cout << "**************************error in SvgModuleMap **************";
} 

void TrackerMap::setText(int idmod, std::string s){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
     mod->text=s;
  }
  else std::cout << "**************************error in IdModuleMap **************";
}


void TrackerMap::setText(int layer, int ring, int nmod, std::string s){

  int key = layer*100000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];
  if(mod!=0){
     mod->text=s;
  }
  else std::cout << "**************************error in SvgModuleMap **************";
} 

void TrackerMap::build(){
  //  ifstream* infile;

  int nmods, pix_sil, fow_bar, ring, nmod, layer;
  unsigned int idex;
  float posx, posy, posz, length, width, thickness, widthAtHalfLength;
  int iModule=0,old_layer=0, ntotMod =0;
  std::string name,dummys;
  ifstream infile(edm::FileInPath(infilename).fullPath().c_str(),std::ios::in);
  while(!infile.eof()) {
    infile >> nmods >> pix_sil >> fow_bar >> layer >> ring >> nmod >> posx >> posy
	   >> posz>> length >> width >> thickness
	   >> widthAtHalfLength >> idex ;
    getline(infile,dummys); //necessary to reach end of record
    getline(infile,name); 
    if(old_layer!=layer){old_layer=layer;iModule=0;}
    iModule++;
    ntotMod++;
    int key=layer*100000+ring*1000+nmod;
    TmModule * mod = smoduleMap[key];
    
    imoduleMap[idex]=mod;

    if(mod==0) std::cout << "error in module "<<key <<std::endl;
    else
      {
          mod->posx = posx;
          mod->posy = posy;
          mod->setUsed();
          mod->value=0;
          mod->count=0;
          mod->posz = posz;
          mod->length = length;
          mod->width = width;
          mod->thickness = thickness;
          mod->widthAtHalfLength = widthAtHalfLength;
          mod->idex = idex;
          mod->name = name;
      }
  }
  infile.close();
  number_modules = ntotMod-1;
}
int TrackerMap::getcolor(float value,int palette){
   int red,green,blue;
   float lminvalue, lmaxvalue;
   lminvalue=minvalue; lmaxvalue=maxvalue;
   if(tkMapLog) {lminvalue=log(minvalue)/log(10);lmaxvalue=log(maxvalue)/log(10); value=log(value)/log(10);}

   
   red=0;green=0;blue=0;
   if(palette==1){//palette1 1 - raibow
   float delta=(lmaxvalue-lminvalue);
   float x =(value-lminvalue);
   if(value<lminvalue){red=0;green=0;blue=255;}
   if(value>lmaxvalue){red=255;green=0;blue=0;}
   if(value>=lminvalue&&value<=lmaxvalue){ 
   red = (int) ( x<(delta/2) ? 0 : ( x > ((3./4.)*delta) ?  255 : 255/(delta/4) * (x-(2./4.)*delta)  ) );
   green= (int) ( x<delta/4 ? (x*255/(delta/4)) : ( x > ((3./4.)*delta) ?  255-255/(delta/4) *(x-(3./4.)*delta) : 255 ) );
   blue = (int) ( x<delta/4 ? 255 : ( x > ((1./2.)*delta) ?  0 : 255-255/(delta/4) * (x-(1./4.)*delta) ) );
     }
     }
     if (palette==2){//palette 2 yellow-green
     green = (int)((value-lminvalue)/(lmaxvalue-lminvalue)*256.);
         if (green > 255) green=255;
         red = 255; blue=0;green=255-green;  
        } 
  // std::cout<<red<<" "<<green<<" "<<blue<<" "<<value <<" "<<lminvalue<<" "<< lmaxvalue<<std::endl;
   return(blue|(green<<8)|(red<<16));
}
void TrackerMap::printonline(){
//Copy interface
  std::ofstream * ofilename;
  std::ifstream * ifilename;
  std::ostringstream ofname;
  std::string ifname;
  std::string command;
  std::string line;
  std::string outputfilename="dqmtmap";
  ifilename=findfile("viewerHeader.xhtml");
  ofname << outputfilename << "viewer.html";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
*ofilename <<"    var tmapname=\"" <<outputfilename << "\""<<std::endl;
*ofilename <<"    var tmaptitle=\"" <<title << "\""<<std::endl;
*ofilename <<"    var ncrates=" <<ncrates << ";"<<std::endl;
*ofilename <<"    var nfeccrates=" <<nfeccrates << ";"<<std::endl;
 *ofilename <<"    var npsuracks=" <<npsuracks << ";"<<std::endl;
 
   ifilename->close();delete ifilename;

  ifilename=findfile("viewerTrailer.xhtml");
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   command = "sed -i \"s/XtmapnameX/"+outputfilename+"/g\" "+ ofname.str();
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
   command = "sed -i \"s/XtmaptitleX/"+title+"/g\" "+ ofname.str();
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
  ofname.str("");
   ifilename->close();delete ifilename;

  ifilename=findfile("jqviewer.js");
  ofname << "jqviewer.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
  ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;

  ifilename=findfile("crate.js");
  ofname <<  "crate.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
  ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;

  ifilename=findfile("feccrate.js");
  ofname <<  "feccrate.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
  ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
  
  ifilename=findfile("layer.js");
  ofname <<  "layer.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
  
  ifilename=findfile("rack.js");
  ofname <<  "rack.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
   ofname.str("");
  
  ifilename=findfile("rackhv.js");
  ofname <<  "rackhv.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofname.str("");
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
    
    
    
    
   std::ostringstream outs,outs1,outs2;
    outs << outputfilename<<".png";
save(true,gminvalue,gmaxvalue,outs.str(),3000,1600);
temporary_file=false;
printlayers(true,gminvalue,gmaxvalue,outputfilename);

//Now print a text file for each layer 
  std::ofstream * txtfile;
for (int layer=1; layer < 44; layer++){
  std::ostringstream outs;
    outs << outputfilename <<"layer"<<layer<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
            int idmod=mod->idex;
            int nchan=0;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
             std::multimap<const int, TmApvPair*>::iterator pos;
             for (pos = apvModuleMap.lower_bound(idmod);
                pos != apvModuleMap.upper_bound(idmod); ++pos) {
               TmApvPair* apvpair = pos->second;
               if(apvpair!=0){
                   nchan++;
                   *txtfile  <<  apvpair->text << std::endl;
                    }

                    }
                   *txtfile  << "</pre><h3>"<< mod->name<<"</h3>"<<std::endl;
                  }
                }
                }
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();delete txtfile;
                }
if(enableFedProcessing){
    outs1 << outputfilename<<"fed.png";
save_as_fedtrackermap(true,gminvalue,gmaxvalue,outs1.str(),6000,3200);
    outs2 << outputfilename<<".xml";
save_as_fedtrackermap(true,gminvalue,gmaxvalue,outs2.str(),3000,1600);
//And a text file for each crate 
  std::map<int , int>::iterator i_fed;
  std::ofstream * txtfile;
  for (int crate=firstcrate; crate < (ncrates+1); crate++){
    std::ostringstream outs;
    outs << outputfilename <<"crate"<<crate<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for (i_fed=fedMap.begin();i_fed != fedMap.end(); i_fed++){
      if(i_fed->second == crate){
	int fedId = i_fed->first;
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= apvMap[key];
	  if(apvPair !=0){
            int idmod=apvPair->idex;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
            *txtfile  <<  apvPair->text << std::endl;
	    std::ostringstream outs;
            outs << "fedchannel "  <<apvPair->getFedId() << "/"<<apvPair->getFedCh()<<" connects to module  " << apvPair->mod->idex ;
            *txtfile  << "</pre><h3>"<< outs.str()<<"</h3>"<<std::endl;
             }
          }
      }
      }
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();delete txtfile;
                }
  }
if(enableFecProcessing){
  std::ostringstream outs1,outs2;
    outs1 << outputfilename<<"fec.png";
save_as_fectrackermap(true,gminvalue,gmaxvalue,outs1.str(),6000,3200);
    outs2 << outputfilename<<".xml";
save_as_fectrackermap(true,gminvalue,gmaxvalue,outs2.str(),3000,1600);
//And a text file for each crate
  std::ofstream * txtfile;
  std::map<int , TmCcu *>::iterator i_ccu;
  std::multimap<TmCcu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmCcu*, TmModule*>::iterator,std::multimap<TmCcu*, TmModule*>::iterator> ret;
  for (int crate=1; crate < (nfeccrates+1); crate++){
    std::ostringstream outs;
    outs << outputfilename <<"feccrate"<<crate<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
     TmCcu *  ccu= i_ccu->second;
      if(ccu!=0&&ccu->getCcuCrate() == crate){
            int idmod=ccu->idex;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
            *txtfile  <<  ccu->text << std::endl;
	    std::ostringstream outs;
            if(ccu->nmod==0)outs << "ccu  is in position" << ccu->mpos<<"in ring but doesn't seem to have any module connected"; else
            {
            outs << "ccu  is in position " << ccu->mpos<<" in ring and connects  " <<ccu->nmod<< " modules" << std::endl;
            ret = fecModuleMap.equal_range(ccu);
        for (it = ret.first; it != ret.second; ++it)
          {
           outs << (*it).second->idex<<" " << (*it).second->name <<" value= "<< (*it).second->value<<"\n\n";
          }

            *txtfile  << "</pre><h4>"<< outs.str()<<"</h4>"<<std::endl;
          }//ifccu->nmod==0
      }//if ccu!=0
      }//for i_ccu
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();delete txtfile;
                }//for int crate
  }
if(enableLVProcessing){
  std::ostringstream outs3,outs4;
    outs3 << outputfilename<<"psu.png";
save_as_psutrackermap(true,gminvalue,gmaxvalue,outs3.str(),6000,3200);

    outs4 << outputfilename<<".xml";
save_as_psutrackermap(true,gminvalue,gmaxvalue,outs4.str(),3000,1600);
//And a text file for each rack 
  
  std::ofstream * txtfile;
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
  for (int rack=1; rack < (npsuracks+1); rack++){
    std::ostringstream outs;
    
    outs << outputfilename <<"psurack"<<rack<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
     for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0 && psu->getPsuRack() == rack){
	*txtfile  << "<a name="<<psu->idex<<"><pre>"<<std::endl;      
        *txtfile  <<  psu->text << std::endl;
	std::ostringstream outs;
        if(psu->nmod==0)outs << "Ps is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outs<< "PS is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmod<<" modules. "<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   outs <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;
	   
	  }
	*txtfile  << "</pre><h4>"<< outs.str()<<"</h4>"<<std::endl;
     }
    }
  }
  *txtfile << "</body></html>" << std::endl;
   txtfile->close();delete txtfile;
  }
 } 


if(enableHVProcessing){
  std::ostringstream outs5,outs6;
    outs5 << outputfilename<<"hv.png";
save_as_HVtrackermap(true,gminvalue,gmaxvalue,outs5.str(),6000,3200);

    outs6 << outputfilename<<".xml";
save_as_HVtrackermap(true,gminvalue,gmaxvalue,outs6.str(),3000,1600);
//And a text file for each rack 

 std::ofstream * txtfile;
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
  for (int rack=1; rack < (npsuracks+1); rack++){
    std::ostringstream outs;
    
    outs << outputfilename <<"HVrack"<<rack<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
     for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0 && psu->getPsuRack() == rack){
	*txtfile  << "<a name="<<psu->idex<<"><pre>"<<std::endl;      
        *txtfile  <<  psu->textHV2 << std::endl;
	std::ostringstream outsHV2;
        if(psu->nmodHV2==0)outsHV2 << "HV Channel002 is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outsHV2<< "HV Channel002 is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmodHV2<<" modules. "<<" <br>"<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   if((*it).second->HVchannel==2){outsHV2 <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;}
	  }
	*txtfile  << "</pre><h4>"<< outsHV2.str()<<"</h4>"<<std::endl;
     }
    
        *txtfile  <<  psu->textHV3 << std::endl;
	std::ostringstream outsHV3;
        if(psu->nmodHV3==0)outsHV3 << "HV Channel003 is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outsHV3<< "HV Channel003 is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmodHV3<<" modules. "<<" <br>"<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   if((*it).second->HVchannel==3){outsHV3 <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;}
	  }
	*txtfile  << "</pre><h4>"<< outsHV3.str()<<"</h4>"<<std::endl;
     }
   
    }
  }
  *txtfile << "</body></html>" << std::endl;
   txtfile->close();delete txtfile;
  }
 } 
 
}
void TrackerMap::printall(bool print_total, float minval1, float maxval1, std::string s,int width, int height){
//Copy interface
 float minval,maxval; minval=minval1; maxval=maxval1; 
  if(tkMapLog && (minval<maxval)) {minval=pow(10.,minval1);maxval=pow(10.,maxval1);}
  std::string filetype=s,outputfilename=s;
  if(saveWebInterface){width=6000;height=3200;}
  else{ 
  size_t found=filetype.find_last_of(".");
  filetype=filetype.substr(found+1);
  found=outputfilename.find_last_of(".");
  outputfilename=outputfilename.substr(0,found);
  }
  std::ofstream * ofilename;
  std::ifstream * ifilename;
  std::ostringstream ofname;
  std::string ifname;
  std::string line;
  std::string command;
  if(saveWebInterface){
  ifilename=findfile("viewerHeader.xhtml");
  ofname << outputfilename << "viewer.html";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
*ofilename <<"    var tmapname=\"" <<outputfilename << "\""<<std::endl;
*ofilename <<"    var tmaptitle=\"" <<title << "\""<<std::endl;
*ofilename <<"    var ncrates=" <<ncrates << ";"<<std::endl;
*ofilename <<"    var nfeccrates=" <<nfeccrates << ";"<<std::endl;
*ofilename <<"    var npsuracks=" <<npsuracks << ";"<<std::endl;
   ifilename->close();delete ifilename;
  ifilename=findfile("viewerTrailer.xhtml");
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
   command = "sed -i \"s/XtmapnameX/"+outputfilename+"/g\" "+ ofname.str();
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
   command = "sed -i \"s/XtmaptitleX/"+title+"/g\" "+ ofname.str();
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
  ofname.str("");
  
ifilename=findfile("jqviewer.js");
  ofname << "jqviewer.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
 
  ofname.str("");
  ifilename=findfile("crate.js");
  ofname <<  "crate.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
 
  ofname.str("");
  ifilename=findfile("feccrate.js");
  ofname <<  "feccrate.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
 
  ofname.str("");
  ifilename=findfile("rack.js");
  ofname <<  "rack.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
 
   ofname.str("");
  ifilename=findfile("rackhv.js");
  ofname <<  "rackhv.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
   
   ofname.str("");
  ifilename=findfile("layer.js");
  ofname <<  "layer.js";
  ofilename = new std::ofstream(ofname.str().c_str(),std::ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << std::endl; }
   ofilename->close();delete ofilename;
   ifilename->close();delete ifilename;
  
   command = "scp -r ../../DQM/TrackerCommon/test/jquery/ .";
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
   command = "scp -r ../../CommonTools/TrackerMap/data/images/ .";
    std::cout << "Executing " << command << std::endl;
    system(command.c_str());
}
 
    std::ostringstream outs;
    outs << outputfilename<<".png";
    if(saveWebInterface)save(true,minval,maxval,outs.str(),3000,1600);
    else {if(saveGeoTrackerMap)save(true,minval,maxval,s,width,height);}
  if(saveWebInterface){
    std::ostringstream outs;
    outs << outputfilename<<".png";
temporary_file=false;
printlayers(true,minval,maxval,outputfilename);

//Now print a text file for each layer 
  std::ofstream * txtfile;
for (int layer=1; layer < 44; layer++){
  std::ostringstream outs;
    outs << outputfilename <<"layer"<<layer<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
            int idmod=mod->idex;
            int nchan=0;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
             std::multimap<const int, TmApvPair*>::iterator pos;
             for (pos = apvModuleMap.lower_bound(idmod);
                pos != apvModuleMap.upper_bound(idmod); ++pos) {
               TmApvPair* apvpair = pos->second;
               if(apvpair!=0){
                   nchan++;
                   *txtfile  <<  apvpair->text << std::endl;
                    }

                    }
                   *txtfile  << "</pre><h3>"<< mod->name<<"</h3>"<<std::endl;
                  }
                }
                }
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();delete txtfile;
}
                }
if(enableFedProcessing){
  std::ostringstream outs1,outs2;
    if(saveWebInterface)outs1 << outputfilename<<"fed.png";
        else outs1 << outputfilename<<"fed."<<filetype;
save_as_fedtrackermap(true,0.,0.,outs1.str(),width,height);
  if(saveWebInterface){
    outs2 << outputfilename<<".xml";
save_as_fedtrackermap(true,0.,0.,outs2.str(),3000,1600);
//And a text file for each crate 
  std::map<int , int>::iterator i_fed;
  std::ofstream * txtfile;
  for (int crate=firstcrate; crate < (ncrates+1); crate++){
    std::ostringstream outs;
    outs << outputfilename <<"crate"<<crate<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for (i_fed=fedMap.begin();i_fed != fedMap.end(); i_fed++){
      if(i_fed->second == crate){
	int fedId = i_fed->first;
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= apvMap[key];
	  if(apvPair !=0){
            int idmod=apvPair->idex;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
            *txtfile  <<  apvPair->text << std::endl;
	    std::ostringstream outs;
            outs << "fedchannel "  <<apvPair->getFedId() << "/"<<apvPair->getFedCh()<<" connects to module  " << apvPair->mod->idex ;
            *txtfile  << "</pre><h3>"<< outs.str()<<"</h3>"<<std::endl;
             }
          }
      }
      }
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();delete txtfile;
                }
   }
  }
if(enableFecProcessing){
  std::ostringstream outs1,outs2;
    if(saveWebInterface)outs1 << outputfilename<<"fec.png";
        else outs1 << outputfilename<<"fec."<<filetype;
save_as_fectrackermap(true,0.,0.,outs1.str(),width,height);
  if(saveWebInterface){
    outs2 << outputfilename<<".xml";
save_as_fectrackermap(true,0.,0.,outs2.str(),3000,1600);
//And a text file for each crate
  std::ofstream * txtfile;
  std::map<int , TmCcu *>::iterator i_ccu;
  std::multimap<TmCcu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmCcu*, TmModule*>::iterator,std::multimap<TmCcu*, TmModule*>::iterator> ret;
  for (int crate=1; crate < (nfeccrates+1); crate++){
    std::ostringstream outs;
    outs << outputfilename <<"feccrate"<<crate<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
    for( i_ccu=ccuMap.begin();i_ccu !=ccuMap.end(); i_ccu++){
     TmCcu *  ccu= i_ccu->second;
      if(ccu!=0&&ccu->getCcuCrate() == crate){
            int idmod=ccu->idex;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<std::endl;
            *txtfile  <<  ccu->text << std::endl;
	    std::ostringstream outs;
            if(ccu->nmod==0)outs << "ccu  is in position" << ccu->mpos<<"in ring but doesn't seem to have any module connected"; else
            {
            outs << "ccu  is in position " << ccu->mpos<<" in ring and connects  " <<ccu->nmod<< " modules" << std::endl;
            ret = fecModuleMap.equal_range(ccu);
        for (it = ret.first; it != ret.second; ++it)
          {
           outs << (*it).second->idex<<" " << (*it).second->name <<" value= "<< (*it).second->value<<"\n\n";
          }

            *txtfile  << "</pre><h4>"<< outs.str()<<"</h4>"<<std::endl;
          }//ifccu->nmod==0
      }//if ccu!=0
      }//for i_ccu
    *txtfile << "</body></html>" << std::endl;
    txtfile->close();
                }//for int crate
  }
  }
if(enableLVProcessing){
  std::ostringstream outs3,outs4;
    if(saveWebInterface)outs3 << outputfilename<<"psu.png";
        else outs3 << outputfilename<<"psu."<<filetype;
save_as_psutrackermap(true,0.,0.,outs3.str(),width,height);
  if(saveWebInterface){
    outs4 << outputfilename<<".xml";
save_as_psutrackermap(true,0.,0.,outs4.str(),3000,1600);
//And a text file for each rack 
  
  std::ofstream * txtfile;
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
  for (int rack=1; rack < (npsuracks+1); rack++){
    std::ostringstream outs;
    
    outs << outputfilename <<"psurack"<<rack<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
     for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0 && psu->getPsuRack() == rack){
	*txtfile  << "<a name="<<psu->idex<<"><pre>"<<std::endl;      
        *txtfile  <<  psu->text << std::endl;
	std::ostringstream outs;
        if(psu->nmod==0)outs << "Ps is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outs<< "PS is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmod<<" modules. "<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   outs <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;
	   
	  }
	*txtfile  << "</pre><h4>"<< outs.str()<<"</h4>"<<std::endl;
     }
    }
  }
  *txtfile << "</body></html>" << std::endl;
   txtfile->close();
  }
  }
 } 


if(enableHVProcessing){
  std::ostringstream outs5,outs6;
    if(saveWebInterface)outs5 << outputfilename<<"hv.png";
        else outs5 << outputfilename<<"hv."<<filetype;
save_as_HVtrackermap(true,0.,0.,outs5.str(),width,height);
  if(saveWebInterface){
    outs6 << outputfilename<<".xml";
save_as_HVtrackermap(true,0.,0.,outs6.str(),3000,1600);
//And a text file for each rack 

  std::ofstream * txtfile;
  std::map<int , TmPsu *>::iterator ipsu;
  std::multimap<TmPsu*, TmModule*>::iterator it;
  std::pair<std::multimap<TmPsu*, TmModule*>::iterator,std::multimap<TmPsu*, TmModule*>::iterator> ret;
  for (int rack=1; rack < (npsuracks+1); rack++){
    std::ostringstream outs;
    
    outs << outputfilename <<"HVrack"<<rack<< ".html";
    txtfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *txtfile << "<html><head></head> <body>" << std::endl;
     for ( ipsu=psuMap.begin();ipsu !=psuMap.end(); ipsu++){
      TmPsu *  psu= ipsu->second;
      if(psu!=0 && psu->getPsuRack() == rack){
	*txtfile  << "<a name="<<psu->idex<<"><pre>"<<std::endl;      
        *txtfile  <<  psu->textHV2 << std::endl;
	std::ostringstream outsHV2;
        if(psu->nmodHV2==0)outsHV2 << "HV Channel002 is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outsHV2<< "HV Channel002 is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmodHV2<<" modules. "<<" <br>"<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   if((*it).second->HVchannel==2){outsHV2 <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;}
	  }
	*txtfile  << "</pre><h4>"<< outsHV2.str()<<"</h4>"<<std::endl;
     }
    
        *txtfile  <<  psu->textHV3 << std::endl;
	std::ostringstream outsHV3;
        if(psu->nmodHV3==0)outsHV3 << "HV Channel003 is in position" << psu->getPsuBoard()<<"in crate but doesn't seem to have any module connected"; else
	{
	outsHV3<< "HV Channel003 is in position "  <<psu->getPsuBoard()<< " in crate and connects to "<<psu->nmodHV3<<" modules. "<<" <br>"<<std::endl;
        
	ret = psuModuleMap.equal_range(psu);
	for (it = ret.first; it != ret.second; ++it)
	  {
	   if((*it).second->HVchannel==3){outsHV3 <<(*it).second->idex << " "<< (*it).second->name<<" value= "<<(*it).second->value<<" <br>"<<std::endl;}
	  }
	*txtfile  << "</pre><h4>"<< outsHV3.str()<<"</h4>"<<std::endl;
     }
   
    }
  }
  *txtfile << "</body></html>" << std::endl;
   txtfile->close();
  }
 } 
 } 


}


std::ifstream * TrackerMap::findfile(std::string filename) {
  std::ifstream * ifilename;
  std::string ifname;
  if(jsPath!=""){
  ifname=jsPath+filename;
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),std::ios::in);
  if(!ifilename){
  ifname="CommonTools/TrackerMap/data/"+filename;
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),std::ios::in);
  }
  }else {
  ifname="CommonTools/TrackerMap/data/"+filename;
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),std::ios::in);
 }
  if(!ifilename)std::cout << "File " << filename << " missing" << std::endl;
  return ifilename;
 }
void TrackerMap::printlayers(bool print_total, float minval, float maxval, std::string outputfilename){
  std::ofstream * xmlfile;
saveAsSingleLayer=true;
if(!print_total){
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()){
	    mod->value = mod->value / mod->count;
	  }
	}
      }
    }
  }
  if(minvalue>=maxvalue){
    minvalue=9999999.;
    maxvalue=-9999999.;
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()){
	    if (minvalue > mod->value)minvalue=mod->value;
	    if (maxvalue < mod->value)maxvalue=mod->value;
	  }
	}
      }
    }
  }
for (int layer=1; layer < 44; layer++){
  std::ostringstream outs;
    outs << outputfilename <<"layer"<<layer<< ".xml";
    xmlfile = new std::ofstream(outs.str().c_str(),std::ios::out);
    *xmlfile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<std::endl;
    *xmlfile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<std::endl;
    *xmlfile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<std::endl;
    *xmlfile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<std::endl;
    *xmlfile << "<script type=\"text/ecmascript\" xlink:href=\"layer.js\" />"<<std::endl;
    *xmlfile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerLayer.init()\">"<<std::endl;
    if(layer<31)*xmlfile << "<g id=\"layer\" transform=\" translate(0,400) rotate(270) scale(1.,1.)\"  > "<<std::endl;
    else *xmlfile << "<g id=\"layer\" transform=\" translate(0,400) rotate(270) scale(1.,0.8)\"  > "<<std::endl;
    *xmlfile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<std::endl;
    *xmlfile << "<svg:polygon id=\"fed\" mapAttribute=\"fed\" points=\"250,40 250,10 230,10 230,40\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" fill=\"rgb(0,127,255)\"/>"<<std::endl;
    *xmlfile << "<svg:polygon id=\"fec\" mapAttribute=\"fec\" points=\"228,40 228,10 208,10 208,40\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" fill=\"rgb(0,127,255)\"/>"<<std::endl;
    *xmlfile << "<svg:polygon id=\"lv\" mapAttribute=\"lv\" points=\"206,40 206,10 186,10 186,40\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" fill=\"rgb(0,127,255)\"/>"<<std::endl;
    *xmlfile << "<svg:polygon id=\"hv\" mapAttribute=\"hv\" points=\"184,40 184,10 164,10 164,40\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" fill=\"rgb(0,127,255)\"/>"<<std::endl;
    *xmlfile << "<svg:polygon id=\"plot\" mapAttribute=\"plot\" points=\"155,45 155,5 135,5 135,45\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" fill=\"rgb(200,0,0)\"/>"<<std::endl;
  
    //    nlay=layer;
    defwindow(layer);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,xmlfile);
        }
      }
    }
    *xmlfile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\">" << std::endl;
    *xmlfile << "<tspan id=\"line1\" x=\"40\" y=\"30\"> </tspan> " << std::endl;
    *xmlfile << "<tspan id=\"line2\" x=\"40\" y=\"60\"> </tspan> " << std::endl;
    *xmlfile << "<tspan id=\"line3\" x=\"40\" y=\"90\"> </tspan> " << std::endl;
    *xmlfile << "<tspan id=\"line4\" x=\"40\" y=\"120\"> </tspan> " << std::endl;
    if(layer > 33){
    *xmlfile << "<tspan  mapAttribute=\"fed\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"15\" y=\"228\" font-size=\"12\" font-family=\"arial\" fill=\"white\">FED</tspan> " <<std::endl;
    *xmlfile << "<tspan  mapAttribute=\"fec\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"15\" y=\"258\" font-size=\"12\" font-family=\"arial\" fill=\"white\">FEC</tspan> " <<std::endl;
    *xmlfile << "<tspan  mapAttribute=\"lv\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"18\" y=\"289\" font-size=\"12\" font-family=\"arial\" fill=\"white\">LV</tspan> " <<std::endl;
    *xmlfile << "<tspan  mapAttribute=\"hv\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"18\" y=\"319\" font-size=\"12\" font-family=\"arial\" fill=\"white\">HV</tspan> " <<std::endl;
    *xmlfile << "<tspan  mapAttribute=\"plot\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"12\" y=\"360\" font-size=\"12\" font-family=\"arial\" fill=\"white\">PLOT</tspan> " <<std::endl;
    }
    else{
    *xmlfile << "<tspan   mapAttribute=\"fed\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"21\" y=\"228\" font-size=\"12\" font-family=\"arial\" fill=\"white\">FED</tspan> " <<std::endl;
    *xmlfile << "<tspan   mapAttribute=\"fec\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"21\" y=\"258\" font-size=\"12\" font-family=\"arial\" fill=\"white\">FEC</tspan> " <<std::endl;
    *xmlfile << "<tspan   mapAttribute=\"lv\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"24\" y=\"289\" font-size=\"12\" font-family=\"arial\" fill=\"white\">LV</tspan> " <<std::endl;
    *xmlfile << "<tspan   mapAttribute=\"hv\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"24\" y=\"319\" font-size=\"12\" font-family=\"arial\" fill=\"white\">HV</tspan> " <<std::endl;
    *xmlfile << "<tspan   mapAttribute=\"plot\" onclick=\"chooseMap(evt);\" onmouseover=\"chooseMap(evt);\" onmouseout=\"chooseMap(evt);\" x=\"17\" y=\"360\" font-size=\"12\" font-family=\"arial\" fill=\"white\">PLOT</tspan> " <<std::endl;
    }
    *xmlfile << " </text> </svg>" << std::endl;
    xmlfile->close();delete xmlfile;
  }
saveAsSingleLayer=false;
}
