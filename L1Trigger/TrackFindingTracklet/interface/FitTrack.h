//This class implementes the track fit
#ifndef FITTRACK_H
#define FITTRACK_H

#include "ProcessBase.h"
#include "TrackDerTable.h"

#ifdef USEHYBRID
#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#endif

using namespace std;

class FitTrack:public ProcessBase{

 public:

  FitTrack(string name, unsigned int iSector):
   ProcessBase(name,iSector){
    trackfit_=0;
   }

  void addOutput(MemoryBase* memory,string output){
   if (writetrace) {
    cout << "In "<<name_<<" adding output to "<<memory->getName()
     << " to output "<<output<<endl;
   }
   if (output=="trackout"){
    TrackFitMemory* tmp=dynamic_cast<TrackFitMemory*>(memory);
    assert(tmp!=0);
    trackfit_=tmp;
    return;
   }

   assert(0);
  }


  void addInput(MemoryBase* memory,string input){
   if (writetrace) {
    cout << "In "<<name_<<" adding input from "<<memory->getName()
     << " to input "<<input<<endl;
   }
   if (input=="tpar1in"||
     input=="tpar2in"||
     input=="tpar3in"||
     input=="tpar4in"||
     input=="tpar5in"||
     input=="tpar6in"||
     input=="tpar7in"||
     input=="tpar8in"||
     input=="tpar9in"||
     input=="tpar10in"||
     input=="tpar11in"||
     input=="tpar12in"||
     input=="tpar13in"||
     input=="tpar14in"||
     input=="tpar15in"){
    TrackletParametersMemory* tmp=dynamic_cast<TrackletParametersMemory*>(memory);
    assert(tmp!=0);
    seedtracklet_.push_back(tmp);
    return; 
   }
   if (input=="fullmatch1in1"||
     input=="fullmatch1in2"||
     input=="fullmatch1in3"||
     input=="fullmatch1in4"||
     input=="fullmatch1in5"||
     input=="fullmatch1in6"||
     input=="fullmatch1in7"||
     input=="fullmatch1in8"||
     input=="fullmatch1in9"||
     input=="fullmatch1in10"||
     input=="fullmatch1in11"||
     input=="fullmatch1in12"||
     input=="fullmatch1in13"||
     input=="fullmatch1in14"||
     input=="fullmatch1in15"||
     input=="fullmatch1in16"
     ){
    FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp!=0);
    fullmatch1_.push_back(tmp);
    return;
   }
   if (input=="fullmatch2in1"||
     input=="fullmatch2in2"||
     input=="fullmatch2in3"||
     input=="fullmatch2in4"||
     input=="fullmatch2in5"||
     input=="fullmatch2in6"||
     input=="fullmatch2in7"||
     input=="fullmatch2in8"||
     input=="fullmatch2in9"||
     input=="fullmatch2in10"||
     input=="fullmatch2in11"||
     input=="fullmatch2in12"||
     input=="fullmatch2in13"||
     input=="fullmatch2in14"||
     input=="fullmatch2in15"||
     input=="fullmatch2in16"
     ){
    FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp!=0);
    fullmatch2_.push_back(tmp);
    return;
   }
   if (input=="fullmatch3in1"||
     input=="fullmatch3in2"||
     input=="fullmatch3in3"||
     input=="fullmatch3in4"||
     input=="fullmatch3in5"||
     input=="fullmatch3in6"||
     input=="fullmatch3in7"||
     input=="fullmatch3in8"||
     input=="fullmatch3in9"||
     input=="fullmatch3in10"||
     input=="fullmatch3in11"||
     input=="fullmatch3in12"||
     input=="fullmatch3in13"||
     input=="fullmatch3in14"||
     input=="fullmatch3in15"||
     input=="fullmatch3in16"||
     input=="fullmatch3in17"
     ){
    FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp!=0);
    fullmatch3_.push_back(tmp);
    return;
   }
   if (input=="fullmatch4in1"||
     input=="fullmatch4in2"||
     input=="fullmatch4in3"||
     input=="fullmatch4in4"||
     input=="fullmatch4in5"||
     input=="fullmatch4in6"||
     input=="fullmatch4in7"||
     input=="fullmatch4in8"||
     input=="fullmatch4in9"||
     input=="fullmatch4in10"||
     input=="fullmatch4in11"||
     input=="fullmatch4in12"||
     input=="fullmatch4in13"||
     input=="fullmatch4in14"||
     input=="fullmatch4in15"||
     input=="fullmatch4in16"||
     input=="fullmatch4in17"
     ){
    FullMatchMemory* tmp=dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp!=0);
    fullmatch4_.push_back(tmp);
    return;
   }
   cout << "Did not find input : "<<input<<endl;
   assert(0);
  }


#ifndef USEHYBRID
  void trackFitNew(Tracklet* tracklet, std::vector<std::pair<Stub*,L1TStub*>> &, std::vector<std::pair<int,int>> &){
#endif

#ifdef USEHYBRID
  void trackFitNew(Tracklet* tracklet, std::vector<std::pair<Stub*,L1TStub*>> &trackstublist, std::vector<std::pair<int,int>> &stubidslist){
   if (doKF) {

    // From full match lists, collect all the stubs associated with the tracklet seed

    // Get seed stubs first
    trackstublist.push_back(std::make_pair(tracklet->innerFPGAStub(), tracklet->innerStub()));
    if(tracklet->getISeed()>=8) trackstublist.push_back(std::make_pair(tracklet->middleFPGAStub(), tracklet->middleStub()));
    trackstublist.push_back(std::make_pair(tracklet->outerFPGAStub(), tracklet->outerStub()));

    // Now get ALL matches (can have multiple per layer)
    for (unsigned int i=0;i<fullmatch1_.size();i++) {
     for (unsigned int j=0;j<fullmatch1_[i]->nMatches();j++) {
      if (fullmatch1_[i]->getFPGATracklet(j)->TCID()==tracklet->TCID()) {
       trackstublist.push_back(fullmatch1_[i]->getMatch(j).second);
      }
     }
    }

    for (unsigned int i=0;i<fullmatch2_.size();i++) {
     for (unsigned int j=0;j<fullmatch2_[i]->nMatches();j++) {
      if (fullmatch2_[i]->getFPGATracklet(j)->TCID()==tracklet->TCID()) {
       trackstublist.push_back(fullmatch2_[i]->getMatch(j).second);
      }
     }
    }

    for (unsigned int i=0;i<fullmatch3_.size();i++) {
     for (unsigned int j=0;j<fullmatch3_[i]->nMatches();j++) {
      if (fullmatch3_[i]->getFPGATracklet(j)->TCID()==tracklet->TCID()) {
       trackstublist.push_back(fullmatch3_[i]->getMatch(j).second);
      }
     }
    }

    for (unsigned int i=0;i<fullmatch4_.size();i++) {
     for (unsigned int j=0;j<fullmatch4_[i]->nMatches();j++) {
      if (fullmatch4_[i]->getFPGATracklet(j)->TCID()==tracklet->TCID()) {
       trackstublist.push_back(fullmatch4_[i]->getMatch(j).second);
      }
     }
    }


    // For merge removal, loop through the resulting list of stubs to calculate their stubids
    if(RemovalType=="merge") {
      for(std::vector<std::pair<Stub*,L1TStub*>>::iterator it=trackstublist.begin(); it!=trackstublist.end(); it++) {
        int layer = it->first->layer().value()+1; // Assume layer (1-6) stub first
        if(it->first->layer().value() < 0) { // if disk stub, though...
          layer = it->first->disk().value()+10*it->first->disk().value()/abs(it->first->disk().value()); //disk = +/- 11-15
        }
        stubidslist.push_back(std::make_pair(layer,(it->first->phiregion().value()<<7)+it->first->stubindex().value()));
      }

      // And that's all we need! The rest is just for fitting, which is duplicated in PurgeDuplicate out of necessity
      return;
    }

    HybridFit hybridFitter(iSector_,extended_,nHelixPar_);
    hybridFitter.Fit(tracklet, trackstublist);
    return;
   }
#endif

   static TrackDerTable derTable;


   //test
   static bool first=true;
   if (first) {
    derTable.readPatternFile(fitpatternfile);
    derTable.fillTable();
    cout << "Number of entries in derivative table: "
     <<derTable.getEntries()<<endl;
    assert(derTable.getEntries()!=0);

    //testDer();
    first=false;
   }

   //First step is to build list of layers and disks.

   int layers[6];
   double r[6];
   unsigned int nlayers=0;
   int disks[5];
   double z[5];
   unsigned int ndisks=0;

   //Why do we need to use 10 entries here?
   double phiresid[10];
   double zresid[10];
   double phiresidexact[10];
   double zresidexact[10];
   int iphiresid[10];
   int izresid[10];
   double alpha[10];

   for(unsigned int i=0;i<10;i++){
    iphiresid[i]=0;
    izresid[i]=0;
    alpha[i]=0.0;

    phiresid[i]=0.0;
    zresid[i]=0.0;
    phiresidexact[i]=0.0;
    zresidexact[i]=0.0;
    iphiresid[i]=0;
    izresid[i]=0;
   }


   static ofstream out2;
   if (writeHitPattern) out2.open("hitpattern.txt");

   char matches[8]="000000\0";
   char matches2[12]="0000000000\0";
   int mult=1;


   unsigned int layermask=0;
   unsigned int diskmask=0;
   unsigned int alphaindex=0;
   unsigned int power=1;

   double t=tracklet->t();
   double rinv=tracklet->rinv();

   if (tracklet->isBarrel()) {

    for (unsigned int l=1;l<=6;l++) {
     if (l==(unsigned int)tracklet->layer()||
       l==(unsigned int)tracklet->layer()+1) {
      matches[l-1]='1';
      layermask|=(1<<(6-l));
      layers[nlayers++]=l;
      continue;
     }
     if (tracklet->match(l)) {
       matches[l-1]='1';
       layermask|=(1<<(6-l));
       phiresid[nlayers]=tracklet->phiresidapprox(l);
       zresid[nlayers]=tracklet->zresidapprox(l);
       phiresidexact[nlayers]=tracklet->phiresid(l);
       zresidexact[nlayers]=tracklet->zresid(l);
       iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
       izresid[nlayers]=tracklet->fpgazresid(l).value();
       
       layers[nlayers++]=l;
     }	
    }



    for (unsigned int d=1;d<=5;d++) {
     if (layermask&(1<<(d-1))) continue;

     if (mult==1<<(3*alphaBitsTable)) continue;

     if (ndisks+nlayers>=6) continue;
     if (tracklet->matchdisk(d)) {
      if (fabs(tracklet->alphadisk(d))<1e-20) {
       matches2[2*(5-d)]='1';
       diskmask|=(1<<(2*(5-d)+1));
      }
      else{
       int ialpha = tracklet->ialphadisk(d).value();
       //cout << "StubA ialpha "<<ialpha<<endl;
       int nalpha = tracklet->ialphadisk(d).nbits();
       //cout << "StubA ialpha nalpha "<<ialpha<<" "<<nalpha<<endl;
       nalpha = nalpha - alphaBitsTable; 
       ialpha = (1<<(alphaBitsTable-1)) + (ialpha>>nalpha);
       //cout << "ialpha ialphatable : "<<tracklet->ialphadisk(d).value()<<" "<<ialpha<<endl;

       alphaindex+=ialpha*power;
       power=power<<alphaBitsTable;
       matches2[2*(d-1)+1]='1';
       diskmask|=(1<<(2*(5-d)));
       mult=mult<<alphaBitsTable;
      }
      alpha[ndisks]=tracklet->alphadisk(d);
      phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
      zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
      phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
      zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
      iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
      izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();

      disks[ndisks++]=d;
     }
    }

    if (mult<=1<<(3*alphaBitsTable)) {
     if (writeHitPattern) {
      out2<<matches<<" "<<matches2<<" "<<mult<<endl;
     }
    }

   } 

   if (tracklet->isDisk()) {

    for (unsigned int l=1;l<=2;l++) {
     if (tracklet->match(l)) {
      matches[l-1]='1';

      layermask|=(1<<(6-l));

      phiresid[nlayers]=tracklet->phiresidapprox(l);
      zresid[nlayers]=tracklet->zresidapprox(l);
      phiresidexact[nlayers]=tracklet->phiresid(l);
      zresidexact[nlayers]=tracklet->zresid(l);
      iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
      izresid[nlayers]=tracklet->fpgazresid(l).value();

      layers[nlayers++]=l;
     }
    }


    for (unsigned int d1=1;d1<=5;d1++) {
     int d=d1;

     // skip F/B5 if there's already a L2 match
     if (d==5 and layermask&(1<<4)) continue;

     if (tracklet->fpgat().value()<0.0) d=-d1;
     if (d==tracklet->disk()||  //All seeds in PS modules
       d==tracklet->disk2()){
      matches2[2*(5-d1)]='1';
      diskmask|=(1<<(2*(5-d1)+1));
      alpha[ndisks]=0.0;
      disks[ndisks++]=d;
      continue;
     }

     if (ndisks+nlayers>=6) continue;	
     if (tracklet->matchdisk(d)) {

      if (fabs(tracklet->alphadisk(d))<1e-20) {
       matches2[2*(5-d1)]='1';
       diskmask|=(1<<(2*(5-d1)+1));
      }
      else{
       int ialpha = tracklet->ialphadisk(d).value();
       //cout << "StubB ialpha "<<ialpha<<endl;
       int nalpha = tracklet->ialphadisk(d).nbits();
       //cout << "StubB ialpha nalpha "<<ialpha<<" "<<nalpha<<endl;
       nalpha = nalpha - alphaBitsTable;
       ialpha = (1<<(alphaBitsTable-1)) + (ialpha>>nalpha);

       alphaindex+=ialpha*power;
       power=power<<alphaBitsTable;
       matches2[2*(d1-1)+1]='1';
       diskmask|=(1<<(2*(5-d1)));
       mult=mult<<alphaBitsTable;
      }

      alpha[ndisks]=tracklet->alphadisk(d);
      assert(fabs(tracklet->phiresidapproxdisk(d))<0.2);
      phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
      zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
      assert(fabs(tracklet->phiresiddisk(d))<0.2);
      phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
      zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
      iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
      izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();

      disks[ndisks++]=d;
     }
    }

   } 

   if (tracklet->isOverlap()) {

    for (unsigned int l=1;l<=2;l++) {
     if (l==(unsigned int)tracklet->layer()) {
      matches[l-1]='1';
      layermask|=(1<<(6-l));
      layers[nlayers++]=l;
      continue;
     }
     if (tracklet->match(l)) {
      matches[l-1]='1';
      layermask|=(1<<(6-l));
      assert(fabs(tracklet->phiresidapprox(l))<0.2);
      phiresid[nlayers]=tracklet->phiresidapprox(l);
      zresid[nlayers]=tracklet->zresidapprox(l);
      assert(fabs(tracklet->phiresid(l))<0.2);
      phiresidexact[nlayers]=tracklet->phiresid(l);
      zresidexact[nlayers]=tracklet->zresid(l);
      iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
      izresid[nlayers]=tracklet->fpgazresid(l).value();

      layers[nlayers++]=l;
     }
    }


    for (unsigned int d1=1;d1<=5;d1++) {

     if (mult==1<<(3*alphaBitsTable)) continue;
     int d=d1;
     if (tracklet->fpgat().value()<0.0) d=-d1;
     if (d==tracklet->disk()){  //All seeds in PS modules
      disks[ndisks]=tracklet->disk();
      matches2[2*(5-d1)]='1';
      diskmask|=(1<<(2*(5-d1)+1));
      //alpha[ndisks]=0.0;
      ndisks++;
      continue;
     }


     if (ndisks+nlayers>=6) continue;
     if (tracklet->matchdisk(d)) {
      if (fabs(tracklet->alphadisk(d))<1e-20) {
       matches2[2*(d1-1)]='1';
       diskmask|=(1<<(2*(5-d1)+1));
       FPGAWord tmp;
       tmp.set(diskmask,10);
      }
      else{
       int ialpha = tracklet->ialphadisk(d).value();
       //cout << "StubC ialpha "<<ialpha<<endl;	    
       int nalpha = tracklet->ialphadisk(d).nbits();
       //cout << "StubC ialpha nalpha "<<ialpha<<" "<<nalpha<<endl;
       nalpha = nalpha - alphaBitsTable;
       ialpha = (1<<(alphaBitsTable-1)) + (ialpha>>nalpha);

       alphaindex+=ialpha*power;
       power=power<<alphaBitsTable;
       matches2[2*(d1-1)+1]='1';
       diskmask|=(1<<(2*(5-d1)));
       FPGAWord tmp;
       tmp.set(diskmask,10);
       mult=mult<<alphaBitsTable;
      }

      alpha[ndisks]=tracklet->alphadisk(d);
      assert(fabs(tracklet->phiresidapproxdisk(d))<0.2);
      phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
      zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
      assert(fabs(tracklet->phiresiddisk(d))<0.2);
      phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
      zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
      iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
      izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();

      disks[ndisks++]=d;
     }
    }

   } 

   int rinvindex=(1<<(nrinvBitsTable-1))*rinv/0.0057+(1<<(nrinvBitsTable-1));
   if (rinvindex<0) rinvindex=0;
   if (rinvindex>=(1<<nrinvBitsTable)) rinvindex=(1<<nrinvBitsTable)-1;

   int ptbin=0;
   if (fabs(rinv)<0.0057/2) ptbin=1;
   if (fabs(rinv)<0.0057/4) ptbin=2;
   if (fabs(rinv)<0.0057/8) ptbin=3;

   TrackDer* derivatives=derTable.getDerivatives(layermask, diskmask,alphaindex,rinvindex);

   if (derivatives==0) {
    FPGAWord tmpl,tmpd;
    tmpl.set(layermask,6);
    tmpd.set(diskmask,10);
    cout << "No derivative for layermask, diskmask : "
     <<layermask<<" "<<tmpl.str()<<" "<<diskmask<<" "<<tmpd.str()<<" eta = "<<asinh(t)<<endl;
    return;
   }

   double ttabi=TrackDerTable::gett(diskmask, layermask);
   if (t<0.0) ttabi=-ttabi;
   double ttab=ttabi;

   if (debug1) {
    cout << "Doing trackfit in  "<<getName()<<endl;
   }

   int sign=1;
   if (t<0.0) sign=-1;

   double rstub[6];

   double realrstub[3];
   realrstub[0]=-1.0;
   realrstub[1]=-1.0;
   realrstub[2]=-1.0;

   for (unsigned i=0;i<nlayers;i++){
    r[i]=rmean[layers[i]-1];
    if (layers[i]==tracklet->layer()) {
	if (tracklet->isOverlap()) {
	  realrstub[i]=tracklet->outerStub()->r();
	} else {
	  realrstub[i]=tracklet->innerStub()->r();
	}
     //r[i]=realrstub[i]; //test
    }
    if (layers[i]==tracklet->layer()+1) {
     realrstub[i]=tracklet->outerStub()->r();
     //r[i]=realrstub[i]; //test
    }
    if (tracklet->validResid(layers[i])&&layers[i]<4) {
     std::pair<Stub*,L1TStub*> stubptrs=tracklet->stubptrs(layers[i]);
     realrstub[i]=stubptrs.second->r();
     //r[i]=realrstub[i];  //test
     assert(fabs(realrstub[i]-r[i])<5.0);
    }
    rstub[i]=r[i];
   }
   for (unsigned i=0;i<ndisks;i++){
    z[i]=sign*zmean[abs(disks[i])-1];
    rstub[i+nlayers]=z[i]/ttabi;
   }

   double D[4][12];
   double MinvDt[4][12];
   int iD[4][12];
   int iMinvDt[4][12];
   double sigma[12];
   double kfactor[12];

   unsigned int n=nlayers+ndisks;

   if (exactderivatives) {
    TrackDerTable::calculateDerivatives(nlayers,r,ndisks,z,alpha,t,rinv,
      D,iD,MinvDt,iMinvDt,sigma,kfactor);
    ttabi=t;
    ttab=t;
   } else {
    if (exactderivativesforfloating) {
     if (useMSFit) {
      TrackDerTable::calculateDerivativesMS(nlayers,r,ndisks,z,alpha,
	t,rinv,D,iD,MinvDt,iMinvDt,
	sigma,kfactor,ptbin);
     } else {

      TrackDerTable::calculateDerivatives(nlayers,r,ndisks,z,alpha,t,
	rinv,D,iD,MinvDt,iMinvDt,
	sigma,kfactor);

     }

     double MinvDtDummy[4][12];
     derivatives->fill(tracklet->fpgat().value(),MinvDtDummy,iMinvDt);
     ttab=t;
    } else {
     derivatives->fill(tracklet->fpgat().value(),MinvDt,iMinvDt);
    }
   }


   if (!exactderivatives) {
     for (unsigned int i=0;i<nlayers;i++){
       if (r[i]>60.0) continue;
       for (unsigned int ii=0;ii<nlayers;ii++){
	 if (r[ii]>60.0) continue;
	 
	 double tder=derivatives->gettdzcorr(i,ii);
	 double zder=derivatives->getz0dzcorr(i,ii);
	 
	 double dr=realrstub[i]-r[i];
	 
	 MinvDt[2][2*ii+1]+=dr*tder;
	 MinvDt[3][2*ii+1]+=dr*zder;
	 
	 int itder=derivatives->getitdzcorr(i,ii);
	 int izder=derivatives->getiz0dzcorr(i,ii);
	 
	 int idr=dr/kr;
	 
	 iMinvDt[2][2*ii+1]+=((idr*itder)>>rcorrbits);
	 iMinvDt[3][2*ii+1]+=((idr*izder)>>rcorrbits);
	 
       }    
     }
   }
   



   double rinvseed=tracklet->rinvapprox();
   double phi0seed=tracklet->phi0approx();
   double tseed=tracklet->tapprox();
   double z0seed=tracklet->z0approx();

   double rinvseedexact=tracklet->rinv();
   double phi0seedexact=tracklet->phi0();
   double tseedexact=tracklet->t();
   double z0seedexact=tracklet->z0();



   double chisqseed=0.0;
   double chisqseedexact=0.0;

   double delta[12];
   double deltaexact[12];
   int idelta[12];

   for(unsigned int i=0;i<12;i++) {
    delta[i]=0.0;
    deltaexact[i]=0.0;
    idelta[i]=0;
   }

   int j=0;

   for(unsigned int i=0;i<n;i++) {

    if (i>=nlayers) {
     iphiresid[i]*=(t/ttabi);
     phiresid[i]*=(t/ttab);
     phiresidexact[i]*=(t/ttab);
    }

    idelta[j]=iphiresid[i];
    delta[j]=phiresid[i];
    if (fabs(phiresid[i])>0.2) {
     cout << getName()<<" WARNING too large phiresid: "
      <<phiresid[i]<<" "<<phiresidexact[i]<<endl;
    }
    assert(fabs(phiresid[i])<1.0);
    assert(fabs(phiresidexact[i])<1.0);
    deltaexact[j++]=phiresidexact[i];

    idelta[j]=izresid[i];
    delta[j]=zresid[i];
    deltaexact[j++]=zresidexact[i];

    chisqseed+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]); 
    chisqseedexact+=(deltaexact[j-2]*deltaexact[j-2]+
      deltaexact[j-1]*deltaexact[j-1]);
   }
   assert(j<=12);

   double drinv=0.0;
   double dphi0=0.0;
   double dt=0.0;
   double dz0=0.0;

   double drinvexact=0.0;
   double dphi0exact=0.0;
   double dtexact=0.0;
   double dz0exact=0.0;

   int idrinv=0;
   int idphi0=0;
   int idt=0;
   int idz0=0;


   double drinv_cov=0.0;
   double dphi0_cov=0.0;
   double dt_cov=0.0;
   double dz0_cov=0.0;

   double drinv_covexact=0.0;
   double dphi0_covexact=0.0;
   double dt_covexact=0.0;
   double dz0_covexact=0.0;



   for(unsigned int j=0;j<2*n;j++) {

    drinv-=MinvDt[0][j]*delta[j];
    dphi0-=MinvDt[1][j]*delta[j];
    dt-=MinvDt[2][j]*delta[j];
    dz0-=MinvDt[3][j]*delta[j];

    drinv_cov+=D[0][j]*delta[j];
    dphi0_cov+=D[1][j]*delta[j];
    dt_cov+=D[2][j]*delta[j];
    dz0_cov+=D[3][j]*delta[j];


    drinvexact-=MinvDt[0][j]*deltaexact[j];
    dphi0exact-=MinvDt[1][j]*deltaexact[j];
    dtexact-=MinvDt[2][j]*deltaexact[j];
    dz0exact-=MinvDt[3][j]*deltaexact[j];

    drinv_covexact+=D[0][j]*deltaexact[j];
    dphi0_covexact+=D[1][j]*deltaexact[j];
    dt_covexact+=D[2][j]*deltaexact[j];
    dz0_covexact+=D[3][j]*deltaexact[j];

    idrinv+=((iMinvDt[0][j]*idelta[j]));
    idphi0+=((iMinvDt[1][j]*idelta[j]));
    idt+=((iMinvDt[2][j]*idelta[j]));
    idz0+=((iMinvDt[3][j]*idelta[j]));

    /*

       double drinvtmp=MinvDt[0][j]*delta[j];
       double idrinvtmp=krinvpars*((iMinvDt[0][j]*idelta[j]+(1<<(fitrinvbitshift-1)))>>fitrinvbitshift);

       if (fabs(drinvtmp-idrinvtmp)>0.000001) {
       cout << "WARNING large fp vs. int difference for drinv "<<tracklet->layer()<<" "<<tracklet->disk()<<" "
       <<j<<" "<<drinvtmp<<" "
       <<idrinvtmp<<" "<<drinvtmp-idrinvtmp<<"  "<<delta[j]<<" "<<idelta[j]*kphiproj123<<" "<<iMinvDt[0][j]<<" "<<idelta[j]<<endl;
       }


       double dz0tmp=MinvDt[3][j]*delta[j];
       double idz0tmp=kz*((iMinvDt[3][j]*idelta[j])>>fitz0bitshift);

       if (fabs(dz0tmp-idz0tmp)>1.0) {
       cout << "WARNING large fp vs. int difference for dz0 "<<tracklet->layer()<<" "<<tracklet->disk()<<" "
       <<j<<" "<<dz0tmp<<" "
       <<idz0tmp<<" "<<dz0tmp-idz0tmp<<"  "<<delta[j]<<" "<<idelta[j]*kz<<endl;
       }

*/

    /*
       if (j%2==0) {
       cout << "j dt idt : "<<j<<" "<<MinvDt[2][j]*delta[j]<<" "<<((iMinvDt[2][j]*idelta[j])>>fittbitshift)*ktpars<<endl;
       if (j/2>=nlayers) {
       cout << "alpha : "<<alpha[j/2-nlayers]*rstub[j/2]*rstub[j/2]<<endl;
       cout << "MinvDt iMinvDt : "<<MinvDt[2][j]<<" "<<iMinvDt[2][j]*ktparsdisk/(1<<fittbitshift)/kphiproj123<<endl;
       cout << "delta idelta : "<<delta[j]<<" "<<idelta[j]*kphiproj123<<endl;
       } else {
       cout << "MinvDt iMinvDt : "<<MinvDt[2][j]<<" "<<iMinvDt[2][j]*ktpars/(1<<fittbitshift)/kphi1<<endl;
       cout << "delta idelta : "<<delta[j]<<" "<<idelta[j]*kphi1<<endl;
       }
       }
       */

    //cout <<  "------------------------------------"<<endl;
    //cout << "j delta idelta : "<<j<<" "<<delta[j]<<" "<<idelta[j]*kfactor[j]<<endl;
    //cout << "j dt idt : "<<j<<" "<<MinvDt[2][j]*delta[j]<<" "<<((iMinvDt[2][j]*idelta[j])>>fittbitshift)*ktpars<<endl;
    //cout << "kfactor kphiproj123 : "<<kfactor[j]<<" "<<kphiproj123<<endl;
    //cout << "j Minv iMinv : "<<j<<" "<<MinvDt[2][j]<<" "<<iMinvDt[2][j]<<" "<<iMinvDt[2][j]*ktpars/kphiproj123/(1<<fittbitshift)<<" "
    //   <<MinvDt[2][j]/(iMinvDt[2][j]*ktpars/kphiproj123/(1<<fittbitshift))<<endl;

    if (0&&j%2==0) {

     cout << "DUMPFITLINNEW1"<<" "<<j
      <<" "<<rinvseed
      <<" + "<<MinvDt[0][j]*delta[j]
      <<" "<<MinvDt[0][j]
      <<" "<<delta[j]*rstub[j/2]*10000
      <<endl;

     cout << "DUMPFITLINNEW2"<<" "<<j
      <<" "<<tracklet->fpgarinv().value()*krinvpars
      <<" + "<<((iMinvDt[0][j]*idelta[j]))*krinvpars/1024.0
      <<" "<<iMinvDt[0][j]*krinvpars/kphiprojdisk/1024.0
      <<" "<<idelta[j]*kphiproj123*rstub[j/2]*10000
      <<" "<<idelta[j]
      <<endl;

    }


   }

   double deltaChisqexact=drinvexact*drinv_covexact+
    dphi0exact*dphi0_covexact+
    dtexact*dt_covexact+
    dz0exact*dz0_covexact;


   int irinvseed=tracklet->fpgarinv().value();
   int iphi0seed=tracklet->fpgaphi0().value();

   int itseed=tracklet->fpgat().value();
   int iz0seed=tracklet->fpgaz0().value();

   int irinvfit=irinvseed+((idrinv+(1<<fitrinvbitshift))>>fitrinvbitshift);
   int iphi0fit=iphi0seed+(idphi0>>fitphi0bitshift);

   int itfit=itseed+(idt>>fittbitshift);

   int iz0fit=iz0seed+(idz0>>fitz0bitshift);

   double rinvfit=rinvseed-drinv;
   double phi0fit=phi0seed-dphi0;

   double tfit=tseed-dt;
   double z0fit=z0seed-dz0;

   double rinvfitexact=rinvseedexact-drinvexact;
   double phi0fitexact=phi0seedexact-dphi0exact;

   double tfitexact=tseedexact-dtexact;
   double z0fitexact=z0seedexact-dz0exact;

   double chisqfitexact=chisqseedexact+deltaChisqexact;

   ////////////// NEW CHISQ /////////////////////
   bool NewChisqDebug=false;
   double chisqfit=0.0;
   uint ichisqfit=0;

   double phifactor;
   double rzfactor;
   double iphifactor;
   double irzfactor;
   int k=0; // column index of D matrix

   if(NewChisqDebug){
    cout << "OG chisq:" << endl;
    cout << "drinv/cov = " << drinv << "/" << drinv_cov << endl;
    cout << "dphi0/cov = " << drinv << "/" << dphi0_cov << endl;
    cout << "dt/cov = " << drinv << "/" << dt_cov << endl;
    cout << "dz0/cov = " << drinv << "/" << dz0_cov << endl << endl;
    cout << "D[0][k]= ";
    for(unsigned int i=0;i<2*n;i++) {
     cout << D[0][i] << ", ";
    }
    cout << endl;
   }



   for(unsigned int i=0;i<n;i++) { // loop over stubs

    phifactor=rstub[k/2]*delta[k]/sigma[k]
     +D[0][k]*drinv+D[1][k]*dphi0+D[2][k]*dt+D[3][k]*dz0;


    iphifactor=kfactor[k]*rstub[k/2]*idelta[k]*(1<<chisqphifactbits)/sigma[k]
     -iD[0][k]*idrinv-iD[1][k]*idphi0-iD[2][k]*idt-iD[3][k]*idz0;

    //double kchisqphi=1.0/(1<<chisqphifactbits);
    //cout << "=================================================="<<endl;
    //cout << "*** old phi resid iresid : "<<rstub[k/2]*delta[k]/sigma[k]<<" "<<kfactor[k]*rstub[k/2]*idelta[k]*(1<<chisqphifactbits)/sigma[k]/(1<<chisqphifactbits)<<" rstub "<<rstub[k/2]<<endl;
    //cout << "phi k  D "<<k<<" "<<D[0][k]*drinv<<" "<<D[1][k]*dphi0<<" "<<D[2][k]*dt<<" "<<D[3][k]*dz0<<endl;
    //cout << "phi k iD "<<k<<" "<<iD[0][k]*idrinv*kchisqphi<<" "<<iD[1][k]*idphi0*kchisqphi<<" "<<iD[2][k]*idt*kchisqphi<<" "<<iD[3][k]*idz0*kchisqphi<<endl;
    //cout << "### new phi resid iresid : "<<phifactor<<" "<<iphifactor*kchisqphi<<endl;

    if(NewChisqDebug){
     cout << "delta[k]/sigma = " << delta[k]/sigma[k] << "  delta[k] = " << delta[k]  << endl;
     cout << "sum = " << phifactor-delta[k]/sigma[k] << "    drinvterm = " << D[0][k]*drinv << "  dphi0term = " << D[1][k]*dphi0 << "  dtterm = " << D[2][k]*dt << "  dz0term = " << D[3][k]*dz0 << endl;
     cout << "  phifactor = " << phifactor << endl;
    }

    chisqfit+=phifactor*phifactor;
    ichisqfit+=iphifactor*iphifactor/(1<<(2*chisqphifactbits-4));

    //cout << "i phi chisq :" << i << " " << phifactor*phifactor << " " << iphifactor*iphifactor/(1<<(2*chisqphifactbits-4))/16.0 << endl;

    k++;


    rzfactor=delta[k]/sigma[k]+D[0][k]*drinv+D[1][k]*dphi0+D[2][k]*dt+D[3][k]*dz0;


    irzfactor=kfactor[k]*idelta[k]*(1<<chisqzfactbits)/sigma[k]-iD[0][k]*idrinv-iD[1][k]*idphi0-iD[2][k]*idt-iD[3][k]*idz0;

    //double kchisqz=1.0/(1<<chisqzfactbits);
    //cout << "idrinv: "<<iD[0][k]<<" "<<idrinv<<endl;
    //cout << "idphi0: "<<iD[1][k]<<" "<<idphi0<<endl;
    //cout << "idt:    "<<iD[2][k]<<" "<<idt<<endl;
    //cout << "idz0:   "<<iD[3][k]<<" "<<idz0<<endl;

    //cout << "dt idt : "<<dt<<" "<<idt*ktpars/(1<<fittbitshift)<<endl;

    //cout << "old z resid : "<<delta[k]/sigma[k]<<" "<<kfactor[k]*idelta[k]/sigma[k]<<endl;
    //cout << "r-z k  D "<<k<<" "<<D[0][k]*drinv<<" "<<D[1][k]*dphi0<<" "<<D[2][k]*dt<<" "<<D[3][k]*dz0<<endl;
    //cout << "r-z k iD "<<k<<" "<<iD[0][k]*idrinv*kchisqz<<" "<<iD[1][k]*idphi0*kchisqz<<" "<<iD[2][k]*idt*kchisqz<<" "<<iD[3][k]*idz0*kchisqz<<endl;

    //cout << "new z resid: "<<rzfactor<<" "<<irzfactor*kchisqz<<endl;

    if(NewChisqDebug){
     cout << "delta[k]/sigma = " << delta[k]/sigma[k] << "  delta[k] = " << delta[k]  << endl;
     cout << "sum = " << rzfactor-delta[k]/sigma[k] << "    drinvterm = " << D[0][k]*drinv << "  dphi0term = " << D[1][k]*dphi0 << "  dtterm = " << D[2][k]*dt << "  dz0term = " << D[3][k]*dz0 << endl;
     cout << "  rzfactor = " << rzfactor << endl;
    }

    chisqfit+=rzfactor*rzfactor;
    ichisqfit+=irzfactor*irzfactor/(1<<(2*chisqzfactbits-4));

    //cout << "i z chisq   :" << i << " " << rzfactor*rzfactor << " " << irzfactor*irzfactor/(1<<(2*chisqzfactbits-4))/16.0 << endl;



    k++;
   }

   //cout <<"chisq ichisq : "<< chisqfit << " " << ichisqfit/16.0<<endl;


   if (writeChiSq) {
    static ofstream out("chisq.txt");
    out << asinh(itfit*ktpars)<<" "<<chisqfit << " " << ichisqfit/16.0<<endl;
   }

   // Divide by degrees of freedom
   // NOT ANYMORE! Raw chisquare!
   //chisqfit=chisqfit/(2*n-4);
   //chisqfitexact=chisqfitexact/(2*n-4);
   //ichisqfit=ichisqfit/(2*n-4);

   // Experimental strategy:
   //    if(ichisqfit < (1<<8));
   //    else if(ichisqfit < (1<<12)) ichisqfit = (1<<8)+(ichisqfit>>4);
   //    else if(ichisqfit < (1<<16)) ichisqfit = (1<<9)+(ichisqfit>>8);
   //    else if(ichisqfit < (1<<20)) ichisqfit = (1<<9)+(1<<8)+(ichisqfit>>12);
   //    else ichisqfit = (1<<10)-1;

   // Cap out at 15 bits -- Any larger is useless
   // Chisquare per DOF capped out at 11 bits, so 15 is an educated guess
   if(ichisqfit >= (1<<15)) {
    //cout << "CHISQUARE (" << ichisqfit << ") LARGER THAN 11 BITS!!" << endl;
    ichisqfit = (1<<15)-1;
   }

   // Eliminate lower bits to fit in 8 bits
   ichisqfit = ichisqfit>>7;
   // Probably redundant... enforce 8 bit cap
   if(ichisqfit >= (1<<8)) ichisqfit = (1<<8)-1;


   double phicrit=phi0fit-asin(0.5*rcrit*rinvfit);
   bool keep=(phicrit>phicritmin)&&(phicrit<phicritmax);

   
   if (!keep) {
     return;
   }
     
   tracklet->setFitPars(rinvfit,phi0fit,0.0,tfit,z0fit,chisqfit,
			rinvfitexact,phi0fitexact,0.0,tfitexact,
			z0fitexact,chisqfitexact,
			irinvfit,iphi0fit,0,itfit,iz0fit,ichisqfit);

  }

  void trackFitFake(Tracklet* tracklet){
    // to be replaced with a five-parameter fit; for now, copy tracklet parameters
    tracklet->setFitPars(tracklet->rinvapprox(),tracklet->phi0approx(),tracklet->d0approx(),tracklet->tapprox(),tracklet->z0approx(),0.,
       tracklet->rinv(),tracklet->phi0(),tracklet->d0(),tracklet->t(),
       tracklet->z0(),0.,
       tracklet->fpgarinv().value(),tracklet->fpgaphi0().value(),tracklet->fpgad0().value(),tracklet->fpgat().value(),tracklet->fpgaz0().value(),0);
  }

  std::vector<Tracklet*> orderedMatches(vector<FullMatchMemory*>& fullmatch) {

   std::vector<Tracklet*> tmp;



   std::vector<unsigned int> indexArray;
   for (unsigned int i=0;i<fullmatch.size();i++) {

     //checks that we have correct order
     if (fullmatch[i]->nMatches()>1) {
       for (unsigned int j=0;j<fullmatch[i]->nMatches()-1;j++){
	 assert(fullmatch[i]->getFPGATracklet(j)->TCID()<=fullmatch[i]->getFPGATracklet(j+1)->TCID());
       }
     }
     
     
    if(debug1 && fullmatch[i]->nMatches()!=0) cout<<"orderedMatches: "<<fullmatch[i]->getName()<<" "<< fullmatch[i]->nMatches()<<"\n";

    indexArray.push_back(0);
   }

   int bestIndex=-1;
   do {
    int bestTCID=(1<<16);
    bestIndex=-1;
    for (unsigned int i=0;i<fullmatch.size();i++) {
     if (indexArray[i]>=fullmatch[i]->nMatches()) {
      //skip as we were at the end
      continue;
     }
     int TCID=fullmatch[i]->getFPGATracklet(indexArray[i])->TCID();
     if (TCID<bestTCID) {
      bestTCID=TCID;
      bestIndex=i;
     }
    }
    if (bestIndex!=-1) {
     tmp.push_back(fullmatch[bestIndex]->getFPGATracklet(indexArray[bestIndex]));
     indexArray[bestIndex]++;
    }
   } while (bestIndex!=-1);

   for (unsigned int i=0;i<tmp.size();i++) {
    if (i>0) {
     //This allows for equal TCIDs. This means that we can e.g. have a track seeded
     //in L1L2 that projects to both L3 and D4. The algorithm will pick up the first hit and
     //drop the second
     if (tmp[i-1]->TCID()>tmp[i]->TCID()){
      cout << "Wrong TCID ordering in "<<getName()<<" : "
       << tmp[i-1]->TCID()<<" "<<tmp[i]->TCID()<<endl;
      //assert(0);
     }
    }
   }

   return tmp;
  }

  void execute() {

   // merge
   std::vector<Tracklet*> matches1=orderedMatches(fullmatch1_);
   std::vector<Tracklet*> matches2=orderedMatches(fullmatch2_);
   std::vector<Tracklet*> matches3=orderedMatches(fullmatch3_);
   std::vector<Tracklet*> matches4=orderedMatches(fullmatch4_);
   
   if (debug1&&(matches1.size()+matches2.size()+matches3.size()+matches4.size())>0) {
    for (unsigned int i=0;i<fullmatch1_.size();i++) {
     cout << fullmatch1_[i]->getName()<<" "<<fullmatch1_[i]->nMatches()<<endl;
    }
    cout << getName()<<"["<<iSector_<<"] matches : "<<matches1.size()<<" "<<matches2.size()<<" "
     <<matches3.size()<<" "<<matches4.size()<<" "<<endl;
   }

   //New trackfit
   unsigned int indexArray[4];
   for (unsigned int i=0;i<4;i++) {
    indexArray[i]=0;
   }

   int countAll=0;
   int countFit=0;

   Tracklet* bestTracklet=0;
   do {
    countAll++;
    bestTracklet=0;

    if (indexArray[0]<matches1.size()) {
     if (bestTracklet==0) {
      bestTracklet=matches1[indexArray[0]];
     } else {
      if (matches1[indexArray[0]]->TCID()<bestTracklet->TCID())
       bestTracklet=matches1[indexArray[0]];
     }
    }

    if (indexArray[1]<matches2.size()) {
     if (bestTracklet==0) {
      bestTracklet=matches2[indexArray[1]];
     } else {
      if (matches2[indexArray[1]]->TCID()<bestTracklet->TCID())
       bestTracklet=matches2[indexArray[1]];
     }
    }

    if (indexArray[2]<matches3.size()) {
     if (bestTracklet==0) {
      bestTracklet=matches3[indexArray[2]];
     } else {
      if (matches3[indexArray[2]]->TCID()<bestTracklet->TCID())
       bestTracklet=matches3[indexArray[2]];
     }
    }

    if (indexArray[3]<matches4.size()) {
     if (bestTracklet==0) {
      bestTracklet=matches4[indexArray[3]];
     } else {
      if (matches4[indexArray[3]]->TCID()<bestTracklet->TCID())
       bestTracklet=matches4[indexArray[3]];
     }
    }

    if (bestTracklet==0) break;

    //Counts total number of matched hits
    int nMatches=0;

    //Counts unique hits in each layer
    int nMatchesUniq=0;
    bool match=false;

    if (indexArray[0]<matches1.size()) {
     while (matches1[indexArray[0]]==bestTracklet && indexArray[0]<matches1.size()) {
      indexArray[0]++;
      nMatches++;
      match=true;
     }
    }

    if (match) nMatchesUniq++;
    match=false;
    
    if (indexArray[1]<matches2.size()) {
     while (matches2[indexArray[1]]==bestTracklet && indexArray[1]<matches2.size()) {
      indexArray[1]++;
      nMatches++;
      match=true;
     }
    }

    if (match) nMatchesUniq++;
    match=false;

    if (indexArray[2]<matches3.size()) {
     while (matches3[indexArray[2]]==bestTracklet && indexArray[2]<matches3.size()) {
      indexArray[2]++;
      nMatches++;
      match=true;
     }
    }

    if (match) nMatchesUniq++;
    match=false;

    if (indexArray[3]<matches4.size()) {
     while (matches4[indexArray[3]]==bestTracklet && indexArray[3]<matches4.size()) {
      indexArray[3]++;
      nMatches++;
      match=true;
     }
    }

    if (match) nMatchesUniq++;

    if(debug1) cout<<getName()<<" : nMatches = "<<nMatches<<" "<<asinh(bestTracklet->t())<<"\n";

    std::vector<std::pair<Stub*,L1TStub*>> trackstublist;
    std::vector<std::pair<int,int>> stubidslist;
    if ((bestTracklet->getISeed()>=8 && nMatchesUniq>=1)||nMatchesUniq>=2) { //For seeds index >=8 (triplet seeds), there are three stubs associated from start.
      countFit++;
      if (fakefit) {
        trackFitFake(bestTracklet);
      } else {
        trackFitNew(bestTracklet,trackstublist,stubidslist);
      }
      if(RemovalType=="merge") {
        trackfit_->addStubList(trackstublist);
        trackfit_->addStubidsList(stubidslist);
        trackfit_->addTrack(bestTracklet);
      }
      else if (bestTracklet->fit()){
        assert(trackfit_!=0);
        if (writeSeeds) {
          ofstream fout("seeds.txt", ofstream::app);
          fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << bestTracklet->getISeed() << endl;
          fout.close();
        }
        trackfit_->addTrack(bestTracklet);
      }
    }

   } while (bestTracklet!=0);

   if (writeFitTrack) {
    static ofstream out("fittrack.txt");
    out<<getName()<<" "<<countAll<<" "<<countFit<<endl;
   }

  }


 private:

  vector<TrackletParametersMemory*> seedtracklet_;
  vector<FullMatchMemory*> fullmatch1_;
  vector<FullMatchMemory*> fullmatch2_;
  vector<FullMatchMemory*> fullmatch3_;
  vector<FullMatchMemory*> fullmatch4_;

  TrackFitMemory* trackfit_;

};

#endif
