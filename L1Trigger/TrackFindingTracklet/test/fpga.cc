// ROOT includes
#include "TMath.h"
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TBranch.h>
#include <TSystem.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TLegend.h>
#include <TLatex.h>

#include "../interface/Constants.h"
#include "../interface/IMATH_TrackletCalculator.h"

#include "../interface/slhcevent.h"

#include "../interface/Sector.h"
#include "../interface/Cabling.h"
#include "../interface/FPGAWord.h"
#include "../interface/CPUTimer.h"
#include "../interface/StubVariance.h"

#include "../interface/GlobalHistTruth.h"
#include "../interface/HistImp.h"

#ifdef IMATH_ROOT
TFile* var_base::h_file_=0;
bool   var_base::use_root = false;
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>


// Include file to define ROOT-Tree
// --------------------------------
#ifdef USEROOT
#include "FPGAEvent.h"
#endif
// --------------------------------

int main(const int argc, const char** argv)
{
  cout << "dphisectorHG = "<<dphisectorHG<<endl;

  krinvpars = TrackletCalculator::ITC_L1L2.rinv_final.get_K();
  kphi0pars = TrackletCalculator::ITC_L1L2.phi0_final.get_K();
  kd0pars   = kd0;
  ktpars    = TrackletCalculator::ITC_L1L2.t_final.get_K();
  kz0pars   = TrackletCalculator::ITC_L1L2.z0_final.get_K();

  krdisk = kr;
  kzpars = kz;  
  krprojshiftdisk = TrackletCalculator::ITC_L1L2.rD_0_final.get_K();

  //those can be made more transparent...
  kphiproj123=kphi0pars*4;
  kphiproj456=kphi0pars/2;
  kzproj=kz;
  kphider=krinvpars*(1<<phiderbitshift);
  kzder=ktpars*(1<<zderbitshift);
  kphiprojdisk=kphi0pars*4.0;
  krprojderdiskshift=krprojderdisk*(1<<rderdiskbitshift);
  krprojderdisk=(1.0/ktpars)/(1<<t2bits);

  cout << "=========================================================" << endl;
  cout << "Conversion factors for global coordinates:"<<endl;
  cout << "z    kz            = "<< kz <<endl;
  cout << "r    kr            = "<< kr <<endl;
  cout << "phi  kphi1         = "<< kphi1 <<endl;
  cout << "=========================================================" << endl;
  cout << "Conversion factors for track(let) parameters:"<<endl;
  cout << "rinv krinvpars     = "<< krinvpars <<endl;
  cout << "phi0 kphi0pars     = "<< kphi0pars <<endl;
  cout << "d0   kd0pars       = "<< kd0pars <<endl;
  cout << "t    ktpars        = "<< ktpars <<endl;
  cout << "z0   kz0pars       = "<< kzpars <<endl;
  cout << "=========================================================" << endl;
  cout << "rinvbitshift = "<<rinvbitshift<<endl;
  cout << "phi0bitshift = "<<phi0bitshift<<endl;
  cout << "d0bitshift   = "<<"???"<<endl;
  cout << "tbitshift    = "<<tbitshift<<endl;
  cout << "z0bitshift   = "<<z0bitshift<<endl;
  cout << endl;
  cout << "=========================================================" << endl;

#include "../plugins/WriteInvTables.icc"
#include "../plugins/WriteDesign.icc"
  
  using namespace std;
  if (argc<4)
    cout << "Need to specify the input ascii file and the number of events to run on and if you want to filter on MC truth" << endl;

  HistImp* histimp=new HistImp;
  histimp->init();
  histimp->bookLayerResidual();
  histimp->bookDiskResidual();
  histimp->bookTrackletParams();
  histimp->bookSeedEff();
  
  GlobalHistTruth::histograms()=histimp;
  
  int nevents = atoi(argv[2]);

  int selectmu = atoi(argv[3]);

  assert((selectmu==0)||(selectmu==1));

  ifstream infile;
  istream* in = &cin;
  if(strcmp(argv[1],"stdin")){
    infile.open(argv[1]);
    in = &infile;
  }

  ofstream outres;
  if (writeResEff) outres.open("trackres.txt");

  ofstream outeff;
  if (writeResEff) outeff.open("trackeff.txt");

  ofstream outpars;
  if (writePars) outpars.open("trackpars.txt");

  //ofstream out;
  //out.open("evlist_skim.txt"); 

//Open file to hold ROOT-Tree
// --------------------------
#ifdef USEROOT
  TFile  *hfile = new TFile("myTest.root","RECREATE","Simple ROOT Ntuple"); 
  TTree *trackTree = new TTree("FPGAEvent","L1Track Tree");
  FPGAEvent *fpgaEvent = new FPGAEvent;  
  fpgaEvent->reset();
  trackTree->Branch("Event",&fpgaEvent);
#endif
// --------------------------



// Define Sectors (boards)	 
  Sector** sectors=new Sector*[NSector];

  Cabling cabling;

  cabling.init("../data/calcNumDTCLinks.txt","../data/modules_T5v3_27SP_nonant_tracklet.dat");


  
  for (unsigned int i=0;i<NSector;i++) {
    sectors[i]=new Sector(i);
  }  


  cout << "Will read memory modules file"<<endl;

  string memfile="../data/memorymodules_"+geomext+".dat";
  ifstream inmem(memfile.c_str());
  assert(inmem.good());

  while (inmem.good()){
    string memType, memName, size;
    inmem >>memType>>memName>>size;
    if (!inmem.good()) continue;
    if (writetrace) {
      cout << "Read memory: "<<memType<<" "<<memName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addMem(memType,memName);
    }
    
  }


  cout << "Will read processing modules file"<<endl;

  string procfile="../data/processingmodules_"+geomext+".dat";
  ifstream inproc(procfile.c_str());
  assert(inproc.good());

  while (inproc.good()){
    string procType, procName;
    inproc >>procType>>procName;
    if (!inproc.good()) continue;
    if (writetrace) {
      cout << "Read process: "<<procType<<" "<<procName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addProc(procType,procName);
    }
    
  }


  cout << "Will read wiring information"<<endl;

  string wirefile="../data/wires_"+geomext+".dat";
  ifstream inwire(wirefile.c_str());
  assert(inwire.good());


  while (inwire.good()){
    string line;
    getline(inwire,line);
    if (!inwire.good()) continue;
    if (writetrace) {
      cout << "Line : "<<line<<endl;
    }
    stringstream ss(line);
    string mem,tmp1,procin,tmp2,procout;
    ss>>mem>>tmp1>>procin;
    //cout <<"procin : "<<procin<<endl;
    if (procin=="output=>") {
      procin="";
      ss>>procout;
    }
    else{
      ss>>tmp2>>procout;
    }

    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addWire(mem,procin,procout);
    }
  
  }

  std::map<string,vector<int> > dtclayerdisk;

  ifstream indtc("../data/dtclinklayerdisk.dat");
  assert(indtc.good());
  string dtc;
  indtc >> dtc;
  while (indtc.good()){
    vector<int> tmp;
    dtclayerdisk[dtc]=tmp;
    int layerdisk;
    indtc >> layerdisk;
    while (layerdisk>0) {
      dtclayerdisk[dtc].push_back(layerdisk);
      indtc >> layerdisk;
    }
    indtc >> dtc;
  }
  
  ofstream skimout;
  if (skimfile!="") skimout.open(skimfile.c_str());

  CPUTimer readTimer;
  CPUTimer cleanTimer;
  CPUTimer addStubTimer;
  CPUTimer VMRouterTimer;  
  CPUTimer TETimer;
  CPUTimer TEDTimer;
  CPUTimer TRETimer;
  CPUTimer TCTimer;
  CPUTimer TCDTimer;
  CPUTimer PRTimer;
  CPUTimer METimer;
  CPUTimer MCTimer;
  CPUTimer MPTimer;
  CPUTimer FTTimer;
  CPUTimer PDTimer;

  if (writeSeeds) {
    ofstream fout("seeds.txt", ofstream::out);
    fout.close();
  }

  bool first=true;

  for (int eventnum=0;eventnum<nevents&&!in->eof();eventnum++){
    
    readTimer.start();
    SLHCEvent ev(*in);
    //ev.read(*in);  
    //ev.read(*in);  
    //ev.read(*in);  
    //ev.read(*in);  
    if (allSector) {
      ev.allSector();
    }
    readTimer.stop();

    GlobalHistTruth::event()=&ev;

    
    L1SimTrack simtrk;


// setup ROOT Tree and Add Monte Carlo tracks to the ROOT-Tree Event
// -----------------------------------------------------------------
#ifdef USEROOT
    fpgaEvent->reset();
    fpgaEvent->nevt = eventnum;
    for(int nst=0; nst<ev.nsimtracks(); nst++) {
      simtrk = ev.simtrack(nst);
      FPGAEventMCTrack *mcTrack = new FPGAEventMCTrack(simtrk.type(),simtrk.pt(),simtrk.eta(),simtrk.phi(),simtrk.vx(),simtrk.vy(),simtrk.vz());
      fpgaEvent->mcTracks.push_back(*mcTrack);
    }
#endif
// ------------------------------------------------------------------	 

    
    if (selectmu==1) {

      //skip if no simtracks - new (180424) ascii files only print out simtrack info if we have clusters
      if (ev.nsimtracks()==0) {
      	eventnum--;
	//cout <<"Skip event"<<endl;
	continue;
      }

      
      simtrk=ev.simtrack(0);

      if (debug1) {
	cout <<"nstub simtrkid pt phi eta t vz:"<<ev.nstubs()<<" "<<simtrk.trackid()<<" "<<simtrk.pt()<<" "<<simtrk.phi()<<" "
	     <<simtrk.eta()<<" "
	     <<sinh(simtrk.eta())<<" "
	     <<simtrk.vz()<<endl;
      }
      
      //double phisector=simtrk.phi();
      //while(phisector<0.0) phisector+=3.141592/4.5;
      //while(phisector>3.141592/4.5) phisector-=3.141592/4.5;
      
      bool good=(fabs(simtrk.pt())>2.0
		 //&&fabs(simtrk.pt())<3.0
                 &&fabs(simtrk.vz())<15.0
		 //&&simtrk.eta()>0.0
		 //&&fabs(fabs(simtrk.eta())-0.0)>1.6
		 //&&fabs(fabs(simtrk.eta())-0.0)<1.9
		 &&fabs(fabs(simtrk.eta())-0.0)<2.4
		 //&&fabs(fabs(simtrk.eta())-0.0)>1.8
		 //&&fabs(fabs(simtrk.phi())-0.05)<0.05
		 //&&fabs(simtrk.eta()-0.0)>1.6
		 //&&fabs(simtrk.eta()-1.5)<0.2
		 //&&fabs(phisector-0.61)<0.03
		 //&&fabs(fabs(simtrk.eta())-1.4)<0.1
                 //&&fabs(simtrk.d0())<1.0
		 );


      if (!good) {
	eventnum--;
	//cout <<"Skip event"<<endl;
	continue;
      }

      if (skimfile!="") ev.write(skimout);
 
    } 

    if (writeSeeds) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << "======== Event " << eventnum << " ========" << endl;
      for(unsigned nst=0; nst<ev.nsimtracks(); nst++) {
        const L1SimTrack &simtrk = ev.simtrack(nst);
        fout << "SimTrk " << simtrk.pt() << " " << simtrk.eta() << " " << simtrk.phi() << " " << simtrk.d0() << " ";

        vector<string> hitPattern;
        for(int i=0; i<ev.nstubs(); i++) {
          const L1TStub stub = ev.stub(i);
          if (!stub.tpmatch(simtrk.trackid()))
            continue;
          if (stub.layer() < 999) {
            switch (stub.layer()) {
              case 0: hitPattern.push_back("L1"); break;
              case 1: hitPattern.push_back("L2"); break;
              case 2: hitPattern.push_back("L3"); break;
              case 3: hitPattern.push_back("L4"); break;
              case 4: hitPattern.push_back("L5"); break;
              case 5: hitPattern.push_back("L6"); break;
              default: cout << "Stub layer: " << stub.layer() << endl; assert(0);
            }
          }
          else {
            string d = (stub.isPSmodule() ? "D" : "d");
            switch (abs(stub.disk())) {
              case 1: hitPattern.push_back(d+"1"); break;
              case 2: hitPattern.push_back(d+"2"); break;
              case 3: hitPattern.push_back(d+"3"); break;
              case 4: hitPattern.push_back(d+"4"); break;
              case 5: hitPattern.push_back(d+"5"); break;
              default: cout << "Stub disk: " << stub.disk() << endl; assert(0);
            }
          }
        }
        bool (*compare)(const string &, const string &) = [](const string &a, const string &b) -> bool {
          if (a.at(0) == 'L' && b.at(0) == 'D')
            return true;
          else if (a.at(0) == 'D' && b.at(0) == 'L')
            return false;
          else
            return a.at(1) < b.at(1);
        };
        sort(hitPattern.begin(), hitPattern.end(), compare);
        hitPattern.erase(unique(hitPattern.begin(), hitPattern.end()), hitPattern.end());
        for (const auto &stub : hitPattern)
          fout << stub;
        if (hitPattern.empty())
          fout << "XX";
        fout << endl;
      }
      fout.close();
    }

    if (writeVariance) {
      StubVariance variance(ev);
    }

    cout <<"Process event: "<<eventnum<<" with "<<ev.nstubs()<<" stubs and "<<ev.nsimtracks()<<" simtracks"<<endl;

    std::vector<Track*> tracks;

    int nlayershit=0;


    
// Processesing done in FPGA.icc
#include "../plugins/FPGA.icc"

// Block for producing ROOT-Tree
// ------------------------------
#ifdef USEROOT
#include "FPGATree.icc"
#endif
// ------------------------------

    if (writeResEff) {
      outeff << simtrk.pt()*simtrk.trackid()/fabs(simtrk.trackid())<<" "<<simtrk.eta()
	     <<" "<<simtrk.phi();
      if (match) outeff << " 1"<<endl;
       else outeff << " 0"<<endl;
    }

    if (writeMatchEff) {
      static ofstream out("matcheff.txt");
      int nsim=0;
      for(unsigned int isimtrack=0;isimtrack<ev.nsimtracks();isimtrack++){
        L1SimTrack simtrack=ev.simtrack(isimtrack);
        if (simtrack.pt()<2.0) continue;
        if (fabs(simtrack.eta())>2.4) continue;
        if (fabs(simtrack.vz())>15.0) continue;
        if (hypot(simtrack.vx(),simtrack.vy())>0.1) continue;
        bool electron=(abs(simtrack.type())==11);
        bool muon=(abs(simtrack.type())==13);
        bool pion=(abs(simtrack.type())==211);
        bool kaon=(abs(simtrack.type())==321);
        bool proton=(abs(simtrack.type())==2212);
        if (!(electron||muon||pion||kaon||proton)) continue;
        int nlayers=0;
        int ndisks=0;
        int simeventid=simtrack.eventid();
        int simtrackid=simtrack.trackid();
        ev.layersHit(simtrackid,nlayers,ndisks);
        //cout << "Simtrack id : "<<simtrackid<<" "<<nlayers<<" "<<ndisks<<endl;                                                
        if (nlayers+ndisks<4) continue;
	nsim++;
        bool eff=false;
        bool effloose=false;
	//int layerdisk=0;
	int itrackmatch=-1;
        for(unsigned int itrack=0;itrack<tracks.size();itrack++) {
          if (tracks[itrack]->duplicate()) continue;
          std::vector<L1TStub*> stubs=tracks[itrack]->stubs();
          //cout << "Track "<<itrack<<" with stubs with simtrackids :";

          unsigned int nmatch=0;
	  //layerdisk=0;
          for(unsigned int istub=0;istub<stubs.size();istub++){
            if (stubs[istub]->tpmatch(simtrackid)) {
              nmatch++;
            } else {
	      if (stubs[istub]->layer()<999) {
		//layerdisk=stubs[istub]->layer()+1;
	      } else {
		//layerdisk=-abs(stubs[istub]->disk());
	      }
	    }
          }

          if (nmatch==stubs.size()) {
	    eff=true;
	    itrackmatch=itrack;
	  }
          if (nmatch>=stubs.size()-1) {
	    effloose=true;
	    if (!eff) itrackmatch=itrack;
	  }

        }
	double dpt=-999;
	double dphi=-999;
	double deta=-999;
	double dz0=-999;
	int q=1;
	if (simtrack.type()==11||simtrack.type()==13||
	    simtrack.type()==-211||simtrack.type()==-321||simtrack.type()==-2212){
	  q=-1;
	}

	if (itrackmatch>=0) {
	  dpt=tracks[itrackmatch]->pt()-q*simtrack.pt();
	  dphi=tracks[itrackmatch]->phi0()-simtrack.phi();
	  if (dphi>0.5*two_pi) dphi-=two_pi;
	  if (dphi<-0.5*two_pi) dphi+=two_pi;
	  deta=tracks[itrackmatch]->eta()-simtrack.eta();
	  dz0=tracks[itrackmatch]->z0()-simtrack.vz();
	  //cout <<" z0 "<<tracks[itrackmatch]->z0()<<" "<<simtrack.vz()<<endl;
	}

	//cout << "dpt dphi deta dz0 "<<dpt<<" "<<dphi<<" "<<deta<<" "<<dz0<<endl;
	
        out <<eventnum<<" "<<simeventid<<" "<<simtrackid<<" "<<simtrack.type()<<" "
            <<simtrack.pt()<<" "<<simtrack.eta()<<" "<<simtrack.phi()<<" "
            <<simtrack.vx()<<" "<<simtrack.vy()<<" "<<simtrack.vz()<<" "
	    <<eff<<" "<<effloose<<" "
	    <<dpt<<" "<<dphi<<" "<<deta<<" "<<dz0
	    <<endl;
      }
      //cout << "nsim : "<<nsim<<endl;
    }

    
    
// Clean up 

    //cout << "Duplicates : ";
    int ntrack=0;
    for(unsigned int l=0;l<tracks.size();l++) {
      //  cout <<tracks[l].duplicate()<<" ";
      if (writePars) {
	double phi=tracks[l]->iphi0()*kphi0pars+tracks[l]->sector()*two_pi/NSector;
	if (phi>0.5*two_pi) phi-=two_pi;
	double phisec=phi-two_pi;
	while (phisec<0.0) phisec+=two_pi/NSector;
	outpars  <<tracks[l]->duplicate()<<" "<<asinh(tracks[l]->it()*ktpars)<<" "
		 <<phi<<" "<<tracks[l]->iz0()*kz<<" "<<phisec/(two_pi/NSector)<<" "
		 <<tracks[l]->irinv()*krinvpars<<endl;
      }   	
      if (!tracks[l]->duplicate()) {
	//cout << "FPGA Track pt, eta, phi, z0, chi2 = " 
	//   << tracks[l]->pt() << " " << tracks[l]->eta() << " " << tracks[l]->phi0() << " " << tracks[l]->z0() << " " << tracks[l]->chisq() 
	//   << " seed " << tracks[l]->seed() << " duplicate " << tracks[l]->duplicate() << endl;
	//cout << " ---------- not duplicate" << endl;
	//cout << "tapprox "<<tracks[l].eta()<<endl;
	ntrack++;
	//cout << "eta = "<<tracks[l].eta()<<endl;
      }
    }
    //cout << endl;
    
    cout << "Number layers/disks hit = "<<nlayershit<<" number of found tracks : "<<tracks.size()
	 <<" unique "<<ntrack<<endl;


  // dump what was found   
//	 printf("Track Parameters: \n");
//	 for(std::vector<Track*>::iterator trk=tracks.begin(); trk!=tracks.end(); trk++){
//	   printf("irinv = %i \n", (*trk)->irinv() );
//		 printf("iphi0 = %i \n", (*trk)->iphi0() );
//		 printf("iz0   = %i \n", (*trk)->iz0() );
//		 printf("it    = %i \n", (*trk)->it() );
//		 printf("stubID=");
//		 std::map<int, int> stubs = (*trk)->stubID();
//		 for(std::map<int, int>::iterator sb=stubs.begin(); sb!=stubs.end(); sb++) printf(" %i -- %i ",sb->first,sb->second);
//		 printf("\n");
//		 printf("dup   = %i\n \n", (*trk)->duplicate());
//		 printf("chisq   = %f\n \n", (*trk)->chisq());
//	 } 




    first=false;

  }

  if (writeCabling) {
    cabling.writephirange();
  }
  
  cout << "Process             Times called   Average time (ms)      Total time (s)"<<endl;
  cout << "Reading               "
       <<setw(10)<<readTimer.ntimes()
       <<setw(20)<<setprecision(3)<<readTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<readTimer.tottime()<<endl;
  cout << "Cleaning              "
       <<setw(10)<<cleanTimer.ntimes()
       <<setw(20)<<setprecision(3)<<cleanTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<cleanTimer.tottime()<<endl;
  cout << "Add Stubs             "
       <<setw(10)<<addStubTimer.ntimes()
       <<setw(20)<<setprecision(3)<<addStubTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<addStubTimer.tottime()<<endl;
  cout << "VMRouter              "
       <<setw(10)<<VMRouterTimer.ntimes()
       <<setw(20)<<setprecision(3)<<VMRouterTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<VMRouterTimer.tottime()<<endl;
  cout << "TrackletEngine        "
       <<setw(10)<<TETimer.ntimes()
       <<setw(20)<<setprecision(3)<<TETimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<TETimer.tottime()<<endl;
  cout << "TrackletEngineDisplaced"
       <<setw(10)<<TEDTimer.ntimes()
       <<setw(20)<<setprecision(3)<<TEDTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<TEDTimer.tottime()<<endl;
  cout << "TripletEngine         "
       <<setw(10)<<TRETimer.ntimes()
       <<setw(20)<<setprecision(3)<<TRETimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<TRETimer.tottime()<<endl;
  cout << "TrackletCalculator    "
       <<setw(10)<<TCTimer.ntimes()
       <<setw(20)<<setprecision(3)<<TCTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<TCTimer.tottime()<<endl;
  cout << "TrackletCalculatorDisplaced"
       <<setw(10)<<TCDTimer.ntimes()
       <<setw(20)<<setprecision(3)<<TCDTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<TCDTimer.tottime()<<endl;
  cout << "ProjectionRouter      "
       <<setw(10)<<PRTimer.ntimes()
       <<setw(20)<<setprecision(3)<<PRTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<PRTimer.tottime()<<endl;
  cout << "MatchEngine           "
       <<setw(10)<<METimer.ntimes()
       <<setw(20)<<setprecision(3)<<METimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<METimer.tottime()<<endl;
  cout << "MatchCalculator       "
       <<setw(10)<<MCTimer.ntimes()
       <<setw(20)<<setprecision(3)<<MCTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<MCTimer.tottime()<<endl;
  cout << "MatchProcessor        "
       <<setw(10)<<MPTimer.ntimes()
       <<setw(20)<<setprecision(3)<<MPTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<MPTimer.tottime()<<endl;
  cout << "FitTrack              "
       <<setw(10)<<FTTimer.ntimes()
       <<setw(20)<<setprecision(3)<<FTTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<FTTimer.tottime()<<endl;
  cout << "PurgeDuplicate        "
       <<setw(10)<<PDTimer.ntimes()
       <<setw(20)<<setprecision(3)<<PDTimer.avgtime()*1000.0
       <<setw(20)<<setprecision(3)<<PDTimer.tottime()<<endl;


  if (skimfile!="") skimout.close();

  histimp->close();
  
// Write and Close ROOT-Tree  
// -------------------------
#ifdef USEROOT
   hfile->Write();
   hfile->Close();
#endif
// --------------------------  

  for (unsigned int i=0;i<NSector;i++)
    delete sectors[i];

} 
