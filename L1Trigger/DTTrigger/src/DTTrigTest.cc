//-------------------------------------------------
//
/**  \class DTTrigTest
 *
 *   EDAnalyzer that generates a rootfile useful
 *   for L1-DTTrigger debugging and performance 
 *   studies
 *
 *
 *   $Date: 2006/09/18 10:47:15 $
 *   $Revision: 1.1 $
 *
 *   \author C. Battilana
 */
//
//--------------------------------------------------

// This class's header
#include "L1Trigger/DTTrigger/interface/DTTrigTest.h"

// Framework headers
#include "FWCore/Framework/interface/ESHandle.h"

// Trigger and DataFormats headers
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// ROOT headers 
#include "TROOT.h"

// Collaborating classes
#include <CLHEP/Vector/LorentzVector.h>

// C++ headers
#include <iostream>
#include <math.h>

using namespace std;



const double DTTrigTest::myTtoTDC = 32./25.;

DTTrigTest::DTTrigTest(const ParameterSet& pset){ 

  debug= pset.getUntrackedParameter<bool>("debug");
  string outputfile = pset.getUntrackedParameter<string>("outputFileName");
  if (debug == true) cout << "[DTTrigTest] Creating rootfile " <<  outputfile <<endl;
  f = new TFile(outputfile.c_str(),"RECREATE");
  theTree = new TTree("h1","GMT",0);
  bool globaldelay = pset.getUntrackedParameter<bool>("globalSync");
  double syncdelay = pset.getUntrackedParameter<double>("syncDelay");
  stringstream myos;
  myos << syncdelay;
  if (globaldelay) {
    if (debug == true) cout << "[DTTrigTest] Using same synchronization for all chambers:" << endl;
    MyTrig = new DTTrig();
    double ftdelay = pset.getUntrackedParameter<double>("globalSyncValue");
    MyTrig->config()->setParam("Programmable Dealy",myos.str());
    MyTrig->config()->setParamValue("BTI setup time","psetdelay",ftdelay*myTtoTDC);
    if (debug == true) cout << "[DTTrigTest] Delay set to " << ftdelay << " ns (as set in parameterset)" << endl; 
  }
  else {
    if (debug == true) cout << "[DTTrigTest] Using chamber by chamber synchronization" << endl;
    MyTrig = new DTTrig(pset.getUntrackedParameter<ParameterSet>("L1DTFineSync"),myos.str());
  }
  //  MyTrig->config()->setParam("Debugging level","fullTRACO");
  if (debug == true) cout << "[DTTrigTest] Constructor executed!!!" << endl;

}

DTTrigTest::~DTTrigTest(){ 

  delete MyTrig;
  delete f;
  if (debug == true) cout << "[DTTrigTest] Destructor executed!!!" << endl;

}

void DTTrigTest::endJob(){

  if (debug == true) cout << "[DTTrigTest] Writing Tree and Closing File" << endl;
  theTree->Write();
  delete theTree;
  f->Close();

}

void DTTrigTest::beginJob(const EventSetup & iEventSetup){   
    
  MyTrig->createTUs(iEventSetup);
  if (debug == true) cout << "[DTTrigTest] TU's Created" << endl;
  
  // BOOKING of the tree's varables
  // GENERAL block branches
  theTree->Branch("Run",&runn,"Run/I");
  theTree->Branch("Event",&eventn,"Event/I");
  theTree->Branch("Weight",&weight,"Weight/F");  
  // GEANT block branches
  theTree->Branch("Ngen",&ngen,"Ngen/I");
  theTree->Branch("Pxgen",pxgen,"Pxgen[Ngen]/F");
  theTree->Branch("Pygen",pygen,"Pygen[Ngen]/F");
  theTree->Branch("Pzgen",pzgen,"Pzgen[Ngen]/F");
  theTree->Branch("Ptgen",ptgen,"Ptgen[Ngen]/F");
  theTree->Branch("Etagen",etagen,"Etagen[Ngen]/F");
  theTree->Branch("Phigen",phigen,"Phigen[Ngen]/F");
  theTree->Branch("Chagen",chagen,"Chagen[Ngen]/I");
  theTree->Branch("Vxgen",vxgen,"Vxgen[Ngen]/F");
  theTree->Branch("Vygen",vygen,"Vygen[Ngen]/F");
  theTree->Branch("Vzgen",vzgen,"Vzgen[Ngen]/F");
  // L1MuDTBtiChipS block
  theTree->Branch("Nbti",&nbti,"Nbti/I");
  theTree->Branch("bwh",bwh,"bwh[Nbti]/I"); 
  theTree->Branch("bstat",bstat,"bstat[Nbti]/I");    
  theTree->Branch("bsect",bsect,"bsect[Nbti]/I");  
  theTree->Branch("bsl",bsl,"bsl[Nbti]/I");
  theTree->Branch("bnum",bnum,"bnum[Nbti]/I");
  theTree->Branch("bbx",bbx,"bbx[Nbti]/I");
  theTree->Branch("bcod",bcod,"bcod[Nbti]/I");
  theTree->Branch("bk",bk,"bk[Nbti]/I");
  theTree->Branch("bx",bx,"bx[Nbti]/I");
  theTree->Branch("bposx",bposx,"bposx[Nbti]/F");
  theTree->Branch("bposy",bposy,"bposy[Nbti]/F");
  theTree->Branch("bposz",bposz,"bposz[Nbti]/F");
  theTree->Branch("bdirx",bdirx,"bdirx[Nbti]/F");
  theTree->Branch("bdiry",bdiry,"bdiry[Nbti]/F");
  theTree->Branch("bdirz",bdirz,"bdirz[Nbti]/F");
  // L1MuDTTracoChipS block
  theTree->Branch("Ntraco",&ntraco,"Ntraco/I");
  theTree->Branch("twh",twh,"twh[Ntraco]/I"); 
  theTree->Branch("tstat",tstat,"tstat[Ntraco]/I");    
  theTree->Branch("tsect",tsect,"tsect[Ntraco]/I");  
  theTree->Branch("tnum",tnum,"tnum[Ntraco]/I"); 
  theTree->Branch("tbx",tbx,"tbx[Ntraco]/I");
  theTree->Branch("tcod",tcod,"tcod[Ntraco]/I");
  theTree->Branch("tk",tk,"tk[Ntraco]/I");
  theTree->Branch("tx",tx,"tx[Ntraco]/I");
  theTree->Branch("tposx",tposx,"tposx[Ntraco]/F");
  theTree->Branch("tposy",tposy,"tposy[Ntraco]/F");
  theTree->Branch("tposz",tposz,"tposz[Ntraco]/F");
  theTree->Branch("tdirx",tdirx,"tdirx[Ntraco]/F");
  theTree->Branch("tdiry",tdiry,"tdiry[Ntraco]/F");
  theTree->Branch("tdirz",tdirz,"tdirz[Ntraco]/F");
  // TSPHI block
  theTree->Branch("Ntsphi",&ntsphi,"Ntsphi/I");
  theTree->Branch("swh",swh,"swh[Ntsphi]/I"); 
  theTree->Branch("sstat",sstat,"sstat[Ntsphi]/I");    
  theTree->Branch("ssect",ssect,"ssect[Ntsphi]/I");  
  theTree->Branch("sbx",sbx,"sbx[Ntsphi]/I");
  theTree->Branch("scod",scod,"scod[Ntsphi]/I");
  theTree->Branch("sphi",sphi,"sphi[Ntsphi]/I");
  theTree->Branch("sphib",sphib,"sphib[Ntsphi]/I");
  theTree->Branch("sposx",sposx,"sposx[Ntsphi]/F");
  theTree->Branch("sposy",sposy,"sposy[Ntsphi]/F");
  theTree->Branch("sposz",sposz,"sposz[Ntsphi]/F");
  theTree->Branch("sdirx",sdirx,"sdirx[Ntsphi]/F");
  theTree->Branch("sdiry",sdiry,"sdiry[Ntsphi]/F");
  theTree->Branch("sdirz",sdirz,"sdirz[Ntsphi]/F");
  // TSTHETA block
  theTree->Branch("Ntstheta",&ntstheta,"Ntstheta/I");
  theTree->Branch("thwh",thwh,"thwh[Ntstheta]/I"); 
  theTree->Branch("thstat",thstat,"thstat[Ntstheta]/I");    
  theTree->Branch("thsect",thsect,"thsect[Ntstheta]/I");  
  theTree->Branch("thbx",thbx,"thbx[Ntstheta]/I");
  theTree->Branch("thcode",thcode,"thcode[Ntstheta][7]/I");
  theTree->Branch("thpos",thpos,"thpos[Ntstheta][7]/I");
  theTree->Branch("thqual",thqual,"thqual[Ntstheta][7]/I");

}

void DTTrigTest::analyze(const Event & iEvent, const EventSetup& iEventSetup){
  
  const int MAXGEN  = 10;
  const float ptcut  = 1.0;
  const float etacut = 2.4;
  
  MyTrig->triggerReco(iEvent,iEventSetup);
  cout << "[DTTrigTest] Trigger algorithm executed for run " << iEvent.id().run() <<" event " << iEvent.id().event() << endl;
  
  // GENERAL Block
  runn   = iEvent.id().run();
  eventn = iEvent.id().event();
  weight = 1; // FIXME what to do with this varable?
  
  // GEANT Block
  Handle<vector<SimTrack> > MyTracks;
  Handle<vector<SimVertex> > MyVertexes;
  iEvent.getByLabel("g4SimHits",MyTracks);
  iEvent.getByLabel("g4SimHits",MyVertexes);
  vector<SimTrack>::const_iterator itrack;
  ngen=0;
  if (debug == true) cout  << "[DTTrigTest] Tracks found in the detector (not only muons) " << MyTracks->size() <<endl;
  for (itrack=MyTracks->begin(); itrack!=MyTracks->end(); itrack++){
    if ( abs(itrack->type())==13){
      float pt  = itrack->momentum().perp();
      float eta = itrack->momentum().pseudoRapidity();
      if ( pt>ptcut && fabs(eta)<etacut ){
	HepLorentzVector momentum = itrack->momentum();
	float phi = momentum.phi();
	int charge = static_cast<int> (-itrack->type()/13); //static_cast<int> (itrack->charge()); charge() still to be implemented
	if ( phi<0 ) phi = 2*M_PI + phi;
	int vtxindex = itrack->vertIndex();
	float gvx=0,gvy=0,gvz=0;
	if (vtxindex >-1){
	  gvx=MyVertexes->at(vtxindex).position().x();
	  gvy=MyVertexes->at(vtxindex).position().y();
	  gvz=MyVertexes->at(vtxindex).position().z();
	}
	if ( ngen < MAXGEN ) {
	  pxgen[ngen]=momentum.x();
	  pygen[ngen]=momentum.y();
	  pzgen[ngen]=momentum.z();
	  ptgen[ngen]=pt;
	  etagen[ngen]=eta;
	  phigen[ngen]=phi;
	  chagen[ngen]=charge;
	  vxgen[ngen]=gvx;
	  vygen[ngen]=gvy;
	  vzgen[ngen]=gvz;
	  ngen++;
	}
      }
    }
  }
  
  // L1 Local Trigger Block
  // BTI
  vector<DTBtiTrigData> btitrigs = MyTrig->BtiTrigs();
  vector<DTBtiTrigData>::const_iterator pbti;
  int ibti = 0;
  if (debug == true) cout << "[DTTrigTest] " << btitrigs.size() << " BTI triggers found" << endl;
  for ( pbti = btitrigs.begin(); pbti != btitrigs.end(); pbti++ ) {
    if ( ibti < 100 ) {
      bwh[ibti]=pbti->wheel();
      bstat[ibti]=pbti->station();
      bsect[ibti]=pbti->sector();
      bsl[ibti]=pbti->btiSL();
      bnum[ibti]=pbti->btiNumber();
      bbx[ibti]=pbti->step();
      bcod[ibti]=pbti->code();
      bk[ibti]=pbti->K();
      bx[ibti]=pbti->X();
      GlobalPoint pos = MyTrig->CMSPosition(&(*pbti));
      GlobalVector dir = MyTrig->CMSDirection(&(*pbti));
      bposx[ibti] = pos.x();
      bposy[ibti] = pos.y();
      bposz[ibti] = pos.z();
      bdirx[ibti] = dir.x();
      bdiry[ibti] = dir.y();
      bdirz[ibti] = dir.z();
      ibti++;
    }
  } 
  nbti = ibti;
  //cout << nbti << endl;
  
  //TRACO
  vector<DTTracoTrigData> tracotrigs = MyTrig->TracoTrigs();
  vector<DTTracoTrigData>::const_iterator ptc;
  int itraco = 0;
  if (debug == true) cout << "[DTTrigTest] " << tracotrigs.size() << " TRACO triggers found" << endl;
  for (ptc=tracotrigs.begin(); ptc!=tracotrigs.end(); ptc++) {
    if (itraco<80) {
      twh[itraco]=ptc->wheel();
      tstat[itraco]=ptc->station();
      tsect[itraco]=ptc->sector();
      tnum[itraco]=ptc->tracoNumber();
      tbx[itraco]=ptc->step();
      tcod[itraco]=ptc->code();
      tk[itraco]=ptc->K();
      tx[itraco]=ptc->X();
      GlobalPoint pos = MyTrig->CMSPosition(&(*ptc));
      GlobalVector dir = MyTrig->CMSDirection(&(*ptc));
      tposx[itraco] = pos.x();
      tposy[itraco] = pos.y();
      tposz[itraco] = pos.z();
      tdirx[itraco] = dir.x();
      tdiry[itraco] = dir.y();
      tdirz[itraco] = dir.z();
      itraco++;
    }
  }
  ntraco = itraco;
  //cout << ntraco << endl;
  
  //TSPHI
  vector<DTChambPhSegm> tsphtrigs = MyTrig->TSPhTrigs();
  vector<DTChambPhSegm>::const_iterator ptsph;
  int itsphi = 0; 
  if (debug == true) cout << "[DTTrigTest] " << tsphtrigs.size() << " TSPhi triggers found" << endl;
  for (ptsph=tsphtrigs.begin(); ptsph!=tsphtrigs.end(); ptsph++) {
    if (itsphi<40 ) {
      const DTChambPhSegm& seg = (*ptsph);
      swh[itsphi] = ptsph->wheel();
      sstat[itsphi] = ptsph->station();
      ssect[itsphi] = ptsph->sector();
      sbx[itsphi] = ptsph->step();      
      scod[itsphi] = ptsph->oldCode();
      sphi[itsphi] = ptsph->phi();
      sphib[itsphi] = ptsph->phiB();
      GlobalPoint pos = MyTrig->CMSPosition(&seg); 
      GlobalVector dir = MyTrig->CMSDirection(&seg);
      sposx[itsphi] = pos.x();
      sposy[itsphi] = pos.y();
      sposz[itsphi] = pos.z();
      sdirx[itsphi] = dir.x();
      sdiry[itsphi] = dir.y();
      sdirz[itsphi] = dir.z();
      itsphi++;
    }
  }
  ntsphi = itsphi;
  //cout << ntsphi << endl;
  
  //TSPHI
  vector<DTChambThSegm> tsthtrigs = MyTrig->TSThTrigs();
  vector<DTChambThSegm>::const_iterator ptsth;
  int itstheta = 0; 
  if (debug == true) cout << "[DTTrigTest] " << tsthtrigs.size() << " TSTheta triggers found" << endl;
  for (ptsth=tsthtrigs.begin(); ptsth!=tsthtrigs.end(); ptsth++) {
    if (itstheta<40 ) {
      thwh[itstheta] = ptsth->ChamberId().wheel();
      thstat[itstheta] = ptsth->ChamberId().station();
      thsect[itstheta] = ptsth->ChamberId().sector();
      thbx[itstheta] = ptsth->step();
      for(int i=0;i<7;i++) {
	  thcode[itstheta][i] = ptsth->code(i);
	  thpos[itstheta][i] = ptsth->position(i);
	  thqual[itstheta][i] = ptsth->quality(i);
      }
      itstheta++;
    }
  }
  ntstheta = itstheta;
  
  //Fill the tree
  theTree->Fill();

}
