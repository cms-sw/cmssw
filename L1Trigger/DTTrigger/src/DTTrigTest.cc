//-------------------------------------------------
//
/**  \class DTTrigTest
 *
 *   EDAnalyzer that generates a rootfile useful
 *   for L1-DTTrigger debugging and performance 
 *   studies
 *
 *
 *   $Date: 2009/12/22 09:36:34 $
 *   $Revision: 1.13 $
 *
 *   \author C. Battilana
 */
//
//--------------------------------------------------

// This class's header
#include "L1Trigger/DTTrigger/interface/DTTrigTest.h"

// Framework related classes
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

// Trigger and DataFormats headers
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// ROOT headers 
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

// Collaborating classes
#include "DataFormats/Math/interface/LorentzVector.h"
#include <CLHEP/Vector/LorentzVector.h>

// C++ headers
#include <iostream>
#include <math.h>
#include<time.h>

using namespace std;
using namespace edm;

const double DTTrigTest::my_TtoTDC = 32./25.;

DTTrigTest::DTTrigTest(const ParameterSet& pset): my_trig(0) { 

  my_debug= pset.getUntrackedParameter<bool>("debug");
  string outputfile = pset.getUntrackedParameter<string>("outputFileName");
  if (my_debug) 
    cout << "[DTTrigTest] Creating rootfile " <<  outputfile <<endl;
  my_rootfile = new TFile(outputfile.c_str(),"RECREATE");
  my_tree = new TTree("h1","GMT",0);
  my_params = pset;
  if (my_debug) cout << "[DTTrigTest] Constructor executed!!!" << endl;


}

DTTrigTest::~DTTrigTest(){ 

  if (my_trig != 0) delete my_trig;
  delete my_rootfile;
  if (my_debug) 
    cout << "[DTTrigTest] Destructor executed!!!" << endl;

}

void DTTrigTest::endJob(){

  if (my_debug) 
    cout << "[DTTrigTest] Writing Tree and Closing File" << endl;
  my_tree->Write();
  delete my_tree;
  my_rootfile->Close();

}

//void DTTrigTest::beginJob(const EventSetup & iEventSetup){   
void DTTrigTest::beginJob(){   
  // get DTConfigManager
  // ESHandle< DTConfigManager > confManager ;
  // iEventSetup.get< DTConfigManagerRcd >().get( confManager ) ;

  //for testing purpose....
  //DTBtiId btiid(1,1,1,1,1);
  //confManager->getDTConfigBti(btiid)->print();

//   my_trig = new DTTrig(my_params);

//   my_trig->createTUs(iEventSetup);
//   if (my_debug) 
//     cout << "[DTTrigTest] TU's Created" << endl;
  
  // BOOKING of the tree's varables
  // GENERAL block branches
  my_tree->Branch("Run",&runn,"Run/I");
  my_tree->Branch("Event",&eventn,"Event/I");
  my_tree->Branch("Weight",&weight,"Weight/F");  
  // GEANT block branches
  my_tree->Branch("Ngen",&ngen,"Ngen/I");
  my_tree->Branch("Pxgen",pxgen,"Pxgen[Ngen]/F");
  my_tree->Branch("Pygen",pygen,"Pygen[Ngen]/F");
  my_tree->Branch("Pzgen",pzgen,"Pzgen[Ngen]/F");
  my_tree->Branch("Ptgen",ptgen,"Ptgen[Ngen]/F");
  my_tree->Branch("Etagen",etagen,"Etagen[Ngen]/F");
  my_tree->Branch("Phigen",phigen,"Phigen[Ngen]/F");
  my_tree->Branch("Chagen",chagen,"Chagen[Ngen]/I");
  my_tree->Branch("Vxgen",vxgen,"Vxgen[Ngen]/F");
  my_tree->Branch("Vygen",vygen,"Vygen[Ngen]/F");
  my_tree->Branch("Vzgen",vzgen,"Vzgen[Ngen]/F");
  // L1MuDTBtiChipS block
  my_tree->Branch("Nbti",&nbti,"Nbti/I");
  my_tree->Branch("bwh",bwh,"bwh[Nbti]/I"); 
  my_tree->Branch("bstat",bstat,"bstat[Nbti]/I");    
  my_tree->Branch("bsect",bsect,"bsect[Nbti]/I");  
  my_tree->Branch("bsl",bsl,"bsl[Nbti]/I");
  my_tree->Branch("bnum",bnum,"bnum[Nbti]/I");
  my_tree->Branch("bbx",bbx,"bbx[Nbti]/I");
  my_tree->Branch("bcod",bcod,"bcod[Nbti]/I");
  my_tree->Branch("bk",bk,"bk[Nbti]/I");
  my_tree->Branch("bx",bx,"bx[Nbti]/I");
  my_tree->Branch("bposx",bposx,"bposx[Nbti]/F");
  my_tree->Branch("bposy",bposy,"bposy[Nbti]/F");
  my_tree->Branch("bposz",bposz,"bposz[Nbti]/F");
  my_tree->Branch("bdirx",bdirx,"bdirx[Nbti]/F");
  my_tree->Branch("bdiry",bdiry,"bdiry[Nbti]/F");
  my_tree->Branch("bdirz",bdirz,"bdirz[Nbti]/F");
  // L1MuDTTracoChipS block
  my_tree->Branch("Ntraco",&ntraco,"Ntraco/I");
  my_tree->Branch("twh",twh,"twh[Ntraco]/I"); 
  my_tree->Branch("tstat",tstat,"tstat[Ntraco]/I");    
  my_tree->Branch("tsect",tsect,"tsect[Ntraco]/I");  
  my_tree->Branch("tnum",tnum,"tnum[Ntraco]/I"); 
  my_tree->Branch("tbx",tbx,"tbx[Ntraco]/I");
  my_tree->Branch("tcod",tcod,"tcod[Ntraco]/I");
  my_tree->Branch("tk",tk,"tk[Ntraco]/I");
  my_tree->Branch("tx",tx,"tx[Ntraco]/I");
  my_tree->Branch("tposx",tposx,"tposx[Ntraco]/F");
  my_tree->Branch("tposy",tposy,"tposy[Ntraco]/F");
  my_tree->Branch("tposz",tposz,"tposz[Ntraco]/F");
  my_tree->Branch("tdirx",tdirx,"tdirx[Ntraco]/F");
  my_tree->Branch("tdiry",tdiry,"tdiry[Ntraco]/F");
  my_tree->Branch("tdirz",tdirz,"tdirz[Ntraco]/F");
  // TSPHI block
  my_tree->Branch("Ntsphi",&ntsphi,"Ntsphi/I");
  my_tree->Branch("swh",swh,"swh[Ntsphi]/I"); 
  my_tree->Branch("sstat",sstat,"sstat[Ntsphi]/I");    
  my_tree->Branch("ssect",ssect,"ssect[Ntsphi]/I");  
  my_tree->Branch("sbx",sbx,"sbx[Ntsphi]/I");
  my_tree->Branch("scod",scod,"scod[Ntsphi]/I");
  my_tree->Branch("sphi",sphi,"sphi[Ntsphi]/I");
  my_tree->Branch("sphib",sphib,"sphib[Ntsphi]/I");
  my_tree->Branch("sposx",sposx,"sposx[Ntsphi]/F");
  my_tree->Branch("sposy",sposy,"sposy[Ntsphi]/F");
  my_tree->Branch("sposz",sposz,"sposz[Ntsphi]/F");
  my_tree->Branch("sdirx",sdirx,"sdirx[Ntsphi]/F");
  my_tree->Branch("sdiry",sdiry,"sdiry[Ntsphi]/F");
  my_tree->Branch("sdirz",sdirz,"sdirz[Ntsphi]/F");
  // TSTHETA block
  my_tree->Branch("Ntstheta",&ntstheta,"Ntstheta/I");
  my_tree->Branch("thwh",thwh,"thwh[Ntstheta]/I"); 
  my_tree->Branch("thstat",thstat,"thstat[Ntstheta]/I");    
  my_tree->Branch("thsect",thsect,"thsect[Ntstheta]/I");  
  my_tree->Branch("thbx",thbx,"thbx[Ntstheta]/I");
  my_tree->Branch("thcode",thcode,"thcode[Ntstheta][7]/I");
  my_tree->Branch("thpos",thpos,"thpos[Ntstheta][7]/I");
  my_tree->Branch("thqual",thqual,"thqual[Ntstheta][7]/I");
  // SC PHI block
  my_tree->Branch("Nscphi",&nscphi,"Nscphi/I");
  my_tree->Branch("scphwh",scphwh,"scphwh[Nscphi]/I"); 
  my_tree->Branch("scphstat",scphstat,"scphstat[Nscphi]/I");    
  my_tree->Branch("scphsect",scphsect,"scphsect[Nscphi]/I");  
  my_tree->Branch("scphbx",scphbx,"scphbx[Nscphi]/I");
  my_tree->Branch("scphcod",scphcod,"scphcod[Nscphi]/I");
  my_tree->Branch("scphphi",scphphi,"scphphi[Nscphi]/I");
  my_tree->Branch("scphphib",scphphib,"scphphib[Nscphi]/I");
  my_tree->Branch("scphposx",scphposx,"scphposx[Nscphi]/F");
  my_tree->Branch("scphposy",scphposy,"scphposy[Nscphi]/F");
  my_tree->Branch("scphposz",scphposz,"scphposz[Nscphi]/F");
  my_tree->Branch("scphdirx",scphdirx,"scphdirx[Nscphi]/F");
  my_tree->Branch("scphdiry",scphdiry,"scphdiry[Nscphi]/F");
  my_tree->Branch("scphdirz",scphdirz,"scphdirz[Nscphi]/F");
  // SC THETA block
  my_tree->Branch("Nsctheta",&nsctheta,"Nsctheta/I");
  my_tree->Branch("scthwh",scthwh,"scthwh[Nsctheta]/I"); 
  my_tree->Branch("scthstat",scthstat,"scthstat[Nsctheta]/I");    
  my_tree->Branch("scthsect",scthsect,"scthsect[Nsctheta]/I");  
  my_tree->Branch("scthbx",scthbx,"scthbx[Nsctheta]/I");
  my_tree->Branch("scthcode",scthcode,"scthcode[Nsctheta][7]/I");
  my_tree->Branch("scthpos",scthpos,"scthpos[Nsctheta][7]/I");
  my_tree->Branch("scthqual",scthqual,"scthqual[Nsctheta][7]/I");

}

void DTTrigTest::beginRun(const edm::Run& iRun, const edm::EventSetup& iEventSetup) {

  if (!my_trig) {
    my_trig = new DTTrig(my_params);
    my_trig->createTUs(iEventSetup);
    if (my_debug)
      cout << "[DTTrigTest] TU's Created" << endl;
  }

}



void DTTrigTest::analyze(const Event & iEvent, const EventSetup& iEventSetup){
  
  const int MAXGEN  = 10;
  const float ptcut  = 1.0;
  const float etacut = 2.4;
  my_trig->triggerReco(iEvent,iEventSetup);
  if (my_debug)
    cout << "[DTTrigTest] Trigger algorithm executed for run " << iEvent.id().run() 
	 <<" event " << iEvent.id().event() << endl;
  
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
  if (my_debug) 
    cout  << "[DTTrigTest] Tracks found in the detector (not only muons) " << MyTracks->size() <<endl;
  
  for (itrack=MyTracks->begin(); itrack!=MyTracks->end(); itrack++){
    if ( abs(itrack->type())==13){
      math::XYZTLorentzVectorD momentum = itrack->momentum();
      float pt  = momentum.Pt();
      float eta = momentum.eta();
      if ( pt>ptcut && fabs(eta)<etacut ){
	float phi = momentum.phi();
	int charge = static_cast<int> (-itrack->type()/13); //static_cast<int> (itrack->charge());
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
  vector<DTBtiTrigData> btitrigs = my_trig->BtiTrigs();
  vector<DTBtiTrigData>::const_iterator pbti;
  int ibti = 0;
  if (my_debug)
    cout << "[DTTrigTest] " << btitrigs.size() << " BTI triggers found" << endl;
  
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
      GlobalPoint pos = my_trig->CMSPosition(&(*pbti));
      GlobalVector dir = my_trig->CMSDirection(&(*pbti));
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
  
  //TRACO
  vector<DTTracoTrigData> tracotrigs = my_trig->TracoTrigs();
  vector<DTTracoTrigData>::const_iterator ptc;
  int itraco = 0;
  if (my_debug)
    cout << "[DTTrigTest] " << tracotrigs.size() << " TRACO triggers found" << endl;
  
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
      GlobalPoint pos = my_trig->CMSPosition(&(*ptc));
      GlobalVector dir = my_trig->CMSDirection(&(*ptc));
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
  
  //TSPHI
  vector<DTChambPhSegm> tsphtrigs = my_trig->TSPhTrigs();
  vector<DTChambPhSegm>::const_iterator ptsph;
  int itsphi = 0; 
  if (my_debug ) 
    cout << "[DTTrigTest] " << tsphtrigs.size() << " TSPhi triggers found" << endl;
  
  for (ptsph=tsphtrigs.begin(); ptsph!=tsphtrigs.end(); ptsph++) {
    if (itsphi<40 ) {
      swh[itsphi] = ptsph->wheel();
      sstat[itsphi] = ptsph->station();
      ssect[itsphi] = ptsph->sector();
      sbx[itsphi] = ptsph->step();
      scod[itsphi] = ptsph->oldCode();
      sphi[itsphi] = ptsph->phi();
      sphib[itsphi] = ptsph->phiB();
      GlobalPoint pos = my_trig->CMSPosition(&(*ptsph)); 
      GlobalVector dir = my_trig->CMSDirection(&(*ptsph));
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
  
  //TSTHETA
  vector<DTChambThSegm> tsthtrigs = my_trig->TSThTrigs();
  vector<DTChambThSegm>::const_iterator ptsth;
  int itstheta = 0; 
  if (my_debug) 
    cout << "[DTTrigTest] " << tsthtrigs.size() << " TSTheta triggers found" << endl;
  
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
  
  //SCPHI
  vector<DTSectCollPhSegm> scphtrigs = my_trig->SCPhTrigs();
  vector<DTSectCollPhSegm>::const_iterator pscph;
  int iscphi = 0; 
  if (my_debug ) 
    cout << "[DTTrigTest] " << scphtrigs.size() << " SectCollPhi triggers found" << endl;
  
  for (pscph=scphtrigs.begin(); pscph!=scphtrigs.end(); pscph++) {
    if (iscphi<40 ) {
      const DTChambPhSegm *seg = (*pscph).tsPhiTrig();
      scphwh[iscphi] = pscph->wheel();
      scphstat[iscphi] = pscph->station();
      scphsect[iscphi] = pscph->sector();
      scphbx[iscphi] = pscph->step();
      scphcod[iscphi] = pscph->oldCode();
      scphphi[iscphi] = pscph->phi();
      scphphib[iscphi] = pscph->phiB();
      GlobalPoint pos = my_trig->CMSPosition(seg); 
      GlobalVector dir = my_trig->CMSDirection(seg);
      scphposx[iscphi] = pos.x();
      scphposy[iscphi] = pos.y();
      scphposz[iscphi] = pos.z();
      scphdirx[iscphi] = dir.x();
      scphdiry[iscphi] = dir.y();
      scphdirz[iscphi] = dir.z();
      iscphi++;
    }
  }
  nscphi = iscphi;
  
  //SCTHETA
  vector<DTSectCollThSegm> scthtrigs = my_trig->SCThTrigs();
  vector<DTSectCollThSegm>::const_iterator pscth;
  int isctheta = 0; 
  if (my_debug) 
    cout << "[DTTrigTest] " << scthtrigs.size() << " SectCollTheta triggers found" << endl;
  
  for (pscth=scthtrigs.begin(); pscth!=scthtrigs.end(); pscth++) {
    if (isctheta<40 ) {
      scthwh[isctheta] = pscth->ChamberId().wheel();
      scthstat[isctheta] = pscth->ChamberId().station();
      scthsect[isctheta] = pscth->ChamberId().sector();
      scthbx[isctheta] = pscth->step();
      for(int i=0;i<7;i++) {
	  scthcode[isctheta][i] = pscth->code(i);
	  scthpos[isctheta][i] = pscth->position(i);
	  scthqual[isctheta][i] = pscth->quality(i);
      }
      isctheta++;
    }
  }
  nsctheta = isctheta;
  
  //Fill the tree
  my_tree->Fill();

}
