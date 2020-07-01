
#include <iostream>

#include "IOMC/EventVertexGenerators/test/BoostTester.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std;

BoostTester::BoostTester(const ParameterSet&) {
  fOutputFile = nullptr;

  ftreevtx = new TTree("vtxtree", "vtxtree");
  ftreevtx->Branch("vx", &fvx, "fvx/D");
  ftreevtx->Branch("vy", &fvy, "fvy/D");
  ftreevtx->Branch("vz", &fvz, "fvz/D");

  ftreep = new TTree("ptree", "ptree");
  ftreep->Branch("px", &fpx, "fpx/D");
  ftreep->Branch("py", &fpy, "fpy/D");
  ftreep->Branch("pz", &fpz, "fpz/D");
  //ftreep->Branch("pt",&fpt,"fpt/D");
  //ftreep->Branch("p",&fp,"fp/D");
  //ftreep->Branch("e",&fe,"fe/D");
  //ftreep->Branch("eta",&feta,"feta/D");
  //ftreep->Branch("phi",&fphi,"fphi/D");

  /*

   fVtxHistz = 0 ;
   fVtxHistx = 0 ;
   fVtxHisty = 0 ;
   fVtxHistxy =0;
   fPhiHistOrg = 0 ;
   fPhiHistSmr = 0 ;
   fEtaHistOrg = 0 ;
   fEtaHistSmr = 0 ;
   fpxHist = fpyHist = fpzHist = fpHist = feHist = fptHist = 0;
*/
}

void BoostTester::beginJob() {
  fOutputFile = new TFile("VtxTest.root", "RECREATE");
  /*
   fVtxHistz      = new TH1D("VtxHistz","Test Z-spread", 150, -250., 250. ) ;
   fVtxHistx      = new TH1D("VtxHistx","Test X-spread", 500, -0.5, 0.5 ) ;
   fVtxHisty      = new TH1D("VtxHisty","Test Y-spread", 500, -0.5, 0.5 ) ;
   fVtxHistxy     = new TH2D("VtxHistxy","Test X-Y spread",700,-0.5,0.5,700,-0.5,0.5);
   
   fpxHist        = new TH1D("pxHist","p_{X} [GeV/c]",100,-15,15.);
   fpyHist        = new TH1D("pyHist","p_{Y} [GeV/c]",100,-15,15.);
   fpzHist        = new TH1D("pzHist","p_{Z} [GeV/c]",100,-15,15.);
   fpHist         = new TH1D("pHist","p [GeV/c]",100,0.,30.);
   feHist         = new TH1D("eHist","E [GeV]", 100,0,20.);
   fptHist        = new TH1D("ptHist","p_{T} [GeV/c]",100,0,10.);

   fPhiHistOrg   = new TH1D("PhiHistOrg","Test Phi, org.", 80, -4., 4. ) ;
   fPhiHistSmr   = new TH1D("PhiHistSmr","Test Phi, smr.", 80, -4., 4. ) ;
   fEtaHistOrg   = new TH1D("EtaHistOrg","Test Eta, org.", 80, -4., 4. ) ;
   fEtaHistSmr   = new TH1D("EtaHistSmr","Test Eta, smr.", 80, -4., 4. ) ;
   */

  return;
}

void BoostTester::analyze(const Event& e, const EventSetup&) {
  ftreevtx->SetBranchAddress("vx", &fvx);
  ftreevtx->SetBranchAddress("vy", &fvy);
  ftreevtx->SetBranchAddress("vz", &fvz);

  ftreep->SetBranchAddress("px", &fpx);
  ftreep->SetBranchAddress("py", &fpy);
  ftreep->SetBranchAddress("pz", &fpz);
  //ftreep->SetBranchAddress("pt",&fpt);
  //ftreep->SetBranchAddress("p",&fp);
  //ftreep->SetBranchAddress("e",&fe);
  //ftreep->SetBranchAddress("eta",&feta);
  //ftreep->SetBranchAddress("phi",&fphi);

  fpx = 0.;
  fpy = 0.;
  fpz = 0.;

  vector<Handle<HepMCProduct> > EvtHandles;
  e.getManyByType(EvtHandles);

  //std::cout << "evthandles= " << EvtHandles.size() << std::endl;

  for (unsigned int i = 0; i < EvtHandles.size(); i++) {
    //std::cout << " i=" << i <<  " name: "<< EvtHandles[i].provenance()->moduleLabel() << std::endl;

    if (EvtHandles[i].isValid()) {
      const HepMC::GenEvent* Evt = EvtHandles[i]->GetEvent();

      // take only 1st vertex for now - it's been tested only of PGuns...
      //

      //HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin() ;

      for (HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin(); Vtx != Evt->vertices_end(); ++Vtx) {
        //if ( (*Vtx)->status() != 1 ) continue;

        fvx = (*Vtx)->position().x();
        fvy = (*Vtx)->position().y();
        fvz = (*Vtx)->position().z();

        ftreevtx->Fill();
      }

      for (HepMC::GenEvent::particle_const_iterator Part = Evt->particles_begin(); Part != Evt->particles_end();
           ++Part) {
        if ((*Part)->status() != 1)
          continue;

        HepMC::FourVector Mon = (*Part)->momentum();

        //if ( EvtHandles[i].provenance()->moduleLabel() == "VtxSmeared" )
        //{

        fpx += Mon.px();
        fpy += Mon.py();
        fpz += Mon.pz();
        /*
		   fp = Mon.mag();
		   fpt = Mon.perp();
		   fe = Mon.e();
		   feta = Mon.eta();//-log(tan(Mom.theta()/2.));
		   fphi = Mon.phi();
		   */

        //ftreep->Fill();

        //std::cout << "particle: p="<<Mon.mag() << " status="<< (*Part)->status() << " pdgid="<<(*Part)->pdg_id() << std::endl;
      }

      //std::cout << " vertex (x,y,z)= " << (*Vtx)->position().x() <<" " << (*Vtx)->position().y() << " " << (*Vtx)->position().z() << std::endl;
      //std::cout << " vertices= " << itotal << std::endl;
    }
  }
  //std::cout << " total px= " << fpx << " py= " << fpy << " pz= " << fpz << std::endl;

  ftreep->Fill();
  return;
}

void BoostTester::endJob() {
  ftreevtx->Write();
  ftreep->Write();

  fOutputFile->Write();
  fOutputFile->Close();

  return;
}

DEFINE_FWK_MODULE(BoostTester);
