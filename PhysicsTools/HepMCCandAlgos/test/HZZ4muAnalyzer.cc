#include <iostream>

#include "PhysicsTools/HepMCCandAlgos/test/HZZ4muAnalyzer.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TFile.h"
#include "TH1.h"

using namespace edm;
using namespace std;

HZZ4muAnalyzer::HZZ4muAnalyzer(const ParameterSet& pset)
    : fToken(consumes<HepMCProduct>(InputTag("source"))),
      fOutputFileName(pset.getUntrackedParameter<string>("HistOutFile", std::string("TestHiggsMass.root"))),
      fOutputFile(nullptr),
      fHist2muMass(nullptr),
      fHist4muMass(nullptr),
      fHistZZMass(nullptr) {}

void HZZ4muAnalyzer::beginJob() {
  fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
  fHist2muMass = new TH1D("Hist2muMass", "2-mu inv. mass", 100, 60., 120.);
  fHist4muMass = new TH1D("Hist4muMass", "4-mu inv. mass", 100, 170., 210.);
  fHistZZMass = new TH1D("HistZZMass", "ZZ inv. mass", 100, 170., 210.);

  return;
}

void HZZ4muAnalyzer::analyze(const Event& e, const EventSetup&) {
  Handle<HepMCProduct> EvtHandle;

  // find initial (unsmeared, unfiltered,...) HepMCProduct
  //
  e.getByToken(fToken, EvtHandle);

  const HepMC::GenEvent* Evt = EvtHandle->GetEvent();

  // this a pointer - and not an array/vector/...
  // because this example explicitely assumes
  // that there one and only Higgs in the record
  //
  HepMC::GenVertex* HiggsDecVtx = nullptr;

  // find the 1st vertex with outgoing Higgs
  // and get Higgs decay vertex from there;
  //
  // in principal, one can look for the vertex
  // with incoming Higgs as well...
  //
  for (HepMC::GenEvent::vertex_const_iterator vit = Evt->vertices_begin(); vit != Evt->vertices_end(); vit++) {
    for (HepMC::GenVertex::particles_out_const_iterator pout = (*vit)->particles_out_const_begin();
         pout != (*vit)->particles_out_const_end();
         pout++) {
      if ((*pout)->pdg_id() == 25 && (*pout)->status() == 2) {
        if ((*pout)->end_vertex() != nullptr) {
          HiggsDecVtx = (*pout)->end_vertex();
          break;
        }
      }
    }
    if (HiggsDecVtx != nullptr) {
      break;  // break the initial loop over vertices
    }
  }

  if (HiggsDecVtx == nullptr) {
    cout << " There is NO Higgs in this event ! " << endl;
    return;
  }

  if (e.id().event() == 1) {
    cout << " " << endl;
    cout << " We do some example printouts in the event 1 " << endl;
    cout << " Higgs decay found at the vertex " << HiggsDecVtx->barcode() << " (barcode)" << endl;

    vector<HepMC::GenParticle*> HiggsChildren;

    for (HepMC::GenVertex::particles_out_const_iterator H0in = HiggsDecVtx->particles_out_const_begin();
         H0in != HiggsDecVtx->particles_out_const_end();
         H0in++) {
      HiggsChildren.push_back(*H0in);
    }
    cout << " Number of Higgs (immediate) children = " << HiggsChildren.size() << endl;
    for (unsigned int ic = 0; ic < HiggsChildren.size(); ic++) {
      HiggsChildren[ic]->print();
    }
  }

  // select and store stable descendants of the Higgs
  //
  vector<HepMC::GenParticle*> StableHiggsDesc;

  if (e.id().event() == 1)
    cout << " Now let us list all descendents of the Higgs" << endl;
  for (HepMC::GenVertex::particle_iterator des = HiggsDecVtx->particles_begin(HepMC::descendants);
       des != HiggsDecVtx->particles_end(HepMC::descendants);
       des++) {
    if (e.id().event() == 1)
      (*des)->print();
    if ((*des)->status() == 1)
      StableHiggsDesc.push_back(*des);
  }

  HepMC::FourVector Mom2part;
  double XMass2part = 0.;
  double XMass4part = 0.;
  double XMass2pairs = 0.;
  vector<HepMC::FourVector> Mom2partCont;

  // browse the array of stable descendants
  // and do 2-mu inv.mass
  //
  for (unsigned int i = 0; i < StableHiggsDesc.size(); i++) {
    // skip other than mu
    //
    if (std::abs(StableHiggsDesc[i]->pdg_id()) != 13)
      continue;

    for (unsigned int j = i + 1; j < StableHiggsDesc.size(); j++) {
      // skip other than mu
      //
      if (std::abs(StableHiggsDesc[j]->pdg_id()) != 13)
        continue;
      //
      // skip same charge combo's
      //
      if ((StableHiggsDesc[i]->pdg_id() * StableHiggsDesc[j]->pdg_id()) > 0)
        continue;
      //
      // OK, opposite charges, do the job
      //
      Mom2part = HepMC::FourVector((StableHiggsDesc[i]->momentum().px() + StableHiggsDesc[j]->momentum().px()),
                                   (StableHiggsDesc[i]->momentum().py() + StableHiggsDesc[j]->momentum().py()),
                                   (StableHiggsDesc[i]->momentum().pz() + StableHiggsDesc[j]->momentum().pz()),
                                   (StableHiggsDesc[i]->momentum().e() + StableHiggsDesc[j]->momentum().e()));

      XMass2part = Mom2part.m();
      fHist2muMass->Fill(XMass2part);
      //cout << " counters : " << StableHiggsDesc[i]->barcode() << " "
      //                       << StableHiggsDesc[j]->barcode()
      //			<< " -> 2-part mass = " << XMass2part << endl ;
      //
      // store if 2-part. inv. mass fits into (roughly) Z-mass interval
      //
      if (XMass2part > 80. && XMass2part < 100.) {
        Mom2partCont.push_back(Mom2part);
      }
    }
  }

  // make 4-part inv.mass
  //
  double px4, py4, pz4, e4;
  px4 = py4 = pz4 = e4 = 0.;
  if (StableHiggsDesc.size() == 4) {
    for (unsigned int i = 0; i < StableHiggsDesc.size(); i++) {
      px4 += StableHiggsDesc[i]->momentum().px();
      py4 += StableHiggsDesc[i]->momentum().py();
      pz4 += StableHiggsDesc[i]->momentum().pz();
      e4 += StableHiggsDesc[i]->momentum().e();
    }
    XMass4part = HepMC::FourVector(px4, py4, pz4, e4).m();
    fHist4muMass->Fill(XMass4part);
  }
  //cout << " 4-part inv. mass = " << XMass4part << endl ;

  // make 2-pairs (ZZ) inv.mass
  //
  //cout << " selected Z-candidates in this event : " << Mom2partCont.size() << endl ;
  for (unsigned int i = 0; i < Mom2partCont.size(); i++) {
    for (unsigned int j = i + 1; j < Mom2partCont.size(); j++) {
      // Mom2pairs = Mom2partCont[i] + Mom2partCont[j] ;
      XMass2pairs = HepMC::FourVector((Mom2partCont[i].px() + Mom2partCont[j].px()),
                                      (Mom2partCont[i].py() + Mom2partCont[j].py()),
                                      (Mom2partCont[i].pz() + Mom2partCont[j].pz()),
                                      (Mom2partCont[i].e() + Mom2partCont[j].e()))
                        .m();
      fHistZZMass->Fill(XMass2pairs);
      //cout << " 2-pairs (ZZ) inv. mass = " << XMass2pairs << endl ;
    }
  }

  return;
}

void HZZ4muAnalyzer::endJob() {
  fOutputFile->Write();
  fOutputFile->Close();

  return;
}

typedef HZZ4muAnalyzer HZZ4muTestAnalyzer;
DEFINE_FWK_MODULE(HZZ4muTestAnalyzer);
