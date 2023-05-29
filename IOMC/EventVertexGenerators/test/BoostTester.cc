#include <iostream>

#include "TTree.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class BoostTester : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BoostTester(const edm::ParameterSet&);
  ~BoostTester() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  TTree* ftreevtx;
  TTree* ftreep;

  double fvx, fvy, fvz;
  double fpx, fpy, fpz, fpt, fp, fe, feta, fphi;
};

BoostTester::BoostTester(const edm::ParameterSet&) {
  usesResource(TFileService::kSharedResource);

  edm::Service<TFileService> tfile;
  ftreevtx = tfile->make<TTree>("vtxtree", "vtxtree");
  ftreevtx->Branch("vx", &fvx, "fvx/D");
  ftreevtx->Branch("vy", &fvy, "fvy/D");
  ftreevtx->Branch("vz", &fvz, "fvz/D");

  ftreep = tfile->make<TTree>("ptree", "ptree");
  ftreep->Branch("px", &fpx, "fpx/D");
  ftreep->Branch("py", &fpy, "fpy/D");
  ftreep->Branch("pz", &fpz, "fpz/D");
}

void BoostTester::analyze(const edm::Event& e, const edm::EventSetup&) {
  ftreevtx->SetBranchAddress("vx", &fvx);
  ftreevtx->SetBranchAddress("vy", &fvy);
  ftreevtx->SetBranchAddress("vz", &fvz);

  ftreep->SetBranchAddress("px", &fpx);
  ftreep->SetBranchAddress("py", &fpy);
  ftreep->SetBranchAddress("pz", &fpz);

  fpx = 0.;
  fpy = 0.;
  fpz = 0.;

  std::vector<edm::Handle<edm::HepMCProduct> > EvtHandles;
  e.getManyByType(EvtHandles);

  edm::LogVerbatim("BoostTest") << "evthandles= " << EvtHandles.size();

  for (unsigned int i = 0; i < EvtHandles.size(); i++) {
    edm::LogVerbatim("BoostTest") << " i=" << i << " name: " << EvtHandles[i].provenance()->moduleLabel();

    if (EvtHandles[i].isValid()) {
      const HepMC::GenEvent* Evt = EvtHandles[i]->GetEvent();

      // take only 1st vertex for now - it's been tested only of PGuns...
      //

      for (auto Vtx = Evt->vertices_begin(); Vtx != Evt->vertices_end(); ++Vtx) {
        fvx = (*Vtx)->position().x();
        fvy = (*Vtx)->position().y();
        fvz = (*Vtx)->position().z();

        ftreevtx->Fill();

        edm::LogVerbatim("BoostTest") << " vertex (x,y,z)= " << (*Vtx)->position().x() << " " << (*Vtx)->position().y()
                                      << " " << (*Vtx)->position().z();
      }

      for (HepMC::GenEvent::particle_const_iterator Part = Evt->particles_begin(); Part != Evt->particles_end();
           ++Part) {
        if ((*Part)->status() != 1)
          continue;

        HepMC::FourVector Mon = (*Part)->momentum();

        fpx += Mon.px();
        fpy += Mon.py();
        fpz += Mon.pz();

        edm::LogVerbatim("BoostTest") << "particle: p=(" << Mon.px() << ", " << Mon.py() << ", " << Mon.pz()
                                      << ") status=" << (*Part)->status() << " pdgid=" << (*Part)->pdg_id();
      }
    }
  }
  edm::LogVerbatim("BoostTest") << " total px= " << fpx << " py= " << fpy << " pz= " << fpz;

  ftreep->Fill();
  return;
}

DEFINE_FWK_MODULE(BoostTester);
