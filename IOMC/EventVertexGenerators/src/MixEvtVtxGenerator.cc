/*
*/

#include "IOMC/EventVertexGenerators/interface/MixEvtVtxGenerator.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <iostream>

using namespace edm;
using namespace std;

MixEvtVtxGenerator::MixEvtVtxGenerator(ParameterSet const& pset, edm::ConsumesCollector& iC)
	: fVertex(new HepMC::FourVector()), boost_(),
	  useRecVertex(pset.exists("useRecVertex") ? pset.getParameter<bool>("useRecVertex") : false) {
   vtxOffset.resize(3);
   if(pset.exists("vtxOffset")) vtxOffset=pset.getParameter<std::vector<double> >("vtxOffset");

   if(useRecVertex) {
     useCF_ = 0;
   } else {
     useCF_ = pset.getUntrackedParameter<bool>("useCF",false);
   }
}

MixEvtVtxGenerator::~MixEvtVtxGenerator() {
}

void MixEvtVtxGenerator::generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) {
  HepMC::GenEvent const* inev = product.GetEvent();
  HepMC::GenVertex* genvtx = inev->signal_process_vertex();

  genvtx = inev->signal_process_vertex();
  if(!genvtx) {
    HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
    HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
    while(!genvtx || (genvtx->particles_in_size() == 1 && pt != ptend)) {
      if(pt == ptend) cout<<"End reached, No Gen Vertex!"<<endl;
      genvtx = (*pt)->production_vertex();
      ++pt;
    }
  }

  double aX = genvtx->position().x();
  double aY = genvtx->position().y();
  double aZ = genvtx->position().z();
  double aT = genvtx->position().t();

  LogInfo("MatchVtx")<<" setting vertex "<<" aX "<<aX<<" aY "<<aY<<" aZ "<<aZ<<" aT "<<aT<<endl;
  fVertex->set(aX,aY,aZ,aT);

  product.applyVtxGen(fVertex.get());
  // product.boostToLab(GetInvLorentzBoost(), "vertex");
  // product.boostToLab(GetInvLorentzBoost(), "momentum");
}
