#include "GeneratorInterface/GenFilters/plugins/PythiaFilterZJetWithOutBg.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <iostream>
#include <list>
#include <vector>
#include <cmath>

PythiaFilterZJetWithOutBg::PythiaFilterZJetWithOutBg(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      etaMuMax(iConfig.getUntrackedParameter<double>("MaxMuonEta", 2.5)),
      ptMuMin(iConfig.getUntrackedParameter<double>("MinMuonPt", 3.5)),
      ptZMin(iConfig.getUntrackedParameter<double>("MinZPt")),
      ptZMax(iConfig.getUntrackedParameter<double>("MaxZPt")),
      maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents", 10000)) {
  m_z = 91.19;
  dm_z = 10.;
  theNumberOfSelected = 0;
}

PythiaFilterZJetWithOutBg::~PythiaFilterZJetWithOutBg() {}

// ------------ method called to produce the data  ------------
bool PythiaFilterZJetWithOutBg::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
  //    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
  //  }

  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  std::vector<const HepMC::GenParticle*> mu;

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if (std::abs((*p)->pdg_id()) == 13 && (*p)->status() == 1)
      mu.push_back(*p);
    if (mu.size() > 1)
      break;
  }

  //    std::cout<<" Number of muons "<<mu.size()<<" "<<mu[0]->pdg_id()<<" "<<mu[1]->pdg_id()<<std::endl;

  if (mu.size() != 2)
    return false;

  if (mu[0]->pdg_id() * (mu[1]->pdg_id()) > 0) {
    return false;
  }

  //      std::cout<<" Muons sign accept "<<mu[0]->momentum().perp()<<" "<<mu[1]->momentum().perp()<<std::endl;

  if (mu[0]->momentum().perp() < ptMuMin || mu[1]->momentum().perp() < ptMuMin)
    return false;

  //    std::cout<<" Muons pt accept "<<std::fabs(mu[0]->momentum().eta())<<" "<<std::fabs(mu[1]->momentum().eta())<<std::endl;

  if (std::fabs(mu[0]->momentum().eta()) > etaMuMax)
    return false;
  if (std::fabs(mu[1]->momentum().eta()) > etaMuMax)
    return false;

  double mmup = mu[0]->generatedMass();
  double mmum = mu[1]->generatedMass();
  double pxZ = mu[0]->momentum().x() + mu[1]->momentum().x();
  double pyZ = mu[0]->momentum().y() + mu[1]->momentum().y();
  double pzZ = mu[0]->momentum().z() + mu[1]->momentum().z();

  double pmup2 = mu[0]->momentum().x() * mu[0]->momentum().x() + mu[0]->momentum().y() * mu[0]->momentum().y() +
                 mu[0]->momentum().z() * mu[0]->momentum().z();
  double pmum2 = mu[1]->momentum().x() * mu[1]->momentum().x() + mu[1]->momentum().y() * mu[1]->momentum().y() +
                 mu[1]->momentum().z() * mu[1]->momentum().z();
  double emup = sqrt(pmup2 + mmup * mmup);
  double emum = sqrt(pmum2 + mmum * mmum);

  double massZ = sqrt((emup + emum) * (emup + emum) - pxZ * pxZ - pyZ * pyZ - pzZ * pzZ);

  //    std::cout<<" Muons eta accept "<<massZ<<std::endl;

  if (std::fabs(massZ - m_z) > dm_z)
    return false;

  //    double ptZ= (mu[0]->momentum() + mu[1]->momentum()).perp();
  //    std::cout<<" MassZ accept "<<ptZ<<std::endl;

  math::XYZTLorentzVector tot_mom(mu[0]->momentum());
  math::XYZTLorentzVector mom2(mu[1]->momentum());
  tot_mom += mom2;
  //    double ptZ= (mu[0]->momentum() + mu[1]->momentum()).perp();
  double ptZ = tot_mom.pt();

  if (ptZ > ptZMin && ptZ < ptZMax)
    accepted = true;

  if (accepted) {
    theNumberOfSelected++;
    std::cout << " Event accept " << theNumberOfSelected << std::endl;
    return true;
  } else
    return false;
}
