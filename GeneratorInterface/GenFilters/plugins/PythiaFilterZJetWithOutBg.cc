/** \class PythiaFilterZJetWithOutBg
 *
 *  PythiaFilterZJetWithOutBg filter implements generator-level preselections
 *  for photon+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 *
 * \author A.Ulyanov, ITEP
 *
 ************************************************************/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

class PythiaFilterZJetWithOutBg : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterZJetWithOutBg(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const double etaMuMax;
  const double ptMuMin;
  const double ptZMin;
  const double ptZMax;
  const double m_z;
  const double dm_z;
};

PythiaFilterZJetWithOutBg::PythiaFilterZJetWithOutBg(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      etaMuMax(iConfig.getUntrackedParameter<double>("MaxMuonEta", 2.5)),
      ptMuMin(iConfig.getUntrackedParameter<double>("MinMuonPt", 3.5)),
      ptZMin(iConfig.getUntrackedParameter<double>("MinZPt")),
      ptZMax(iConfig.getUntrackedParameter<double>("MaxZPt")),
      m_z(91.19),
      dm_z(10.) {}

bool PythiaFilterZJetWithOutBg::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
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
  double emup = std::sqrt(pmup2 + mmup * mmup);
  double emum = std::sqrt(pmum2 + mmum * mmum);

  double massZ = std::sqrt((emup + emum) * (emup + emum) - pxZ * pxZ - pyZ * pyZ - pzZ * pzZ);

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

  return accepted;
}

DEFINE_FWK_MODULE(PythiaFilterZJetWithOutBg);
