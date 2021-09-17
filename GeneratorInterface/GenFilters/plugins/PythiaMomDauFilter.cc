// -*- C++ -*-
//
// Package:    PythiaMomDauFilter
// Class:      PythiaMomDauFilter
//
/**\class PythiaMomDauFilter PythiaMomDauFilter.cc

 Description: Filter events using MotherId and ChildrenIds infos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Oct 27 2015
// Fixed : Ta-Wei Wang, Dec 11 2015
// $Id: PythiaMomDauFilter.h,v 1.1 2015/10/27  pedrini Exp $
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include "HepMC/PythiaWrapper6_4.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <string>
#include <vector>

#define PYCOMP pycomp_
extern "C" {
int PYCOMP(int& ip);
}

// Must be a "one" because it uses a Pythia 6 FORTRAN function
class PythiaMomDauFilter : public edm::one::EDFilter<edm::one::SharedResources> {
public:
  explicit PythiaMomDauFilter(const edm::ParameterSet&);

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> label_;
  std::vector<int> dauIDs;
  std::vector<int> desIDs;
  int particleID;
  int daughterID;
  bool chargeconju;
  int ndaughters;
  int ndescendants;
  double minptcut;
  double maxptcut;
  double minetacut;
  double maxetacut;
  double mom_minptcut;
  double mom_maxptcut;
  double mom_minetacut;
  double mom_maxetacut;
  double betaBoost;
};

using namespace std;

PythiaMomDauFilter::PythiaMomDauFilter(const edm::ParameterSet& iConfig)
    : label_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleID(iConfig.getUntrackedParameter<int>("ParticleID", 0)),
      daughterID(iConfig.getUntrackedParameter<int>("DaughterID", 0)),
      chargeconju(iConfig.getUntrackedParameter<bool>("ChargeConjugation", true)),
      ndaughters(iConfig.getUntrackedParameter<int>("NumberDaughters", 0)),
      ndescendants(iConfig.getUntrackedParameter<int>("NumberDescendants", 0)),
      minptcut(iConfig.getUntrackedParameter<double>("MinPt", 0.)),
      maxptcut(iConfig.getUntrackedParameter<double>("MaxPt", 14000.)),
      minetacut(iConfig.getUntrackedParameter<double>("MinEta", -10.)),
      maxetacut(iConfig.getUntrackedParameter<double>("MaxEta", 10.)),
      mom_minptcut(iConfig.getUntrackedParameter<double>("MomMinPt", 0.)),
      mom_maxptcut(iConfig.getUntrackedParameter<double>("MomMaxPt", 14000.)),
      mom_minetacut(iConfig.getUntrackedParameter<double>("MomMinEta", -10.)),
      mom_maxetacut(iConfig.getUntrackedParameter<double>("MomMaxEta", 10.)),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)) {
  usesResource(edm::SharedResourceNames::kPythia6);

  //now do what ever initialization is needed
  vector<int> defdauID;
  defdauID.push_back(0);
  vector<int> defdesID;
  defdesID.push_back(0);
  dauIDs = iConfig.getUntrackedParameter<vector<int> >("DaughterIDs", defdauID);
  desIDs = iConfig.getUntrackedParameter<vector<int> >("DescendantsIDs", defdesID);
}

bool PythiaMomDauFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;
  bool mom_accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(label_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();
  int ndauac = 0;
  int ndau = 0;
  int ndesac = 0;
  int ndes = 0;

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->pdg_id() != particleID)
      continue;
    HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
    if ((*p)->momentum().perp() > mom_minptcut && (*p)->momentum().perp() < mom_maxptcut && mom.eta() > mom_minetacut &&
        mom.eta() < mom_maxetacut) {
      mom_accepted = true;
      ndauac = 0;
      ndau = 0;
      ndesac = 0;
      ndes = 0;
      if ((*p)->end_vertex()) {
        for (HepMC::GenVertex::particle_iterator dau = (*p)->end_vertex()->particles_begin(HepMC::children);
             dau != (*p)->end_vertex()->particles_end(HepMC::children);
             ++dau) {
          ++ndau;
          for (unsigned int i = 0; i < dauIDs.size(); ++i) {
            if ((*dau)->pdg_id() != dauIDs[i])
              continue;
            ++ndauac;
            break;
          }
          if ((*dau)->pdg_id() == daughterID) {
            for (HepMC::GenVertex::particle_iterator des = (*dau)->end_vertex()->particles_begin(HepMC::children);
                 des != (*des)->end_vertex()->particles_end(HepMC::children);
                 ++des) {
              ++ndes;
              for (unsigned int i = 0; i < desIDs.size(); ++i) {
                if ((*des)->pdg_id() != desIDs[i])
                  continue;
                HepMC::FourVector dau_i = MCFilterZboostHelper::zboost((*des)->momentum(), betaBoost);
                if ((*des)->momentum().perp() > minptcut && (*des)->momentum().perp() < maxptcut &&
                    dau_i.eta() > minetacut && dau_i.eta() < maxetacut) {
                  ++ndesac;
                  break;
                }
              }
            }
          }
        }
      }
    }
    if (ndau == ndaughters && ndauac == ndaughters && mom_accepted && ndes == ndescendants && ndesac == ndescendants) {
      accepted = true;
      break;
    }
  }

  if (!accepted && chargeconju) {
    mom_accepted = false;
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      if ((*p)->pdg_id() != -particleID)
        continue;
      HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
      if ((*p)->momentum().perp() > mom_minptcut && (*p)->momentum().perp() < mom_maxptcut &&
          mom.eta() > mom_minetacut && mom.eta() < mom_maxetacut) {
        mom_accepted = true;
        ndauac = 0;
        ndau = 0;
        ndesac = 0;
        ndes = 0;
        if ((*p)->end_vertex()) {
          for (HepMC::GenVertex::particle_iterator dau = (*p)->end_vertex()->particles_begin(HepMC::children);
               dau != (*p)->end_vertex()->particles_end(HepMC::children);
               ++dau) {
            ++ndau;
            for (unsigned int i = 0; i < dauIDs.size(); ++i) {
              int IDanti = -dauIDs[i];
              int pythiaCode = PYCOMP(dauIDs[i]);
              int has_antipart = pydat2.kchg[3 - 1][pythiaCode - 1];
              if (has_antipart == 0)
                IDanti = dauIDs[i];
              if ((*dau)->pdg_id() != IDanti)
                continue;
              ++ndauac;
              break;
            }
            int daughterIDanti = -daughterID;
            int pythiaCode = PYCOMP(daughterID);
            int has_antipart = pydat2.kchg[3 - 1][pythiaCode - 1];
            if (has_antipart == 0)
              daughterIDanti = daughterID;
            if ((*dau)->pdg_id() == daughterIDanti) {
              for (HepMC::GenVertex::particle_iterator des = (*dau)->end_vertex()->particles_begin(HepMC::children);
                   des != (*des)->end_vertex()->particles_end(HepMC::children);
                   ++des) {
                ++ndes;
                for (unsigned int i = 0; i < desIDs.size(); ++i) {
                  int IDanti = -desIDs[i];
                  int pythiaCode = PYCOMP(desIDs[i]);
                  int has_antipart = pydat2.kchg[3 - 1][pythiaCode - 1];
                  if (has_antipart == 0)
                    IDanti = desIDs[i];
                  if ((*des)->pdg_id() != IDanti)
                    continue;
                  HepMC::FourVector dau_i = MCFilterZboostHelper::zboost((*des)->momentum(), betaBoost);
                  if ((*des)->momentum().perp() > minptcut && (*des)->momentum().perp() < maxptcut &&
                      dau_i.eta() > minetacut && dau_i.eta() < maxetacut) {
                    ++ndesac;
                    break;
                  }
                }
              }
            }
          }
        }
      }
      if (ndau == ndaughters && ndauac == ndaughters && mom_accepted && ndes == ndescendants &&
          ndesac == ndescendants) {
        accepted = true;
        break;
      }
    }
  }
  return accepted;
}

DEFINE_FWK_MODULE(PythiaMomDauFilter);
