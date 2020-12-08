#ifndef PYTHIAFILTERMULTIANCESTOR_h
#define PYTHIAFILTERMULTIANCESTOR_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class PythiaFilterMultiAncestor : public edm::EDFilter {
public:
  explicit PythiaFilterMultiAncestor(const edm::ParameterSet&);
  ~PythiaFilterMultiAncestor() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  bool isAncestor(HepMC::GenParticle* particle, int IDtoMatch);

  edm::EDGetTokenT<edm::HepMCProduct> token_;
  int particleID;
  double minpcut;
  double maxpcut;
  double minptcut;
  double maxptcut;
  double minetacut;
  double maxetacut;
  double minrapcut;
  double maxrapcut;
  double minphicut;
  double maxphicut;

  double rapidity;

  int status;
  std::vector<int> motherIDs;
  std::vector<int> daughterIDs;
  std::vector<double> daughterMinPts;
  std::vector<double> daughterMaxPts;
  std::vector<double> daughterMinEtas;
  std::vector<double> daughterMaxEtas;

  int processID;

  double betaBoost;
};
#endif
