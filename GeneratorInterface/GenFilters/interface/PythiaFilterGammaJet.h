#ifndef PythiaFilterGammaJet_h
#define PythiaFilterGammaJet_h

/** \class PythiaFilterGammaJet
 *
 *  PythiaFilterGammaJet filter implements generator-level preselections 
 *  for photon+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 * 
 * \author A.Ulyanov, ITEP
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class HepMCProduct;
}

class PythiaFilterGammaJet : public edm::EDFilter {
public:
  explicit PythiaFilterGammaJet(const edm::ParameterSet&);
  ~PythiaFilterGammaJet() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  double etaMax;
  double ptSeed;
  double ptMin;
  double ptMax;
  double dphiMin;
  double detaMax;
  double etaPhotonCut2;

  double cone;
  double ebEtaMax;
  double deltaEB;
  double deltaEE;

  int theNumberOfSelected;
  int maxnumberofeventsinrun;
};
#endif
