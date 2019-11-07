#ifndef PythiaFilterZJetWithOutBg_h
#define PythiaFilterZJetWithOutBg_h

/** \class PythiaFilterZJetWithOutBg
 *
 *  PythiaFilterZJetWithOutBg filter implements generator-level preselections 
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

class PythiaFilterZJetWithOutBg : public edm::EDFilter {
public:
  explicit PythiaFilterZJetWithOutBg(const edm::ParameterSet&);
  ~PythiaFilterZJetWithOutBg() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  double etaMuMax;
  double ptMuMin;
  double ptZMin;
  double ptZMax;
  double m_z;
  double dm_z;
  int nmu;

  int theNumberOfSelected;
  int maxnumberofeventsinrun;
};
#endif
