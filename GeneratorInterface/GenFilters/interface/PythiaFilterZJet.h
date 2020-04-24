#ifndef PythiaFilterZJet_h
#define PythiaFilterZJet_h

/** \class PythiaFilterZJet
 *
 *  PythiaFilterZJet filter implements generator-level preselections 
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

class PythiaFilterZJet : public edm::EDFilter {
   public:
      explicit PythiaFilterZJet(const edm::ParameterSet&);
      ~PythiaFilterZJet();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       double etaMuMax;
       double ptZMin;
       double ptZMax;

       int theNumberOfSelected;
       int maxnumberofeventsinrun;

};
#endif
