#ifndef PythiaFilterIsolatedTrack_h
#define PythiaFilterIsolatedTrack_h

/** \class PythiaFilterIsolatedTrack
 *
 *  PythiaFilterGammaJet filter implements generator-level preselections 
 *  for ChargedHadron+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 * 
 * \author O.Kodolova, SINP
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


class PythiaFilterIsolatedTrack : public edm::EDFilter {
   public:
      explicit PythiaFilterIsolatedTrack(const edm::ParameterSet&);
      ~PythiaFilterIsolatedTrack();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      
       std::string label_;
       double etaMax;
       double ptSeed;
       double cone;
       double ebEtaMax;
       double deltaEB;
       double deltaEE;
       int theNumberOfSelected;

};
#endif
