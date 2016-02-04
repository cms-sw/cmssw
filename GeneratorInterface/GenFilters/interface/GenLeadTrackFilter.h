#ifndef GenLeadTrackFilter_h
#define GenLeadTrackFilter_h

/***********************************************************
*                 GenLeadTrackFilter                       *
*                 ------------------                       *
*                                                          *
* Original Author: Souvik Das, Cornell University          *
* Created        : 7 August 2009                           *
*                                                          *
*  Allows events which have at least one generator level   *
*  charged particle with pT greater than X GeV within      *
*  |eta| less than Y, where X and Y are specified in the   *
*  cfi configuration file.                                 *
***********************************************************/

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Particle.h"

class GenLeadTrackFilter : public edm::EDFilter 
{
  public:
  explicit GenLeadTrackFilter(const edm::ParameterSet&);
  ~GenLeadTrackFilter();

  private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------
  edm::InputTag hepMCProduct_label_;
  double   genLeadTrackPt_,
           genEta_;
};

#endif
