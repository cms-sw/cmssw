#ifndef _JpsieeFilter_h_
#define _JpsieeFilter_h_


// A filter to select J/Psi or Upsilon ->ee 
// The mother type and the type of lepton can be specified. 
// This is a simplified version of BsJpsiPhiFilter from fabstoec


// system include files
#include <memory>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "HepMC/GenVertex.h"


// 


class JpsieeFilter : public edm::EDFilter 
{

public:

  explicit JpsieeFilter(const edm::ParameterSet&);
  ~JpsieeFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:

  struct CutStruct {
    int type;
    double etaMin, etaMax, ptMin;
  };

  typedef std::vector< HepMC::GenParticle * > GenPartVect;
  typedef std::vector< HepMC::GenParticle * >::const_iterator GenPartVectIt;

  //  HepMC::GenParticle * findParticle(const GenPartVect genPartVect, const int requested_id) ;

  HepMC::GenEvent::particle_const_iterator getNextParticle(const HepMC::GenEvent::particle_const_iterator start, 
			       const HepMC::GenEvent::particle_const_iterator end);


  CutStruct leptonCuts ;

  std::string label_;
  int noAccepted;
  std::vector<int> motherId;
};


#endif
