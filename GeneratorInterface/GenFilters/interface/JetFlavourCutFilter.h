#ifndef _JetFlavourCutCutFilter_h_
#define _JetFlavourCutCutFilter_h_


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

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"


// 


class JetFlavourCutFilter : public edm::EDFilter 
{

public:

  explicit JetFlavourCutFilter(const edm::ParameterSet&);
  ~JetFlavourCutFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:

  typedef std::vector< HepMC::GenParticle * > GenPartVect;
  typedef std::vector< HepMC::GenParticle * >::const_iterator GenPartVectIt;

  HepMC::GenParticle * findParticle(const GenPartVect& genPartVect, const int requested_id) ;

  void printHisto(const HepMC::GenEvent::particle_iterator start, 
			       const HepMC::GenEvent::particle_iterator end);


  int jetType;

  edm::EDGetTokenT<edm::HepMCProduct> token_;
  int noAccepted;
};


#endif
