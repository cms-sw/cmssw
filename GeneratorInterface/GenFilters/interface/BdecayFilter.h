#ifndef _BdecayFilter_h_
#define _BdecayFilter_h_


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

#include "HepMC/GenVertex.h"


// 


class BdecayFilter : public edm::EDFilter 
{

public:

  explicit BdecayFilter(const edm::ParameterSet&);
  ~BdecayFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:

  struct CutStruct {
    int type;
    std::vector<int> decayProduct;
    double etaMin, etaMax, ptMin;
  };

  typedef std::vector< HepMC::GenParticle * > GenPartVect;
  typedef std::vector< HepMC::GenParticle * >::const_iterator GenPartVectIt;

  //  HepMC::GenParticle * findParticle(const GenPartVect genPartVect, const int requested_id) ;

  //***
  HepMC::GenParticle * findParticle(HepMC::GenVertex* , const int requested_id) ;
  //***

  HepMC::GenEvent::particle_const_iterator getNextBs(const HepMC::GenEvent::particle_const_iterator start, 
						     const HepMC::GenEvent::particle_const_iterator end);
  
  
  bool cuts(const HepMC::GenParticle * jpsi, const CutStruct& cut);
  bool etaInRange(float eta, float etamin, float etamax);

  CutStruct firstDaughter, secondDaughter;

  std::string label_;
  int noAccepted;
  int motherParticle;
};


#endif
