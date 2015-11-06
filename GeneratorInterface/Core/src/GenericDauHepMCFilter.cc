
#include "GeneratorInterface/Core/interface/GenericDauHepMCFilter.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


GenericDauHepMCFilter::GenericDauHepMCFilter(const edm::ParameterSet& iConfig) :
        particleID(iConfig.getParameter<int>("ParticleID")),
        chargeconju(iConfig.getParameter<bool>("ChargeConjugation")),
        ndaughters(iConfig.getParameter<int>("NumberDaughters")),
        dauIDs(iConfig.getParameter<std::vector<int> >("DaughterIDs")),
        minptcut(iConfig.getParameter<double>("MinPt")),
        maxptcut(iConfig.getParameter<double>("MaxPt")),
        minetacut(iConfig.getParameter<double>("MinEta")),
        maxetacut(iConfig.getParameter<double>("MaxEta"))
{

}


GenericDauHepMCFilter::~GenericDauHepMCFilter()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool GenericDauHepMCFilter::filter(const HepMC::GenEvent* evt)
{
  
  for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
                  p != evt->particles_end(); ++p ) {

          if( (*p)->pdg_id() != particleID ) continue ;

          int ndauac = 0;
          int ndau = 0;     
          if ( (*p)->end_vertex() ) {     
                  for ( HepMC::GenVertex::particle_iterator 
                                  des=(*p)->end_vertex()->particles_begin(HepMC::children);
                                  des != (*p)->end_vertex()->particles_end(HepMC::children);
                                  ++des ) {
                          ++ndau;       
                          //       cout << "   -> Daughter = " << (*des)->pdg_id() << endl;
                          for( unsigned int i=0; i<dauIDs.size(); ++i) {
                                  if( (*des)->pdg_id() != dauIDs[i] ) continue ;
                                  if(   (*des)->momentum().perp() >  minptcut  &&
                                                  (*des)->momentum().perp() <  maxptcut  &&
                                                  (*des)->momentum().eta()  >  minetacut && 
                                                  (*des)->momentum().eta()  <  maxetacut ) {
                                          ++ndauac;
                                          //           cout << "     -> pT = " << (*des)->momentum().perp() << endl;
                                          break;
                                  } 
                          }                            
                  }
          }  
          if( ndau ==  ndaughters && ndauac == ndaughters ) {
                  return true;
          }    

  }


  if (chargeconju) {

          for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
                          p != evt->particles_end(); ++p ) {

                  if( (*p)->pdg_id() != -particleID ) continue ;
                  int ndauac = 0;
                  int ndau = 0;     
                  if ( (*p)->end_vertex() ) {
                          for ( HepMC::GenVertex::particle_iterator 
                                          des=(*p)->end_vertex()->particles_begin(HepMC::children);
                                          des != (*p)->end_vertex()->particles_end(HepMC::children);
                                          ++des ) {
                                  ++ndau;
                                  for( unsigned int i=0; i<dauIDs.size(); ++i) {
                                          bool has_antipart = !(dauIDs[i]==22 || dauIDs[i]==23);
                                          int IDanti = has_antipart ? -dauIDs[i] : dauIDs[i];
                                          if( (*des)->pdg_id() != IDanti ) continue ;
                                          if(   (*des)->momentum().perp() >  minptcut  &&
                                                          (*des)->momentum().perp() <  maxptcut  &&
                                                          (*des)->momentum().eta()  >  minetacut && 
                                                          (*des)->momentum().eta()  <  maxetacut ) {
                                                  ++ndauac;
                                                  break;
                                          } 
                                  }                            
                          }
                  }
                  if( ndau ==  ndaughters && ndauac == ndaughters ) {
                          return true;
                  }    
          }

  }    

  return false;

}
