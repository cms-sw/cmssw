
#include "GeneratorInterface/HepMCFilters/interface/GenericDauHepMCFilter.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


// GenericDauHepMCFilter::GenericDauHepMCFilter(const edm::ParameterSet& iConfig) :
//         particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
//         chargeconju(iConfig.getUntrackedParameter("ChargeConjugation", true)),
//         ndaughters(iConfig.getUntrackedParameter("NumberDaughters", 0)),
//         minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
//         maxptcut(iConfig.getUntrackedParameter("MaxPt", 14000.)),
//         minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
//         maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.))
// {
//         //now do what ever initialization is needed
//         vector<int> defdauID;
//         defdauID.push_back(0);
//         dauIDs = iConfig.getUntrackedParameter< vector<int> >("DaughterIDs",defdauID);
// 
// }

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

          //     cout << "=> found a particle with : " << particleID << endl;
          //
          //     if ( (*p)->production_vertex() ) {
          //       for ( HepMC::GenVertex::particle_iterator 
          //             mother = (*p)->production_vertex()->particles_begin(HepMC::parents);
          //                          mother != (*p)->production_vertex()->particles_end(HepMC::parents); 
          //                          ++mother ) {
          //                        std::cout << "\t";
          //                        (*mother)->print();
          //       }
          // }

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
