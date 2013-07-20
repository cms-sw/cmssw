#ifndef METSIG_SIGINPUTOBJ_H
#define METSIG_SIGINPUTOBJ_H
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SigInputObj
// 
/**\class METSignificance SigInputObj.h DataFormats/METReco/include/SigInputObj.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: SigInputObj.h,v 1.1 2012/08/31 08:57:55 veelken Exp $
//
//



#include <vector>
#include <string>
#include <iostream>
#include <sstream>

//=== Class SigInputObj ==============================//
namespace metsig{
  class SigInputObj{

  public:
    SigInputObj():
      type(""),energy(0.),phi(0.),sigma_e(0.),sigma_tan(0.)
      {;}// default constructor

    SigInputObj( const std::string& m_type,  double m_energy, 
		 double m_phi, double m_sigm_e,  double m_sigma_phi);
    ~SigInputObj() {;}
    
    std::string get_type() const {return(type);};
    double get_energy() const {return(energy);};
    double get_phi() const {return(phi);};
    double get_sigma_e() const {return(sigma_e);};
    double get_sigma_tan() const {return(sigma_tan);};
    
    void set(const std::string & m_type, const double & m_energy,
	     const double & m_phi, const double & m_sigma_e,
	     const double & m_sigma_tan){
      type.clear(); type.append(m_type);
      energy = m_energy;
      phi = m_phi;
      sigma_e = m_sigma_e;
      sigma_tan = m_sigma_tan;
    }

  private:
    std::string type;       //type of physics object
    /*** note: type = "jet", "uc" (un-clustered energy),
	 "electron", "muon", "hot-cell", "vertex" ***/
    double energy;    //magnitude of the energy
    double phi;       //azimuthal angle
    double sigma_e;   //gaus width in radial direction
    double sigma_tan; //gaus width in phi-hat direction (not in rad)
    
    void set_type(const std::string & m_type){type.clear(); type.append(m_type);};
    void set_energy(const double & m_energy){energy=m_energy;};
    void set_phi(const double & m_phi){phi=m_phi;};
    void set_sigma_e(const double & m_sigma_e){sigma_e=m_sigma_e;};
    void set_sigma_tan(const double & m_sigma_tan){sigma_tan=m_sigma_tan;};
  };
}

#endif
