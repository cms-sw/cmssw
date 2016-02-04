#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SigInputObj
// 
/**\class METSignificance SigInputObj.cc RecoMET/METAlgorithms/src/SigInputObj.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: SigInputObj.cc,v 1.1 2008/04/18 10:12:55 fblekman Exp $
//
//

//=== Constructors ===============================//
metsig::SigInputObj::SigInputObj( std::string & m_type,  double & m_energy,  double & m_phi,
		  double & m_sigma_e,  double & m_sigma_tan) 
{
  type.clear(); 
  type.append(m_type);
  energy = m_energy;
  phi = m_phi;
  sigma_e = m_sigma_e;
  sigma_tan = m_sigma_tan;
}

//================================================//

//=== Methods ====================================//
// none yet...
//================================================//

