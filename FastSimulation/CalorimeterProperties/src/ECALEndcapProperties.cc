#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALEndcapProperties.h"
#include "TMath.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ECALEndcapProperties::ECALEndcapProperties(const edm::ParameterSet& fastDet)
{

  edm::ParameterSet fastDetEndcap = fastDet.getParameter<edm::ParameterSet>("EndcapCalorimeterProperties");

  lightColl_ =  fastDetEndcap.getParameter<double>("lightColl");  
  lightCollUnif_ = fastDetEndcap.getParameter<double>("lightCollUnif");
  photoStatistics_ = fastDetEndcap.getParameter<double>("photoStatistics");
  thickness_ = fastDetEndcap.getParameter<double>("thickness");
  interactionLength_ = fastDetEndcap.getParameter<double>("interactionLength");

  Aeff_ = fastDetEndcap.getParameter<double>("Aeff");
  Zeff_ = fastDetEndcap.getParameter<double>("Zeff");
  rho_ = fastDetEndcap.getParameter<double>("rho");
  radLenIngcm2_ = fastDetEndcap.getParameter<double>("radLenIngcm2");

  // Parameters that might be calculated out of the formulas

  radLenIncm_ = fastDetEndcap.getParameter<double>("radLenIncm");
  radLenIncm_ = (radLenIncm_ < 0) ? radLenIngcm2_/rho_ : radLenIncm_;

  criticalEnergy_ = fastDetEndcap.getParameter<double>("criticalEnergy");
  criticalEnergy_ = (criticalEnergy_ < 0) ? 2.66E-3*TMath::Power((radLenIngcm2_*Zeff_/Aeff_),1.1) : criticalEnergy_;

  moliereRadius_  = fastDetEndcap.getParameter<double>("moliereRadius");
  moliereRadius_  = (moliereRadius_ < 0) ? scaleEnergy_/criticalEnergy_*radLenIncm_ : moliereRadius_;

  Fs_ = fastDetEndcap.getParameter<double>("Fs");
  ehat_ = fastDetEndcap.getParameter<double>("ehat");
  resE_ = fastDetEndcap.getParameter<double>("resE");

  da_ = fastDetEndcap.getParameter<double>("da");
  dp_ = fastDetEndcap.getParameter<double>("dp");

  bHom_ = fastDetEndcap.getParameter<bool>("bHom");

  bool debug = fastDetEndcap.getParameter<bool>("debug");

  if (debug)
    edm::LogInfo("ECALProperties") <<" ========== Endcap ========= \n"
				   <<" isHom ? " << bHom_ << "\n"
				   <<" da = " << da_ << " dp = " << dp_ 
				   <<" lightColl = " << lightColl_ << "\n"
				   <<" lightCollUnif_ = " <<  lightCollUnif_  << "\n"
				   <<" photoStatistics_ = " << photoStatistics_ << " photons/GeV\n"
				   <<" thickness_ = " << thickness_ << " in cm \n"
				   <<" interactionLength_ = " << interactionLength_ << " cm \n"
				   <<" Aeff_ = " << Aeff_ << "\n"
				   <<" Zeff_ = " << Zeff_ << "\n"
				   <<" rho_ = " << rho_ << " g/cm3\n"
				   <<" radLenIngcm2_ = " << radLenIngcm2_ << " g*cm2\n"
				   <<" radLenIncm_ = " << radLenIncm_ << " cm\n"
				   <<" moliereRadius_ = " << moliereRadius_ << " cm\n"
				   <<" criticalEnergy_ = " << criticalEnergy_ << " GeV\n"	 
				   <<" scaleEnergy_ = " << scaleEnergy_ << " GeV\n"
				   <<" Fs = " << Fs_ << " ehat = " << ehat_ << " resE = " << resE_ << "\n";

}
