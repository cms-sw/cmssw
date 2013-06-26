#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALBarrelProperties.h"
#include "TMath.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ECALBarrelProperties::ECALBarrelProperties(const edm::ParameterSet& fastDet)
{
 
  edm::ParameterSet fastDetBarrel = fastDet.getParameter<edm::ParameterSet>("BarrelCalorimeterProperties");

  lightColl_ =  fastDetBarrel.getParameter<double>("lightColl");  
  lightCollUnif_ = fastDetBarrel.getParameter<double>("lightCollUnif");
  photoStatistics_ = fastDetBarrel.getParameter<double>("photoStatistics");
  thickness_ = fastDetBarrel.getParameter<double>("thickness");
  interactionLength_ = fastDetBarrel.getParameter<double>("interactionLength");

  Aeff_ = fastDetBarrel.getParameter<double>("Aeff");
  Zeff_ = fastDetBarrel.getParameter<double>("Zeff");
  rho_ = fastDetBarrel.getParameter<double>("rho");
  radLenIngcm2_ = fastDetBarrel.getParameter<double>("radLenIngcm2");

  radLenIncm_ = fastDetBarrel.getParameter<double>("radLenIncm");
  radLenIncm_ = (radLenIncm_ < 0) ? radLenIngcm2_/rho_ : radLenIncm_;

  criticalEnergy_ = fastDetBarrel.getParameter<double>("criticalEnergy");
  criticalEnergy_ = (criticalEnergy_ < 0) ? 2.66E-3*TMath::Power((radLenIngcm2_*Zeff_/Aeff_),1.1) : criticalEnergy_;

  moliereRadius_  = fastDetBarrel.getParameter<double>("moliereRadius");
  moliereRadius_  = (moliereRadius_ < 0) ? scaleEnergy_/criticalEnergy_*radLenIncm_ : moliereRadius_;

  Fs_ = fastDetBarrel.getParameter<double>("Fs");
  ehat_ = fastDetBarrel.getParameter<double>("ehat");
  resE_ = fastDetBarrel.getParameter<double>("resE");

  da_ = fastDetBarrel.getParameter<double>("da");
  dp_ = fastDetBarrel.getParameter<double>("dp");

  bHom_ = fastDetBarrel.getParameter<bool>("bHom");

  bool debug = fastDetBarrel.getParameter<bool>("debug");


  if (debug)
    edm::LogInfo("ECALProperties")  <<" ========== Barrel ========= " << "\n"
				    <<" isHom ? " << bHom_ << "\n"
				    <<" da = " << da_ << " dp = " << dp_ 
				    <<" lightColl = " << lightColl_ << "\n"
				    <<" lightCollUnif_ = " <<  lightCollUnif_  << "\n"
				    <<" photoStatistics_ = " << photoStatistics_ << " photons/GeV\n"
				    <<" thickness_ = " << thickness_ << " cm\n"
				    <<" interactionLength_ = " << interactionLength_ << " cm\n"
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
