#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALBarrelProperties.h"
#include "TMath.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ECALBarrelProperties::ECALBarrelProperties(const edm::ParameterSet& fastDet)
{

  using namespace std;

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

  bHom_ = fastDetBarrel.getParameter<bool>("bHom");

  bool debug = fastDetBarrel.getParameter<bool>("debug");

  if (debug)
    LogDebug("ECALEndcapProperties")  <<" ========== Barrel ========= " << endl
	 <<" \t\t isHom ? " << bHom_ << endl
	 <<" lightColl = " << lightColl_ << endl
	 <<" lightCollUnif_ = " <<  lightCollUnif_  << endl
	 <<" photoStatistics_ = " << photoStatistics_ << endl
	 <<" thickness_ = " << thickness_ << endl
	 <<" interactionLength_ = " << interactionLength_ << endl
	 <<" Aeff_ = " << Aeff_ << endl
	 <<" Zeff_ = " << Zeff_ << endl
	 <<" rho_ = " << rho_ << endl
	 <<" radLenIngcm2_ = " << radLenIngcm2_ << endl
	 <<" radLenIncm_ = " << radLenIncm_ << endl
	 <<" moliereRadius_ = " << moliereRadius_ << endl
	 <<" criticalEnergy_ = " << criticalEnergy_ << endl
	 <<" scaleEnergy_ = " << scaleEnergy_ << endl
	 <<" Fs = " << Fs_ << " ehat = " << ehat_ << " resE = " << resE_ << endl;

  
  


}
