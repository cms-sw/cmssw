#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALEndcapProperties.h"
#include "TMath.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ECALEndcapProperties::ECALEndcapProperties(const edm::ParameterSet& fastDet)
{

  using namespace std;

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

  bHom_ = fastDetEndcap.getParameter<bool>("bHom");

  bool debug = fastDetEndcap.getParameter<bool>("debug");

  if (debug)
    LogDebug("ECALEndcapProperties") <<" ========== Endcap ========= " << endl
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
