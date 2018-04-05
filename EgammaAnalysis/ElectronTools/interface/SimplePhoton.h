#ifndef SimplePhoton_H
#define SimplePhoton_H
 
#ifndef SimpleElectron_STANDALONE
#include "DataFormats/EgammaCandidates/interface/Photon.h"        
#endif

class SimplePhoton {
 public:
  SimplePhoton() {}
 SimplePhoton(double run, 
	      double eClass,
	      double r9, 
	      double scEnergy, 
	      double scEnergyError, 
	      double regEnergy, 
	      double regEnergyError, 
	      double eta, 
	      bool isEB, 
	      bool isMC
	      ) : 
  run_(run),
    eClass_(eClass),
    r9_(r9),
    scEnergy_(scEnergy), 
    scEnergyError_(scEnergyError), 
    regEnergy_(regEnergy), 
    regEnergyError_(regEnergyError), 
    eta_(eta), 
    isEB_(isEB), 
    isMC_(isMC), 
    newEnergy_(regEnergy_), 
    newEnergyError_(regEnergyError_),
    scale_(1.0), smearing_(0.0)
    {}
  ~SimplePhoton() {}	

#ifndef SimplePhoton_STANDALONE
  explicit SimplePhoton(const reco::Photon &in, unsigned int runNumber, bool isMC) ;
  void writeTo(reco::Photon& out) const ;
#endif

  //accessors
  double getNewEnergy() const {return newEnergy_;}
  double getNewEnergyError() const {return newEnergyError_;}
  double getScale() const {return scale_;}
  double getSmearing() const {return smearing_;}
  double getSCEnergy() const {return scEnergy_;}
  double getSCEnergyError() const {return scEnergyError_;}
  double getRegEnergy() const {return regEnergy_;}
  double getRegEnergyError() const {return regEnergyError_;}
  double getEta() const {return eta_;}
  float getR9() const {return r9_;}
  int getElClass() const {return eClass_;}
  int getRunNumber() const {return run_;}
  bool isEB() const {return isEB_;}
  bool isMC() const {return isMC_;}
	    
  //setters
  void setNewEnergy(double newEnergy){newEnergy_ = newEnergy;}
  void setNewEnergyError(double newEnergyError){newEnergyError_ = newEnergyError;}
	    
 private:
  double run_; 
  double eClass_;
  double r9_;
  double scEnergy_; 
  double scEnergyError_; 
  double regEnergy_; 
  double regEnergyError_;
  double eta_;
  bool isEB_;
  bool isMC_;
  double newEnergy_; 
  double newEnergyError_; 
  double scale_; 
  double smearing_;
};

#endif
