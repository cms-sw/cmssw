//
// $Id: TopEletron.cc,v 1.1 2007/09/20 18:12:22 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"


/// default constructor
TopElectron::TopElectron() : TopLepton<TopElectronType>() {
}


/// constructor from TopElectronType
TopElectron::TopElectron(const TopElectronType & anElectron) : TopLepton<TopElectronType>(anElectron) {
}


/// destructor
TopElectron::~TopElectron() {
}


/// return the tracker isolation variable
double TopElectron::getTrackIso() const {
  return trackIso_;
}


/// return the calorimeter isolation variable
double TopElectron::getCaloIso() const {
  return caloIso_;
}


/// return the lepton ID discriminator
double TopElectron::getLeptonID() const {
  return leptonID_;
}

/// return tracker isolation as calc. by Egamma POG producer
double TopElectron::getEgammaTkIso() const {
  return egammaTkIso_;
}

/// return "number of tracks" isolation as calc. by Egamma POG producer
int TopElectron::getEgammaTkNumIso() const {
  return egammaTkNumIso_;
}

/// return ecal isolation as calc. by Egamma POG producer
double TopElectron::getEgammaEcalIso() const {
  return egammaEcalIso_;
}

/// return hcal isolation as calc. by Egamma POG producer
double TopElectron::getEgammaHcalIso() const {
  return egammaHcalIso_;
}

/// method to set the tracker isolation variable
void TopElectron::setTrackIso(double trackIso) {
  trackIso_ = trackIso;
}


/// method to set the calorimeter isolation variable
void TopElectron::setCaloIso(double caloIso) {
  caloIso_ = caloIso;
}


/// method to set the lepton ID discriminator
void TopElectron::setLeptonID(double id) {
  leptonID_ = id;
}

/// methods to set the isolation from the Egamma POG's producer
void TopElectron::setEgammaTkIso(double iso) {
  egammaTkIso_=iso;
}
void TopElectron::setEgammaTkNumIso(int iso) {
  egammaTkNumIso_=iso;
}
void TopElectron::setEgammaEcalIso(double iso) {
  egammaEcalIso_=iso;
}
void TopElectron::setEgammaHcalIso(double iso) {
  egammaHcalIso_=iso;
}
