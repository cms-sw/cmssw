//
// $Id$
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
