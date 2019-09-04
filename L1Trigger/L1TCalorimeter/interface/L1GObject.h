#ifndef L1GObject_h
#define L1GObject_h

#include <iostream>

#include <string>

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

/**
 * L1GObject represents a calorimeter global trigger object that
 * is made from global calorimeter quantities, total ET, missing ET.
 * These objects are created/filled while processing the calorimeter
 * trigger information at the card level.
 */

class L1GObject : public reco::LeafCandidate {
public:
  // Constructors

  //L1GObject() : myEt(0), myEta(999), myPhi(999), myName("L1GObject") {initialize();}
  L1GObject() {}

  L1GObject(unsigned int et, unsigned int eta, unsigned int phi)
      : myEt(et), myEta(eta), myPhi(phi), myName("L1GObject") {
    initialize();
  }

  L1GObject(unsigned int et, unsigned int eta, unsigned int phi, std::string name)
      : myEt(et), myEta(eta), myPhi(phi), myName(name) {
    initialize();
  }

  L1GObject(unsigned int packedObject, std::string name = "L1GObject") {
    myEt = (packedObject & 0xFFFF0000) >> 16;
    myEta = (packedObject & 0x0000FF00) >> 8;
    myPhi = (packedObject & 0x000000FF);
    myName = name;
    initialize();
  }

  unsigned int packedObject() {
    if (myEt > 0xFFFF)
      myEt = 0xFFFF;
    unsigned int etBits = (myEt << 16);
    if (myEta < 0xFF) {
      unsigned int etaBits = (myEta << 8);
      if (myPhi < 0xFF) {
        return (etBits + etaBits + myPhi);
      }
    }
    std::cerr << "L1GObject: Cannot pack content - fatal error: " << myEt << ", " << myEta << ", " << myPhi
              << std::endl;
    return (etBits);
  }

  L1GObject(const L1GObject& t) : reco::LeafCandidate::LeafCandidate() {
    myName = t.myName;
    myPhi = t.myPhi;
    myEta = t.myEta;
    myEt = t.myEt;
    associatedRegionEt_ = t.associatedRegionEt_;
    associatedJetPt_ = t.associatedJetPt_;
    ellIsolation_ = t.ellIsolation_;
    puLevel_ = t.puLevel_;
    tauVeto_ = t.tauVeto_;
    mipBit_ = t.mipBit_;
    initialize();
  }

  L1GObject& operator=(const L1GObject& t) {
    if (this != &t) {
      myName = t.myName;
      myPhi = t.myPhi;
      myEta = t.myEta;
      myEt = t.myEt;
      associatedRegionEt_ = t.associatedRegionEt_;
      associatedJetPt_ = t.associatedJetPt_;
      ellIsolation_ = t.ellIsolation_;
      puLevel_ = t.puLevel_;
      tauVeto_ = t.tauVeto_;
      mipBit_ = t.mipBit_;
      initialize();
    }
    return *this;
  }

  // Destructor

  ~L1GObject() override {}

  // Access functions

  std::string name() const { return myName; }

  bool empty() const { return false; }

  double ptValue() const { return myLSB * myEt; }

  double etaValue() const {
    if (myEta < 11) {
      return -etaValues[-(myEta - 10)];  // 0-10 are negative eta values
    } else if (myEta < 22) {
      return etaValues[myEta - 11];  // 11-21 are positive eta values
    }
    return 999.;
  }

  double phiValue() const {
    if (myPhi < 18)
      return phiValues[myPhi];
    else
      return 999.;
  }

  unsigned int ptCode() const { return myEt; }

  unsigned int etaIndex() const { return myEta; }

  unsigned int phiIndex() const { return myPhi; }

  // Operators required for sorting lists of these objects

  bool operator==(const L1GObject& t) const {
    if (myEt == t.myEt)
      return true;
    else
      return false;
  }

  bool operator<(const L1GObject& t) const {
    if (myEt < t.myEt)
      return true;
    else
      return false;
  }

  bool operator>(const L1GObject& t) const {
    if (myEt > t.myEt)
      return true;
    else
      return false;
  }

  bool operator<=(const L1GObject& t) const {
    if (myEt <= t.myEt)
      return true;
    else
      return false;
  }

  bool operator>=(const L1GObject& t) const {
    if (myEt >= t.myEt)
      return true;
    else
      return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const L1GObject& t) {
    os << "L1GObject : Name = " << t.name() << "(Et, Eta, Phi) = (" << t.myEt << ", " << t.myEta << ", " << t.myPhi
       << ") (" << t.ptValue() << ", " << t.etaValue() << ", " << t.phiValue() << ")";
    return os;
  }

  void setEt(unsigned int et) { myEt = et; }
  void setEta(unsigned int eta) { myEta = eta; }
  void setPhi(unsigned int phi) { myPhi = phi; }
  void setName(std::string name) { myName = name; }
  void setLSB(double lsb) { myLSB = lsb; }

  void initialize() {
    for (unsigned int i = 0; i < 10; i++) {
      phiValues[i] = 2. * 3.1415927 * i / 18;
    }
    for (unsigned int j = 10; j < 18; j++) {
      phiValues[j] = -3.1415927 + 2. * 3.1415927 * (j - 9) / 18;
    }
    etaValues[0] = 0.174;  // HB and inner HE bins are 0.348 wide
    etaValues[1] = 0.522;
    etaValues[2] = 0.870;
    etaValues[3] = 1.218;
    etaValues[4] = 1.566;
    etaValues[5] = 1.956;  // Last two HE bins are 0.432 and 0.828 wide
    etaValues[6] = 2.586;
    etaValues[7] = 3.250;  // HF bins are 0.5 wide
    etaValues[8] = 3.750;
    etaValues[9] = 4.250;
    etaValues[10] = 4.750;
    myLSB = 1.0;

    // Initialize tuning parameters
    //associatedRegionEt = -1;
    //associatedJetPt = -1;
    //puLevel = -1;

    // Setup the reco::Candidate (physics) 4-vector
    math::PtEtaPhiMLorentzVector myP4(this->ptValue(), this->etaValue(), this->phiValue(), 0);
    this->setP4(myP4);
  }

  // Extra values for tuning UCT parameters - just public members, to
  // eventually be removed
  double associatedJetPt() const { return associatedJetPt_; }
  unsigned int puLevel() const { return puLevel_; }
  double associatedRegionEt() const { return associatedRegionEt_; }
  bool ellIsolation() const { return ellIsolation_; };
  bool tauVeto() const { return tauVeto_; }
  bool mipBit() const { return mipBit_; }

  double associatedJetPt_;
  unsigned int puLevel_;
  double associatedRegionEt_;
  bool ellIsolation_;

  // For the EG objects, don't require this to build the object, just embed it.
  bool tauVeto_;
  bool mipBit_;

private:
  unsigned int myEt;
  unsigned int myEta;
  unsigned int myPhi;
  std::string myName;

  double myLSB;
  double etaValues[11];
  double phiValues[18];
};

#endif
