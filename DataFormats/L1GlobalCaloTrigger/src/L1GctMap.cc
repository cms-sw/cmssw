
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctMap.h"

#include "FWCore/Utilities/interface/Exception.h"

L1GctMap* L1GctMap::m_instance = 0;

const unsigned L1GctMap::N_RGN_PHI = 18;
const unsigned L1GctMap::N_RGN_ETA = 22;


/// constructor
L1GctMap::L1GctMap() { 

}

/// destructor
L1GctMap::~L1GctMap() {

}

/// get the RCT crate number
unsigned L1GctMap::rctCrate(L1GctRegion r) {
  unsigned eta = this->eta(r);
  unsigned phiCrate = ((N_RGN_PHI+4-(this->phi(r)))%N_RGN_PHI)/2;
  return (eta<(N_RGN_ETA/2) ? phiCrate : phiCrate+N_RGN_PHI/2) ;
}

/// get the SC number
unsigned L1GctMap::sourceCard(L1GctRegion r) {
  return 3*(this->rctCrate(r)) + (this->sourceCardType(r)==2 ? 1 : 2) ;
}

/// get the SC type
unsigned L1GctMap::sourceCardType(L1GctRegion r) {
  return this->sourceCardType(this->rctEta(r), this->rctPhi(r));
}

// get the SC input number
unsigned L1GctMap::sourceCardOutput(L1GctRegion r) {
  return this->sourceCardOutput(this->rctEta(r), this->rctPhi(r));
}

/// get the eta index within an RCT crate
/// counting from 0 at the Wheel boundary eta=0
unsigned L1GctMap::rctEta(L1GctRegion r) {
  unsigned eta = this->eta(r);
  return (eta<(N_RGN_ETA/2) ? ((N_RGN_ETA/2-1)-eta) : (eta - (N_RGN_ETA/2))) ;
}

/// get the phi index within an RCT crate
unsigned L1GctMap::rctPhi(L1GctRegion r) {
  // even or odd?
  return ((this->phi(r))%2);
}

/// get global eta index
unsigned L1GctMap::eta(L1GctRegion r) {
  // id = phi + 18eta
  return (r.id() / N_RGN_PHI);
}

/// get global phi index 
unsigned L1GctMap::phi(L1GctRegion r) {
  // id = phi + 18eta
  return (r.id() % N_RGN_PHI);
}

/// get physical eta 
//double L1GctMap::eta(L1GctRegion r) {
//
//}

/// get physical phi
//double L1GctMap::phi(L1GctRegion r) {
//
//}

/// get ID from eta, phi indices
unsigned L1GctMap::id(unsigned ieta, unsigned iphi) {
  return (ieta * N_RGN_PHI) + iphi;
}

/// get ID from position in the system
unsigned L1GctMap::id(unsigned rctCrate, unsigned scType, unsigned in) {
  if (rctCrate<N_RGN_PHI && ((scType==2 && in<12) || (scType==3 && in<10))) {
    unsigned ieta = this->rctEta(scType, in);
    unsigned iphi = this->rctPhi(scType, in);
    if (rctCrate>=(N_RGN_PHI/2)) { ieta = ((N_RGN_ETA-1) - ieta); }
    iphi = this->globalPhi(iphi, rctCrate%(N_RGN_PHI/2));
    return this->id(ieta, iphi);
  }
  // Shouldn't get here!
  throw cms::Exception("L1GctProcessingError")
    << " gctMap->id(rctCrate, scType, in) called with invalid inputs "
    << " crate " << rctCrate
    << " card type " << scType
    << " input " << in << std::endl;
}

/// convert phi from rctCrate/jetFinder local to global coordinates
unsigned L1GctMap::globalPhi(unsigned iphi, unsigned jfphi)
{
  if (iphi<2 && jfphi<(N_RGN_PHI)/2) {
    return ((N_RGN_PHI+4-(2*jfphi + iphi))%N_RGN_PHI);
  }
  // Here if arguments are invalid
  throw cms::Exception("L1GctProcessingError")
    << " gctMap->globalPhi(iphi, jfphi) called with invalid inputs "
    << " iphi " << iphi << ", should be 0 or 1"
    << " jfphi " << jfphi << ", should be less than " << N_RGN_PHI/2 << std::endl;
}

/// convert eta from rctCrate/jetFinder local to global coordinates
unsigned L1GctMap::globalEta(unsigned ieta, unsigned wheel)
{
  if (ieta<(N_RGN_ETA)/2 && wheel<2) {
    return ( (wheel==0) ? (N_RGN_ETA/2) - 1 - ieta : (N_RGN_ETA/2+ieta) );
  }
  // Here if arguments are invalid
  throw cms::Exception("L1GctProcessingError")
    << " gctMap->globalEta(ieta, wheel) called with invalid inputs "
    << " ieta " << ieta << ", should be less than " << N_RGN_ETA/2 
    << " wheel " << wheel << ", should be 0 or 1" << std::endl;
}

/// Conversion routines
/// local eta from source card type and output
unsigned L1GctMap::rctEta(unsigned scType, unsigned in)
{
  // initialise ieta to invalid bin number
  // to suppress compiler warnings
  unsigned ieta=N_RGN_ETA;

  if (scType==2) {
    if (in<12) {
      if (in<2)  { ieta=6-in; }
      if (in==2) { ieta=4;    }
      if (in==3) { ieta=4;    }
      if (in>3)  { ieta=3-(in%4); }
    }
  } else {
    if (scType==3) {
      if (in<10) {
	if (in<6) { ieta=10-in; }
 	     else { ieta=16-in; }
      }
    }
  }
  return ieta;
}

/// local phi from source card type and output
unsigned L1GctMap::rctPhi(unsigned scType, unsigned in)
{
  // initialise iphi to invalid bin number
  // to suppress compiler warnings
  unsigned iphi=N_RGN_PHI;

  if (scType==2) {
    if (in<12) {
      if (in<2)  { iphi=1; }
      if (in==2) { iphi=0; }
      if (in==3) { iphi=1; }
      if (in>3)  { iphi=(in/4)-1; }
    }
  } else {
    if (scType==3) {
      if (in<10) {
	if (in<6) { iphi=0; }
 	     else { iphi=1; }
      }
    }
  }
  return iphi;
}

/// source card type from local eta and phi
unsigned L1GctMap::sourceCardType(unsigned localEta, unsigned localPhi)
{
  bool barrelSourceCard =
    ((localPhi==0) && (localEta<6)) ||
    ((localPhi==1) && (localEta<4));
  return (barrelSourceCard ? 3 : 2);
}

/// source card number from local eta and phi
unsigned L1GctMap::sourceCardOutput(unsigned localEta, unsigned localPhi)
{
  unsigned result=99;
  // localPhi should be 0 or 1
  // localEta is 0-10, going from the centre of the barrel towards the endcaps
  if (localPhi==0) {
    if (localEta<6)  { result = localEta; }   //cardType3: inputs 0-5
    if (localEta==6) { result = 2; }          //cardType2: input  2
    if (localEta>6)  { result = localEta-3; } //cardType2: inputs 4-7
  } else {
    if (localEta<4)  { result = localEta+6; } //cardType3: inputs 6-9
    if (localEta==4) { result = 0; }          //cardType2: input  0
    if (localEta==5) { result = 1; }          //cardType2: input  1
    if (localEta==6) { result = 3; }          //cardType2: input  3
    if (localEta>6)  { result = localEta+1; } //cardType2: inputs 8-11
  }

  return result;
}

