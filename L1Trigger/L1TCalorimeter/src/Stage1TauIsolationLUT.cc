// Stage1TauIsolationLUT.cc
// Author: Leonard Apanasevich
//

#include "L1Trigger/L1TCalorimeter/interface/Stage1TauIsolationLUT.h"
#include <vector>

using namespace l1t;

const unsigned int Stage1TauIsolationLUT::nbitsJet=NBITS_JET_ET_LUT;
const unsigned int Stage1TauIsolationLUT::nbitsTau=NBITS_TAU_ET_LUT;
const unsigned int Stage1TauIsolationLUT::nbits_data=NBITS_DATA;
const unsigned int Stage1TauIsolationLUT::lut_version=LUT_VERSION;

Stage1TauIsolationLUT::Stage1TauIsolationLUT(CaloParamsStage1* params): params_(params) 
{};

unsigned Stage1TauIsolationLUT::lutAddress(unsigned int tauPt, unsigned int jetPt) const
{
  const unsigned int maxJet = pow(2,nbitsJet)-1;
  const unsigned int maxTau = pow(2,nbitsTau)-1;
  
  // lut only defined for 8 bit Jet ET
  if ( (nbitsJet != 8) ||  (nbitsTau != 8) ) return 0;

  double jetLsb=params_->jetLsb();
  if (std::abs(jetLsb-0.5) > 0.0001){
    std::cout << "%Stage1TauIsolationLUT-E-Unexpected jetLsb " << jetLsb << " IsoTau calculation will be broken"<< std::endl;
    return 0;
  }

  tauPt=tauPt>>1;
  jetPt=jetPt>>1;

  if (jetPt>maxJet) jetPt=maxJet;
  if (tauPt>maxTau) tauPt=maxTau;

  unsigned int address= (jetPt << nbitsTau) + tauPt;
  return address;
}

int Stage1TauIsolationLUT::lutPayload(unsigned int address) const
{

  const unsigned int maxJet = pow(2,nbitsJet)-1;
  const unsigned int maxTau = pow(2,nbitsTau)-1;

  const double tauMaxJetIsolationA = params_->tauMaxJetIsolationA();
  const double tauMaxJetIsolationB = params_->tauMaxJetIsolationB();
  const double tauMinPtJetIsolationB = params_->tauMinPtJetIsolationB();
  
  unsigned int maxAddress = pow(2,nbitsJet+nbitsTau)-1;    
  if (address > maxAddress){  // check that address is right length
    std::cout << "%Stage1TauIsolationLUT-E-Address: " << address 
	      << " exceeds maximum value allowed. Setting value to maximum (" << maxAddress << ")" << std::endl;
    address = maxAddress;
  }
  // extract the jet and tau et from the address
  int ijetet = address >> nbitsTau;
  int itauet = address & 0xff;

  
  double jet_pt = static_cast <float> (ijetet);  // no need convert to physical eT, as right shift (>>1) operation
  double tau_pt = static_cast <float> (itauet);  // in lutAddress automatically converts to physical eT, assuming lsb=0.5

  //std::cout << "ijetet: " << ijetet << "\titauet: " << itauet << std::endl;
  //std::cout << "jetet: " << jet_pt << "\ttauet: " << tau_pt << std::endl;

  int isol=0;
  if (maxTau == (unsigned) itauet){
    isol=1;
  } else if (maxJet == (unsigned) ijetet){
    isol=1;
  } else {
    double relativeJetIsolationTau = (jet_pt / tau_pt) -1;
        
    double isolCut=tauMaxJetIsolationA;
    if (tau_pt >= tauMinPtJetIsolationB)isolCut=tauMaxJetIsolationB;
    if (relativeJetIsolationTau < isolCut) isol=1;
  }
  return isol;
}

Stage1TauIsolationLUT::~Stage1TauIsolationLUT(){};
