// This is the file "FactorizedJetCorrector.cc".
// This is the implementation of the class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/Vector3D.h"
#include "Math/LorentzVector.h"
#include <vector>
#include <string>
#include <sstream>

//------------------------------------------------------------------------
//--- Default FactorizedJetCorrector constructor -------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector()
{
}
//------------------------------------------------------------------------
//--- FactorizedJetCorrector constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions):
mCalc(fLevels,fFiles,fOptions)
{
}
//------------------------------------------------------------------------
//--- FactorizedJetCorrector constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector(const std::vector<JetCorrectorParameters>& fParameters):
  mCalc(fParameters)
{
}

//------------------------------------------------------------------------
//--- Returns the correction ---------------------------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getCorrection()
{
  return mCalc.getCorrection(mValues);
}
//------------------------------------------------------------------------
//--- Returns the vector of subcorrections, up to a given level ----------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrector::getSubCorrections()
{
  return mCalc.getSubCorrections(mValues);
}
//------------------------------------------------------------------------
//--- Setters ------------------------------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrector::setNPV(int fNPV)
{

  mValues.setNPV (fNPV);
}
void FactorizedJetCorrector::setJetEta(float fEta)
{
  mValues.setJetEta( fEta );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetPt(float fPt)
{
  mValues.setJetPt(fPt);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetPhi(float fPhi)
{
  mValues.setJetPhi( fPhi );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetE(float fE)
{
  mValues.setJetE(fE);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetEMF(float fEMF)
{
  mValues.setJetEMF( fEMF );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetA(float fA)
{
  mValues.setJetA( fA );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setRho(float fRho)
{
  mValues.setRho(fRho);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJPTrawP4(const TLorentzVector& fJPTrawP4)
{
  mValues.setJPTrawP4(fJPTrawP4);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJPTrawOff(float fJPTrawOff)
{
  mValues.setJPTrawOff(fJPTrawOff);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPx(float fPx)
{
  mValues.setLepPx( fPx );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPy(float fPy)
{
  mValues.setLepPy( fPy );
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPz(float fPz)
{
  mValues.setLepPz(fPz);
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setAddLepToJet(bool fAddLepToJet)
{
  mValues.setAddLepToJet(fAddLepToJet);
}
