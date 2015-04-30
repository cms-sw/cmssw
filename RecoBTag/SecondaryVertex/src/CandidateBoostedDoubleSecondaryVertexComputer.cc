#include "RecoBTag/SecondaryVertex/interface/CandidateBoostedDoubleSecondaryVertexComputer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"


CandidateBoostedDoubleSecondaryVertexComputer::CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters)
{
  uses(0, "ipTagInfos");
  uses(1, "svTagInfos");
  uses(2, "muonTagInfos");
  uses(3, "elecTagInfos");
}

float CandidateBoostedDoubleSecondaryVertexComputer::discriminator(const TagInfoHelper & tagInfo) const
{
  //const reco::CandIPTagInfo              & ipTagInfo = tagInfo.get<reco::CandIPTagInfo>(0); // for possible future use
  const reco::CandSecondaryVertexTagInfo & svTagInfo = tagInfo.get<reco::CandSecondaryVertexTagInfo>(1);
  const reco::CandSoftLeptonTagInfo      & muonTagInfo = tagInfo.get<reco::CandSoftLeptonTagInfo>(2);
  const reco::CandSoftLeptonTagInfo      & elecTagInfo = tagInfo.get<reco::CandSoftLeptonTagInfo>(3);

  // default discriminator value
  float value = -10.;

  // do stuff here
  int nSV = 0;
  int nSL = 0, nSM = 0, nSE = 0;

  nSV = svTagInfo.nVertices();
  nSM = muonTagInfo.leptons();
  nSE = elecTagInfo.leptons();
  nSL = nSM + nSE;
  if (nSL > nSV )
    edm::LogInfo("MoreLeptonsThanSVs") << "nSV: " << nSV << " nSL: " << nSL;
  else
    edm::LogInfo("MoreSVsThanLeptons") << "nSV: " << nSV << " nSL: " << nSL;
  // ...


  // return the final discriminator value
  return value;
}
