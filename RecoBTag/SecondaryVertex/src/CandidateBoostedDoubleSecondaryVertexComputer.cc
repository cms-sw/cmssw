#include "RecoBTag/SecondaryVertex/interface/CandidateBoostedDoubleSecondaryVertexComputer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"


CandidateBoostedDoubleSecondaryVertexComputer::CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters) :
  useCondDB_(parameters.getParameter<bool>("useCondDB")),
  gbrForestLabel_(parameters.existsAs<std::string>("gbrForestLabel") ? parameters.getParameter<std::string>("gbrForestLabel") : ""),
  weightFile_(parameters.existsAs<edm::FileInPath>("weightFile") ? parameters.getParameter<edm::FileInPath>("weightFile") : edm::FileInPath()),
  useGBRForest_(parameters.existsAs<bool>("useGBRForest") ? parameters.getParameter<bool>("useGBRForest") : false),
  useAdaBoost_(parameters.existsAs<bool>("useAdaBoost") ? parameters.getParameter<bool>("useAdaBoost") : false)
{
  uses(0, "svTagInfos");

  mvaID.reset(new TMVAEvaluator());
}

void CandidateBoostedDoubleSecondaryVertexComputer::initialize(const JetTagComputerRecord & record)
{
  // variable names and order need to be the same as in the training
  std::vector<std::string> variables({"z_ratio",
                                      "trackSipdSig_3","trackSipdSig_2","trackSipdSig_1","trackSipdSig_0",
                                      "trackSipdSig_1_0","trackSipdSig_0_0","trackSipdSig_1_1","trackSipdSig_0_1",
                                      "trackSip2dSigAboveCharm_0","trackSip2dSigAboveBottom_0","trackSip2dSigAboveBottom_1",
                                      "tau0_trackEtaRel_0","tau0_trackEtaRel_1","tau0_trackEtaRel_2",
                                      "tau1_trackEtaRel_0","tau1_trackEtaRel_1","tau1_trackEtaRel_2",
                                      "tau_vertexMass_0","tau_vertexEnergyRatio_0","tau_vertexDeltaR_0","tau_flightDistance2dSig_0",
                                      "tau_vertexMass_1","tau_vertexEnergyRatio_1","tau_flightDistance2dSig_1",
                                      "jetNTracks","nSV"});
  // book TMVA readers
  std::vector<std::string> spectators({"massPruned", "flavour", "nbHadrons", "ptPruned", "etaPruned"});

  if (useCondDB_)
  {
     const GBRWrapperRcd & gbrWrapperRecord = record.getRecord<GBRWrapperRcd>();

     edm::ESHandle<GBRForest> gbrForestHandle;
     gbrWrapperRecord.get(gbrForestLabel_.c_str(), gbrForestHandle);

     mvaID->initializeGBRForest(gbrForestHandle.product(), variables, spectators, useAdaBoost_);
  }
  else
    mvaID->initialize("Color:Silent:Error", "BDT", weightFile_.fullPath(), variables, spectators, useGBRForest_, useAdaBoost_);
}

float CandidateBoostedDoubleSecondaryVertexComputer::discriminator(const TagInfoHelper & tagInfo) const
{
  // get the TagInfo
  const reco::BoostedDoubleSVTagInfo & bdsvTagInfo = tagInfo.get<reco::BoostedDoubleSVTagInfo>(0);

  // get the TaggingVariables
  const reco::TaggingVariableList vars = bdsvTagInfo.taggingVariables();

  // default discriminator value
  float value = -10.;

  std::map<std::string,float> inputs;
  inputs["z_ratio"] = vars.get(reco::btau::z_ratio);
  inputs["trackSipdSig_3"] = vars.get(reco::btau::trackSip3dSig_3);
  inputs["trackSipdSig_2"] = vars.get(reco::btau::trackSip3dSig_2);
  inputs["trackSipdSig_1"] = vars.get(reco::btau::trackSip3dSig_1);
  inputs["trackSipdSig_0"] = vars.get(reco::btau::trackSip3dSig_0);
  inputs["trackSipdSig_1_0"] = vars.get(reco::btau::tau2_trackSip3dSig_0);
  inputs["trackSipdSig_0_0"] = vars.get(reco::btau::tau1_trackSip3dSig_0);
  inputs["trackSipdSig_1_1"] = vars.get(reco::btau::tau2_trackSip3dSig_1);
  inputs["trackSipdSig_0_1"] = vars.get(reco::btau::tau1_trackSip3dSig_1);
  inputs["trackSip2dSigAboveCharm_0"] = vars.get(reco::btau::trackSip2dSigAboveCharm);
  inputs["trackSip2dSigAboveBottom_0"] = vars.get(reco::btau::trackSip2dSigAboveBottom_0);
  inputs["trackSip2dSigAboveBottom_1"] = vars.get(reco::btau::trackSip2dSigAboveBottom_1);
  inputs["tau1_trackEtaRel_0"] = vars.get(reco::btau::tau2_trackEtaRel_0);
  inputs["tau1_trackEtaRel_1"] = vars.get(reco::btau::tau2_trackEtaRel_1);
  inputs["tau1_trackEtaRel_2"] = vars.get(reco::btau::tau2_trackEtaRel_2);
  inputs["tau0_trackEtaRel_0"] = vars.get(reco::btau::tau1_trackEtaRel_0);
  inputs["tau0_trackEtaRel_1"] = vars.get(reco::btau::tau1_trackEtaRel_1);
  inputs["tau0_trackEtaRel_2"] = vars.get(reco::btau::tau1_trackEtaRel_2);
  inputs["tau_vertexMass_0"] = vars.get(reco::btau::tau1_vertexMass);
  inputs["tau_vertexEnergyRatio_0"] = vars.get(reco::btau::tau1_vertexEnergyRatio);
  inputs["tau_vertexDeltaR_0"] = vars.get(reco::btau::tau1_vertexDeltaR);
  inputs["tau_flightDistance2dSig_0"] = vars.get(reco::btau::tau1_flightDistance2dSig);
  inputs["tau_vertexMass_1"] = vars.get(reco::btau::tau2_vertexMass);
  inputs["tau_vertexEnergyRatio_1"] = vars.get(reco::btau::tau2_vertexEnergyRatio);
  inputs["tau_flightDistance2dSig_1"] = vars.get(reco::btau::tau2_flightDistance2dSig);
  inputs["jetNTracks"] = vars.get(reco::btau::jetNTracks);
  inputs["nSV"] = vars.get(reco::btau::jetNSecondaryVertices);

  // evaluate the MVA
  value = mvaID->evaluate(inputs);

  // return the final discriminator value
  return value;
}
