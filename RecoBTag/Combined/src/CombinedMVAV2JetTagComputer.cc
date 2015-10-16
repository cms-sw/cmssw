#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTag/Combined/interface/CombinedMVAV2JetTagComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

using namespace reco;

CombinedMVAV2JetTagComputer::CombinedMVAV2JetTagComputer(const edm::ParameterSet &params) :

  inputComputerNames( params.getParameter<std::vector<std::string> >("jetTagComputers") ),
  mvaName( params.getParameter<std::string >("mvaName") ),
  variables( params.getParameter<std::vector<std::string> >("variables") ),
  spectators( params.getParameter<std::vector<std::string> >("spectators") ),
  useCondDB(params.getParameter<bool>("useCondDB")),
  gbrForestLabel(params.existsAs<std::string>("gbrForestLabel") ? params.getParameter<std::string>("gbrForestLabel") : ""),
  weightFile(params.existsAs<edm::FileInPath>("weightFile") ? params.getParameter<edm::FileInPath>("weightFile") : edm::FileInPath()),
  useGBRForest(params.existsAs<bool>("useGBRForest") ? params.getParameter<bool>("useGBRForest") : false),
  useAdaBoost(params.existsAs<bool>("useAdaBoost") ? params.getParameter<bool>("useAdaBoost") : false)

{
  uses(0, "ipTagInfos");
  uses(1, "svAVRTagInfos");
  uses(2, "svIVFTagInfos");
  uses(3, "smTagInfos");
  uses(4, "seTagInfos");

}

CombinedMVAV2JetTagComputer::~CombinedMVAV2JetTagComputer()
{
}

void CombinedMVAV2JetTagComputer::initialize(const JetTagComputerRecord & record) {

  mvaID.reset(new TMVAEvaluator());

  if (useCondDB)
  {
     const GBRWrapperRcd & gbrWrapperRecord = record.getRecord<GBRWrapperRcd>();

     edm::ESHandle<GBRForest> gbrForestHandle;
     gbrWrapperRecord.get(gbrForestLabel.c_str(), gbrForestHandle);

     mvaID->initializeGBRForest(gbrForestHandle.product(), variables, spectators, useAdaBoost);
  }
  else {
    mvaID->initialize(
        "Color:Silent:Error", mvaName.c_str(),
        weightFile.fullPath(), variables, spectators, useGBRForest, useAdaBoost
    );
  }
  for (auto & name : inputComputerNames) {
    edm::ESHandle<JetTagComputer> computerHandle;
    record.get(name, computerHandle);
    const JetTagComputer* comp = computerHandle.product();
    computers.push_back(comp);
  }
}

float CombinedMVAV2JetTagComputer::discriminator(const JetTagComputer::TagInfoHelper &info) const
{
  // default discriminator value
  float value = -10.;

  // TagInfos for JP taggers
  std::vector<const BaseTagInfo*> jpTagInfos({ &info.getBase(0) });

  // TagInfos for the CSVv2AVR tagger
  std::vector<const BaseTagInfo*> avrTagInfos({ &info.getBase(0), &info.getBase(1) });

  // TagInfos for the CSVv2IVF tagger
  std::vector<const BaseTagInfo*> ivfTagInfos({ &info.getBase(0), &info.getBase(2) });

  // TagInfos for the SoftMuon tagger
  std::vector<const BaseTagInfo*> smTagInfos({ &info.getBase(3) });

  // TagInfos for the SoftElectron tagger
  std::vector<const BaseTagInfo*> seTagInfos({ &info.getBase(4) });

  std::map<std::string,float> inputs;
  inputs["Jet_JP"]       = (*(computers[0]))( TagInfoHelper(jpTagInfos) );
  inputs["Jet_JBP"]      = (*(computers[1]))( TagInfoHelper(jpTagInfos) );
  inputs["Jet_CSV"]      = (*(computers[2]))( TagInfoHelper(avrTagInfos) );
  inputs["Jet_CSVIVF"]   = (*(computers[2]))( TagInfoHelper(ivfTagInfos) );
  inputs["Jet_SoftMu"]   = (*(computers[3]))( TagInfoHelper(smTagInfos) );
  inputs["Jet_SoftEl"]   = (*(computers[4]))( TagInfoHelper(seTagInfos) );
//  inputs["Jet_pt"]       = 0.0;
//  inputs["Jet_eta"]      = 0.0;
//  inputs["Jet_flavour"]  = 0.0;
  
  if (inputs["Jet_JP"] <= 0) {
      inputs["Jet_JP"] = 0;
  }
  if (inputs["Jet_JBP"] <= 0) {
      inputs["Jet_JBP"] = 0;
  }
  if (inputs["Jet_CSV"] <= 0) {
      inputs["Jet_CSV"] = 0;
  }
  if (inputs["Jet_CSVIVF"] <= 0) {
      inputs["Jet_CSVIVF"] = 0;
  }
  if (inputs["Jet_SoftMu"] <= 0) {
      inputs["Jet_SoftMu"] = 0;
  }
  if (inputs["Jet_SoftEl"] <= 0) {
      inputs["Jet_SoftEl"] = 0;
  }
  
  if (inputs["Jet_CSV"] >= 1) {
      inputs["Jet_CSV"] = 1;
  }
  if (inputs["Jet_CSVIVF"] >= 1) {
      inputs["Jet_CSVIVF"] = 1;
  }
  if (inputs["Jet_SoftMu"] >= 1) {
      inputs["Jet_SoftMu"] = 1;
  }
  if (inputs["Jet_SoftEl"] >= 1) {
      inputs["Jet_SoftEl"] = 1;
  }

  // evaluate the MVA
  value = mvaID->evaluate(inputs);

  // return the final discriminator value
  return value;
}
