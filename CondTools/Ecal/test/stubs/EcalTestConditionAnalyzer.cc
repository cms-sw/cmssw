/**
   \file
   Test analyzer for ecal conditions

   \author Stefano ARGIRO
   \version $Id: EcalTestConditionAnalyzer.cc,v 1.6 2009/07/01 08:16:25 argiro Exp $
   \date 05 Nov 2008
*/

#include <string>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"

#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalWeightGroupXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"

/**
 *
 * Test analyzer that reads ecal records from event setup and writes XML files
 *
 */

class EcalTestConditionAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTestConditionAnalyzer(const edm::ParameterSet&);
  ~EcalTestConditionAnalyzer() override = default;

  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVConstantToken_;
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelStatusToken_;
  const edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstantsToken_;
  const edm::ESGetToken<EcalIntercalibConstantsMC, EcalIntercalibConstantsMCRcd> intercalibConstantsMCToken_;
  const edm::ESGetToken<EcalIntercalibErrors, EcalIntercalibErrorsRcd> intercalibErrorsToken_;
  const edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tbWeightsToken_;
  const edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> weightXtalGroupsToken_;
};

EcalTestConditionAnalyzer::EcalTestConditionAnalyzer(const edm::ParameterSet&)
    : adcToGeVConstantToken_(esConsumes()),
      channelStatusToken_(esConsumes()),
      gainRatiosToken_(esConsumes()),
      intercalibConstantsToken_(esConsumes()),
      intercalibConstantsMCToken_(esConsumes()),
      intercalibErrorsToken_(esConsumes()),
      tbWeightsToken_(esConsumes()),
      weightXtalGroupsToken_(esConsumes()) {}

void EcalTestConditionAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& iSetup) {
  // retrieve records from setup and write XML
  EcalCondHeader header;
  header.method_ = "testmethod";
  header.version_ = "testversion";
  header.datasource_ = "testdata";
  header.since_ = 123;
  header.tag_ = "testtag";
  header.date_ = "Mar 24 1973";

  const auto& adctogev = iSetup.getData(adcToGeVConstantToken_);
  const auto& chstatus = iSetup.getData(channelStatusToken_);
  const auto& gainratios = iSetup.getData(gainRatiosToken_);
  const auto& intercalib = iSetup.getData(intercalibConstantsToken_);
  const auto& intercalibmc = iSetup.getData(intercalibConstantsMCToken_);
  const auto& intercaliberr = iSetup.getData(intercalibErrorsToken_);
  const auto& tbweights = iSetup.getData(tbWeightsToken_);
  const auto& wgroup = iSetup.getData(weightXtalGroupsToken_);

  edm::LogInfo("EcalTestConditionAnalyzer") << "Got all records";

  const std::string ADCfile = "EcalADCToGeVConstant.xml";
  const std::string ChStatusfile = "EcalChannelStatus.xml";
  const std::string Grfile = "EcalGainRatios.xml";
  const std::string InterFile = "EcalIntercalibConstants.xml";
  const std::string InterMCFile = "EcalIntercalibConstantsMC.xml";
  const std::string WFile = "EcalTBWeights.xml";
  const std::string WGFile = "EcalWeightXtalGroups.xml";

  EcalADCToGeVXMLTranslator::writeXML(ADCfile, header, adctogev);
  EcalChannelStatusXMLTranslator::writeXML(ChStatusfile, header, chstatus);
  EcalGainRatiosXMLTranslator::writeXML(Grfile, header, gainratios);
  EcalIntercalibConstantsXMLTranslator::writeXML(InterFile, header, intercalib);
  EcalTBWeightsXMLTranslator::writeXML(WFile, header, tbweights);
  EcalWeightGroupXMLTranslator::writeXML(WGFile, header, wgroup);
}

DEFINE_FWK_MODULE(EcalTestConditionAnalyzer);
