/**
   \file
   Test analyzer for ecal conditions

   \author Stefano ARGIRO
   \version $Id: EcalTestConditionAnalyzer.cc,v 1.7 2009/12/17 23:26:04 wmtan Exp $
   \date 05 Nov 2008
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"


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


static const char CVSId[] = "$Id: EcalTestConditionAnalyzer.cc,v 1.7 2009/12/17 23:26:04 wmtan Exp $";

/**
 *
 * Test analyzer that reads ecal records from event setup and writes XML files
 *
 */

class EcalTestConditionAnalyzer : public edm::EDAnalyzer {

public:
  explicit EcalTestConditionAnalyzer (const edm::ParameterSet&){}
    

private:
  virtual void beginJob(){}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}

};


void EcalTestConditionAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& iSetup){

  using std::string;

  // retrieve records from setup and write XML
 
   EcalCondHeader header;
   header.method_="testmethod";
   header.version_="testversion";
   header.datasource_="testdata";
   header.since_=123;
   header.tag_="testtag";
   header.date_="Mar 24 1973";
  
   edm::ESHandle<EcalADCToGeVConstant> adctogev;
   iSetup.get<EcalADCToGeVConstantRcd>().get(adctogev );
 
   edm::ESHandle<EcalChannelStatus> chstatus;
   iSetup.get<EcalChannelStatusRcd>().get(chstatus);

   edm::ESHandle<EcalGainRatios> gainratios;
   iSetup.get<EcalGainRatiosRcd>().get(gainratios);

   edm::ESHandle<EcalIntercalibConstants> intercalib;
   iSetup.get<EcalIntercalibConstantsRcd>().get(intercalib);

   edm::ESHandle<EcalIntercalibConstantsMC> intercalibmc;
   iSetup.get<EcalIntercalibConstantsMCRcd>().get(intercalibmc);

   edm::ESHandle<EcalIntercalibErrors> intercaliberr;
   iSetup.get<EcalIntercalibErrorsRcd>().get(intercaliberr);


   edm::ESHandle<EcalTBWeights> tbweights;
   iSetup.get<EcalTBWeightsRcd>().get(tbweights);

   edm::ESHandle<EcalWeightXtalGroups> wgroup;
   iSetup.get<EcalWeightXtalGroupsRcd>().get(wgroup);

   std::cout << "Got all records " << std::endl;
   
   string ADCfile      = "EcalADCToGeVConstant.xml";
   string ChStatusfile = "EcalChannelStatus.xml";
   string Grfile       = "EcalGainRatios.xml";
   string InterFile    = "EcalIntercalibConstants.xml";
   string InterMCFile  = "EcalIntercalibConstantsMC.xml";
   string WFile        = "EcalTBWeights.xml";
   string WGFile       = "EcalWeightXtalGroups.xml";

   EcalADCToGeVXMLTranslator::writeXML(ADCfile,header,*adctogev); 
   EcalChannelStatusXMLTranslator::writeXML(ChStatusfile,header,*chstatus);
   EcalGainRatiosXMLTranslator::writeXML(Grfile,header,*gainratios);
   EcalIntercalibConstantsXMLTranslator::writeXML(InterFile,header,
						  *intercalib);
   EcalTBWeightsXMLTranslator::writeXML(WFile,header,*tbweights);
   EcalWeightGroupXMLTranslator::writeXML(WGFile,header,*wgroup);

}

DEFINE_FWK_MODULE(EcalTestConditionAnalyzer);



// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End:
