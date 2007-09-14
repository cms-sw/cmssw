
/*----------------------------------------------------------------------

Sh. Rahatlou, University of Rome & INFN
simple analyzer to dump information about ECAL cond objects

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CLHEP/Matrix/Matrix.h"




using namespace std;

class EcalObjectAnalyzer : public edm::EDAnalyzer
{
public:
  explicit  EcalObjectAnalyzer(edm::ParameterSet const& p) 
  { }
  explicit  EcalObjectAnalyzer(int i) 
  { }
  virtual ~ EcalObjectAnalyzer() { }
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
private:
};

void
EcalObjectAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
{
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout <<">>> EcalObjectAnalyzer: processing run "<<e.id().run() << " event: " << e.id().event() << std::endl;

  edm::ESHandle<EcalPedestals> pPeds;
  context.get<EcalPedestalsRcd>().get(pPeds);
  
  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  context.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  std::cout << "Global ADC->GeV scale: EB " << agc->getEBValue() << " GeV/ADC count" 
	    << " EE " << agc->getEEValue() << " GeV/ADC count" <<std::endl; 

  const EcalPedestals* myped=pPeds.product();
  for(std::map<const unsigned int,EcalPedestals::Item>::const_iterator it=myped->m_pedestals.begin(); it!=myped->m_pedestals.end(); it++)
    {
      
      std::cout << "EcalPedestal: " << it->first
		<< "  mean_x1:  " <<it->second.mean_x1 << " rms_x1: " << it->second.rms_x1
		<< "  mean_x6:  " <<it->second.mean_x6 << " rms_x6: " << it->second.rms_x6
		<< "  mean_x12: " <<it->second.mean_x12 << " rms_x12: " << it->second.rms_x12
		<< std::endl;
    } 

  // fetch map of groups of xtals
  edm::ESHandle<EcalWeightXtalGroups> pGrp;
  context.get<EcalWeightXtalGroupsRcd>().get(pGrp);
  const EcalWeightXtalGroups* grp = pGrp.product();
  
  for (EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().begin(); git!= grp->getMap().end() ;git++)
    {
      EcalXtalGroupId gid;
      if( git != grp->getMap().end() ) {
	std::cout << "XtalGroupId " << git->first << " gid: "  
		  << git->second.id() << std:: endl;
      }
    }
  
  // Gain Ratios
  edm::ESHandle<EcalGainRatios> pRatio;
  context.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatios* gr = pRatio.product();

  for (EcalGainRatios::EcalGainRatioMap::const_iterator grit=gr->getMap().begin(); grit!= gr->getMap().end() ; grit++)
    {
      EcalMGPAGainRatio mgpa;
      mgpa = grit->second;
      std::cout << "EcalMGPAGainRatio: " << grit->first  
		<< " gain 12/6:  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1()
		<< std::endl;
    } 

  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  context.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();

  for(EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().begin();icalit!=ical->getMap().end();icalit++)
    {
      EcalIntercalibConstants::EcalIntercalibConstant icalconst;
      icalconst = icalit->second;
      std::cout << "EcalIntercalibConstant: " << icalit->first
		<< " icalconst: " << icalconst
		<< std::endl;
    } 

//   // fetch TB weights
//   std::cout <<"Fetching EcalTBWeights from DB " << std::endl;
   edm::ESHandle<EcalTBWeights> pWgts;
   context.get<EcalTBWeightsRcd>().get(pWgts);
   const EcalTBWeights* wgts = pWgts.product();
   std::cout << "EcalTBWeightMap.size(): " << wgts->getMap().size() << std::endl;

//   // look up the correct weights for this  xtal
//   //EcalXtalGroupId gid( git->second );
//   EcalTBWeights::EcalTDCId tdcid(1);

//   std::cout << "Lookup EcalWeightSet for groupid: " << gid.id() << " and TDC id " << tdcid << std::endl;
   for (EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().begin(); wit != wgts->getMap().end() ; wit++)
     {
       std::cout << "EcalWeights " << wit->first.first.id() << "," << wit->first.second << std::endl;
       wit->second.print(std::cout);
       std::cout << std::endl;
     }
   
} //end of ::Analyze()
DEFINE_FWK_MODULE(EcalObjectAnalyzer);
