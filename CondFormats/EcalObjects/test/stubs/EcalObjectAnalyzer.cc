
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

  // use a channel to fetch values from DB
  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
  int ieta =  int( 1 + r1*85 );
  r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
  int iphi =  int( 1 + r1*20 );

  EBDetId ebid(ieta,iphi); //eta,phi
  std::cout << "EcalObjectAnalyzer: using EBDetId: " << ebid << std::endl;

  const EcalPedestals* myped=pPeds.product();
  std::map<const unsigned int,EcalPedestals::Item>::const_iterator it=myped->m_pedestals.find(ebid.rawId());
  if( it!=myped->m_pedestals.end() ){
    std::cout << "EcalPedestal: "
	      << "  mean_x1:  " <<it->second.mean_x1 << " rms_x1: " << it->second.rms_x1
	      << "  mean_x6:  " <<it->second.mean_x6 << " rms_x6: " << it->second.rms_x6
	      << "  mean_x12: " <<it->second.mean_x12 << " rms_x12: " << it->second.rms_x12
	      << std::endl;
  } else {
    std::cout << "No pedestal found for this xtal! something wrong with EcalPedestals in your DB? "
	      << std::endl;
  }

  // fetch map of groups of xtals
  edm::ESHandle<EcalWeightXtalGroups> pGrp;
  context.get<EcalWeightXtalGroupsRcd>().get(pGrp);
  const EcalWeightXtalGroups* grp = pGrp.product();

  EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().find( ebid.rawId() );
  EcalXtalGroupId gid;
  if( git != grp->getMap().end() ) {
    std::cout << "XtalGroupId.id() = " << git->second.id() << std:: endl;
    gid = git->second;
  } else {
    std::cout << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
	      << std::endl;
  }

  // Gain Ratios
  edm::ESHandle<EcalGainRatios> pRatio;
  context.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatios* gr = pRatio.product();

  EcalGainRatios::EcalGainRatioMap::const_iterator grit=gr->getMap().find(ebid.rawId());
  EcalMGPAGainRatio mgpa;
  if( grit!=gr->getMap().end() ){
    mgpa = grit->second;

    std::cout << "EcalMGPAGainRatio: "
	      << "gain 12/6 :  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1()
	      << std::endl;
  } else {
    std::cout << "No MGPA Gain Ratio found for this xtal! something wrong with EcalGainRatios in your DB? "
	      << std::endl;
  }

  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  context.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();

  EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(ebid.rawId());
  EcalIntercalibConstants::EcalIntercalibConstant icalconst;
  if( icalit!=ical->getMap().end() ){
    icalconst = icalit->second;

    std::cout << "EcalIntercalibConstant: "
	      << icalconst
	      << std::endl;
  } else {
    std::cout << "No intercalib const found for this xtal! something wrong with EcalIntercalibConstants in your DB? "
	      << std::endl;
  }

  // fetch TB weights
  std::cout <<"Fetching EcalTBWeights from DB " << std::endl;
  edm::ESHandle<EcalTBWeights> pWgts;
  context.get<EcalTBWeightsRcd>().get(pWgts);
  const EcalTBWeights* wgts = pWgts.product();
  std::cout << "EcalTBWeightMap.size(): " << wgts->getMap().size() << std::endl;

  // look up the correct weights for this  xtal
  //EcalXtalGroupId gid( git->second );
  EcalTBWeights::EcalTDCId tdcid(1);

  std::cout << "Lookup EcalWeightSet for groupid: " << gid.id() << " and TDC id " << tdcid << std::endl;
  EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
  EcalWeightSet  wset;
  if( wit != wgts->getMap().end() ) {
    wset = wit->second;
    //std::cout << "weight set it: " << wit << std::endl;

    std::cout << "check size of data members in EcalWeightSet" << std::endl;
    wit->second.print(std::cout);


    //typedef std::vector< std::vector<EcalWeight> > EcalWeightMatrix;
    EcalWeightSet::EcalWeightMatrix mat1 = wit->second.getWeightsBeforeGainSwitch();
    EcalWeightSet::EcalWeightMatrix mat2 = wit->second.getWeightsAfterGainSwitch();

    //std::cout << "WeightsBeforeGainSwitch.size: " << mat1.size() << ", WeightsAfterGainSwitch.size: " << mat2.size() << std::endl;
  } else {
    std::cout << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid << std::endl;
  }


   
} //end of ::Analyze()
DEFINE_FWK_MODULE(EcalObjectAnalyzer);
