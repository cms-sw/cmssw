
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

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

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"

using namespace std;

namespace edmtest
{
  class HcalCalibrationsAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  HcalCalibrationsAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  HcalCalibrationsAnalyzer(int i) 
    { }
    virtual ~ HcalCalibrationsAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
   HcalCalibrationsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<"HcalCalibrationsAnalyzer::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<"HcalCalibrationsAnalyzer::analyze->  ---EVENT NUMBER "<<e.id().run() <<std::endl;
    edm::ESHandle<HcalPedestals> pPeds;
    context.get<HcalPedestalsRcd>().get(pPeds);
//    edm::eventsetup::ESHandle<HcalPedestalWidths> pPedWs;
//    context.get<HcalPedestalWidthsRcd>().get(pPedWs);
//    edm::eventsetup::ESHandle<HcalGains> pGains;
//    context.get<HcalGainsRcd>().get(pGains);
//    edm::eventsetup::ESHandle<HcalGainWidths> pGainWs;
//    context.get<HcalGainWidthsRcd>().get(pGainWs);
    //call tracker code
    //

    const HcalPedestals* myped = pPeds.product();
    std::vector<DetId> mychannels = myped->getAllChannels();
    //    int cnt=0;
    for (std::vector<DetId>::const_iterator it=mychannels.begin(); it!=mychannels.end(); it++)
      {
	HcalDetId mydetid(*it);
	int eta = mydetid.ieta();
	int phi = mydetid.iphi();
	int depth = mydetid.depth();
	int subdet = mydetid.subdet();
	char id;
	int rawid = mydetid.rawId();
	switch (subdet)
	  {
	    case 1 : id = 'B'; break;
	    case 2 : id = 'E'; break;
	    case 3 : id = 'O'; break;
	    case 4 : id = 'F'; break;
	  }

	//	std::cout << eta << " " << phi << " " << depth << " " << subdet << " H" << id << " " << rawid;
	const float* values = (myped->getValues(mydetid))->getValues();
//	if (values) std::cout << ", pedestals: "
//			      << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 

	printf("   %d   %d   %d   H%c   %f   %f   %f   %f   %X\n", eta, phi, depth, id, values [0], values [1], values [2], values [3], rawid);


      }



//    std::cout <<" Hcal peds for channel HB eta=15, phi=5, depth=2 "<<std::endl;
//    int channelID = HcalDetId (HcalBarrel, 15, 5, 2).rawId();
//    const HcalPedestals* myped=pPeds.product();
//    const HcalPedestalWidths* mypedW=pPedWs.product();
//    const HcalGains* mygain=pGains.product();
//    const HcalGainWidths* mygainW=pGainWs.product();
//
//    const float* values = myped->getValues (channelID);
//    if (values) std::cout << "pedestals for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mypedW->getValues (channelID);
//    if (values) std::cout << "pedestal widths for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mygain->getValues (channelID);
//    if (values) std::cout << "gains for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mygainW->getValues (channelID);
//    if (values) std::cout << "gain widts for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
  }
  DEFINE_FWK_MODULE(HcalCalibrationsAnalyzer);
}
