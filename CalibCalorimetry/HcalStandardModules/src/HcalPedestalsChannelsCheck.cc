#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsChannelsCheck.h"

HcalPedestalsChannelsCheck::HcalPedestalsChannelsCheck(edm::ParameterSet const& ps)
{
   epsilon = .1;
}

HcalPedestalsChannelsCheck::~HcalPedestalsChannelsCheck()
{
}

void HcalPedestalsChannelsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
   using namespace edm::eventsetup;
 
   // get fake pedestals from file ("new pedestals")
   edm::ESHandle<HcalPedestals> newPeds;
   es.get<HcalPedestalsRcd>().get("update",newPeds);
   const HcalPedestals* myNewPeds = newPeds.product();
 
   // get DB pedestals from Frontier/OrcoX ("reference")
   edm::ESHandle<HcalPedestals> refPeds;
   es.get<HcalPedestalsRcd>().get("reference",refPeds);
   const HcalPedestals* myRefPeds = refPeds.product();
 
   // get e-map from reference
   edm::ESHandle<HcalElectronicsMap> refEMap;
   es.get<HcalElectronicsMapRcd>().get("reference",refEMap);
   const HcalElectronicsMap* myRefEMap = refEMap.product();
 
   std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
   std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
   std::vector<DetId>::iterator cell;
   bool failflag = false;

   // store channels which have changed by more that epsilon
   HcalPedestals *changedchannels = new HcalPedestals();
   for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
         DetId mydetid = *it;
         HcalDetId hocheck(mydetid);
         if(hocheck.subdet()==3) continue;
         cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
         if (cell == listNewChan.end())
            {
               continue;
            }
         else
            { 
               const float* values = (myNewPeds->getValues( mydetid ))->getValues();
               const float* oldvalue = (myRefPeds->getValues( mydetid ))->getValues();
               if( (fabs(*oldvalue-*values)>epsilon) || (fabs(*(oldvalue+1)-*(values+1))>epsilon) || (fabs(*(oldvalue+2)-*(values+2))>epsilon) || (fabs(*(oldvalue+3)-*(values+3))>epsilon) ){
// 	       throw cms::Exception("DataDoesNotMatch") << "Values differ by more than deltaP";
               failflag = true;
               const HcalPedestal* item = myNewPeds->getValues(mydetid);
               changedchannels->addValues(*item);
             }
             listNewChan.erase(cell);  // fix 25.02.08
           }
        
       } 
     // first get the list of all channels from the update
     std::vector<DetId> listChangedChan = changedchannels->getAllChannels();
 
     HcalPedestals *resultPeds = new HcalPedestals(); //myRefPeds->isADC() );
     for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
       {
         DetId mydetid = *it;
         cell = std::find(listChangedChan.begin(), listChangedChan.end(), mydetid);
         if (cell == listChangedChan.end()) // not present in new list, take old pedestals
           {
             //   bool addValue (DetId fId, const float fValues [4]);
 	    const HcalPedestal* item = myRefPeds->getValues(mydetid);
             std::cout << "o";
             resultPeds->addValues(*item);
           }
         else // present in new list, take new pedestals
           {
             const HcalPedestal* item = myNewPeds->getValues(mydetid);
             std::cout << "n";
             resultPeds->addValues(*item);
             // compare the values of the pedestals for valid channels between update and reference
             listChangedChan.erase(cell);  // fix 25.02.08
           }
       }
 
     std::vector<DetId> listResult = resultPeds->getAllChannels();
     // get the e-map list of channels
     std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
     // look up if emap channels are all present in pedestals, if not then cerr
     if(1)//checkemapflag)
     {
       for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
       {
   	  DetId mydetid = DetId(it->rawId());
 	HcalGenericDetId mygenid(it->rawId());
 	//	std::cout << "id = " << mygenid << ", hashed id = " << mygenid.hashedId() << std::endl;
 	if (std::find(listResult.begin(), listResult.end(), mydetid ) == listResult.end())
 	  {
 	    std::cout << "Conditions not found for DetId = " << HcalGenericDetId(it->rawId()) << std::endl;
 	  }
       }
     }
 
     // dump the resulting list of pedestals into a file
     if(failflag)
     {
        std::ofstream outStream3("dump.txt");//outfile.c_str());
        std::cout << "--- Dumping Pedestals - thei merged ones ---" << std::endl;
        HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );
     }
  
}


DEFINE_FWK_MODULE(HcalPedestalsChannelsCheck);
