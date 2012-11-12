#include "CondTools/Hcal/interface/HcalPedestalsCheck.h"

HcalPedestalsCheck::HcalPedestalsCheck(edm::ParameterSet const& ps)
{
  outfile = ps.getUntrackedParameter<std::string>("outFile","null");
  dumprefs = ps.getUntrackedParameter<std::string>("dumpRefPedsTo","null");
  dumpupdate = ps.getUntrackedParameter<std::string>("dumpUpdatePedsTo","null");
  checkemapflag = ps.getUntrackedParameter<bool>("checkEmap",true);
  validatepedestalsflag = ps.getUntrackedParameter<bool>("validatePedestals",false);
  epsilon = ps.getUntrackedParameter<double>("deltaP",0);
}

HcalPedestalsCheck::~HcalPedestalsCheck()
{
}

void HcalPedestalsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
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

  // dump pedestals:
  if(!(dumprefs.compare("null")==0)){
    std::ofstream outStream(dumprefs.c_str());
    std::cout << "--- Dumping Pedestals - reference ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myRefPeds) );
  }
  if(!(dumpupdate.compare("null")==0)){
    std::ofstream outStream2(dumpupdate.c_str());
    std::cout << "--- Dumping Pedestals - updated ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream2, (*myNewPeds) );
  }

  if(validatepedestalsflag){
    std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
    std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
    std::vector<DetId>::iterator cell;
    bool failflag = false;
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
	DetId mydetid = *it;
	cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
	if (cell == listNewChan.end()) // not present in new list, take old pedestals
	  {
            throw cms::Exception("DataDoesNotMatch") << "Channel missing";
	    failflag = true;
  	    break;
	  }
	else // present in new list, take new pedestals
	  {
	    const float* values = (myNewPeds->getValues( mydetid ))->getValues();
	    const float* oldvalue = (myRefPeds->getValues( mydetid ))->getValues();
	    if( (*oldvalue != *values) || (*(oldvalue + 1)!=*(values+1)) || (*(oldvalue + 2)!=*(values+2)) || (*(oldvalue + 3)!=*(values+3)) ){
               throw cms::Exception("DataDoesNotMatch") << "Value does not match";
	       failflag = true;
               break;
            }
	    // compare the values of the pedestals for valid channels between update and reference 

	    listNewChan.erase(cell);  // fix 25.02.08
	  }      
	}
       if(!failflag) std::cout << "These are identical" << std::endl;
    }
 
    if(epsilon!=0){   
    std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
    std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
    std::vector<DetId>::iterator cell;
    bool failflag = false;
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
        DetId mydetid = *it;
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
	       throw cms::Exception("DataDoesNotMatch") << "Values differ by more than deltaP";
	       failflag = true;
               break;
            }
            listNewChan.erase(cell);  // fix 25.02.08
          }
      }
    if(!failflag) std::cout << "These are identical to within deltaP" << std::endl;
    }
    if(!(outfile.compare("null")==0))
    {
    // first get the list of all channels from the update
    std::vector<DetId> listNewChan = myNewPeds->getAllChannels();

    // go through list of valid channels from reference, look up if pedestals exist for update
    // push back into new vector the corresponding updated pedestals,
    // or if it doesn't exist, the reference
    HcalPedestals *resultPeds = new HcalPedestals(myRefPeds->topo(), myRefPeds->isADC() );
    std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
    std::vector<DetId>::iterator cell;
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
        DetId mydetid = *it;
        cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
        if (cell == listNewChan.end()) // not present in new list, take old pedestals
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
            listNewChan.erase(cell);  // fix 25.02.08
          }
      }


    for (std::vector<DetId>::iterator it = listNewChan.begin(); it != listNewChan.end(); it++)  // fix 25.02.08
      {
	DetId mydetid = *it;
	const HcalPedestal* item = myNewPeds->getValues(mydetid);
	std::cout << "N";
	resultPeds->addValues(*item);
      }


    std::cout << std::endl;

    std::vector<DetId> listResult = resultPeds->getAllChannels();
    // get the e-map list of channels
    std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
    // look up if emap channels are all present in pedestals, if not then cerr
    if(checkemapflag){
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
    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Dumping Pedestals - the combined ones ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );
    }

    // const float* values = myped->getValues (channelID);
    //    if (values) std::cout << "pedestals for channel " << channelID << ": "
    //			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 

}

//vecDetId HcalPedestalsCheck::getMissingDetIds(vector<HcalPedestals> & myPedestals)
//{
//  HcalGeometry myHcalGeometry;
//  // get the valid detid from the various subdetectors
//  vecDetId validHB = myHcalGeometry.getValidDetIds(Hcal,HcalBarrel);  // check these numbers
//  vecDetId validHE = myHcalGeometry.getValidDetIds(Hcal,HcalEndcap);
//  vecDetId validHF = myHcalGeometry.getValidDetIds(Hcal,HcalForward);
//  vecDetId validHO = myHcalGeometry.getValidDetIds(Hcal,HcalOuter);
//  vecDetId validZDC = myHcalGeometry.getValidDetIds(Calo,2);
//
//  // check if everything is there in pedestals
//
//
//}


DEFINE_FWK_MODULE(HcalPedestalsCheck);
