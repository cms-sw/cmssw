#include "CondTools/Hcal/interface/HcalPedestalWidthsCheck.h"

HcalPedestalWidthsCheck::HcalPedestalWidthsCheck(edm::ParameterSet const& ps)
{
  outfile = ps.getUntrackedParameter<std::string>("outFile","Dump");
}

HcalPedestalWidthsCheck::~HcalPedestalWidthsCheck()
{
}

void HcalPedestalWidthsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  using namespace edm::eventsetup;

  // get fake pedestals from file ("new pedestals")
  edm::ESHandle<HcalPedestalWidths> newPeds;
  es.get<HcalPedestalWidthsRcd>().get("update",newPeds);
  const HcalPedestalWidths* myNewPeds = newPeds.product();

  // get DB pedestals from Frontier/OrcoX ("reference")
  edm::ESHandle<HcalPedestalWidths> refPeds;
  es.get<HcalPedestalWidthsRcd>().get("reference",refPeds);
  const HcalPedestalWidths* myRefPeds = refPeds.product();

  // get e-map from reference
  edm::ESHandle<HcalElectronicsMap> refEMap;
  es.get<HcalElectronicsMapRcd>().get("reference",refEMap);
  const HcalElectronicsMap* myRefEMap = refEMap.product();


    // dump pedestals:
//    std::ostringstream filename;
//    filename << "test_update.txt";
//    std::ofstream outStream(filename.str().c_str());
//    std::cout << "--- Dumping PedestalWidths - update ---" << std::endl;
//    HcalDbASCIIIO::dumpObject (outStream, (*myNewPeds) );
//
//    std::ostringstream filename2;
//    filename2 << "test_reference.txt";
//    std::ofstream outStream2(filename2.str().c_str());
//    std::cout << "--- Dumping PedestalWidths - reference ---" << std::endl;
//    HcalDbASCIIIO::dumpObject (outStream2, (*myRefPeds) );

    // first get the list of all channels from the update
    std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
    
    // go through list of valid channels from reference, look up if pedestals exist for update
    // push back into new vector the corresponding updated pedestals,
    // or if it doesn't exist, the reference
    HcalPedestalWidths *resultPeds = new HcalPedestalWidths();
    std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
    std::vector<DetId>::const_iterator cell;
    for (std::vector<DetId>::const_iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
	DetId mydetid = *it;
	cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
	if (cell == listNewChan.end()) // not present in new list, take old pedestals
	  {
	    const HcalPedestalWidth* mywidth = myRefPeds->getValues( mydetid );
	    std::cout << "o";
	    resultPeds->setWidth( *mywidth );
//	    //   bool addValue (DetId fId, const float fValues [4]);
//	    const float* values = (myRefPeds->getValues( mydetid ))->getValues();
//	    std::cout << "o";
//	    resultPeds->addValue( (*it), values );
	  }
	else // present in new list, take new pedestals
	  {
	    const HcalPedestalWidth* mywidth = myNewPeds->getValues( mydetid );
	    std::cout << "n";
	    resultPeds->setWidth( *mywidth );
//	    const float* values = (myNewPeds->getValues( mydetid ))->getValues();
//	    std::cout << "n";
//	    resultPeds->addValue( (*it), values );
	    // compare the values of the pedestals for valid channels between update and reference
	    

	  }
      }
    std::cout << std::endl;

    std::vector<DetId> listResult = resultPeds->getAllChannels();
    // get the e-map list of channels
    std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
    // look up if emap channels are all present in pedestals, if not then cerr
    for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {
      DetId mydetid = DetId(it->rawId());
	if (std::find(listResult.begin(), listResult.end(), mydetid ) == listResult.end()  )
	  {
	    std::cout << "Conditions not found for DetId = " << HcalGenericDetId(it->rawId()) << std::endl;
	  }
      }


    // dump the resulting list of pedestals into a file
    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Dumping PedestalWidths - the combined ones ---" << std::endl;
    resultPeds->sort();
    HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );


    // const float* values = myped->getValues (channelID);
    //    if (values) std::cout << "pedestals for channel " << channelID << ": "
    //			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 

}


//vecDetId HcalPedestalWidthsCheck::getMissingDetIds(vector<HcalPedestalWidths> & myPedestalWidths)
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


DEFINE_FWK_MODULE(HcalPedestalWidthsCheck);
