#include "CondTools/Hcal/interface/HcalPedestalWidthsCheck.h"

HcalPedestalWidthsCheck::HcalPedestalWidthsCheck(edm::ParameterSet const& ps)
{
  outfile = ps.getUntrackedParameter<std::string>("outFile","null");
  dumprefs = ps.getUntrackedParameter<std::string>("dumpRefWidthsTo","null");
  dumpupdate = ps.getUntrackedParameter<std::string>("dumpUpdateWidthsTo","null");
  checkemapflag = ps.getUntrackedParameter<bool>("checkEmap",false);
  validateflag = ps.getUntrackedParameter<bool>("validateWidths",false);
  epsilon = ps.getUntrackedParameter<double>("deltaW",0);
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
   if(dumpupdate.compare("null")!=0){
    std::ofstream outStream(dumpupdate.c_str());
    std::cout << "--- Dumping PedestalWidths - update ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myNewPeds) );
   }
   if(dumprefs.compare("null")!=0){
    std::ofstream outStream2(dumprefs.c_str());
    std::cout << "--- Dumping PedestalWidths - reference ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream2, (*myRefPeds) );
   }
    // first get the list of all channels from the update
    std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
    
    // go through list of valid channels from reference, look up if pedestals exist for update
    // push back into new vector the corresponding updated pedestals,
    // or if it doesn't exist, the reference
    HcalPedestalWidths *resultPeds = new HcalPedestalWidths(myRefPeds->topo(), myRefPeds->isADC() );
    std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
    std::vector<DetId>::iterator cell;

    if(validateflag){
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
        DetId mydetid = *it;
        cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
        if (cell == listNewChan.end()) // not present in new list, take old pedestals
          {
		throw cms::Exception("DataDoesNotMatch")<<"Value not found in reference" << std::endl;
          }
        else // present in new list, take new pedestals
          {
            const HcalPedestalWidth* first = myNewPeds->getValues( mydetid );
            const HcalPedestalWidth* second = myRefPeds->getValues( mydetid );
            const float* newwidth = first->getValues();
            const float* oldwidth = second->getValues();
            if( (*newwidth != *oldwidth) || (*(newwidth+1)!=*(oldwidth+1)) || (*(newwidth+2)!=*(oldwidth+2)) || (*(newwidth+3)!=*(oldwidth+3)) || (*(newwidth+4)!=*(oldwidth+4)) || (*(newwidth+5)!=*(oldwidth+5)) || (*(newwidth+6)!=*(oldwidth+6)) || (*(newwidth+7)!=*(oldwidth+7)) || (*(newwidth+8)!=*(oldwidth+8)) || (*(newwidth+9)!=*(oldwidth+9))){
                 throw cms::Exception("DataDoesNotMatch") << "Values are not identical" << std::endl;
            }
            listNewChan.erase(cell);  // fix 25.02.08
          }
      }
      std::cout << "These are identical" << std::endl;
    }




  if(epsilon!=0){
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
        DetId mydetid = *it;
        cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
        if (cell == listNewChan.end()) // not present in new list, take old pedestals
          {
                throw cms::Exception("DataDoesNotMatch")<<"Value not found in reference" << std::endl;
          }
        else // present in new list, take new pedestals
          {
            const HcalPedestalWidth* first = myNewPeds->getValues( mydetid );
            const HcalPedestalWidth* second = myRefPeds->getValues( mydetid );
            const float* newwidth = first->getValues();
            const float* oldwidth = second->getValues();
            if( fabs(*newwidth-*oldwidth)>epsilon || fabs(*(newwidth+1)-*(oldwidth+1))>epsilon || fabs(*(newwidth+2)-*(oldwidth+2))>epsilon || fabs(*(newwidth+3)-*(oldwidth+3))>epsilon || fabs(*(newwidth+4)-*(oldwidth+4))>epsilon || fabs(*(newwidth+5)-*(oldwidth+5))>epsilon || fabs(*(newwidth+6)-*(oldwidth+6))>epsilon || fabs(*(newwidth+7)-*(oldwidth+7))>epsilon || fabs(*(newwidth+8)-*(oldwidth+8))>epsilon || fabs(*(newwidth+9)-*(oldwidth+9))>epsilon){
                 throw cms::Exception("DataDoesNotMatch") << "Values differ by more than deltaW" << std::endl;
            }
            listNewChan.erase(cell);  // fix 25.02.08
          }
      }
      std::cout << "These are identical" << std::endl;
    }
   if(outfile.compare("null")!=0){
   for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
	DetId mydetid = *it;
	cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
	if (cell == listNewChan.end()) // not present in new list, take old pedestals
	  {
	    const HcalPedestalWidth* mywidth = myRefPeds->getValues( mydetid );
	    std::cout << "o";
	    resultPeds->addValues( *mywidth );
	  }
	else // present in new list, take new pedestals
	  {
	    const HcalPedestalWidth* mywidth = myNewPeds->getValues( mydetid );
	    std::cout << "n";
	    resultPeds->addValues( *mywidth );

	    listNewChan.erase(cell);  // fix 25.02.08
	  }
      }

    for (std::vector<DetId>::iterator it = listNewChan.begin(); it != listNewChan.end(); it++)  // fix 25.02.08
      {
	DetId mydetid = *it;
	const HcalPedestalWidth* mywidth = myNewPeds->getValues( mydetid );
	std::cout << "N";
	resultPeds->addValues( *mywidth );
      }
    // dump the resulting list of pedestals into a file
    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Dumping PedestalWidths - the combined ones ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );



    }
    std::cout << std::endl;
    if(checkemapflag){
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
    }

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
