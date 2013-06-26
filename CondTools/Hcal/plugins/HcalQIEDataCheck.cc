#include "CondTools/Hcal/interface/HcalQIEDataCheck.h"

HcalQIEDataCheck::HcalQIEDataCheck(edm::ParameterSet const& ps)
{
  outfile = ps.getUntrackedParameter<std::string>("outFile","null");
  dumprefs = ps.getUntrackedParameter<std::string>("dumpRefQIEsTo","null");
  dumpupdate = ps.getUntrackedParameter<std::string>("dumpUpdateQIEsTo","null");
  checkemapflag = ps.getUntrackedParameter<bool>("checkEmap",false);
  validateflag = ps.getUntrackedParameter<bool>("validateQIEs",false);
//  epsilon = ps.getUntrackedParameter<double>("deltaQIE",0);
}

HcalQIEDataCheck::~HcalQIEDataCheck() {}

void HcalQIEDataCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  using namespace edm::eventsetup;

  edm::ESHandle<HcalQIEData> newQIEs;
  es.get<HcalQIEDataRcd>().get("update",newQIEs);
  const HcalQIEData* myNewQIEs = newQIEs.product();

  edm::ESHandle<HcalQIEData> refQIEs;
  es.get<HcalQIEDataRcd>().get("reference",refQIEs);
  const HcalQIEData* myRefQIEs = refQIEs.product();

  edm::ESHandle<HcalElectronicsMap> refEMap;
  es.get<HcalElectronicsMapRcd>().get("reference",refEMap);
  const HcalElectronicsMap* myRefEMap = refEMap.product();

  if(dumpupdate.compare("null")!=0){
    std::ofstream outStream(dumpupdate.c_str());
    std::cout << "--- Dumping QIEs - update ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myNewQIEs) );
  }
  if(dumprefs.compare("null")!=0){
    std::ofstream outStream2(dumprefs.c_str());
    std::cout << "--- Dumping QIEs - reference ---" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream2, (*myRefQIEs) );
  }

    // first get the list of all channels from the update
    std::vector<DetId> listNewChan = myNewQIEs->getAllChannels();   

    HcalQIEData *resultQIEs = new HcalQIEData(myRefQIEs->topo());
    std::vector<DetId> listRefChan = myRefQIEs->getAllChannels();
    std::vector<DetId>::iterator cell;

    if(validateflag){
    for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
        DetId mydetid = *it;
        cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
        if (cell == listNewChan.end()) // not present in new list
          {
		throw cms::Exception("DataDoesNotMatch") << "Value not found in reference" << std::endl;
          }
        else // present in new list
          {

            const HcalQIECoder* first = myNewQIEs->getCoder( mydetid );
            const HcalQIECoder* second = myRefQIEs->getCoder( mydetid );
	    {
	    bool failflag = false;
            if(first->offset(0,0) != second->offset(0,0)) failflag = true;
            if(first->offset(0,1) != second->offset(0,1)) failflag = true;
            if(first->offset(0,2) != second->offset(0,2)) failflag = true;
            if(first->offset(0,3) != second->offset(0,3)) failflag = true;
            if(first->offset(1,0) != second->offset(1,0)) failflag = true;
            if(first->offset(1,1) != second->offset(1,1)) failflag = true;
            if(first->offset(1,2) != second->offset(1,2)) failflag = true;
            if(first->offset(1,3) != second->offset(1,3)) failflag = true;
            if(first->offset(2,0) != second->offset(2,0)) failflag = true;
            if(first->offset(2,1) != second->offset(2,1)) failflag = true;
            if(first->offset(2,2) != second->offset(2,2)) failflag = true;
            if(first->offset(2,3) != second->offset(2,3)) failflag = true;
            if(first->offset(3,0) != second->offset(3,0)) failflag = true;
            if(first->offset(3,1) != second->offset(3,1)) failflag = true;
            if(first->offset(3,2) != second->offset(3,2)) failflag = true;
            if(first->offset(3,3) != second->offset(3,3)) failflag = true;
            if(first->slope(0,0) != second->slope(0,0)) failflag = true;
            if(first->slope(0,1) != second->slope(0,1)) failflag = true;
            if(first->slope(0,2) != second->slope(0,2)) failflag = true;
            if(first->slope(0,3) != second->slope(0,3)) failflag = true;
            if(first->slope(1,0) != second->slope(1,0)) failflag = true;
            if(first->slope(1,1) != second->slope(1,1)) failflag = true;
            if(first->slope(1,2) != second->slope(1,2)) failflag = true;
            if(first->slope(1,3) != second->slope(1,3)) failflag = true;
            if(first->slope(2,0) != second->slope(2,0)) failflag = true;
            if(first->slope(2,1) != second->slope(2,1)) failflag = true;
            if(first->slope(2,2) != second->slope(2,2)) failflag = true;
            if(first->slope(2,3) != second->slope(2,3)) failflag = true;
            if(first->slope(3,0) != second->slope(3,0)) failflag = true;
            if(first->slope(3,1) != second->slope(3,1)) failflag = true;
            if(first->slope(3,2) != second->slope(3,2)) failflag = true;
            if(first->slope(3,3) != second->slope(3,3)) failflag = true;
	    if(failflag) throw cms::Exception("DataDoesNotMatch") << "Values are do not match";
            }
            listNewChan.erase(cell);  // fix 25.02.08
          }
      }
      std::cout << "These are identical" << std::endl;
    }



//  if(epsilon!=0){
	//implement compare qies -- different epsilon for slope and offset?
  //  }

   if(outfile.compare("null")!=0){
   for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
	DetId mydetid = *it;
	cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
	if (cell == listNewChan.end()) // not present in new list
	  {
	    const HcalQIECoder* myCoder = myRefQIEs->getCoder( mydetid );
	    std::cout << "o";
	    resultQIEs->addCoder( *myCoder );
	  }
	else // present in new list
	  {
	    const HcalQIECoder* myCoder = myNewQIEs->getCoder( mydetid );
	    std::cout << "n";
	    resultQIEs->addCoder( *myCoder );
	    listNewChan.erase(cell);  // fix 25.02.08
	  }
      }
    for (std::vector<DetId>::iterator it = listNewChan.begin(); it != listNewChan.end(); it++)  // fix 25.02.08
      {
	DetId mydetid = *it;
	const HcalQIECoder* myCoder = myNewQIEs->getCoder( mydetid );
	std::cout << "N";
	resultQIEs->addCoder( *myCoder );
      }

    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Dumping QIEs - the combined ones ---" << std::endl;
    resultQIEs->sort();
    HcalDbASCIIIO::dumpObject (outStream3, (*resultQIEs) );
    }

    std::cout << std::endl;
    if(checkemapflag){
    std::vector<DetId> listResult = resultQIEs->getAllChannels();
    // get the e-map list of channels
    std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
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

DEFINE_FWK_MODULE(HcalQIEDataCheck);
