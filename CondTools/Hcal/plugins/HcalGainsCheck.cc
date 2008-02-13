#include "CondTools/Hcal/interface/HcalGainsCheck.h"

HcalGainsCheck::HcalGainsCheck(edm::ParameterSet const& ps)
{
  rootfile = ps.getUntrackedParameter<std::string>("rootfile");
  outfile = ps.getUntrackedParameter<std::string>("outFile","Dump");

}

void HcalGainsCheck::beginJob(const edm::EventSetup& es)
{
  f = new TFile(rootfile.c_str(),"RECREATE");

  //book histos:
  ocMapUp = new TH2F("ocMapUp","occupancy_map_updated_gains",83,-41.5,41.5,72,0.5,72.5);
  ocMapRef = new TH2F("ocMapUp","occupancy_map_updated_gains",83,-41.5,41.5,72,0.5,72.5);
//  valMapUp;
//  valMapRef;

  diffUpRefCap0 = new TH1F("diffUpRefCap0","difference_update_reference_Cap0",100,-0.5,0.5);
  ratioUpRefCap0 = new TH1F("ratioUpRefCap0", "ration_update_reference_Cap0",100,0.5,1.5);
  gainsUpCap0 = new TH1F("gainsUpCap0","gains_update_Cap0",100,0.0,0.6);
  gainsRefCap0 = new TH1F("gainsRefCap0","gains_reference_Cap0",100,0.0,0.6);

  diffUpRefCap1 = new TH1F("diffUpRefCap1","difference_update_reference_Cap1",100,-0.5,0.5);
  ratioUpRefCap1 = new TH1F("ratioUpRefCap1", "ration_update_reference_Cap1",100,0.5,1.5);
  gainsUpCap1 = new TH1F("gainsUpCap1","gains_update_Cap1",100,0.0,0.6);
  gainsRefCap1 = new TH1F("gainsRefCap1","gains_reference_Cap1",100,0.0,0.6);

  diffUpRefCap2 = new TH1F("diffUpRefCap2","difference_update_reference_Cap2",100,-0.5,0.5);
  ratioUpRefCap2 = new TH1F("ratioUpRefCap2", "ration_update_reference_Cap2",100,0.5,1.5);
  gainsUpCap2 = new TH1F("gainsUpCap2","gains_update_Cap2",100,0.0,0.6);
  gainsRefCap2 = new TH1F("gainsRefCap2","gains_reference_Cap2",100,0.0,0.6);

  diffUpRefCap3 = new TH1F("diffUpRefCap3","difference_update_reference_Cap3",100,-0.5,0.5);
  ratioUpRefCap3 = new TH1F("ratioUpRefCap3", "ration_update_reference_Cap3",100,0.5,1.5);
  gainsUpCap3 = new TH1F("gainsUpCap3","gains_update_Cap3",100,0.0,0.6);
  gainsRefCap3 = new TH1F("gainsRefCap3","gains_reference_Cap3",100,0.0,0.6);

  //  gainsUpCap0vsEta = new TGraph("gainsUpCap0vsEta","gains_update_Cap0_vsEta",100,-41,0.6);
  //  gainsRefCap0vsEta = new TGraph("gainsRefCap0vsEta","gains_reference_Cap0_vsEta",100,0.0,0.6);
}


void HcalGainsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  using namespace edm::eventsetup;

  // get new gains
  edm::ESHandle<HcalGains> newGains;
  es.get<HcalGainsRcd>().get("update",newGains);
  const HcalGains* myNewGains = newGains.product();

  // get reference gains
  edm::ESHandle<HcalGains> refGains;
  es.get<HcalGainsRcd>().get("reference",refGains);
  const HcalGains* myRefGains = refGains.product();

  // get e-map from reference
  edm::ESHandle<HcalElectronicsMap> refEMap;
  es.get<HcalElectronicsMapRcd>().get("reference",refEMap);
  const HcalElectronicsMap* myRefEMap = refEMap.product();


    // dump gains:
//    std::ostringstream filename;
//    filename << "test_update.txt";
//    std::ofstream outStream(filename.str().c_str());
//    std::cout << "--- Dumping Gains - update ---" << std::endl;
//    HcalDbASCIIIO::dumpObject (outStream, (*myNewGains) );
//
//    std::ostringstream filename2;
//    filename2 << "test_reference.txt";
//    std::ofstream outStream2(filename2.str().c_str());
//    std::cout << "--- Dumping Gains - reference ---" << std::endl;
//    HcalDbASCIIIO::dumpObject (outStream2, (*myRefGains) );

    // get the list of all channels
    std::vector<DetId> listNewChan = myNewGains->getAllChannels();
    std::vector<DetId> listRefChan = myRefGains->getAllChannels();
    
    std::vector<DetId>::const_iterator cell;

    //plots: occupancy map, value map, difference, ratio, gains:
    for (std::vector<DetId>::const_iterator it = listRefChan.begin(); it!=listRefChan.end(); it++)
      {
	HcalGenericDetId myId(*it);
	//	ocMapRef->Fill(myId->);

	float valCap0 = myRefGains->getValue( (*it), 0);
	float valCap1 = myRefGains->getValue( (*it), 1);
	float valCap2 = myRefGains->getValue( (*it), 2);
	float valCap3 = myRefGains->getValue( (*it), 3);

	gainsRefCap0->Fill(valCap0);
	gainsRefCap1->Fill(valCap1);
	gainsRefCap2->Fill(valCap2);
	gainsRefCap3->Fill(valCap3);

	cell = std::find(listNewChan.begin(), listNewChan.end(), (*it));
	if (cell != listNewChan.end() ) //found
	  {
	    float valCap0up = myNewGains->getValue( (*it), 0);
	    float valCap1up = myNewGains->getValue( (*it), 1);
	    float valCap2up = myNewGains->getValue( (*it), 2);
	    float valCap3up = myNewGains->getValue( (*it), 3);
	    
	    diffUpRefCap0->Fill(valCap0up - valCap0);
	    diffUpRefCap1->Fill(valCap1up - valCap1);
	    diffUpRefCap2->Fill(valCap2up - valCap2);
	    diffUpRefCap3->Fill(valCap3up - valCap3);

	    ratioUpRefCap0->Fill(valCap0up / valCap0);
	    ratioUpRefCap1->Fill(valCap1up / valCap1);
	    ratioUpRefCap2->Fill(valCap2up / valCap2);
	    ratioUpRefCap3->Fill(valCap3up / valCap3);
	  }
      }
    for (std::vector<DetId>::const_iterator it = listNewChan.begin(); it!=listNewChan.end(); it++)
      {
	float valCap0 = myNewGains->getValue( (*it), 0);
	float valCap1 = myNewGains->getValue( (*it), 1);
	float valCap2 = myNewGains->getValue( (*it), 2);
	float valCap3 = myNewGains->getValue( (*it), 3);

	gainsUpCap0->Fill(valCap0);
	gainsUpCap1->Fill(valCap1);
	gainsUpCap2->Fill(valCap2);
	gainsUpCap3->Fill(valCap3);
      }

    // go through list of valid channels from reference, look up if conditions exist for update
    // push back into new vector the corresponding updated conditions,
    // or if it doesn't exist, the reference
    HcalGains *resultGains = new HcalGains();
    for (std::vector<DetId>::const_iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
	DetId mydetid = *it;
	HcalGenericDetId myId(*it);
	cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
	if (cell == listNewChan.end()) // not present in new list, take old conditions
	  {
	    const float* values = (myRefGains->getValues( mydetid ))->getValues();
	    std::cout << "o";
	    resultGains->addValue( (*it), values );
	  }
	else // present in new list, take new pedestals
	  {
	    const float* values = (myNewGains->getValues( mydetid ))->getValues();
	    std::cout << "n";
	    resultGains->addValue( (*it), values );
	  }
      }
    std::cout << std::endl;

    std::vector<DetId> listResult = resultGains->getAllChannels();
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
    //    std::ostringstream filename3;
    //    filename3 << "test_combined.txt";
    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Dumping Gains - the combined ones ---" << std::endl;
    resultGains->sort();
    HcalDbASCIIIO::dumpObject (outStream3, (*resultGains) );


    // const float* values = myped->getValues (channelID);
    //    if (values) std::cout << "pedestals for channel " << channelID << ": "
    //			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 

}


// ------------ method called once each job just after ending the event loop  ------------
void 
HcalGainsCheck::endJob() 
{

  f->Write();

  f->Close();

}

DEFINE_FWK_MODULE(HcalGainsCheck);
