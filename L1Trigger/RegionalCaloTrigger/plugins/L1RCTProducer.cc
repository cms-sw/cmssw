#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// default scales
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

// debug scales
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigis(conf.getParameter<std::vector<edm::InputTag> >("ecalDigis")),
  hcalDigis(conf.getParameter<std::vector<edm::InputTag> >("hcalDigis")),
  bunchCrossings(conf.getParameter<std::vector<int> >("BunchCrossings")),
  useDebugTpgScales(conf.getParameter<bool>("useDebugTpgScales"))
{
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void L1RCTProducer::beginJob(const edm::EventSetup& eventSetup)
{
}

void L1RCTProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{

  // Refresh configuration information every event
  // Hopefully, this does not take too much time
  // There should be a call back function in future to
  // handle changes in configuration

  // parameters to configure RCT (thresholds, etc)
  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  eventSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
  const L1RCTChannelMask* c = channelMask.product();
  
  // energy scale to convert eGamma output
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();

  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(c);
  rctLookupTables->setL1CaloEtScale(s);

  // use these dummies to get the delete right when using old-style
  // scales to create set of L1CaloXcalScales
  L1CaloEcalScale* dummyE(0);
  L1CaloHcalScale* dummyH(0);

  //Really wanted to drop that too but 
  //I thought it might be OK to leave it for a while more
 
  if (useDebugTpgScales) // generate new-style scales from tpg scales
    {
      // old version of hcal energy scale to convert input
      edm::ESHandle<CaloTPGTranscoder> transcoder;
      eventSetup.get<CaloTPGRecord>().get(transcoder);
      const CaloTPGTranscoder* h_tpg = transcoder.product();

      // old version of ecal energy scale to convert input
      EcalTPGScale* e_tpg = new EcalTPGScale();
      e_tpg->setEventSetup(eventSetup);

      L1CaloEcalScale* ecalScale = new L1CaloEcalScale();
      L1CaloHcalScale* hcalScale = new L1CaloHcalScale();

      // generate L1CaloXcalScales from old-style scales (thanks, werner!)
      // ECAL

      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	  {
	    EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;
	    double etGeVPos =
	      e_tpg->getTPGInGeV
	      ( irank, EcalTrigTowerDetId(1, // +ve eta
					  subdet,
					  ieta,
					  1 )); // dummy phi value
	    ecalScale->setBin( irank, ieta, 1, etGeVPos ) ;
	  }
      
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	  {
	    EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;
	    double etGeVNeg =
	      e_tpg->getTPGInGeV
	      ( irank,
		EcalTrigTowerDetId(-1, // -ve eta
				   subdet,
				   ieta,
				   2 )); // dummy phi value
	    ecalScale->setBin( irank, ieta, -1, etGeVNeg ) ;
	  }
      
      //HCAL -  positive eta
      for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta )
	for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank )
	  {
	    double etGeVPos = h_tpg->hcaletValue( ieta, irank ) ;
	    hcalScale->setBin( irank, ieta, 1, etGeVPos ) ;
	  }

      //HCAL - negative eta
      for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta )
	for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank )
	  {
	    double etGeVNeg = h_tpg->hcaletValue( -ieta, irank ) ;
	    hcalScale->setBin( irank, ieta, -1, etGeVNeg ) ;
	    
	  }

      // set the input scales
      rctLookupTables->setEcalScale(ecalScale);
      rctLookupTables->setHcalScale(hcalScale);

      dummyE = ecalScale;
      dummyH = hcalScale;

      delete e_tpg;
      delete h_tpg; 
    }
  else
    {

      // get energy scale to convert input from ECAL
      edm::ESHandle<L1CaloEcalScale> ecalScale;
      eventSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
      const L1CaloEcalScale* e = ecalScale.product();
      
      // get energy scale to convert input from HCAL
      edm::ESHandle<L1CaloHcalScale> hcalScale;
      eventSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
      const L1CaloHcalScale* h = hcalScale.product();

      // set scales
      rctLookupTables->setEcalScale(e);
      rctLookupTables->setHcalScale(h);

    }




  std::auto_ptr<L1CaloEmCollection> rctEmCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);


  if(!(ecalDigis.size()==hcalDigis.size()&&hcalDigis.size()==bunchCrossings.size()))
      throw cms::Exception("BadInput")
	<< "From what I see the number of your your ECAL input digi collections.\n"
        <<"is different from the size of your HCAL digi input collections\n"
	<<"or the size of your BX factor collection" 
        <<"They must be the same to correspond to the same Bxs\n"
	<< "It does not matter if one of them is empty\n"; 




  // loop through and process each bx
    for (unsigned short sample = 0; sample < bunchCrossings.size(); sample++)
      {
	edm::Handle<EcalTrigPrimDigiCollection> ecal;
	edm::Handle<HcalTrigPrimDigiCollection> hcal;

	if (useEcal) { event.getByLabel(ecalDigis[sample], ecal); }
	if (useHcal) {  }

	EcalTrigPrimDigiCollection ecalIn;
	HcalTrigPrimDigiCollection hcalIn;


	if(useHcal&&event.getByLabel(hcalDigis[sample], hcal))
	  hcalIn = *hcal;

	if(useEcal&&event.getByLabel(ecalDigis[sample],ecal))
	  ecalIn = *ecal;

	rct->digiInput(ecalIn,hcalIn);
	rct->processEvent();

      // Stuff to create
	for (int j = 0; j<18; j++)
	  {
	    L1CaloEmCollection isolatedEGObjects = rct->getIsolatedEGObjects(j);
	    L1CaloEmCollection nonisolatedEGObjects = rct->getNonisolatedEGObjects(j);
	    for (int i = 0; i<4; i++) 
	      {
		isolatedEGObjects.at(i).setBx(bunchCrossings[sample]);
		nonisolatedEGObjects.at(i).setBx(bunchCrossings[sample]);
		rctEmCands->push_back(isolatedEGObjects.at(i));
		rctEmCands->push_back(nonisolatedEGObjects.at(i));
	      }
	  }
      
      
	for (int i = 0; i < 18; i++)
	  {
	    vector<L1CaloRegion> regions = rct->getRegions(i);
	    for (int j = 0; j < 22; j++)
	      {
		regions.at(j).setBx(bunchCrossings[sample]);
		rctRegions->push_back(regions.at(j));
	      }
	  }

      }

  
  //putting stuff back into event
  event.put(rctEmCands);
  event.put(rctRegions);

  if (dummyE != 0) delete dummyE;
  if (dummyH != 0) delete dummyH;
  
}
