#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 


#include <vector>
using std::vector;
#include <iostream>



using std::cout;
using std::endl;
const int L1RCTProducer::crateFED[18][5]=
      {{613, 614, 603, 702, 718},
    {611, 612, 602, 700, 718},
    {627, 610, 601,716,   722},
    {625, 626, 609, 714, 722},
    {623, 624, 608, 712, 722},
    {621, 622, 607, 710, 720},
    {619, 620, 606, 708, 720},
    {617, 618, 605, 706, 720},
    {615, 616, 604, 704, 718},
    {631, 632, 648, 703, 719},
    {629, 630, 647, 701, 719},
    {645, 628, 646, 717, 723},
    {643, 644, 654, 715, 723},
    {641, 642, 653, 713, 723},
    {639, 640, 652, 711, 721},
    {637, 638, 651, 709, 721},
    {635, 636, 650, 707, 721},
    {633, 634, 649, 705, 719}};



L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigis(conf.getParameter<std::vector<edm::InputTag> >("ecalDigis")),
  hcalDigis(conf.getParameter<std::vector<edm::InputTag> >("hcalDigis")),
  bunchCrossings(conf.getParameter<std::vector<int> >("BunchCrossings")),
  fedUpdatedMask(0)
{
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

 

}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
  if(fedUpdatedMask != 0) delete fedUpdatedMask;
}


void L1RCTProducer::beginRun(edm::Run& run, const edm::EventSetup& eventSetup)
{

}

void L1RCTProducer::beginLuminosityBlock(edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context)
{
  updateConfiguration(context);
} 



void L1RCTProducer::updateConfiguration(const edm::EventSetup& eventSetup)
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
  const L1RCTChannelMask* cEs = channelMask.product();


  // list of Noisy RCT channels to mask
  edm::ESHandle<L1RCTNoisyChannelMask> hotChannelMask;
  eventSetup.get<L1RCTNoisyChannelMaskRcd>().get(hotChannelMask);
  const L1RCTNoisyChannelMask* cEsNoise = hotChannelMask.product();
  rctLookupTables->setNoisyChannelMask(cEsNoise);


  
  //Update the channel mask according to the FED VECTOR
  //This is the beginning of run. We delete the old
  //create the new and set it in the LUTs

  if(fedUpdatedMask!=0) delete fedUpdatedMask;

  fedUpdatedMask = new L1RCTChannelMask();
  // copy a constant object
   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       fedUpdatedMask->ecalMask[i][j][k] = cEs->ecalMask[i][j][k];
	       fedUpdatedMask->hcalMask[i][j][k] =cEs->hcalMask[i][j][k] ;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       fedUpdatedMask->hfMask[i][j][k] = cEs->hfMask[i][j][k];
	     }
	 }
     }


  // adding fed mask into channel mask
  edm::ESHandle<RunInfo> sum;
  eventSetup.get<RunInfoRcd>().get(sum);
  const RunInfo* summary=sum.product();


  std::vector<int> caloFeds;  // pare down the feds to the intresting ones
  // is this unneccesary?
  // Mike B : This will decrease the find speed so better do it
  const std::vector<int> Feds = summary->m_fed_in;
  for(std::vector<int>::const_iterator cf = Feds.begin(); cf != Feds.end(); ++cf){
    int fedNum = *cf;
    if(fedNum > 600 && fedNum <724) 
      caloFeds.push_back(fedNum);
  }

  for(int  cr = 0; cr < 18; ++cr){

    for(crateSection cs = c_min; cs <= c_max; cs = crateSection(cs +1)) {
      bool fedFound = false;


      //Try to find the FED
      std::vector<int>::iterator fv = std::find(caloFeds.begin(),caloFeds.end(),crateFED[cr][cs]);
      if(fv!=caloFeds.end())
	fedFound = true;

      if(!fedFound) {
	int eta_min=0;
	int eta_max=0;
	bool phi_even[2] = {false};//, phi_odd = false;
	bool ecal=false;
	
	switch (cs) {
	case ebEvenFed :
	  eta_min = minBarrel;
	  eta_max = maxBarrel;
	  phi_even[0] = true;
	  ecal = true;	
	  break;
	  
	case ebOddFed:
	  eta_min = minBarrel;
	  eta_max = maxBarrel;
	  phi_even[1] = true;
	  ecal = true;	
	  break;
	  
	case eeFed:
	  eta_min = minEndcap;
	  eta_max = maxEndcap;
	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = true;	
	  break;	
	  
	case hbheFed:
	  eta_min = minBarrel;
	  eta_max = maxEndcap;
	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = false;
	  break;

	case hfFed:	
	  eta_min = minHF;
	  eta_max = maxHF;

	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = false;
	  break;
	default:
	  break;

	}
	for(int ieta = eta_min; ieta <= eta_max; ++ieta){
	  if(ieta<=28) // barrel and endcap
	    for(int even = 0; even<=1 ; even++){	 
	      if(phi_even[even]){
		if(ecal)
		  fedUpdatedMask->ecalMask[cr][even][ieta-1] = true;
		else
		  fedUpdatedMask->hcalMask[cr][even][ieta-1] = true;
	      }
	    }
	  else
	    for(int even = 0; even<=1 ; even++)
	      if(phi_even[even])
		  fedUpdatedMask->hfMask[cr][even][ieta-29] = true;
	
	}
      }
    }
  }


  //SCALES

  // energy scale to convert eGamma output
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();


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


  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(fedUpdatedMask);
  rctLookupTables->setL1CaloEtScale(s);
}


void L1RCTProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{


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
	    std::vector<L1CaloRegion> regions = rct->getRegions(i);
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
  
}
