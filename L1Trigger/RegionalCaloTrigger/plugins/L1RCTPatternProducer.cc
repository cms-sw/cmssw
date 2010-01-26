#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
using std::ostream;
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>




#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTPatternProducer.h"

L1RCTPatternProducer::L1RCTPatternProducer(const edm::ParameterSet& conf) :
  fileName(conf.getUntrackedParameter<std::string>("rctTestInputFile")),
  rctLookupTables(new L1RCTLookupTables), 
  fgEcalE(conf.getUntrackedParameter<int>("fgEcalE")),
  testName(conf.getUntrackedParameter<std::string>("testName","none")),
  randomPercent(conf.getUntrackedParameter<int>("randomPercent",5)),
  randomSeed(conf.getUntrackedParameter<int>("randomSeed",12345)),
  regionSums(conf.getUntrackedParameter<bool>("regionSums",true)),
 nPatternEv(conf.getUntrackedParameter<int>("nPatternEv",64)),
   nSamplesPerEv(conf.getUntrackedParameter<int>("nSamplesPerEv",1)),
  nPreSamples(conf.getUntrackedParameter<int>("nPreSamples",1))
{
  produces<EcalTrigPrimDigiCollection>();
  //produces<EcalTrigPrimDigiCollection>("formatTCP");
  //  produces<HcalTrigPrimDigiCollection>();
  produces<HBHEDigiCollection>();
  produces<HFDigiCollection>();
  
  cout << " nsamples per ev " << nSamplesPerEv << " nPatternEv " << nPatternEv << endl;

  fileName = testName+"Input.txt";

//   ofs.open(fileName.c_str());//, std::ios::app);
//   if(!ofs)
//     {
//       std::cerr << "Could not create " << fileName << std::endl;
//       exit(1);
//     }
}

L1RCTPatternProducer::~L1RCTPatternProducer()
{
  //  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void
L1RCTPatternProducer::produce(edm::Event& event,
			      const edm::EventSetup& eventSetup)
{
  cout << "producing patterns" << endl;
  using namespace edm;

  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();
  //edm::ESHandle<CaloTPGTranscoder> transcoder;
  //eventSetup.get<CaloTPGRecord>().get(transcoder);
  //const CaloTPGTranscoder* h_tpg = transcoder.product();
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();
  rctLookupTables->setRCTParameters(r);
  // rctLookupTables->setTranscoder(t);
  rctLookupTables->setL1CaloEtScale(s);

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  eventSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
  const L1RCTChannelMask* c = channelMask.product();
  rctLookupTables->setChannelMask(c);

  std::auto_ptr<EcalTrigPrimDigiCollection>
    ecalTPs(new EcalTrigPrimDigiCollection());
  //std::auto_ptr<EcalTrigPrimDigiCollection>
    // tcpEcalTPs(new EcalTrigPrimDigiCollection());
  //  std::auto_ptr<HcalTrigPrimDigiCollection> 
  //    hcalTPs(new HcalTrigPrimDigiCollection());
  ecalTPs->reserve(56*72); 
  //  hcalTPs->reserve(56*72+18*8);  // includes HF

  std::auto_ptr<HBHEDigiCollection> 
    digiHBHE(new HBHEDigiCollection());
  std::auto_ptr<HFDigiCollection> 
    digiHF(new HFDigiCollection());

  const int nEcalSamples = nSamplesPerEv;   // we only use 1 sample for each
  const int nHcalSamples = nSamplesPerEv;
  const int nHBHESamples = nSamplesPerEv;
  const int nHFSamples = nSamplesPerEv;

  long randomNum = 0;
  long NineBit = 0;
  long Overflow = 0;
  static int nEvents = 0;

//   if(nEvents==0) {
//     ofs
//       << "Crate = 0-17" << std::endl
//       << "Card = 0-7 within the crate" << std::endl
//       << "Tower = 0-31 covers 4 x 8 covered by the card" << std::endl
//       << "EMAddr(0:8) = EMFGBit(0:0)+CompressedEMET(1:8)" << std::endl
//       << "HDAddr(0:8) = HDFGBit(0:0)+CompressedHDET(1:8) - note: HDFGBit(0:0) is not part of the hardware LUT address" << std::endl
//       << "LutOut(0:17)= LinearEMET(0:6)+HoEFGVetoBit(7:7)+LinearJetET(8:16)+ActivityBit(17:17)" << std::endl
//       << "Event" << "\t"
//       << "Crate" << "\t"
//       << "Card" << "\t"
//       << "Tower" << "\t"
//       << "EMAddr" << "\t"
//       << "HDAddr" << "\t"
//       << "LUTOut"
//       << std::endl;
//   }

  if(nEvents < nPatternEv)    {

      for(unsigned short iCrate = 0; iCrate < 18; iCrate++){
	for(unsigned short iCard = 0; iCard < 7; iCard++){    
	  //  unsigned short ecal, fgbit, hcal, mubit;  
	  int tt = (nEvents%3)*2;
	  //int s = 0;

	  fgbit =0;
	  hcal = 0;
	  mubit =0;
	  ecal = 0;
	  //int TowersHit = (int) rand()%32+1;
	  //int Energy=(int)rand()%(0x80);
	  Etot = 0;
	  randomNum = rand()%100;
	  if(randomNum>70&&randomNum<=90)
	    NineBit = 1;
	  if(randomNum>90)
	    Overflow = 1;
	  //int k = 0;

	  //set up event 'bunches'
	  bool eBunch4[16] = {false};
	  bool eBunch8[8] = {false};
	  int eNum4 = (nEvents-1)/4;
	  int eNum8 = (nEvents-1)/8;
	  eBunch4[eNum4]=true;
	  eBunch8[eNum8]=true;

	  // tower numbered from 0-31
	  for(unsigned short iTower = 0; iTower < 32; iTower++)
	    {
	      if(nEvents==0)
		firstEvent(iCrate,iCard,iTower);
	      // Add other events here.....
	  
	      else if(testName=="random") {  //Random
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		  if(rand()%100<randomPercent) //fill some % of the towers
		  ecal = rand()%(0x1f); //with random energy < 64
 		if(rand()%100<(randomPercent/5))  //include fgbit 1/5 as often of the time
		  fgbit = 1;
		if(regionSums){
		  //Compatable with HCAL link test
		  //iEta>21 --> phi divisions are 2 towers wide
		  if(iCard==6 || ((iCard==5 || iCard==4) && iTower>=16)){
		    if(iTower%2!=0) { 
		      if(rand()%100<randomPercent)
			hcal = rand()%(0x2f);
		    }
		  }
		  else {
		    if(rand()%100<randomPercent)
		      hcal = rand()%(0x2f);
		  }
		}
	      }	      
	      else if(testName=="walkingOnes") { //Walking ones
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;

		if(iTower==10||iTower==5)
		  {
		    if(nEvents<32){
		      if(iCard==2||iCard==3)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = ((1<<(tt+1))&0x3F);
		    }
		    if(nEvents>=32){
		      if(iCard==0||iCard==1)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==2||iCard==3)
			ecal = ((1<<(tt+1))&0x3F);
		    }		      
		  }
		//to make tower 5 non-iso
		if((iCard<=3)&&(iTower==15)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="walkingZeros") { //Walking zeros
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(iTower==10||iTower==5)
		  {
		    if(nEvents<32){
		      if(iCard==2||iCard==3)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = (~(1<<(tt+1))&0x3F);
		    }
		    if(nEvents>=32){
		      if(iCard==0||iCard==1)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==2||iCard==3)
			ecal = (~(1<<(tt+1))&0x3F);
		    }		      
		  }
		//to make tower 5 non-iso
		if((iCard<=3)&&(iTower==15)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}		
		//Jet info
		if(regionSums){
		  walkZeroHCAL(nEvents, iCard, iTower);
		}
	      }
	      else if(testName=="walkingOnesZeros") { //Walking ones--iso, zeros--niso
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		//ISO
		if(nEvents<32){
		  if(iTower==5)
		    {
		      if(iCard==2||iCard==3)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = ((1<<(tt+1))&0x3F);		      
		    }
		  //NISO
		  if(iTower==10)
		    {
		      if(iCard==2||iCard==3)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		if(nEvents>=32){
		  if(iTower==5)
		    {
		      if(iCard==2||iCard==3)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = ((1<<(tt+1))&0x3F);		      
		    }
		  //NISO
		  if(iTower==10)
		    {
		      if(iCard==2||iCard==3)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==0||iCard==1)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		//to make tower 10 non-iso
		if((iCard<=3)&&(iTower==14)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}
		//Jet info
		if(regionSums){
		  walkHCAL(nEvents, iCard, iTower);
		}
	      }
	      else if(testName=="cornerSharing") {  //checking corner sharing using fg or quiet corner veto
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
		bool botRight=false, botLeft=false, topRight =false, topLeft = false;
		if(iTower==0&&iCard==0&&iCrate<9) botRight = true;
		if(iTower==0&&iCard==0&&iCrate>8) botLeft = true;
		if(iTower==3&&iCard==1&&iCrate<9) topRight = true;
		if(iTower==3&&iCard==1&&iCrate>8) topLeft = true;
		//single electron in one of the four eta=0 corners
		if((eBunch8[0]&&botRight) || 
		   (eBunch8[1]&&botLeft) || 
		   (eBunch8[2]&&topRight)||
		   (eBunch8[3]&&topLeft))
		  ecal = 0x3f;
		//set fgbit on opposite corner
		//will find some iso electrons of rank 4 (unless threshold set lower...)
		if((eBunch4[0]&&topLeft)||
		   (eBunch4[2]&&topRight)||
		   (eBunch4[4]&&botLeft)||
		   (eBunch4[6]&&botRight))
		  {
		    ecal = fgEcalE;
		    fgbit = nEvents%2;
		  }
		//set quiet corners (energy in opposite corner tower, and tower in same card)
		if((eBunch4[1]&&(topLeft ||(iTower==1&&iCard==0&&iCrate<9)))||
		   (eBunch4[3]&&(topRight||(iTower==1&&iCard==0&&iCrate>8)))||
		   (eBunch4[5]&&(botLeft ||(iTower==2&&iCard==1&&iCrate<9)))||
		   (eBunch4[7]&&(botRight||(iTower==2&&iCard==1&&iCrate>8))))
		  {
		    if(nEvents%2==0)
		      ecal = 0x04; //currently shows up as an isolated electron too
		    if(nEvents%4==0)
		      ecal = 0x03;
		  }

		//check eta=0 edge too
		if((iCard==0||iCard==1)&&iTower==nEvents%4){
		  if((eBunch8[4]&&iCrate<9)||
		     (eBunch8[5]&&iCrate>8))
		    ecal = 0x3f;
		  if((eBunch8[4]&&iCrate>8)||
		     (eBunch8[5]&&iCrate<9))
		    {
		      fgbit = 1;
		      ecal = fgEcalE;
		    }
		  if((eBunch8[6]&&iCrate<9)||
		     (eBunch8[7]&&iCrate>8))
		    ecal = 0x2a;
		  if((eBunch8[6]&&iCrate>8)||
		     (eBunch8[7]&&iCrate<9))
		    ecal = 0x15;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }
	      else if (testName=="edges") {
		//eBunch8[0]
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		int walk = (nEvents%8)*4; //selects tower based on event number
		if(iCrate<9) walk = ((nEvents+1)%8)*4; //to avoid interference at eta=0 boundary
		//ISO
		if((eBunch8[0]&&iTower==walk && (iCard==0||iCard==2))||
		   (eBunch8[1]&&iTower==walk+3 && (iCard==1||iCard==3))||
		   (eBunch8[2]&&iTower==walk && iCard==4)||
		   (eBunch8[3]&&iTower==walk+3 && iCard==5))
		  ecal = 0x15;
		else if((eBunch8[0]&&iTower==walk+3 && (iCard==1||iCard==3))||
			(eBunch8[1]&&iTower==walk && (iCard==0||iCard==2))||
			(eBunch8[2]&&iTower==walk+3 && iCard==5)||
			(eBunch8[3]&&iTower==walk && iCard==4))
		  ecal = 0x2a;
		
		//NONISO
		if((eBunch8[4]&&iTower==walk   && (iCard==0||iCard==2))||
		   (eBunch8[5]&&iTower==walk+3 && (iCard==1||iCard==3))||
		   (eBunch8[6]&&iTower==walk   && iCard==4)||
		   (eBunch8[7]&&iTower==walk+3 && iCard==5))
		  {
		    fgbit = 1;
		    ecal = fgEcalE; 
		  } 
		else if((eBunch8[4]&&iTower==walk+3 && (iCard==1||iCard==3))||
			(eBunch8[5]&&iTower==walk   && (iCard==0||iCard==2))||
			(eBunch8[6]&&iTower==walk+3 && iCard==5)||
			(eBunch8[7]&&iTower==walk   && iCard==4))
		  ecal = 0x3f;	

		//ISO
		//card 6 has different numbering:
		int walk6 =(nEvents%4)*4;
		if(iCard==6) {
		  if(eBunch8[2]) {//test ecal sharing
		    if(iTower==walk6) //walk along top edge, based on event number
		      ecal = 0x15;
		    if(iTower==(31-walk6)) //walk along bottom edge, based on event number
		      ecal = 0x2a;
		  }
		  if(eBunch8[3]){//upper region, larger energy
		    if(iTower==walk6){
		      ecal = 0x2a;
		    }
		    if(iTower==(31-walk6))
		      ecal = 0x15;
		  }
		}
		//NONISO
		if(iCard==6) {  
		  //test fine grain veto
		  if((eBunch4[12]&&iTower==walk6) ||
		     (eBunch4[13]&&iTower==(31-walk6)))
		    {
		      fgbit = 1;
		      ecal = fgEcalE; 
		    } 
		  else if((eBunch4[12]&&iTower==(31-walk6))||
			  (eBunch4[13]&&iTower==walk6))
		    ecal = 0x3f;

		  //test quiet corner veto
		  if((eBunch4[14]&&iTower==walk6) ||
		     (eBunch4[15]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[14]&&iTower==walk6+1) ||
			  (eBunch4[15]&&iTower==(30-walk6)))
		    ecal = 0x05;
		  else if((eBunch4[14]&&iTower==(31-walk6))||
			  (eBunch4[15]&&iTower==walk6))
		    ecal = 0x0a;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="intraEdges") { //checks sharing between cards within a crate
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
		//want to switch active pins between events (even/odd) -- necessary???
		int walk =(nEvents%8)*4;
		int walk4 = (nEvents%4);
		//top/bottom sharing
		if((nEvents<17&&nEvents%2&&iTower==walk+3 && (iCard==0||iCard==2))||
		   (nEvents<17&&nEvents%2==0&&iTower==walk && (iCard==1||iCard==3))||
		   ((eBunch8[2]||eBunch8[3])&&nEvents%2&&iTower==walk+3 && iCard==4)||
		   ((eBunch8[3]||eBunch8[2])&&nEvents%2==0&&iTower==walk && iCard==5))
		  ecal = 0x2a;
		else if((nEvents<17&&nEvents%2&&iTower==walk && (iCard==1||iCard==3))||
			(nEvents<17&&nEvents%2==0&&iTower==walk+3 && (iCard==0||iCard==2))||
			((eBunch8[2]||eBunch8[3])&&nEvents%2&&iTower==walk && iCard==5)||
			((eBunch8[3]||eBunch8[2])&&nEvents%2==0&&iTower==walk+3 && iCard==4))
		  ecal = 0x15;

		//left/right sharing
		if((eBunch4[8]&&(iCard==2||iCard==3)&&iTower==walk4)||
		   (eBunch4[9]&&(iCard==0||iCard==1)&&iTower==walk4+28))
		  ecal = 0x2a;
		if((eBunch4[8]&&(iCard==0||iCard==1)&&iTower==walk4+28)||
		   (eBunch4[9]&&(iCard==2||iCard==3)&&iTower==walk4))
		  ecal = 0x15;
		if((eBunch4[10]&&(iCard==2||iCard==3)&&iTower==walk4+28)||
		   (eBunch4[11]&&(iCard==4||iCard==5)&&iTower==walk4))
		  ecal = 0x2a;
		if((eBunch4[10]&&(iCard==4||iCard==5)&&iTower==walk4)||
		   (eBunch4[11]&&(iCard==2||iCard==3)&&iTower==walk4+28))
		  ecal = 0x15;
		//left/right with card 6
		if((eBunch4[12]&&(iCard==6)&&(iTower==walk4||iTower==walk4+28))||
		   (eBunch4[13]&&(iCard==4||iCard==5)&&iTower==walk4+28))
		  ecal = 0x2a;
		if((eBunch4[12]&&(iCard==4||iCard==5)&&iTower==walk4+28)||
		   (eBunch4[13]&&(iCard==6)&&(iTower==walk4||iTower==walk4+28)))
		  ecal = 0x15;

		//put walking ones into hcal
		//if(regionSums)
		//walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="testCard6"){  //checking card 6 sharing using energies and fg or quiet corner veto for non-iso
		fgbit = 0; 
		hcal  = 0; 
		mubit = 0; 
		ecal  = 0; 
		int walk6 = (nEvents%4)*4;
		if(iCard==6){

		  if((eBunch4[0]&&iTower==walk6) ||
		     (eBunch4[1]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[0]&&iTower==walk6+1) ||
			  (eBunch4[1]&&iTower==(30-walk6)))
		    ecal = 0xa;
		  else if((eBunch4[0]&&iTower==(31-walk6))||
			  (eBunch4[1]&&iTower==walk6))
		    ecal = 0x03;

		  if((eBunch4[2]&&iTower==walk6) ||
		     (eBunch4[3]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[2]&&iTower==walk6+1)||
			  (eBunch4[3]&&iTower==(30-walk6)))
		    ecal = 0xa;
		  else if((eBunch4[2]&&iTower==(31-walk6))||
			  (eBunch4[3]&&iTower==walk6))
		    ecal = 0x04;

		  if((eBunch4[4]&&iTower==walk6) ||
		     (eBunch4[5]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[4]&&iTower==walk6+1) ||
			  (eBunch4[5]&&iTower==(30-walk6)))
		    ecal = 0xa;
		  else if((eBunch4[4]&&iTower==(31-walk6))||
			  (eBunch4[5]&&iTower==walk6))
		    ecal = 0x08;

		  if((eBunch4[6]&&iTower==walk6) ||
		     (eBunch4[7]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[6]&&iTower==walk6+1) ||
			  (eBunch4[7]&&iTower==(30-walk6)))
		    ecal = 0x04; // HACK 0x4;
		  else if((eBunch4[6]&&iTower==(31-walk6))||
			  (eBunch4[7]&&iTower==walk6))
		    ecal = 0x05;
		  
		  if((eBunch4[8]&&iTower==walk6) ||
		     (eBunch4[9]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[8]&&iTower==(31-walk6))||
			  (eBunch4[9]&&iTower==walk6))
		    ecal = 0x10;
		  
		  if((eBunch4[10]&&iTower==walk6) ||
		     (eBunch4[11]&&iTower==(31-walk6)))
		    ecal = 0x10;
		  else if((eBunch4[10]&&iTower==(31-walk6))||
			  (eBunch4[11]&&iTower==walk6))
		    ecal = 0x20;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);
	      }
	      else if(testName=="flooding") {//Tom's test
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
		//single electron in one of the four eta=0 corners, even events 
		if(nEvents<16)
		  if(iTower==0&&iCard==0&&iCrate<9)
		    ecal = (0x3f);
		if(nEvents>15&&nEvents<32)
		  if(iTower==0&&iCard==0&&iCrate>8)
		    ecal = (0x3f);
		if(nEvents>31&&nEvents<48)
		  if(iTower==3&&iCard==1&&iCrate<9)
		    ecal = (0x3f);
		if(nEvents>47&&nEvents<64)
		  if(iTower==3&&iCard==1&&iCrate>8)
		    ecal = (0x3f);
		//flood with energy on odd events (should not find iso electrons)
		if(nEvents%2==1&&iCard<2) {
		  ecal = (rand()%(0x37)+4); //need at least ecal=4 for fg bit
		  fgbit = 1;
		}
		//put walking ones into hcal
		//if(regionSums)
		  //walkHCAL(nEvents, iCard, iTower);
	      }
	      else if(testName=="allTowers") { //All Tower tests (ISO)
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
		//Fill ecal towers corresponding to event number (convenient)
		if(nEvents<32&&iTower==nEvents) {
		  if(iCrate<9&&iCard<4)
		    ecal = (0x3f);
		  if(iCrate>8&&iCard>3)
		    ecal = (0x3f);
		}
		//Fill hcal tower not nearby
		if(nEvents<16&&iTower==nEvents+16){
		  if(iCrate<9&&iCard<4)
		    hcal = (0x3f);
		  if(iCrate>8&&iCard>3)
		    hcal = (0x3f);
		}
		if(nEvents>15&&iTower==nEvents-16){
		  if(iCrate<9&&iCard<4)
		    hcal = (0x3f);
		  if(iCrate>8&&iCard>3)
		    hcal = (0x3f);
		}
		//Fill second set of ecal towers
		if(nEvents>31 && iTower==(nEvents-32)){
		  if(iCrate<9&&iCard>3)
		    ecal = (0x3f);
		  if(iCrate>8&&iCard<4)
		    ecal = (0x3f);
		}
		//Fill second set of hcal towers
		if(nEvents<48&&nEvents>31&&iTower==nEvents-16){
		  if(iCrate<9&&iCard>3)
		    hcal = (0x3f);
		  if(iCrate>8&&iCard<4)
		    hcal = (0x3f);
		}
		if(nEvents>47&&nEvents<64&&iTower==nEvents-48){
		  if(iCrate<9&&iCard>3)
		    hcal = (0x3f);
		  if(iCrate>8&&iCard<4)
		    hcal = (0x3f);
		}

		//put walking ones into hcal
		//if(regionSums)
		//walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="zeroMax") { //Switch tower to max (0x3F) and zero
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
		if(nEvents<32) //Iso
		  if(iTower==6&&iCard>3&&(nEvents%2))
		    ecal = (0x3f);
		if(nEvents>31&&nEvents<64){ //non-iso
		  if(iTower==6&&iCard>3&&(nEvents%2))
		    ecal = (0x3f);
		  if(iTower==10&&iCard>3&&(nEvents%2)) {
		    ecal = (fgEcalE);
		    fgbit = 1*(nEvents%2);
		  }
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="regionSumPins0") { //Check Region Sum pins specifically (see in04_009)
		// 000 [card] 0 [region] 000000 [rank]
		//tower 5 = region 0, tower 26 = region 1
		//tower 10/21&15/16, create noniso in region 0/1
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents>0) { //OKKKKKK

		  //Electrons
		  if(iCard==0) {
		    if(iTower==5)  // 0000000001
		      ecal = 0x01;
		    if(iTower==10)
		      ecal = 0x01;
		    if(iTower==15) {
		      ecal = 0x01;
		      fgbit = 1;
		    }
		  }
		  if(iCard==6){
		    if(iTower==26) // 1101111110
		      ecal = 0x3e;
		    if(iTower==21)
		      ecal = 0x3e;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==5){
		    if(iTower==26) // 1011011100
		      ecal = 0x1c;
		    if(iTower==21)
		      ecal = 0x1c;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==2){
		    if(iTower==5)  // 0100100011
		      ecal = 0x23;
		    if(iTower==10)
		      ecal = 0x23;
		    if(iTower==15) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  //jet/sum
		  if(regionSums)
		    jetSumPins(nEvents, iCard, iTower,0);
		}
	      }
	      else if(testName=="regionSumPins1")///not gooooood (maybe)
		{
		  fgbit =0;
		  hcal = 0;
		  mubit =0;
		  ecal = 0;
		  if(nEvents>0){ //OKKKKK
		    if(iCard==5){
		      if(iTower==5)  // 0101010101
			ecal = 0x15;
		      if(iTower==10)
			ecal = 0x15;
		      if(iTower==15) {
			ecal = fgEcalE;
			fgbit = 1;
		      }
		    }
		    if(iCard==2){
		      if(iTower==26) // 1010101010
			ecal = 0x10;
		      if(iTower==21)
			ecal = 0x10;
		      if(iTower==16) {
			ecal = fgEcalE;
			fgbit = 1;
		      }
		    }
		    if(iCard==6){
		      if(iTower==26) // 0010110011
			ecal = 0xc;
		      if(iTower==21)
			ecal = 0xc;
		      if(iTower==16) {
			ecal = fgEcalE;
			fgbit = 1;
		      }
		    }
		    if(iCard==1){
		      if(iTower==5) // 1101001100
			ecal = 0x33;
		      if(iTower==10)
			ecal = 0x33;
		      if(iTower==15) {
			ecal = fgEcalE;
			fgbit = 1;
		      }
		    }
		    //jet/sum
		    if(regionSums)
		      jetSumPins(nEvents, iCard, iTower,1);
		  }
		}
	      else if(testName=="regionSumPins2"){
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents>0) { //OKKKKKKK
		  if(iCard==2){
		    if(iTower==5)  // 1011111001
		      ecal = 0x06;
		    if(iTower==10)
		      ecal = 0x06;
		    if(iTower==15) {
		      ecal = 0x05;
		      fgbit = 1;
		    }
		  }
		  if(iCard==5){
		    if(iTower==26) // 0100000110
		      ecal = 0x39;
		    if(iTower==21)
		      ecal = 0x39;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==1){
		    if(iTower==26) // 1100011101
		      ecal = 0x22;
		    if(iTower==21)
		      ecal = 0x22;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==6){
		    if(iTower==5)  // 0011100010
		      ecal = 0x1d;
		    if(iTower==10)
		      ecal = 0x1d;
		    if(iTower==15) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  //jet/sum
		  if(regionSums)
		    jetSumPins(nEvents, iCard, iTower,2);
		}
	      }
	      else if(testName=="regionSumPins3") {
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents>0) { //OKKKKKK
		  if(iCard==3){
		    if(iTower==5)  // 0110011001
		      ecal = 0x19;
		    if(iTower==10)
		      ecal = 0x19;
		    if(iTower==15) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==4){
		    if(iTower==26) // 1001100110
		      ecal = 0x26;
		    if(iTower==21)
		      ecal = 0x26;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==0){
		    if(iTower==26) // 0001101011
		      ecal = 0x2b;
		    if(iTower==21)
		      ecal = 0x2b;
		    if(iTower==16) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  if(iCard==5){
		    if(iTower==5)  // 1010010100
		      ecal = 0x14;
		    if(iTower==10)
		      ecal = 0x14;
		    if(iTower==15) {
		      ecal = fgEcalE;
		      fgbit = 1;
		    }
		  }
		  //jet/sum
		  if(regionSums)
		    jetSumPins(nEvents, iCard, iTower,3);
		}
	      }
	      else if(testName=="walkingOnes456") { //Walking ones cards 4&5&6
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents<32){
		  if(iTower==10||iTower==5)
		    {
		      if(iCard==4)
			ecal = (1<<tt&0x3F);  
		      if(iCard==5)
			ecal = (1<<(tt+1)&0x3F);	
		      if(iCard==6)
			ecal = (1<<tt&0x3F);
		    }
		}
		if(nEvents>=32){
		  if(iTower==10||iTower==5)
		    {
		      if(iCard==5)
			ecal = (1<<tt&0x3F);  
		      if(iCard==4)
			ecal = (1<<(tt+1)&0x3F);	
		      if(iCard==6)
			ecal = (1<<(tt+1)&0x3F);
		    }
		}
		//to make tower 5 non-iso
		if((iCard==4||iCard==5||iCard==6)&&(iTower==15)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }	
	      else if(testName=="walkingZeros456") { //Walking zeros cards 4&5&6
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents<32){
		  if(iTower==10||iTower==5)
		    {
		      if(iCard==4)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==5)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		if(nEvents>=32){
		  if(iTower==10||iTower==5)
		    {
		      if(iCard==5)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = (~(1<<(tt+1))&0x3F);  
		      if(iCard==4)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		//to make tower 5 non-iso
		if((iCard==4||iCard==5||iCard==6)&&(iTower==15)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}

		//put walking zeros into hcal
		if(regionSums)
		  walkZeroHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="walkingOnesZeros456") { //Walking ones--iso, zeros--niso cards 4&5&6
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		//ISO
		if(nEvents<32){
		  if(iTower==5)
		    {
		      if(iCard==4)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==5)
			ecal = ((1<<(tt+1))&0x3F);		      
		    }
		  //NISO
		  if(iTower==10)
		    {
		      if(iCard==4)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = (~(1<<tt)&0x3F);
		      if(iCard==5)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		//ISO
		if(nEvents>=32){
		  if(iTower==5)
		    {
		      if(iCard==4)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = ((1<<tt)&0x3F);  
		      if(iCard==5)
			ecal = ((1<<(tt+1))&0x3F);		      
		    }
		  //NISO
		  if(iTower==10)
		    {
		      if(iCard==4)
			ecal = (~(1<<tt)&0x3F);  
		      if(iCard==6)
			ecal = (~(1<<tt)&0x3F);
		      if(iCard==5)
			ecal = (~(1<<(tt+1))&0x3F);		      
		    }
		}
		//to make tower 10 non-iso
		if((iCard==5||iCard==4||iCard==6)&&(iTower==15)) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}

		//put walking ones into hcal
		if(regionSums)
		  walkHCAL(nEvents, iCard, iTower);

	      }
	      else if(testName=="testCrateNumber") {
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(iTower==6&iCard==0)
		  ecal = iCrate;
		if(iTower==6&&iCard==1)
		  ecal = iCrate+16;
		if(iTower==9&&iCard==1) {
		  ecal = fgEcalE;
		  fgbit = 1;
		}

		int iRegion = 0;
		if(iTower>=16) 
		  iRegion = 1;
		else 
		  iRegion = 0;
		//note: For HCAL link patterns need towers not next to each other in phi 
		//for endcap where phi divisions are two towers wide
		if(iTower==0||iTower==16)
		  Etot = ((iCrate<<4)|(iCard<<1)|iRegion);
		if(iTower == 11 || iTower == 27) {
		  if(Etot < 0xff)
		    hcal = (Etot)&0xff;
		  else 
		    hcal = 0xfe; 
		  Etot = Etot-hcal;
		}
		if(iTower == 14 || iTower == 30)
		  hcal = (Etot)&0xff;
	      }
	      else if(testName=="max") {
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;
		if(nEvents%10==0) {
		  if(iTower==9) {
		    ecal = 0x3f;
		  }
		}
		if(nEvents%5==0 && !(nEvents%10==0)) {
		  if(iTower==25) {
		    ecal = 0x3f;
		  }
		}
	      }
	      else if(testName=="test") { 
		fgbit =0;
		hcal = 0;
		mubit =0;
		ecal = 0;

		//hcal is combination of iTower, iCrate, iCard
		//hcal = iTower + iCrate + iCard;
		Etot = ((iCard<<5)|iTower);
		hcal = Etot;
	      }
	      else if(testName=="zeros") {
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
	      }
	      else if("count") {
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;

		if(iTower==5 || iTower==21 )
		  //ecal = nEvents;
		  hcal = nEvents;
	      } 
	      else { //no test specified
		cout << "NO TEST SPECIFIED!" <<endl;
		fgbit = 0;
		hcal  = 0;
		mubit = 0;
		ecal  = 0;
	      }
      
	      // Let's analyze this event
	    
	      //Printout now done in L1RCTSaveInput.cc
// 	      unsigned eAddr = ecal + fgbit; // ecal * 2 + fgbit; 
// 	      unsigned hAddr = hcal + mubit;// hcal * 2 + mubit; 

// 	      unsigned long lutOutput = rctLookupTables->lookup(ecal, hcal, fgbit, iCrate, iCard, iTower);
	     
// 	      ofs
// 		<< std::hex 
// 		<< nEvents << "\t"
// 		<< iCrate << "\t"
// 		<< iCard << "\t"
// 		<< iTower << "\t"
// 		<< eAddr << "\t"
// 		<< hAddr << "\t"
// 		<< lutOutput 
// 		<< std::dec 
// 		<< std::endl;
	      
	      //  Add the item into emulutor
	      int iEta = rctLookupTables->rctParameters()->calcIEta(iCrate,iCard,iTower);
	      int iPhi = rctLookupTables->rctParameters()->calcIPhi(iCrate,iCard,iTower);
	      // transform rct iphi coords into global coords

	      iPhi = ((72 + 18 - iPhi) % 72);
	      if (iPhi == 0) {iPhi = 72;}
	      int zSide = (iEta/abs(iEta));
	  
	      // args to detid are zside, type of tower, absieta, iphi
	      // absieta and iphi must be between 1 and 127 inclusive

	      EcalSubdetector subdet = (iEta <=17 ) ? EcalBarrel : EcalEndcap ;

	      EcalTriggerPrimitiveDigi 
		ecalDigi(EcalTrigTowerDetId(zSide, subdet, abs(iEta),
					      iPhi));

		ecalDigi.setSize(nEcalSamples);

	  

	      // last arg is 3-bit trigger tower flag, which we don't use
	      // we only use 8-bit encoded et and 1-bit fg
	      for(int nSample =0 ; nSample< nEcalSamples; ++nSample){
		if(nSample == 0)
		  ecalDigi.setSample(0, EcalTriggerPrimitiveSample(ecal, 
								   fgbit, 0));
		else 
		  ecalDigi.setSample(nSample, EcalTriggerPrimitiveSample(0, 
									 0, 0));
	      }

	    
	      ecalTPs->push_back(ecalDigi);
	      HcalTriggerPrimitiveDigi
		hcalDigi(HcalTrigTowerDetId(iEta, iPhi));
	  
	      hcalDigi.setSize(nHcalSamples);
	  
	      // last two arg's are slb and slb channel, which we don't need
	      hcalDigi.setSample(0, HcalTriggerPrimitiveSample(hcal,
							       mubit,
							       0, 0));


	      //Used for making HCAL pattern files (input to HCAL instead of RCT)
	      // HB valid DetIds: phi=1-72,eta=1-14,depth=1; phi=1-72,eta=15-16,depth=1-2
  
	      // HE valid DetIds: phi=1-72,eta=16-17,depth=1; phi=1-72,eta=18-20,depth=1-2; 
	      //                  phi=1-71(in steps of 2),eta=21-26,depth=1-2; phi=1-71(in steps of 2),eta=27-28,depth=1-3
	      //                  phi=1-71(in steps of 2),eta=29,depth=1-2

	      int depth=1;
	      int adc = hcal;
	      int capid = 0; //capacitor id (0-3), possibly anything?
	      bool dv=true; bool er=false; //data valid and error
	      int fiber = 0; int fiberchan = 0; //maybe need this???
	      int aiEta = abs(iEta);

// 	      if(aiEta<15) depth = 1;
// 	      if(aiEta==16 || aiEta==15) depth = 1; //or 2
	      HBHEDataFrame hbDataFrame(HcalDetId((HcalSubdetector) HcalBarrel, iEta, iPhi, depth));
	      hbDataFrame.setSize(nHBHESamples);
	      hbDataFrame.setSample(0,HcalQIESample(adc,capid,fiber,fiberchan,dv,er));

// 	      if(aiEta==16 || aiEta==17) depth = 1;
// 	      if(aiEta >=18 && aiEta <=26 || aiEta==29) depth = 1; // or 2
// 	      if(aiEta==27 || aiEta==28) depth = 1; // or 2 or 3

	      //MAYBE keep this???
	      if(aiEta >= 21) {
		if(iPhi%2==0) {
		  iPhi=iPhi-1;
		  depth = 2;
		}
	      }
	      
	      HBHEDataFrame heDataFrame(HcalDetId((HcalSubdetector) HcalEndcap, iEta, iPhi, depth));
	      heDataFrame.setSize(nHBHESamples);
	      heDataFrame.setSample(0,HcalQIESample(adc,capid,fiber,fiberchan,dv,er));

	      if(aiEta<=16)
		digiHBHE->push_back(hbDataFrame);
	      if(aiEta>16)
		digiHBHE->push_back(heDataFrame);
	}
      }
    }
      
      for (int i = 0; i < 18; i++) //Crate
	{	  
	  for (int j = 0; j < 8; j++) //HF "tower"
	    {	 
	      fgbit = 0;
	      hcal  = 0;
	      mubit = 0;
	      ecal  = 0;
	      hf=0;
	      if(regionSums){
		int tt = (nEvents%3)*2;
		if(testName=="testCrateNumber"&&nEvents>0) {
		  hf = ((i<<3)|(j))&0xFF;
		}
		else if(testName=="random"&&nEvents>0) {
		  if(rand()%100<randomPercent*2)
		    hf = rand()%0xFF;
		}
		else if(testName=="zeros"||testName=="flooding" || testName=="test") 
		  hf = 0x00;
		else if(testName=="max")
		  hf = 0x00;
		else {
		  if(nEvents==0)
		    hf=0;
		  else if((j<2||j==4||j==5)&&nEvents<32)
		    hf = (1<<tt)&0xFF;
		  else if((j<2||j==4||j==5)&&nEvents>=32)
		    hf = (1<<(tt+1))&0xFF;
		  else if((j>5||j==2||j==3)&&nEvents>=32)
		    hf = (1<<tt)&0x3F;
		  else if((j>5||j==2||j==3)&&nEvents<32)
		    hf = (1<<(tt+1))&0xFF;
		}
	        if(testName=="flooding12") {
		  if(nEvents%12==0) {
		    hf = rand()%0xFF;
		    fgbit = 1;
		  }
		  else {
		    hf = 0;
		    fgbit = 0;
		  }
		}
// 		unsigned long lutOutput = rctLookupTables->lookup(hf,i, 999, j);
// 		ofs
// 		  << std::hex 
// 		  << nEvents << "\t"
// 		  << i << "\t"
// 		  << "999" << "\t"
// 		  << j << "\t"
// 		  << "0" << "\t"
// 		  << hf << "\t"
// 		  << lutOutput // << "Wha happen'd"
// 		  << std::dec 
// 		  << std::endl;
	      }

	      // HF ieta: +- 29 through 32.  HF iphi: 1,5,9,13,etc.
	      int hfIEta = (j%4)+29;
	      if (i < 9)
		{
		  hfIEta = hfIEta*(-1);
		}
	      //int hfIPhi= i%9*2+j/4;
	      //int hfIPhi = (i%9*2 + j/4)*4 + 1;
	      int hfIPhi = i%9*8 + (j/4)*4 + 1;
	      hfIPhi = (72 + 18 - hfIPhi)%72;
              
	      HcalTriggerPrimitiveDigi
		hfDigi(HcalTrigTowerDetId(hfIEta, hfIPhi));
	      hfDigi.setSize(1);
	      hfDigi.setSample(0, HcalTriggerPrimitiveSample(hf,fgbit,0,0));
	      //	      hcalTPs->push_back(hfDigi);
	      int subdet = 4; 
	      int ieta = hfIEta; int iphi = hfIPhi; int depth = 1;
	      HFDataFrame hfDataFrame(HcalDetId((HcalSubdetector) subdet, ieta, iphi, depth));
	      int adc = hf;  
	      int capid = 0; //capacitor id (0-3), possibly anything?
	      bool dv=true; bool er=false; //data valid and error
	      int fiber = 0; int fiberchan = 0; //maybe need this???
	      hfDataFrame.setSize(nHFSamples);
	      hfDataFrame.setSample(0,HcalQIESample(adc,capid,fiber,fiberchan,dv,er));
	      digiHF->push_back(hfDataFrame);
	      

	    }
	}

    }
  event.put(ecalTPs);
  //  event.put(hcalTPs);

  event.put(digiHBHE);
  event.put(digiHF);

  nEvents++;
  }


/**********************************************
 *
 * Put known pattern into the first event
 *
 **********************************************/
void L1RCTPatternProducer::firstEvent(unsigned short iCrate,unsigned short iCard,unsigned short iTower) {
  fgbit =0;
  hcal = 0;
  mubit =0;
  ecal = 0;
  if(iTower==11) { //Easy to Find tower (i think)

    //if((iCrate%2)==0) {
    switch(iCard){
    case 0:
      ecal = 0x0d;
      break;
    case 1:
      ecal = 0x1e;
      break;
    case 2:
      ecal = 0x2a;
      break;  
    case 3:
      ecal = 0x3d;
      break;
    default:
      ecal = 0;
      break;
    }
    // } else {
    switch(iCard){
    case 0:
      ecal = 0x2e;
      break;
    case 1:
      ecal = 0x3f;
      break;
    case 2:
      ecal = 0x0b;
      break;
    case 3:
      ecal = 0x1e;
      break;
    default:
      ecal = 0;
      break;
      // }
    }
	    
  }
}
/*****************************************
 *
 * Simple walking ones through hcal
 *
 ****************************************/
void L1RCTPatternProducer::walkHCAL(int nEvents, unsigned short iCard, unsigned short iTower) {
  //No tau bit stuff yet
  //Putting energy in hcal for jets
  //Emulator **and not the hardware** currently set up so that if any tower is saturated (0xff) the entire region is saturated (0x3ff) -- so the following code avoids saturating any towers, but also allows for a saturated region.

  //For HCAL link patterns need towers not next to each other in phi 
  //for endcap where phi divisions are two towers wide
  hcal = 0;
  int regID = 0;
  if(nEvents<32)
    if(iTower>=16) 
      regID = 1;
    else 
      regID = 0;
  else
    if(iTower < 16) 
      regID = 1;
    else 
      regID = 0;
  
  int s = (nEvents*2+regID)%12;
  
  if(s<10&&nEvents<64&&(iTower==0||iTower==17)){
    Etot = (1<<s)&0x3ff;
  }
  else if(s==10&&nEvents<60&&(iTower==0||iTower==17)) {
    Etot = 0x3ff;
  }
  else if(s==11||nEvents>=60)
    Etot = 0x0;
  
  if(s>=8 && (iTower == 11 || iTower == 13 || iTower == 15 || iTower == 27 || iTower == 29 || iTower == 31)){
    if(Etot >=0xff)
      hcal = 0xf0;
    else 
      hcal = Etot;
    Etot = Etot - hcal;
  }
  if((iTower == 3 || (iTower == 19) )&& Etot!=0) {
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = rand()%(Etot+1);
    Etot=Etot-hcal;   
  }
  if((iTower == 12 || (iTower == 25 && iCard<6)) && Etot!=0) {
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = Etot;
    Etot=Etot-hcal;
  }
  if(s<8 && (iTower == 17 || iTower == 31) && iCard==6)
    hcal = Etot/2;
  if((iTower == 17 ) && iCard==6){
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = Etot;
    Etot = Etot-hcal;
  }
}

/****************************************
 *
 * Simple walking ones through hcal
 *
 ****************************************/
void L1RCTPatternProducer::walkZeroHCAL(int nEvents, unsigned short iCard, unsigned short iTower) {
  //No tau bit stuff yet
  //Putting energy in hcal for jets
  //Emulator **and not the hardware** currently set up so that if any tower is saturated (0xff) the entire region is saturated (0x3ff) -- so the following code avoids saturating any towers, but also allows for a saturated region.
  //cout<<"walkZeroHCAL " << nEvents << " card " << iCard << " tower " << iTower << endl;

  //For HCAL link patterns need towers not next to each other in phi 
  //for endcap where phi divisions are two towers wide
  hcal = 0;
  int regID = 0;
  if(nEvents<32) {
    if(iTower>=16) 
      regID = 1;
    else 
      regID = 0;
  }
  else 
    if(iTower < 16) 
      regID = 1;
    else 
      regID = 0;
  
  int s = (nEvents*2+regID)%12;
  if(s<12&&nEvents<64&&(iTower==0||iTower==17)){
    Etot = ~(1<<s)&0x3ff;
  }
  /*
    else if(s==10&&nEvents<60&&(iTower==0||iTower==16)) {
    Etot = 0x3ff;
    }
    else if(s==11||nEvents>=64)
    Etot = 0x0;
  */
  //if(iCard == 6)
  //cout << "Etot 0 " << Etot << endl;

  if((iTower == 11 || iTower == 13 || iTower == 15 || iTower == 27 || iTower == 29 || iTower == 31 )){
    if(Etot >=0xff)
      hcal = 0xf0;
    else 
      hcal = Etot;
    Etot = Etot - hcal;

  }
  //if(iCard == 6)
  //cout << "Etot 1 " << Etot << endl;
  if((iTower == 3 || (iTower == 19 )) && Etot!=0) {
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = rand()%(Etot+1);
    Etot=Etot-hcal;   
  }
  //if(iCard == 6)
  //cout << "Etot 2 " << Etot << endl;
  if((iTower == 12 || (iTower == 25 && iCard<6)) && Etot!=0) {
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = Etot;
    Etot=Etot-hcal;
  }
  //if(iCard == 6)
  //cout << "Etot 3 " << Etot << endl;
  if((iTower == 17) && iCard==6) {
    if(Etot>=0xff)
      hcal = 0xf0;
    else
      hcal = 0xf0;
    Etot=Etot-hcal;
  }
  //if(iCard == 6)
  //cout << "Etot 5 " << Etot << endl;
  //if(iCard == 6)
  //cout<<"walkZeroHCAL6 " << hcal << endl;
}

/***********************************************************
 *
 * Set up on/off to test bit switching on jet sum pins
 *
 ***********************************************************/
void L1RCTPatternProducer::jetSumPins(int nEvents, unsigned short iCard, unsigned short iTower, int num) {
  //No tau bit stuff yet
  //Putting energy in hcal for jets

  //For HCAL link patterns need towers not next to each other in phi 
  //for endcap where phi divisions are two towers wide

  hcal = 0;
  int oneZero = nEvents%2; 
  switch(num){
  case 0:
    if((iTower == 19 || iTower == 28 || iTower == 25) && iCard<6)
      hcal = (0xff)*(1-oneZero);
    if(iTower == 3 || iTower == 12 || iTower == 9)
      hcal = (0xff)*(oneZero);
    if((iTower == 17 || iTower == 31 || iTower == 21) && iCard==6) //16->17
      hcal = (0xff)*(1-oneZero);
    break;
  case 1:
    if((iTower == 19  || iTower == 28 ) && iCard<6)
      hcal = 0xE6>>oneZero;
    if(iTower == 3 || iTower == 12)
      hcal = 0xE6>>oneZero;
    if((iTower == 17 || iTower == 31) && iCard==6)//16->17
      hcal = 0xE6>>oneZero;
    if((iTower == 25 && iCard<6) || iTower == 9 || (iCard==6 && iTower == 21))
      hcal = 0xDE>>oneZero;
    break;
  case 2:
    if((iTower == 19 || iTower == 28 || iTower == 25) && iCard<6)
      hcal = (0xff)*(1-oneZero);
    if(iTower == 3 || iTower == 12 || iTower == 9)
      hcal = (0xff)*(oneZero);
    if((iTower == 17 || iTower == 31 || iTower == 21) && iCard==6)
      hcal = (0xff)*(1-oneZero);
    break;
  case 3:
    if((iTower == 19  || iTower == 28 ) && iCard<6)
      hcal = 0xE6>>oneZero;
    if(iTower == 3 || iTower == 12)
      hcal = 0xE6>>oneZero;
    if((iTower == 17 || iTower == 31) && iCard==6)
      hcal = 0xE6>>oneZero;
    if((iTower == 25 && iCard<6) || iTower == 9 || (iCard==6 && iTower == 21))
      hcal = 0xDE>>oneZero;
    break;
  }
}

// void L1RCTPatternProducer::writeEcalFiles(ofstream& of, int TCC) {
  
//   std::string ecalFileName;
//   ecalFileName = "data/" + testName + "/ecal_tcc" + TCC + ".txt";
  
//   of.open(ecalFileName.c_str());
//   if(!of)
//     {
//       std::cerr << "Could not create " << fileName << std::endl;
//       exit(1);
//     }
  


// }
