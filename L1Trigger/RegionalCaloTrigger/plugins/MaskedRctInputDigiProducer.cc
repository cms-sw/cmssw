#include "L1Trigger/RegionalCaloTrigger/interface/MaskedRctInputDigiProducer.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
using std::ifstream;
using std::vector;
using std::endl;

//
// constructors and destructor
//

MaskedRctInputDigiProducer::MaskedRctInputDigiProducer(const edm::ParameterSet& iConfig):
  useEcal_(iConfig.getParameter<bool>("useEcal")),
  useHcal_(iConfig.getParameter<bool>("useHcal")),
  ecalDigisLabel_(iConfig.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel_(iConfig.getParameter<edm::InputTag>("hcalDigisLabel")),
  maskFile_(iConfig.getParameter<edm::FileInPath>("maskFile"))
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/

  produces<EcalTrigPrimDigiCollection>();
  produces<HcalTrigPrimDigiCollection>();

   //now do what ever other initialization is needed
  
}


MaskedRctInputDigiProducer::~MaskedRctInputDigiProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MaskedRctInputDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

   edm::Handle<EcalTrigPrimDigiCollection> ecal;
   edm::Handle<HcalTrigPrimDigiCollection> hcal;

   if (useEcal_) { iEvent.getByLabel(ecalDigisLabel_, ecal); }
   if (useHcal_) { iEvent.getByLabel(hcalDigisLabel_, hcal); }

   EcalTrigPrimDigiCollection ecalColl;
   HcalTrigPrimDigiCollection hcalColl;
   if (ecal.isValid()) { ecalColl = *ecal; }
   if (hcal.isValid()) { hcalColl = *hcal; }

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
 
   ifstream maskFileStream(maskFile_.fullPath().c_str());
   if (!maskFileStream.is_open())
     {
       throw cms::Exception("FileInPathError")
	 << "RCT mask file not opened" << endl;;
     }

   // read and process file (transform mask to new position in phi to match digis)
   //char junk[256];
   std::string junk;
   //char junk[32];
   do
     {
       //maskFileStream.getline(junk, 256);
       maskFileStream >> junk;
     }
   while (junk.compare("ECAL:") != 0);

   //   vector<vector<vector<unsigned short> > > temp(2,vector<vector<unsigned short> >(72,vector<unsigned short>(28,1)));
   vector<vector<vector<unsigned short> > > ecalMask(2,vector<vector<unsigned short> >(72,vector<unsigned short>(28,1)));
   vector<vector<vector<unsigned short> > > hcalMask(2,vector<vector<unsigned short> >(72,vector<unsigned short>(28,1)));
   vector<vector<vector<unsigned short> > > hfMask(2,vector<vector<unsigned short> >(18,vector<unsigned short>(8,1)));

   // read ECAL mask first
   // loop through rct iphi
   for (int i = 1; i <= 72; i++)
     {
       int phi_index = (72 + 18 - i) % 72;  // transfrom from RCT coords
       if (phi_index == 0) {phi_index = 72;}
       //std::cout << "ecal phi index is " << phi_index << endl;
       for (int j = 28; j >= 1; j--)
	 {
	   maskFileStream >> junk;
	   if (junk.compare("-") == 0) 
	     {}
	   else if ((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       ecalMask.at(0).at(phi_index-1).at(j-1) = 0;
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }
	 }
       for (int j = 1; j <= 28; j++)
	 {
	   maskFileStream >> junk;
	   if(junk.compare("-") == 0)
	     {}
	   else if((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       ecalMask.at(1).at(phi_index-1).at(j-1) = 0;
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }	   
	 }
     }
   //std::cout << "done reading ECAL" << endl;

   maskFileStream >> junk;
   if (junk.compare("HCAL:") != 0)
     {
       throw cms::Exception("FileInPathError")
	 << "RCT mask producer: error reading ECAL mask" << endl;;
     }

   //std::cout << "Starting HCAL read" << endl;

   // now read HCAL mask
   // loop through rct iphi
   for (int i = 1; i <= 72; i++)
     {
       int phi_index = (72 + 18 - i) % 72;  // transfrom from RCT coords
       if (phi_index == 0) {phi_index = 72;}
       //std::cout << "hcal phi index is " << phi_index << endl;
       for (int j = 28; j >= 1; j--)
	 {
	   maskFileStream >> junk;
	   if (junk.compare("-") == 0) 
	     {}
	   else if ((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       hcalMask.at(0).at(phi_index-1).at(j-1) = 0;
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }
	 }
       for (int j = 1; j <= 28; j++)
	 {
	   maskFileStream >> junk;
	   if(junk.compare("-") == 0)
	     {}
	   else if((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       hcalMask.at(1).at(phi_index-1).at(j-1) = 0;
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }	   
	 }
     }

   maskFileStream >> junk;
   if (junk.compare("HF:") != 0)
     {
       throw cms::Exception("FileInPathError")
	 << "RCT mask producer: error reading HCAL mask" << endl;;
     }       

   // loop through the hf phi columns in file
   for(int i = 0; i < 18; i++)
     {
       //std::cout << "iphi for hf file read is " << i << endl;
       for (int j = 4; j >= 1; j--)
	 {
	   if (maskFileStream >> junk) {}
	   else { std::cerr << "RCT mask producer: error reading HF mask" << endl; }
	   if (junk.compare("-") == 0) 
	     {}
	   else if ((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       hfMask.at(0).at(i).at(j-1) = 0;  // just save iphi as 0-17, transform later
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }
	 }
       for (int j = 1; j <= 4; j++)
	 {
	   if (maskFileStream >> junk) {}
	   else { std::cerr << "RCT mask producer: error reading HF mask" << endl; }
	   if (junk.compare("-") == 0) 
	     {}
	   else if ((junk.compare("X") == 0) || (junk.compare("x") == 0))
	     {
	       hfMask.at(1).at(i).at(j-1) = 0;
	     }
	   else
	     {
	       std::cerr << "RCT mask producer: error -- unrecognized character" << endl;
	     }
	 }	   
     }

   //std::cout << "HF read done" << endl;

   maskFileStream.close();

   //std::cout << "file closed" << endl;

   // apply mask

   std::auto_ptr<EcalTrigPrimDigiCollection>
     maskedEcalTPs(new EcalTrigPrimDigiCollection());
   std::auto_ptr<HcalTrigPrimDigiCollection>
     maskedHcalTPs(new HcalTrigPrimDigiCollection());
   maskedEcalTPs->reserve(56*72);
   maskedHcalTPs->reserve(56*72+18*8);
   const int nEcalSamples = 1;
   const int nHcalSamples = 1;

   for (unsigned int i = 0; i < ecalColl.size(); i++)
     {
       short ieta = (short) ecalColl[i].id().ieta();
       unsigned short absIeta = (unsigned short) abs(ieta);
       int sign = ieta / absIeta;
       short iphi = (unsigned short) ecalColl[i].id().iphi();
       //if (i < 20) {std::cout << "ieta is " << ieta << ", absIeta is " << absIeta
       //		      << ", iphi is " << iphi << endl;}

       int energy;
       bool fineGrain;

       if (sign < 0)
	 {
	   //std::cout << "eta-: mask is " << ecalMask.at(0).at(iphi-1).at(absIeta-1) << endl;
	   energy = ecalMask.at(0).at(iphi-1).at(absIeta-1) * ecalColl[i].compressedEt();
	   fineGrain = ecalMask.at(0).at(iphi-1).at(absIeta-1) * ecalColl[i].fineGrain();
	 }
       else if (sign > 0)
	 {
	   //std::cout << "eta+: mask is " << ecalMask.at(1).at(iphi-1).at(absIeta-1) << endl;	   
	   energy = ecalMask.at(1).at(iphi-1).at(absIeta-1) * ecalColl[i].compressedEt();
	   fineGrain = ecalMask.at(1).at(iphi-1).at(absIeta-1) * ecalColl[i].fineGrain();
	 }

       EcalTriggerPrimitiveDigi
	 ecalDigi(EcalTrigTowerDetId(sign, EcalTriggerTower, absIeta,iphi));
       ecalDigi.setSize(nEcalSamples);
       ecalDigi.setSample(0, EcalTriggerPrimitiveSample(energy,
							fineGrain,
							0));
       maskedEcalTPs->push_back(ecalDigi);
     }
   //std::cout << "End of ecal digi masking" << endl;

   for (unsigned int i = 0; i < hcalColl.size(); i++)
     {
       short ieta = (short) hcalColl[i].id().ieta();
       unsigned short absIeta = (unsigned short) abs(ieta);
       int sign = ieta / absIeta;
       short iphi = (unsigned short) hcalColl[i].id().iphi();

       int energy;
       bool fineGrain;

       if (absIeta < 29)
	 {
	   if (sign < 0)
	     {
	       energy = hcalMask.at(0).at(iphi-1).at(absIeta-1) * hcalColl[i].SOI_compressedEt();
	       fineGrain = hcalMask.at(0).at(iphi-1).at(absIeta-1) * hcalColl[i].SOI_fineGrain();
	     }
	   else if (sign > 0)
	     {
	       energy = hcalMask.at(1).at(iphi-1).at(absIeta-1) * hcalColl[i].SOI_compressedEt();
	       fineGrain = hcalMask.at(1).at(iphi-1).at(absIeta-1) * hcalColl[i].SOI_fineGrain();
	     }
	 }
       else if ((absIeta >= 29) && (absIeta <= 32))
	 {
	   //std::cout << "hf iphi: " << iphi << endl;
	   short hf_phi_index = iphi/4;
	   //iphi = iphi/4;  // change from 1,5,9, etc to access vector positions
	   //std::cout << "hf phi index: " << hf_phi_index << endl;
	   if (sign < 0)
	     {
	       //std::cout << "ieta is " << ieta << ", absIeta is " << absIeta << ", iphi is " << iphi << endl;
	       //std::cout << "eta-: mask is " << hfMask.at(0).at(hf_phi_index).at(absIeta-29) << endl; // hf ieta 0-3
	       energy = hfMask.at(0).at(hf_phi_index).at(absIeta-29) * hcalColl[i].SOI_compressedEt();  // for hf, iphi starts at 0
	       fineGrain = hfMask.at(0).at(hf_phi_index).at(absIeta-29) * hcalColl[i].SOI_fineGrain();
	     }
	   else if (sign > 0)
	     {
	       //std::cout << "ieta is " << ieta << ", absIeta is " << absIeta << ", iphi is " << iphi << endl;
	       //std::cout << "eta+: mask is " << hfMask.at(1).at(hf_phi_index).at(absIeta-29) << endl;
	       energy = hfMask.at(1).at(hf_phi_index).at(absIeta-29) * hcalColl[i].SOI_compressedEt();
	       fineGrain = hfMask.at(1).at(hf_phi_index).at(absIeta-29) * hcalColl[i].SOI_fineGrain();
	     }
	   //iphi = iphi*4 + 1; // change back to original
	   //std::cout << "New hf iphi = " << iphi << endl;
	 }

       HcalTriggerPrimitiveDigi
	 hcalDigi(HcalTrigTowerDetId(ieta,iphi));
       hcalDigi.setSize(nHcalSamples);
       hcalDigi.setSample(0, HcalTriggerPrimitiveSample(energy,
							fineGrain,
							0, 0));
       maskedHcalTPs->push_back(hcalDigi);
     }
   //std::cout << "End of hcal digi masking" << endl;

   // put new data into event

   iEvent.put(maskedEcalTPs);
   iEvent.put(maskedHcalTPs);

}

// ------------ method called once each job just before starting event loop  ------------
void 
MaskedRctInputDigiProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MaskedRctInputDigiProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MaskedRctInputDigiProducer);
