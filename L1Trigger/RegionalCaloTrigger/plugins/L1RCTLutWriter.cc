#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLutWriter.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1RCTLutWriter::L1RCTLutWriter(const edm::ParameterSet& iConfig) :
  lookupTable_(new L1RCTLookupTables)
{
   //now do what ever initialization is needed
 
}


L1RCTLutWriter::~L1RCTLutWriter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
  if (lookupTable_ != 0) delete lookupTable_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1RCTLutWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //   using namespace edm;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
   
   // get all the configuration information from the event, set it
   // in the lookuptable
   edm::ESHandle<L1RCTParameters> rctParameters;
   iSetup.get<L1RCTParametersRcd>().get(rctParameters);
   //const L1RCTParameters* r = rctParameters.product();
   rctParameters_ = rctParameters.product();
   // don't get channel mask, make dummy below
   //edm::ESHandle<L1RCTChannelMask> channelMask;
   //iSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
   //const L1RCTChannelMask* m = channelMask.product();
   edm::ESHandle<CaloTPGTranscoder> transcoder;
   iSetup.get<CaloTPGRecord>().get(transcoder);
   const CaloTPGTranscoder* t = transcoder.product();
   edm::ESHandle<L1CaloEtScale> emScale;
   iSetup.get<L1EmEtScaleRcd>().get(emScale);
   const L1CaloEtScale* s = emScale.product();

   EcalTPGScale* e = new EcalTPGScale();
   e->setEventSetup(iSetup);

   // make dummy channel mask -- we don't want to mask
   // any channels when writing LUTs, that comes afterwards
   L1RCTChannelMask* m = new L1RCTChannelMask;
   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       m->ecalMask[i][j][k] = false;
	       m->hcalMask[i][j][k] = false;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       m->hfMask[i][j][k] = false;
	     }
	 }
     }
   
   lookupTable_->setRCTParameters(rctParameters_);
   lookupTable_->setChannelMask(m);
   lookupTable_->setTranscoder(t);
   lookupTable_->setL1CaloEtScale(s);
   lookupTable_->setEcalTPGScale(e);


   for (unsigned short nCard = 0; nCard <= 6; nCard = nCard + 2)
     {
       writeRcLutFile(nCard);
       writeEicLutFile(nCard);
     }
   writeJscLutFile();

   delete e;

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1RCTLutWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1RCTLutWriter::endJob() {
}


// ------------ method to write one receiver card lut file
void
L1RCTLutWriter::writeRcLutFile(unsigned short card)
{

  // don't mess yet with timestamp etc.
  char filename[64];
  if (card != 6)
    {
      int card2 = card + 1;
      sprintf(filename,"rctReceiverCard%i%i.dat",card,card2);
    }
  else
    {
      sprintf(filename,"rctReceiverCard6.dat");
    }
  // open file for writing, delete any existing content
  lutFile_.open(filename, ios::trunc);
  lutFile_ << "Emulator-parameter generated lut file, card " 
	   << card << std::endl;

  unsigned long data = 0;

  // write all memory addresses in increasing order
  // address = (1<<22)+(nLUT<<19)+(hcalEt<<10)+(ecalfg<<9)+(ecalEt<<1)

  // loop through the physical LUTs on the card, 0-7
  for (unsigned short nLUT = 0; nLUT < 8; nLUT++)
    {
      // determine ieta, iphi, etc. everything
      unsigned short iAbsEta = 0;
      if (card != 6)
	{
	  iAbsEta = (card/2)*8 + nLUT + 1;
	}
      else
	{
	  iAbsEta = (card/2)*8 + (nLUT%4) + 1;
	}

      unsigned short iPhi = 1;
      float dummy = 0;
      short sign = 1;
      do
	{
	  dummy = lookupTable_->convertEcal(127, iAbsEta, iPhi, sign);
	  iPhi++;
	}
      while (dummy == 0 && iPhi < 73);
      if (dummy == 0)
	{
	  sign = -1;
	  iPhi = 1;
	  do
	    {
	      dummy = lookupTable_->convertEcal(127, iAbsEta, iPhi, sign);
	      iPhi++;
	    }
	  while (dummy == 0 && iPhi < 73);
	}
      if (iPhi == 73)
	{
	  iPhi = 1;
	}

      short iEta = sign * ( (short) iAbsEta);

      unsigned short crate = rctParameters_->calcCrate(iPhi, iEta);
      unsigned short tower = rctParameters_->calcTower(iPhi, iAbsEta);

      // first do region sums half of LUTs, bit 18 of address is 0
      // loop through 8 bits of hcal energy, 2^8 is 256
      for (unsigned int hcalEt = 0; hcalEt < 256; hcalEt++)
	{
	  // loop through 1 bit of ecal fine grain
	  for (unsigned short ecalfg = 0; ecalfg < 2; ecalfg++)
	    {
	      // loop through 8 bits of ecal energy
	      for (unsigned int ecalEt = 0; ecalEt < 256; ecalEt++)
		{
		  // assign 10-bit (9et,1mip) sums data here!
		  unsigned long output = lookupTable_->
		    lookup(ecalEt, hcalEt, ecalfg, crate, card, tower);
		  unsigned short etIn9Bits = (output>>8)&511;
		  unsigned short tauActivityBit = (output>>17)&1;
		  data = (tauActivityBit<<9)+etIn9Bits;
		  lutFile_ << std::hex << data << std::dec << std::endl;
		}
	    }
	}
      // second do egamma half of LUTs, bit 18 of address is 1
      // loop through 8 bits of hcal energy
      for (unsigned int hcalEt = 0; hcalEt < 256; hcalEt++)
        {
          // loop through 1 bit of ecal fine grain
          for (unsigned short ecalfg = 0; ecalfg < 2; ecalfg++)
            {
              // loop through 8 bits of ecal energy
              for (unsigned int ecalEt = 0; ecalEt < 256; ecalEt++)
                {
                  // assign 8-bit (7et,1veto) egamma data here!
		  unsigned long output = lookupTable_->
		    lookup(ecalEt, hcalEt, ecalfg, crate, card, tower);
		  unsigned short etIn7Bits = output&127;
		  unsigned short heFgVetoBit = (output>>7)&1;
		  data = (heFgVetoBit<<7)+etIn7Bits;
		  lutFile_ << std::hex << data << std::dec << std::endl;
		  /*
		  if (((ecalEt % 100) == 0) && ((hcalEt % 45) == 0)) 
		    {
		      std::cout << "Writer: ecalEt=" << ecalEt << " hcalEt=" 
				<< hcalEt
				<< " ecalfg=" << ecalfg << " etIn7Bits="
				<< etIn7Bits << " heFgVetoBit="
				<< heFgVetoBit << std::endl;
		    }
		  */
                }
            }
        }
    }

  lutFile_.close();
  return;

}

// ------------ method to write one electron isolation card lut file
void
L1RCTLutWriter::writeEicLutFile(unsigned short card)
{
  // try timestamp
  char filename[64];
  char command[64];
  if (card != 6)
    {
      int card2 = card + 1;
      sprintf(filename,"rctElectronIsolationCard%i%i.dat",card,card2);
    }
  else
    {
      sprintf(filename,"rctElectronIsolationCard6.dat");
    }
  // open file for writing, delete any existing content
  lutFile_.open(filename, ios::trunc);
  lutFile_ << "Emulator-parameter generated EIC lut file, card " 
	   << card << std::endl;
  // close to append timestamp info
  lutFile_.close();
  sprintf(command, "date >> %s", filename);
  system(command);

  // reopen file for writing values
  lutFile_.open(filename, ios::app);

  unsigned long data = 0;

  // write all memory addresses in increasing order
  // address = (1<<22) + (etIn7Bits<<1)

  // 2^7 = 0x7f = 128
  for (int etIn7Bits = 0; etIn7Bits < 128; etIn7Bits++)
    {
      data = lookupTable_->emRank(etIn7Bits);
      if (data > 0x3f)
	{
	  data = 0x3f;
	}
      lutFile_ << std::hex << data << std::dec << std::endl;
    }
  lutFile_.close();
  return;
}

// ------------ method to write one jet summary card lut file
void L1RCTLutWriter::writeJscLutFile()
{
  char filename[64];
  char command[64];
  sprintf(filename, "rctJetSummaryCard.dat");

  // open file; if it already existed, delete existing content
  lutFile_.open(filename, ios::trunc);
  lutFile_ << "Emulator parameter-generated lut file    ";
  // close to append time-stamp
  lutFile_.close();
  sprintf(command, "date >> %s", filename);
  system(command);
  // reopen file for writing
  lutFile_.open(filename, ios::app);

  unsigned long data = 0;
  unsigned long data0 = 0;
  unsigned long data1 = 0;

  // write all memory addresses in increasing order
  // address = (1<<22) + (lutbits<<17) + (phi1et<<9) + (phi0et<<1);

  // ecl and U93/U225 lut id bits, identify eta segment of hf
  for (int lutbits = 0; lutbits < 4; lutbits++)
    {
      // 8-bit phi_1 et for each eta partition
      for (unsigned int phi1et = 0; phi1et < 256; phi1et++)
	{
	  // 8-bit phi_0 et for each eta
	  for (unsigned int phi0et = 0; phi0et < 256; phi0et++)
	    {
	      // lookup takes "(hf_et, crate, card, tower)"
	      // "card" convention for hf is 999, tower is 0-7
	      // but equivalent to 0-3 == lutbits
	      // crate doesn't matter, take 0
	      // only |ieta| matters
	      data0 = lookupTable_->lookup(phi0et, 0, 999, lutbits);
	      if (data0 > 0xff)
		{
		  data0 = 0xff;   // 8-bit output energy for each phi region
		}
	      data1 = lookupTable_->lookup(phi1et, 0, 999, lutbits);
	      if (data1 > 0xff)
		{
		  data1 = 0xff;   // 8-bit output energy for each phi region
		}
	      data = (data1<<8) + (data0);
	      lutFile_ << hex << data << dec << endl;
	      /*
	      if (phi0et < 10 && phi1et < 10)
		{
		  std::cout << "Writer: jsc. lutbits=" << lutbits 
			    << " phi0et=" << phi0et << " data0=" << data0
			    << " phi1et=" << phi1et << " data1=" << data1
			    << std::endl;
		}
	      */
	    }
	}
    }
  
  lutFile_.close();
  return;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1RCTLutWriter); // done in SealModule.cc
