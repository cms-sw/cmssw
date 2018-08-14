#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLutWriter.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"

// default scales
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"

// debug scales
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

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
  lookupTable_(new L1RCTLookupTables),
  keyName_(iConfig.getParameter<std::string>("key")),
  useDebugTpgScales_(iConfig.getParameter<bool>("useDebugTpgScales"))
{
   //now do what ever initialization is needed
 
}


L1RCTLutWriter::~L1RCTLutWriter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
  
  if (lookupTable_ != nullptr) delete lookupTable_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1RCTLutWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
   // get all the configuration information from the event, set it
   // in the lookuptable
   edm::ESHandle<L1RCTParameters> rctParameters;
   iSetup.get<L1RCTParametersRcd>().get(rctParameters);
   rctParameters_ = rctParameters.product();
   edm::ESHandle<L1CaloEtScale> emScale;
   iSetup.get<L1EmEtScaleRcd>().get(emScale);
   const L1CaloEtScale* s = emScale.product();

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


   //Same for Noisy mask
   // make dummy channel mask -- we don't want to mask
   // any channels when writing LUTs, that comes afterwards
   L1RCTNoisyChannelMask* m2 = new L1RCTNoisyChannelMask;
   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       m2->ecalMask[i][j][k] = false;
	       m2->hcalMask[i][j][k] = false;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       m2->hfMask[i][j][k] = false;
	     }
	 }
     }
   m2->ecalThreshold = 0.0;
   m2->hcalThreshold = 0.0;
   m2->hfThreshold = 0.0;


   
  // use these dummies to get the delete right when using old-style
  // scales to create set of L1CaloXcalScales
  L1CaloEcalScale* dummyE(nullptr);
  L1CaloHcalScale* dummyH(nullptr);

  if (useDebugTpgScales_) // generate new-style scales from tpg scales
    {

      std::cout << "Using old-style TPG scales!" << std::endl;

      // old version of hcal energy scale to convert input
      edm::ESHandle<CaloTPGTranscoder> transcoder;
      iSetup.get<CaloTPGRecord>().get(transcoder);
      const CaloTPGTranscoder* h_tpg = transcoder.product();

      // old version of ecal energy scale to convert input
      EcalTPGScale* e_tpg = new EcalTPGScale();
      e_tpg->setEventSetup(iSetup);

      L1CaloEcalScale* ecalScale = new L1CaloEcalScale();
      L1CaloHcalScale* hcalScale = new L1CaloHcalScale();

      // generate L1CaloXcalScales from old-style scales (thanks, werner!)

      // ECAL
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	{
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
	}
      
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	{
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
	}

      // HCAL
      for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta )
	{
	  for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank )
	    {
	      double etGeV = h_tpg->hcaletValue( ieta, irank ) ;

	      hcalScale->setBin( irank, ieta, 1, etGeV ) ;
	      hcalScale->setBin( irank, ieta, -1, etGeV ) ;
	    }
	}

      // set the input scales
      lookupTable_->setEcalScale(ecalScale);
      lookupTable_->setHcalScale(hcalScale);

      dummyE = ecalScale;
      dummyH = hcalScale;

      delete e_tpg;

    }
  else
    {

      // get energy scale to convert input from ECAL
      edm::ESHandle<L1CaloEcalScale> ecalScale;
      iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
      const L1CaloEcalScale* e = ecalScale.product();
      
      // get energy scale to convert input from HCAL
      edm::ESHandle<L1CaloHcalScale> hcalScale;
      iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
      const L1CaloHcalScale* h = hcalScale.product();

      // set scales
      lookupTable_->setEcalScale(e);
      lookupTable_->setHcalScale(h);

    }

   lookupTable_->setRCTParameters(rctParameters_);
   lookupTable_->setChannelMask(m);
   lookupTable_->setNoisyChannelMask(m2);
   //lookupTable_->setHcalScale(h);
   //lookupTable_->setEcalScale(e);
   lookupTable_->setL1CaloEtScale(s);

   for (unsigned short nCard = 0; nCard <= 6; nCard = nCard + 2)
     {
       writeRcLutFile(nCard);
       writeEicLutFile(nCard);
     }
   writeJscLutFile();

   unsigned int eicThreshold = rctParameters_->eicIsolationThreshold();
   unsigned int jscThresholdBarrel = rctParameters_->jscQuietThresholdBarrel();
   unsigned int jscThresholdEndcap = rctParameters_->jscQuietThresholdEndcap();
   writeThresholdsFile(eicThreshold, jscThresholdBarrel, jscThresholdEndcap);

  if (dummyE != nullptr) delete dummyE;
  if (dummyH != nullptr) delete dummyH;

}




// ------------ method called once each job just after ending the event loop  ------------
void 
L1RCTLutWriter::endJob() {
}


// ------------ method to write one receiver card lut file
void
L1RCTLutWriter::writeRcLutFile(unsigned short card)
{

  // don't mess yet with name
  char filename[256];
  char command[264];
  if (card != 6)
    {
      int card2 = card + 1;
      sprintf(filename,"RC%i%i-%s.dat",card,card2,keyName_.c_str() );
      //sprintf(filename, "RC%i%i.dat", card, card2);
    }
  else
    {
      sprintf(filename,"RC6-%s.dat",keyName_.c_str() );
      //sprintf(filename, "RC6.dat");
    }
  // open file for writing, delete any existing content
  lutFile_.open(filename, std::ios::trunc);
  lutFile_ << "Emulator-parameter generated lut file, card " 
	   << card << " key " << keyName_ << "   ";

  // close to append timestamp info
  lutFile_.close();
  sprintf(command, "date >> %s", filename);
  system(command);

  // reopen file for writing values
  lutFile_.open(filename, std::ios::app);

  unsigned long data = 0;

  // write all memory addresses in increasing order
  // address = (1<<22)+(nLUT<<19)+(eG?<18)+(hcalEt<<10)+(ecalfg<<9)+(ecalEt<<1)

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
	  if (nLUT < 4)
	    {
	      iAbsEta = (card/2)*8 + nLUT + 1;
	    }
	  else
	    {
	      iAbsEta = (card/2)*8 + (3 - (nLUT%4) ) + 1;	      
	    }
	  //std::cout << "LUT is " << nLUT << " iAbsEta is " << iAbsEta << std::endl;
	}


      // All RCT stuff uniform in phi, symmetric wrt eta = 0

      // below line always gives crate in +eta; makes no difference to us
      unsigned short crate = rctParameters_->calcCrate(1, iAbsEta); 
      unsigned short tower = rctParameters_->calcTower(1, iAbsEta);

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
  char filename[256];
  char command[264];
  if (card != 6)
    {
      int card2 = card + 1;
      sprintf(filename,"EIC%i%i-%s.dat", card, card2, keyName_.c_str() );
    }
  else
    {
      sprintf(filename,"EIC6-%s.dat", keyName_.c_str() );
    }
  // open file for writing, delete any existing content
  lutFile_.open(filename, std::ios::trunc);
  lutFile_ << "Emulator-parameter generated EIC lut file, card " 
	   << card << " key " << keyName_ << "   ";
  // close to append timestamp info
  lutFile_.close();
  sprintf(command, "date >> %s", filename);
  system(command);

  // reopen file for writing values
  lutFile_.open(filename, std::ios::app);

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
void 
L1RCTLutWriter::writeJscLutFile()
{
  char filename[256];
  char command[264];
  sprintf(filename, "JSC-%s.dat", keyName_.c_str() );

  // open file; if it already existed, delete existing content
  lutFile_.open(filename, std::ios::trunc);
  lutFile_ << "Emulator parameter-generated lut file, key " << keyName_ 
	   << "   ";
  // close to append time-stamp
  lutFile_.close();
  sprintf(command, "date >> %s", filename);
  system(command);
  // reopen file for writing
  lutFile_.open(filename, std::ios::app);

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
	      lutFile_ << std::hex << data << std::dec << std::endl;
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

//-----------Write text file containing the 1 JSC and 2 EIC thresholds
void
L1RCTLutWriter::writeThresholdsFile(unsigned int eicThreshold, 
				    unsigned int jscThresholdBarrel,
				    unsigned int jscThresholdEndcap)
{
  //
  std::ofstream thresholdsFile;
  char filename[256];
  sprintf(filename, "Thresholds-%s.dat", keyName_.c_str() );
  thresholdsFile.open(filename, std::ios::trunc);

  thresholdsFile << "key is " << keyName_ << std::endl << std::endl;
  thresholdsFile << "eicIsolationThreshold " << eicThreshold << std::endl;
  thresholdsFile << "jscQuietThresholdBarrel " << jscThresholdBarrel << std::endl;
  thresholdsFile << "jscQuietThresholdEndcap " << jscThresholdEndcap << std::endl;

  thresholdsFile.close();
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1RCTLutWriter); // done in SealModule.cc
