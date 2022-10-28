#include "RctInputTextToDigi.h"

//
// constructors and destructor
//

RctInputTextToDigi::RctInputTextToDigi(const edm::ParameterSet &iConfig)
    : inputFile_(iConfig.getParameter<edm::FileInPath>("inputFile")),
      inputStream_(inputFile_.fullPath().c_str()),
      lookupTables_(new L1RCTLookupTables),
      paramsToken_(esConsumes()),
      nEvent_(0),
      oldVersion_(false) {
  // register your products
  /* Examples
     produces<ExampleData2>();

     //if do put with a label
     produces<ExampleData2>("label");
  */

  produces<EcalTrigPrimDigiCollection>();
  produces<HcalTrigPrimDigiCollection>();

  // now do what ever other initialization is needed

  if ((!inputStream_.is_open()) || (!inputStream_)) {
    // not good!!
    std::cerr << "Input file didn't open!!" << std::endl;
  }
  // if (inputStream_.eof()) {std::cout << "Real zeroth eof! " << std::endl;}
}

RctInputTextToDigi::~RctInputTextToDigi() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  inputStream_.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void RctInputTextToDigi::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // std::cout << std::endl << std::endl << "Event number " << nEvent_ <<
  // std::endl;

  // This next section taken directly from
  // L1Trigger/RegionalCaloTrigger/plugins/L1RCTProducer.cc rev. 1.6
  // Refresh configuration information every event
  // Hopefully doesn't take too much time
  const L1RCTParameters *r = &iSetup.getData(paramsToken_);
  lookupTables_->setRCTParameters(r);

  std::unique_ptr<EcalTrigPrimDigiCollection> ecalTPs(new EcalTrigPrimDigiCollection());
  std::unique_ptr<HcalTrigPrimDigiCollection> hcalTPs(new HcalTrigPrimDigiCollection());
  ecalTPs->reserve(56 * 72);
  hcalTPs->reserve(56 * 72 + 18 * 8);  // includes HF
  const int nEcalSamples = 1;          // we only use 1 sample for each
  const int nHcalSamples = 1;

  int fileEventNumber;

  // check to see if need to skip file header and do so before
  // looping through entire event

  std::string junk;
  // bool old_version = false;
  if (nEvent_ == 0) {
    // std::string junk;
    unsigned short junk_counter = 0;
    // bool old_version = false;
    do {
      if (inputStream_ >> junk) { /*std::cout << "Good: ";*/
      }
      // std::cout << "header junk was input: \"" << junk << "\"."
      //		     << std::endl;
      // for oldest version, which is same as newest version
      //	  if((junk_counter == 11) && (junk.compare("0-32") == 0))
      //	    {
      //	      oldVersion_ = true;
      //	    }
      if ((junk_counter == 11) && (junk == "1-32")) {
        oldVersion_ = true;
      }
      junk_counter++;
    } while (junk != "LUTOut");
    std::cout << "Skipped file header" << std::endl;
    if (oldVersion_) {
      std::cout << "oldVersion_ TRUE (tower 1-32)" << std::endl;
    } else {
      std::cout << "oldVersion_ FALSE (tower 0-31)" << std::endl;
    }
  }

  // can't actually read in phi and eta, file has crate card tower instead
  // do a while loop for event number instead??  dunno
  for (int i = 0; i < 72; i++) {
    // negative eta, iEta -28 to -1
    for (int j = 0; j < 56; j++) {
      // calc ieta, iphi coords of tower
      // ieta -28 to -1 or 1 to 28, iphi 1 to 72
      // methods in CondFormats/L1TObjects/src/L1RCTParameters.cc

      unsigned short crate;
      unsigned short card;
      unsigned short tower;
      unsigned eAddr;
      unsigned hAddr;

      inputStream_ >> std::hex >> fileEventNumber >> crate >> card >> tower >> eAddr >> hAddr >> junk >> std::dec;

      if (oldVersion_) {
        tower = tower - 1;
      }
      int encodedEtEcal = (int)(eAddr >> 1);
      bool fineGrainEcal = (bool)(eAddr & 1);
      int encodedEtHcal = (int)(hAddr >> 1);
      bool fineGrainHcal = (bool)(hAddr & 1);  // mip bit

      // std::cout << "Eventnumber " << fileEventNumber << "\tCrate "
      //    << crate << "\tCard " << card << "\tTower "
      //    << tower << " \teAddr " << eAddr <<"\thAddr "
      //    << hAddr << "\tjunk " << junk << std::endl;

      int iEta = lookupTables_->rctParameters()->calcIEta(crate, card, tower);
      int iPhi = lookupTables_->rctParameters()->calcIPhi(crate, card, tower);
      // transform rct iphi coords into global coords
      iPhi = ((72 + 18 - iPhi) % 72);
      if (iPhi == 0) {
        iPhi = 72;
      }
      unsigned absIeta = abs(iEta);
      int zSide = (iEta / absIeta);

      /*std::cout << "iEta " << iEta << "\tabsiEta " << absIeta
        << "\tiPhi " << iPhi << "\tzSide "
        << zSide << std::endl;
      */

      // args to detid are zside, type of tower, absieta, iphi
      // absieta and iphi must be between 1 and 127 inclusive

      EcalTriggerPrimitiveDigi ecalDigi(EcalTrigTowerDetId(zSide, EcalTriggerTower, absIeta, iPhi));
      ecalDigi.setSize(nEcalSamples);

      // last arg is 3-bit trigger tower flag, which we don't use
      // we only use 8-bit encoded et and 1-bit fg
      ecalDigi.setSample(0, EcalTriggerPrimitiveSample(encodedEtEcal, fineGrainEcal, 0));
      // std::cout << ecalDigi << std::endl;
      ecalTPs->push_back(ecalDigi);

      HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId(iEta, iPhi));

      hcalDigi.setSize(nHcalSamples);

      // last two arg's are slb and slb channel, which we don't need
      hcalDigi.setSample(0, HcalTriggerPrimitiveSample(encodedEtHcal, fineGrainHcal, 0, 0));
      // std::cout << hcalDigi << std::endl;
      hcalTPs->push_back(hcalDigi);
    }

    // also need to push_back HF digis!
    // file input doesn't include HF, so need empty digis
    for (int i = 0; i < 18; i++) {
      for (int j = 0; j < 8; j++) {
        // HF ieta: +- 29 through 32.  HF iphi: 1,5,9,13,etc.
        int hfIEta = (j % 4) + 29;
        if (i < 9) {
          hfIEta = hfIEta * (-1);
        }
        // iphi shift not implemented, but not necessary here --
        // everything's filled with zeros so it's symmetric anyhow
        int hfIPhi = (i % 9) * 8 + (j / 4) * 4 + 1;

        HcalTriggerPrimitiveDigi hfDigi(HcalTrigTowerDetId(hfIEta, hfIPhi));
        hfDigi.setSize(1);
        hfDigi.setSample(0, HcalTriggerPrimitiveSample(0, false, 0, 0));
        hcalTPs->push_back(hfDigi);
      }
    }
  }
  iEvent.put(std::move(ecalTPs));
  iEvent.put(std::move(hcalTPs));

  nEvent_++;
  // std::cout << "Produce done" << std::endl;
}

// ------------ method called once each job just before starting event loop
// ------------
void RctInputTextToDigi::beginJob() {
  // open input file to read all events
  // inputStream_.open(inputFile_.fullPath().c_str());
  // std::cout << "beginJob entered" << std::endl;
}

// ------------ method called once each job just after ending the event loop
// ------------
void RctInputTextToDigi::endJob() {
  // close input file
  // inputStream_.close();
}

// define this as a plug-in
DEFINE_FWK_MODULE(RctInputTextToDigi);
