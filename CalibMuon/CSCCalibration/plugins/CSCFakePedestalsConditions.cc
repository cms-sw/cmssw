
#include "CalibMuon/CSCCalibration/interface/CSCFakePedestalsConditions.h"

CSCPedestals *CSCFakePedestalsConditions::prefillPedestals() {
  const CSCDetId &detId = CSCDetId();
  CSCPedestals *cnpedestals = new CSCPedestals();

  int max_istrip, id_layer, max_ring, max_cham;
  seed = 10000;
  srand(seed);
  meanped = 600.0, meanrms = 1.5, M = 1000;

  // endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6

  for (int iendcap = detId.minEndcapId(); iendcap <= detId.maxEndcapId(); iendcap++) {
    for (int istation = detId.minStationId(); istation <= detId.maxStationId(); istation++) {
      max_ring = detId.maxRingId();
      // station 4 ring 4 not there(36 chambers*2 missing)
      // 3 rings max this way of counting (ME1a & b)
      if (istation == 1)
        max_ring = 3;
      if (istation == 2)
        max_ring = 2;
      if (istation == 3)
        max_ring = 2;
      if (istation == 4)
        max_ring = 1;

      for (int iring = detId.minRingId(); iring <= max_ring; iring++) {
        max_istrip = 80;
        max_cham = detId.maxChamberId();
        if (istation == 1 && iring == 1)
          max_cham = 36;
        if (istation == 1 && iring == 2)
          max_cham = 36;
        if (istation == 1 && iring == 3)
          max_cham = 36;
        if (istation == 2 && iring == 1)
          max_cham = 18;
        if (istation == 2 && iring == 2)
          max_cham = 36;
        if (istation == 3 && iring == 1)
          max_cham = 18;
        if (istation == 3 && iring == 2)
          max_cham = 36;
        if (istation == 4 && iring == 1)
          max_cham = 18;

        for (int ichamber = detId.minChamberId(); ichamber <= max_cham; ichamber++) {
          for (int ilayer = detId.minLayerId(); ilayer <= detId.maxLayerId(); ilayer++) {
            // station 1 ring 3 has 64 strips per layer instead of 80
            if (istation == 1 && iring == 3)
              max_istrip = 64;

            std::vector<CSCPedestals::Item> itemvector;
            itemvector.resize(max_istrip);
            id_layer = 100000 * iendcap + 10000 * istation + 1000 * iring + 10 * ichamber + ilayer;

            for (int istrip = 0; istrip < max_istrip; istrip++) {
              itemvector[istrip].ped = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) * 100 + meanped;
              itemvector[istrip].rms = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + meanrms;
              cnpedestals->pedestals[id_layer] = itemvector;
            }
          }
        }
      }
    }
  }
  return cnpedestals;
}

CSCFakePedestalsConditions::CSCFakePedestalsConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCFakePedestalsConditions::producePedestals);
  findingRecord<CSCPedestalsRcd>();
  // now do what ever other initialization is needed
}

CSCFakePedestalsConditions::~CSCFakePedestalsConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakePedestalsConditions::ReturnType CSCFakePedestalsConditions::producePedestals(const CSCPedestalsRcd &iRecord) {
  return CSCFakePedestalsConditions::ReturnType(prefillPedestals());
}

void CSCFakePedestalsConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                                const edm::IOVSyncValue &,
                                                edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
