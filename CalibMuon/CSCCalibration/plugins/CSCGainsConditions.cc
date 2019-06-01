#include <fstream>
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCGainsConditions.h"

CSCGains *CSCGainsConditions::prefillGains() {
  float mean, min, minchi;
  int seed;
  int old_chamber_id, old_strip, new_chamber_id, new_strip;
  float old_gainslope, old_intercpt, old_chisq;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_slope;
  std::vector<float> old_intercept;
  std::vector<float> old_chi2;
  float new_gainslope, new_intercpt, new_chisq;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope;
  std::vector<float> new_intercept;
  std::vector<float> new_chi2;

  const CSCDetId &detId = CSCDetId();
  CSCGains *cngains = new CSCGains();

  int max_istrip, id_layer, max_ring, max_cham;
  unsigned int old_nrlines = 0;
  unsigned int new_nrlines = 0;
  seed = 10000;
  srand(seed);
  mean = 6.8, min = -10.0, minchi = 1.0;

  std::ifstream olddata;
  olddata.open("old_gains.dat", std::ios::in);
  if (!olddata) {
    std::cerr << "Error: old_gains.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!olddata.eof()) {
    olddata >> old_chamber_id >> old_strip >> old_gainslope >> old_intercpt >> old_chisq;
    old_cham_id.push_back(old_chamber_id);
    old_strips.push_back(old_strip);
    old_slope.push_back(old_gainslope);
    old_intercept.push_back(old_intercpt);
    old_chi2.push_back(old_chisq);
    old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  newdata.open("new_gains.txt", std::ios::in);
  if (!newdata) {
    std::cerr << "Error: new_gains.txt -> no such file!" << std::endl;
    exit(1);
  }

  while (!newdata.eof()) {
    newdata >> new_chamber_id >> new_strip >> new_gainslope >> new_intercpt >> new_chisq;
    new_cham_id.push_back(new_chamber_id);
    new_strips.push_back(new_strip);
    new_slope.push_back(new_gainslope);
    new_intercept.push_back(new_intercpt);
    new_chi2.push_back(new_chisq);
    new_nrlines++;
  }
  newdata.close();

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

            std::vector<CSCGains::Item> itemvector;
            itemvector.resize(max_istrip);
            id_layer = 100000 * iendcap + 10000 * istation + 1000 * iring + 10 * ichamber + ilayer;

            for (int istrip = 0; istrip < max_istrip; istrip++) {
              itemvector[istrip].gain_slope = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + mean;
              itemvector[istrip].gain_intercept = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + min;
              itemvector[istrip].gain_chi2 = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + minchi;
              cngains->gains[id_layer] = itemvector;
            }
          }
        }
      }
    }
  }

  // overwrite fakes with old values from DB
  int istrip = 0;
  std::vector<CSCGains::Item> itemvector;
  itemvector.resize(80);

  for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
    if (old_strips[mystrip] == 0)
      istrip = 0;
    itemvector[istrip].gain_slope = old_slope[mystrip];
    itemvector[istrip].gain_intercept = old_intercept[mystrip];
    itemvector[istrip].gain_chi2 = old_chi2[mystrip];
    cngains->gains[old_cham_id[mystrip]] = itemvector;
    istrip++;
  }

  itemvector.resize(64);
  for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
    if (old_strips[mystrip] == 0)
      istrip = 0;
    if (old_cham_id[mystrip] >= 113000 && old_cham_id[mystrip] <= 113999) {
      itemvector[istrip].gain_slope = old_slope[mystrip];
      itemvector[istrip].gain_intercept = old_intercept[mystrip];
      itemvector[istrip].gain_chi2 = old_chi2[mystrip];
      cngains->gains[old_cham_id[mystrip]] = itemvector;
      istrip++;
    }
  }

  itemvector.resize(64);
  for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
    if (old_strips[mystrip] == 0)
      istrip = 0;
    if (old_cham_id[mystrip] >= 213000 && old_cham_id[mystrip] <= 213999) {
      itemvector[istrip].gain_slope = old_slope[mystrip];
      itemvector[istrip].gain_intercept = old_intercept[mystrip];
      itemvector[istrip].gain_chi2 = old_chi2[mystrip];
      cngains->gains[old_cham_id[mystrip]] = itemvector;
      istrip++;
    }
  }

  // overwrite old values with ones from new runs
  itemvector.resize(80);
  for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
    if (new_strips[mystrip] == 0)
      istrip = 0;
    itemvector[istrip].gain_slope = new_slope[mystrip];
    itemvector[istrip].gain_intercept = new_intercept[mystrip];
    itemvector[istrip].gain_chi2 = new_chi2[mystrip];
    cngains->gains[new_cham_id[mystrip]] = itemvector;
    istrip++;
  }

  itemvector.resize(64);
  for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
    if (new_strips[mystrip] == 0)
      istrip = 0;
    if (new_cham_id[mystrip] >= 113000 && new_cham_id[mystrip] <= 113999) {
      itemvector[istrip].gain_slope = new_slope[mystrip];
      itemvector[istrip].gain_intercept = new_intercept[mystrip];
      itemvector[istrip].gain_chi2 = new_chi2[mystrip];
      cngains->gains[new_cham_id[mystrip]] = itemvector;
      istrip++;
    }
  }

  itemvector.resize(64);
  for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
    if (new_strips[mystrip] == 0)
      istrip = 0;
    if (new_cham_id[mystrip] >= 213000 && new_cham_id[mystrip] <= 213999) {
      itemvector[istrip].gain_slope = new_slope[mystrip];
      itemvector[istrip].gain_intercept = new_intercept[mystrip];
      itemvector[istrip].gain_chi2 = new_chi2[mystrip];
      cngains->gains[new_cham_id[mystrip]] = itemvector;
      istrip++;
    }
  }
  return cngains;
}

CSCGainsConditions::CSCGainsConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCGainsConditions::produceGains);
  findingRecord<CSCGainsRcd>();
  // now do what ever other initialization is needed
}

CSCGainsConditions::~CSCGainsConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCGainsConditions::ReturnType CSCGainsConditions::produceGains(const CSCGainsRcd &iRecord) {
  // Added by Zhen, need a new object so to not be deleted at exit
  return CSCGainsConditions::ReturnType(prefillGains());
}

void CSCGainsConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                        const edm::IOVSyncValue &,
                                        edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
