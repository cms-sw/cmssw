#include <fstream>
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkConditions.h"

CSCcrosstalk *CSCCrosstalkConditions::prefillCrosstalk() {
  float mean, min, minchi;
  int seed;
  int old_chamber_id, old_strip, new_chamber_id, new_strip;
  float old_slope_right, old_slope_left, old_intercept_right;
  float old_intercept_left, old_chi2_right, old_chi2_left;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_slope_r;
  std::vector<float> old_intercept_r;
  std::vector<float> old_chi2_r;
  std::vector<float> old_slope_l;
  std::vector<float> old_intercept_l;
  std::vector<float> old_chi2_l;
  float new_slope_right, new_slope_left, new_intercept_right;
  float new_intercept_left, new_chi2_right, new_chi2_left;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope_r;
  std::vector<float> new_intercept_r;
  std::vector<float> new_chi2_r;
  std::vector<float> new_slope_l;
  std::vector<float> new_intercept_l;
  std::vector<float> new_chi2_l;

  const CSCDetId &detId = CSCDetId();
  CSCcrosstalk *cncrosstalk = new CSCcrosstalk();

  int max_istrip, id_layer, max_ring, max_cham;
  unsigned int old_nrlines = 0;
  unsigned int new_nrlines = 0;
  seed = 10000;
  srand(seed);
  mean = -0.0009, min = 0.035, minchi = 1.5;

  // endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6
  std::ifstream olddata;
  olddata.open("old_xtalk.dat", std::ios::in);
  if (!olddata) {
    std::cerr << "Error: old_xtalk.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!olddata.eof()) {
    olddata >> old_chamber_id >> old_strip >> old_slope_right >> old_intercept_right >> old_chi2_right >>
        old_slope_left >> old_intercept_left >> old_chi2_left;
    old_cham_id.push_back(old_chamber_id);
    old_strips.push_back(old_strip);
    old_slope_r.push_back(old_slope_right);
    old_slope_l.push_back(old_slope_left);
    old_intercept_r.push_back(old_intercept_right);
    old_intercept_l.push_back(old_intercept_left);
    old_chi2_r.push_back(old_chi2_right);
    old_chi2_l.push_back(old_chi2_left);
    old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  newdata.open("new_xtalk.txt", std::ios::in);
  if (!newdata) {
    std::cerr << "Error: new_xtalk.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!newdata.eof()) {
    newdata >> new_chamber_id >> new_strip >> new_slope_right >> new_intercept_right >> new_chi2_right >>
        new_slope_left >> new_intercept_left >> new_chi2_left;
    new_cham_id.push_back(new_chamber_id);
    new_strips.push_back(new_strip);
    new_slope_r.push_back(new_slope_right);
    new_slope_l.push_back(new_slope_left);
    new_intercept_r.push_back(new_intercept_right);
    new_intercept_l.push_back(new_intercept_left);
    new_chi2_r.push_back(new_chi2_right);
    new_chi2_l.push_back(new_chi2_left);
    new_nrlines++;
  }
  newdata.close();

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
        // station 1 ring 3 has 64 strips per layer instead of 80(minus & plus
        // side!!!)

        for (int ichamber = detId.minChamberId(); ichamber <= max_cham; ichamber++) {
          for (int ilayer = detId.minLayerId(); ilayer <= detId.maxLayerId(); ilayer++) {
            // station 1 ring 3 has 64 strips per layer instead of 80
            if (istation == 1 && iring == 3)
              max_istrip = 64;

            std::vector<CSCcrosstalk::Item> itemvector;
            itemvector.resize(max_istrip);
            id_layer = 100000 * iendcap + 10000 * istation + 1000 * iring + 10 * ichamber + ilayer;

            for (int istrip = 0; istrip < max_istrip; istrip++) {
              // create fake values
              itemvector[istrip].xtalk_slope_right =
                  -((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 10000 + mean;
              itemvector[istrip].xtalk_intercept_right =
                  ((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 100 + min;
              itemvector[istrip].xtalk_chi2_right = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + minchi;
              itemvector[istrip].xtalk_slope_left =
                  -((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 10000 + mean;
              itemvector[istrip].xtalk_intercept_left =
                  ((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 100 + min;
              itemvector[istrip].xtalk_chi2_left = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + minchi;
              cncrosstalk->crosstalk[id_layer] = itemvector;

              if (istrip == 0) {
                itemvector[istrip].xtalk_slope_right =
                    -((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 10000 + mean;
                itemvector[istrip].xtalk_intercept_right =
                    ((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 100 + min;
                itemvector[istrip].xtalk_chi2_right = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + minchi;
                itemvector[istrip].xtalk_slope_left = 0.0;
                itemvector[istrip].xtalk_intercept_left = 0.0;
                itemvector[istrip].xtalk_chi2_left = 0.0;
                cncrosstalk->crosstalk[id_layer] = itemvector;
              }

              if (istrip == 79) {
                itemvector[istrip].xtalk_slope_right = 0.0;
                itemvector[istrip].xtalk_intercept_right = 0.0;
                itemvector[istrip].xtalk_chi2_right = 0.0;
                itemvector[istrip].xtalk_slope_left =
                    -((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 10000 + mean;
                itemvector[istrip].xtalk_intercept_left =
                    ((double)rand() / ((double)(RAND_MAX) + (double)(1))) / 100 + min;
                itemvector[istrip].xtalk_chi2_left = ((double)rand() / ((double)(RAND_MAX) + (double)(1))) + minchi;
                cncrosstalk->crosstalk[id_layer] = itemvector;
              }
            }
          }
        }
      }
    }

    // overwrite fakes with old values from DB
    int istrip = 0;
    std::vector<CSCcrosstalk::Item> itemvector;
    itemvector.resize(80);

    for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
      if (old_strips[mystrip] == 0)
        istrip = 0;
      itemvector[istrip].xtalk_slope_right = old_slope_r[mystrip];
      itemvector[istrip].xtalk_intercept_right = old_intercept_r[mystrip];
      itemvector[istrip].xtalk_chi2_right = old_chi2_r[mystrip];
      itemvector[istrip].xtalk_slope_left = old_slope_l[mystrip];
      itemvector[istrip].xtalk_intercept_left = old_intercept_l[mystrip];
      itemvector[istrip].xtalk_chi2_left = old_chi2_l[mystrip];
      cncrosstalk->crosstalk[old_cham_id[mystrip]] = itemvector;
      istrip++;
    }

    itemvector.resize(64);
    for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
      if (old_strips[mystrip] == 0)
        istrip = 0;
      if (old_cham_id[mystrip] >= 113000 && old_cham_id[mystrip] <= 113999) {
        itemvector[istrip].xtalk_slope_right = old_slope_r[mystrip];
        itemvector[istrip].xtalk_intercept_right = old_intercept_r[mystrip];
        itemvector[istrip].xtalk_chi2_right = old_chi2_r[mystrip];
        itemvector[istrip].xtalk_slope_left = old_slope_l[mystrip];
        itemvector[istrip].xtalk_intercept_left = old_intercept_l[mystrip];
        itemvector[istrip].xtalk_chi2_left = old_chi2_l[mystrip];
        cncrosstalk->crosstalk[old_cham_id[mystrip]] = itemvector;
        istrip++;
      }
    }

    itemvector.resize(64);
    for (unsigned int mystrip = 0; mystrip < old_nrlines - 1; mystrip++) {
      if (old_strips[mystrip] == 0)
        istrip = 0;
      if (old_cham_id[mystrip] >= 213000 && old_cham_id[mystrip] <= 213999) {
        itemvector[istrip].xtalk_slope_right = old_slope_r[mystrip];
        itemvector[istrip].xtalk_intercept_right = old_intercept_r[mystrip];
        itemvector[istrip].xtalk_chi2_right = old_chi2_r[mystrip];
        itemvector[istrip].xtalk_slope_left = old_slope_l[mystrip];
        itemvector[istrip].xtalk_intercept_left = old_intercept_l[mystrip];
        itemvector[istrip].xtalk_chi2_left = old_chi2_l[mystrip];
        cncrosstalk->crosstalk[old_cham_id[mystrip]] = itemvector;
        istrip++;
      }
    }

    // overwrite old values with ones from new runs
    itemvector.resize(80);
    for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
      if (new_strips[mystrip] == 0)
        istrip = 0;
      itemvector[istrip].xtalk_slope_right = new_slope_r[mystrip];
      itemvector[istrip].xtalk_intercept_right = new_intercept_r[mystrip];
      itemvector[istrip].xtalk_chi2_right = new_chi2_r[mystrip];
      itemvector[istrip].xtalk_slope_left = new_slope_l[mystrip];
      itemvector[istrip].xtalk_intercept_left = new_intercept_l[mystrip];
      itemvector[istrip].xtalk_chi2_left = new_chi2_l[mystrip];
      cncrosstalk->crosstalk[new_cham_id[mystrip]] = itemvector;
      istrip++;
    }

    itemvector.resize(64);
    for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
      if (new_strips[mystrip] == 0)
        istrip = 0;
      if (new_cham_id[mystrip] >= 113000 && new_cham_id[mystrip] <= 113999) {
        itemvector[istrip].xtalk_slope_right = new_slope_r[mystrip];
        itemvector[istrip].xtalk_intercept_right = new_intercept_r[mystrip];
        itemvector[istrip].xtalk_chi2_right = new_chi2_r[mystrip];
        itemvector[istrip].xtalk_slope_left = new_slope_l[mystrip];
        itemvector[istrip].xtalk_intercept_left = new_intercept_l[mystrip];
        itemvector[istrip].xtalk_chi2_left = new_chi2_l[mystrip];
        cncrosstalk->crosstalk[new_cham_id[mystrip]] = itemvector;
        istrip++;
      }
    }

    itemvector.resize(64);
    for (unsigned int mystrip = 0; mystrip < new_nrlines - 1; mystrip++) {
      if (new_strips[mystrip] == 0)
        istrip = 0;
      if (new_cham_id[mystrip] >= 213000 && new_cham_id[mystrip] <= 213999) {
        itemvector[istrip].xtalk_slope_right = new_slope_r[mystrip];
        itemvector[istrip].xtalk_intercept_right = new_intercept_r[mystrip];
        itemvector[istrip].xtalk_chi2_right = new_chi2_r[mystrip];
        itemvector[istrip].xtalk_slope_left = new_slope_l[mystrip];
        itemvector[istrip].xtalk_intercept_left = new_intercept_l[mystrip];
        itemvector[istrip].xtalk_chi2_left = new_chi2_l[mystrip];
        cncrosstalk->crosstalk[new_cham_id[mystrip]] = itemvector;
        istrip++;
      }
    }
  }

  return cncrosstalk;
}

CSCCrosstalkConditions::CSCCrosstalkConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCCrosstalkConditions::produceCrosstalk);
  findingRecord<CSCcrosstalkRcd>();
  // now do what ever other initialization is needed
}

CSCCrosstalkConditions::~CSCCrosstalkConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCCrosstalkConditions::ReturnType CSCCrosstalkConditions::produceCrosstalk(const CSCcrosstalkRcd &iRecord) {
  // Added by Zhen, need a new object so to not be deleted at exit
  return CSCCrosstalkConditions::ReturnType(prefillCrosstalk());
}

void CSCCrosstalkConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                            const edm::IOVSyncValue &,
                                            edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
