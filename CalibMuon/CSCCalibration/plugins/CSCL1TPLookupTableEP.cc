// system include files
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableCCLUTRcd.h"
#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableME11ILTRcd.h"
#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableME21ILTRcd.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableCCLUT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME21ILT.h"

// user include files
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

class CSCL1TPLookupTableEP : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCL1TPLookupTableEP(const edm::ParameterSet&);
  ~CSCL1TPLookupTableEP() override {}

  std::unique_ptr<CSCL1TPLookupTableCCLUT> produceCCLUT(const CSCL1TPLookupTableCCLUTRcd&);
  std::unique_ptr<CSCL1TPLookupTableME11ILT> produceME11ILT(const CSCL1TPLookupTableME11ILTRcd&);
  std::unique_ptr<CSCL1TPLookupTableME21ILT> produceME21ILT(const CSCL1TPLookupTableME21ILTRcd&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  std::vector<unsigned> load(std::string fileName) const;
  const edm::ParameterSet pset_;
};

CSCL1TPLookupTableEP::CSCL1TPLookupTableEP(const edm::ParameterSet& pset) : pset_(pset) {
  setWhatProduced(this, &CSCL1TPLookupTableEP::produceCCLUT);
  setWhatProduced(this, &CSCL1TPLookupTableEP::produceME11ILT);
  setWhatProduced(this, &CSCL1TPLookupTableEP::produceME21ILT);
  findingRecord<CSCL1TPLookupTableCCLUTRcd>();
  findingRecord<CSCL1TPLookupTableME11ILTRcd>();
  findingRecord<CSCL1TPLookupTableME21ILTRcd>();
}

void CSCL1TPLookupTableEP::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                          const edm::IOVSyncValue& iosv,
                                          edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

std::unique_ptr<CSCL1TPLookupTableCCLUT> CSCL1TPLookupTableEP::produceCCLUT(const CSCL1TPLookupTableCCLUTRcd&) {
  // make the LUT object
  std::unique_ptr<CSCL1TPLookupTableCCLUT> lut = std::make_unique<CSCL1TPLookupTableCCLUT>();

  // get the text files
  std::vector<std::string> positionLUTFiles_ = pset_.getParameter<std::vector<std::string>>("positionLUTFiles");
  std::vector<std::string> slopeLUTFiles_ = pset_.getParameter<std::vector<std::string>>("slopeLUTFiles");

  std::unordered_map<unsigned, std::vector<unsigned>> cclutPosition;
  std::unordered_map<unsigned, std::vector<unsigned>> cclutSlope;

  // read the text files and extract the data
  for (int i = 0; i < 5; ++i) {
    cclutPosition[i] = load(positionLUTFiles_[i]);
    cclutSlope[i] = load(slopeLUTFiles_[i]);
  }

  // set the data in the LUT object
  lut->set_cclutPosition(std::move(cclutPosition));
  lut->set_cclutSlope(std::move(cclutSlope));

  return lut;
}

std::unique_ptr<CSCL1TPLookupTableME11ILT> CSCL1TPLookupTableEP::produceME11ILT(const CSCL1TPLookupTableME11ILTRcd&) {
  // make the LUT object
  std::unique_ptr<CSCL1TPLookupTableME11ILT> lut = std::make_unique<CSCL1TPLookupTableME11ILT>();

  // get the text files
  std::vector<std::string> padToEsME11aFiles_ = pset_.getParameter<std::vector<std::string>>("padToEsME11aFiles");
  std::vector<std::string> padToEsME11bFiles_ = pset_.getParameter<std::vector<std::string>>("padToEsME11bFiles");

  std::vector<std::string> rollToMaxWgME11Files_ = pset_.getParameter<std::vector<std::string>>("rollToMaxWgME11Files");
  std::vector<std::string> rollToMinWgME11Files_ = pset_.getParameter<std::vector<std::string>>("rollToMinWgME11Files");

  std::vector<std::string> gemCscSlopeCosiFiles_ = pset_.getParameter<std::vector<std::string>>("gemCscSlopeCosiFiles");
  std::vector<std::string> gemCscSlopeCosiCorrectionFiles_ =
      pset_.getParameter<std::vector<std::string>>("gemCscSlopeCosiCorrectionFiles");
  std::vector<std::string> gemCscSlopeCorrectionFiles_ =
      pset_.getParameter<std::vector<std::string>>("gemCscSlopeCorrectionFiles");

  std::vector<std::string> esDiffToSlopeME11aFiles_ =
      pset_.getParameter<std::vector<std::string>>("esDiffToSlopeME11aFiles");
  std::vector<std::string> esDiffToSlopeME11bFiles_ =
      pset_.getParameter<std::vector<std::string>>("esDiffToSlopeME11bFiles");

  // read the text files and extract the data
  auto GEM_pad_CSC_es_ME11a_even_ = load(padToEsME11aFiles_[0]);
  auto GEM_pad_CSC_es_ME11a_odd_ = load(padToEsME11aFiles_[1]);
  auto GEM_pad_CSC_es_ME11b_even_ = load(padToEsME11bFiles_[0]);
  auto GEM_pad_CSC_es_ME11b_odd_ = load(padToEsME11bFiles_[1]);

  auto GEM_roll_CSC_min_wg_ME11_even_ = load(rollToMinWgME11Files_[0]);
  auto GEM_roll_CSC_min_wg_ME11_odd_ = load(rollToMinWgME11Files_[1]);
  auto GEM_roll_CSC_max_wg_ME11_even_ = load(rollToMaxWgME11Files_[0]);
  auto GEM_roll_CSC_max_wg_ME11_odd_ = load(rollToMaxWgME11Files_[1]);

  auto CSC_slope_cosi_2to1_L1_ME11a_even_ = load(gemCscSlopeCosiFiles_[0]);
  auto CSC_slope_cosi_2to1_L1_ME11a_odd_ = load(gemCscSlopeCosiFiles_[1]);
  auto CSC_slope_cosi_3to1_L1_ME11a_even_ = load(gemCscSlopeCosiFiles_[2]);
  auto CSC_slope_cosi_3to1_L1_ME11a_odd_ = load(gemCscSlopeCosiFiles_[3]);

  auto CSC_slope_cosi_2to1_L1_ME11b_even_ = load(gemCscSlopeCosiFiles_[4]);
  auto CSC_slope_cosi_2to1_L1_ME11b_odd_ = load(gemCscSlopeCosiFiles_[5]);
  auto CSC_slope_cosi_3to1_L1_ME11b_even_ = load(gemCscSlopeCosiFiles_[6]);
  auto CSC_slope_cosi_3to1_L1_ME11b_odd_ = load(gemCscSlopeCosiFiles_[7]);

  auto CSC_slope_cosi_corr_L1_ME11a_even_ = load(gemCscSlopeCosiCorrectionFiles_[0]);
  auto CSC_slope_cosi_corr_L1_ME11b_even_ = load(gemCscSlopeCosiCorrectionFiles_[1]);
  auto CSC_slope_cosi_corr_L1_ME11a_odd_ = load(gemCscSlopeCosiCorrectionFiles_[3]);
  auto CSC_slope_cosi_corr_L1_ME11b_odd_ = load(gemCscSlopeCosiCorrectionFiles_[4]);

  auto CSC_slope_corr_L1_ME11a_even_ = load(gemCscSlopeCorrectionFiles_[0]);
  auto CSC_slope_corr_L1_ME11b_even_ = load(gemCscSlopeCorrectionFiles_[1]);
  auto CSC_slope_corr_L1_ME11a_odd_ = load(gemCscSlopeCorrectionFiles_[3]);
  auto CSC_slope_corr_L1_ME11b_odd_ = load(gemCscSlopeCorrectionFiles_[4]);
  auto CSC_slope_corr_L2_ME11a_even_ = load(gemCscSlopeCorrectionFiles_[6]);
  auto CSC_slope_corr_L2_ME11b_even_ = load(gemCscSlopeCorrectionFiles_[7]);
  auto CSC_slope_corr_L2_ME11a_odd_ = load(gemCscSlopeCorrectionFiles_[9]);
  auto CSC_slope_corr_L2_ME11b_odd_ = load(gemCscSlopeCorrectionFiles_[10]);

  auto es_diff_slope_L1_ME11a_even_ = load(esDiffToSlopeME11aFiles_[0]);
  auto es_diff_slope_L1_ME11a_odd_ = load(esDiffToSlopeME11aFiles_[1]);
  auto es_diff_slope_L2_ME11a_even_ = load(esDiffToSlopeME11aFiles_[2]);
  auto es_diff_slope_L2_ME11a_odd_ = load(esDiffToSlopeME11aFiles_[3]);

  auto es_diff_slope_L1_ME11b_even_ = load(esDiffToSlopeME11bFiles_[0]);
  auto es_diff_slope_L1_ME11b_odd_ = load(esDiffToSlopeME11bFiles_[1]);
  auto es_diff_slope_L2_ME11b_even_ = load(esDiffToSlopeME11bFiles_[2]);
  auto es_diff_slope_L2_ME11b_odd_ = load(esDiffToSlopeME11bFiles_[3]);

  // set the data in the LUT object
  lut->set_GEM_pad_CSC_es_ME11b_even(std::move(GEM_pad_CSC_es_ME11b_even_));
  lut->set_GEM_pad_CSC_es_ME11a_even(std::move(GEM_pad_CSC_es_ME11a_even_));
  lut->set_GEM_pad_CSC_es_ME11b_odd(std::move(GEM_pad_CSC_es_ME11b_odd_));
  lut->set_GEM_pad_CSC_es_ME11a_odd(std::move(GEM_pad_CSC_es_ME11a_odd_));

  lut->set_GEM_roll_CSC_min_wg_ME11_even(std::move(GEM_roll_CSC_min_wg_ME11_even_));
  lut->set_GEM_roll_CSC_min_wg_ME11_odd(std::move(GEM_roll_CSC_min_wg_ME11_odd_));
  lut->set_GEM_roll_CSC_max_wg_ME11_even(std::move(GEM_roll_CSC_max_wg_ME11_even_));
  lut->set_GEM_roll_CSC_max_wg_ME11_odd(std::move(GEM_roll_CSC_max_wg_ME11_odd_));

  // GEM-CSC trigger: slope correction
  lut->set_CSC_slope_cosi_2to1_L1_ME11a_even(std::move(CSC_slope_cosi_2to1_L1_ME11a_even_));
  lut->set_CSC_slope_cosi_2to1_L1_ME11a_odd(std::move(CSC_slope_cosi_2to1_L1_ME11a_odd_));
  lut->set_CSC_slope_cosi_3to1_L1_ME11a_even(std::move(CSC_slope_cosi_3to1_L1_ME11a_even_));
  lut->set_CSC_slope_cosi_3to1_L1_ME11a_odd(std::move(CSC_slope_cosi_3to1_L1_ME11a_odd_));

  lut->set_CSC_slope_cosi_2to1_L1_ME11b_even(std::move(CSC_slope_cosi_2to1_L1_ME11b_even_));
  lut->set_CSC_slope_cosi_2to1_L1_ME11b_odd(std::move(CSC_slope_cosi_2to1_L1_ME11b_odd_));
  lut->set_CSC_slope_cosi_3to1_L1_ME11b_even(std::move(CSC_slope_cosi_3to1_L1_ME11b_even_));
  lut->set_CSC_slope_cosi_3to1_L1_ME11b_odd(std::move(CSC_slope_cosi_3to1_L1_ME11b_odd_));

  lut->set_CSC_slope_corr_L1_ME11a_even(std::move(CSC_slope_corr_L1_ME11a_even_));
  lut->set_CSC_slope_corr_L1_ME11a_odd(std::move(CSC_slope_corr_L1_ME11a_odd_));
  lut->set_CSC_slope_corr_L1_ME11b_even(std::move(CSC_slope_corr_L1_ME11b_even_));
  lut->set_CSC_slope_corr_L1_ME11b_odd(std::move(CSC_slope_corr_L1_ME11b_odd_));
  lut->set_CSC_slope_corr_L2_ME11a_even(std::move(CSC_slope_corr_L2_ME11a_even_));
  lut->set_CSC_slope_corr_L2_ME11a_odd(std::move(CSC_slope_corr_L2_ME11a_odd_));
  lut->set_CSC_slope_corr_L2_ME11b_even(std::move(CSC_slope_corr_L2_ME11b_even_));
  lut->set_CSC_slope_corr_L2_ME11b_odd(std::move(CSC_slope_corr_L2_ME11b_odd_));

  // GEM-CSC trigger: 1/8-strip difference to slope
  lut->set_es_diff_slope_L1_ME11a_even(std::move(es_diff_slope_L1_ME11a_even_));
  lut->set_es_diff_slope_L1_ME11a_odd(std::move(es_diff_slope_L1_ME11a_odd_));
  lut->set_es_diff_slope_L2_ME11a_even(std::move(es_diff_slope_L2_ME11a_even_));
  lut->set_es_diff_slope_L2_ME11a_odd(std::move(es_diff_slope_L2_ME11a_odd_));

  lut->set_es_diff_slope_L1_ME11b_even(std::move(es_diff_slope_L1_ME11b_even_));
  lut->set_es_diff_slope_L1_ME11b_odd(std::move(es_diff_slope_L1_ME11b_odd_));
  lut->set_es_diff_slope_L2_ME11b_even(std::move(es_diff_slope_L2_ME11b_even_));
  lut->set_es_diff_slope_L2_ME11b_odd(std::move(es_diff_slope_L2_ME11b_odd_));

  return lut;
}

std::unique_ptr<CSCL1TPLookupTableME21ILT> CSCL1TPLookupTableEP::produceME21ILT(const CSCL1TPLookupTableME21ILTRcd&) {
  // make the LUT object
  std::unique_ptr<CSCL1TPLookupTableME21ILT> lut = std::make_unique<CSCL1TPLookupTableME21ILT>();

  // get the text files
  std::vector<std::string> padToEsME21Files_ = pset_.getParameter<std::vector<std::string>>("padToEsME21Files");

  std::vector<std::string> rollToMaxWgME21Files_ = pset_.getParameter<std::vector<std::string>>("rollToMaxWgME21Files");
  std::vector<std::string> rollToMinWgME21Files_ = pset_.getParameter<std::vector<std::string>>("rollToMinWgME21Files");

  std::vector<std::string> gemCscSlopeCosiFiles_ = pset_.getParameter<std::vector<std::string>>("gemCscSlopeCosiFiles");
  std::vector<std::string> gemCscSlopeCosiCorrectionFiles_ =
      pset_.getParameter<std::vector<std::string>>("gemCscSlopeCosiCorrectionFiles");
  std::vector<std::string> gemCscSlopeCorrectionFiles_ =
      pset_.getParameter<std::vector<std::string>>("gemCscSlopeCorrectionFiles");

  std::vector<std::string> esDiffToSlopeME21Files_ =
      pset_.getParameter<std::vector<std::string>>("esDiffToSlopeME21Files");

  // read the text files and extract the data
  auto GEM_pad_CSC_es_ME21_even_ = load(padToEsME21Files_[0]);
  auto GEM_pad_CSC_es_ME21_odd_ = load(padToEsME21Files_[1]);

  auto GEM_roll_L1_CSC_min_wg_ME21_even_ = load(rollToMinWgME21Files_[0]);
  auto GEM_roll_L1_CSC_min_wg_ME21_odd_ = load(rollToMinWgME21Files_[1]);
  auto GEM_roll_L2_CSC_min_wg_ME21_even_ = load(rollToMinWgME21Files_[2]);
  auto GEM_roll_L2_CSC_min_wg_ME21_odd_ = load(rollToMinWgME21Files_[3]);

  auto GEM_roll_L1_CSC_max_wg_ME21_even_ = load(rollToMaxWgME21Files_[0]);
  auto GEM_roll_L1_CSC_max_wg_ME21_odd_ = load(rollToMaxWgME21Files_[1]);
  auto GEM_roll_L2_CSC_max_wg_ME21_even_ = load(rollToMaxWgME21Files_[2]);
  auto GEM_roll_L2_CSC_max_wg_ME21_odd_ = load(rollToMaxWgME21Files_[3]);

  auto es_diff_slope_L1_ME21_even_ = load(esDiffToSlopeME21Files_[0]);
  auto es_diff_slope_L1_ME21_odd_ = load(esDiffToSlopeME21Files_[1]);
  auto es_diff_slope_L2_ME21_even_ = load(esDiffToSlopeME21Files_[2]);
  auto es_diff_slope_L2_ME21_odd_ = load(esDiffToSlopeME21Files_[3]);

  auto CSC_slope_cosi_2to1_L1_ME21_even_ = load(gemCscSlopeCosiFiles_[8]);
  auto CSC_slope_cosi_2to1_L1_ME21_odd_ = load(gemCscSlopeCosiFiles_[9]);
  auto CSC_slope_cosi_3to1_L1_ME21_even_ = load(gemCscSlopeCosiFiles_[10]);
  auto CSC_slope_cosi_3to1_L1_ME21_odd_ = load(gemCscSlopeCosiFiles_[11]);

  auto CSC_slope_cosi_corr_L1_ME21_even_ = load(gemCscSlopeCosiCorrectionFiles_[2]);
  auto CSC_slope_cosi_corr_L1_ME21_odd_ = load(gemCscSlopeCosiCorrectionFiles_[5]);

  auto CSC_slope_corr_L1_ME21_even_ = load(gemCscSlopeCorrectionFiles_[2]);
  auto CSC_slope_corr_L1_ME21_odd_ = load(gemCscSlopeCorrectionFiles_[5]);
  auto CSC_slope_corr_L2_ME21_even_ = load(gemCscSlopeCorrectionFiles_[8]);
  auto CSC_slope_corr_L2_ME21_odd_ = load(gemCscSlopeCorrectionFiles_[11]);

  // set the data in the LUT object
  lut->set_GEM_pad_CSC_es_ME21_even(std::move(GEM_pad_CSC_es_ME21_even_));
  lut->set_GEM_pad_CSC_es_ME21_odd(std::move(GEM_pad_CSC_es_ME21_odd_));

  lut->set_GEM_roll_L1_CSC_min_wg_ME21_even(std::move(GEM_roll_L1_CSC_min_wg_ME21_even_));
  lut->set_GEM_roll_L1_CSC_max_wg_ME21_even(std::move(GEM_roll_L1_CSC_max_wg_ME21_even_));
  lut->set_GEM_roll_L1_CSC_min_wg_ME21_odd(std::move(GEM_roll_L1_CSC_min_wg_ME21_odd_));
  lut->set_GEM_roll_L1_CSC_max_wg_ME21_odd(std::move(GEM_roll_L1_CSC_max_wg_ME21_odd_));

  lut->set_GEM_roll_L2_CSC_min_wg_ME21_even(std::move(GEM_roll_L2_CSC_min_wg_ME21_even_));
  lut->set_GEM_roll_L2_CSC_max_wg_ME21_even(std::move(GEM_roll_L2_CSC_max_wg_ME21_even_));
  lut->set_GEM_roll_L2_CSC_min_wg_ME21_odd(std::move(GEM_roll_L2_CSC_min_wg_ME21_odd_));
  lut->set_GEM_roll_L2_CSC_max_wg_ME21_odd(std::move(GEM_roll_L2_CSC_max_wg_ME21_odd_));

  lut->set_es_diff_slope_L1_ME21_even(std::move(es_diff_slope_L1_ME21_even_));
  lut->set_es_diff_slope_L1_ME21_odd(std::move(es_diff_slope_L1_ME21_odd_));
  lut->set_es_diff_slope_L2_ME21_even(std::move(es_diff_slope_L2_ME21_even_));
  lut->set_es_diff_slope_L2_ME21_odd(std::move(es_diff_slope_L2_ME21_odd_));

  lut->set_CSC_slope_cosi_2to1_L1_ME21_even(std::move(CSC_slope_cosi_2to1_L1_ME21_even_));
  lut->set_CSC_slope_cosi_2to1_L1_ME21_odd(std::move(CSC_slope_cosi_2to1_L1_ME21_odd_));
  lut->set_CSC_slope_cosi_3to1_L1_ME21_even(std::move(CSC_slope_cosi_3to1_L1_ME21_even_));
  lut->set_CSC_slope_cosi_3to1_L1_ME21_odd(std::move(CSC_slope_cosi_3to1_L1_ME21_odd_));

  lut->set_CSC_slope_corr_L1_ME21_even(std::move(CSC_slope_corr_L1_ME21_even_));
  lut->set_CSC_slope_corr_L1_ME21_odd(std::move(CSC_slope_corr_L1_ME21_odd_));
  lut->set_CSC_slope_corr_L2_ME21_even(std::move(CSC_slope_corr_L2_ME21_even_));
  lut->set_CSC_slope_corr_L2_ME21_odd(std::move(CSC_slope_corr_L2_ME21_odd_));

  return lut;
}

std::vector<unsigned> CSCL1TPLookupTableEP::load(std::string fileName) const {
  std::vector<unsigned> returnV;
  std::ifstream fstream;
  fstream.open(edm::FileInPath(fileName.c_str()).fullPath());
  // empty file, return empty lut
  if (!fstream.good()) {
    fstream.close();
    return returnV;
  }

  std::string line;

  while (std::getline(fstream, line)) {
    //ignore comments
    line.erase(std::find(line.begin(), line.end(), '#'), line.end());
    std::istringstream lineStream(line);
    std::pair<unsigned, unsigned> entry;
    while (lineStream >> entry.first >> entry.second) {
      returnV.push_back(entry.second);
    }
  }
  return returnV;
}

DEFINE_FWK_EVENTSETUP_SOURCE(CSCL1TPLookupTableEP);
