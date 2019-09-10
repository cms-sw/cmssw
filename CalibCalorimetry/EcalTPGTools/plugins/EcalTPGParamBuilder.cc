#include "EcalTPGParamBuilder.h"

#include "CalibCalorimetry/EcalTPGTools/plugins/EcalTPGDBApp.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"

//modif-alex-27-july-2015
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

#include <TF1.h>
#include <TH2F.h>
#include <TFile.h>
#include <TNtuple.h>
#include <iomanip>
#include <fstream>

using namespace std;

double oneOverEtResolEt(double* x, double* par) {
  double Et = x[0];
  if (Et < 1e-6)
    return 1. / par[1];  // to avoid division by 0.
  double resolEt_overEt =
      sqrt((par[0] / sqrt(Et)) * (par[0] / sqrt(Et)) + (par[1] / Et) * (par[1] / Et) + par[2] * par[2]);
  return 1. / (Et * resolEt_overEt);
}

EcalTPGParamBuilder::EcalTPGParamBuilder(edm::ParameterSet const& pSet)
    : xtal_LSB_EB_(0), xtal_LSB_EE_(0), nSample_(5), complement2_(7), useDBShape_(true) {
  ped_conf_id_ = 0;
  lin_conf_id_ = 0;
  lut_conf_id_ = 0;
  wei_conf_id_ = 0;
  fgr_conf_id_ = 0;
  sli_conf_id_ = 0;
  spi_conf_id_ = 0;  //modif-alex 21/01/11
  del_conf_id_ = 0;  //modif-alex 21/01/11
  bxt_conf_id_ = 0;
  btt_conf_id_ = 0;
  bst_conf_id_ = 0;
  tag_ = "";
  version_ = 0;

  m_write_ped = 1;
  m_write_lin = 1;
  m_write_lut = 1;
  m_write_wei = 1;
  m_write_fgr = 1;
  m_write_sli = 1;
  m_write_spi = 1;  //modif-alex 21/01/11
  m_write_del = 1;  //modif-alex 21/01/11
  m_write_bxt = 1;
  m_write_btt = 1;
  m_write_bst = 1;

  writeToDB_ = pSet.getParameter<bool>("writeToDB");
  DBEE_ = pSet.getParameter<bool>("allowDBEE");
  string DBsid = pSet.getParameter<std::string>("DBsid");
  string DBuser = pSet.getParameter<std::string>("DBuser");
  string DBpass = pSet.getParameter<std::string>("DBpass");
  //uint32_t DBport = pSet.getParameter<unsigned int>("DBport") ;

  tag_ = pSet.getParameter<std::string>("TPGtag");
  version_ = pSet.getParameter<unsigned int>("TPGversion");

  m_write_ped = pSet.getParameter<unsigned int>("TPGWritePed");
  m_write_lin = pSet.getParameter<unsigned int>("TPGWriteLin");
  m_write_lut = pSet.getParameter<unsigned int>("TPGWriteLut");
  m_write_wei = pSet.getParameter<unsigned int>("TPGWriteWei");
  m_write_fgr = pSet.getParameter<unsigned int>("TPGWriteFgr");
  m_write_sli = pSet.getParameter<unsigned int>("TPGWriteSli");
  m_write_spi = pSet.getParameter<unsigned int>("TPGWriteSpi");  //modif-alex 21/01/11
  m_write_del = pSet.getParameter<unsigned int>("TPGWriteDel");  //modif-alex 21/01/11
  m_write_bxt = pSet.getParameter<unsigned int>("TPGWriteBxt");
  m_write_btt = pSet.getParameter<unsigned int>("TPGWriteBtt");
  m_write_bst = pSet.getParameter<unsigned int>("TPGWriteBst");

  btt_conf_id_ = m_write_btt;
  bxt_conf_id_ = m_write_bxt;
  bst_conf_id_ = m_write_bst;

  if (m_write_ped != 0 && m_write_ped != 1)
    ped_conf_id_ = m_write_ped;

  if (writeToDB_)
    edm::LogInfo("TopInfo") << "data will be saved with tag and version=" << tag_ << ".version" << version_ << "\n";
  db_ = new EcalTPGDBApp(DBsid, DBuser, DBpass);

  writeToFiles_ = pSet.getParameter<bool>("writeToFiles");
  if (writeToFiles_) {
    std::string outFile = pSet.getParameter<std::string>("outFile");
    out_file_ = new std::ofstream(outFile.c_str(), std::ios::out);
  }
  geomFile_ = new std::ofstream("geomFile.txt", std::ios::out);

  useTransverseEnergy_ = pSet.getParameter<bool>("useTransverseEnergy");

  Et_sat_EB_ = pSet.getParameter<double>("Et_sat_EB");
  Et_sat_EE_ = pSet.getParameter<double>("Et_sat_EE");
  sliding_ = pSet.getParameter<unsigned int>("sliding");
  weight_timeShift_ = pSet.getParameter<double>("weight_timeShift");
  sampleMax_ = pSet.getParameter<unsigned int>("weight_sampleMax");
  weight_unbias_recovery_ = pSet.getParameter<bool>("weight_unbias_recovery");

  forcedPedestalValue_ = pSet.getParameter<int>("forcedPedestalValue");
  forceEtaSlice_ = pSet.getParameter<bool>("forceEtaSlice");

  LUT_option_ = pSet.getParameter<std::string>("LUT_option");
  LUT_threshold_EB_ = pSet.getParameter<double>("LUT_threshold_EB");
  LUT_threshold_EE_ = pSet.getParameter<double>("LUT_threshold_EE");
  LUT_stochastic_EB_ = pSet.getParameter<double>("LUT_stochastic_EB");
  LUT_noise_EB_ = pSet.getParameter<double>("LUT_noise_EB");
  LUT_constant_EB_ = pSet.getParameter<double>("LUT_constant_EB");
  LUT_stochastic_EE_ = pSet.getParameter<double>("LUT_stochastic_EE");
  LUT_noise_EE_ = pSet.getParameter<double>("LUT_noise_EE");
  LUT_constant_EE_ = pSet.getParameter<double>("LUT_constant_EE");

  TTF_lowThreshold_EB_ = pSet.getParameter<double>("TTF_lowThreshold_EB");
  TTF_highThreshold_EB_ = pSet.getParameter<double>("TTF_highThreshold_EB");
  TTF_lowThreshold_EE_ = pSet.getParameter<double>("TTF_lowThreshold_EE");
  TTF_highThreshold_EE_ = pSet.getParameter<double>("TTF_highThreshold_EE");

  FG_lowThreshold_EB_ = pSet.getParameter<double>("FG_lowThreshold_EB");
  FG_highThreshold_EB_ = pSet.getParameter<double>("FG_highThreshold_EB");
  FG_lowRatio_EB_ = pSet.getParameter<double>("FG_lowRatio_EB");
  FG_highRatio_EB_ = pSet.getParameter<double>("FG_highRatio_EB");
  FG_lut_EB_ = pSet.getParameter<unsigned int>("FG_lut_EB");
  FG_Threshold_EE_ = pSet.getParameter<double>("FG_Threshold_EE");
  FG_lut_strip_EE_ = pSet.getParameter<unsigned int>("FG_lut_strip_EE");
  FG_lut_tower_EE_ = pSet.getParameter<unsigned int>("FG_lut_tower_EE");
  SFGVB_Threshold_ = pSet.getParameter<unsigned int>("SFGVB_Threshold");
  SFGVB_lut_ = pSet.getParameter<unsigned int>("SFGVB_lut");
  SFGVB_SpikeKillingThreshold_ = pSet.getParameter<int>("SFGVB_SpikeKillingThreshold");  //modif-alex 21/01/11
  pedestal_offset_ = pSet.getParameter<unsigned int>("pedestal_offset");

  useInterCalibration_ = pSet.getParameter<bool>("useInterCalibration");
  H2_ = pSet.getUntrackedParameter<bool>("H2", false);

  useTransparencyCorr_ = false;
  useTransparencyCorr_ = pSet.getParameter<bool>("useTransparencyCorr");            //modif-alex-25/04/2012
  Transparency_Corr_ = pSet.getParameter<std::string>("transparency_corrections");  //modif-alex-30/01/2012

  //modif-alex-23/02/2011
  //convert the spike killing first from GeV to ADC (10 bits)
  //depending on the saturation scale: Et_sat_EB_
  if (SFGVB_SpikeKillingThreshold_ == -1 || (SFGVB_SpikeKillingThreshold_ > Et_sat_EB_))
    SFGVB_SpikeKillingThreshold_ = 1023;  //nokilling
  else
    SFGVB_SpikeKillingThreshold_ = int(SFGVB_SpikeKillingThreshold_ * 1024 / Et_sat_EB_);
  edm::LogInfo("TopInfo") << "INFO:SPIKE KILLING THRESHOLD (ADC)=" << SFGVB_SpikeKillingThreshold_ << "\n";

  //modif-alex-02/02/11
  //TIMING information
  TimingDelays_EB_ = pSet.getParameter<std::string>("timing_delays_EB");
  TimingDelays_EE_ = pSet.getParameter<std::string>("timing_delays_EE");
  TimingPhases_EB_ = pSet.getParameter<std::string>("timing_phases_EB");
  TimingPhases_EE_ = pSet.getParameter<std::string>("timing_phases_EE");

  std::ostringstream ss;
  //  edm::LogInfo("TopInfo") << "INFO: READING timing files" << "\n";
  ss << "INFO: READING timing files\n";
  std::ifstream delay_eb(TimingDelays_EB_.c_str());
  if (!delay_eb)
    edm::LogError("TopInfo") << "ERROR: File " << TimingDelays_EB_.c_str() << " could not be opened"
                             << "\n";
  std::ifstream delay_ee(TimingDelays_EE_.c_str());
  if (!delay_ee)
    edm::LogError("TopInfo") << "ERROR: File " << TimingDelays_EE_.c_str() << " could not be opened"
                             << "\n";
  std::ifstream phase_eb(TimingPhases_EB_.c_str());
  if (!phase_eb)
    edm::LogError("TopInfo") << "ERROR: File " << TimingPhases_EB_.c_str() << " could not be opened"
                             << "\n";
  std::ifstream phase_ee(TimingPhases_EE_.c_str());
  if (!phase_ee)
    edm::LogError("TopInfo") << "ERROR: File " << TimingPhases_EE_.c_str() << " could not be opened"
                             << "\n";

  char buf[1024];
  //READING DELAYS EB
  delay_eb.getline(buf, sizeof(buf), '\n');
  while (delay_eb) {
    std::stringstream sin(buf);

    int tcc;
    sin >> tcc;

    vector<int> vec_delays_eb;
    for (int ieb = 0; ieb < 68; ++ieb) {
      int time_delay = -1;
      sin >> time_delay;
      vec_delays_eb.push_back(time_delay);
      if (time_delay == -1)
        edm::LogError("TopInfo") << "ERROR:Barrel timing delay -1, check file"
                                 << "\n";
    }

    if (vec_delays_eb.size() != 68)
      edm::LogError("TopInfo") << "ERROR:Barrel timing delay wrong, not enough towers, check file"
                               << "\n";

    if (delays_EB_.find(tcc) == delays_EB_.end())
      delays_EB_.insert(make_pair(tcc, vec_delays_eb));

    //    edm::LogInfo("TopInfo") << tcc << "\n";
    ss << tcc;
    for (unsigned int ieb = 0; ieb < vec_delays_eb.size(); ++ieb)
      //      edm::LogInfo("TopInfo") << vec_delays_eb[ieb] << "\n";
      ss << " " << vec_delays_eb[ieb];
    delay_eb.getline(buf, sizeof(buf), '\n');
    ss << "\n";
  }  //loop delay file EB
  delay_eb.close();

  //READING PHASES EB
  phase_eb.getline(buf, sizeof(buf), '\n');
  while (phase_eb) {
    std::stringstream sin(buf);
    int tcc;
    sin >> tcc;

    vector<int> vec_phases_eb;
    for (unsigned int ieb = 0; ieb < 68; ++ieb) {
      int time_phase = -1;
      sin >> time_phase;
      vec_phases_eb.push_back(time_phase);
      if (time_phase == -1)
        edm::LogError("TopInfo") << "ERROR:Barrel timing phase -1, check file"
                                 << "\n";
    }

    if (vec_phases_eb.size() != 68)
      edm::LogError("TopInfo") << "ERROR:Barrel timing phase wrong, not enough towers, check file"
                               << "\n";

    if (phases_EB_.find(tcc) == phases_EB_.end())
      phases_EB_.insert(make_pair(tcc, vec_phases_eb));

    //    edm::LogInfo("TopInfo") << tcc << "\n";
    ss << tcc;
    for (unsigned int ieb = 0; ieb < vec_phases_eb.size(); ++ieb)
      //      edm::LogInfo("TopInfo") << vec_phases_eb[ieb] << "\n";
      ss << " " << vec_phases_eb[ieb];
    phase_eb.getline(buf, sizeof(buf), '\n');
    ss << "\n";
  }  //loop phase file EB
  phase_eb.close();

  //READING DELAYS EE//------------------------------------------------
  delay_ee.getline(buf, sizeof(buf), '\n');
  while (delay_ee) {
    std::stringstream sin(buf);
    int tcc;
    sin >> tcc;

    vector<int> vec_delays_ee;
    for (unsigned int iee = 0; iee < 48; ++iee) {
      int time_delay = -1;
      sin >> time_delay;
      vec_delays_ee.push_back(time_delay);
      if (time_delay == -1)
        edm::LogError("TopInfo") << "ERROR:EE timing delay -1, check file"
                                 << "\n";
    }

    if (vec_delays_ee.size() != 48)
      edm::LogError("TopInfo") << "ERROR:EE timing delay wrong, not enough towers, check file"
                               << "\n";

    if (delays_EE_.find(tcc) == delays_EE_.end())
      delays_EE_.insert(make_pair(tcc, vec_delays_ee));

    //    edm::LogInfo("TopInfo") << tcc << "\n";
    ss << tcc;
    for (unsigned int iee = 0; iee < vec_delays_ee.size(); ++iee)
      //      edm::LogInfo("TopInfo") << vec_delays_ee[iee] << "\n";
      ss << " " << vec_delays_ee[iee];
    ss << "\n";
    delay_ee.getline(buf, sizeof(buf), '\n');
  }  //loop delay file EE
  delay_ee.close();

  //READING PHASES EE
  phase_ee.getline(buf, sizeof(buf), '\n');
  while (phase_ee) {
    std::stringstream sin(buf);
    int tcc;
    sin >> tcc;

    vector<int> vec_phases_ee;
    for (unsigned int iee = 0; iee < 48; ++iee) {
      int time_phase = -1;
      sin >> time_phase;
      vec_phases_ee.push_back(time_phase);
      if (time_phase == -1)
        edm::LogError("TopInfo") << "ERROR:EE timing phase -1, check file"
                                 << "\n";
    }

    if (vec_phases_ee.size() != 48)
      edm::LogError("TopInfo") << "ERROR:EE timing phase wrong, not enough towers, check file"
                               << "\n";

    if (phases_EE_.find(tcc) == phases_EE_.end())
      phases_EE_.insert(make_pair(tcc, vec_phases_ee));
    //    edm::LogInfo("TopInfo") << tcc << "\n";
    ss << tcc;
    for (unsigned int iee = 0; iee < vec_phases_ee.size(); ++iee)
      //      edm::LogInfo("TopInfo") << vec_phases_ee[iee] << "\n";
      ss << " " << vec_phases_ee[iee];
    ss << "\n";
    phase_ee.getline(buf, sizeof(buf), '\n');
  }  //loop phase file EE
  phase_ee.close();

  //  edm::LogInfo("TopInfo") << "INFO: DONE reading timing files for EB and EE" << "\n";
  ss << "INFO: DONE reading timing files for EB and EE\n";
  edm::LogInfo("TopInfo") << ss.str();
}

EcalTPGParamBuilder::~EcalTPGParamBuilder() {
  if (writeToFiles_) {
    (*out_file_) << "EOF" << std::endl;
    out_file_->close();
    delete out_file_;
  }
}

bool EcalTPGParamBuilder::checkIfOK(EcalPedestals::Item item) {
  bool result = true;
  if (item.mean_x1 < 150. || item.mean_x1 > 250)
    result = false;
  if (item.mean_x6 < 150. || item.mean_x6 > 250)
    result = false;
  if (item.mean_x12 < 150. || item.mean_x12 > 250)
    result = false;
  if (item.rms_x1 < 0 || item.rms_x1 > 2)
    result = false;
  if (item.rms_x6 < 0 || item.rms_x6 > 3)
    result = false;
  if (item.rms_x12 < 0 || item.rms_x12 > 5)
    result = false;
  return result;
}

int EcalTPGParamBuilder::getEtaSlice(int tccId, int towerInTCC) {
  int etaSlice = (towerInTCC - 1) / 4 + 1;
  // barrel
  if (tccId > 36 || tccId < 73)
    return etaSlice;
  //endcap
  else {
    if (tccId >= 1 && tccId <= 18)
      etaSlice += 21;  // inner -
    if (tccId >= 19 && tccId <= 36)
      etaSlice += 17;  // outer -
    if (tccId >= 91 && tccId <= 108)
      etaSlice += 21;  // inner +
    if (tccId >= 73 && tccId <= 90)
      etaSlice += 17;  // outer +
  }
  return etaSlice;
}

void EcalTPGParamBuilder::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  using namespace edm;
  using namespace std;

  // geometry
  ESHandle<CaloGeometry> theGeometry;
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<CaloGeometryRecord>().get(theGeometry);
  evtSetup.get<EcalEndcapGeometryRecord>().get("EcalEndcap", theEndcapGeometry_handle);
  evtSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel", theBarrelGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get(eTTmap_);
  theEndcapGeometry_ = theEndcapGeometry_handle.product();
  theBarrelGeometry_ = theBarrelGeometry_handle.product();

  // electronics mapping
  ESHandle<EcalElectronicsMapping> ecalmapping;
  evtSetup.get<EcalMappingRcd>().get(ecalmapping);
  theMapping_ = ecalmapping.product();

  // get record for alpha
  std::ostringstream ss;
  ss << "EcalLaserDbAnalyzer::analyze\n";
  edm::ESHandle<EcalLaserAlphas> handle;
  evtSetup.get<EcalLaserAlphasRcd>().get(handle);
  ss << "EcalLaserDbAnalyzer::analyze-> got EcalLaserDbRecord: \n";
  const EcalLaserAlphaMap& laserAlphaMap = handle.product()->getMap();  // map of apdpns

  //modif-alex-27-july-2015-+ Jean june 2016 beg
  // use alpha to check
  EcalLaserAlphaMap::const_iterator italpha;
  int cnt = 0;
  // Barrel loop
  for (italpha = laserAlphaMap.barrelItems().begin(); italpha != laserAlphaMap.barrelItems().end(); ++italpha) {
    if (cnt % 1000 == 0) {
      EBDetId ebdetid = EBDetId::unhashIndex(cnt);
      ss << " Barrel ALPHA = " << (*italpha) << " cmsswId " << ebdetid.rawId() << "\n";
    }
    cnt++;
  }
  ss << "Number of barrel Alpha parameters : " << cnt << "\n";
  // Endcap loop
  cnt = 0;
  for (italpha = laserAlphaMap.endcapItems().begin(); italpha != laserAlphaMap.endcapItems().end(); ++italpha) {
    if (cnt % 1000 == 0) {
      EEDetId eedetid = EEDetId::unhashIndex(cnt);
      ss << "EndCap ALPHA = " << (*italpha) << " cmsswId " << eedetid.rawId() << "\n";
    }
    cnt++;
  }
  ss << "Number of Endcap Alpha parameters : " << cnt << "\n";
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");
  //modif-alex-27-july-2015 +-Jean june 2016-end

  //modif-alex-30/01/2012   displaced in analyze Jean 2018
  ss.str("");
  if (useTransparencyCorr_) {
    if (Transparency_Corr_ != "tag") {  // Jean 2018
      edm::LogInfo("TopInfo") << "INFO: READING transparency correction files"
                              << "\n";
      ss << "INFO: READING transparency correction files\n";
      std::ifstream transparency(Transparency_Corr_.c_str());
      if (!transparency)
        edm::LogError("TopInfo") << "ERROR: File " << Transparency_Corr_.c_str() << " could not be opened"
                                 << "\n";

      char buf[1024];
      transparency.getline(buf, sizeof(buf), '\n');
      int xtalcounter = 0;
      while (transparency) {
        std::stringstream sin(buf);

        int raw_xtal_id;
        sin >> raw_xtal_id;

        double xtal_trans_corr;
        sin >> xtal_trans_corr;

        //      edm::LogInfo("TopInfo") << raw_xtal_id << " " << xtal_trans_corr << "\n";
        ss << raw_xtal_id << " " << xtal_trans_corr << "\n";

        Transparency_Correction_.insert(make_pair(raw_xtal_id, xtal_trans_corr));

        xtalcounter++;
        transparency.getline(buf, sizeof(buf), '\n');
      }  //loop transparency
      transparency.close();
      ss << "INFO: DONE transparency correction files " << xtalcounter << "\n";
      edm::LogInfo("TopInfo") << ss.str();
      ss.str("");
    }       // file
    else {  // Jean 2018
      edm::LogInfo("TopInfo") << "INFO: READING transparency correction tag"
                              << "\n";
      //      std::cout << "new feature, read a tag" << std::endl;
      ESHandle<EcalLaserAPDPNRatios> pAPDPNRatios;
      evtSetup.get<EcalLaserAPDPNRatiosRcd>().get(pAPDPNRatios);
      const EcalLaserAPDPNRatios* lratio = pAPDPNRatios.product();
      //      std::cout << " laser map size " << lratio->getLaserMap().size() << std::endl;

      EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator itratio;
      // Barrel loop
      for (int ib = 0; ib < 61200; ib++) {
        EBDetId ebdetid = EBDetId::unhashIndex(ib);
        itratio = lratio->getLaserMap().find(ebdetid.rawId());
        //	std::cout  << ib << " " << ebdetid.rawId() << " Barrel APDPNRatios = " << (*itratio).p1 << " " << (*itratio).p2 << " " << (*itratio).p3 << std::endl;
        Transparency_Correction_.insert(make_pair(ebdetid.rawId(), (*itratio).p2));
      }
      // Endcap loop
      for (int ie = 0; ie < 14648; ie++) {
        EEDetId eedetid = EEDetId::unhashIndex(ie);
        itratio = lratio->getLaserMap().find(eedetid.rawId());
        //	std::cout  << ie << " " << eedetid.rawId() << " Endcap APDPNRatios = " << (*itratio).p1 << " " << (*itratio).p2 << " " << (*itratio).p3 << std::endl;
        Transparency_Correction_.insert(make_pair(eedetid.rawId(), (*itratio).p2));
      }
      ss << "INFO: DONE transparency correction tag\n";
      edm::LogInfo("TopInfo") << ss.str();
      ss.str("");
    }  // Jean 2018
  }    //if transparency

  // histo
  TFile saving("EcalTPGParam.root", "recreate");
  saving.cd();
  TH2F* ICEB = new TH2F("ICEB", "IC: Barrel", 360, 1, 361, 172, -86, 86);
  ICEB->GetYaxis()->SetTitle("eta index");
  ICEB->GetXaxis()->SetTitle("phi index");
  TH2F* tpgFactorEB = new TH2F("tpgFactorEB", "tpgFactor: Barrel", 360, 1, 361, 172, -86, 86);
  tpgFactorEB->GetYaxis()->SetTitle("eta index");
  tpgFactorEB->GetXaxis()->SetTitle("phi index");

  TH2F* ICEEPlus = new TH2F("ICEEPlus", "IC: Plus Endcap", 120, -9, 111, 120, -9, 111);
  ICEEPlus->GetYaxis()->SetTitle("y index");
  ICEEPlus->GetXaxis()->SetTitle("x index");
  TH2F* tpgFactorEEPlus = new TH2F("tpgFactorEEPlus", "tpgFactor: Plus Endcap", 120, -9, 111, 120, -9, 111);
  tpgFactorEEPlus->GetYaxis()->SetTitle("y index");
  tpgFactorEEPlus->GetXaxis()->SetTitle("x index");
  TH2F* ICEEMinus = new TH2F("ICEEMinus", "IC: Minus Endcap", 120, -9, 111, 120, -9, 111);
  ICEEMinus->GetYaxis()->SetTitle("y index");
  ICEEMinus->GetXaxis()->SetTitle("x index");
  TH2F* tpgFactorEEMinus = new TH2F("tpgFactorEEMinus", "tpgFactor: Minus Endcap", 120, -9, 111, 120, -9, 111);
  tpgFactorEEMinus->GetYaxis()->SetTitle("y index");
  tpgFactorEEMinus->GetXaxis()->SetTitle("x index");

  TH2F* IC = new TH2F("IC", "IC", 720, -acos(-1.), acos(-1.), 600, -3., 3.);
  IC->GetYaxis()->SetTitle("eta");
  IC->GetXaxis()->SetTitle("phi");
  TH2F* tpgFactor = new TH2F("tpgFactor", "tpgFactor", 720, -acos(-1.), acos(-1.), 600, -3., 3.);
  tpgFactor->GetYaxis()->SetTitle("eta");
  tpgFactor->GetXaxis()->SetTitle("phi");

  TH1F* hshapeEB = new TH1F("shapeEB", "shapeEB", 250, 0., 10.);
  TH1F* hshapeEE = new TH1F("shapeEE", "shapeEE", 250, 0., 10.);

  TTree* ntuple = new TTree("tpgmap", "TPG geometry map");
  const std::string branchFloat[26] = {
      "fed",    "tcc",    "tower", "stripInTower", "xtalInStrip", "CCU",      "VFE",     "xtalInVFE", "xtalInCCU",
      "ieta",   "iphi",   "ix",    "iy",           "iz",          "hashedId", "ic",      "cmsswId",   "dbId",
      "ietaTT", "iphiTT", "TCCch", "TCCslot",      "SLBch",       "SLBslot",  "ietaGCT", "iphiGCT"};
  ntupleInts_ = new Int_t[26];
  for (int i = 0; i < 26; i++)
    ntuple->Branch(branchFloat[i].c_str(), &ntupleInts_[i], (branchFloat[i] + string("/I")).c_str());
  ntuple->Branch("det", ntupleDet_, "det/C");
  ntuple->Branch("crate", ntupleCrate_, "crate/C");

  TNtuple* ntupleSpike = new TNtuple("spikeParam", "Spike parameters", "gainId:theta:G:g:ped:pedLin");

  ////////////////////////////
  // Initialization section //
  ////////////////////////////
  list<uint32_t> towerListEB;
  list<uint32_t> stripListEB;
  list<uint32_t> towerListEE;
  list<uint32_t> stripListEE;
  list<uint32_t>::iterator itList;

  map<int, uint32_t> stripMapEB;               // <EcalLogicId.hashed, strip elec id>
  map<uint32_t, uint32_t> stripMapEBsintheta;  // <strip elec id, sintheta>

  // Pedestals

  EcalPedestalsMap pedMap;

  if (m_write_ped == 1) {
    ss << "Getting the pedestals from offline DB...\n";

    ESHandle<EcalPedestals> pedHandle;
    evtSetup.get<EcalPedestalsRcd>().get(pedHandle);
    pedMap = pedHandle.product()->getMap();

    EcalPedestalsMapIterator pedIter;
    int nPed = 0;
    for (pedIter = pedMap.begin(); pedIter != pedMap.end() && nPed < 10; ++pedIter, nPed++) {
      EcalPedestals::Item aped = (*pedIter);
      ss << aped.mean_x12 << ", " << aped.mean_x6 << ", " << aped.mean_x1 << "\n";
    }
  } else if (m_write_ped == 0) {
    ss << "Getting the pedestals from previous configuration\n";

    EcalPedestals peds;

    FEConfigMainInfo fe_main_info;
    fe_main_info.setConfigTag(tag_);
    ss << "trying to read previous tag if it exists tag=" << tag_ << ".version" << version_ << "\n";
    db_->fetchConfigSet(&fe_main_info);
    if (fe_main_info.getPedId() > 0)
      ped_conf_id_ = fe_main_info.getPedId();

    FEConfigPedInfo fe_ped_info;
    fe_ped_info.setId(ped_conf_id_);
    db_->fetchConfigSet(&fe_ped_info);
    std::map<EcalLogicID, FEConfigPedDat> dataset_TpgPed;
    db_->fetchDataSet(&dataset_TpgPed, &fe_ped_info);

    typedef std::map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
    EcalLogicID ecid_xt;
    FEConfigPedDat rd_ped;
    int icells = 0;
    for (CIfeped p = dataset_TpgPed.begin(); p != dataset_TpgPed.end(); p++) {
      ecid_xt = p->first;
      rd_ped = p->second;

      std::string ecid_name = ecid_xt.getName();

      // EB data
      if (ecid_name == "EB_crystal_number") {
        int sm_num = ecid_xt.getID1();
        int xt_num = ecid_xt.getID2();

        EBDetId ebdetid(sm_num, xt_num, EBDetId::SMCRYSTALMODE);
        EcalPedestals::Item item;
        item.mean_x1 = rd_ped.getPedMeanG1();
        item.mean_x6 = rd_ped.getPedMeanG6();
        item.mean_x12 = rd_ped.getPedMeanG12();
        item.rms_x1 = 0.5;
        item.rms_x6 = 1.;
        item.rms_x12 = 1.2;

        if (icells < 10)
          ss << " copy the EB data "
             << " ped = " << item.mean_x12 << "\n";

        peds.insert(std::make_pair(ebdetid.rawId(), item));

        ++icells;
      } else if (ecid_name == "EE_crystal_number") {
        // EE data
        int z = ecid_xt.getID1();
        int x = ecid_xt.getID2();
        int y = ecid_xt.getID3();
        EEDetId eedetid(x, y, z, EEDetId::XYMODE);
        EcalPedestals::Item item;
        item.mean_x1 = rd_ped.getPedMeanG1();
        item.mean_x6 = rd_ped.getPedMeanG6();
        item.mean_x12 = rd_ped.getPedMeanG12();
        item.rms_x1 = 0.5;
        item.rms_x6 = 1.;
        item.rms_x12 = 1.2;

        peds.insert(std::make_pair(eedetid.rawId(), item));
        ++icells;
      }
    }

    pedMap = peds.getMap();

    EcalPedestalsMapIterator pedIter;
    int nPed = 0;
    for (pedIter = pedMap.begin(); pedIter != pedMap.end() && nPed < 10; ++pedIter, nPed++) {
      EcalPedestals::Item aped = (*pedIter);
      ss << aped.mean_x12 << ", " << aped.mean_x6 << ", " << aped.mean_x1 << "\n";
    }

  } else if (m_write_ped > 1) {
    ss << "Getting the pedestals from configuration number" << m_write_ped << "\n";

    EcalPedestals peds;

    FEConfigPedInfo fe_ped_info;
    fe_ped_info.setId(m_write_ped);
    db_->fetchConfigSet(&fe_ped_info);
    std::map<EcalLogicID, FEConfigPedDat> dataset_TpgPed;
    db_->fetchDataSet(&dataset_TpgPed, &fe_ped_info);

    typedef std::map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
    EcalLogicID ecid_xt;
    FEConfigPedDat rd_ped;
    int icells = 0;
    for (CIfeped p = dataset_TpgPed.begin(); p != dataset_TpgPed.end(); p++) {
      ecid_xt = p->first;
      rd_ped = p->second;

      std::string ecid_name = ecid_xt.getName();

      // EB data
      if (ecid_name == "EB_crystal_number") {
        if (icells < 10)
          edm::LogInfo("TopInfo") << " copy the EB data "
                                  << " icells = " << icells << "\n";
        int sm_num = ecid_xt.getID1();
        int xt_num = ecid_xt.getID2();

        EBDetId ebdetid(sm_num, xt_num, EBDetId::SMCRYSTALMODE);
        EcalPedestals::Item item;
        item.mean_x1 = (unsigned int)rd_ped.getPedMeanG1();
        item.mean_x6 = (unsigned int)rd_ped.getPedMeanG6();
        item.mean_x12 = (unsigned int)rd_ped.getPedMeanG12();
        item.rms_x1 = 0.5;
        item.rms_x6 = 1.;
        item.rms_x12 = 1.2;

        peds.insert(std::make_pair(ebdetid.rawId(), item));
        ++icells;
      } else if (ecid_name == "EE_crystal_number") {
        // EE data
        int z = ecid_xt.getID1();
        int x = ecid_xt.getID2();
        int y = ecid_xt.getID3();
        EEDetId eedetid(x, y, z, EEDetId::XYMODE);
        EcalPedestals::Item item;
        item.mean_x1 = (unsigned int)rd_ped.getPedMeanG1();
        item.mean_x6 = (unsigned int)rd_ped.getPedMeanG6();
        item.mean_x12 = (unsigned int)rd_ped.getPedMeanG12();
        item.rms_x1 = 0.5;
        item.rms_x6 = 1.;
        item.rms_x12 = 1.2;

        peds.insert(std::make_pair(eedetid.rawId(), item));
        ++icells;
      }
    }

    pedMap = peds.getMap();

    EcalPedestalsMapIterator pedIter;
    int nPed = 0;
    for (pedIter = pedMap.begin(); pedIter != pedMap.end() && nPed < 10; ++pedIter, nPed++) {
      EcalPedestals::Item aped = (*pedIter);
      ss << aped.mean_x12 << ", " << aped.mean_x6 << ", " << aped.mean_x1 << "\n";
    }
  }

  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  // Intercalib constants
  ss << "Getting intercalib from offline DB...\n";
  ESHandle<EcalIntercalibConstants> pIntercalib;
  evtSetup.get<EcalIntercalibConstantsRcd>().get(pIntercalib);
  const EcalIntercalibConstants* intercalib = pIntercalib.product();
  const EcalIntercalibConstantMap& calibMap = intercalib->getMap();
  EcalIntercalibConstantMap::const_iterator calIter;
  int nCal = 0;
  for (calIter = calibMap.begin(); calIter != calibMap.end() && nCal < 10; ++calIter, nCal++) {
    ss << (*calIter) << "\n";
  }
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");
  float calibvec[1700];
  if (H2_) {
    edm::LogInfo("TopInfo") << "H2: overwriting IC coef with file"
                            << "\n";
    std::ifstream calibfile("calib_sm36.txt", std::ios::out);
    int idata, icry;
    float fdata, fcali;
    std::string strdata;
    if (calibfile.is_open()) {
      calibfile >> strdata >> strdata >> strdata >> strdata >> strdata;
      while (!calibfile.eof()) {
        calibfile >> idata >> icry >> fcali >> fdata >> fdata;
        calibvec[icry - 1] = fcali;
        if (calibfile.eof()) {
          break;  // avoid last line duplication
        }
      }
    }
  }

  // Gain Ratios
  ss << "Getting the gain ratios from offline DB...\n";
  ESHandle<EcalGainRatios> pRatio;
  evtSetup.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatioMap& gainMap = pRatio.product()->getMap();
  EcalGainRatioMap::const_iterator gainIter;
  int nGain = 0;
  for (gainIter = gainMap.begin(); gainIter != gainMap.end() && nGain < 10; ++gainIter, nGain++) {
    const EcalMGPAGainRatio& aGain = (*gainIter);
    ss << aGain.gain12Over6() << ", " << aGain.gain6Over1() * aGain.gain12Over6() << "\n";
  }
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  // ADCtoGeV
  ss << "Getting the ADC to GeV from offline DB...\n";
  ESHandle<EcalADCToGeVConstant> pADCToGeV;
  evtSetup.get<EcalADCToGeVConstantRcd>().get(pADCToGeV);
  const EcalADCToGeVConstant* ADCToGeV = pADCToGeV.product();
  xtal_LSB_EB_ = ADCToGeV->getEBValue();
  xtal_LSB_EE_ = ADCToGeV->getEEValue();
  ss << "xtal_LSB_EB_ = " << xtal_LSB_EB_ << "\n";
  ss << "xtal_LSB_EE_ = " << xtal_LSB_EE_ << "\n";

  vector<EcalLogicID> my_EcalLogicId;
  vector<EcalLogicID> my_TTEcalLogicId;
  vector<EcalLogicID> my_StripEcalLogicId;
  EcalLogicID my_EcalLogicId_EB;
  // Endcap identifiers
  EcalLogicID my_EcalLogicId_EE;
  vector<EcalLogicID> my_TTEcalLogicId_EE;
  vector<EcalLogicID> my_RTEcalLogicId_EE;
  vector<EcalLogicID> my_StripEcalLogicId1_EE;
  vector<EcalLogicID> my_StripEcalLogicId2_EE;
  vector<EcalLogicID> my_CrystalEcalLogicId_EE;
  vector<EcalLogicID> my_TTEcalLogicId_EB_by_TCC;
  vector<EcalLogicID> my_StripEcalLogicId_EE_strips_by_TCC;

  my_EcalLogicId_EB = db_->getEcalLogicID("EB", EcalLogicID::NULLID, EcalLogicID::NULLID, EcalLogicID::NULLID, "EB");
  my_EcalLogicId_EE = db_->getEcalLogicID("EE", EcalLogicID::NULLID, EcalLogicID::NULLID, EcalLogicID::NULLID, "EE");
  my_EcalLogicId = db_->getEcalLogicIDSetOrdered(
      "EB_crystal_number", 1, 36, 1, 1700, EcalLogicID::NULLID, EcalLogicID::NULLID, "EB_crystal_number", 12);
  my_TTEcalLogicId = db_->getEcalLogicIDSetOrdered(
      "EB_trigger_tower", 1, 36, 1, 68, EcalLogicID::NULLID, EcalLogicID::NULLID, "EB_trigger_tower", 12);
  my_StripEcalLogicId = db_->getEcalLogicIDSetOrdered(
      "EB_VFE", 1, 36, 1, 68, 1, 5, "EB_VFE", 123);  //last digi means ordered 1st by SM,then TT, then strip
  ss << "got the 3 ecal barrel logic id set\n";

  // EE crystals identifiers
  my_CrystalEcalLogicId_EE =
      db_->getEcalLogicIDSetOrdered("EE_crystal_number", -1, 1, 0, 200, 0, 200, "EE_crystal_number", 123);

  // EE Strip identifiers
  // DCC=601-609 TT = ~40 EEstrip = 5
  my_StripEcalLogicId1_EE =
      db_->getEcalLogicIDSetOrdered("ECAL_readout_strip", 601, 609, 1, 100, 0, 5, "ECAL_readout_strip", 123);
  // EE Strip identifiers
  // DCC=646-654 TT = ~40 EEstrip = 5
  my_StripEcalLogicId2_EE =
      db_->getEcalLogicIDSetOrdered("ECAL_readout_strip", 646, 654, 1, 100, 0, 5, "ECAL_readout_strip", 123);
  // ----> modif here 31/1/2011
  // EE Strip identifiers by TCC tower strip
  // TCC=1-108 TT = ~40 EEstrip = 1-5
  my_StripEcalLogicId_EE_strips_by_TCC =
      db_->getEcalLogicIDSetOrdered("EE_trigger_strip", 1, 108, 1, 100, 1, 5, "EE_trigger_strip", 123);

  // ----> modif here 31/1/2011
  // ECAL Barrel trigger towers by TCC/tower identifiers
  // TTC=38-72 TT = 1-68
  my_TTEcalLogicId_EB_by_TCC = db_->getEcalLogicIDSetOrdered(
      "ECAL_trigger_tower", 1, 108, 1, 68, EcalLogicID::NULLID, EcalLogicID::NULLID, "ECAL_trigger_tower", 12);

  // EE TT identifiers
  // TTC=72 TT = 1440
  my_TTEcalLogicId_EE = db_->getEcalLogicIDSetOrdered(
      "EE_trigger_tower", 1, 108, 1, 40, EcalLogicID::NULLID, EcalLogicID::NULLID, "EE_trigger_tower", 12);

  // EE TT identifiers
  // TTC=72 TT = 1440
  my_RTEcalLogicId_EE = db_->getEcalLogicIDSetOrdered(
      "EE_readout_tower", 1, 1000, 1, 100, EcalLogicID::NULLID, EcalLogicID::NULLID, "EE_readout_tower", 12);
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  if (writeToDB_) {
    ss << "Getting the latest ids for this tag (latest version) "
       << "\n";

    FEConfigMainInfo fe_main_info;
    fe_main_info.setConfigTag(tag_);

    ss << "trying to read previous tag if it exists tag=" << tag_ << ".version" << version_ << "\n";

    db_->fetchConfigSet(&fe_main_info);
    if (fe_main_info.getPedId() > 0 && ped_conf_id_ == 0)
      ped_conf_id_ = fe_main_info.getPedId();
    lin_conf_id_ = fe_main_info.getLinId();
    lut_conf_id_ = fe_main_info.getLUTId();
    wei_conf_id_ = fe_main_info.getWeiId();
    fgr_conf_id_ = fe_main_info.getFgrId();
    sli_conf_id_ = fe_main_info.getSliId();
    spi_conf_id_ = fe_main_info.getSpiId();  //modif-alex 21/01/11
    del_conf_id_ = fe_main_info.getTimId();  //modif-alex 21/01/11
    if (fe_main_info.getBxtId() > 0 && bxt_conf_id_ == 0)
      bxt_conf_id_ = fe_main_info.getBxtId();
    if (fe_main_info.getBttId() > 0 && btt_conf_id_ == 0)
      btt_conf_id_ = fe_main_info.getBttId();
    if (fe_main_info.getBstId() > 0 && bst_conf_id_ == 0)
      bst_conf_id_ = fe_main_info.getBstId();
    // those that are not written specifically in this program are propagated
    // from the previous record with the same tag and the highest version
  }

  /////////////////////////////////////////
  // Compute linearization coeff section //
  /////////////////////////////////////////

  map<EcalLogicID, FEConfigPedDat> pedset;
  map<EcalLogicID, FEConfigLinDat> linset;
  map<EcalLogicID, FEConfigLinParamDat> linparamset;
  map<EcalLogicID, FEConfigLUTParamDat> lutparamset;
  map<EcalLogicID, FEConfigFgrParamDat> fgrparamset;

  map<int, linStruc> linEtaSlice;
  map<vector<int>, linStruc> linMap;

  // count number of strip per tower
  int NbOfStripPerTCC[108][68];
  for (int i = 0; i < 108; i++)
    for (int j = 0; j < 68; j++)
      NbOfStripPerTCC[i][j] = 0;
  const std::vector<DetId>& ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  const std::vector<DetId>& eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  for (vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it);
    const EcalTrigTowerDetId towid = id.tower();
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    int tccNb = theMapping_->TCCid(towid);
    int towerInTCC = theMapping_->iTT(towid);
    int stripInTower = elId.pseudoStripId();
    if (stripInTower > NbOfStripPerTCC[tccNb - 1][towerInTCC - 1])
      NbOfStripPerTCC[tccNb - 1][towerInTCC - 1] = stripInTower;
  }
  for (vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    int tccNb = theMapping_->TCCid(towid);
    int towerInTCC = theMapping_->iTT(towid);
    int stripInTower = elId.pseudoStripId();
    if (stripInTower > NbOfStripPerTCC[tccNb - 1][towerInTCC - 1])
      NbOfStripPerTCC[tccNb - 1][towerInTCC - 1] = stripInTower;
  }

  // loop on EB xtals
  if (writeToFiles_)
    (*out_file_) << "COMMENT ====== barrel crystals ====== " << std::endl;

  // special case of eta slices
  for (vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it);
    double theta = theBarrelGeometry_->getGeometry(id)->getPosition().theta();
    if (!useTransverseEnergy_)
      theta = acos(0.);
    const EcalTrigTowerDetId towid = id.tower();
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    int dccNb = theMapping_->DCCid(towid);
    int tccNb = theMapping_->TCCid(towid);
    int towerInTCC = theMapping_->iTT(towid);  // from 1 to 68 (EB)
    int stripInTower = elId.pseudoStripId();   // from 1 to 5
    int xtalInStrip = elId.channelId();        // from 1 to 5
    const EcalElectronicsId Id = theMapping_->getElectronicsId(id);
    int CCUid = Id.towerId();
    int VFEid = Id.stripId();
    int xtalInVFE = Id.xtalId();
    int xtalWithinCCUid = 5 * (VFEid - 1) + xtalInVFE - 1;  // Evgueni expects [0,24]

    (*geomFile_) << "dccNb = " << dccNb << " tccNb = " << tccNb << " towerInTCC = " << towerInTCC
                 << " stripInTower = " << stripInTower << " xtalInStrip = " << xtalInStrip << " CCUid = " << CCUid
                 << " VFEid = " << VFEid << " xtalInVFE = " << xtalInVFE << " xtalWithinCCUid = " << xtalWithinCCUid
                 << " ieta = " << id.ieta() << " iphi = " << id.iphi() << " xtalhashedId = " << id.hashedIndex()
                 << " xtalNb = " << id.ic() << " ietaTT = " << towid.ieta() << " iphiTT = " << towid.iphi() << endl;

    int TCCch = towerInTCC;
    int SLBslot = int((towerInTCC - 1) / 8.) + 1;
    int SLBch = (towerInTCC - 1) % 8 + 1;
    int cmsswId = id.rawId();
    int ixtal = (id.ism() - 1) * 1700 + (id.ic() - 1);
    EcalLogicID logicId = my_EcalLogicId[ixtal];
    int dbId = logicId.getLogicID();
    int val[] = {dccNb + 600,
                 tccNb,
                 towerInTCC,
                 stripInTower,
                 xtalInStrip,
                 CCUid,
                 VFEid,
                 xtalInVFE,
                 xtalWithinCCUid,
                 id.ieta(),
                 id.iphi(),
                 -999,
                 -999,
                 towid.ieta() / abs(towid.ieta()),
                 id.hashedIndex(),
                 id.ic(),
                 cmsswId,
                 dbId,
                 towid.ieta(),
                 towid.iphi(),
                 TCCch,
                 getCrate(tccNb).second,
                 SLBch,
                 SLBslot,
                 getGCTRegionEta(towid.ieta()),
                 getGCTRegionPhi(towid.iphi())};
    for (int i = 0; i < 26; i++)
      ntupleInts_[i] = val[i];

    strcpy(ntupleDet_, getDet(tccNb).c_str());
    strcpy(ntupleCrate_, getCrate(tccNb).first.c_str());
    ntuple->Fill();

    if (tccNb == 37 && stripInTower == 3 && xtalInStrip == 3 && (towerInTCC - 1) % 4 == 0) {
      int etaSlice = towid.ietaAbs();
      coeffStruc coeff;
      //getCoeff(coeff, calibMap, id.rawId()) ;
      //modif-alex-27-july-2015
      string str;
      getCoeff(coeff, calibMap, laserAlphaMap, id.rawId(), str);
      ss << str;
      getCoeff(coeff, gainMap, id.rawId());
      getCoeff(coeff, pedMap, id.rawId());
      linStruc lin;
      for (int i = 0; i < 3; i++) {
        int mult, shift;
        bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EB", mult, shift);
        if (!ok)
          edm::LogError("TopInfo") << "unable to compute the parameters for SM=" << id.ism() << " xt=" << id.ic() << " "
                                   << dec << id.rawId() << "\n";
        else {
          lin.pedestal_[i] = coeff.pedestals_[i];
          lin.mult_[i] = mult;
          lin.shift_[i] = shift;
        }
      }

      bool ok(true);
      if (forcedPedestalValue_ == -2)
        ok = realignBaseline(lin, 0);
      if (!ok)
        ss << "SM=" << id.ism() << " xt=" << id.ic() << " " << dec << id.rawId() << "\n";
      linEtaSlice[etaSlice] = lin;
    }
  }

  // general case
  for (vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it);
    double theta = theBarrelGeometry_->getGeometry(id)->getPosition().theta();
    if (!useTransverseEnergy_)
      theta = acos(0.);
    const EcalTrigTowerDetId towid = id.tower();
    towerListEB.push_back(towid.rawId());
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    stripListEB.push_back(elId.rawId() & 0xfffffff8);
    int dccNb = theMapping_->DCCid(towid);
    //int tccNb = theMapping_->TCCid(towid) ;
    int towerInTCC = theMapping_->iTT(towid);  // from 1 to 68 (EB)
    //int stripInTower = elId.pseudoStripId() ;  // from 1 to 5
    //int xtalInStrip = elId.channelId() ;       // from 1 to 5
    const EcalElectronicsId Id = theMapping_->getElectronicsId(id);
    int CCUid = Id.towerId();
    int VFEid = Id.stripId();
    int xtalInVFE = Id.xtalId();
    int xtalWithinCCUid = 5 * (VFEid - 1) + xtalInVFE - 1;  // Evgueni expects [0,24]
    int etaSlice = towid.ietaAbs();

    // hashed index of strip EcalLogicID:
    int hashedStripLogicID = 68 * 5 * (id.ism() - 1) + 5 * (towerInTCC - 1) + (VFEid - 1);
    stripMapEB[hashedStripLogicID] = elId.rawId() & 0xfffffff8;
    stripMapEBsintheta[elId.rawId() & 0xfffffff8] = SFGVB_Threshold_ + abs(int(sin(theta) * pedestal_offset_));

    //modif-debug
    /*
    FEConfigFgrEEStripDat stripdebug;
    EcalLogicID thestrip_debug = my_StripEcalLogicId[hashedStripLogicID] ;
    if(towid.ieta() == 6 && towid.iphi() == 7){
      std::cout << "xtal info=" << id << " VFE=" << VFEid << std::endl;
      std::cout << "TOWER DEBUG ieta=" << towid.ieta() << " iphi=" << towid.iphi() << " SFGVB=" << SFGVB_Threshold_ + abs(int(sin(theta)*pedestal_offset_)) 
		<< " dbId=" << (elId.rawId() & 0xfffffff8) << " " << hashedStripLogicID << " " << thestrip_debug.getLogicID() << std::endl;  //modif-debug
    }//EB+3 TT24
    */
    //std::cout<<std::dec<<SFGVB_Threshold_ + abs(int(sin(theta)*pedestal_offset_))<<" "<<SFGVB_Threshold_<<" "<<abs(int(sin(theta)*pedestal_offset_))<<std::endl ;

    FEConfigPedDat pedDB;
    FEConfigLinDat linDB;
    if (writeToFiles_)
      (*out_file_) << "CRYSTAL " << dec << id.rawId() << std::endl;
    //  if (writeToDB_) logicId = db_->getEcalLogicID ("EB_crystal_number", id.ism(), id.ic()) ;

    coeffStruc coeff;
    //getCoeff(coeff, calibMap, id.rawId()) ;
    //modif-alex-27-july-2015
    std::string str;
    getCoeff(coeff, calibMap, laserAlphaMap, id.rawId(), str);
    ss << str;

    if (H2_)
      coeff.calibCoeff_ = calibvec[id.ic() - 1];
    getCoeff(coeff, gainMap, id.rawId());
    getCoeff(coeff, pedMap, id.rawId());
    ICEB->Fill(id.iphi(), id.ieta(), coeff.calibCoeff_);
    IC->Fill(theBarrelGeometry_->getGeometry(id)->getPosition().phi(),
             theBarrelGeometry_->getGeometry(id)->getPosition().eta(),
             coeff.calibCoeff_);

    vector<int> xtalCCU;
    xtalCCU.push_back(dccNb + 600);
    xtalCCU.push_back(CCUid);
    xtalCCU.push_back(xtalWithinCCUid);
    xtalCCU.push_back(id.rawId());

    // compute and fill linearization parameters
    // case of eta slice
    if (forceEtaSlice_) {
      map<int, linStruc>::const_iterator itLin = linEtaSlice.find(etaSlice);
      if (itLin != linEtaSlice.end()) {
        linMap[xtalCCU] = itLin->second;
        if (writeToFiles_) {
          for (int i = 0; i < 3; i++)
            (*out_file_) << hex << " 0x" << itLin->second.pedestal_[i] << " 0x" << itLin->second.mult_[i] << " 0x"
                         << itLin->second.shift_[i] << std::endl;
        }
        if (writeToDB_) {
          for (int i = 0; i < 3; i++) {
            if (i == 0) {
              pedDB.setPedMeanG12(itLin->second.pedestal_[i]);
              linDB.setMultX12(itLin->second.mult_[i]);
              linDB.setShift12(itLin->second.shift_[i]);
            }
            if (i == 1) {
              pedDB.setPedMeanG6(itLin->second.pedestal_[i]);
              linDB.setMultX6(itLin->second.mult_[i]);
              linDB.setShift6(itLin->second.shift_[i]);
            }
            if (i == 2) {
              pedDB.setPedMeanG1(itLin->second.pedestal_[i]);
              linDB.setMultX1(itLin->second.mult_[i]);
              linDB.setShift1(itLin->second.shift_[i]);
            }
          }
        }
        float factor = float(itLin->second.mult_[0]) * pow(2., -itLin->second.shift_[0]) / xtal_LSB_EB_;
        tpgFactorEB->Fill(id.iphi(), id.ieta(), factor);
        tpgFactor->Fill(theBarrelGeometry_->getGeometry(id)->getPosition().phi(),
                        theBarrelGeometry_->getGeometry(id)->getPosition().eta(),
                        factor);
      } else
        ss << "current EtaSlice = " << etaSlice << " not found in the EtaSlice map"
           << "\n";
    } else {
      // general case
      linStruc lin;
      int forceBase12 = 0;
      for (int i = 0; i < 3; i++) {
        int mult, shift;
        bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EB", mult, shift);
        if (!ok)
          edm::LogError("TopInfo") << "unable to compute the parameters for SM=" << id.ism() << " xt=" << id.ic() << " "
                                   << dec << id.rawId() << "\n";
        else {
          //PP begin
          //  mult = 0 ; shift = 0 ;
          //  if (CCUid==1 && xtalWithinCCUid==21) {
          //    if (i==0) {mult = 0x80 ; shift = 0x3 ;}
          //    if (i==1) {mult = 0x80 ; shift = 0x2 ;}
          //    if (i==2) {mult = 0xc0 ; shift = 0x0 ;}
          //  }
          //PP end
          double base = coeff.pedestals_[i];
          if (forcedPedestalValue_ == -3 && i == 0) {
            double G = mult * pow(2.0, -(shift + 2));
            double g = G / sin(theta);
            // int pedestal = coeff.pedestals_[i] ;
            base = double(coeff.pedestals_[i]) - pedestal_offset_ / g;
            if (base < 0.)
              base = 0;
            forceBase12 = int(base);
          }
          lin.pedestal_[i] = coeff.pedestals_[i];
          lin.mult_[i] = mult;
          lin.shift_[i] = shift;

          // 	  if (xtalWithinCCUid != 14) {
          // 	    forceBase12 = 0 ;
          // 	    lin.pedestal_[i] = 0 ;
          // 	    lin.mult_[i] = 0 ;
          // 	    lin.shift_[i] = 0 ;
          // 	  }
        }
      }

      bool ok(true);
      if (forcedPedestalValue_ == -2)
        ok = realignBaseline(lin, 0);
      if (forcedPedestalValue_ == -3)
        ok = realignBaseline(lin, forceBase12);
      if (!ok)
        ss << "SM=" << id.ism() << " xt=" << id.ic() << " " << dec << id.rawId() << "\n";

      for (int i = 0; i < 3; i++) {
        if (writeToFiles_)
          (*out_file_) << hex << " 0x" << lin.pedestal_[i] << " 0x" << lin.mult_[i] << " 0x" << lin.shift_[i]
                       << std::endl;
        if (writeToDB_) {
          if (i == 0) {
            pedDB.setPedMeanG12(lin.pedestal_[i]);
            linDB.setMultX12(lin.mult_[i]);
            linDB.setShift12(lin.shift_[i]);
          }
          if (i == 1) {
            pedDB.setPedMeanG6(lin.pedestal_[i]);
            linDB.setMultX6(lin.mult_[i]);
            linDB.setShift6(lin.shift_[i]);
          }
          if (i == 2) {
            pedDB.setPedMeanG1(lin.pedestal_[i]);
            linDB.setMultX1(lin.mult_[i]);
            linDB.setShift1(lin.shift_[i]);
          }
        }
        if (i == 0) {
          float factor = float(lin.mult_[i]) * pow(2., -lin.shift_[i]) / xtal_LSB_EB_;
          tpgFactorEB->Fill(id.iphi(), id.ieta(), factor);
          tpgFactor->Fill(theBarrelGeometry_->getGeometry(id)->getPosition().phi(),
                          theBarrelGeometry_->getGeometry(id)->getPosition().eta(),
                          factor);
        }
        double G = lin.mult_[i] * pow(2.0, -(lin.shift_[i] + 2));
        double g = G / sin(theta);
        float val[] = {float(i),
                       float(theta),
                       float(G),
                       float(g),
                       float(coeff.pedestals_[i]),
                       float(lin.pedestal_[i])};  // first arg = gainId (0 means gain12)
        ntupleSpike->Fill(val);
      }
      linMap[xtalCCU] = lin;
    }
    if (writeToDB_) {
      // 1700 crystals/SM in the ECAL barrel
      int ixtal = (id.ism() - 1) * 1700 + (id.ic() - 1);
      EcalLogicID logicId = my_EcalLogicId[ixtal];
      pedset[logicId] = pedDB;
      linset[logicId] = linDB;
    }
  }  //ebCells

  if (writeToDB_) {
    // EcalLogicID  of the whole barrel is: my_EcalLogicId_EB
    FEConfigLinParamDat linparam;
    linparam.setETSat(Et_sat_EB_);
    linparamset[my_EcalLogicId_EB] = linparam;

    FEConfigFgrParamDat fgrparam;
    fgrparam.setFGlowthresh(FG_lowThreshold_EB_);
    fgrparam.setFGhighthresh(FG_highThreshold_EB_);
    fgrparam.setFGlowratio(FG_lowRatio_EB_);
    fgrparam.setFGhighratio(FG_highRatio_EB_);
    fgrparamset[my_EcalLogicId_EB] = fgrparam;

    FEConfigLUTParamDat lutparam;
    lutparam.setETSat(Et_sat_EB_);
    lutparam.setTTThreshlow(TTF_lowThreshold_EB_);
    lutparam.setTTThreshhigh(TTF_highThreshold_EB_);
    lutparamset[my_EcalLogicId_EB] = lutparam;
  }

  // loop on EE xtals
  if (writeToFiles_)
    (*out_file_) << "COMMENT ====== endcap crystals ====== " << std::endl;

  // special case of eta slices
  for (vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    double theta = theEndcapGeometry_->getGeometry(id)->getPosition().theta();
    if (!useTransverseEnergy_)
      theta = acos(0.);
    const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    const EcalElectronicsId Id = theMapping_->getElectronicsId(id);
    int dccNb = Id.dccId();
    int tccNb = theMapping_->TCCid(towid);
    int towerInTCC = theMapping_->iTT(towid);
    int stripInTower = elId.pseudoStripId();
    int xtalInStrip = elId.channelId();
    int CCUid = Id.towerId();
    int VFEid = Id.stripId();
    int xtalInVFE = Id.xtalId();
    int xtalWithinCCUid = 5 * (VFEid - 1) + xtalInVFE - 1;  // Evgueni expects [0,24]

    (*geomFile_) << "dccNb = " << dccNb << " tccNb = " << tccNb << " towerInTCC = " << towerInTCC
                 << " stripInTower = " << stripInTower << " xtalInStrip = " << xtalInStrip << " CCUid = " << CCUid
                 << " VFEid = " << VFEid << " xtalInVFE = " << xtalInVFE << " xtalWithinCCUid = " << xtalWithinCCUid
                 << " ix = " << id.ix() << " iy = " << id.iy() << " xtalhashedId = " << id.hashedIndex()
                 << " xtalNb = " << id.isc() << " ietaTT = " << towid.ieta() << " iphiTT = " << towid.iphi() << endl;

    int TCCch = stripInTower;
    int SLBslot, SLBch;
    if (towerInTCC < 5) {
      SLBslot = 1;
      SLBch = 4 + towerInTCC;
    } else {
      SLBslot = int((towerInTCC - 5) / 8.) + 2;
      SLBch = (towerInTCC - 5) % 8 + 1;
    }
    for (int j = 0; j < towerInTCC - 1; j++)
      TCCch += NbOfStripPerTCC[tccNb - 1][j];

    int cmsswId = id.rawId();
    EcalLogicID logicId;
    int iz = id.positiveZ();
    if (iz == 0)
      iz = -1;
    for (int k = 0; k < (int)my_CrystalEcalLogicId_EE.size(); k++) {
      int z = my_CrystalEcalLogicId_EE[k].getID1();
      int x = my_CrystalEcalLogicId_EE[k].getID2();
      int y = my_CrystalEcalLogicId_EE[k].getID3();
      if (id.ix() == x && id.iy() == y && iz == z)
        logicId = my_CrystalEcalLogicId_EE[k];
    }
    int dbId = logicId.getLogicID();

    int val[] = {dccNb + 600,
                 tccNb,
                 towerInTCC,
                 stripInTower,
                 xtalInStrip,
                 CCUid,
                 VFEid,
                 xtalInVFE,
                 xtalWithinCCUid,
                 -999,
                 -999,
                 id.ix(),
                 id.iy(),
                 towid.ieta() / abs(towid.ieta()),
                 id.hashedIndex(),
                 id.ic(),
                 cmsswId,
                 dbId,
                 towid.ieta(),
                 towid.iphi(),
                 TCCch,
                 getCrate(tccNb).second,
                 SLBch,
                 SLBslot,
                 getGCTRegionEta(towid.ieta()),
                 getGCTRegionPhi(towid.iphi())};
    for (int i = 0; i < 26; i++)
      ntupleInts_[i] = val[i];
    strcpy(ntupleDet_, getDet(tccNb).c_str());
    strcpy(ntupleCrate_, getCrate(tccNb).first.c_str());
    ntuple->Fill();

    if ((tccNb == 76 || tccNb == 94) && stripInTower == 1 && xtalInStrip == 3 && (towerInTCC - 1) % 4 == 0) {
      int etaSlice = towid.ietaAbs();
      coeffStruc coeff;
      //getCoeff(coeff, calibMap, id.rawId()) ;
      //modif-alex-27-july-2015
      std::string str;
      getCoeff(coeff, calibMap, laserAlphaMap, id.rawId(), str);
      ss << str;
      getCoeff(coeff, gainMap, id.rawId());
      getCoeff(coeff, pedMap, id.rawId());
      linStruc lin;
      for (int i = 0; i < 3; i++) {
        int mult, shift;
        bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EE", mult, shift);
        if (!ok)
          edm::LogError("TopInfo") << "unable to compute the parameters for Quadrant=" << id.iquadrant()
                                   << " xt=" << id.ic() << " " << dec << id.rawId() << "\n";
        else {
          lin.pedestal_[i] = coeff.pedestals_[i];
          lin.mult_[i] = mult;
          lin.shift_[i] = shift;
        }
      }

      bool ok(true);
      if (forcedPedestalValue_ == -2 || forcedPedestalValue_ == -3)
        ok = realignBaseline(lin, 0);
      if (!ok)
        ss << "Quadrant=" << id.iquadrant() << " xt=" << id.ic() << " " << dec << id.rawId() << "\n";

      linEtaSlice[etaSlice] = lin;
    }
  }

  // general case
  for (vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    double theta = theEndcapGeometry_->getGeometry(id)->getPosition().theta();
    if (!useTransverseEnergy_)
      theta = acos(0.);
    const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
    const EcalElectronicsId Id = theMapping_->getElectronicsId(id);
    towerListEE.push_back(towid.rawId());
    // special case of towers in inner rings of EE
    if (towid.ietaAbs() == 27 || towid.ietaAbs() == 28) {
      EcalTrigTowerDetId additionalTower(towid.zside(), towid.subDet(), towid.ietaAbs(), towid.iphi() + 1);
      towerListEE.push_back(additionalTower.rawId());
    }
    stripListEE.push_back(elId.rawId() & 0xfffffff8);
    int dccNb = Id.dccId();
    //int tccNb = theMapping_->TCCid(towid) ;
    //int towerInTCC = theMapping_->iTT(towid) ;
    //int stripInTower = elId.pseudoStripId() ;
    //int xtalInStrip = elId.channelId() ;
    int CCUid = Id.towerId();
    int VFEid = Id.stripId();
    int xtalInVFE = Id.xtalId();
    int xtalWithinCCUid = 5 * (VFEid - 1) + xtalInVFE - 1;  // Evgueni expects [0,24]
    int etaSlice = towid.ietaAbs();

    EcalLogicID logicId;
    FEConfigPedDat pedDB;
    FEConfigLinDat linDB;
    if (writeToFiles_)
      (*out_file_) << "CRYSTAL " << dec << id.rawId() << std::endl;
    if (writeToDB_ && DBEE_) {
      int iz = id.positiveZ();
      if (iz == 0)
        iz = -1;
      for (int k = 0; k < (int)my_CrystalEcalLogicId_EE.size(); k++) {
        int z = my_CrystalEcalLogicId_EE[k].getID1();
        int x = my_CrystalEcalLogicId_EE[k].getID2();
        int y = my_CrystalEcalLogicId_EE[k].getID3();
        if (id.ix() == x && id.iy() == y && iz == z)
          logicId = my_CrystalEcalLogicId_EE[k];
      }
    }

    coeffStruc coeff;
    //getCoeff(coeff, calibMap, id.rawId()) ;
    //modif-alex-27-july-2015
    string str;
    getCoeff(coeff, calibMap, laserAlphaMap, id.rawId(), str);
    ss << str;
    getCoeff(coeff, gainMap, id.rawId());
    getCoeff(coeff, pedMap, id.rawId());
    if (id.zside() > 0)
      ICEEPlus->Fill(id.ix(), id.iy(), coeff.calibCoeff_);
    else
      ICEEMinus->Fill(id.ix(), id.iy(), coeff.calibCoeff_);
    IC->Fill(theEndcapGeometry_->getGeometry(id)->getPosition().phi(),
             theEndcapGeometry_->getGeometry(id)->getPosition().eta(),
             coeff.calibCoeff_);

    vector<int> xtalCCU;
    xtalCCU.push_back(dccNb + 600);
    xtalCCU.push_back(CCUid);
    xtalCCU.push_back(xtalWithinCCUid);
    xtalCCU.push_back(id.rawId());

    // compute and fill linearization parameters
    // case of eta slice
    if (forceEtaSlice_) {
      map<int, linStruc>::const_iterator itLin = linEtaSlice.find(etaSlice);
      if (itLin != linEtaSlice.end()) {
        linMap[xtalCCU] = itLin->second;
        if (writeToFiles_) {
          for (int i = 0; i < 3; i++)
            (*out_file_) << hex << " 0x" << itLin->second.pedestal_[i] << " 0x" << itLin->second.mult_[i] << " 0x"
                         << itLin->second.shift_[i] << std::endl;
        }
        if (writeToDB_ && DBEE_) {
          for (int i = 0; i < 3; i++) {
            if (i == 0) {
              pedDB.setPedMeanG12(itLin->second.pedestal_[i]);
              linDB.setMultX12(itLin->second.mult_[i]);
              linDB.setShift12(itLin->second.shift_[i]);
            }
            if (i == 1) {
              pedDB.setPedMeanG6(itLin->second.pedestal_[i]);
              linDB.setMultX6(itLin->second.mult_[i]);
              linDB.setShift6(itLin->second.shift_[i]);
            }
            if (i == 2) {
              pedDB.setPedMeanG1(itLin->second.pedestal_[i]);
              linDB.setMultX1(itLin->second.mult_[i]);
              linDB.setShift1(itLin->second.shift_[i]);
            }
          }
        }
        float factor = float(itLin->second.mult_[0]) * pow(2., -itLin->second.shift_[0]) / xtal_LSB_EE_;
        if (id.zside() > 0)
          tpgFactorEEPlus->Fill(id.ix(), id.iy(), factor);
        else
          tpgFactorEEMinus->Fill(id.ix(), id.iy(), factor);
        tpgFactor->Fill(theEndcapGeometry_->getGeometry(id)->getPosition().phi(),
                        theEndcapGeometry_->getGeometry(id)->getPosition().eta(),
                        factor);
      } else
        ss << "current EtaSlice = " << etaSlice << " not found in the EtaSlice map"
           << "\n";
    } else {
      // general case
      linStruc lin;
      for (int i = 0; i < 3; i++) {
        int mult, shift;
        bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EE", mult, shift);
        if (!ok)
          edm::LogError("TopInfo") << "unable to compute the parameters for " << dec << id.rawId() << "\n";
        else {
          lin.pedestal_[i] = coeff.pedestals_[i];
          lin.mult_[i] = mult;
          lin.shift_[i] = shift;
        }
      }

      bool ok(true);
      if (forcedPedestalValue_ == -2 || forcedPedestalValue_ == -3)
        ok = realignBaseline(lin, 0);
      if (!ok)
        ss << "Quadrant=" << id.iquadrant() << " xt=" << id.ic() << " " << dec << id.rawId() << "\n";

      for (int i = 0; i < 3; i++) {
        if (writeToFiles_)
          (*out_file_) << hex << " 0x" << lin.pedestal_[i] << " 0x" << lin.mult_[i] << " 0x" << lin.shift_[i]
                       << std::endl;
        if (writeToDB_ && DBEE_) {
          if (i == 0) {
            pedDB.setPedMeanG12(lin.pedestal_[i]);
            linDB.setMultX12(lin.mult_[i]);
            linDB.setShift12(lin.shift_[i]);
          }
          if (i == 1) {
            pedDB.setPedMeanG6(lin.pedestal_[i]);
            linDB.setMultX6(lin.mult_[i]);
            linDB.setShift6(lin.shift_[i]);
          }
          if (i == 2) {
            pedDB.setPedMeanG1(lin.pedestal_[i]);
            linDB.setMultX1(lin.mult_[i]);
            linDB.setShift1(lin.shift_[i]);
          }
        }
        if (i == 0) {
          float factor = float(lin.mult_[i]) * pow(2., -lin.shift_[i]) / xtal_LSB_EE_;
          if (id.zside() > 0)
            tpgFactorEEPlus->Fill(id.ix(), id.iy(), factor);
          else
            tpgFactorEEMinus->Fill(id.ix(), id.iy(), factor);
          tpgFactor->Fill(theEndcapGeometry_->getGeometry(id)->getPosition().phi(),
                          theEndcapGeometry_->getGeometry(id)->getPosition().eta(),
                          factor);
        }
      }
      linMap[xtalCCU] = lin;
    }
    if (writeToDB_ && DBEE_) {
      pedset[logicId] = pedDB;
      linset[logicId] = linDB;
    }
  }  //eeCells
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  if (writeToDB_) {
    // EcalLogicID  of the whole barrel is: my_EcalLogicId_EB
    FEConfigLinParamDat linparam;
    linparam.setETSat(Et_sat_EE_);
    linparamset[my_EcalLogicId_EE] = linparam;

    FEConfigLUTParamDat lutparam;
    lutparam.setETSat(Et_sat_EE_);
    lutparam.setTTThreshlow(TTF_lowThreshold_EE_);
    lutparam.setTTThreshhigh(TTF_highThreshold_EE_);
    lutparamset[my_EcalLogicId_EE] = lutparam;

    FEConfigFgrParamDat fgrparam;
    fgrparam.setFGlowthresh(FG_Threshold_EE_);
    fgrparam.setFGhighthresh(FG_Threshold_EE_);
    fgrparamset[my_EcalLogicId_EE] = fgrparam;
  }

  if (writeToDB_) {
    ostringstream ltag;
    ltag.str("EB_");
    ltag << Et_sat_EB_ << "_EE_" << Et_sat_EE_;
    std::string lin_tag = ltag.str();
    ss << " LIN tag " << lin_tag << "\n";

    if (m_write_ped == 1) {
      ped_conf_id_ = db_->writeToConfDB_TPGPedestals(pedset, 1, "from_OfflineDB");
    } else {
      ss << "the ped id =" << ped_conf_id_ << " will be used for the pedestals "
         << "\n";
    }

    if (m_write_lin == 1)
      lin_conf_id_ = db_->writeToConfDB_TPGLinearCoef(linset, linparamset, 1, lin_tag);
  }

  /////////////////////
  // Evgueni interface
  ////////////////////
  std::ofstream evgueni("TPG_hardcoded.hh", std::ios::out);
  evgueni << "void getLinParamTPG_hardcoded(int fed, int ccu, int xtal," << endl;
  evgueni << "                        int & mult12, int & shift12, int & base12," << endl;
  evgueni << "                        int & mult6, int & shift6, int & base6," << endl;
  evgueni << "                        int & mult1, int & shift1, int & base1)" << endl;
  evgueni << "{" << endl;
  evgueni << "  mult12 = 0 ; shift12 = 0 ; base12 = 0 ; mult6 = 0 ; shift6 = 0 ; base6 = 0 ; mult1 = 0 ; shift1 = 0 ; "
             "base1 = 0 ;"
          << endl;
  map<vector<int>, linStruc>::const_iterator itLinMap;
  for (itLinMap = linMap.begin(); itLinMap != linMap.end(); itLinMap++) {
    vector<int> xtalInCCU = itLinMap->first;
    evgueni << "  if (fed==" << xtalInCCU[0] << " && ccu==" << xtalInCCU[1] << " && xtal==" << xtalInCCU[2] << ") {";
    evgueni << "  mult12 = " << itLinMap->second.mult_[0] << " ; shift12 = " << itLinMap->second.shift_[0]
            << " ; base12 = " << itLinMap->second.pedestal_[0] << " ; ";
    evgueni << "  mult6 = " << itLinMap->second.mult_[1] << " ; shift6 = " << itLinMap->second.shift_[1]
            << " ; base6 = " << itLinMap->second.pedestal_[1] << " ; ";
    evgueni << "  mult1 = " << itLinMap->second.mult_[2] << " ; shift1 = " << itLinMap->second.shift_[2]
            << " ; base1 = " << itLinMap->second.pedestal_[2] << " ; ";
    evgueni << "  return ;}" << endl;
  }
  evgueni << "}" << endl;
  evgueni.close();

  /////////////////////////////
  // Compute weights section //
  /////////////////////////////

  const int NWEIGROUPS = 2;
  std::vector<unsigned int> weights[NWEIGROUPS];

  bool useDBShape = useDBShape_;
  EBShape shapeEB(useDBShape);
  shapeEB.setEventSetup(evtSetup);  // EBShape, EEShape are fetched now from DB (2018.05.22 K. Theofilatos)
  EEShape shapeEE(useDBShape);
  shapeEE.setEventSetup(evtSetup);  //
  weights[0] = computeWeights(shapeEB, hshapeEB);
  weights[1] = computeWeights(shapeEE, hshapeEE);

  map<EcalLogicID, FEConfigWeightGroupDat> dataset;

  for (int igrp = 0; igrp < NWEIGROUPS; igrp++) {
    if (weights[igrp].size() == 5) {
      if (writeToFiles_) {
        (*out_file_) << std::endl;
        (*out_file_) << "WEIGHT " << igrp << endl;
        for (unsigned int sample = 0; sample < 5; sample++)
          (*out_file_) << "0x" << hex << weights[igrp][sample] << " ";
        (*out_file_) << std::endl;
        (*out_file_) << std::endl;
      }
      if (writeToDB_) {
        ss << "going to write the weights for groupe:" << igrp << "\n";
        FEConfigWeightGroupDat gut;
        gut.setWeightGroupId(igrp);
        //PP WARNING: weights order is reverted when stored in the DB
        gut.setWeight0(weights[igrp][4]);
        gut.setWeight1(weights[igrp][3] +
                       0x80);  //0x80 to identify the max of the pulse in the FENIX (doesn't exist in emulator)
        gut.setWeight2(weights[igrp][2]);
        gut.setWeight3(weights[igrp][1]);
        gut.setWeight4(weights[igrp][0]);
        EcalLogicID ecid = EcalLogicID("DUMMY", igrp, igrp);  //1 dummy ID per group
        // Fill the dataset
        dataset[ecid] = gut;
      }
    }
  }

  if (writeToDB_) {
    // now we store in the DB the correspondence btw channels and groups
    map<EcalLogicID, FEConfigWeightDat> dataset2;

    // EB loop
    for (int ich = 0; ich < (int)my_StripEcalLogicId.size(); ich++) {
      FEConfigWeightDat wut;
      int igroup = 0;  // this group is for EB
      wut.setWeightGroupId(igroup);
      dataset2[my_StripEcalLogicId[ich]] = wut;
    }

    // EE loop
    for (int ich = 0; ich < (int)my_StripEcalLogicId1_EE.size(); ich++) {
      FEConfigWeightDat wut;
      int igroup = 1;  // this group is for EE
      wut.setWeightGroupId(igroup);
      // Fill the dataset
      dataset2[my_StripEcalLogicId1_EE[ich]] = wut;
    }
    // EE loop 2 (we had to split the ids of EE in 2 vectors to avoid crash!)
    for (int ich = 0; ich < (int)my_StripEcalLogicId2_EE.size(); ich++) {
      FEConfigWeightDat wut;
      int igroup = 1;  // this group is for EE
      wut.setWeightGroupId(igroup);
      // Fill the dataset
      dataset2[my_StripEcalLogicId2_EE[ich]] = wut;
    }

    // Insert the datasets
    ostringstream wtag;
    wtag.str("");
    wtag << "Shape_NGroups_" << NWEIGROUPS;
    std::string weight_tag = wtag.str();
    ss << " weight tag " << weight_tag << "\n";
    if (m_write_wei == 1)
      wei_conf_id_ = db_->writeToConfDB_TPGWeight(dataset, dataset2, NWEIGROUPS, weight_tag);
  }
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  /////////////////////////
  // Compute FG section //
  /////////////////////////

  // barrel
  unsigned int lowRatio, highRatio, lowThreshold, highThreshold, lutFG;
  computeFineGrainEBParameters(lowRatio, highRatio, lowThreshold, highThreshold, lutFG);
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    (*out_file_) << "FG 0" << std::endl;
    (*out_file_) << hex << "0x" << lowThreshold << " 0x" << highThreshold << " 0x" << lowRatio << " 0x" << highRatio
                 << " 0x" << lutFG << std::endl;
  }

  // endcap
  unsigned int threshold, lut_tower;
  unsigned int lut_strip;
  computeFineGrainEEParameters(threshold, lut_strip, lut_tower);

  // and here we store the fgr part

  if (writeToDB_) {
    ss << "going to write the fgr "
       << "\n";
    map<EcalLogicID, FEConfigFgrGroupDat> dataset;
    // we create 1 group
    int NFGRGROUPS = 1;
    for (int ich = 0; ich < NFGRGROUPS; ich++) {
      FEConfigFgrGroupDat gut;
      gut.setFgrGroupId(ich);
      gut.setThreshLow(lowRatio);
      gut.setThreshHigh(highRatio);
      gut.setRatioLow(lowThreshold);
      gut.setRatioHigh(highThreshold);
      gut.setLUTValue(lutFG);
      EcalLogicID ecid = EcalLogicID("DUMMY", ich, ich);
      // Fill the dataset
      dataset[ecid] = gut;  // we use any logic id but different, because it is in any case ignored...
    }

    // now we store in the DB the correspondence btw channels and groups
    map<EcalLogicID, FEConfigFgrDat> dataset2;
    // in this case I decide in a stupid way which channel belongs to which group
    for (int ich = 0; ich < (int)my_TTEcalLogicId.size(); ich++) {
      FEConfigFgrDat wut;
      int igroup = 0;
      wut.setFgrGroupId(igroup);
      // Fill the dataset
      // the logic ids are ordered by SM (1,...36) and TT (1,...68)
      // you have to calculate the right index here
      dataset2[my_TTEcalLogicId[ich]] = wut;
    }

    // endcap loop
    for (int ich = 0; ich < (int)my_RTEcalLogicId_EE.size(); ich++) {
      //	std::cout << " endcap FGR " << std::endl;
      FEConfigFgrDat wut;
      int igroup = 0;
      wut.setFgrGroupId(igroup);
      // Fill the dataset
      // the logic ids are ordered by .... ?
      // you have to calculate the right index here
      dataset2[my_RTEcalLogicId_EE[ich]] = wut;
    }

    // endcap TT loop for the FEfgr EE Tower
    map<EcalLogicID, FEConfigFgrEETowerDat> dataset3;
    for (int ich = 0; ich < (int)my_TTEcalLogicId_EE.size(); ich++) {
      FEConfigFgrEETowerDat fgreett;
      fgreett.setLutValue(lut_tower);
      dataset3[my_TTEcalLogicId_EE[ich]] = fgreett;
    }

    // endcap strip loop for the FEfgr EE strip
    // and barrel strip loop for the spike parameters (same structure than EE FGr)
    map<EcalLogicID, FEConfigFgrEEStripDat> dataset4;
    for (int ich = 0; ich < (int)my_StripEcalLogicId1_EE.size(); ich++) {
      FEConfigFgrEEStripDat zut;
      zut.setThreshold(threshold);
      zut.setLutFgr(lut_strip);
      dataset4[my_StripEcalLogicId1_EE[ich]] = zut;
    }
    for (int ich = 0; ich < (int)my_StripEcalLogicId2_EE.size(); ich++) {
      FEConfigFgrEEStripDat zut;
      zut.setThreshold(threshold);
      zut.setLutFgr(lut_strip);
      // Fill the dataset
      dataset4[my_StripEcalLogicId2_EE[ich]] = zut;
    }
    for (int ich = 0; ich < (int)my_StripEcalLogicId.size(); ich++) {
      // EB
      FEConfigFgrEEStripDat zut;
      EcalLogicID thestrip = my_StripEcalLogicId[ich];
      uint32_t elStripId = stripMapEB[ich];
      map<uint32_t, uint32_t>::const_iterator it = stripMapEBsintheta.find(elStripId);
      if (it != stripMapEBsintheta.end())
        zut.setThreshold(it->second);
      else {
        edm::LogError("TopInfo") << "ERROR: strip SFGVB threshold parameter not found for that strip:"
                                 << thestrip.getID1() << " " << thestrip.getID3() << " " << thestrip.getID3() << "\n";
        edm::LogError("TopInfo") << " using value = " << SFGVB_Threshold_ + pedestal_offset_ << "\n";
        zut.setThreshold(SFGVB_Threshold_ + pedestal_offset_);
      }
      zut.setLutFgr(SFGVB_lut_);
      // Fill the dataset
      dataset4[thestrip] = zut;
    }

    // Insert the dataset
    ostringstream wtag;
    wtag.str("");
    wtag << "FGR_" << lutFG << "_N_" << NFGRGROUPS << "_eb_" << FG_lowThreshold_EB_ << "_EB_" << FG_highThreshold_EB_;
    std::string weight_tag = wtag.str();
    ss << " weight tag " << weight_tag << "\n";
    if (m_write_fgr == 1)
      fgr_conf_id_ =
          db_->writeToConfDB_TPGFgr(dataset, dataset2, fgrparamset, dataset3, dataset4, NFGRGROUPS, weight_tag);

    //modif-alex 21/01/11
    map<EcalLogicID, FEConfigSpikeDat> datasetspike;  //loob EB TT
    for (int ich = 0; ich < (int)my_TTEcalLogicId.size(); ich++) {
      FEConfigSpikeDat spiketh;
      spiketh.setSpikeThreshold(SFGVB_SpikeKillingThreshold_);
      datasetspike[my_TTEcalLogicId[ich]] = spiketh;
    }  //loop EB TT towers

    //modif-alex 21/01/11
    ostringstream stag;
    stag.str("");
    stag << "SpikeTh" << SFGVB_SpikeKillingThreshold_;
    std::string spike_tag = stag.str();
    ss << " spike tag " << spike_tag << "\n";
    if (m_write_spi == 1)
      spi_conf_id_ = db_->writeToConfDB_Spike(datasetspike, spike_tag);  //modif-alex 21/01/11

    //modif-alex 31/01/11
    //DELAYS EB
    map<EcalLogicID, FEConfigTimingDat>
        datasetdelay;  // the loop goes from TCC 38 to 72 and throught the towers from 1 to 68
    for (int ich = 0; ich < (int)my_TTEcalLogicId_EB_by_TCC.size(); ich++) {
      FEConfigTimingDat delay;

      EcalLogicID logiciddelay = my_TTEcalLogicId_EB_by_TCC[ich];
      int id1_tcc = my_TTEcalLogicId_EB_by_TCC[ich].getID1();  // the TCC
      int id2_tt = my_TTEcalLogicId_EB_by_TCC[ich].getID2();   // the tower
      std::map<int, vector<int> >::const_iterator ittEB = delays_EB_.find(id1_tcc);
      std::vector<int> TimingDelaysEB = ittEB->second;

      if (ittEB != delays_EB_.end()) {
        if (TimingDelaysEB[id2_tt - 1] == -1) {
          edm::LogError("TopInfo") << "ERROR: Barrel timing delay not specified, check file, putting default value 1"
                                   << "\n";
          delay.setTimingPar1(1);
        } else
          delay.setTimingPar1(TimingDelaysEB[id2_tt - 1]);
      } else {
        edm::LogError("TopInfo") << "ERROR:Barrel Could not find delay parameter for that trigger tower "
                                 << "\n";
        edm::LogError("TopInfo") << "Using default value = 1"
                                 << "\n";
        delay.setTimingPar1(1);
      }

      std::map<int, vector<int> >::const_iterator ittpEB = phases_EB_.find(id1_tcc);
      std::vector<int> TimingPhasesEB = ittpEB->second;

      if (ittpEB != phases_EB_.end()) {
        if (TimingPhasesEB[id2_tt - 1] == -1) {
          edm::LogError("TopInfo") << "ERROR: Barrel timing phase not specified, check file, putting default value 0"
                                   << "\n";
          delay.setTimingPar2(0);
        } else
          delay.setTimingPar2(TimingPhasesEB[id2_tt - 1]);
      } else {
        edm::LogError("TopInfo") << "ERROR:Barrel Could not find phase parameter for that trigger tower "
                                 << "\n";
        edm::LogError("TopInfo") << "Using default value = 0"
                                 << "\n";
        delay.setTimingPar2(0);
      }

      ss << ich << " tcc=" << id1_tcc << " TT=" << id2_tt << " logicId=" << logiciddelay.getLogicID()
         << " delay=" << TimingDelaysEB[id2_tt - 1] << " phase=" << TimingPhasesEB[id2_tt - 1] << "\n";

      //delay.setTimingPar1(1);
      //delay.setTimingPar2(2);
      datasetdelay[my_TTEcalLogicId_EB_by_TCC[ich]] = delay;
    }  //loop EB TT towers

    //DELAYS EE
    int stripindex = 0;
    int tccin = 1;
    for (int ich = 0; ich < (int)my_StripEcalLogicId_EE_strips_by_TCC.size(); ich++) {
      FEConfigTimingDat delay;
      //int id1_strip=my_StripEcalLogicId_EE_strips_by_TCC[ich].getID1(); // the TCC
      //int id2_strip=my_StripEcalLogicId_EE_strips_by_TCC[ich].getID2(); // the Tower
      //int id3_strip=my_StripEcalLogicId_EE_strips_by_TCC[ich].getID3(); // the strip

      EcalLogicID logiciddelay = my_StripEcalLogicId_EE_strips_by_TCC[ich];
      int id1_tcc = my_StripEcalLogicId_EE_strips_by_TCC[ich].getID1();  // the TCC
      int id2_tt = my_StripEcalLogicId_EE_strips_by_TCC[ich].getID2();   // the tower
      int id3_st = my_StripEcalLogicId_EE_strips_by_TCC[ich].getID3();   // the strip

      //reset strip counter
      if (id1_tcc != tccin) {
        tccin = id1_tcc;
        stripindex = 0;
      }

      std::map<int, vector<int> >::const_iterator ittEE = delays_EE_.find(id1_tcc);
      std::vector<int> TimingDelaysEE = ittEE->second;

      if (ittEE != delays_EE_.end()) {
        if (TimingDelaysEE[stripindex] == -1) {
          edm::LogError("TopInfo") << "ERROR: Endcap timing delay not specified, check file, putting default value 1"
                                   << "\n";
          delay.setTimingPar1(1);
        } else
          delay.setTimingPar1(TimingDelaysEE[stripindex]);
      } else {
        edm::LogError("TopInfo") << "ERROR:Endcap Could not find delay parameter for that trigger tower "
                                 << "\n";
        edm::LogError("TopInfo") << "Using default value = 1"
                                 << "\n";
        delay.setTimingPar1(1);
      }

      std::map<int, vector<int> >::const_iterator ittpEE = phases_EE_.find(id1_tcc);
      std::vector<int> TimingPhasesEE = ittpEE->second;

      if (ittpEE != phases_EE_.end()) {
        if (TimingPhasesEE[stripindex] == -1) {
          edm::LogError("TopInfo") << "ERROR: Endcap timing phase not specified, check file, putting default value 0"
                                   << "\n";
          delay.setTimingPar2(0);
        } else
          delay.setTimingPar2(TimingPhasesEE[stripindex]);
      } else {
        edm::LogError("TopInfo") << "ERROR:Endcap Could not find phase parameter for that trigger tower "
                                 << "\n";
        edm::LogError("TopInfo") << "Using default value = 0"
                                 << "\n";
        delay.setTimingPar2(0);
      }

      ss << ich << " stripindex=" << stripindex << " tcc=" << id1_tcc << " TT=" << id2_tt << " id3_st=" << id3_st
         << " logicId=" << logiciddelay.getLogicID() << " delay=" << TimingDelaysEE[stripindex]
         << " phase=" << TimingPhasesEE[stripindex] << "\n";

      //delay.setTimingPar1(1);
      //delay.setTimingPar2(2);
      datasetdelay[my_StripEcalLogicId_EE_strips_by_TCC[ich]] = delay;
      stripindex++;
    }  //loop EE strip towers

    ostringstream de_tag;
    de_tag.str("");
    de_tag << "DelaysFromFile";
    std::string delay_tag = de_tag.str();
    ss << " delay tag " << delay_tag << "\n";
    if (m_write_del == 1)
      del_conf_id_ = db_->writeToConfDB_Delay(datasetdelay, delay_tag);  //modif-alex 31/01/11

  }  //write to DB

  if (writeToDB_) {
    ss << "going to write the sliding "
       << "\n";
    map<EcalLogicID, FEConfigSlidingDat> dataset;
    // in this case I decide in a stupid way which channel belongs to which group
    for (int ich = 0; ich < (int)my_StripEcalLogicId.size(); ich++) {
      FEConfigSlidingDat wut;
      wut.setSliding(sliding_);
      // Fill the dataset
      // the logic ids are ordered by SM (1,...36) , TT (1,...68) and strip (1..5)
      // you have to calculate the right index here
      dataset[my_StripEcalLogicId[ich]] = wut;
    }

    // endcap loop
    for (int ich = 0; ich < (int)my_StripEcalLogicId1_EE.size(); ich++) {
      FEConfigSlidingDat wut;
      wut.setSliding(sliding_);
      // Fill the dataset
      // the logic ids are ordered by fed tower strip
      // you have to calculate the right index here
      dataset[my_StripEcalLogicId1_EE[ich]] = wut;
    }
    for (int ich = 0; ich < (int)my_StripEcalLogicId2_EE.size(); ich++) {
      FEConfigSlidingDat wut;
      wut.setSliding(sliding_);
      // Fill the dataset
      // the logic ids are ordered by ... ?
      // you have to calculate the right index here
      dataset[my_StripEcalLogicId2_EE[ich]] = wut;
    }

    // Insert the dataset
    ostringstream wtag;
    wtag.str("");
    wtag << "Sliding_" << sliding_;
    std::string justatag = wtag.str();
    ss << " sliding tag " << justatag << "\n";
    int iov_id = 0;  // just a parameter ...
    if (m_write_sli == 1)
      sli_conf_id_ = db_->writeToConfDB_TPGSliding(dataset, iov_id, justatag);
  }

  /////////////////////////
  // Compute LUT section //
  /////////////////////////

  int lut_EB[1024], lut_EE[1024];

  // barrel
  computeLUT(lut_EB, "EB");
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    (*out_file_) << "LUT 0" << std::endl;
    for (int i = 0; i < 1024; i++)
      (*out_file_) << "0x" << hex << lut_EB[i] << endl;
    (*out_file_) << endl;
  }

  // endcap
  computeLUT(lut_EE, "EE");
  // check first if lut_EB and lut_EE are the same
  bool newLUT(false);
  for (int i = 0; i < 1024; i++)
    if (lut_EE[i] != lut_EB[i])
      newLUT = true;
  if (newLUT && writeToFiles_) {
    (*out_file_) << std::endl;
    (*out_file_) << "LUT 1" << std::endl;
    for (int i = 0; i < 1024; i++)
      (*out_file_) << "0x" << hex << lut_EE[i] << endl;
    (*out_file_) << endl;
  }

  if (writeToDB_) {
    map<EcalLogicID, FEConfigLUTGroupDat> dataset;
    // we create 1 LUT group
    int NLUTGROUPS = 0;
    int ich = 0;
    FEConfigLUTGroupDat lut;
    lut.setLUTGroupId(ich);
    for (int i = 0; i < 1024; i++) {
      lut.setLUTValue(i, lut_EB[i]);
    }
    EcalLogicID ecid = EcalLogicID("DUMMY", ich, ich);
    // Fill the dataset
    dataset[ecid] = lut;  // we use any logic id but different, because it is in any case ignored...

    ich++;

    FEConfigLUTGroupDat lute;
    lute.setLUTGroupId(ich);
    for (int i = 0; i < 1024; i++) {
      lute.setLUTValue(i, lut_EE[i]);
    }
    EcalLogicID ecide = EcalLogicID("DUMMY", ich, ich);
    // Fill the dataset
    dataset[ecide] = lute;  // we use any logic id but different, because it is in any case ignored...

    ich++;

    NLUTGROUPS = ich;

    // now we store in the DB the correspondence btw channels and LUT groups
    map<EcalLogicID, FEConfigLUTDat> dataset2;
    // in this case I decide in a stupid way which channel belongs to which group
    for (int ich = 0; ich < (int)my_TTEcalLogicId.size(); ich++) {
      FEConfigLUTDat lut;
      int igroup = 0;
      lut.setLUTGroupId(igroup);
      // calculate the right TT - in the vector they are ordered by SM and by TT
      // Fill the dataset
      dataset2[my_TTEcalLogicId[ich]] = lut;
    }

    // endcap loop
    for (int ich = 0; ich < (int)my_TTEcalLogicId_EE.size(); ich++) {
      FEConfigLUTDat lut;
      int igroup = 1;
      lut.setLUTGroupId(igroup);
      // calculate the right TT
      // Fill the dataset
      dataset2[my_TTEcalLogicId_EE[ich]] = lut;
    }

    // Insert the dataset
    ostringstream ltag;
    ltag.str("");
    ltag << LUT_option_ << "_NGroups_" << NLUTGROUPS;
    std::string lut_tag = ltag.str();
    ss << " LUT tag " << lut_tag << "\n";
    if (m_write_lut == 1)
      lut_conf_id_ = db_->writeToConfDB_TPGLUT(dataset, dataset2, lutparamset, NLUTGROUPS, lut_tag);
  }

  // last we insert the FE_CONFIG_MAIN table
  if (writeToDB_) {
    //int conf_id_=db_->writeToConfDB_TPGMain(ped_conf_id_,lin_conf_id_, lut_conf_id_, fgr_conf_id_,
    //				sli_conf_id_, wei_conf_id_, bxt_conf_id_, btt_conf_id_, tag_, version_) ;
    int conf_id_ = db_->writeToConfDB_TPGMain(ped_conf_id_,
                                              lin_conf_id_,
                                              lut_conf_id_,
                                              fgr_conf_id_,
                                              sli_conf_id_,
                                              wei_conf_id_,
                                              spi_conf_id_,
                                              del_conf_id_,
                                              bxt_conf_id_,
                                              btt_conf_id_,
                                              bst_conf_id_,
                                              tag_,
                                              version_);  //modif-alex 21/01/11

    ss << "\n Conf ID = " << conf_id_ << "\n";
  }

  ////////////////////////////////////////////////////
  // loop on strips and associate them with  values //
  ////////////////////////////////////////////////////

  // Barrel
  stripListEB.sort();
  stripListEB.unique();
  ss << "Number of EB strips=" << dec << stripListEB.size() << "\n";
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    for (itList = stripListEB.begin(); itList != stripListEB.end(); itList++) {
      (*out_file_) << "STRIP_EB " << dec << (*itList) << endl;
      (*out_file_) << hex << "0x" << sliding_ << std::endl;
      (*out_file_) << "0" << std::endl;
      (*out_file_) << "0x" << stripMapEBsintheta[(*itList)] << " 0x" << SFGVB_lut_ << std::endl;
    }
  }

  // Endcap
  stripListEE.sort();
  stripListEE.unique();
  ss << "Number of EE strips=" << dec << stripListEE.size() << "\n";
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    for (itList = stripListEE.begin(); itList != stripListEE.end(); itList++) {
      (*out_file_) << "STRIP_EE " << dec << (*itList) << endl;
      (*out_file_) << hex << "0x" << sliding_ << std::endl;
      //(*out_file_) <<" 0" << std::endl ;
      (*out_file_) << " 1" << std::endl;  //modif-debug to get the correct EE TPG
      (*out_file_) << hex << "0x" << threshold << " 0x" << lut_strip << std::endl;
    }
  }
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  ///////////////////////////////////////////////////////////
  // loop on towers and associate them with default values //
  ///////////////////////////////////////////////////////////

  // Barrel
  towerListEB.sort();
  towerListEB.unique();
  ss << "Number of EB towers=" << dec << towerListEB.size() << "\n";
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    for (itList = towerListEB.begin(); itList != towerListEB.end(); itList++) {
      (*out_file_) << "TOWER_EB " << dec << (*itList) << endl;
      (*out_file_) << " 0\n 0\n";
      (*out_file_) << " " << SFGVB_SpikeKillingThreshold_ << std::endl;  //modif-alex
    }
  }

  // Endcap
  towerListEE.sort();
  towerListEE.unique();
  ss << "Number of EE towers=" << dec << towerListEE.size() << "\n";
  if (writeToFiles_) {
    (*out_file_) << std::endl;
    for (itList = towerListEE.begin(); itList != towerListEE.end(); itList++) {
      (*out_file_) << "TOWER_EE " << dec << (*itList) << endl;
      if (newLUT)
        (*out_file_) << " 1\n";
      else
        (*out_file_) << " 0\n";
      (*out_file_) << hex << "0x" << lut_tower << std::endl;
    }
  }
  edm::LogInfo("TopInfo") << ss.str();
  ss.str("");

  //////////////////////////
  // store control histos //
  //////////////////////////
  ICEB->Write();
  tpgFactorEB->Write();
  ICEEPlus->Write();
  tpgFactorEEPlus->Write();
  ICEEMinus->Write();
  tpgFactorEEMinus->Write();
  IC->Write();
  tpgFactor->Write();
  hshapeEB->Write();
  hshapeEE->Write();
  ntuple->Write();
  ntupleSpike->Write();
  saving.Close();
}

void EcalTPGParamBuilder::beginJob() {
  using namespace edm;
  using namespace std;

  edm::LogInfo("TopInfo") << "we are in beginJob\n";

  create_header();

  DetId eb(DetId::Ecal, EcalBarrel);
  DetId ee(DetId::Ecal, EcalEndcap);

  if (writeToFiles_) {
    (*out_file_) << "PHYSICS_EB " << dec << eb.rawId() << std::endl;
    (*out_file_) << Et_sat_EB_ << " " << TTF_lowThreshold_EB_ << " " << TTF_highThreshold_EB_ << std::endl;
    (*out_file_) << FG_lowThreshold_EB_ << " " << FG_highThreshold_EB_ << " " << FG_lowRatio_EB_ << " "
                 << FG_highRatio_EB_ << std::endl;
    //(*out_file_) << SFGVB_SpikeKillingThreshold_ << std::endl; //modif-alex02/02/2011
    (*out_file_) << std::endl;

    (*out_file_) << "PHYSICS_EE " << dec << ee.rawId() << std::endl;
    (*out_file_) << Et_sat_EE_ << " " << TTF_lowThreshold_EE_ << " " << TTF_highThreshold_EE_ << std::endl;
    (*out_file_) << FG_Threshold_EE_ << " " << -1 << " " << -1 << " " << -1 << std::endl;
    (*out_file_) << std::endl;
  }
}

bool EcalTPGParamBuilder::computeLinearizerParam(
    double theta, double gainRatio, double calibCoeff, std::string subdet, int& mult, int& shift) {
  /*
    Linearization coefficient are determined in order to satisfy:
    tpg(ADC_sat) = 1024
    where: 
    tpg() is a model of the linearized tpg response on 10b 
    ADC_sat is the number of ADC count corresponding the Et_sat, the maximum scale of the transverse energy
    
    Since we have:
    Et_sat = xtal_LSB * ADC_sat * gainRatio * calibCoeff * sin(theta)
    and a simple model of tpg() being given by:
    tpg(X) = [ (X*mult) >> (shift+2) ] >> (sliding+shiftDet) 
    we must satisfy:
    [ (Et_sat/(xtal_LSB * gainRatio * calibCoeff * sin(theta)) * mult) >> (shift+2) ] >> (sliding+shiftDet) = 1024 
    that is:
    mult = 1024/Et_sat * xtal_LSB * gainRatio * calibCoeff * sin(theta) * 2^-(sliding+shiftDet+2) * 2^-shift
    mult = factor * 2^-shift
  */

  // case barrel:
  int shiftDet = 2;  //fixed, due to FE FENIX TCP format
  double ratio = xtal_LSB_EB_ / Et_sat_EB_;
  // case endcap:
  if (subdet == "EE") {
    shiftDet = 2;  //applied in TCC-EE and not in FE FENIX TCP... This parameters is setable in the TCC-EE
    //shiftDet = 0 ; //was like this before with FE bug
    ratio = xtal_LSB_EE_ / Et_sat_EE_;
  }

  //modif-alex-30/01/2012
  //std::cout << "calibCoeff="<<calibCoeff<<endl;

  double factor = 1024 * ratio * gainRatio * calibCoeff * sin(theta) * (1 << (sliding_ + shiftDet + 2));
  // Let's try first with shift = 0 (trivial solution)
  mult = (int)(factor + 0.5);
  for (shift = 0; shift < 15; shift++) {
    if (mult >= 128 && mult < 256)
      return true;
    factor *= 2;
    mult = (int)(factor + 0.5);
  }
  edm::LogError("TopInfo") << "too bad we did not manage to calculate the factor for calib=" << calibCoeff << "\n";
  return false;
}

void EcalTPGParamBuilder::create_header() {
  if (!writeToFiles_)
    return;
  (*out_file_) << "COMMENT put your comments here" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           physics EB structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)" << std::endl;
  (*out_file_) << "COMMENT  FG_lowThreshold (GeV), FG_highThreshold (GeV), FG_lowRatio, FG_highRatio" << std::endl;
  //(*out_file_) <<"COMMENT  SFGVB_SpikeKillingThreshold (GeV)"<<std::endl ; //modif-alex-02/02/2011
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           physics EE structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)" << std::endl;
  (*out_file_) << "COMMENT  FG_Threshold (GeV), dummy, dummy, dummy" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           crystal structure (same for EB and EE)" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  ped, mult, shift [gain12]" << std::endl;
  (*out_file_) << "COMMENT  ped, mult, shift [gain6]" << std::endl;
  (*out_file_) << "COMMENT  ped, mult, shift [gain1]" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           strip EB structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  sliding_window" << std::endl;
  (*out_file_) << "COMMENT  weightGroupId" << std::endl;
  (*out_file_) << "COMMENT  threshold_sfg lut_sfg" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           strip EE structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  sliding_window" << std::endl;
  (*out_file_) << "COMMENT  weightGroupId" << std::endl;
  (*out_file_) << "COMMENT  threshold_fg lut_fg" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           tower EB structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  LUTGroupId" << std::endl;
  (*out_file_) << "COMMENT  FgGroupId" << std::endl;
  (*out_file_) << "COMMENT  spike_killing_threshold" << std::endl;  //modif alex
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           tower EE structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  LUTGroupId" << std::endl;
  (*out_file_) << "COMMENT  tower_lut_fg" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           Weight structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  weightGroupId" << std::endl;
  (*out_file_) << "COMMENT  w0, w1, w2, w3, w4" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           lut structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  LUTGroupId" << std::endl;
  (*out_file_) << "COMMENT  LUT[1-1024]" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT           fg EB structure" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;
  (*out_file_) << "COMMENT  FgGroupId" << std::endl;
  (*out_file_) << "COMMENT  el, eh, tl, th, lut_fg" << std::endl;
  (*out_file_) << "COMMENT =================================" << std::endl;
  (*out_file_) << "COMMENT" << std::endl;

  (*out_file_) << std::endl;
}

int EcalTPGParamBuilder::uncodeWeight(double weight, int complement2) {
  int iweight;
  unsigned int max = (unsigned int)(pow(2., complement2) - 1);
  if (weight > 0)
    iweight = int((1 << 6) * weight + 0.5);  // +0.5 for rounding pb
  else
    iweight = max - int(-weight * (1 << 6) + 0.5) + 1;
  iweight = iweight & max;
  return iweight;
}

double EcalTPGParamBuilder::uncodeWeight(int iweight, int complement2) {
  double weight = double(iweight) / pow(2., 6.);
  // test if negative weight:
  if ((iweight & (1 << (complement2 - 1))) != 0)
    weight = (double(iweight) - pow(2., complement2)) / pow(2., 6.);
  return weight;
}

std::vector<unsigned int> EcalTPGParamBuilder::computeWeights(EcalShapeBase& shape, TH1F* histo) {
  std::ostringstream ss;
  ss << "Computing Weights...\n";
  double timeMax = shape.timeOfMax() - shape.timeOfThr();  // timeMax w.r.t begining of pulse
  double max = shape(timeMax);

  double sumf = 0.;
  double sumf2 = 0.;
  for (unsigned int sample = 0; sample < nSample_; sample++) {
    double time = timeMax - ((double)sampleMax_ - (double)sample) * 25.;
    time -= weight_timeShift_;
    sumf += shape(time) / max;
    sumf2 += shape(time) / max * shape(time) / max;
    for (int subtime = 0; subtime < 25; subtime++)
      histo->Fill(float(sample * 25. + subtime) / 25., shape(time + subtime));
  }
  double lambda = 1. / (sumf2 - sumf * sumf / nSample_);
  double gamma = -lambda * sumf / nSample_;
  double* weight = new double[nSample_];
  for (unsigned int sample = 0; sample < nSample_; sample++) {
    double time = timeMax - ((double)sampleMax_ - (double)sample) * 25.;
    time -= weight_timeShift_;
    weight[sample] = lambda * shape(time) / max + gamma;
  }

  int* iweight = new int[nSample_];
  for (unsigned int sample = 0; sample < nSample_; sample++)
    iweight[sample] = uncodeWeight(weight[sample], complement2_);

  // Let's check:
  int isumw = 0;
  for (unsigned int sample = 0; sample < nSample_; sample++)
    isumw += iweight[sample];
  unsigned int imax = (unsigned int)(pow(2., int(complement2_)) - 1);
  isumw = (isumw & imax);

  double ampl = 0.;
  double sumw = 0.;
  for (unsigned int sample = 0; sample < nSample_; sample++) {
    double time = timeMax - ((double)sampleMax_ - (double)sample) * 25.;
    time -= weight_timeShift_;
    ampl += weight[sample] * shape(time);
    sumw += weight[sample];
    ss << "weight=" << weight[sample] << " shape=" << shape(time) << "\n";
  }
  ss << "Weights: sum=" << isumw << " in float =" << uncodeWeight(isumw, complement2_) << " sum of floats =" << sumw
     << "\n";
  ss << "Weights: sum (weight*shape) = " << ampl << "\n";

  // Let's correct for bias if any
  if (weight_unbias_recovery_) {
    int count = 0;
    while (isumw != 0 && count < 10) {
      double min = 99.;
      unsigned int index = 0;
      if ((isumw & (1 << (complement2_ - 1))) != 0) {
        // add 1:
        ss << "Correcting for bias: adding 1\n";
        for (unsigned int sample = 0; sample < nSample_; sample++) {
          int new_iweight = iweight[sample] + 1;
          double new_weight = uncodeWeight(new_iweight, complement2_);
          if (fabs(new_weight - weight[sample]) < min) {
            min = fabs(new_weight - weight[sample]);
            index = sample;
          }
        }
        iweight[index]++;
      } else {
        // Sub 1:
        ss << "Correcting for bias: subtracting 1\n";
        for (unsigned int sample = 0; sample < nSample_; sample++) {
          int new_iweight = iweight[sample] - 1;
          double new_weight = uncodeWeight(new_iweight, complement2_);
          if (fabs(new_weight - weight[sample]) < min) {
            min = fabs(new_weight - weight[sample]);
            index = sample;
          }
        }
        iweight[index]--;
      }
      isumw = 0;
      for (unsigned int sample = 0; sample < nSample_; sample++)
        isumw += iweight[sample];
      imax = (unsigned int)(pow(2., int(complement2_)) - 1);
      isumw = (isumw & imax);
      ss << "Correcting weight number: " << index << " sum weights = " << isumw << "\n";
      count++;
    }
  }

  // let's check again
  isumw = 0;
  for (unsigned int sample = 0; sample < nSample_; sample++)
    isumw += iweight[sample];
  imax = (unsigned int)(pow(2., int(complement2_)) - 1);
  isumw = (isumw & imax);
  ampl = 0.;
  for (unsigned int sample = 0; sample < nSample_; sample++) {
    double time = timeMax - ((double)sampleMax_ - (double)sample) * 25.;
    time -= weight_timeShift_;
    double new_weight = uncodeWeight(iweight[sample], complement2_);
    sumw += uncodeWeight(iweight[sample], complement2_);
    ampl += new_weight * shape(time);
    ss << "weight unbiased after integer conversion=" << new_weight << " shape=" << shape(time) << "\n";
  }
  ss << "Weights: sum=" << isumw << " in float =" << uncodeWeight(isumw, complement2_) << " sum of floats =" << sumw
     << "\n";
  ss << "Weights: sum (weight*shape) = " << ampl << "\n";
  edm::LogInfo("TopInfo") << ss.str();

  std::vector<unsigned int> theWeights;
  for (unsigned int sample = 0; sample < nSample_; sample++)
    theWeights.push_back(iweight[sample]);

  delete[] weight;
  delete[] iweight;
  return theWeights;
}

void EcalTPGParamBuilder::computeLUT(int* lut, std::string det) {
  double Et_sat = Et_sat_EB_;
  double LUT_threshold = LUT_threshold_EB_;
  double LUT_stochastic = LUT_stochastic_EB_;
  double LUT_noise = LUT_noise_EB_;
  double LUT_constant = LUT_constant_EB_;
  double TTF_lowThreshold = TTF_lowThreshold_EB_;
  double TTF_highThreshold = TTF_highThreshold_EB_;
  if (det == "EE") {
    Et_sat = Et_sat_EE_;
    LUT_threshold = LUT_threshold_EE_;
    LUT_stochastic = LUT_stochastic_EE_;
    LUT_noise = LUT_noise_EE_;
    LUT_constant = LUT_constant_EE_;
    TTF_lowThreshold = TTF_lowThreshold_EE_;
    TTF_highThreshold = TTF_highThreshold_EE_;
  }

  // initialisation with identity
  for (int i = 0; i < 1024; i++) {
    lut[i] = i;
    if (lut[i] > 0xff)
      lut[i] = 0xff;
  }

  // case linear LUT
  if (LUT_option_ == "Linear") {
    int mylut = 0;
    for (int i = 0; i < 1024; i++) {
      lut[i] = mylut;
      if ((i + 1) % 4 == 0)
        mylut++;
      //if ((i+1)%8 == 0 ) mylut++ ;//modif-alex 16/12/2010 LSB==500MeV ONLY USED FOR BEAMV4 key
    }
  }

  // case LUT following Ecal resolution
  if (LUT_option_ == "EcalResolution") {
    TF1* func = new TF1("func", oneOverEtResolEt, 0., Et_sat, 3);
    func->SetParameters(LUT_stochastic, LUT_noise, LUT_constant);
    double norm = func->Integral(0., Et_sat);
    for (int i = 0; i < 1024; i++) {
      double Et = i * Et_sat / 1024.;
      lut[i] = int(0xff * func->Integral(0., Et) / norm + 0.5);
    }
  }

  // Now, add TTF thresholds to LUT and apply LUT threshold if needed
  for (int j = 0; j < 1024; j++) {
    double Et_GeV = Et_sat / 1024 * (j + 0.5);
    if (Et_GeV <= LUT_threshold)
      lut[j] = 0;  // LUT threshold
    int ttf = 0x0;
    if (Et_GeV >= TTF_highThreshold)
      ttf = 3;
    if (Et_GeV >= TTF_lowThreshold && Et_GeV < TTF_highThreshold)
      ttf = 1;
    ttf = ttf << 8;
    lut[j] += ttf;
  }
}

void EcalTPGParamBuilder::getCoeff(coeffStruc& coeff,
                                   const EcalIntercalibConstantMap& calibMap,
                                   const EcalLaserAlphaMap& laserAlphaMap,
                                   uint rawId,
                                   std::string& st) {
  // get current intercalibration coeff
  coeff.calibCoeff_ = 1.;
  if (!useInterCalibration_)
    return;
  EcalIntercalibConstantMap::const_iterator icalit = calibMap.find(rawId);

  //modif-alex-30/01/2012
  std::map<int, double>::const_iterator itCorr = Transparency_Correction_.find(rawId);
  double icorr = 1.0;
  double alpha_factor = 1.0;

  if (useTransparencyCorr_) {
    icorr = itCorr->second;
    if (itCorr != Transparency_Correction_.end()) {
      stringstream ss;
      ss << rawId;
      stringstream ss1;
      ss1 << icorr;
      stringstream ss2;
      ss2 << (*icalit);
      st = "Transparency correction found for xtal " + ss.str() + " corr=" + ss1.str() + " intercalib=" + ss2.str() +
           "\n";
    } else
      edm::LogError("TopInfo") << "ERROR = Transparency correction not found for xtal " << rawId << "\n";

    //modif-alex-27-july-2015
    DetId ECALdetid(rawId);
    //    ss << "DETID=" << ECALdetid.subdetId() << "\n";
    stringstream ss;
    ss << ECALdetid.subdetId();
    st += "DETID=" + ss.str() + "\n";
    if (ECALdetid.subdetId() == 1) {  //ECAL BARREL
      EBDetId barrel_detid(rawId);
      EcalLaserAlphaMap::const_iterator italpha = laserAlphaMap.find(barrel_detid);
      if (italpha != laserAlphaMap.end())
        alpha_factor = (*italpha);
      else
        edm::LogError("TopInfo") << "ERROR:LaserAlphe parameter note found!!"
                                 << "\n";
    }
    if (ECALdetid.subdetId() == 2) {  //ECAL ENDCAP
      EEDetId endcap_detid(rawId);
      EcalLaserAlphaMap::const_iterator italpha = laserAlphaMap.find(endcap_detid);
      if (italpha != laserAlphaMap.end())
        alpha_factor = (*italpha);
      else
        edm::LogError("TopInfo") << "ERROR:LaserAlphe parameter note found!!"
                                 << "\n";
    }

  }  //transparency corrections applied

  //if( icalit != calibMap.end() ) coeff.calibCoeff_ = (*icalit) ;
  //if( icalit != calibMap.end() ) coeff.calibCoeff_ = (*icalit)/icorr; //modif-alex-30/01/2010 tansparency corrections
  //  ss << "rawId " << (*icalit) << " " << icorr << " " << alpha_factor << "\n";
  stringstream ss;
  ss << (*icalit);
  stringstream ss1;
  ss1 << icorr;
  stringstream ss2;
  ss2 << alpha_factor;
  st += "rawId " + ss.str() + " " + ss1.str() + " " + ss2.str() + "\n";
  if (icalit != calibMap.end())
    coeff.calibCoeff_ =
        (*icalit) /
        std::pow(icorr, alpha_factor);  //modif-alex-27/07/2015 tansparency corrections with alpha parameters

  else
    edm::LogError("TopInfo") << "getCoeff: " << rawId << " not found in EcalIntercalibConstantMap"
                             << "\n";
}

void EcalTPGParamBuilder::getCoeff(coeffStruc& coeff, const EcalGainRatioMap& gainMap, unsigned int rawId) {
  // get current gain ratio
  coeff.gainRatio_[0] = 1.;
  coeff.gainRatio_[1] = 2.;
  coeff.gainRatio_[2] = 12.;
  EcalGainRatioMap::const_iterator gainIter = gainMap.find(rawId);
  if (gainIter != gainMap.end()) {
    const EcalMGPAGainRatio& aGain = (*gainIter);
    coeff.gainRatio_[1] = aGain.gain12Over6();
    coeff.gainRatio_[2] = aGain.gain6Over1() * aGain.gain12Over6();
  } else
    edm::LogError("TopInfo") << "getCoeff: " << rawId << " not found in EcalGainRatioMap"
                             << "\n";
}

void EcalTPGParamBuilder::getCoeff(coeffStruc& coeff, const EcalPedestalsMap& pedMap, unsigned int rawId) {
  coeff.pedestals_[0] = 0;
  coeff.pedestals_[1] = 0;
  coeff.pedestals_[2] = 0;

  if (forcedPedestalValue_ >= 0) {
    coeff.pedestals_[0] = forcedPedestalValue_;
    coeff.pedestals_[1] = forcedPedestalValue_;
    coeff.pedestals_[2] = forcedPedestalValue_;
    return;
  }

  // get current pedestal
  EcalPedestalsMapIterator pedIter = pedMap.find(rawId);
  if (pedIter != pedMap.end()) {
    EcalPedestals::Item aped = (*pedIter);
    coeff.pedestals_[0] = int(aped.mean_x12 + 0.5);
    coeff.pedestals_[1] = int(aped.mean_x6 + 0.5);
    coeff.pedestals_[2] = int(aped.mean_x1 + 0.5);
  } else
    edm::LogError("TopInfo") << "getCoeff: " << rawId << " not found in EcalPedestalsMap\n";
}

void EcalTPGParamBuilder::getCoeff(coeffStruc& coeff,
                                   const map<EcalLogicID, MonPedestalsDat>& pedMap,
                                   const EcalLogicID& logicId) {
  // get current pedestal
  coeff.pedestals_[0] = 0;
  coeff.pedestals_[1] = 0;
  coeff.pedestals_[2] = 0;

  map<EcalLogicID, MonPedestalsDat>::const_iterator it = pedMap.find(logicId);
  if (it != pedMap.end()) {
    MonPedestalsDat ped = it->second;
    coeff.pedestals_[0] = int(ped.getPedMeanG12() + 0.5);
    coeff.pedestals_[1] = int(ped.getPedMeanG6() + 0.5);
    coeff.pedestals_[2] = int(ped.getPedMeanG1() + 0.5);
  } else
    edm::LogError("TopInfo") << "getCoeff: " << logicId.getID1() << ", " << logicId.getID2() << ", " << logicId.getID3()
                             << " not found in map<EcalLogicID, MonPedestalsDat\n";
}

void EcalTPGParamBuilder::computeFineGrainEBParameters(unsigned int& lowRatio,
                                                       unsigned int& highRatio,
                                                       unsigned int& lowThreshold,
                                                       unsigned int& highThreshold,
                                                       unsigned int& lut) {
  lowRatio = int(0x80 * FG_lowRatio_EB_ + 0.5);
  if (lowRatio > 0x7f)
    lowRatio = 0x7f;
  highRatio = int(0x80 * FG_highRatio_EB_ + 0.5);
  if (highRatio > 0x7f)
    highRatio = 0x7f;

  // lsb at the stage of the FG calculation is:
  double lsb_FG = Et_sat_EB_ / 1024. / 4;
  lowThreshold = int(FG_lowThreshold_EB_ / lsb_FG + 0.5);
  if (lowThreshold > 0xff)
    lowThreshold = 0xff;
  highThreshold = int(FG_highThreshold_EB_ / lsb_FG + 0.5);
  if (highThreshold > 0xff)
    highThreshold = 0xff;

  // FG lut: FGVB response is LUT(adress) where adress is:
  // bit3: maxof2/ET >= lowRatio, bit2: maxof2/ET >= highRatio, bit1: ET >= lowThreshold, bit0: ET >= highThreshold
  // FGVB =1 if jet-like (veto active), =0 if E.M.-like
  // the condition for jet-like is: ET>Threshold and  maxof2/ET < Ratio (only TT with enough energy are vetoed)

  // With the following lut, what matters is only max(TLow, Thigh) and max(Elow, Ehigh)
  // So, jet-like if maxof2/ettot<max(TLow, Thigh) && ettot >= max(Elow, Ehigh)
  if (FG_lut_EB_ == 0)
    lut = 0x0888;
  else
    lut = FG_lut_EB_;  // let's use the users value (hope he/she knows what he/she does!)
}

void EcalTPGParamBuilder::computeFineGrainEEParameters(unsigned int& threshold,
                                                       unsigned int& lut_strip,
                                                       unsigned int& lut_tower) {
  // lsb for EE:
  double lsb_FG = Et_sat_EE_ / 1024.;  // FIXME is it true????
  threshold = int(FG_Threshold_EE_ / lsb_FG + 0.5);
  lut_strip = FG_lut_strip_EE_;
  lut_tower = FG_lut_tower_EE_;
}

bool EcalTPGParamBuilder::realignBaseline(linStruc& lin, float forceBase12) {
  bool ok(true);
  float base[3] = {forceBase12, float(lin.pedestal_[1]), float(lin.pedestal_[2])};
  for (int i = 1; i < 3; i++) {
    if (lin.mult_[i] > 0) {
      base[i] = float(lin.pedestal_[i]) - float(lin.mult_[0]) / float(lin.mult_[i]) *
                                              pow(2., -(lin.shift_[0] - lin.shift_[i])) * (lin.pedestal_[0] - base[0]);
    } else
      base[i] = 0;
  }

  for (int i = 0; i < 3; i++) {
    lin.pedestal_[i] = base[i];
    //cout<<lin.pedestal_[i]<<" "<<base[i]<<endl ;
    if (base[i] < 0 || lin.pedestal_[i] > 1000) {
      edm::LogError("TopInfo") << "WARNING: base= " << base[i] << ", " << lin.pedestal_[i] << " for gainId[0-2]=" << i
                               << " ==> forcing at 0"
                               << "\n";
      lin.pedestal_[i] = 0;
      ok = false;
    }
  }
  return ok;
}

int EcalTPGParamBuilder::getGCTRegionPhi(int ttphi) {
  int gctphi = 0;
  gctphi = (ttphi + 1) / 4;
  if (ttphi <= 2)
    gctphi = 0;
  if (ttphi >= 71)
    gctphi = 0;
  return gctphi;
}

int EcalTPGParamBuilder::getGCTRegionEta(int tteta) {
  int gcteta = 0;
  if (tteta > 0)
    gcteta = (tteta - 1) / 4 + 11;
  else if (tteta < 0)
    gcteta = (tteta + 1) / 4 + 10;
  return gcteta;
}

std::string EcalTPGParamBuilder::getDet(int tcc) {
  std::stringstream sdet;

  if (tcc > 36 && tcc < 55)
    sdet << "EB-" << tcc - 36;
  else if (tcc >= 55 && tcc < 73)
    sdet << "EB+" << tcc - 54;
  else if (tcc <= 36)
    sdet << "EE-";
  else
    sdet << "EE+";

  if (tcc <= 36 || tcc >= 73) {
    if (tcc >= 73)
      tcc -= 72;
    if (tcc == 1 || tcc == 18 || tcc == 19 || tcc == 36)
      sdet << 7;
    else if (tcc == 2 || tcc == 3 || tcc == 20 || tcc == 21)
      sdet << 8;
    else if (tcc == 4 || tcc == 5 || tcc == 22 || tcc == 23)
      sdet << 9;
    else if (tcc == 6 || tcc == 7 || tcc == 24 || tcc == 25)
      sdet << 1;
    else if (tcc == 8 || tcc == 9 || tcc == 26 || tcc == 27)
      sdet << 2;
    else if (tcc == 10 || tcc == 11 || tcc == 28 || tcc == 29)
      sdet << 3;
    else if (tcc == 12 || tcc == 13 || tcc == 30 || tcc == 31)
      sdet << 4;
    else if (tcc == 14 || tcc == 15 || tcc == 32 || tcc == 33)
      sdet << 5;
    else if (tcc == 16 || tcc == 17 || tcc == 34 || tcc == 35)
      sdet << 6;
  }

  return sdet.str();
}

std::pair<std::string, int> EcalTPGParamBuilder::getCrate(int tcc) {
  std::stringstream crate;
  std::string pos;
  int slot = 0;

  crate << "S2D";
  if (tcc >= 40 && tcc <= 42) {
    crate << "02d";
    slot = 5 + (tcc - 40) * 6;
  }
  if (tcc >= 43 && tcc <= 45) {
    crate << "03d";
    slot = 5 + (tcc - 43) * 6;
  }
  if (tcc >= 37 && tcc <= 39) {
    crate << "04d";
    slot = 5 + (tcc - 37) * 6;
  }
  if (tcc >= 52 && tcc <= 54) {
    crate << "06d";
    slot = 5 + (tcc - 52) * 6;
  }
  if (tcc >= 46 && tcc <= 48) {
    crate << "07d";
    slot = 5 + (tcc - 46) * 6;
  }
  if (tcc >= 49 && tcc <= 51) {
    crate << "08d";
    slot = 5 + (tcc - 49) * 6;
  }
  if (tcc >= 58 && tcc <= 60) {
    crate << "02h";
    slot = 5 + (tcc - 58) * 6;
  }
  if (tcc >= 61 && tcc <= 63) {
    crate << "03h";
    slot = 5 + (tcc - 61) * 6;
  }
  if (tcc >= 55 && tcc <= 57) {
    crate << "04h";
    slot = 5 + (tcc - 55) * 6;
  }
  if (tcc >= 70 && tcc <= 72) {
    crate << "06h";
    slot = 5 + (tcc - 70) * 6;
  }
  if (tcc >= 64 && tcc <= 66) {
    crate << "07h";
    slot = 5 + (tcc - 64) * 6;
  }
  if (tcc >= 67 && tcc <= 69) {
    crate << "08h";
    slot = 5 + (tcc - 67) * 6;
  }

  if (tcc >= 76 && tcc <= 81) {
    crate << "02l";
    if (tcc % 2 == 0)
      slot = 2 + (tcc - 76) * 3;
    else
      slot = 4 + (tcc - 77) * 3;
  }
  if (tcc >= 94 && tcc <= 99) {
    crate << "02l";
    if (tcc % 2 == 0)
      slot = 3 + (tcc - 94) * 3;
    else
      slot = 5 + (tcc - 95) * 3;
  }

  if (tcc >= 22 && tcc <= 27) {
    crate << "03l";
    if (tcc % 2 == 0)
      slot = 2 + (tcc - 22) * 3;
    else
      slot = 4 + (tcc - 23) * 3;
  }
  if (tcc >= 4 && tcc <= 9) {
    crate << "03l";
    if (tcc % 2 == 0)
      slot = 3 + (tcc - 4) * 3;
    else
      slot = 5 + (tcc - 5) * 3;
  }

  if (tcc >= 82 && tcc <= 87) {
    crate << "07l";
    if (tcc % 2 == 0)
      slot = 2 + (tcc - 82) * 3;
    else
      slot = 4 + (tcc - 83) * 3;
  }
  if (tcc >= 100 && tcc <= 105) {
    crate << "07l";
    if (tcc % 2 == 0)
      slot = 3 + (tcc - 100) * 3;
    else
      slot = 5 + (tcc - 101) * 3;
  }

  if (tcc >= 28 && tcc <= 33) {
    crate << "08l";
    if (tcc % 2 == 0)
      slot = 2 + (tcc - 28) * 3;
    else
      slot = 4 + (tcc - 29) * 3;
  }
  if (tcc >= 10 && tcc <= 15) {
    crate << "08l";
    if (tcc % 2 == 0)
      slot = 3 + (tcc - 10) * 3;
    else
      slot = 5 + (tcc - 11) * 3;
  }

  if (tcc == 34) {
    crate << "04l";
    slot = 2;
  }
  if (tcc == 16) {
    crate << "04l";
    slot = 3;
  }
  if (tcc == 35) {
    crate << "04l";
    slot = 4;
  }
  if (tcc == 17) {
    crate << "04l";
    slot = 5;
  }
  if (tcc == 36) {
    crate << "04l";
    slot = 8;
  }
  if (tcc == 18) {
    crate << "04l";
    slot = 9;
  }
  if (tcc == 19) {
    crate << "04l";
    slot = 10;
  }
  if (tcc == 1) {
    crate << "04l";
    slot = 11;
  }
  if (tcc == 20) {
    crate << "04l";
    slot = 14;
  }
  if (tcc == 2) {
    crate << "04l";
    slot = 15;
  }
  if (tcc == 21) {
    crate << "04l";
    slot = 16;
  }
  if (tcc == 3) {
    crate << "04l";
    slot = 17;
  }

  if (tcc == 88) {
    crate << "06l";
    slot = 2;
  }
  if (tcc == 106) {
    crate << "06l";
    slot = 3;
  }
  if (tcc == 89) {
    crate << "06l";
    slot = 4;
  }
  if (tcc == 107) {
    crate << "06l";
    slot = 5;
  }
  if (tcc == 90) {
    crate << "06l";
    slot = 8;
  }
  if (tcc == 108) {
    crate << "06l";
    slot = 9;
  }
  if (tcc == 73) {
    crate << "06l";
    slot = 10;
  }
  if (tcc == 91) {
    crate << "06l";
    slot = 11;
  }
  if (tcc == 74) {
    crate << "06l";
    slot = 14;
  }
  if (tcc == 92) {
    crate << "06l";
    slot = 15;
  }
  if (tcc == 75) {
    crate << "06l";
    slot = 16;
  }
  if (tcc == 93) {
    crate << "06l";
    slot = 17;
  }

  return std::pair<std::string, int>(crate.str(), slot);
}
