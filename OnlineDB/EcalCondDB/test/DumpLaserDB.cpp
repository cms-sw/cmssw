/*******************************************

 DumpLaserDB.cpp
 Author: Giovanni.Organtini@roma1.infn.it
 Date:   may 2011

 Description: reads the laser DB and builds
   a plain root ntuple with primitives and
   associated quantities.

 ******************************************/
#include <climits>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>

#include "boost/program_options.hpp"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"

typedef struct lmffdata {
  int detId;
  int logic_id;
  int eta;
  int phi;
  int x;
  int y;
  int z;
  int nevts;
  int quality;
  float pnMean;
  float pnRMS;
  float pnM3;
  float pnAoverpnBMean;
  float pnAoverpnBRMS;
  float pnAoverpnBM3;
  int pnFlag;
  float Mean;
  float RMS;
  float M3;
  float APDoverPNAMean;
  float APDoverPNARMS;
  float APDoverPNAM3;
  float APDoverPNBMean;
  float APDoverPNBRMS;
  float APDoverPNBM3;
  float APDoverPNMean;
  float APDoverPNRMS;
  float APDoverPNM3;
  float alpha;
  float beta;
  int flag;
} LMFData;

typedef struct matacq {
  int fit_method;
  float mtq_ampl;
  float mtq_time;
  float mtq_rise;
  float mtq_fwhm;
  float mtq_fw20;
  float mtq_fw80;
  float mtq_sliding;
} LMFMtq;

typedef struct laser_config {
  int wavelength;
  float vfe_gain;
  float pn_gain;
  float lsr_power;
  float lsr_attenuator;
  float lsr_current;
  float lsr_delay_1;
  float lsr_delay_2;
} LMFLaserConfig;

typedef struct DetIdData {
  int hashedIndex;
  int eta;
  int phi;
  int x;
  int y;
  int z;
} detIdData;

class CondDBApp {
public:
  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(std::string sid, std::string user, std::string pass, run_t r1, run_t r2) {
    try {
      std::cout << "Making connection to " << sid << " using username " << user << std::flush;
      econn = new EcalCondDBInterface(sid, user, pass);
      std::cout << "Done." << std::endl;
      std::cout << "Getting data for run" << std::flush;
      if (r2 <= 0) {
        std::cout << " " << r1 << std::endl;
      } else {
        std::cout << "s between " << r1 << " and " << r2 << std::endl;
      }
      run_min = r1;
      run_max = r2;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      exit(-1);
    }
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() {
    tree->Write();
    tfile->Close();
    delete econn;
  }

  void zero(std::vector<LMFMtq> &lmfmtq) {
    for (unsigned int i = 0; i < lmfmtq.size(); i++) {
      lmfmtq[i].fit_method = -1;
      lmfmtq[i].mtq_ampl = -1000;
      lmfmtq[i].mtq_time = -1000;
      lmfmtq[i].mtq_rise = -1000;
      lmfmtq[i].mtq_fwhm = -1000;
      lmfmtq[i].mtq_fw20 = -1000;
      lmfmtq[i].mtq_fw80 = -1000;
      lmfmtq[i].mtq_sliding = -1000;
    }
  }

  void zero(std::vector<LMFLaserConfig> &lmfConf) {
    for (unsigned int i = 0; i < lmfConf.size(); i++) {
      lmfConf[i].wavelength = -1;
      lmfConf[i].vfe_gain = -1000;
      lmfConf[i].pn_gain = -1000;
      lmfConf[i].lsr_power = -1000;
      lmfConf[i].lsr_attenuator = -1000;
      lmfConf[i].lsr_current = -1000;
      lmfConf[i].lsr_delay_1 = -1000;
      lmfConf[i].lsr_delay_2 = -1000;
    }
  }

  void zero(std::vector<LMFData> &lmfdata) {
    for (unsigned int i = 0; i < lmfdata.size(); i++) {
      lmfdata[i].detId = -1;
      lmfdata[i].logic_id = -1;
      lmfdata[i].eta = -1000;
      lmfdata[i].phi = -1000;
      lmfdata[i].x = -1000;
      lmfdata[i].y = -1000;
      lmfdata[i].z = -1000;
      lmfdata[i].nevts = 0;
      lmfdata[i].quality = -1;
      lmfdata[i].pnMean = -1000.;
      lmfdata[i].pnRMS = -1000.;
      lmfdata[i].pnM3 = -1000.;
      lmfdata[i].pnAoverpnBMean = -1000.;
      lmfdata[i].pnAoverpnBRMS = -1000.;
      lmfdata[i].pnAoverpnBM3 = -1000.;
      lmfdata[i].pnFlag = -1000;
      lmfdata[i].Mean = -1000.;
      lmfdata[i].RMS = -1000.;
      lmfdata[i].M3 = -1000.;
      lmfdata[i].APDoverPNAMean = -1000.;
      lmfdata[i].APDoverPNARMS = -1000.;
      lmfdata[i].APDoverPNAM3 = -1000.;
      lmfdata[i].APDoverPNBMean = -1000.;
      lmfdata[i].APDoverPNBRMS = -1000.;
      lmfdata[i].APDoverPNBM3 = -1000.;
      lmfdata[i].APDoverPNMean = -1000.;
      lmfdata[i].APDoverPNRMS = -1000.;
      lmfdata[i].APDoverPNM3 = -1000.;
      lmfdata[i].alpha = -1000.;
      lmfdata[i].beta = -1000.;
      lmfdata[i].flag = -1000;
    }
  }

  detIdData getCoords(int detid) {
    detIdData ret;
    DetId d(detid);
    if (d.subdetId() == EcalBarrel) {
      EBDetId eb(detid);
      ret.hashedIndex = eb.hashedIndex();
      ret.eta = eb.ieta();
      ret.phi = eb.iphi();
      ret.x = -1000;
      ret.y = -1000;
      ret.z = -1000;
    } else {
      EEDetId ee(detid);
      ret.hashedIndex = ee.hashedIndex() + 61200;
      ret.x = ee.ix();
      ret.y = ee.iy();
      ret.z = ee.zside();
      ret.eta = -1000;
      ret.phi = -1000;
    }
    return ret;
  }

  bool userCheck(LMFData d) {
    /*** EXPERTS ONLY ***/
    bool ret = true;
    return ret;
  }

  void doRun() {
    init();
    std::map<int, int> detids = econn->getLogicId2DetIdMap();
    std::cout << "Crystal map got: " << detids.size() << std::endl;
    for (run_t run = run_min; run <= run_max; run++) {
      std::cout << "Analyzing run " << run << std::endl << std::flush;
      run_num = run;
      // for each run collect
      LMFSeqDat s(econn);
      std::map<int, LMFSeqDat> sequences = s.fetchByRunNumber(run);
      std::map<int, LMFSeqDat>::const_iterator si = sequences.begin();
      std::map<int, LMFSeqDat>::const_iterator se = sequences.end();
      while (si != se) {
        // seq start
        seqStart = si->second.getSequenceStart().epoch();
        // seq stop
        seqStop = si->second.getSequenceStop().epoch();
        // seq number
        seqNum = si->second.getSequenceNumber();
        std::cout << std::endl << " Seq. " << seqNum;
        LMFRunIOV riov(econn);
        std::list<LMFRunIOV> run_iovs = riov.fetchBySequence(si->second);
        std::list<LMFRunIOV>::const_iterator ri = run_iovs.begin();
        std::list<LMFRunIOV>::const_iterator re = run_iovs.end();
        while (ri != re) {
          // lmr
          lmr = ri->getLmr();
          std::cout << std::endl << "  LMR " << std::setw(2) << lmr << std::flush;
          // color
          sprintf(&color[0], "%s", ri->getColorShortName().c_str());
          // subrun times
          subrunstart = ri->getSubRunStart().epoch();
          subrunstop = ri->getSubRunEnd().epoch();
          // blue laser configuration for this run
          LMFLaserPulseDat mtqConf(econn, "BLUE");
          mtqConf.setLMFRunIOV(*ri);
          mtqConf.fetch();
          std::list<int> mtqLogicIds = mtqConf.getLogicIds();
          LMFLaserConfigDat laserConfig(econn);
          laserConfig.setLMFRunIOV(*ri);
          laserConfig.fetch();
          std::list<int> laserConfigLogicIds = laserConfig.getLogicIds();
          // *** get data ***
          LMFPrimDat prim(econn, color, "LASER");
          prim.setLMFRunIOV(*ri);
          std::vector<std::string> channels;
          // uncomment the following line to selects just three channels
          /*
	  channels.push_back("2012043034");
	  channels.push_back("2012060033");
	  channels.push_back("2012034040");
	  prim.setWhereClause("(LOGIC_ID = :I1 OR LOGIC_ID = :I2 OR LOGIC_ID = :I3)", channels); // selects only endcap primitives
	  */
          prim.fetch();
          if (!prim.getLogicIds().empty()) {
            LMFRunDat run_dat(econn);
            run_dat.setLMFRunIOV(*ri);
            /* uncomment the following to select only endcaps
	    run_dat.setWhereClause("LOGIC_ID > 2000000000"); // selects only endcap primitives
	    */
            run_dat.fetch();
            LMFPnPrimDat pnPrim(econn, color, "LASER");
            pnPrim.setLMFRunIOV(*ri);
            /* uncomment the following to select only endcaps
	    pnPrim.setWhereClause("LOGIC_ID > 2000000000"); // selects only endcap primitives
	    */
            pnPrim.fetch();
            // *** run dat ***
            std::list<int> logic_ids = run_dat.getLogicIds();
            std::list<int>::const_iterator li = logic_ids.begin();
            std::list<int>::const_iterator le = logic_ids.end();
            int count = 0;
            int xcount = 0;
            std::vector<LMFData> lmfdata(detids.size());
            std::vector<LMFMtq> lmfmtq(detids.size());
            std::vector<LMFLaserConfig> lmflasConf(detids.size());
            zero(lmfdata);
            zero(lmfmtq);
            zero(lmflasConf);
            while (li != le) {
              // here the logic_id's are those of a LMR: transform into crystals
              int logic_id = *li;
              std::vector<EcalLogicID> xtals = econn->getEcalLogicIDForLMR(logic_id);
              std::cout << " Size = " << std::setw(4) << xtals.size();
              for (unsigned int j = 0; j < xtals.size(); j++) {
                int xtal_id = xtals[j].getLogicID();
                int detId = detids[xtal_id];
                detIdData dd = getCoords(detId);
                int index = dd.hashedIndex;
                lmfdata[index].nevts = run_dat.getEvents(logic_id);
                lmfdata[index].detId = detId;
                lmfdata[index].logic_id = xtal_id;
                lmfdata[index].quality = run_dat.getQualityFlag(logic_id);
                lmfdata[index].eta = dd.eta;
                lmfdata[index].phi = dd.phi;
                lmfdata[index].x = dd.x;
                lmfdata[index].y = dd.y;
                lmfdata[index].z = dd.z;

                std::list<int>::iterator logicIdIsThere = std::find(mtqLogicIds.begin(), mtqLogicIds.end(), logic_id);
                if (logicIdIsThere != mtqLogicIds.end()) {
                  lmfmtq[index].fit_method = mtqConf.getFitMethod(logic_id);
                  lmfmtq[index].mtq_ampl = mtqConf.getMTQAmplification(logic_id);
                  lmfmtq[index].mtq_time = mtqConf.getMTQTime(logic_id);
                  lmfmtq[index].mtq_rise = mtqConf.getMTQRise(logic_id);
                  lmfmtq[index].mtq_fwhm = mtqConf.getMTQFWHM(logic_id);
                  lmfmtq[index].mtq_fw20 = mtqConf.getMTQFW20(logic_id);
                  lmfmtq[index].mtq_fw80 = mtqConf.getMTQFW80(logic_id);
                  lmfmtq[index].mtq_sliding = mtqConf.getMTQSliding(logic_id);
                }

                logicIdIsThere = std::find(laserConfigLogicIds.begin(), laserConfigLogicIds.end(), logic_id);
                if (logicIdIsThere != laserConfigLogicIds.end()) {
                  lmflasConf[index].wavelength = laserConfig.getWavelength(logic_id);
                  lmflasConf[index].vfe_gain = laserConfig.getVFEGain(logic_id);
                  lmflasConf[index].pn_gain = laserConfig.getPNGain(logic_id);
                  lmflasConf[index].lsr_power = laserConfig.getLSRPower(logic_id);
                  lmflasConf[index].lsr_attenuator = laserConfig.getLSRAttenuator(logic_id);
                  lmflasConf[index].lsr_current = laserConfig.getLSRCurrent(logic_id);
                  lmflasConf[index].lsr_delay_1 = laserConfig.getLSRDelay1(logic_id);
                  lmflasConf[index].lsr_delay_2 = laserConfig.getLSRDelay1(logic_id);
                }
                xcount++;
              }
              //
              li++;
              count++;
            }
            std::cout << "   RunDat: " << std::setw(4) << count << " (" << std::setw(4) << xcount << ")";
            count = 0;
            xcount = 0;
            /*** pnPrim Dat ***/
            logic_ids.clear();
            logic_ids = pnPrim.getLogicIds();
            li = logic_ids.begin();
            le = logic_ids.end();
            while (li != le) {
              int logic_id = *li;
              std::vector<EcalLogicID> xtals = econn->getEcalLogicIDForLMPN(logic_id);
              for (unsigned int j = 0; j < xtals.size(); j++) {
                int xtal_id = xtals[j].getLogicID();
                int detId = detids[xtal_id];
                detIdData dd = getCoords(detId);
                int index = dd.hashedIndex;
                lmfdata[index].pnMean = pnPrim.getMean(logic_id);
                lmfdata[index].pnRMS = pnPrim.getRMS(logic_id);
                lmfdata[index].pnM3 = pnPrim.getM3(logic_id);
                lmfdata[index].pnAoverpnBMean = pnPrim.getPNAoverBMean(logic_id);
                lmfdata[index].pnAoverpnBRMS = pnPrim.getPNAoverBRMS(logic_id);
                lmfdata[index].pnAoverpnBM3 = pnPrim.getPNAoverBM3(logic_id);
                lmfdata[index].pnFlag = pnPrim.getFlag(logic_id);
                lmfdata[index].detId = detids[xtal_id];
                lmfdata[index].logic_id = xtal_id;
                lmfdata[index].eta = dd.eta;
                lmfdata[index].phi = dd.phi;
                lmfdata[index].x = dd.x;
                lmfdata[index].y = dd.y;
                lmfdata[index].z = dd.z;
                xcount++;
              }
              li++;
              count++;
            }
            std::cout << "   PnDat: " << std::setw(4) << count << " (" << std::setw(4) << xcount << ")";
            count = 0;
            xcount = 0;
            /*** Prm Dat ***/
            logic_ids.clear();
            logic_ids = prim.getLogicIds();
            li = logic_ids.begin();
            le = logic_ids.end();
            while (li != le) {
              int logic_id = *li;
              int detId = detids[logic_id];
              detIdData dd = getCoords(detId);
              int index = dd.hashedIndex;
              lmfdata[index].detId = detId;
              lmfdata[index].logic_id = logic_id;
              lmfdata[index].eta = dd.eta;
              lmfdata[index].phi = dd.phi;
              lmfdata[index].x = dd.x;
              lmfdata[index].y = dd.y;
              lmfdata[index].z = dd.z;
              //
              lmfdata[index].Mean = prim.getMean(logic_id);
              lmfdata[index].RMS = prim.getRMS(logic_id);
              lmfdata[index].M3 = prim.getM3(logic_id);
              lmfdata[index].APDoverPNAMean = prim.getAPDoverAMean(logic_id);
              lmfdata[index].APDoverPNARMS = prim.getAPDoverARMS(logic_id);
              lmfdata[index].APDoverPNAM3 = prim.getAPDoverAM3(logic_id);
              lmfdata[index].APDoverPNBMean = prim.getAPDoverBMean(logic_id);
              lmfdata[index].APDoverPNBRMS = prim.getAPDoverBRMS(logic_id);
              lmfdata[index].APDoverPNBM3 = prim.getAPDoverBM3(logic_id);
              lmfdata[index].APDoverPNMean = prim.getAPDoverPnMean(logic_id);
              lmfdata[index].APDoverPNRMS = prim.getAPDoverPnRMS(logic_id);
              lmfdata[index].APDoverPNM3 = prim.getAPDoverPnM3(logic_id);
              lmfdata[index].alpha = prim.getAlpha(logic_id);
              lmfdata[index].beta = prim.getBeta(logic_id);
              lmfdata[index].flag = prim.getFlag(logic_id);
              li++;
              count++;
              xcount++;
            }
            std::cout << "   PrimDat: " << std::setw(4) << count << " (" << std::setw(4) << xcount << ")";
            // fill the tree
            xcount = 0;
            for (unsigned int i = 0; i < lmfdata.size(); i++) {
              if ((userCheck(lmfdata[i])) && (lmfdata[i].Mean > 0)) {
                detId = lmfdata[i].detId;
                logic_id = lmfdata[i].logic_id;
                eta = lmfdata[i].eta;
                phi = lmfdata[i].phi;
                x = lmfdata[i].x;
                y = lmfdata[i].y;
                z = lmfdata[i].z;
                nevts = lmfdata[i].nevts;
                quality = lmfdata[i].quality;
                pnMean = lmfdata[i].pnMean;
                pnRMS = lmfdata[i].pnRMS;
                pnM3 = lmfdata[i].pnM3;
                pnAoverBMean = lmfdata[i].pnAoverpnBMean;
                pnAoverBRMS = lmfdata[i].pnAoverpnBRMS;
                pnAoverBM3 = lmfdata[i].pnAoverpnBM3;
                pnFlag = lmfdata[i].pnFlag;
                Mean = lmfdata[i].Mean;
                RMS = lmfdata[i].RMS;
                M3 = lmfdata[i].M3;
                APDoverPNAMean = lmfdata[i].APDoverPNAMean;
                APDoverPNARMS = lmfdata[i].APDoverPNARMS;
                APDoverPNAM3 = lmfdata[i].APDoverPNAM3;
                APDoverPNBMean = lmfdata[i].APDoverPNBMean;
                APDoverPNBRMS = lmfdata[i].APDoverPNBRMS;
                APDoverPNBM3 = lmfdata[i].APDoverPNBM3;
                APDoverPNMean = lmfdata[i].APDoverPNMean;
                APDoverPNRMS = lmfdata[i].APDoverPNRMS;
                APDoverPNM3 = lmfdata[i].APDoverPNM3;
                alpha = lmfdata[i].alpha;
                beta = lmfdata[i].beta;
                flag = lmfdata[i].flag;
                // matacq variables
                fit_method = lmfmtq[i].fit_method;
                mtq_ampl = lmfmtq[i].mtq_ampl;
                mtq_time = lmfmtq[i].mtq_time;
                mtq_rise = lmfmtq[i].mtq_rise;
                mtq_fwhm = lmfmtq[i].mtq_fwhm;
                mtq_fw20 = lmfmtq[i].mtq_fw20;
                mtq_fw80 = lmfmtq[i].mtq_fw80;
                mtq_sliding = lmfmtq[i].mtq_sliding;
                // laser conf variables
                wavelength = lmflasConf[i].wavelength;
                vfe_gain = lmflasConf[i].vfe_gain;
                pn_gain = lmflasConf[i].pn_gain;
                lsr_power = lmflasConf[i].lsr_power;
                lsr_attenuator = lmflasConf[i].lsr_attenuator;
                lsr_current = lmflasConf[i].lsr_current;
                lsr_delay_1 = lmflasConf[i].lsr_delay_1;
                lsr_delay_2 = lmflasConf[i].lsr_delay_2;

                tree->Fill();
                xcount++;
              }
            }
            std::cout << " Tree: " << xcount;
          }
          ri++;
        }
        si++;
      }
      std::cout << std::endl;
    }
  }

private:
  CondDBApp() = delete;  // hidden default constructor
  void init();
  TTree *tree;
  TFile *tfile;
  EcalCondDBInterface *econn;
  run_t run_min;
  run_t run_max;

  int run_num;
  uint64_t seqStart;
  uint64_t seqStop;
  int seqNum;
  int lmr;
  char color[35];
  int nevts;
  uint64_t subrunstart;
  uint64_t subrunstop;
  int detId;
  int logic_id;
  int eta;
  int phi;
  int x;
  int y;
  int z;
  int quality;
  float pnMean;
  float pnRMS;
  float pnM3;
  float pnAoverBMean;
  float pnAoverBRMS;
  float pnAoverBM3;
  float pnFlag;
  float Mean;
  float RMS;
  float M3;
  float APDoverPNAMean;
  float APDoverPNARMS;
  float APDoverPNAM3;
  float APDoverPNBMean;
  float APDoverPNBRMS;
  float APDoverPNBM3;
  float APDoverPNMean;
  float APDoverPNRMS;
  float APDoverPNM3;
  float alpha;
  float beta;
  int flag;
  // matacq
  int fit_method;
  float mtq_ampl;
  float mtq_time;
  float mtq_rise;
  float mtq_fwhm;
  float mtq_fw20;
  float mtq_fw80;
  float mtq_sliding;
  // laser config
  int wavelength;
  float vfe_gain;
  float pn_gain;
  float lsr_power;
  float lsr_attenuator;
  float lsr_current;
  float lsr_delay_1;
  float lsr_delay_2;
};

void CondDBApp::init() {
  std::stringstream title;
  std::stringstream fname;
  title << "Dump of Laser data from online DB for run";
  fname << "DumpLaserDB";
  if (run_max <= 0) {
    title << " " << run_min;
    fname << "-" << run_min;
    run_max = run_min;
  } else {
    title << "s " << run_min << " - " << run_max;
    fname << "-" << run_min << "-" << run_max;
  }
  fname << ".root";
  std::cout << "Building tree " << title.str() << " on file " << fname.str() << std::endl;
  tfile = new TFile(fname.str().c_str(), "RECREATE", title.str().c_str());
  tree = new TTree("LDB", title.str().c_str());
  tree->Branch("run", &run_num, "run/I");
  tree->Branch("seqStart", &seqStart, "seqStart/l");
  tree->Branch("seqStop", &seqStop, "seqStop/l");
  tree->Branch("seqNum", &seqNum, "seqNum/I");
  tree->Branch("lmr", &lmr, "lmr/I");
  tree->Branch("nevts", &nevts, "nevts/I");
  tree->Branch("color", &color, "color/C");
  tree->Branch("subrunstart", &subrunstart, "subrunstart/l");
  tree->Branch("subrunstop", &subrunstart, "subrunstop/l");
  tree->Branch("detId", &detId, "detId/I");
  tree->Branch("logic_id", &logic_id, "logic_id/I");
  tree->Branch("eta", &eta, "eta/I");
  tree->Branch("phi", &phi, "phi/I");
  tree->Branch("x", &x, "x/I");
  tree->Branch("y", &y, "y/I");
  tree->Branch("z", &z, "z/I");
  tree->Branch("quality", &quality, "quality/I");
  tree->Branch("pnMean", &pnMean, "pnMean/F");
  tree->Branch("pnRMS", &pnRMS, "pnRMS/F");
  tree->Branch("pnM3", &pnM3, "pnM3/F");
  tree->Branch("pnAoverBMean", &pnAoverBMean, "pnAoverBMean/F");
  tree->Branch("pnAoverBRMS", &pnAoverBRMS, "pnAoverBRMS/F");
  tree->Branch("pnAoverBM3", &pnAoverBM3, "pnAoverBM3/F");
  tree->Branch("pnFlag", &pnFlag, "pnFlag/F");
  tree->Branch("Mean", &pnMean, "Mean/F");
  tree->Branch("RMS", &pnRMS, "RMS/F");
  tree->Branch("M3", &pnM3, "M3/F");
  tree->Branch("APDoverPnAMean", &APDoverPNAMean, "APDoverPNAMean/F");
  tree->Branch("APDoverPnARMS", &APDoverPNARMS, "APDoverPNARMS/F");
  tree->Branch("APDoverPnAM3", &APDoverPNAM3, "APDoverPNAM3/F");
  tree->Branch("APDoverPnBMean", &APDoverPNBMean, "APDoverPNBMean/F");
  tree->Branch("APDoverPnBRMS", &APDoverPNBRMS, "APDoverPNBRMS/F");
  tree->Branch("APDoverPnBM3", &APDoverPNBM3, "APDoverPNBM3/F");
  tree->Branch("APDoverPnMean", &APDoverPNMean, "APDoverPNMean/F");
  tree->Branch("APDoverPnRMS", &APDoverPNRMS, "APDoverPNRMS/F");
  tree->Branch("APDoverPnM3", &APDoverPNM3, "APDoverPNM3/F");
  tree->Branch("alpha", &alpha, "alpha/F");
  tree->Branch("beta", &beta, "beta/F");
  tree->Branch("flag", &flag, "flag/I");

  tree->Branch("fit_method", &fit_method, "fit_method/I");
  tree->Branch("mtq_ampl", &mtq_ampl, "mtq_ampl/F");
  tree->Branch("mtq_time", &mtq_time, "mtq_time/F");
  tree->Branch("mtq_rise", &mtq_rise, "mtq_rise/F");
  tree->Branch("mtq_fwhm", &mtq_fwhm, "mtq_fwhm/F");
  tree->Branch("mtq_fw20", &mtq_fw20, "mtq_fw20/F");
  tree->Branch("mtq_fw80", &mtq_fw80, "mtq_fw80/F");
  tree->Branch("mtq_sliding", &mtq_sliding, "mtq_sliding/F");

  tree->Branch("wavelength", &wavelength, "wavelength/I");
  tree->Branch("vfe_gain", &vfe_gain, "vfe_gain/F");
  tree->Branch("pn_gain", &pn_gain, "pn_gain/F");
  tree->Branch("lsr_power", &lsr_power, "lsr_power/F");
  tree->Branch("lsr_attenuator", &lsr_attenuator, "lsr_attenuator/F");
  tree->Branch("lsr_current", &lsr_current, "lsr_current/F");
  tree->Branch("lsr_delay_1", &lsr_delay_1, "lsr_delay_1/F");
  tree->Branch("lsr_delay_2", &lsr_delay_2, "lsr_delay_2/F");

  std::cout << "Tree created" << std::endl;
}

void help() {
  std::cout << "DumpLaserDB\n";
  std::cout << " Reads laser DB and build a root ntuple.\n";
  std::cout << std::endl;
  std::cout << " Loops on runs between run_min and (if present) run_max.\n";
  std::cout << " For each run loops on each Laser Sequence. Each sequence\n";
  std::cout << " contains up to 92 LMR.\n";
  std::cout << " For each region gets:\n";
  std::cout << "   a) run data, such as the number of events acquired\n";
  std::cout << "   b) pn primitives\n";
  std::cout << "   c) laser primitives\n";
  std::cout << std::endl;
  std::cout << " During execution it shows the progress on Sequences/LMR's\n";
  std::cout << " For each LMR shows its size (crystals belonging to it).\n";
  std::cout << " For each group of data shows the number of rows found\n";
  std::cout << " in the DB and the no. of channels (in parenthesis).\n";
  std::cout << " Usually there is 1 row per RunDat (common to all channels\n";
  std::cout << " in the LMR) and as many channels as crystals belonging to\n";
  std::cout << " the LMR. There are between between 8 and 10 rows per PnDat\n";
  std::cout << " grouped in such a way that the number of channels must be\n";
  std::cout << " equal to the crystals in the supermodule. Primitives are \n";
  std::cout << " less or equal to the number of channels in the LMR.\n";
  std::cout << " Finally, the number of rows in the Tree is shown\n\n";
}

int main(int argc, char *argv[]) {
  namespace po = boost::program_options;

  int run1 = 0;
  int run2 = 0;

  std::string sid = "CMS_ORCOFF_PROD";
  std::string user = "CMS_ECAL_R";
  std::string pass = "";

  std::string confFile = "";

  try {
    // options definition
    po::options_description desc("Allowed options");
    desc.add_options()("help", "shows help message")(
        "rmin", po::value<int>()->default_value(0), "set minimum run number to analyze (mandatory)")(
        "rmax", po::value<int>()->default_value(0), "set maximum run number to analyze (optional)")(
        "sid", po::value<std::string>()->default_value(sid), "Oracle System ID (SID) (defaulted)")(
        "user", po::value<std::string>()->default_value(user), "SID user (defaulted)")(
        "pass", po::value<std::string>(), "password (mandatory)")(
        "file", po::value<std::string>(), "configuration file name (optional)");

    po::positional_options_description p;
    p.add("sid", 1);
    p.add("user", 1);
    p.add("pass", 1);
    p.add("rmin", 1);
    p.add("rmax", 1);

    // parsing and decoding options
    po::variables_map vm;
    //    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      help();
      return 1;
    }

    if (vm.count("rmin") && (vm["rmin"].as<int>() > 0)) {
      run1 = vm["rmin"].as<int>();
      run2 = vm["rmax"].as<int>();
    } else {
      std::cout << desc << std::endl;
      return 1;
    }
    if (vm.count("rmax")) {
      run2 = vm["rmax"].as<int>();
    }
    if (vm.count("sid")) {
      sid = vm["sid"].as<std::string>();
    }
    if (vm.count("user")) {
      user = vm["user"].as<std::string>();
    }
    if (vm.count("pass")) {
      pass = vm["pass"].as<std::string>();
    } else {
      std::cout << desc << std::endl;
      return 1;
    }
    if (vm.count("file")) {
      confFile = vm["file"].as<std::string>();
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  try {
    CondDBApp app(sid, user, pass, run1, run2);
    app.doRun();
  } catch (std::exception &e) {
    std::cout << "ERROR:  " << e.what() << std::endl;
  } catch (...) {
    std::cout << "Unknown error caught" << std::endl;
  }

  std::cout << "All Done." << std::endl;

  return 0;
}
