#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "Utilities/Xerces/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include "CondTools/Ecal/interface/EcalSRPHandler.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

#include "OnlineDB/EcalCondDB/interface/RunConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/ODRunConfigInfo.h"
#include "OnlineDB/EcalCondDB/interface/ODRunConfigSeqInfo.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadTTInfo.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSRCondTools.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;

popcon::EcalSRPHandler::EcalSRPHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalSRPHandler")) {
  m_firstRun = (unsigned long)atoi(ps.getParameter<std::string>("firstRun").c_str());
  m_lastRun = (unsigned long)atoi(ps.getParameter<std::string>("lastRun").c_str());
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_location = ps.getParameter<std::string>("location");
  m_runtype = ps.getParameter<std::string>("runtype");
  m_gentag = ps.getParameter<std::string>("gentag");
  m_i_tag = "";
  m_debug = ps.getParameter<bool>("debug");
  m_i_version = 0;
}

popcon::EcalSRPHandler::~EcalSRPHandler() {}

void popcon::EcalSRPHandler::getNewObjects() {
  std::ostringstream ss;
  ss << "ECAL ";

  unsigned long long max_since = 1;

  // this is the last inserted run
  max_since = tagInfo().lastInterval.first;

  // this is the last object in the DB
  Ref srp_db = lastPayload();
  if (m_debug)
    std::cout << " max_since : " << max_since << "\n retrieved last payload " << std::endl;

  // we copy the last valid record to a temporary object
  EcalSRSettings* sref = new EcalSRSettings();
  sref->deltaEta_ = srp_db->deltaEta_;
  sref->deltaPhi_ = srp_db->deltaPhi_;
  sref->ecalDccZs1stSample_ = srp_db->ecalDccZs1stSample_;
  sref->ebDccAdcToGeV_ = srp_db->ebDccAdcToGeV_;
  sref->eeDccAdcToGeV_ = srp_db->eeDccAdcToGeV_;
  sref->dccNormalizedWeights_ = srp_db->dccNormalizedWeights_;
  sref->symetricZS_ = srp_db->symetricZS_;
  sref->srpLowInterestChannelZS_ = srp_db->srpLowInterestChannelZS_;
  sref->srpHighInterestChannelZS_ = srp_db->srpHighInterestChannelZS_;
  sref->actions_ = srp_db->actions_;
  sref->tccMasksFromConfig_ = srp_db->tccMasksFromConfig_;
  sref->srpMasksFromConfig_ = srp_db->srpMasksFromConfig_;
  sref->dccMasks_ = srp_db->dccMasks_;
  sref->srfMasks_ = srp_db->srfMasks_;
  sref->substitutionSrfs_ = srp_db->substitutionSrfs_;
  sref->testerTccEmuSrpIds_ = srp_db->testerTccEmuSrpIds_;
  sref->testerSrpEmuSrpIds_ = srp_db->testerSrpEmuSrpIds_;
  sref->testerDccTestSrpIds_ = srp_db->testerDccTestSrpIds_;
  sref->testerSrpTestSrpIds_ = srp_db->testerSrpTestSrpIds_;
  sref->bxOffsets_ = srp_db->bxOffsets_;
  sref->bxGlobalOffset_ = srp_db->bxGlobalOffset_;
  sref->automaticMasks_ = srp_db->automaticMasks_;
  sref->automaticSrpSelect_ = srp_db->automaticSrpSelect_;

  // now read the actual status from the online DB

  std::cout << "Retrieving DAQ status from OMDS DB ... " << std::endl;
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  std::cout << "Connection done" << std::endl;

  if (!econn) {
    std::cout << " Problem with OMDS: connection parameters " << m_sid << "/" << m_user << std::endl;
    throw cms::Exception("OMDS not available");
  }

  LocationDef my_locdef;
  my_locdef.setLocation(m_location);

  RunTypeDef my_rundef;
  my_rundef.setRunType(m_runtype);

  RunTag my_runtag;
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);

  // range of validity
  int min_run = 0;
  if (m_firstRun <= (unsigned long)max_since) {
    min_run = (int)max_since + 1;  // we have to add 1 to the last transferred one
  } else {
    min_run = (int)m_firstRun;
  }

  int max_run = (int)m_lastRun;
  if (m_debug)
    std::cout << "min_run " << min_run << " max_run " << max_run << std::endl;

  std::ofstream fout;
  if (m_debug) {
    char outfile[800];
    sprintf(outfile, "SRP_run%d.txt", min_run);
    fout.open(outfile, std::fstream::out);
    fout << " ORCOFF last run max_since : " << max_since << std::endl;
    PrintPayload(*sref, fout);
  }

  RunList my_list;
  my_list = econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef);

  std::vector<RunIOV> run_vec = my_list.getRuns();
  int num_runs = run_vec.size();

  if (m_debug) {
    fout << " number of runs is : " << num_runs << std::endl;
  }
  unsigned long irun = 0;
  if (num_runs > 0) {
    int fe_conf_id_old = 0;
    //    int krmax = std::min(num_runs, 100);
    //    for(int kr = 0; kr < krmax; kr++) {
    for (int kr = 0; kr < num_runs; kr++) {  // allow any number of runs, exit on transfert size
      irun = (unsigned long)run_vec[kr].getRunNumber();
      std::string geneTag = run_vec[kr].getRunTag().getGeneralTag();
      if (geneTag != "GLOBAL") {
        if (m_debug)
          fout << "\n New run " << irun << " with tag " << geneTag << " giving up " << std::endl;
        continue;
      }
      if (m_debug)
        fout << "\n New run " << irun << " geneTag " << geneTag << std::endl;

      // First, RUN_CONFIGURATION
      std::map<EcalLogicID, RunConfigDat> dataset;
      econn->fetchDataSet(&dataset, &run_vec[kr]);
      std::string myconfig_tag = "";
      int myconfig_version = 0;
      std::map<EcalLogicID, RunConfigDat>::const_iterator it;
      if (dataset.size() != 1) {
        std::cout << "\n\n run " << irun << " strange number of dataset " << dataset.size() << std::endl;
        if (m_debug)
          fout << "\n\n run " << irun << " strange number of dataset " << dataset.size() << " giving up " << std::endl;
        continue;
      }

      it = dataset.begin();
      RunConfigDat dat = it->second;
      myconfig_tag = dat.getConfigTag();
      //      if (myconfig_tag.substr(0, 15) == "ZeroSuppression") {
      if (myconfig_tag.substr(0, 15) == "ZeroSuppression" || myconfig_tag.substr(0, 11) == "FullReadout" ||
          myconfig_tag.substr(0, 11) == "AlmostEmpty") {
        if (m_debug)
          fout << " run " << irun << " tag " << myconfig_tag << " giving up " << std::endl;
        continue;
      }

      // Now  FE_DAQ_CONFIG
      typedef std::map<EcalLogicID, RunFEConfigDat>::const_iterator feConfIter;
      std::map<EcalLogicID, RunFEConfigDat> feconfig;
      econn->fetchDataSet(&feconfig, &run_vec[kr]);
      if (feconfig.size() != 1) {
        if (m_debug)
          fout << "\n\n run " << irun << " strange number of FE config " << feconfig.size() << " giving up "
               << std::endl;
        continue;
      }
      RunFEConfigDat rd_fe;
      int fe_conf_id = 0;
      feConfIter p = feconfig.begin();
      rd_fe = p->second;
      fe_conf_id = rd_fe.getConfigId();

      myconfig_version = dat.getConfigVersion();
      if (m_debug)
        fout << " run " << irun << " tag " << myconfig_tag << " version " << myconfig_version << " Fe config "
             << fe_conf_id << std::endl;
      // here we should check if it is the same as previous run.
      if (myconfig_tag != m_i_tag || myconfig_version != m_i_version || fe_conf_id != fe_conf_id_old) {
        if (m_debug)
          fout << " run= " << irun << " different tag  ... retrieving last config set from DB" << std::endl;

        bool FromCLOB = false;
        EcalSRSettings* sr = new EcalSRSettings;
        sr->ebDccAdcToGeV_ = 0.035;
        sr->eeDccAdcToGeV_ = 0.060;
        sr->symetricZS_.push_back(0);

        ODRunConfigInfo od_run_info;
        od_run_info.setTag(myconfig_tag);
        od_run_info.setVersion(myconfig_version);

        try {
          econn->fetchConfigSet(&od_run_info);
          int config_id = od_run_info.getId();

          ODRunConfigSeqInfo seq;
          seq.setEcalConfigId(config_id);
          seq.setSequenceNumber(0);
          econn->fetchConfigSet(&seq);
          int sequenceid = seq.getSequenceId();

          ODEcalCycle ecal_cycle;
          ecal_cycle.setSequenceId(sequenceid);
          econn->fetchConfigSet(&ecal_cycle);
          int cycle_id = ecal_cycle.getId();
          int srp_id = ecal_cycle.getSRPId();
          if (srp_id == 0) {
            if (m_debug)
              fout << " no SRP config for this run, give up " << std::endl;
            delete sr;
            continue;  //  no SRP config
          }
          int dcc_id = ecal_cycle.getDCCId();
          if (m_debug)
            fout << " cycleid " << cycle_id << " SRP id " << srp_id << " DCC id " << dcc_id << std::endl;
          /**************************/
          /*          SRP           */
          /**************************/
          ODSRPConfig srp;
          srp.setId(srp_id);
          econn->fetchConfigSet(&srp);

          unsigned char* cbuffer = srp.getSRPClob();
          unsigned int SRPClobSize = srp.getSRPClobSize();
          std::string srpstr((char*)cbuffer, SRPClobSize);
          std::string SRPClob = srpstr;
          std::fstream myfile;
          myfile.open("srp.txt", std::fstream::out);
          for (std::string::iterator it = SRPClob.begin(); it < SRPClob.end(); it++)
            myfile << *it;
          myfile.close();
          std::ifstream f("srp.txt");
          if (!f.good()) {
            throw cms::Exception("EcalSRPHandler") << " Failed to open file srp.txt";
            if (m_debug)
              fout << " Failed to open file srp.txt" << std::endl;
          }
          EcalSRCondTools::importSrpConfigFile(*sr, f, m_debug);
          f.close();
          int rv = system("rm srp.txt");
          if (m_debug && rv != 0)
            fout << "rm srp.txt result code: " << rv << "\n";

          sr->bxGlobalOffset_ = srp.getSRP0BunchAdjustPosition();
          sr->automaticMasks_ = srp.getAutomaticMasks();
          sr->automaticSrpSelect_ = srp.getAutomaticSrpSelect();
          /**************************/
          /*          DCC           */
          /**************************/
          ODDCCConfig dcc;
          dcc.setId(dcc_id);
          econn->fetchConfigSet(&dcc);
          std::string weightsMode = dcc.getDCCWeightsMode();
          if (m_debug)
            fout << " DCC weightsMode " << weightsMode << std::endl
                 << " weight size beg " << sr->dccNormalizedWeights_.size() << std::endl;
          if (weightsMode == "CLOB") {
            FromCLOB = true;
            if (m_debug)
              fout << " will read weights from DCC CLOB " << std::endl;
          }
          cbuffer = dcc.getDCCClob();
          unsigned int DCCClobSize = dcc.getDCCClobSize();
          std::string dccstr((char*)cbuffer, DCCClobSize);
          std::string DCCClob = dccstr;
          std::ostringstream osd;
          osd << "dcc.txt";
          std::string fname = osd.str();
          myfile.open(fname.c_str(), std::fstream::out);
          for (std::string::iterator it = DCCClob.begin(); it < DCCClob.end(); it++)
            myfile << *it;
          myfile.close();
          importDccConfigFile(*sr, fname, FromCLOB);
          if (m_debug)
            fout << " weight size after CLOB " << sr->dccNormalizedWeights_.size() << std::endl;
          rv = system("rm dcc.txt");
          if (m_debug && rv != 0)
            fout << "rm dcc.txt result code: " << rv << "\n";
        } catch (std::exception& e) {
          // we should not come here...
          std::cout << "ERROR: This config does not exist: tag " << myconfig_tag << " version " << myconfig_version
                    << std::endl;
          if (m_debug)
            fout << "ERROR: This config does not exist: tag " << myconfig_tag << " version " << myconfig_version
                 << std::endl;
          //	    m_i_run_number = irun;
        }
        if (m_debug)
          fout << " FromCLOB " << FromCLOB << std::endl;
        if (!FromCLOB) {  // data from FE, we need to get FE_DAQ_CONFIG table
          // reading this configuration
          ODFEDAQConfig myconfig;
          myconfig.setId(fe_conf_id);
          econn->fetchConfigSet(&myconfig);

          // list weights
          int mywei = myconfig.getWeightId();
          if (m_debug)
            fout << " WEI_ID " << mywei << std::endl;

          if (mywei != 0) {
            ODWeightsSamplesDat samp;
            samp.setId(mywei);
            econn->fetchConfigSet(&samp);
            int mySample = samp.getSampleId();
            if (m_debug)
              fout << " SAMPLE_ID " << mySample << std::endl;
            sr->ecalDccZs1stSample_.push_back(mySample);

            ODWeightsDat weights;
            weights.setId(mywei);
            econn->fetchConfigSet(&weights);

            std::vector<std::vector<float> > my_dccw = weights.getWeight();
            int imax = my_dccw.size();
            if (m_debug)
              fout << " weight size before check " << imax << std::endl;
            if (imax == 75848) {  // all the channel present. Check for change
              bool WeightsChange = false, WeightsChangeEB = false, WeightsChangeEE = false;
              for (int i = 1; i < 61200; i++)  // EB
                for (int ich = 0; ich < 6; ich++)
                  if (my_dccw[i][ich] != my_dccw[0][ich])
                    WeightsChangeEB = true;
              if (m_debug)
                fout << " WeightsChangeEB " << WeightsChangeEB << std::endl;
              for (int i = 61201; i < 75848; i++)  // EE
                for (int ich = 0; ich < 6; ich++)
                  if (my_dccw[i][ich] != my_dccw[61200][ich])
                    WeightsChangeEE = true;
              if (m_debug)
                fout << " WeightsChangeEE " << WeightsChangeEE << std::endl;
              for (int ich = 0; ich < 6; ich++)
                if (my_dccw[0][ich] != my_dccw[61200][ich])
                  WeightsChange = true;
              if (m_debug)
                fout << " WeightsChange " << WeightsChange << std::endl;

              if (WeightsChangeEB || WeightsChangeEE)  // differences between channels, keep them all
                sr->dccNormalizedWeights_ = my_dccw;
              else if (WeightsChange) {  // difference between EB and EE, keep only 1 channel from each
                std::vector<float> dccwRowEB, dccwRowEE;
                for (int ich = 0; ich < 6; ich++) {
                  dccwRowEB.push_back(my_dccw[0][ich]);
                  dccwRowEE.push_back(my_dccw[61200][ich]);
                }
                sr->dccNormalizedWeights_.push_back(dccwRowEB);
                sr->dccNormalizedWeights_.push_back(dccwRowEE);
              } else {  // no difference keep only one
                std::vector<float> dccwRow;
                for (int ich = 0; ich < 6; ich++) {
                  dccwRow.push_back(my_dccw[0][ich]);
                }
                sr->dccNormalizedWeights_.push_back(dccwRow);
                if (m_debug) {
                  fout << " weight ";
                  for (int ich = 0; ich < 6; ich++)
                    fout << " ch " << ich << " " << sr->dccNormalizedWeights_[0][ich];
                  fout << std::endl;
                }
              }
            }  // all channels
            else {
              sr->dccNormalizedWeights_ = my_dccw;
            }
            if (m_debug)
              fout << " weight size after DB " << sr->dccNormalizedWeights_.size() << std::endl;
          }  // WEI_ID != 0
        }    // weights got from FE

        // check if we have found the weights
        if (sr->dccNormalizedWeights_.empty()) {  // use the firmware default weights
          //	      float opt[] = { -383, -383, -372, 279, 479, 380};
          float def[] = {-1215, 20, 297, 356, 308, 232};
          std::vector<float> dccw(def, def + 6);
          if (m_debug)
            fout << " default weights ";
          for (int i = 0; i < 6; i++) {
            if (m_debug)
              fout << " i " << i << " def " << def[i] << " dccw " << dccw[i] << " \n";
          }
          sr->dccNormalizedWeights_.push_back(dccw);  // vector vector
        }
        // normalize online weights
        int imax = sr->dccNormalizedWeights_.size();
        if (m_debug)
          fout << " weight size " << imax << " normalized weights : " << std::endl;
        for (int i = 0; i < imax; i++)
          for (int ich = 0; ich < 6; ich++) {
            sr->dccNormalizedWeights_[i][ich] /= 1024.;
            if (m_debug && i == 0)
              fout << " ch " << ich << " weight " << sr->dccNormalizedWeights_[i][ich] << std::endl;
          }

        /**************************/
        /*  checking for change   */
        /**************************/
        if (m_debug)
          fout << " checking for change " << std::endl;
        bool nochange = true;
        int imaxref, imaxnew;

        if (sref->deltaEta_ != sr->deltaEta_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->deltaEta_.size();
            imaxnew = sr->deltaEta_.size();
            if (imaxref != imaxnew) {
              fout << " deltaEta_ size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->deltaEta_[i] != sr->deltaEta_[i]) {
                  fout << " deltaEta_[" << i << "] ref " << sref->deltaEta_[i] << " now " << sr->deltaEta_[i]
                       << std::endl;
                }
              }
            }
          }
        }

        if (sref->deltaPhi_ != sr->deltaPhi_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->deltaPhi_.size();
            imaxnew = sr->deltaPhi_.size();
            if (imaxref != imaxnew) {
              fout << " deltaPhi size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->deltaPhi_[i] != sr->deltaPhi_[i]) {
                  fout << " deltaPhi[" << i << "] ref " << sref->deltaPhi_[i] << " now " << sr->deltaPhi_[i]
                       << std::endl;
                }
              }
            }
          }
        }

        if (sref->ecalDccZs1stSample_ != sr->ecalDccZs1stSample_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->ecalDccZs1stSample_.size();
            imaxnew = sr->ecalDccZs1stSample_.size();
            if (imaxref != imaxnew) {
              fout << " ecalDccZs1stSample size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->ecalDccZs1stSample_[i] != sr->ecalDccZs1stSample_[i]) {
                  fout << " ecalDccZs1stSample_[" << i << "] ref " << sref->ecalDccZs1stSample_[i] << " now "
                       << sr->ecalDccZs1stSample_[i] << std::endl;
                }
              }
            }
          }
        }

        if (sref->ebDccAdcToGeV_ != sr->ebDccAdcToGeV_ || sref->eeDccAdcToGeV_ != sr->eeDccAdcToGeV_) {
          nochange = false;
          if (m_debug)
            fout << " ebDccAdcToGeV ref " << sref->ebDccAdcToGeV_ << " ee " << sref->eeDccAdcToGeV_ << " now "
                 << sr->ebDccAdcToGeV_ << " ee " << sr->eeDccAdcToGeV_ << std::endl;
        }

        if (sref->dccNormalizedWeights_ != sr->dccNormalizedWeights_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->dccNormalizedWeights_.size();
            imaxnew = sr->dccNormalizedWeights_.size();
            if (imaxref != imaxnew) {
              fout << " dccNormalizedWeights size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              int i = 0;
              for (int ich = 0; ich < 6; ich++) {
                if (sref->dccNormalizedWeights_[i][ich] != sr->dccNormalizedWeights_[i][ich]) {
                  fout << " dccNormalizedWeights_[" << i << "][" << ich << "] ref "
                       << sref->dccNormalizedWeights_[i][ich] << " now " << sr->dccNormalizedWeights_[i][ich]
                       << std::endl;
                }
              }
            }
          }
        }

        if (sref->symetricZS_ != sr->symetricZS_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->symetricZS_.size();
            imaxnew = sr->symetricZS_.size();
            if (imaxref != imaxnew) {
              fout << " symetricZS size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->symetricZS_[i] != sr->symetricZS_[i]) {
                  fout << " symetricZS[" << i << "] ref " << sref->symetricZS_[i] << " now " << sr->symetricZS_[i]
                       << std::endl;
                }
              }
            }
          }
        }

        if (sref->srpLowInterestChannelZS_ != sr->srpLowInterestChannelZS_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->srpLowInterestChannelZS_.size();
            imaxnew = sr->srpLowInterestChannelZS_.size();
            if (imaxref != imaxnew) {
              fout << " srpLowInterestChannelZS size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->srpLowInterestChannelZS_[i] != sr->srpLowInterestChannelZS_[i]) {
                  fout << " srpLowInterestChannelZS[" << i << "] ref " << sref->srpLowInterestChannelZS_[i] << " now "
                       << sr->srpLowInterestChannelZS_[i] << std::endl;
                }
              }
            }
          }
        }

        if (sref->srpHighInterestChannelZS_ != sr->srpHighInterestChannelZS_) {
          nochange = false;
          if (m_debug) {
            imaxref = sref->srpHighInterestChannelZS_.size();
            imaxnew = sr->srpHighInterestChannelZS_.size();
            if (imaxref != imaxnew) {
              fout << " srpHighInterestChannelZS size ref " << imaxref << " now " << imaxnew << std::endl;
            } else {
              for (int i = 0; i < imaxref; i++) {
                if (sref->srpHighInterestChannelZS_[i] != sr->srpHighInterestChannelZS_[i]) {
                  fout << " srpHighInterestChannelZS[" << i << "] ref " << sref->srpHighInterestChannelZS_[i] << " now "
                       << sr->srpHighInterestChannelZS_[i] << std::endl;
                }
              }
            }
          }
        }

        if (sref->actions_ != sr->actions_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 4; i++) {
              if (sref->actions_[i] != sr->actions_[i]) {
                fout << " actions " << i << " ref " << sref->actions_[i] << " now " << sr->actions_[i] << std::endl;
              }
            }
          }
        }

        if (sref->tccMasksFromConfig_ != sr->tccMasksFromConfig_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 108; i++) {
              if (sref->tccMasksFromConfig_[i] != sr->tccMasksFromConfig_[i]) {
                fout << " tccMasks " << i << " ref " << sref->tccMasksFromConfig_[i] << " now "
                     << sr->tccMasksFromConfig_[i] << std::endl;
              }
            }
          }
        }

        if (sref->srpMasksFromConfig_ != sr->srpMasksFromConfig_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 12; i++) {
              for (int ich = 0; ich < 8; ich++) {
                if (sref->srpMasksFromConfig_[i][ich] != sr->srpMasksFromConfig_[i][ich]) {
                  fout << " srpMasks " << i << " " << ich << " ref " << sref->srpMasksFromConfig_[i][ich] << " now "
                       << sr->srpMasksFromConfig_[i][ich] << std::endl;
                }
              }
            }
          }
        }

        if (sref->dccMasks_ != sr->dccMasks_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 54; i++) {
              if (sref->dccMasks_[i] != sr->dccMasks_[i]) {
                fout << " dccMasks " << i << " ref " << sref->dccMasks_[i] << " now " << sr->dccMasks_[i] << std::endl;
              }
            }
          }
        }

        if (sref->srfMasks_ != sr->srfMasks_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 12; i++) {
              if (sref->srfMasks_[i] != sr->srfMasks_[i]) {
                fout << " srfMasks " << i << " ref " << sref->srfMasks_[i] << " now " << sr->srfMasks_[i] << std::endl;
              }
            }
          }
        }

        if (sref->substitutionSrfs_ != sr->substitutionSrfs_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 12; i++) {
              for (int ich = 0; ich < 68; ich++) {
                if (sref->substitutionSrfs_[i][ich] != sr->substitutionSrfs_[i][ich]) {
                  fout << " substitutionSrfs " << i << " " << ich << " ref " << sref->substitutionSrfs_[i][ich]
                       << " now " << sr->substitutionSrfs_[i][ich] << std::endl;
                }
              }
            }
          }
        }

        if (sref->testerTccEmuSrpIds_ != sr->testerTccEmuSrpIds_) {
          nochange = false;
          if (m_debug) {
            for (int i = 0; i < 12; i++) {
              fout << " testerTccEmuSrpIds " << i << " ref " << sref->testerTccEmuSrpIds_[i] << " now "
                   << sr->testerTccEmuSrpIds_[i] << std::endl;
            }
          }
        }

        if (sref->testerSrpEmuSrpIds_ != sr->testerSrpEmuSrpIds_) {
          nochange = false;
          if (m_debug)
            for (int i = 0; i < 12; i++) {
              fout << " testerSrpEmuSrpIds " << i << " ref " << sref->testerSrpEmuSrpIds_[i] << " now "
                   << sr->testerSrpEmuSrpIds_[i] << std::endl;
            }
        }

        if (sref->testerDccTestSrpIds_ != sr->testerDccTestSrpIds_) {
          nochange = false;
          if (m_debug)
            for (int i = 0; i < 12; i++) {
              fout << " testerDccTestSrpIds " << i << " ref " << sref->testerDccTestSrpIds_[i] << " now "
                   << sr->testerDccTestSrpIds_[i] << std::endl;
            }
        }

        if (sref->testerSrpTestSrpIds_ != sr->testerSrpTestSrpIds_) {
          nochange = false;
          if (m_debug)
            for (int i = 0; i < 12; i++) {
              fout << " testerSrpTestSrpIds " << i << " ref " << sref->testerSrpTestSrpIds_[i] << " now "
                   << sr->testerSrpTestSrpIds_[i] << std::endl;
            }
        }

        if (sref->bxOffsets_ != sr->bxOffsets_) {
          nochange = false;
          if (m_debug)
            for (int i = 0; i < 12; i++) {
              fout << " bxOffsets " << i << " ref " << sref->bxOffsets_[i] << " now " << sr->bxOffsets_[i] << std::endl;
            }
        }

        if (sref->bxGlobalOffset_ != sr->bxGlobalOffset_) {
          nochange = false;
          if (m_debug)
            fout << " bxGlobalOffset ref " << sr->bxGlobalOffset_ << " now " << sr->bxGlobalOffset_ << std::endl;
        }

        if (sref->automaticMasks_ != sr->automaticMasks_) {
          nochange = false;
          if (m_debug)
            fout << " automaticMasks ref " << sref->automaticMasks_ << " now " << sr->automaticMasks_ << std::endl;
        }

        if (sref->automaticSrpSelect_ != sr->automaticSrpSelect_) {
          nochange = false;
          if (m_debug)
            fout << " automaticSrpSelect ref " << sref->automaticSrpSelect_ << " now " << sr->automaticSrpSelect_
                 << std::endl;
        }

        if (nochange) {
          if (m_debug)
            fout << " no change has been found " << std::endl;
          std::ostringstream ss;
          ss << "Run=" << irun << "_SRPunchanged_" << std::endl;
          m_userTextLog = ss.str() + ";";
        } else {
          if (m_debug) {
            fout << " Change has been found !\n   New payload :" << std::endl;
            PrintPayload(*sr, fout);
          }
          ChangePayload(*sref, *sr);  // new reference
          // write the new payload to ORCON/ORCOFF
          EcalSRSettings* srp_pop = new EcalSRSettings();
          ChangePayload(*srp_pop, *sr);  // add this payload to DB
          m_to_transfer.push_back(std::make_pair(srp_pop, irun));

          std::ostringstream ss;
          ss << "Run=" << irun << "_SRPchanged_" << std::endl;
          m_userTextLog = ss.str() + ";";

          if (m_to_transfer.size() >= 20)
            break;
        }
        delete sr;
        m_i_tag = myconfig_tag;
        m_i_version = myconfig_version;
        fe_conf_id_old = fe_conf_id;
      }  // different tag or version or fe config
    }    // loop over runs
  }      // test on number of runs

  // disconnect from DB
  delete econn;
  fout.close();
}

void popcon::EcalSRPHandler::importDccConfigFile(EcalSRSettings& sr, const std::string& filename, bool useCLOB) {
  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme(XercesDOMParser::Val_Never);
  parser->setDoNamespaces(false);
  parser->setDoSchema(false);

  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "importDccConfigFile Error parsing document" << std::endl;
  }

  DOMElement* element = xmlDoc->getDocumentElement();
  std::string type = cms::xerces::toString(element->getTagName());

  // 1st level
  int EBEE = -1;
  int L1ZS[2] = {0, 0}, L2ZS[2] = {0, 0};
  for (DOMNode* childNode = element->getFirstChild(); childNode; childNode = childNode->getNextSibling()) {
    if (childNode->getNodeType() == DOMNode::ELEMENT_NODE) {
      const std::string foundName = cms::xerces::toString(childNode->getNodeName());
      DOMElement* child = static_cast<DOMElement*>(childNode);
      DOMNamedNodeMap* attributes = child->getAttributes();
      unsigned int numAttributes = attributes->getLength();
      for (unsigned int j = 0; j < numAttributes; ++j) {
        DOMNode* attributeNode = attributes->item(j);
        DOMAttr* attribute = static_cast<DOMAttr*>(attributeNode);
        const std::string info = cms::xerces::toString(attribute->getName());
        const std::string scope = cms::xerces::toString(attribute->getValue());
        if (info == "_scope") {
          if (scope.substr(0, 2) == "EE")
            EBEE = 1;
          else
            EBEE = 0;
        }
      }
      // 2nd level
      for (DOMNode* subchildNode = childNode->getFirstChild(); subchildNode;
           subchildNode = subchildNode->getNextSibling()) {
        if (subchildNode->getNodeType() == DOMNode::ELEMENT_NODE) {
          const std::string subName = cms::xerces::toString(subchildNode->getNodeName());
          // 3rd level
          for (DOMNode* subsubchildNode = subchildNode->getFirstChild(); subsubchildNode;
               subsubchildNode = subsubchildNode->getNextSibling()) {
            if (subsubchildNode->getNodeType() == DOMNode::ELEMENT_NODE) {
              const std::string subName = cms::xerces::toString(subsubchildNode->getNodeName());
              if (subName == "L1ZSUPPRESSION")
                GetNodeData(subsubchildNode, L1ZS[EBEE]);
              if (subName == "L2ZSUPPRESSION")
                GetNodeData(subsubchildNode, L2ZS[EBEE]);
              if (subName == "FIRSTZSSAMPLE") {
                int ZS;
                GetNodeData(subsubchildNode, ZS);
                if (useCLOB)
                  sr.ecalDccZs1stSample_.push_back(ZS);
              }
              if (subName == "CXTALWEIGHTS") {
                std::vector<float> dcc(6);
                float w;
                for (int iw = 0; iw < 6; iw++) {
                  GetNodeData(subsubchildNode, w);
                  dcc.push_back(w);
                }
                if (useCLOB)
                  sr.dccNormalizedWeights_.push_back(dcc);  // vector vector
              }
            }
          }  // loop over subsubchild
        }
      }  // loop over subchild
    }
  }                                                             // loop over child
  sr.srpLowInterestChannelZS_.push_back(L1ZS[0] * 0.035 / 4);   //  EB
  sr.srpLowInterestChannelZS_.push_back(L1ZS[1] * 0.060 / 4);   //  EE
  sr.srpHighInterestChannelZS_.push_back(L2ZS[0] * 0.035 / 4);  //  EB
  sr.srpHighInterestChannelZS_.push_back(L2ZS[1] * 0.060 / 4);  //  EE
  delete parser;
  cms::concurrency::xercesTerminate();
}

void popcon::EcalSRPHandler::PrintPayload(EcalSRSettings& sr, std::ofstream& fout) {
  int imax = sr.deltaEta_.size();
  fout << " deltaEta[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.deltaEta_[i] << " ";
  }
  fout << std::endl;

  imax = sr.deltaPhi_.size();
  fout << " deltaPhi[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.deltaPhi_[i] << " ";
  }
  fout << std::endl;

  imax = sr.ecalDccZs1stSample_.size();
  fout << " ecalDccZs1stSample[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.ecalDccZs1stSample_[i] << " ";
  }
  fout << std::endl;

  fout << " ebDccAdcToGeV " << sr.ebDccAdcToGeV_ << std::endl;

  fout << " eeDccAdcToGeV " << sr.eeDccAdcToGeV_ << std::endl;

  fout << " dccNormalizedWeights" << std::endl;
  for (int i = 0; i < (int)sr.dccNormalizedWeights_.size(); ++i) {
    fout << " Channel " << i;
    for (int j = 0; j < (int)sr.dccNormalizedWeights_[i].size(); ++j)
      fout << " " << sr.dccNormalizedWeights_[i][j];
    fout << std::endl;
  }

  imax = sr.symetricZS_.size();
  fout << " symetricZS[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.symetricZS_[i] << " ";
  }
  fout << std::endl;

  imax = sr.srpLowInterestChannelZS_.size();
  fout << " srpLowInterestChannelZS[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.srpLowInterestChannelZS_[i] << " ";
  }
  fout << std::endl;

  imax = sr.srpHighInterestChannelZS_.size();
  fout << " srpHighInterestChannelZS[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.srpHighInterestChannelZS_[i] << " ";
  }
  fout << std::endl;

  imax = sr.actions_.size();
  fout << " actions[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.actions_[i] << " ";
  }
  fout << std::endl;

  imax = sr.tccMasksFromConfig_.size();
  fout << " tccMasksFromConfig[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.tccMasksFromConfig_[i] << " ";
  }
  fout << std::endl;

  fout << " srpMasksFromConfig" << std::endl;
  for (int i = 0; i < (int)sr.srpMasksFromConfig_.size(); ++i) {
    for (int j = 0; j < (int)sr.srpMasksFromConfig_[i].size(); ++j)
      fout << sr.srpMasksFromConfig_[i][j] << " ";
    fout << std::endl;
  }

  imax = sr.dccMasks_.size();
  fout << " dccMasks[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.dccMasks_[i] << " ";
  }
  fout << std::endl;

  imax = sr.srfMasks_.size();
  fout << " srfMasks[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.srfMasks_[i] << " ";
  }
  fout << std::endl;

  fout << "substitutionSrfs" << std::endl;
  for (int i = 0; i < (int)sr.substitutionSrfs_.size(); ++i) {
    for (int j = 0; j < (int)sr.substitutionSrfs_[i].size(); ++j)
      fout << sr.substitutionSrfs_[i][j] << " ";
    fout << std::endl;
  }

  imax = sr.testerTccEmuSrpIds_.size();
  fout << " testerTccEmuSrpIds[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.testerTccEmuSrpIds_[i] << " ";
  }
  fout << std::endl;

  imax = sr.testerSrpEmuSrpIds_.size();
  fout << " testerSrpEmuSrpIds[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.testerSrpEmuSrpIds_[i] << " ";
  }
  fout << std::endl;

  imax = sr.testerDccTestSrpIds_.size();
  fout << " testerDccTestSrpIds[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.testerDccTestSrpIds_[i] << " ";
  }
  fout << std::endl;

  imax = sr.testerSrpTestSrpIds_.size();
  fout << " testerSrpTestSrpIds[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.testerSrpTestSrpIds_[i] << " ";
  }
  fout << std::endl;

  imax = sr.bxOffsets_.size();
  fout << " bxOffsets[" << imax << "] ";
  for (int i = 0; i < imax; i++) {
    fout << sr.bxOffsets_[i] << " ";
  }
  fout << std::endl;

  fout << " bxGlobalOffset " << sr.bxGlobalOffset_ << std::endl;
  fout << " automaticMasks " << sr.automaticMasks_ << std::endl;
  fout << " automaticSrpSelect " << sr.automaticSrpSelect_ << std::endl;
}

void popcon::EcalSRPHandler::ChangePayload(EcalSRSettings& sref, EcalSRSettings& sr) {
  sref.deltaEta_ = sr.deltaEta_;
  sref.deltaPhi_ = sr.deltaPhi_;
  sref.ecalDccZs1stSample_ = sr.ecalDccZs1stSample_;
  sref.ebDccAdcToGeV_ = sr.ebDccAdcToGeV_;
  sref.eeDccAdcToGeV_ = sr.eeDccAdcToGeV_;
  sref.dccNormalizedWeights_ = sr.dccNormalizedWeights_;
  sref.symetricZS_ = sr.symetricZS_;
  sref.srpLowInterestChannelZS_ = sr.srpLowInterestChannelZS_;
  sref.srpHighInterestChannelZS_ = sr.srpHighInterestChannelZS_;
  sref.actions_ = sr.actions_;
  sref.tccMasksFromConfig_ = sr.tccMasksFromConfig_;
  sref.srpMasksFromConfig_ = sr.srpMasksFromConfig_;
  sref.dccMasks_ = sr.dccMasks_;
  sref.srfMasks_ = sr.srfMasks_;
  sref.substitutionSrfs_ = sr.substitutionSrfs_;
  sref.testerTccEmuSrpIds_ = sr.testerTccEmuSrpIds_;
  sref.testerSrpEmuSrpIds_ = sr.testerSrpEmuSrpIds_;
  sref.testerDccTestSrpIds_ = sr.testerDccTestSrpIds_;
  sref.testerSrpTestSrpIds_ = sr.testerSrpTestSrpIds_;
  sref.bxOffsets_ = sr.bxOffsets_;
  sref.bxGlobalOffset_ = sr.bxGlobalOffset_;
  sref.automaticMasks_ = sr.automaticMasks_;
  sref.automaticSrpSelect_ = sr.automaticSrpSelect_;
}
