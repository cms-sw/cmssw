// -*- C++ -*-
//
// Class:      L1TMuonBarrelParamsESProducer
//
// Original Author:  Giannis Flouris
//         Created:
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"
#include "CondFormats/L1TObjects/interface/DTTFBitArray.h"

#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/Parameter.h"
#include "L1Trigger/L1TCommon/interface/Mask.h"

#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsHelper.h"

// class declaration
//
typedef std::map<short, short, std::less<short> > LUT;

class L1TMuonBarrelParamsESProducer : public edm::ESProducer {
   public:
      L1TMuonBarrelParamsESProducer(const edm::ParameterSet&);
      ~L1TMuonBarrelParamsESProducer() override;
      int load_pt(std::vector<LUT>& , std::vector<int>&, unsigned short int, std::string);
      int load_phi(std::vector<LUT>& , unsigned short int, unsigned short int, std::string);
      int load_ext(std::vector<L1TMuonBarrelParams::LUTParams::extLUT>&, unsigned short int, unsigned short int );
      //void print(std::vector<LUT>& , std::vector<int>& ) const;
      int getPtLutThreshold(int ,std::vector<int>& ) const;
      using ReturnType = std::unique_ptr<L1TMuonBarrelParams>;

      ReturnType produce(const L1TMuonBarrelParamsRcd&);
   private:
    //L1TMuonBarrelParams m_params;
    L1TMuonBarrelParamsHelper m_params_helper;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TMuonBarrelParamsESProducer::L1TMuonBarrelParamsESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   // Firmware version
   unsigned fwVersion = iConfig.getParameter<unsigned>("fwVersion");
   std::string AssLUTpath = iConfig.getParameter<std::string>("AssLUTPath");
   //m_params.setAssLUTPath(AssLUTpath);

   // int PT_Assignment_nbits_Phi = iConfig.getParameter<int>("PT_Assignment_nbits_Phi");
   // int PT_Assignment_nbits_PhiB = iConfig.getParameter<int>("PT_Assignment_nbits_PhiB");
   // int PHI_Assignment_nbits_Phi = iConfig.getParameter<int>("PHI_Assignment_nbits_Phi");
   // int PHI_Assignment_nbits_PhiB = iConfig.getParameter<int>("PHI_Assignment_nbits_PhiB");
   // int Extrapolation_nbits_Phi = iConfig.getParameter<int>("Extrapolation_nbits_Phi");
   // int Extrapolation_nbits_PhiB = iConfig.getParameter<int>("Extrapolation_nbits_PhiB");
   // int BX_min = iConfig.getParameter<int>("BX_min");
   // int BX_max = iConfig.getParameter<int>("BX_max");
   // int Extrapolation_Filter = iConfig.getParameter<int>("Extrapolation_Filter");
   // int OutOfTime_Filter_Window = iConfig.getParameter<int>("OutOfTime_Filter_Window");
   // bool OutOfTime_Filter = iConfig.getParameter<bool>("OutOfTime_Filter");
   // bool Open_LUTs = iConfig.getParameter<bool>("Open_LUTs");
   // bool EtaTrackFinder = iConfig.getParameter<bool>("EtaTrackFinder");
   // bool Extrapolation_21 = iConfig.getParameter<bool>("Extrapolation_21");
   bool configFromXML = iConfig.getParameter<bool>("configFromXML");
   // bool DisableNewAlgo = iConfig.getParameter<bool>("DisableNewAlgo");

   std::map<std::string, int> allInts;
   std::map<std::string, bool> allBools;
   std::map<std::string, std::vector<std::string> > allMasks;

   allInts["PT_Assignment_nbits_Phi"] = iConfig.getParameter<int>("PT_Assignment_nbits_Phi");
   allInts["PT_Assignment_nbits_PhiB"] = iConfig.getParameter<int>("PT_Assignment_nbits_PhiB");
   allInts["PHI_Assignment_nbits_Phi"] = iConfig.getParameter<int>("PHI_Assignment_nbits_Phi");
   allInts["PHI_Assignment_nbits_PhiB"] = iConfig.getParameter<int>("PHI_Assignment_nbits_PhiB");
   allInts["Extrapolation_nbits_Phi"] = iConfig.getParameter<int>("Extrapolation_nbits_Phi");
   allInts["Extrapolation_nbits_PhiB"] = iConfig.getParameter<int>("Extrapolation_nbits_PhiB");
   allInts["BX_min"] = iConfig.getParameter<int>("BX_min");
   allInts["BX_max"] = iConfig.getParameter<int>("BX_max");
   allInts["Extrapolation_Filter"] = iConfig.getParameter<int>("Extrapolation_Filter");
   allInts["OutOfTime_Filter_Window"] = iConfig.getParameter<int>("OutOfTime_Filter_Window");
   allBools["OutOfTime_Filter"] = iConfig.getParameter<bool>("OutOfTime_Filter");
   allBools["Open_LUTs"] = iConfig.getParameter<bool>("Open_LUTs");
   allBools["EtaTrackFinder"] = iConfig.getParameter<bool>("EtaTrackFinder");
   allBools["Extrapolation_21"] = iConfig.getParameter<bool>("Extrapolation_21");
   allBools["configFromXML"] = iConfig.getParameter<bool>("configFromXML");
   allBools["DisableNewAlgo"] = iConfig.getParameter<bool>("DisableNewAlgo");

   allMasks["mask_phtf_st1"] = iConfig.getParameter< std::vector <string>  >("mask_phtf_st1");
   allMasks["mask_phtf_st2"] = iConfig.getParameter< std::vector <string>  >("mask_phtf_st2");
   allMasks["mask_phtf_st3"] = iConfig.getParameter< std::vector <string>  >("mask_phtf_st3");
   allMasks["mask_phtf_st4"] = iConfig.getParameter< std::vector <string>  >("mask_phtf_st4");

   allMasks["mask_ettf_st1"] = iConfig.getParameter< std::vector <string>  >("mask_ettf_st1");
   allMasks["mask_ettf_st2"] = iConfig.getParameter< std::vector <string>  >("mask_ettf_st2");
   allMasks["mask_ettf_st3"] = iConfig.getParameter< std::vector <string>  >("mask_ettf_st3");

/*
       m_params.set_PT_Assignment_nbits_Phi(PT_Assignment_nbits_Phi);
       m_params.set_PT_Assignment_nbits_PhiB(PT_Assignment_nbits_PhiB);
       m_params.set_PHI_Assignment_nbits_Phi(PHI_Assignment_nbits_Phi);
       m_params.set_PHI_Assignment_nbits_PhiB(PHI_Assignment_nbits_PhiB);
       m_params.set_Extrapolation_nbits_Phi(Extrapolation_nbits_Phi);
       m_params.set_Extrapolation_nbits_PhiB(Extrapolation_nbits_PhiB);
       m_params.set_BX_min(BX_min);
       m_params.set_BX_max(BX_max);
       m_params.set_Extrapolation_Filter(Extrapolation_Filter);
       m_params.set_OutOfTime_Filter_Window(OutOfTime_Filter_Window);
       m_params.set_OutOfTime_Filter(OutOfTime_Filter);
       m_params.set_Open_LUTs(Open_LUTs);
       m_params.set_EtaTrackFinder(EtaTrackFinder);
       m_params.set_Extrapolation_21(Extrapolation_21);
       m_params.setFwVersion(fwVersion);
       m_params.set_DisableNewAlgo(DisableNewAlgo);


    ///Read Pt assignment Luts
        std::vector<LUT> pta_lut(0); pta_lut.reserve(19);
        std::vector<int> pta_threshold(6); pta_threshold.reserve(9);
        if ( load_pt(pta_lut,pta_threshold, PT_Assignment_nbits_Phi, AssLUTpath) != 0 ) {
          cout << "Can not open files to load pt-assignment look-up tables for L1TMuonBarrelTrackProducer!" << endl;
        }
       m_params.setpta_lut(pta_lut);
       m_params.setpta_threshold(pta_threshold);

    ///Read Phi assignment Luts
        std::vector<LUT> phi_lut(0); phi_lut.reserve(2);
        if ( load_phi(phi_lut, PHI_Assignment_nbits_Phi, PHI_Assignment_nbits_PhiB, AssLUTpath) != 0 ) {
          cout << "Can not open files to load phi-assignment look-up tables for L1TMuonBarrelTrackProducer!" << endl;
        }
        m_params.setphi_lut(phi_lut);




       m_params.l1mudttfparams.reset();

       std::vector <std::string>  mask_phtf_st1 = iConfig.getParameter< std::vector <string>  >("mask_phtf_st1");
       std::vector <std::string>  mask_phtf_st2 = iConfig.getParameter< std::vector <string>  >("mask_phtf_st2");
       std::vector <std::string>  mask_phtf_st3 = iConfig.getParameter< std::vector <string>  >("mask_phtf_st3");
       std::vector <std::string>  mask_phtf_st4 = iConfig.getParameter< std::vector <string>  >("mask_phtf_st4");

       std::vector <std::string>  mask_ettf_st1 = iConfig.getParameter< std::vector <string>  >("mask_ettf_st1");
       std::vector <std::string>  mask_ettf_st2 = iConfig.getParameter< std::vector <string>  >("mask_ettf_st2");
       std::vector <std::string>  mask_ettf_st3 = iConfig.getParameter< std::vector <string>  >("mask_ettf_st3");

        for( int wh=-3; wh<4; wh++ ) {
           int sec = 0;
           for(char& c : mask_phtf_st1[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_inrec_chdis_st1(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_phtf_st2[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_inrec_chdis_st2(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_phtf_st3[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_inrec_chdis_st3(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_phtf_st4[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_inrec_chdis_st4(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_ettf_st1[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_etsoc_chdis_st1(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_ettf_st2[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_etsoc_chdis_st2(wh,sec,mask);
                sec++;
            }
           sec = 0;
           for(char& c : mask_ettf_st3[wh+3]) {
                int mask = c - '0';
                m_params.l1mudttfmasks.set_etsoc_chdis_st3(wh,sec,mask);
                //Not used in BMTF - mask
                m_params.l1mudttfmasks.set_inrec_chdis_csc(wh,sec,true);
                sec++;
            }

        }


    ///Read Extrapolation Luts
        std::vector<L1TMuonBarrelParams::LUTParams::extLUT> ext_lut(0); ext_lut.reserve(12);
        if ( load_ext(ext_lut, PHI_Assignment_nbits_Phi, PHI_Assignment_nbits_PhiB) != 0 ) {
          cout << "Can not open files to load extrapolation look-up tables for L1TMuonBarrelTrackProducer!" << endl;
        }
        m_params.setext_lut(ext_lut);

    //m_params.l1mudttfextlut.load();

    */

        m_params_helper.configFromPy(allInts, allBools, allMasks, fwVersion, AssLUTpath);
   if(configFromXML){
      cout<<"Configuring BMTF Emulator from xml files.\n";
      edm::FileInPath hwXmlFile(iConfig.getParameter<std::string>("hwXmlFile"));
      edm::FileInPath topCfgXmlFile(iConfig.getParameter<std::string>("topCfgXmlFile"));
      std::string xmlCfgKey = iConfig.getParameter<std::string>("xmlCfgKey");

      l1t::TriggerSystem trgSys;
      trgSys.configureSystemFromFiles(hwXmlFile.fullPath().c_str(),topCfgXmlFile.fullPath().c_str(),xmlCfgKey.c_str());

     /* std::map<std::string, std::string> procRole = trgSys.getProcRole();

      for(auto it_proc=procRole.begin(); it_proc!=procRole.end(); it_proc++ ){

          std::string procId = it_proc->first;

          std::map<std::string, l1t::Setting> settings = trgSys.getSettings(procId);
          std::vector<l1t::TableRow>  tRow = settings["regTable"].getTableRows();
          for(auto it=tRow.begin(); it!=tRow.end(); it++)
          {
            if (it->getRowValue<std::string>("register_path").find("open_lut") != std::string::npos){
              //std::cout << "Value is: " << it->getRowValue<bool>("register_value") << std::endl;
              m_params.set_Open_LUTs(it->getRowValue<bool>("register_value"));
            }
            if (it->getRowValue<std::string>("register_path").find("sel_21") != std::string::npos){
              //std::cout << "Value is: " << it->getRowValue<bool>("register_value") << std::endl;
              m_params.set_Extrapolation_21(it->getRowValue<bool>("register_value"));
            }

            if (it->getRowValue<std::string>("register_path").find("dis_newalgo") != std::string::npos){
              //std::cout << "Value is: " << it->getRowValue<int>("register_value") << std::endl;
              //int fwv = (it->getRowValue<int>("register_value")==1) ? 1 : 2;
              //m_params.setFwVersion(fwv);
              bool disnewalgo = (it->getRowValue<int>("register_value")==1);
              m_params.set_DisableNewAlgo(disnewalgo);
            }

            string masks[5] = {"mask_ctrl_N2", "mask_ctrl_N1", "mask_ctrl_0", "mask_ctrl_P1", "mask_ctrl_P2"};

            for(int m=0; m<5; m++){

                if (it->getRowValue<std::string>("register_path").find(masks[m]) != std::string::npos){
                  string mask_ctrl = it->getRowValue<string>("register_value");
                  const char *hexstring = mask_ctrl.c_str();
                  ///Converts the last bit from str to int
                  int mask = (int)strtol((hexstring+7), NULL, 16);
                  int mask_all = (int)strtol((hexstring), NULL, 16);
                  ///All bits must be the same
                  if(!( mask_all==0x111111 || mask_all==0x222222 || mask_all==0x333333 || mask_all==0x444444 ||
                     mask_all==0x555555 || mask_all==0x666666 || mask_all==0x777777) )
                    cerr<<"BMTF: Cannot re-emulate properly. Individual link masking cannot be handled."<<endl;

                  if((mask&1)>0)  {
                     for(int sec=0; sec<12; sec++){
                      if(masks[m]=="mask_ctrl_N2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(-3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(-3,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_N1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(-2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(-2,sec,true);
                      }

                      if(masks[m]=="mask_ctrl_0"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(-1,sec,true);
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(-1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(1,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(2,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st1(3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st1(3,sec,true);
                      }
                    }

                  }

                  if((mask&2)>0)  {
                    for(int sec=0; sec<12; sec++){
                      if(masks[m]=="mask_ctrl_N2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(-3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(-3,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_N1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(-2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(-2,sec,true);
                      }

                      if(masks[m]=="mask_ctrl_0"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(-1,sec,true);
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(-1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(1,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(2,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st2(3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st2(3,sec,true);
                      }
                    }
                  }

                  if((mask&4)>0)  {
                    for(int sec=0; sec<12; sec++){
                      if(masks[m]=="mask_ctrl_N2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(-3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(-3,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_N1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(-2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(-2,sec,true);
                      }

                      if(masks[m]=="mask_ctrl_0"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(-1,sec,true);
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(-1,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(1,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(2,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(2,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st3(3,sec,true);
                                            m_params.l1mudttfmasks.set_etsoc_chdis_st3(3,sec,true);
                      }
                    }
                  }

                  if((mask&8)>0)  {
                    for(int sec=0; sec<12; sec++){
                      if(masks[m]=="mask_ctrl_N2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(-3,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_N1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(-2,sec,true);
                      }

                      if(masks[m]=="mask_ctrl_0"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(-1,sec,true);
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(1,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P1"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(2,sec,true);
                      }
                      if(masks[m]=="mask_ctrl_P2"){
                                            m_params.l1mudttfmasks.set_inrec_chdis_st4(3,sec,true);
                      }
                    }
                  }
               }///if register path
             }///for masks
          }///for it tRow
      }///for it procRole */
	m_params_helper.configFromDB(trgSys);
  }///if configDB
	//m_params = cast_to_L1TMuonBarrelParams((L1TMuonBarrelParams_PUBLIC)m_params_helper);
 

}


L1TMuonBarrelParamsESProducer::~L1TMuonBarrelParamsESProducer()
{

}

/*int L1TMuonBarrelParamsESProducer::load_pt(std::vector<LUT>& pta_lut,
                                  std::vector<int>& pta_threshold,
                                  unsigned short int nbitphi,
                                  std::string AssLUTpath
                                  ){


// maximal number of pt assignment methods
const int MAX_PTASSMETH = 19;
const int MAX_PTASSMETHA = 12;

// pt assignment methods
enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                   PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H,
                   PB12H,  PB13H,  PB14H,  PB21H,  PB23H,  PB24H, PB34H,
                   NODEF };

  // get directory name
  string pta_str = "";
  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (address) and  pt ...5 bits
  // now convert phi and phib to the required precision
  int nbit_phi = nbitphi;
  int sh_phi  = 12 - nbit_phi;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < MAX_PTASSMETH; pam++ ) {
    switch ( pam ) {
      case PT12L  : { pta_str = "pta12l"; break; }
      case PT12H  : { pta_str = "pta12h"; break; }
      case PT13L  : { pta_str = "pta13l"; break; }
      case PT13H  : { pta_str = "pta13h"; break; }
      case PT14L  : { pta_str = "pta14l"; break; }
      case PT14H  : { pta_str = "pta14h"; break; }
      case PT23L  : { pta_str = "pta23l"; break; }
      case PT23H  : { pta_str = "pta23h"; break; }
      case PT24L  : { pta_str = "pta24l"; break; }
      case PT24H  : { pta_str = "pta24h"; break; }
      case PT34L  : { pta_str = "pta34l"; break; }
      case PT34H  : { pta_str = "pta34h"; break; }
      case PB12H  : { pta_str = "ptb12h_Feb2016"; break; }
      case PB13H  : { pta_str = "ptb13h_Feb2016"; break; }
      case PB14H  : { pta_str = "ptb14h_Feb2016"; break; }
      case PB21H  : { pta_str = "ptb21h_Feb2016"; break; }
      case PB23H  : { pta_str = "ptb23h_Feb2016"; break; }
      case PB24H  : { pta_str = "ptb24h_Feb2016"; break; }
      case PB34H  : { pta_str = "ptb34h_Feb2016"; break; }

    }

    // assemble file name
    string lutpath = AssLUTpath;
    edm::FileInPath lut_f = edm::FileInPath(string(lutpath + pta_str + ".lut"));
    string pta_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(pta_file);
    if ( file.open() != 0 ) return -1;

    // get the right shift factor
    int shift = sh_phi;
    int adr_old = -2048 >> shift;
    if (pam >= MAX_PTASSMETHA) adr_old = -512 >> shift;

    LUT tmplut;

    int number = -1;
    int sum_pt = 0;

    if ( file.good() ) {
      int threshold = file.readInteger();
      pta_threshold[pam/2] = threshold;
    }

    // read values and shift to correct precision
    while ( file.good() ) {

      int adr = (file.readInteger()) >> shift;
      int pt  = file.readInteger();

      number++;
      //cout<<pam<<"    "<<number<<"   "<<MAX_PTASSMETHA<<endl;
      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, (sum_pt/number) ));

        adr_old = adr;
        number = 0;
        sum_pt = 0;
      }

      sum_pt += pt;

      if ( !file.good() ) file.close();

    }

    file.close();
    pta_lut.push_back(tmplut);
  }
  return 0;

}




int L1TMuonBarrelParamsESProducer::load_phi(std::vector<LUT>& phi_lut,
                                  unsigned short int nbit_phi,
                                  unsigned short int nbit_phib,
                                  std::string AssLUTpath
                                  ) {


  // precision : in the look-up tables the following precision is used :
  // address (phib) ...10 bits, phi ... 12 bits

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  string phi_str;
  // loop over all phi-assignment methods
  for ( int idx = 0; idx < 2; idx++ ) {
    switch ( idx ) {
      case 0 : { phi_str = "phi12"; break; }
      case 1 : { phi_str = "phi42"; break; }
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(AssLUTpath + phi_str + ".lut"));
    string phi_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(phi_file);
    if ( file.open() != 0 ) return -1;

    LUT tmplut;

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_phi = 0;

    // read values
    while ( file.good() ) {

      int adr = (file.readInteger()) >> sh_phib;
      int phi =  file.readInteger();

      number++;

      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, ((sum_phi/number) >> sh_phi) ));

        adr_old = adr;
        number = 0;
        sum_phi  = 0;
      }

      sum_phi += phi;

      if ( !file.good() ) file.close();

    }

    file.close();
    phi_lut.push_back(tmplut);
  }
  return 0;

}


int L1TMuonBarrelParamsESProducer::getPtLutThreshold(int pta_ind, std::vector<int>& pta_threshold) const {

  if ( pta_ind >= 0 && pta_ind < 13/2 ) {
    return pta_threshold[pta_ind];
  }
  else {
    cerr << "PtaLut::getPtLutThreshold : can not find threshold " << pta_ind << endl;
    return 0;
  }

}

*/




//
// load extrapolation look-up tables
//
/*
int L1TMuonBarrelParamsESProducer::load_ext(std::vector<L1TMuonBarrelParams::LUTParams::extLUT>& ext_lut,
                                            unsigned short int nbit_phi,
                                            unsigned short int nbit_phib) {

  //max. number of Extrapolations
  const int MAX_EXT = 12;

  // extrapolation types
  enum Extrapolation { EX12, EX13, EX14, EX21, EX23, EX24, EX34,
                     EX15, EX16, EX25, EX26, EX56 };

  // get directory name
  string defaultPath = "L1Trigger/L1TMuon/data/bmtf_luts/";
  string ext_dir = "LUTs_Ext/";
  string ext_str = "";

  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (low, high), phib ...10 bits (address)
  // now convert phi and phib to the required precision

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  // loop over all extrapolations
  for ( int ext = 0; ext < MAX_EXT; ext++ ) {
    switch (ext) {
      case EX12 : ext_str = "ext12"; break;
      case EX13 : ext_str = "ext13"; break;
      case EX14 : ext_str = "ext14"; break;
      case EX21 : ext_str = "ext21"; break;
      case EX23 : ext_str = "ext23"; break;
      case EX24 : ext_str = "ext24"; break;
      case EX34 : ext_str = "ext34"; break;
      case EX15 : ext_str = "ext15"; break;
      case EX16 : ext_str = "ext16"; break;
      case EX25 : ext_str = "ext25"; break;
      case EX26 : ext_str = "ext26"; break;
      case EX56 : ext_str = "ext56"; break;
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + ext_dir + ext_str + ".lut"));
    string ext_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(ext_file);
    if ( file.open() != 0 ) return -1;
    //    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : "
    //                                         << file.getName() << endl;

    L1TMuonBarrelParams::LUTParams::extLUT tmplut;

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_low = 0;
    int sum_high = 0;

    // read values and shift to correct precision
    while ( file.good() ) {

      int adr  = ( file.readInteger() ) >> sh_phib;	// address (phib)
      int low  = ( file.readInteger() );    		// low value (phi)
      int high = ( file.readInteger() );	        // high value (phi)

      number++;

      if ( adr != adr_old ) {

        tmplut.low[adr_old]  = sum_low  >> sh_phi;
        tmplut.high[adr_old] = sum_high >> sh_phi;

        adr_old = adr;
        number = 0;
        sum_low  = 0;
        sum_high = 0;

      }

      if (number == 0) sum_low  = low;
      if (number == 0) sum_high = high;

      if ( !file.good() ) file.close();
    }

    file.close();
    ext_lut.push_back(tmplut);
  }
  return 0;

}


//
// member functions
//
*/
// ------------ method called to produce the data  ------------
L1TMuonBarrelParamsESProducer::ReturnType
L1TMuonBarrelParamsESProducer::produce(const L1TMuonBarrelParamsRcd& iRecord)
{
   return std::make_unique<L1TMuonBarrelParams>(m_params_helper);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonBarrelParamsESProducer);
