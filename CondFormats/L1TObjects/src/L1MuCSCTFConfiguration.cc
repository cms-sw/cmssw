#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include <sstream>
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

edm::ParameterSet L1MuCSCTFConfiguration::parameters(int sp) const {
  
  //std::cout << "==============================================="<<std::endl;
  //std::cout << "SP:"<<int(sp)<< std::endl;
  
  edm::ParameterSet pset;
  if(sp>=12) return pset;

  pset.addParameter<int>("CoreLatency",8);

  //initialization of the DAT_ETA registers with default values
  //the DAT_ETA registers meaning are explained at Table 2 of
  //http://www.phys.ufl.edu/~uvarov/SP05/LU-SP_ReferenceGuide_090915_Update.pdf
  std::vector<unsigned int> etamin(8), etamax(8), etawin(7);

  unsigned int mindetap      = 8;
  unsigned int mindetap_halo = 8;

  etamin[0] = 22;
  etamin[1] = 22;
  etamin[2] = 14;
  etamin[3] = 14;
  etamin[4] = 14;
  etamin[5] = 14;
  etamin[6] = 10;
  etamin[7] = 22;

  unsigned int mindeta12_accp  =  8; 
  unsigned int mindeta13_accp  = 19;
  unsigned int mindeta112_accp = 19;
  unsigned int mindeta113_accp = 30;

  etamax[0] = 127;
  etamax[1] = 127;
  etamax[2] = 127;
  etamax[3] = 127;
  etamax[4] = 127;
  etamax[5] =  24;
  etamax[6] =  24;
  etamax[7] = 127;

  unsigned int maxdeta12_accp  = 14;
  unsigned int maxdeta13_accp  = 25;
  unsigned int maxdeta112_accp = 25;
  unsigned int maxdeta113_accp = 36;

  etawin[0] = 4;
  etawin[1] = 4;
  etawin[2] = 4;
  etawin[3] = 4;
  etawin[4] = 4;
  etawin[5] = 4;
  etawin[6] = 4;

  unsigned int maxdphi12_accp  = 64;
  unsigned int maxdphi13_accp  = 64;
  unsigned int maxdphi112_accp = 64;
  unsigned int maxdphi113_accp = 64;
                               
  unsigned int mindphip      = 128;
  unsigned int mindphip_halo = 128;
               
  unsigned int straightp =   60;
  unsigned int curvedp   =  200;
  unsigned int mbaPhiOff =    0;
  unsigned int mbbPhiOff = 2048;

  int eta_cnt=0;

  // default printout
  LogDebug("L1MuCSCTFParametersTester") << "DEFAULT VALUES FOR DAT_ETA" << std::endl
                                        << "mindetap     =" << mindetap      << std::endl
                                        << "mindetap_halo=" << mindetap_halo << std::endl
      
                                        << "etamin[0]=" << etamin[0] << std::endl
                                        << "etamin[1]=" << etamin[1] << std::endl
                                        << "etamin[2]=" << etamin[2] << std::endl
                                        << "etamin[3]=" << etamin[3] << std::endl
                                        << "etamin[4]=" << etamin[4] << std::endl
                                        << "etamin[5]=" << etamin[5] << std::endl
                                        << "etamin[6]=" << etamin[6] << std::endl
                                        << "etamin[7]=" << etamin[7] << std::endl

                                        << "mindeta12_accp =" << mindeta12_accp  << std::endl 
                                        << "mindeta13_accp =" << mindeta13_accp  << std::endl
                                        << "mindeta112_accp=" << mindeta112_accp << std::endl
                                        << "mindeta113_accp=" << mindeta113_accp << std::endl
      
                                        << "etamax[0]=" << etamax[0] << std::endl
                                        << "etamax[1]=" << etamax[1] << std::endl
                                        << "etamax[2]=" << etamax[2] << std::endl
                                        << "etamax[3]=" << etamax[3] << std::endl
                                        << "etamax[4]=" << etamax[4] << std::endl
                                        << "etamax[5]=" << etamax[5] << std::endl
                                        << "etamax[6]=" << etamax[6] << std::endl
                                        << "etamax[7]=" << etamax[7] << std::endl

                                        << "maxdeta12_accp =" << maxdeta12_accp  << std::endl
                                        << "maxdeta13_accp =" << maxdeta13_accp  << std::endl
                                        << "maxdeta112_accp=" << maxdeta112_accp << std::endl
                                        << "maxdeta113_accp=" << maxdeta113_accp << std::endl
  
                                        << "etawin[0]=" << etawin[0] << std::endl
                                        << "etawin[1]=" << etawin[1] << std::endl
                                        << "etawin[2]=" << etawin[2] << std::endl
                                        << "etawin[3]=" << etawin[3] << std::endl
                                        << "etawin[4]=" << etawin[4] << std::endl
                                        << "etawin[5]=" << etawin[5] << std::endl
                                        << "etawin[6]=" << etawin[6] << std::endl
  
                                        << "maxdphi12_accp =" << maxdphi12_accp  << std::endl
                                        << "maxdphi13_accp =" << maxdphi13_accp  << std::endl
                                        << "maxdphi112_accp=" << maxdphi112_accp << std::endl
                                        << "maxdphi113_accp=" << maxdphi113_accp << std::endl
                           
                                        << "mindphip     =" << mindphip      << std::endl
                                        << "mindphip_halo=" << mindphip_halo << std::endl
  
                                        << "straightp=" << straightp << std::endl
                                        << "curvedp  =" << curvedp   << std::endl
                                        << "mbaPhiOff=" << mbaPhiOff << std::endl
                                        << "mbbPhiOff=" << mbbPhiOff << std::endl;
  
  // start filling the registers with the values in the DBS
  std::stringstream conf(registers[sp]);
  while( !conf.eof() ){
    char buff[1024];
    conf.getline(buff,1024);
    std::stringstream line(buff);
    //std::cout<<"buff:"<<buff<<std::endl;
    std::string register_;     line>>register_;
    std::string chip_;         line>>chip_;
    std::string muon_;         line>>muon_;
    std::string writeValue_;   line>>writeValue_;
    std::string comments_;     std::getline(line,comments_);

    if( register_=="CSR_REQ" && chip_=="SP" ){
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);
      pset.addParameter<bool>("run_core",       value&0x8000);
      pset.addParameter<bool>("trigger_on_ME1a",value&0x0001);
      pset.addParameter<bool>("trigger_on_ME1b",value&0x0002);
      pset.addParameter<bool>("trigger_on_ME2", value&0x0004);
      pset.addParameter<bool>("trigger_on_ME3", value&0x0008);
      pset.addParameter<bool>("trigger_on_ME4", value&0x0010);
      pset.addParameter<bool>("trigger_on_MB1a",value&0x0100);
      pset.addParameter<bool>("trigger_on_MB1d",value&0x0200);
    }
    if( register_=="DAT_FTR" && chip_=="SP" ){
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);
      pset.addParameter<unsigned int>("singlesTrackPt",value); // 0x1F - rank, 0x60 - Q1,Q0, 0x80 - charge
    }
    if( register_=="CSR_SFC" && chip_=="SP" ){
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);
      pset.addParameter<unsigned int>("singlesTrackOutput",(value&0x3000)>>12);
    }
    if( register_=="CNT_ETA" && chip_=="SP" ){
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);
      eta_cnt = value;
    }
    if( register_=="CSR_SCC" && chip_=="SP" ){
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);
      pset.addParameter<unsigned int>("BXAdepth",    value&0x3      );
      pset.addParameter<unsigned int>("useDT",      (value&0x80)>>7 );
      pset.addParameter<unsigned int>("widePhi",    (value&0x40)>>6 );
      pset.addParameter<unsigned int>("PreTrigger", (value&0x300)>>8);
      
      // this were two old settings, not used anymore. Set them to zero
      // ask Alex if he can remove them altogether
      pset.addParameter<bool>        ("AllowALCTonly"    , 0);
      pset.addParameter<bool>        ("AllowCLCTonly"    , 0);
      pset.addParameter<bool>        ("rescaleSinglesPhi", 0);
    }

    // LATEST VERSION FROM CORE 2010-01-22 at http://www.phys.ufl.edu/~madorsky/sp/2010-01-22
    if( register_=="DAT_ETA" && chip_=="SP" ){
      
      unsigned int value = strtol(writeValue_.c_str(),'\0',16);

      //std::cout<<"DAT_ETA SP value:"<<value<<std::endl;

      if (eta_cnt== 0) mindetap = value;
      if (eta_cnt== 1) mindetap_halo = value;

      if (eta_cnt>= 2 && eta_cnt<10 ) etamin[eta_cnt-2] = value;

      if (eta_cnt==10) mindeta12_accp  = value;
      if (eta_cnt==11) mindeta13_accp  = value;
      if (eta_cnt==12) mindeta112_accp = value;
      if (eta_cnt==13) mindeta113_accp = value;

      if (eta_cnt>=14 && eta_cnt<22 ) etamax[eta_cnt-14] = value;

      if (eta_cnt==22) maxdeta12_accp  = value;
      if (eta_cnt==23) maxdeta13_accp  = value;
      if (eta_cnt==24) maxdeta112_accp = value;
      if (eta_cnt==25) maxdeta113_accp = value;

      if( eta_cnt>=26 && eta_cnt<33) etawin[eta_cnt-26] = value;

      if (eta_cnt==33) maxdphi12_accp  = value;
      if (eta_cnt==34) maxdphi13_accp  = value;
      if (eta_cnt==35) maxdphi112_accp = value;
      if (eta_cnt==36) maxdphi113_accp = value;

      if (eta_cnt==37) mindphip      = value;
      if (eta_cnt==38) mindphip_halo = value;

      if (eta_cnt==39) straightp = value;
      if (eta_cnt==40) curvedp   = value;
      if (eta_cnt==41) mbaPhiOff = value;
      if (eta_cnt==42) mbbPhiOff = value;
      
      eta_cnt++;
    }

    if( register_=="CSR_LQE" && chip_=="F1" && muon_=="M1" )
      pset.addParameter<unsigned int>("QualityEnableME1a",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F1" && muon_=="M2" )
      pset.addParameter<unsigned int>("QualityEnableME1b",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F1" && muon_=="M3" )
      pset.addParameter<unsigned int>("QualityEnableME1c",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F2" && muon_=="M1" )
      pset.addParameter<unsigned int>("QualityEnableME1d",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F2" && muon_=="M2" )
      pset.addParameter<unsigned int>("QualityEnableME1e",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F2" && muon_=="M3" )
      pset.addParameter<unsigned int>("QualityEnableME1f",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F3" && muon_=="M1" )
      pset.addParameter<unsigned int>("QualityEnableME2a",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F3" && muon_=="M2" )
      pset.addParameter<unsigned int>("QualityEnableME2b",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F3" && muon_=="M3" )
      pset.addParameter<unsigned int>("QualityEnableME2c",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F4" && muon_=="M1" )
      pset.addParameter<unsigned int>("QualityEnableME3a",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F4" && muon_=="M2" )
      pset.addParameter<unsigned int>("QualityEnableME3b",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F4" && muon_=="M3" )
      pset.addParameter<unsigned int>("QualityEnableME3c",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F5" && muon_=="M1" )
      pset.addParameter<unsigned int>("QualityEnableME4a",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F5" && muon_=="M2" )
      pset.addParameter<unsigned int>("QualityEnableME4b",strtol(writeValue_.c_str(),'\0',16));
    if( register_=="CSR_LQE" && chip_=="F5" && muon_=="M3" )
      pset.addParameter<unsigned int>("QualityEnableME4c",strtol(writeValue_.c_str(),'\0',16));

    if( register_=="CSR_KFL" )//&& chip_=="SP" && muon_=="MA" )
      pset.addParameter<unsigned int>("kill_fiber",strtol(writeValue_.c_str(),'\0',16));
  }


  // add the DAT_ETA registers to the pset
  pset.addParameter<unsigned int>("mindetap"     , mindetap     );
  pset.addParameter<unsigned int>("mindetap_halo", mindetap_halo);
  
  pset.addParameter< std::vector<unsigned int> >("EtaMin",etamin);

  pset.addParameter<unsigned int>("mindeta12_accp",  mindeta12_accp );
  pset.addParameter<unsigned int>("mindeta13_accp" , mindeta13_accp );
  pset.addParameter<unsigned int>("mindeta112_accp", mindeta112_accp);
  pset.addParameter<unsigned int>("mindeta113_accp", mindeta113_accp);

  pset.addParameter< std::vector<unsigned int> >("EtaMax",etamax);

  pset.addParameter<unsigned int>("maxdeta12_accp",  maxdeta12_accp );
  pset.addParameter<unsigned int>("maxdeta13_accp" , maxdeta13_accp );
  pset.addParameter<unsigned int>("maxdeta112_accp", maxdeta112_accp);
  pset.addParameter<unsigned int>("maxdeta113_accp", maxdeta113_accp);

  pset.addParameter< std::vector<unsigned int> >("EtaWindows",etawin);

  pset.addParameter<unsigned int>("maxdphi12_accp",  maxdphi12_accp );
  pset.addParameter<unsigned int>("maxdphi13_accp" , maxdphi13_accp );
  pset.addParameter<unsigned int>("maxdphi112_accp", maxdphi112_accp);
  pset.addParameter<unsigned int>("maxdphi113_accp", maxdphi113_accp);
  
  pset.addParameter<unsigned int>("mindphip",      mindphip     );
  pset.addParameter<unsigned int>("mindphip_halo", mindphip_halo);
  
  pset.addParameter<unsigned int>("straightp", straightp);
  pset.addParameter<unsigned int>("curvedp"  , curvedp  );
  pset.addParameter<unsigned int>("mbaPhiOff", mbaPhiOff);
  pset.addParameter<unsigned int>("mbbPhiOff", mbbPhiOff);

  
  LogDebug("L1MuCSCTFParametersTester") << "AFTER READING THE DBS VALUES" << std::endl
                                        << "mindetap     =" << mindetap      << std::endl
                                        << "mindetap_halo=" << mindetap_halo << std::endl
      
                                        << "etamin[0]=" << etamin[0] << std::endl
                                        << "etamin[1]=" << etamin[1] << std::endl
                                        << "etamin[2]=" << etamin[2] << std::endl
                                        << "etamin[3]=" << etamin[3] << std::endl
                                        << "etamin[4]=" << etamin[4] << std::endl
                                        << "etamin[5]=" << etamin[5] << std::endl
                                        << "etamin[6]=" << etamin[6] << std::endl
                                        << "etamin[7]=" << etamin[7] << std::endl

                                        << "mindeta12_accp =" << mindeta12_accp  << std::endl 
                                        << "mindeta13_accp =" << mindeta13_accp  << std::endl
                                        << "mindeta112_accp=" << mindeta112_accp << std::endl
                                        << "mindeta113_accp=" << mindeta113_accp << std::endl
      
                                        << "etamax[0]=" << etamax[0] << std::endl
                                        << "etamax[1]=" << etamax[1] << std::endl
                                        << "etamax[2]=" << etamax[2] << std::endl
                                        << "etamax[3]=" << etamax[3] << std::endl
                                        << "etamax[4]=" << etamax[4] << std::endl
                                        << "etamax[5]=" << etamax[5] << std::endl
                                        << "etamax[6]=" << etamax[6] << std::endl
                                        << "etamax[7]=" << etamax[7] << std::endl

                                        << "maxdeta12_accp =" << maxdeta12_accp  << std::endl
                                        << "maxdeta13_accp =" << maxdeta13_accp  << std::endl
                                        << "maxdeta112_accp=" << maxdeta112_accp << std::endl
                                        << "maxdeta113_accp=" << maxdeta113_accp << std::endl
  
                                        << "etawin[0]=" << etawin[0] << std::endl
                                        << "etawin[1]=" << etawin[1] << std::endl
                                        << "etawin[2]=" << etawin[2] << std::endl
                                        << "etawin[3]=" << etawin[3] << std::endl
                                        << "etawin[4]=" << etawin[4] << std::endl
                                        << "etawin[5]=" << etawin[5] << std::endl
                                        << "etawin[6]=" << etawin[6] << std::endl
  
                                        << "maxdphi12_accp =" << maxdphi12_accp  << std::endl
                                        << "maxdphi13_accp =" << maxdphi13_accp  << std::endl
                                        << "maxdphi112_accp=" << maxdphi112_accp << std::endl
                                        << "maxdphi113_accp=" << maxdphi113_accp << std::endl
                           
                                        << "mindphip     =" << mindphip      << std::endl
                                        << "mindphip_halo=" << mindphip_halo << std::endl
  
                                        << "straightp=" << straightp << std::endl
                                        << "curvedp  =" << curvedp   << std::endl
                                        << "mbaPhiOff=" << mbaPhiOff << std::endl
                                        << "mbbPhiOff=" << mbbPhiOff << std::endl;
  
  // ---------------------------------------------------------

  return pset;

}



void L1MuCSCTFConfiguration::print(std::ostream& myStr) const {
  myStr << "\nL1 Mu CSCTF Parameters \n" << std::endl;

  for (int iSP=0;iSP<12;iSP++) {
    myStr << "============================================="    << std::endl;
    myStr << "Printing out Global Tag Content for SP " << iSP+1 << std::endl;
    myStr << registers[iSP];
    myStr << "============================================="    << std::endl;
  }
}
