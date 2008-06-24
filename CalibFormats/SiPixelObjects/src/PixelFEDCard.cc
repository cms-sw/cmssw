// Read the pixelFED setup parameters from an ASCII file
// Will Johns & Danek Kotlinski 04/06.

#include <iostream>

#include "CalibFormats/SiPixelObjects/interface/PixelFEDCard.h"

#include <cassert>
#include <sstream>
#include <map>

using namespace std;

using namespace pos;


PixelFEDCard::PixelFEDCard(): 
  PixelConfigBase(" "," "," "){}

// modified by MR on 24-04-2008 12:05:42
// Read configuration from DB
PixelFEDCard::PixelFEDCard(vector<vector<string> > &tableMat):PixelConfigBase(" "," "," ")
{
  vector<string> ins = tableMat[0];
  map<string , int > colM;
  vector<string> colNames;
  bool first = true ;
  /**
    
    Name                                           Null?    Type
    ----------------------------------------- -------- ----------------------------

    CONFIG_KEY_ID                             NOT NULL      NUMBER(38)
    CONFIG_KEY                                NOT NULL      VARCHAR2(80)
    FED_CHAN_VERS                                           VARCHAR2(40)
    FED_CONFIG_VERS                                         VARCHAR2(40)
    SERIAL_NUMBER                                           VARCHAR2(40)
    PIXEL_FED                                 NOT NULL      VARCHAR2(200)
    CRATE_NUMBER                              NOT NULL      NUMBER(38)
    SLOT_NUMBER                               NOT NULL      NUMBER(38)
    VME_ADDRS_HEX                                           VARCHAR2(17)
    CHANNEL_ID                                NOT NULL      NUMBER(10)
    NUM_ROCS                                  NOT NULL      NUMBER(4)
    CHAN_OFFST_DAC                            NOT NULL      NUMBER(10)
    CHAN_DELAY                                NOT NULL      NUMBER(10)
    CHAN_BHIGH                                NOT NULL      NUMBER(10)
    CHAN_BLOW                                 NOT NULL      NUMBER(10)
    CHAN_UB                                   NOT NULL      NUMBER(10)
    OPT1_CAP                                  NOT NULL      NUMBER(38)
    OPT2_CAP                                  NOT NULL      NUMBER(38)
    OPT3_CAP                                  NOT NULL      NUMBER(38)
    OPT1_INP                                  NOT NULL      NUMBER(38)
    OPT2_INP                                  NOT NULL      NUMBER(38)
    OPT3_INP                                  NOT NULL      NUMBER(38)
    OPT1_OUT                                  NOT NULL      NUMBER(38)
    OPT2_OUT                                  NOT NULL      NUMBER(38)
    OPT3_OUT                                  NOT NULL      NUMBER(38)
    NORTH_CLKPHB                              NOT NULL      NUMBER(38)
    NORTHCENTER_CLKPHB                        NOT NULL      NUMBER(38)
    SOUTHCENTER_CLKPHB                        NOT NULL      NUMBER(38)
    SOUTH_CLKPHB                              NOT NULL      NUMBER(38)
    NORTH_CTRL                                NOT NULL      NUMBER(38)
    NORTHCENTER_CTRL                          NOT NULL      NUMBER(38)
    SOUTHCENTER_CTRL                          NOT NULL      NUMBER(38)
    SOUTH_CTRL                                NOT NULL      NUMBER(38)
    REG1_TTCRX_FDLA                           NOT NULL      NUMBER(38)
    REG2_TTCRX_CDLA                           NOT NULL      NUMBER(38)
    REG3_TTCRX_CLKD2                          NOT NULL      NUMBER(38)
    CENTER_CTRL                               NOT NULL      NUMBER(38)
    CENTER_MODE                               NOT NULL      NUMBER(38)
    B1_ADCGN                                  NOT NULL      NUMBER(38)
    B2_ADCGN                                  NOT NULL      NUMBER(38)
    B3_ADCGN                                  NOT NULL      NUMBER(38)
    B4_ADCGN                                  NOT NULL      NUMBER(38)
    NORTH_BADJ                                NOT NULL      NUMBER(38)
    NORTHCENTER_BADJ                          NOT NULL      NUMBER(38)
    SOUTHCENTER_BADJ                          NOT NULL      NUMBER(38)
    SOUTH_BADJ                                NOT NULL      NUMBER(38)
    NORTH_TBMMASK                                           NUMBER(38)
    NORTHCENTER_TBMMASK                                     NUMBER(38)
    SOUTHCENTER_TBMMASK                                     NUMBER(38)
    SOUTH_TBMMASK                                           NUMBER(38)
    NORTH_PWORD                                             NUMBER(38)
    NORTHCENTER_PWORD                                       NUMBER(38)
    SOUTH_PWORD                                             NUMBER(38)
    SOUTHCENTER_PWORD                                       NUMBER(38)
    SPECDAC                                                 NUMBER(38)
    OOS_LVL                                                 NUMBER(38)
    ERR_LVL                                                 NUMBER(38)
    NORTH_FIFO1_BZ_LVL                                      NUMBER(38)
    NORTHCENTER_FIFO1_BZ_LVL                                NUMBER(38)
    SOUTHCENTER_FIFO1_BZ_LVL                                NUMBER(38)
    SOUTH_FIFO1_BZ_LVL                                      NUMBER(38)
  */
  colNames.push_back("CONFIG_KEY_ID"            );
  colNames.push_back("CONFIG_KEY"               );
  colNames.push_back("FED_CHAN_VERS"            );
  colNames.push_back("FED_CONFIG_VERS"          );
  colNames.push_back("SERIAL_NUMBER"            );
  colNames.push_back("PIXEL_FED"                );
  colNames.push_back("CRATE_NUMBER"             );
  colNames.push_back("SLOT_NUMBER"              );
  colNames.push_back("VME_ADDRS_HEX"            );
  colNames.push_back("CHANNEL_ID"               );
  colNames.push_back("NUM_ROCS"                 );
  colNames.push_back("CHAN_OFFST_DAC"           );
  colNames.push_back("CHAN_DELAY"               );
  colNames.push_back("CHAN_BHIGH"               );
  colNames.push_back("CHAN_BLOW"                );
  colNames.push_back("CHAN_UB"                  );
  colNames.push_back("OPT1_CAP"                 );
  colNames.push_back("OPT2_CAP"                 );
  colNames.push_back("OPT3_CAP"                 );
  colNames.push_back("OPT1_INP"                 );
  colNames.push_back("OPT2_INP"                 );
  colNames.push_back("OPT3_INP"                 );
  colNames.push_back("OPT1_OUT"                 );
  colNames.push_back("OPT2_OUT"                 );
  colNames.push_back("OPT3_OUT"                 );
  colNames.push_back("NORTH_CLKPHB"             );
  colNames.push_back("NORTHCENTER_CLKPHB"       );
  colNames.push_back("SOUTHCENTER_CLKPHB"       );
  colNames.push_back("SOUTH_CLKPHB"             );
  colNames.push_back("NORTH_CTRL"               );
  colNames.push_back("NORTHCENTER_CTRL"         );
  colNames.push_back("SOUTHCENTER_CTRL"         );
  colNames.push_back("SOUTH_CTRL"               );
  colNames.push_back("REG1_TTCRX_FDLA"          );
  colNames.push_back("REG2_TTCRX_CDLA"          );
  colNames.push_back("REG3_TTCRX_CLKD2"         );
  colNames.push_back("CENTER_CTRL"              );
  colNames.push_back("CENTER_MODE"              );
  colNames.push_back("B1_ADCGN"                 );
  colNames.push_back("B2_ADCGN"                 );
  colNames.push_back("B3_ADCGN"                 );
  colNames.push_back("B4_ADCGN"                 );
  colNames.push_back("NORTH_BADJ"               );
  colNames.push_back("NORTHCENTER_BADJ"         );
  colNames.push_back("SOUTHCENTER_BADJ"         );
  colNames.push_back("SOUTH_BADJ"               );
  colNames.push_back("NORTH_TBMMASK"            );
  colNames.push_back("NORTHCENTER_TBMMASK"      );
  colNames.push_back("SOUTHCENTER_TBMMASK"      );
  colNames.push_back("SOUTH_TBMMASK"            );
  colNames.push_back("NORTH_PWORD"              );
  colNames.push_back("NORTHCENTER_PWORD"        );
  colNames.push_back("SOUTH_PWORD"              );
  colNames.push_back("SOUTHCENTER_PWORD"        );
  colNames.push_back("SPECDAC"                  );
  colNames.push_back("OOS_LVL"                  );
  colNames.push_back("ERR_LVL"                  );
  colNames.push_back("NORTH_FIFO1_BZ_LVL"       );
  colNames.push_back("NORTHCENTER_FIFO1_BZ_LVL" );
  colNames.push_back("SOUTHCENTER_FIFO1_BZ_LVL" );
  colNames.push_back("SOUTH_FIFO1_BZ_LVL"       );
  
  for(unsigned int c = 0 ; c < ins.size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[0][c] == colNames[n]){
	    colM[colNames[n]] = c;
	    break;
	  }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << "[PixelTBMSettings::PixelTBMSettings()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  // disentagle different tables
  int size[3]   ;
  int indexsize = 0 ;   
  for(unsigned int r = 0 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
    if(tableMat[r].size() == 0)
      {
	cout << "__________________ NEW TABLE __________________"<< endl ;
        size[indexsize] = r ;
        cout << "size[" << indexsize << "] = " << size[indexsize] << endl ;
        indexsize++ ;
	continue ;
      }
    for(vector<string>::iterator it = tableMat[r].begin() ; it != tableMat[r].end() ; it++)
      {
// 	cout << *it <<"["<<&*it<<"]\t"  ;
	cout << *it <<"\t"  ;
      }
    cout << endl ;
  }

  // Read below quantities pertaining to a single FED that are equal accross 36 channels
  if(first)
    {
      first = false ;
      //VME base address 
      string fakeHex = "0x" + tableMat[1][colM["VME_ADDRS_HEX"]] ;
      //Fed Base Address
      sscanf(fakeHex.c_str(),"%lx",&FEDBASE_0);
      sscanf(tableMat[1][colM["PIXEL_FED"]].c_str(), "PxlFED_%ld",&fedNumber);
      //Settable optical input parameters (one for each 12-receiver)
      opt_cap[0]   = atoi(tableMat[1][colM["OPT1_CAP"]].c_str()) ;
      opt_cap[1]   = atoi(tableMat[1][colM["OPT2_CAP"]].c_str()) ;
      opt_cap[2]   = atoi(tableMat[1][colM["OPT3_CAP"]].c_str()) ;
      opt_inadj[0] = atoi(tableMat[1][colM["OPT1_INP"]].c_str()) ;
      opt_inadj[1] = atoi(tableMat[1][colM["OPT2_INP"]].c_str()) ;
      opt_inadj[2] = atoi(tableMat[1][colM["OPT3_INP"]].c_str()) ;
      opt_ouadj[0] = atoi(tableMat[1][colM["OPT1_OUT"]].c_str()) ;
      opt_ouadj[1] = atoi(tableMat[1][colM["OPT2_OUT"]].c_str()) ;
      opt_ouadj[2] = atoi(tableMat[1][colM["OPT3_OUT"]].c_str()) ;

      //clock phases, use bits 0-8, select the clock edge
      clkphs1_9   = atoi(tableMat[1][colM["NORTH_CLKPHB"]].c_str()       ) ; // TO BE VERIFIED
      clkphs10_18 = atoi(tableMat[1][colM["NORTHCENTER_CLKPHB"]].c_str() ) ; // TO BE VERIFIED
      clkphs19_27 = atoi(tableMat[1][colM["SOUTHCENTER_CLKPHB"]].c_str() ) ; // TO BE VERIFIED
      clkphs28_36 = atoi(tableMat[1][colM["SOUTH_CLKPHB"]].c_str()       ) ; // TO BE VERIFIED

      // Control register and delays for the TTCrx
      CoarseDel   = atoi(tableMat[1][colM["REG2_TTCRX_CDLA"]].c_str()  ) ;
      ClkDes2     = atoi(tableMat[1][colM["REG3_TTCRX_CLKD2"]].c_str() ) ;
      FineDes2Del = atoi(tableMat[1][colM["REG1_TTCRX_FDLA"]].c_str()  ) ;
      
      //data Regs adjustable fifo Almost Full levels
      Nfifo1Bzlvl  =  atoi(tableMat[1][colM["NORTH_FIFO1_BZ_LVL"]].c_str()       ) ;
      NCfifo1Bzlvl =  atoi(tableMat[1][colM["NORTHCENTER_FIFO1_BZ_LVL"]].c_str() ) ;
      SCfifo1Bzlvl =  atoi(tableMat[1][colM["SOUTHCENTER_FIFO1_BZ_LVL"]].c_str() ) ;
      Sfifo1Bzlvl  =  atoi(tableMat[1][colM["SOUTTH_FIFO1_BZ_LVL"]].c_str()      ) ;

      //Bits (1st 8) used to mask TBM trailer bits
      N_TBMmask  =   atoi(tableMat[1][colM["NORTH_TBMMASK"]].c_str()       ) ;
      NC_TBMmask =   atoi(tableMat[1][colM["NORTHCENTER_TBMMASK"]].c_str() ) ;
      SC_TBMmask =   atoi(tableMat[1][colM["SOUTHCENTER_TBMMASK"]].c_str() ) ;
      S_TBMmask  =   atoi(tableMat[1][colM["SOUTH_TBMMASK"]].c_str()       ) ;
      
      //Bits (1st 8) used to set the Private Word in the gap and filler words
      N_Pword  =   atoi(tableMat[1][colM["NORTH_PWORD"]].c_str()       ) ;
      NC_Pword =   atoi(tableMat[1][colM["NORTHCENTER_PWORD"]].c_str() ) ;
      SC_Pword =   atoi(tableMat[1][colM["SOUTHCENTER_PWORD"]].c_str() ) ;
      S_Pword  =   atoi(tableMat[1][colM["SOUTH_PWORD"]].c_str()       ) ;

       //These bit sets the special dac mode for random triggers 
      SpecialDac = atoi(tableMat[1][colM["SPECDAC"]].c_str()       ) ;

      //These bits set the number of Out of consecutive out of sync events until a TTs OOs 
      Ooslvl =  atoi(tableMat[1][colM["OOS_LVL"]].c_str()       ) ;
      //These bits set the number of Empty events until a TTs Error 
      Errlvl =  atoi(tableMat[1][colM["ERR_LVL"]].c_str()       ) ;

      //Control Regs for setting ADC 1Vpp and 2Vpp
      Nadcg  =   atoi(tableMat[1][colM["B1_ADCGN"]].c_str()       ) ;
      NCadcg =   atoi(tableMat[1][colM["B2_ADCGN"]].c_str()       ) ;
      SCadcg =   atoi(tableMat[1][colM["B3_ADCGN"]].c_str()       ) ;
      Sadcg  =   atoi(tableMat[1][colM["B4_ADCGN"]].c_str()       ) ;

    } // end of 'first' condition
  for(int r = 1 ; r < size[0] ; r++)    //Goes to every row of the FIRST Matrix (MUST BE 36, one for each FED channel)
    {
      //Number of ROCS per FED channel
      NRocs[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1]    = atoi(tableMat[r][colM["NUM_ROCS"]].c_str()       ) ; 
      //input offset dac (one for each channel)                                                                     
      offs_dac[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1] = atoi(tableMat[r][colM["CHAN_OFFST_DAC"]].c_str() ) ; 
      //Channel delays, one for each channel, 0=15                                                                  
      DelayCh[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1]  = atoi(tableMat[r][colM["CHAN_DELAY"]].c_str()     ) ; 
      //Blacks and Ultra-blacks, 3 limit per channel                                                                
      BlackHi[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1]  = atoi(tableMat[r][colM["CHAN_BHIGH"]].c_str()     ) ; 
      BlackLo[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1]  = atoi(tableMat[r][colM["CHAN_BLOW"]].c_str()      ) ; 
      Ublack[atoi(tableMat[r][colM["CHANNEL_ID"]].c_str())-1]   = atoi(tableMat[r][colM["CHAN_UB"]].c_str()        ) ; 
    }
  readDBTBMLevels(tableMat, size[0]+1, size[1]) ;
  readDBROCLevels(tableMat, size[1]+1, size[2]) ;
}

void PixelFEDCard::readDBTBMLevels(std::vector<std::vector<std::string> > &tableMat, int firstRow, int lastRow)
{
  string mthn = "[PixelFEDCard::readDBTBMLevels()] ";
  vector<string> ins = tableMat[firstRow];
  map<string , int > colM;
  vector<string> colNames;

  /**
     CONFIG_KEY_ID                                      NUMBER(38)
     CONFIG_KEY                                         VARCHAR2(80)
     VERSION                                            VARCHAR2(40)
     CONDITION_DATA_SET_ID                              NUMBER(38)
     KIND_OF_CONDITION_ID                               NUMBER(38)
     KIND_OF_COND                                       VARCHAR2(40)
     PXLFED_NAME                                        VARCHAR2(200)
     FED_CHAN                                           NUMBER(38)
     TBM_PART_ID                                        NUMBER(38)
     TBM_SER_NUM                                        VARCHAR2(40)
     PANEL_NAME                                         VARCHAR2(99)
     HUB_ADDRS                                          NUMBER(38)
     TBMA_HEAD_L0                                       VARCHAR2(200)
     TBMA_HEAD_L1                                       VARCHAR2(200)
     TBMA_HEAD_L2                                       VARCHAR2(200)
     TBMA_HEAD_L3                                       VARCHAR2(200)
     TBMA_HEAD_L4                                       VARCHAR2(200)
     TBMA_TRAIL_L0                                      VARCHAR2(200)
     TBMA_TRAIL_L1                                      VARCHAR2(200)
     TBMA_TRAIL_L2                                      VARCHAR2(200)
     TBMA_TRAIL_L3                                      VARCHAR2(200)
     TBMA_TRAIL_L4                                      VARCHAR2(200)
  */
  
  colNames.push_back("CONFIG_KEY_ID"           );
  colNames.push_back("CONFIG_KEY"              );
  colNames.push_back("VERSION"                 );
  colNames.push_back("CONDITION_DATA_SET_ID"   );
  colNames.push_back("KIND_OF_CONDITION_ID"    );
  colNames.push_back("KIND_OF_COND"            );
  colNames.push_back("PXLFED_NAME"             );
  colNames.push_back("FED_CHAN"                );
  colNames.push_back("TBM_PART_ID"             );
  colNames.push_back("TBM_SER_NUM"             );
  colNames.push_back("PANEL_NAME"              );
  colNames.push_back("HUB_ADDRS"               );
  colNames.push_back("TBMA_HEAD_L0"            );
  colNames.push_back("TBMA_HEAD_L1"            );
  colNames.push_back("TBMA_HEAD_L2"            );
  colNames.push_back("TBMA_HEAD_L3"            );
  colNames.push_back("TBMA_HEAD_L4"            );
  colNames.push_back("TBMA_TRAIL_L0"           );
  colNames.push_back("TBMA_TRAIL_L1"           );
  colNames.push_back("TBMA_TRAIL_L2"           );
  colNames.push_back("TBMA_TRAIL_L3"           );
  colNames.push_back("TBMA_TRAIL_L4"           );

  // Retrieve header row and cross check that everyfield is there.
  for(unsigned int c = 0 ; c < ins.size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[firstRow][c] == colNames[n]){
	    colM[colNames[n]] = c;
	    break;
	  }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << mthn << "\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  for(int r = firstRow + 1 ; r < lastRow ; r++)    //Goes to every row of the Matrix (MUST BE 36, one for each FED channel)
    {
      //Signal levels for the TBM, one per channel
      TBM_L0[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_HEAD_L0"]].c_str()	 ) ;
      TBM_L1[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_HEAD_L1"]].c_str()	 ) ;
      TBM_L2[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_HEAD_L2"]].c_str()	 ) ;
      TBM_L3[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_HEAD_L3"]].c_str()	 ) ;
      TBM_L4[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_HEAD_L4"]].c_str()	 ) ;
      TRL_L0[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_TRAIL_L0"]].c_str()    ) ;
      TRL_L1[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_TRAIL_L1"]].c_str()    ) ;
      TRL_L2[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_TRAIL_L2"]].c_str()    ) ;
      TRL_L3[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_TRAIL_L3"]].c_str()    ) ;
      TRL_L4[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1] = atoi(tableMat[r][colM["TBMA_TRAIL_L4"]].c_str()    ) ;
    }
}

void PixelFEDCard::readDBROCLevels(std::vector<std::vector<std::string> > &tableMat, int firstRow, int lastRow)
{
  string mthn = "[PixelFEDCard::readDBROCLevels()] ";
  map<string , int > colM;
  vector<string> colNames;

  /**
     CONFIG_KEY_ID                                      NUMBER(38)
     CONFIG_KEY                                         VARCHAR2(80)
     VERSION                                            VARCHAR2(40)
     KIND_OF_COND                                       VARCHAR2(40)
     PXLFED_NAME                                        VARCHAR2(200)
     FED_CHAN                                           NUMBER(38)
     AOH_CHAN                                           NUMBER(38)
     ROC_NAME                                           VARCHAR2(187)
     HUB_ADDRS                                          NUMBER(38)
     PORT_NUMBER                                        NUMBER(10)
     I2C_ADDR                                           NUMBER
     GEOM_ROC_NUM                                       NUMBER(10)
     FED_ROC_NUM                                        NUMBER(10)
     ROC_L0                                             VARCHAR2(200)
     ROC_L1                                             VARCHAR2(200)
     ROC_L2                                             VARCHAR2(200)
     ROC_L3                                             VARCHAR2(200)
     ROC_L4                                             VARCHAR2(200)
  */
  
  colNames.push_back("CONFIG_KEY_ID"    );
  colNames.push_back("CONFIG_KEY"       );
  colNames.push_back("VERSION"          );
  colNames.push_back("KIND_OF_COND"     );
  colNames.push_back("PXLFED_NAME"      );
  colNames.push_back("FED_CHAN"         );
  colNames.push_back("AOH_CHAN"         );
  colNames.push_back("ROC_NAME"         );
  colNames.push_back("HUB_ADDRS"        );
  colNames.push_back("PORT_NUMBER"      );
  colNames.push_back("ROC_I2C_ADDR"     );
  colNames.push_back("GEOM_ROC_NUM"     );
  colNames.push_back("FED_ROC_NUM"      );
  colNames.push_back("ROC_L0"           );
  colNames.push_back("ROC_L1"           );
  colNames.push_back("ROC_L2"           );
  colNames.push_back("ROC_L3"           );
  colNames.push_back("ROC_L4"           );
  
  // Retrieve header row and cross check that everyfield is there.
  for(unsigned int c = 0 ; c < tableMat[firstRow].size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[firstRow][c] == colNames[n]){
	    colM[colNames[n]] = c;
	    break;
	  }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << mthn << "\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  // Address levels 1 per channel (36) per roc(max=26)
//   int ROC_L0[36][26],ROC_L1[36][26],ROC_L2[36][26],ROC_L3[36][26],ROC_L4[36][26];

  for(int r = firstRow + 1 ; r < lastRow ; r++)    //Goes to every row of the Matrix (MUST BE 36, one for each FED channel)
    {
      ROC_L0[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1][atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str())-1] = atoi(tableMat[r][colM["ROC_L0"]].c_str()) ;
      ROC_L1[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1][atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str())-1] = atoi(tableMat[r][colM["ROC_L1"]].c_str()) ;
      ROC_L2[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1][atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str())-1] = atoi(tableMat[r][colM["ROC_L2"]].c_str()) ;
      ROC_L3[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1][atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str())-1] = atoi(tableMat[r][colM["ROC_L3"]].c_str()) ;
      ROC_L4[atoi(tableMat[r][colM["FED_CHAN"]].c_str())-1][atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str())-1] = atoi(tableMat[r][colM["ROC_L4"]].c_str()) ;
    }
  
}


// Read the configuration parameters from file
PixelFEDCard::PixelFEDCard(string fileName):
  PixelConfigBase(" "," "," ")
{

  //const bool localDEBUG = true;
  const bool localDEBUG = false;
  
  // Added by Dario (March 26th, 2008): insure variables are all cleared before read-in
  clear() ;

  //cout<<" Get setup parameters from file "<<fileName<<endl;
  FILE *infile = fopen((fileName.c_str()),"r");
  if (infile == NULL) {
    cout<<"No parameter file:"<<fileName<<endl; 
    return;
  }
  
  //Fed Base Address
  fscanf(infile,"FED Base address                         :%lx\n",
         &FEDBASE_0);
  fscanf(infile,"FEDID Number                             :%lx\n",
         &fedNumber);

  printf("FED Base address, FED # :%lx\n",FEDBASE_0);
  //if(FEDBASE != FEDBASE_0) cout<<" Inconsistent FED base address?"<<endl;
  printf("FEDID # :%lx\n",fedNumber);
 
  // Number of ROCs
  int ijx=0;
  for(int i=0;i<36;i++){
  ijx=i+1;
    fscanf(infile,"Number of ROCs Chnl %d:%d \n",&ijx,&NRocs[i]);
    if(localDEBUG)printf("Number of ROCs per Chnl %d:%d \n",ijx,NRocs[i]);
  }

  //Settable optical input parameters
  fscanf(infile,"Optical reciever 1  Capacitor Adjust(0-3):%d\n",&opt_cap[0]);
  fscanf(infile,"Optical reciever 2  Capacitor Adjust(0-3):%d\n",&opt_cap[1]);
  fscanf(infile,"Optical reciever 3  Capacitor Adjust(0-3):%d\n",&opt_cap[2]);
  fscanf(infile,"Optical reciever 1  Input Offset (0-15)  :%d\n",&opt_inadj[0]);
  fscanf(infile,"Optical reciever 2  Input Offset (0-15)  :%d\n",&opt_inadj[1]);
  fscanf(infile,"Optical reciever 3  Input Offset (0-15)  :%d\n",&opt_inadj[2]);
  fscanf(infile,"Optical reciever 1 Output Offset (0-3)   :%d\n",&opt_ouadj[0]);
  fscanf(infile,"Optical reciever 2 Output Offset (0-3)   :%d\n",&opt_ouadj[1]);
  fscanf(infile,"Optical reciever 3 Output Offset (0-3)   :%d\n",&opt_ouadj[2]);
  
  if(localDEBUG) {
    printf("Optical reciever 1  Capacitor Adjust(0-3):%d\n",opt_cap[0]);
    printf("Optical reciever 2  Capacitor Adjust(0-3):%d\n",opt_cap[1]);
    printf("Optical reciever 3  Capacitor Adjust(0-3):%d\n",opt_cap[2]);
    printf("Optical reciever 1  Input Offset (0-15)   :%d\n",opt_inadj[0]);
    printf("Optical reciever 2  Input Offset (0-15)   :%d\n",opt_inadj[1]);
    printf("Optical reciever 3  Input Offset (0-15)   :%d\n",opt_inadj[2]);
    printf("Optical reciever 1 Output Offset (0-3)  :%d\n",opt_ouadj[0]);
    printf("Optical reciever 2 Output Offset (0-3)  :%d\n",opt_ouadj[1]);
    printf("Optical reciever 3 Output Offset (0-3)  :%d\n",opt_ouadj[2]);
  }

  //input offset dac
  for(int i=0;i<36;i++) {
    fscanf(infile,"Offset DAC channel %d:%d\n",&ijx,&offs_dac[i]);
    if(localDEBUG) printf("Offset DAC channel %d:%d\n",i+1,offs_dac[i]);
  }
  
  //clock phases
  fscanf(infile,"Clock Phase Bits ch   1-9:%x\n",& clkphs1_9 );
  fscanf(infile,"Clock Phase Bits ch 10-18:%x\n",&clkphs10_18);
  fscanf(infile,"Clock Phase Bits ch 19-27:%x\n",&clkphs19_27);
  fscanf(infile,"Clock Phase Bits ch 28-36:%x\n",&clkphs28_36);
  if(localDEBUG)printf("Clock Phase Bits ch    1-9:%x\n",clkphs1_9 );
  if(localDEBUG)printf("Clock Phase Bits ch  10-18:%x\n",clkphs10_18 );
  if(localDEBUG)printf("Clock Phase Bits ch  19-27:%x\n",clkphs19_27 );
  if(localDEBUG)printf("Clock Phase Bits ch  28-36:%x\n",clkphs28_36 );
  
  //Blacks 
  for(int i=0;i<36;i++){
    fscanf(infile,"Black HiThold ch %d:%d \n",&ijx,&BlackHi[i]);
    fscanf(infile,"Black LoThold ch %d:%d \n",&ijx,&BlackLo[i]);
    fscanf(infile,"ULblack Thold ch %d:%d \n",&ijx, &Ublack[i]);
    if(localDEBUG)printf("Black HiThold ch %d:%d\n",ijx,BlackHi[i]);
    if(localDEBUG)printf("Black LoThold ch %d:%d\n",ijx,BlackLo[i]);
    if(localDEBUG)printf("ULblack Thold ch %d:%d\n",ijx, Ublack[i]);
  }
  
  //Channel delays
  for(int i=0;i<36;i++) {
    fscanf(infile,"Delay channel %d(0-15):%d\n",&ijx,&DelayCh[i]);
    if(localDEBUG) 
      printf("Delay channel %d(0-15):%d\n",i+1,DelayCh[i]);
  }
  
  //Signal levels
  for(int i=0;i<36;i++) {
    fscanf(infile,"TBM level 0 Channel  %d:%d\n",&ijx,&TBM_L0[i]);
    fscanf(infile,"TBM level 1 Channel  %d:%d\n",&ijx,&TBM_L1[i]);
    fscanf(infile,"TBM level 2 Channel  %d:%d\n",&ijx,&TBM_L2[i]);
    fscanf(infile,"TBM level 3 Channel  %d:%d\n",&ijx,&TBM_L3[i]);
    fscanf(infile,"TBM level 4 Channel  %d:%d\n",&ijx,&TBM_L4[i]);
    if(localDEBUG)printf("TBM level 0 Channel  %d:%d\n",ijx,TBM_L0[i]);
    if(localDEBUG)printf("TBM level 1 Channel  %d:%d\n",ijx,TBM_L1[i]);
    if(localDEBUG)printf("TBM level 2 Channel  %d:%d\n",ijx,TBM_L2[i]);
    if(localDEBUG)printf("TBM level 3 Channel  %d:%d\n",ijx,TBM_L3[i]);
    if(localDEBUG)printf("TBM level 4 Channel  %d:%d\n",ijx,TBM_L4[i]);
    
    int ijy=0;
    for(int j=0;j<NRocs[i];j++) {
      fscanf(infile,"ROC%d level 0 Channel  %d :%d\n",
             &ijy,&ijx,&ROC_L0[i][j]);
      fscanf(infile,"ROC%d level 1 Channel  %d :%d\n",
             &ijy,&ijx,&ROC_L1[i][j]);
      fscanf(infile,"ROC%d level 2 Channel  %d :%d\n",
             &ijy,&ijx,&ROC_L2[i][j]);
      fscanf(infile,"ROC%d level 3 Channel  %d :%d\n",
             &ijy,&ijx,&ROC_L3[i][j]);
      fscanf(infile,"ROC%d level 4 Channel  %d :%d\n",
             &ijy,&ijx,&ROC_L4[i][j]);
      if(localDEBUG)
        printf("ROC%d level 0 Channel  %d :%d\n",ijy,ijx,ROC_L0[i][j]);
      if(localDEBUG)
        printf("ROC%d level 1 Channel  %d :%d\n",ijy,ijx,ROC_L1[i][j]);
      if(localDEBUG)
        printf("ROC%d level 2 Channel  %d :%d\n",ijy,ijx,ROC_L2[i][j]);
      if(localDEBUG)
        printf("ROC%d level 3 Channel  %d :%d\n",ijy,ijx,ROC_L3[i][j]);
      if(localDEBUG)
        printf("ROC%d level 4 Channel  %d :%d\n",ijy,ijx,ROC_L4[i][j]);
    }
      
    fscanf(infile,"TRLR level 0 Channel %d:%d\n",&ijx,&TRL_L0[i]);
    fscanf(infile,"TRLR level 1 Channel %d:%d\n",&ijx,&TRL_L1[i]);
    fscanf(infile,"TRLR level 2 Channel %d:%d\n",&ijx,&TRL_L2[i]);
    fscanf(infile,"TRLR level 3 Channel %d:%d\n",&ijx,&TRL_L3[i]);
    fscanf(infile,"TRLR level 4 Channel %d:%d\n",&ijx,&TRL_L4[i]);
    if(localDEBUG)printf("TRLR level 0 Channel %d:%d\n",ijx,TRL_L0[i]);
    if(localDEBUG)printf("TRLR level 1 Channel %d:%d\n",ijx,TRL_L1[i]);
    if(localDEBUG)printf("TRLR level 2 Channel %d:%d\n",ijx,TRL_L2[i]);
    if(localDEBUG)printf("TRLR level 3 Channel %d:%d\n",ijx,TRL_L3[i]);
    if(localDEBUG)printf("TRLR level 4 Channel %d:%d\n",ijx,TRL_L4[i]);
  }
  
  
  //These bits turn off(1) and on(0) channels
  fscanf(infile,"Channel Enbable bits chnls 1-9  (on = 0):%x\n",
         &Ncntrl);
  fscanf(infile,"Channel Enbable bits chnls 10-18(on = 0):%x\n",
         &NCcntrl);
  fscanf(infile,"Channel Enbable bits chnls 19-27(on = 0):%x\n",
         &SCcntrl);
  fscanf(infile,"Channel Enbable bits chnls 28-36(on = 0):%x\n",
         &Scntrl);
  if(localDEBUG)
    printf("Channel Enbable bits chnls 1-9  (on = 0):%x\n",Ncntrl);
  if(localDEBUG)
    printf("Channel Enbable bits chnls 10-18(on = 0):%x\n",NCcntrl);
  if(localDEBUG)
    printf("Channel Enbable bits chnls 19-27(on = 0):%x\n",SCcntrl);
  if(localDEBUG)
    printf("Channel Enbable bits chnls 28-36(on = 0):%x\n",Scntrl);
  
  //These are delays to the TTCrx
  fscanf(infile,"TTCrx Coarse Delay Register 2:%d\n",&CoarseDel);
  fscanf(infile,"TTCrc      ClkDes2 Register 3:%x\n",&ClkDes2);
  fscanf(infile,"TTCrc Fine Dlay ClkDes2 Reg 1:%d\n",&FineDes2Del);
  if(localDEBUG)printf("TTCrx Coarse Delay Register 2:%d\n",CoarseDel);
  if(localDEBUG)printf("TTCrc      ClkDes2 Register 3:%x\n",ClkDes2);
  if(localDEBUG)printf("TTCrc Fine Dlay ClkDes2 Reg 1:%d\n",FineDes2Del);
  
  // Control register
  fscanf(infile,"Center Chip Control Reg:%x\n",&Ccntrl);
  printf("Control Reg:0x%x\n",Ccntrl);
  fscanf(infile,"Initial Slink DAQ mode:%d\n",&modeRegister);
  printf("Mode Reg:%d\n",modeRegister);
  
   //These bits set ADC Gain/Range 1Vpp(0) and 2Vpp(1) for channels
  fscanf(infile,"Channel ADC Gain bits chnls  1-12(1Vpp = 0):%x\n",
         &Nadcg);
  fscanf(infile,"Channel ADC Gain bits chnls 13-20(1Vpp = 0):%x\n",
         &NCadcg);
  fscanf(infile,"Channel ADC Gain bits chnls 21-28(1Vpp = 0):%x\n",
         &SCadcg);
  fscanf(infile,"Channel ADC Gain bits chnls 29-36(1Vpp = 0):%x\n",
         &Sadcg);
  if(localDEBUG)
    printf("Channel ADC Gain bits chnls  1-12(1Vpp = 0):%x\n",Nadcg);
  if(localDEBUG)
    printf("Channel ADC Gain bits chnls 13-20(1Vpp = 0):%x\n",NCadcg);
  if(localDEBUG)
    printf("Channel ADC Gain bits chnls 21-28(1Vpp = 0):%x\n",SCadcg);
  if(localDEBUG)
    printf("Channel ADC Gain bits chnls 29-36(1Vpp = 0):%x\n",Sadcg);

       //These bits set Baseline adjustment value (common by FPGA)//can turn on by channel 
  fscanf(infile,"Channel Baseline Enbable chnls 1-9  (on = (0x1ff<<16)+):%x\n",
         &Nbaseln);
  fscanf(infile,"Channel Baseline Enbable chnls 10-18(on = (0x1ff<<16)+):%x\n",
         &NCbaseln);
  fscanf(infile,"Channel Baseline Enbable chnls 19-27(on = (0x1ff<<16)+):%x\n",
         &SCbaseln);
  fscanf(infile,"Channel Baseline Enbable chnls 28-36(on = (0x1ff<<16)+):%x\n",
         &Sbaseln);
  if(localDEBUG)
    printf("Channel Baseline Enbable chnls 1-9  (on = (0x1ff<<16)+):%x\n",Nbaseln);
  if(localDEBUG)
    printf("Channel Baseline Enbable chnls 10-18(on = (0x1ff<<16)+):%x\n",NCbaseln);
  if(localDEBUG)
    printf("Channel Baseline Enbable chnls 19-27(on = (0x1ff<<16)+):%x\n",SCbaseln);
  if(localDEBUG)
    printf("Channel Baseline Enbable chnls 28-36(on = (0x1ff<<16)+):%x\n",Sbaseln);

       //These bits set TBM trailer mask (common by FPGA) 
  fscanf(infile,"TBM trailer mask chnls 1-9  (0xff = all masked):%x\n",
         &N_TBMmask);
  fscanf(infile,"TBM trailer mask chnls 10-18(0xff = all masked):%x\n",
         &NC_TBMmask);
  fscanf(infile,"TBM trailer mask chnls 19-27(0xff = all masked):%x\n",
         &SC_TBMmask);
  fscanf(infile,"TBM trailer mask chnls 28-36(0xff = all masked):%x\n",
         &S_TBMmask);
  if(localDEBUG)
    printf("TBM trailer mask chnls 1-9  (0xff = all masked):%x\n",N_TBMmask);
  if(localDEBUG)
    printf("TBM trailer mask chnls 10-18(0xff = all masked):%x\n",NC_TBMmask);
  if(localDEBUG)
    printf("TBM trailer mask chnls 19-27(0xff = all masked):%x\n",SC_TBMmask);
  if(localDEBUG)
    printf("TBM trailer mask chnls 28-36(0xff = all masked):%x\n",S_TBMmask);

       //These bits set the Private fill/gap word value (common by FPGA) 
  fscanf(infile,"Private 8 bit word chnls 1-9  :%x\n",
         &N_Pword);
  fscanf(infile,"Private 8 bit word chnls 10-18:%x\n",
         &NC_Pword);
  fscanf(infile,"Private 8 bit word chnls 19-27:%x\n",
         &SC_Pword);
  fscanf(infile,"Private 8 bit word chnls 28-36:%x\n",
         &S_Pword);
  if(localDEBUG)
    printf("Private 8 bit word chnls 1-9  :%x\n",N_Pword);
  if(localDEBUG)
    printf("Private 8 bit word chnls 10-18:%x\n",NC_Pword);
  if(localDEBUG)
    printf("Private 8 bit word chnls 19-27:%x\n",SC_Pword);
  if(localDEBUG)
    printf("Private 8 bit word chnls 28-36:%x\n",S_Pword);

       //These bit sets the special dac mode for random triggers 
  fscanf(infile,"Special Random testDAC mode (on = 0x1, off=0x0):%x\n",
         &SpecialDac);
  if(localDEBUG)
    printf("Special Random testDAC mode (on = 0x1, off=0x0):%x\n",SpecialDac);


      //These bits set the number of Out of consecutive out of sync events until a TTs OOs 
  fscanf(infile,"Number of Consecutive (max 1023) Out of Syncs till TTs OOS set:%d\n",
         &Ooslvl);
  if(localDEBUG)
    printf("Number of Consecutive (max 1023) Out of Syncs till TTs OOS set:%d\n",Ooslvl);

      //These bits set the number of Empty events until a TTs Error 
  fscanf(infile,"Number of Consecutive (max 1023) Empty events till TTs ERR set:%d\n",
         &Errlvl);
  if(localDEBUG)
    printf("Number of Consecutive (max 1023) Empty events till TTs ERR set:%d\n",Errlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 N
  fscanf(infile,"N Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         &Nfifo1Bzlvl);
  if(localDEBUG)
    printf("N Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",Nfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 NC
  fscanf(infile,"NC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         &NCfifo1Bzlvl);
  if(localDEBUG)
    printf("NC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",NCfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 SC
  fscanf(infile,"SC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         &SCfifo1Bzlvl);
  if(localDEBUG)
    printf("SC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",SCfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 S
  fscanf(infile,"S Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         &Sfifo1Bzlvl);
  if(localDEBUG)
    printf("S Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",Sfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-3, Almost full = TTs WARN in fifo-3
  fscanf(infile,"Fifo-3 almost full level,sets TTs WARN (max 8191):%d\n",
         &fifo3Wrnlvl);
  if(localDEBUG)
    printf("Fifo-3 almost full level,sets TTs WARN (max 8191):%d\n",fifo3Wrnlvl);




   
  fclose(infile);

  Ncntrl_original=Ncntrl;
  NCcntrl_original=NCcntrl;
  SCcntrl_original=SCcntrl;
  Scntrl_original=Scntrl;

  Nbaseln_original=Nbaseln;
  NCbaseln_original=NCbaseln;
  SCbaseln_original=SCbaseln;
  Sbaseln_original=Sbaseln;

  return;
}

//==================================================================================
// Added by Dario (March 26th 2008)
void PixelFEDCard::clear(void) 
{
  FEDBASE_0 = 0 ;
  fedNumber = 0 ;
  for(int i=0;i<36;i++){
    NRocs[i]    = 0;
    offs_dac[i] = 0;
    BlackHi[i]  = 0;
    BlackLo[i]  = 0;
    Ublack[i]   = 0;
    DelayCh[i]  = 0;  
    TBM_L0[i]   = 0;               
    TBM_L1[i]   = 0;               
    TBM_L2[i]   = 0;               
    TBM_L3[i]   = 0;               
    TBM_L4[i]   = 0;
    TRL_L0[i]   = 0;
    TRL_L1[i]   = 0;
    TRL_L2[i]   = 0;
    TRL_L3[i]   = 0;
    TRL_L4[i]   = 0;
  }             
  for(int i=0;i<3;i++){
    opt_cap[i]   = 0;
    opt_inadj[i] = 0;
    opt_ouadj[i] = 0;
  }
  clkphs1_9   = 0;
  clkphs10_18 = 0;
  clkphs19_27 = 0;
  clkphs28_36 = 0;
  
  for(int i=0;i<36;i++) {
    for(int j=0;j<26;j++) {
      ROC_L0[i][j] = 0;
      ROC_L1[i][j] = 0;
      ROC_L2[i][j] = 0;
      ROC_L3[i][j] = 0;
      ROC_L4[i][j] = 0;
    }
  }
  Ncntrl       = 0;  
  NCcntrl      = 0;  
  SCcntrl      = 0;  
  Scntrl       = 0;  
  CoarseDel    = 0;
  ClkDes2      = 0;
  FineDes2Del  = 0;
  Ccntrl       = 0;
  modeRegister = 0;
  Nadcg        = 0;
  NCadcg       = 0;
  SCadcg       = 0;
  Sadcg        = 0;
  Nbaseln      = 0;
  NCbaseln     = 0;
  SCbaseln     = 0;
  Sbaseln      = 0;
  N_TBMmask    = 0;
  NC_TBMmask   = 0;
  SC_TBMmask   = 0;
  S_TBMmask    = 0;
  N_Pword      = 0;
  NC_Pword     = 0;
  SC_Pword     = 0;
  S_Pword      = 0;
  SpecialDac   = 0;
  Ooslvl       = 0;
  Errlvl       = 0;
  Nfifo1Bzlvl  = 0;
  NCfifo1Bzlvl = 0;
  SCfifo1Bzlvl = 0;
  Sfifo1Bzlvl  = 0;
  fifo3Wrnlvl  = 0;
}
//==================================================================================

void PixelFEDCard::writeASCII(std::string dir) const{

  //  cout << "PixelFEDCard::writeASCII"<<endl;

  ostringstream s1;
  s1<<fedNumber;
  std::string fedNum=s1.str();

  if (dir!="") dir+="/";

  std::string filename=dir+"params_fed_"+fedNum+".dat";

  FILE *outfile = fopen((filename.c_str()),"w");
  if (outfile == NULL) {
    cout<<"Could not open file:"<<filename<<" for writing"<<endl; 
    return;
  }
  
  //Fed Base Address
  fprintf(outfile,"FED Base address                         :0x%lx\n",
         FEDBASE_0);
  fprintf(outfile,"FEDID Number                             :0x%lx\n",
         fedNumber);

  // Number of ROCs
  int ijx=0;
  for(int i=0;i<36;i++){
  ijx=i+1;
    fprintf(outfile,"Number of ROCs Chnl %d:%d\n",ijx,NRocs[i]);
}

  //Settable optical input parameters
  fprintf(outfile,"Optical reciever 1  Capacitor Adjust(0-3):%d\n",opt_cap[0]);
  fprintf(outfile,"Optical reciever 2  Capacitor Adjust(0-3):%d\n",opt_cap[1]);
  fprintf(outfile,"Optical reciever 3  Capacitor Adjust(0-3):%d\n",opt_cap[2]);
  fprintf(outfile,"Optical reciever 1  Input Offset (0-15)  :%d\n",opt_inadj[0]);
  fprintf(outfile,"Optical reciever 2  Input Offset (0-15)  :%d\n",opt_inadj[1]);
  fprintf(outfile,"Optical reciever 3  Input Offset (0-15)  :%d\n",opt_inadj[2]);
  fprintf(outfile,"Optical reciever 1 Output Offset (0-3)   :%d\n",opt_ouadj[0]);
  fprintf(outfile,"Optical reciever 2 Output Offset (0-3)   :%d\n",opt_ouadj[1]);
  fprintf(outfile,"Optical reciever 3 Output Offset (0-3)   :%d\n",opt_ouadj[2]);
  
  //input offset dac
  for(int i=0;i<36;i++) {
    fprintf(outfile,"Offset DAC channel %d:%d\n",i+1,offs_dac[i]);
  }
  
  //clock phases
  fprintf(outfile,"Clock Phase Bits ch   1-9:0x%x\n",clkphs1_9 );
  fprintf(outfile,"Clock Phase Bits ch 10-18:0x%x\n",clkphs10_18);
  fprintf(outfile,"Clock Phase Bits ch 19-27:0x%x\n",clkphs19_27);
  fprintf(outfile,"Clock Phase Bits ch 28-36:0x%x\n",clkphs28_36);
  
  //Blacks 
  for(int i=0;i<36;i++){
    fprintf(outfile,"Black HiThold ch %d:%d \n",i+1,BlackHi[i]);
    fprintf(outfile,"Black LoThold ch %d:%d \n",i+1,BlackLo[i]);
    fprintf(outfile,"ULblack Thold ch %d:%d \n",i+1,Ublack[i]);
  }
  
  //Channel delays
  for(int i=0;i<36;i++) {
    fprintf(outfile,"Delay channel %d(0-15):%d\n",i+1,DelayCh[i]);
  }
  
  //Signal levels
  for(int i=0;i<36;i++) {
    fprintf(outfile,"TBM level 0 Channel  %d:%d\n",i+1,TBM_L0[i]);
    fprintf(outfile,"TBM level 1 Channel  %d:%d\n",i+1,TBM_L1[i]);
    fprintf(outfile,"TBM level 2 Channel  %d:%d\n",i+1,TBM_L2[i]);
    fprintf(outfile,"TBM level 3 Channel  %d:%d\n",i+1,TBM_L3[i]);
    fprintf(outfile,"TBM level 4 Channel  %d:%d\n",i+1,TBM_L4[i]);
    
    for(int j=0;j<NRocs[i];j++) {
      fprintf(outfile,"ROC%d level 0 Channel  %d :%d\n",
             j,i+1,ROC_L0[i][j]);
      fprintf(outfile,"ROC%d level 1 Channel  %d :%d\n",
             j,i+1,ROC_L1[i][j]);
      fprintf(outfile,"ROC%d level 2 Channel  %d :%d\n",
             j,i+1,ROC_L2[i][j]);
      fprintf(outfile,"ROC%d level 3 Channel  %d :%d\n",
             j,i+1,ROC_L3[i][j]);
      fprintf(outfile,"ROC%d level 4 Channel  %d :%d\n",
             j,i+1,ROC_L4[i][j]);
    }
      
    fprintf(outfile,"TRLR level 0 Channel %d:%d\n",i+1,TRL_L0[i]);
    fprintf(outfile,"TRLR level 1 Channel %d:%d\n",i+1,TRL_L1[i]);
    fprintf(outfile,"TRLR level 2 Channel %d:%d\n",i+1,TRL_L2[i]);
    fprintf(outfile,"TRLR level 3 Channel %d:%d\n",i+1,TRL_L3[i]);
    fprintf(outfile,"TRLR level 4 Channel %d:%d\n",i+1,TRL_L4[i]);
  }
  
  
  //These bits turn off(1) and on(0) channels
  fprintf(outfile,"Channel Enbable bits chnls 1-9  (on = 0):0x%x\n",
         Ncntrl);
  fprintf(outfile,"Channel Enbable bits chnls 10-18(on = 0):0x%x\n",
         NCcntrl);
  fprintf(outfile,"Channel Enbable bits chnls 19-27(on = 0):0x%x\n",
         SCcntrl);
  fprintf(outfile,"Channel Enbable bits chnls 28-36(on = 0):0x%x\n",
         Scntrl);
  
  //These are delays to the TTCrx
  fprintf(outfile,"TTCrx Coarse Delay Register 2:%d\n",CoarseDel);
  fprintf(outfile,"TTCrc      ClkDes2 Register 3:0x%x\n",ClkDes2);
  fprintf(outfile,"TTCrc Fine Dlay ClkDes2 Reg 1:%d\n",FineDes2Del);
  
  // Control register
  fprintf(outfile,"Center Chip Control Reg:0x%x\n",Ccntrl);
  fprintf(outfile,"Initial Slink DAQ mode:%d\n",modeRegister);
  
   //These bits set ADC Gain/Range 1Vpp(0) and 2Vpp(1) for channels
  fprintf(outfile,"Channel ADC Gain bits chnls  1-12(1Vpp = 0):0x%x\n",
         Nadcg);
  fprintf(outfile,"Channel ADC Gain bits chnls 13-20(1Vpp = 0):0x%x\n",
         NCadcg);
  fprintf(outfile,"Channel ADC Gain bits chnls 21-28(1Vpp = 0):0x%x\n",
         SCadcg);
  fprintf(outfile,"Channel ADC Gain bits chnls 29-36(1Vpp = 0):0x%x\n",
         Sadcg);
   
       //These bits set Baseline adjustment value (common by FPGA)//can turn on by channel 
  fprintf(outfile,"Channel Baseline Enbable chnls 1-9  (on = (0x1ff<<16)+):0x%x\n",
         Nbaseln);
  fprintf(outfile,"Channel Baseline Enbable chnls 10-18(on = (0x1ff<<16)+):0x%x\n",
         NCbaseln);
  fprintf(outfile,"Channel Baseline Enbable chnls 19-27(on = (0x1ff<<16)+):0x%x\n",
         SCbaseln);
  fprintf(outfile,"Channel Baseline Enbable chnls 28-36(on = (0x1ff<<16)+):0x%x\n",
         Sbaseln);

       //These bits set TBM trailer mask (common by FPGA) 
  fprintf(outfile,"TBM trailer mask chnls 1-9  (0xff = all masked):0x%x\n",
         N_TBMmask);
  fprintf(outfile,"TBM trailer mask chnls 10-18(0xff = all masked):0x%x\n",
         NC_TBMmask);
  fprintf(outfile,"TBM trailer mask chnls 19-27(0xff = all masked):0x%x\n",
         SC_TBMmask);
  fprintf(outfile,"TBM trailer mask chnls 28-36(0xff = all masked):0x%x\n",
         S_TBMmask);

       //These bits set the Private fill/gap word value (common by FPGA) 
  fprintf(outfile,"Private 8 bit word chnls 1-9  :0x%x\n",
         N_Pword);
  fprintf(outfile,"Private 8 bit word chnls 10-18:0x%x\n",
         NC_Pword);
  fprintf(outfile,"Private 8 bit word chnls 19-27:0x%x\n",
         SC_Pword);
  fprintf(outfile,"Private 8 bit word chnls 28-36:0x%x\n",
         S_Pword);

       //These bit sets the special dac mode for random triggers 
  fprintf(outfile,"Special Random testDAC mode (on = 0x1, off=0x0):0x%x\n",
         SpecialDac);

      //These bits set the number of Out of consecutive out of sync events until a TTs OOs 
  fprintf(outfile,"Number of Consecutive (max 1023) Out of Syncs till TTs OOS set:%d\n",
         Ooslvl);
 
      //These bits set the number of Empty events until a TTs Error 
  fprintf(outfile,"Number of Consecutive (max 1023) Empty events till TTs ERR set:%d\n",
         Errlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 N
  fprintf(outfile,"N Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         Nfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 NC
  fprintf(outfile,"NC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         NCfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 SC
  fprintf(outfile,"SC Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         SCfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-1, Almost full = TTs BUSY in fifo-1 S
  fprintf(outfile,"S Fifo-1 almost full level,sets TTs BUSY (max 1023):%d\n",
         Sfifo1Bzlvl);

      //These bits set the Almost Full level in fifo-3, Almost full = TTs WARN in fifo-3
  fprintf(outfile,"Fifo-3 almost full level,sets TTs WARN (max 8191):%d\n",
         fifo3Wrnlvl);

  fclose(outfile);


}

unsigned long long PixelFEDCard::enabledChannels() {
  unsigned long long channels=0;
// return a 64-bit word with low 36 bits set if a channel is enabled
// if bits are set in the control registers, transfer of data from 
// fifo1 to fifo 2 is not done, meaning the channel is disabled.
  channels  = (Ncntrl  & 0x1ffLL);  // Add LL for SLC4, d.k. 12/07
  channels += (NCcntrl & 0x1ffLL) << 9;
  channels += (SCcntrl & 0x1ffLL) << 18;
  channels += (Scntrl  & 0x1ffLL) << 27;
  return ~channels;  //bitwise complement to get enabled channels
}

bool PixelFEDCard::useChannel(unsigned int iChannel){
  assert(iChannel>0&&iChannel<37);
  return (enabledChannels()>>(iChannel-1))&0x1LL;
} 

void PixelFEDCard::setChannel(unsigned int iChannel, bool mode){
  assert(iChannel>0&&iChannel<37);
  long long mask=enabledChannels();
  long long bit=0x1LL<<(iChannel-1);
  if (mode) {
    mask=mask|bit;
  }
  else{
    bit=~bit;
    mask=mask&bit;
  }
  mask=~mask;
  Ncntrl=mask & 0x1ffLL;
  mask=mask>>9;
  NCcntrl=mask & 0x1ffLL;
  mask=mask>>9;
  SCcntrl=mask & 0x1ffLL;
  mask=mask>>9;
  Scntrl=mask & 0x1ffLL;
 
}  

void PixelFEDCard::restoreBaselinAndChannelMasks(){

  Ncntrl=Ncntrl_original;
  NCcntrl=NCcntrl_original;
  SCcntrl=SCcntrl_original;
  Scntrl=Scntrl_original;

  Nbaseln=Nbaseln_original;
  NCbaseln=NCbaseln_original;
  SCbaseln=SCbaseln_original;
  Sbaseln=Sbaseln_original;


}


/* Emacs specific customization
   ;;; Local Variables:     ***
   ;;; indent-tabs-mode:nil ***
   ;;; c-set-style:gnu      ***
   ;;; End:                 ***
*/
