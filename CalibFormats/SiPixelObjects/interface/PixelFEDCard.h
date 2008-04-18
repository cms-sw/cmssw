#ifndef TP_PIXELFEDCARD_H
#define TP_PIXELFEDCARD_H
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelFEDCard.h
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

#include <string>

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelFEDCard PixelFEDCard.h
*  \brief This is the documentation about PixelFEDCard...
*
*  The structure which holds all the informations needed to setup 
*  a pixel FED. Danek Kotlinski 18/4/06
*/
  class PixelFEDCard : public PixelConfigBase{

  public:
    
    //Return true or false depending on if iChannel is used
    //iChannel=1..36
    bool useChannel(unsigned int iChannel);

    //Set iChannel enable to mode
    //iChannel=1..36
    void setChannel(unsigned int iChannel, bool mode);

    void restoreBaselinAndChannelMasks();

    // Constructor and destructor
    PixelFEDCard(); // empty
    PixelFEDCard(std::string filename); // create from files
    ~PixelFEDCard() {};

    void writeASCII(std::string dir="") const; // write to files
    unsigned long long enabledChannels();  // returns 64-bit integer mask 35..0


    //Settable optical input parameters (one for each 12-receiver)
    int opt_cap[3];   // Capacitor adjust
    int opt_inadj[3]; // DC-input offset
    int opt_ouadj[3]; // DC-output offset
  
    //input offset dac (one for each channel)
    int offs_dac[36];
  
    //clock phases, use bits 0-8, select the clock edge
    int clkphs1_9,clkphs10_18,clkphs19_27,clkphs28_36;

    //Channel delays, one for each channel, 0=15
    int DelayCh[36];
  
    //Blacks and Ultra-blacks, 3 limit per channel
    int BlackHi[36];
    int BlackLo[36];
    int Ublack[36];
    
    //Signal levels for the TBM, one per channel
    int TBM_L0[36],TBM_L1[36],TBM_L2[36],TBM_L3[36],TBM_L4[36];
    int TRL_L0[36],TRL_L1[36],TRL_L2[36],TRL_L3[36],TRL_L4[36];
    // Address levels 1 per channel (36) per roc(max=26)
    int ROC_L0[36][26],ROC_L1[36][26],ROC_L2[36][26],ROC_L3[36][26],
      ROC_L4[36][26];

    //These bits turn off(1) and on(0) channels
    int Ncntrl,NCcntrl,SCcntrl,Scntrl;

    //The values as read from file so that they can be restored after
    //calibration
    int Ncntrl_original,NCcntrl_original,SCcntrl_original,Scntrl_original;

     //Bits (1st 8) used to mask TBM trailer bits
    int N_TBMmask,NC_TBMmask,SC_TBMmask,S_TBMmask;
    
    //Bits (1st 8) used to set the Private Word in the gap and filler words
    int N_Pword,NC_Pword,SC_Pword,S_Pword;

    // 1 = Special Random trigger DAC mode on, 0=off
    int SpecialDac;
 
    // Control register and delays for the TTCrx
    int CoarseDel,ClkDes2,FineDes2Del;
 
    //Main control reg for determining the DAQ mode
    int Ccntrl; // "CtrlReg" in LAD_C
 
    //Mode register
    int modeRegister; // "ModeReg" in LAD_C
  
    //Number of ROCS per FED channel
    int NRocs[36];

    //Control Regs for setting ADC 1Vpp and 2Vpp
    int Nadcg,NCadcg,SCadcg,Sadcg;

    //Control and data Regs for setting Baseline Adjustment
    int Nbaseln,NCbaseln,SCbaseln,Sbaseln;

    //data Regs for TTs adjustable levels
    int Ooslvl,Errlvl;

    //data Regs adjustable fifo Almost Full levels
    int Nfifo1Bzlvl,NCfifo1Bzlvl,SCfifo1Bzlvl,Sfifo1Bzlvl,fifo3Wrnlvl;

    //The values as read from file so that they can be restored after
    //calibration
    int Nbaseln_original,NCbaseln_original,SCbaseln_original,
        Sbaseln_original;


    //VME base address 
    unsigned long FEDBASE_0, fedNumber;

 private: 
 
    // Added by Dario (March 26th 2008)
    void clear(void) ;

  }; // end class PixelFEDCard
}
/* @} */
#endif // ifdef include
