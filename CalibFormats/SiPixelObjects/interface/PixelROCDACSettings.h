#ifndef PixelROCDACSettings_h
#define PixelROCDACSettings_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h
*   \brief This class provide the data structure for the ROC DAC parameters
*
*   At this point I do not see a reason to make an abstract layer for this code.
*/

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigKey.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

namespace pos{
  typedef unsigned char bits8;
  typedef unsigned char bits4;

/*! \class PixelROCDACSettings PixelROCDACSettings.h "interface/PixelROCDACSettings.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelROCDACSettings{

  public:

    PixelROCDACSettings();
    virtual ~PixelROCDACSettings(){;};

    PixelROCDACSettings(const PixelROCName& rocid){rocid_= rocid;}

    //Build the commands needed to configure ROC
    //Need to use the mask bits also for this
    std::string getConfigCommand();

    int read(std::ifstream& in, const PixelROCName& rocid);
    int read(std::istringstream& in, const PixelROCName& rocid);

    int readBinary(std::ifstream& in, const PixelROCName& rocid);

    void writeBinary(std::ofstream& out) const;

    void writeASCII(std::ostream& out) const;
    void 	 writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, int version, std::string path, std::ofstream *out) const {;}
    virtual void writeXML(                                                              std::ofstream *out) const  ;
    virtual void writeXMLTrailer(                                                       std::ofstream *out) const {;}

    void getDACs(std::vector<unsigned int>& dacs) const;
    void getDACs(std::map<std::string, unsigned int>& dacs) const;


    void setDAC(unsigned int dacaddress, unsigned int dacvalue);
    void setDACs(std::map<std::string, unsigned int>& dacs) ;

    void compareDACs(std::map<std::string, unsigned int> & dacs, 
                     std::map<std::string, bool>         & changes,
		     std::map<std::string, unsigned int> & previous) ;

    void checkTag(std::string tag, 
		  std::string dacName,
		  const PixelROCName& rocid);
      
    void setDac(std::string dacName, int value);

    unsigned int getDac(std::string dacName) const;

    bits4 getVdd() {return Vdd_;}
    void setVdd(bits4 vdd) {Vdd_=vdd;}

    bits8 getVana() {return Vana_;}
    void setVana(bits8 vana) {Vana_=vana;}

    bits8 getVsf() {return Vsf_;}
    void setVsf(bits8 vsf) {Vsf_=vsf;}

    bits4 getVcomp() {return Vcomp_;}
    void setVcomp(bits4 vcomp) {Vcomp_=vcomp;}

    bits8 getVleak() {return Vleak_;}
    void setVleak(bits8 vleak) {Vleak_=vleak;}

    bits4 getVrgPr() {return VrgPr_;}
    void setVrgPr(bits4 vrgPr) {VrgPr_=vrgPr;}

    bits8 getVwllPr() {return VwllPr_;}
    void setVwllPr_(bits8 vwllPr) {VwllPr_=vwllPr;}

    bits4 getVrgSh() {return VrgSh_;}
    void setVrgSh(bits4 vrgSh) {VrgSh_=vrgSh;}

    bits8 getVwllSh() {return VwllSh_;}
    void setVwllSh(bits8 vwllSh) {VwllSh_=vwllSh;}

    bits8 getVHldDel() {return VHldDel_;}
    void setVHldDel(bits8 vHldDel) {VHldDel_=vHldDel;}

    bits8 getVtrim() {return Vtrim_;}
    void setVtrim(bits8 vtrim) {Vtrim_=vtrim;}

    bits8 getVcThr() {return VcThr_;}
    void setVcThr(bits8 vcThr) {VcThr_=vcThr;}

    bits8 getVIbias_bus() {return VIbias_bus_;}
    void setVIbias_bus(bits8 vIbias_bus) {VIbias_bus_=vIbias_bus;}

    bits4 getVIbias_sf() {return VIbias_sf_;}
    void setVIbias_sf(bits4 vIbias_sf) {VIbias_sf_=vIbias_sf;}

    bits8 getVOffsetOp() {return VOffsetOp_;}
    void setVOffsetOp(bits8 vOffsetOp) {VOffsetOp_=vOffsetOp;}

    bits8 getVbiasOp() {return VbiasOp_;}
    void setVbiasOp(bits8 vbiasOp) {VbiasOp_=vbiasOp;}

    bits8 getVOffsetRO() {return VOffsetRO_;}
    void setVOffsetRO(bits8 vOffsetRO) {VOffsetRO_=vOffsetRO;}

    bits8 getVIon() {return VIon_;}
    void setVIon(bits8 vIon) {VIon_=vIon;}

    bits8 getVIbias_PH() {return VIbias_PH_;}
    void setVIbias_PH(bits8 vIbias_PH) {VIbias_PH_=vIbias_PH;}

    bits8 getVIbias_DAC() {return VIbias_DAC_;}
    void setVIbias_DAC(bits8 vIbias_DAC) {VIbias_DAC_=vIbias_DAC;}

    bits8 getVIbias_roc() {return VIbias_roc_;}
    void setVIbias_roc(bits8 vIbias_roc) {VIbias_roc_=vIbias_roc;}

    bits8 getVIColOr() {return VIColOr_;}
    void setVIColOr(bits8 vIColOr) {VIColOr_=vIColOr;}

    bits8 getVnpix() {return Vnpix_;}
    void setVnpix(bits8 vnpix) {Vnpix_=vnpix;}

    bits8 getVsumCol() {return VsumCol_;}
    void setVsumCol(bits8 vsumCol) {VsumCol_=vsumCol;}

    bits8 getVcal() {return Vcal_;}
    void setVcal(bits8 vcal) {Vcal_=vcal;}

    bits8 getCalDel() {return CalDel_;}
    void setCalDel(bits8 calDel) {CalDel_=calDel;}

    bits8 getTempRange() {return TempRange_;}
    void setTempRange(bits8 tempRange) {TempRange_=tempRange;}

    bits8 getWBC() {return WBC_;}
    void setWBC(bits8 wbc) {WBC_=wbc;}

    bits8 getControlRegister() const {return ChipContReg_;}
    void setControlRegister(bits8 controlregister) {ChipContReg_=controlregister;}

    PixelROCName getROCName() const {return rocid_;} 

    friend std::ostream& operator<<(std::ostream& s, const PixelROCDACSettings& dacs);

  private:

    PixelROCName rocid_;

    //The dac settings used by the ROC

    bits4 Vdd_;              //addr 1
    bits8 Vana_;             //addr 2
    bits8 Vsf_;              //addr 3
    bits4 Vcomp_;            //addr 4
    bits8 Vleak_;            //addr 5
    bits4 VrgPr_;            //addr 6
    bits8 VwllPr_;           //addr 7
    bits4 VrgSh_;            //addr 8
    bits8 VwllSh_;           //addr 9
    bits8 VHldDel_;          //addr 10
    bits8 Vtrim_;            //addr 11
    bits8 VcThr_;            //addr 12
    bits8 VIbias_bus_;       //addr 13
    bits4 VIbias_sf_;        //addr 14
    bits8 VOffsetOp_;        //addr 15
    bits8 VbiasOp_;          //addr 16
    bits8 VOffsetRO_;        //addr 17
    bits8 VIon_;             //addr 18
    bits8 VIbias_PH_;        //addr 19
    bits8 VIbias_DAC_;       //addr 20
    bits8 VIbias_roc_;       //addr 21
    bits8 VIColOr_;          //addr 22
    bits8 Vnpix_;            //addr 23
    bits8 VsumCol_;          //addr 24
    bits8 Vcal_;             //addr 25
    bits8 CalDel_;           //addr 26
    bits8 TempRange_;        //addr 27
    bits8 WBC_;              //addr 254
    bits8 ChipContReg_;      //addr 253
    
    std::string ToLower(std::string) ;
    
  };
}
#endif
