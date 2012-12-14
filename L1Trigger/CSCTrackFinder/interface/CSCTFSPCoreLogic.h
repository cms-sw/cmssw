#ifndef CSCTrackFinder_CSCTFSPCoreLogic_h
#define CSCTrackFinder_CSCTFSPCoreLogic_h

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>

// different cores
#include <L1Trigger/CSCTrackFinder/src/core_2010_01_22/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2010_07_28/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2010_09_01/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2010_10_11/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2010_12_10/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2011_01_18/vpp_generated.h>
#include <L1Trigger/CSCTrackFinder/src/core_2012_01_31/vpp_generated.h> 
#include <L1Trigger/CSCTrackFinder/src/core_2012_03_13/vpp_generated.h> 
#include <L1Trigger/CSCTrackFinder/src/core_2012_07_30/vpp_generated.h> 

class vpp_generated_2010_01_22;
class vpp_generated_2010_07_28;
class vpp_generated_2010_09_01;
class vpp_generated_2010_10_11;
class vpp_generated_2010_12_10;
class vpp_generated_2011_01_18;
class vpp_generated_2012_01_31;
class vpp_generated_2012_03_13;
class vpp_generated_2012_07_30;

class CSCTFSPCoreLogic
{
   /**change input and output to Signal   */
    struct SPio {

      unsigned me1aVp; unsigned me1aQp; unsigned me1aEtap; unsigned me1aPhip; unsigned me1aAmp; unsigned me1aCSCIdp; unsigned me1aCLCTp;
      unsigned me1bVp; unsigned me1bQp; unsigned me1bEtap; unsigned me1bPhip; unsigned me1bAmp; unsigned me1bCSCIdp; unsigned me1bCLCTp;
      unsigned me1cVp; unsigned me1cQp; unsigned me1cEtap; unsigned me1cPhip; unsigned me1cAmp; unsigned me1cCSCIdp; unsigned me1cCLCTp;
      unsigned me1dVp; unsigned me1dQp; unsigned me1dEtap; unsigned me1dPhip; unsigned me1dAmp; unsigned me1dCSCIdp; unsigned me1dCLCTp;
      unsigned me1eVp; unsigned me1eQp; unsigned me1eEtap; unsigned me1ePhip; unsigned me1eAmp; unsigned me1eCSCIdp; unsigned me1eCLCTp;
      unsigned me1fVp; unsigned me1fQp; unsigned me1fEtap; unsigned me1fPhip; unsigned me1fAmp; unsigned me1fCSCIdp; unsigned me1fCLCTp;

      unsigned me2aVp; unsigned me2aQp; unsigned me2aEtap; unsigned me2aPhip;	unsigned me2aAmp;
      unsigned me2bVp; unsigned me2bQp; unsigned me2bEtap; unsigned me2bPhip;	unsigned me2bAmp;
      unsigned me2cVp; unsigned me2cQp; unsigned me2cEtap; unsigned me2cPhip;	unsigned me2cAmp;

      unsigned me3aVp; unsigned me3aQp; unsigned me3aEtap; unsigned me3aPhip;	unsigned me3aAmp;
      unsigned me3bVp; unsigned me3bQp; unsigned me3bEtap; unsigned me3bPhip;	unsigned me3bAmp;
      unsigned me3cVp; unsigned me3cQp; unsigned me3cEtap; unsigned me3cPhip;	unsigned me3cAmp;

      unsigned me4aVp; unsigned me4aQp; unsigned me4aEtap; unsigned me4aPhip;	unsigned me4aAmp;
      unsigned me4bVp; unsigned me4bQp; unsigned me4bEtap; unsigned me4bPhip;	unsigned me4bAmp;
      unsigned me4cVp; unsigned me4cQp; unsigned me4cEtap; unsigned me4cPhip;	unsigned me4cAmp;

      unsigned mb1aVp; unsigned mb1aQp; unsigned mb1aPhip; unsigned mb1aBendp;
      unsigned mb1bVp; unsigned mb1bQp; unsigned mb1bPhip; unsigned mb1bBendp;
      unsigned mb1cVp; unsigned mb1cQp; unsigned mb1cPhip; unsigned mb1cBendp;
      unsigned mb1dVp; unsigned mb1dQp; unsigned mb1dPhip; unsigned mb1dBendp;

      unsigned mb2aVp; unsigned mb2aQp; unsigned mb2aPhip;
      unsigned mb2bVp; unsigned mb2bQp; unsigned mb2bPhip;
      unsigned mb2cVp; unsigned mb2cQp; unsigned mb2cPhip;
      unsigned mb2dVp; unsigned mb2dQp; unsigned mb2dPhip;

      unsigned ptHp; unsigned signHp; unsigned modeMemHp; unsigned etaPTHp; unsigned FRHp; unsigned phiHp; unsigned phdiff_aHp;  unsigned phdiff_bHp; 
      unsigned ptMp; unsigned signMp; unsigned modeMemMp; unsigned etaPTMp; unsigned FRMp; unsigned phiMp; unsigned phdiff_aMp;  unsigned phdiff_bMp; 
      unsigned ptLp; unsigned signLp; unsigned modeMemLp; unsigned etaPTLp; unsigned FRLp; unsigned phiLp; unsigned phdiff_aLp;  unsigned phdiff_bLp; 

      unsigned me1idH; unsigned me2idH; unsigned me3idH; unsigned me4idH; unsigned mb1idH; unsigned mb2idH;
      unsigned me1idM; unsigned me2idM; unsigned me3idM; unsigned me4idM; unsigned mb1idM; unsigned mb2idM;
      unsigned me1idL; unsigned me2idL; unsigned me3idL; unsigned me4idL; unsigned mb1idL; unsigned mb2idL;
    };

 public:

    CSCTFSPCoreLogic() : runme(false),
      spFirmwareVersion(0), coreFirmwareVersion(0),
      verboseCore(false){}

  void loadData(const CSCTriggerContainer<csctf::TrackStub>&,
		const unsigned& endcap, const unsigned& sector,
		const int& minBX, const int& maxBX);

  bool run(const unsigned& endcap, const unsigned& sector, const unsigned& latency,
	   const unsigned& etamin1, const unsigned& etamin2, const unsigned& etamin3, const unsigned& etamin4,
	   const unsigned& etamin5, const unsigned& etamin6, const unsigned& etamin7, const unsigned& etamin8,
	   const unsigned& etamax1, const unsigned& etamax2, const unsigned& etamax3, const unsigned& etamax4,
	   const unsigned& etamax5, const unsigned& etamax6, const unsigned& etamax7, const unsigned& etamax8,
	   const unsigned& etawin1, const unsigned& etawin2, const unsigned& etawin3,
	   const unsigned& etawin4, const unsigned& etawin5, const unsigned& etawin6, const unsigned& etawin7,
	   const unsigned& mindphip, const unsigned& mindetap,
	   const unsigned& mindeta12_accp,
	   const unsigned& maxdeta12_accp, const unsigned& maxdphi12_accp,
	   const unsigned& mindeta13_accp,
	   const unsigned& maxdeta13_accp, const unsigned& maxdphi13_accp,
	   const unsigned& mindeta112_accp,
	   const unsigned& maxdeta112_accp, const unsigned& maxdphi112_accp,
	   const unsigned& mindeta113_accp,
	   const unsigned& maxdeta113_accp, const unsigned& maxdphi113_accp,
	   const unsigned& mindphip_halo, const unsigned& mindetap_halo,
	   const unsigned& straightp, const unsigned& curvedp,
	   const unsigned& mbaPhiOff, const unsigned& mbbPhiOff,
	   const unsigned& m_extend_length,
	   const unsigned& m_allowALCTonly, const unsigned& m_allowCLCTonly,
	   const unsigned& m_preTrigger, const unsigned& m_widePhi,
	   const int& minBX, const int& maxBX);

  CSCTriggerContainer<csc::L1Track> tracks();
  
  void SetSPFirmwareVersion(const unsigned int fwVer) {spFirmwareVersion=fwVer; }
  unsigned int GetSPFirmwareVersion() {return spFirmwareVersion; }

  void SetCoreFirmwareVersion(const unsigned int fwVer) {coreFirmwareVersion=fwVer; }
  unsigned int GetCoreFirmwareVersion() {return coreFirmwareVersion; }

  void SetVerbose(const bool verb) { verboseCore=verb; }
  bool IsVerbose() { return verboseCore; }
  void setNLBTables();
  
 private:
  static vpp_generated_2010_01_22 sp_2010_01_22_;
  static vpp_generated_2010_07_28 sp_2010_07_28_;
  static vpp_generated_2010_09_01 sp_2010_09_01_;
  static vpp_generated_2010_10_11 sp_2010_10_11_;
  static vpp_generated_2010_12_10 sp_2010_12_10_;
  static vpp_generated_2011_01_18 sp_2011_01_18_;
  static vpp_generated_2012_01_31 sp_2012_01_31_;
  static vpp_generated_2012_03_13 sp_2012_03_13_;
  static vpp_generated_2012_07_30 sp_2012_07_30_;
   
  std::vector<SPio> io_;
  bool runme;
  CSCTriggerContainer<csc::L1Track> mytracks;
  unsigned int spFirmwareVersion;
  unsigned int coreFirmwareVersion;
  bool verboseCore;
};

#endif
