#ifndef CSCTrackFinder_CSCTFSPCoreLogic_h
#define CSCTrackFinder_CSCTFSPCoreLogic_h

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>

class SPvpp;

class CSCTFSPCoreLogic
{
   /**change input and output to Signal   */
    struct SPio {                                       
      
      unsigned me1aVp; unsigned me1aQp; unsigned me1aEtap; unsigned me1aPhip; unsigned me1aAmp; unsigned me1aCSCIdp;
      unsigned me1bVp; unsigned me1bQp; unsigned me1bEtap; unsigned me1bPhip; unsigned me1bAmp; unsigned me1bCSCIdp;
      unsigned me1cVp; unsigned me1cQp; unsigned me1cEtap; unsigned me1cPhip; unsigned me1cAmp; unsigned me1cCSCIdp;
      unsigned me1dVp; unsigned me1dQp; unsigned me1dEtap; unsigned me1dPhip; unsigned me1dAmp; unsigned me1dCSCIdp;
      unsigned me1eVp; unsigned me1eQp; unsigned me1eEtap; unsigned me1ePhip; unsigned me1eAmp; unsigned me1eCSCIdp;
      unsigned me1fVp; unsigned me1fQp; unsigned me1fEtap; unsigned me1fPhip; unsigned me1fAmp; unsigned me1fCSCIdp;
      
      unsigned me2aVp; unsigned me2aQp; unsigned me2aEtap; unsigned me2aPhip;	unsigned me2aAmp;  
      unsigned me2bVp; unsigned me2bQp; unsigned me2bEtap; unsigned me2bPhip;	unsigned me2bAmp;  
      unsigned me2cVp; unsigned me2cQp; unsigned me2cEtap; unsigned me2cPhip;	unsigned me2cAmp;  
      
      unsigned me3aVp; unsigned me3aQp; unsigned me3aEtap; unsigned me3aPhip;	unsigned me3aAmp;  
      unsigned me3bVp; unsigned me3bQp; unsigned me3bEtap; unsigned me3bPhip;	unsigned me3bAmp;  
      unsigned me3cVp; unsigned me3cQp; unsigned me3cEtap; unsigned me3cPhip;	unsigned me3cAmp;  
      
      unsigned me4aVp; unsigned me4aQp; unsigned me4aEtap; unsigned me4aPhip;	unsigned me4aAmp;  
      unsigned me4bVp; unsigned me4bQp; unsigned me4bEtap; unsigned me4bPhip;	unsigned me4bAmp;  
      unsigned me4cVp; unsigned me4cQp; unsigned me4cEtap; unsigned me4cPhip;	unsigned me4cAmp;  
      
      unsigned mb1aVp; unsigned mb1aQp; unsigned mb1aPhip;                                   
      unsigned mb1bVp; unsigned mb1bQp; unsigned mb1bPhip;                                   
      unsigned mb1cVp; unsigned mb1cQp; unsigned mb1cPhip;                                   
      unsigned mb1dVp; unsigned mb1dQp; unsigned mb1dPhip;                                   
      
      unsigned mb2aVp; unsigned mb2aQp; unsigned mb2aPhip;                                   
      unsigned mb2bVp; unsigned mb2bQp; unsigned mb2bPhip;                                   
      unsigned mb2cVp; unsigned mb2cQp; unsigned mb2cPhip;                                   
      unsigned mb2dVp; unsigned mb2dQp; unsigned mb2dPhip;                                   
      
      unsigned ptHp; unsigned signHp; unsigned modeMemHp; unsigned etaPTHp; unsigned FRHp; unsigned phiHp;  
      unsigned ptMp; unsigned signMp; unsigned modeMemMp; unsigned etaPTMp; unsigned FRMp; unsigned phiMp;  
      unsigned ptLp; unsigned signLp; unsigned modeMemLp; unsigned etaPTLp; unsigned FRLp; unsigned phiLp;  
      
      unsigned me1idH; unsigned me2idH; unsigned me3idH; unsigned me4idH; unsigned mb1idH; unsigned mb2idH;
      unsigned me1idM; unsigned me2idM; unsigned me3idM; unsigned me4idM; unsigned mb1idM; unsigned mb2idM;
      unsigned me1idL; unsigned me2idL; unsigned me3idL; unsigned me4idL; unsigned mb1idL; unsigned mb2idL;
    };

 public:

  CSCTFSPCoreLogic() : runme(false) {}

  void loadData(const CSCTriggerContainer<CSCTrackStub>&,
		const unsigned& endcap, const unsigned& sector, 
		const int& minBX, const int& maxBX);

  bool run(const unsigned& endcap, const unsigned& sector, const unsigned& latency, 
	   const unsigned& etawin1, const unsigned& etawin2, const unsigned& etawin3, 
	   const unsigned& etawin4, const unsigned& etawin5, const unsigned& etawin6,
	   const unsigned& bxa_on, const unsigned& extend, const int& minBX, 
	   const int& maxBX);

  CSCTriggerContainer<csc::L1Track> tracks();

 private:
  static SPvpp sp_; 
  std::vector<SPio> io_;
  bool runme;
  CSCTriggerContainer<csc::L1Track> mytracks;
};

#endif
