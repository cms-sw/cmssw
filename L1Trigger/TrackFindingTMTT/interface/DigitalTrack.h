#ifndef __DIGITALTRACK_H__
#define __DIGITALTRACK_H__

#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include <string>
#include <set>

using namespace std;

namespace TMTT {

class Settings;

//====================================================================================================
/**
* Used to digitize the fitted track helix params.
* WARNING: Digitizes according to common format agreed for KF and SimpleLR fitters, 
* and uses KF digitisation cfg for all fitters except SimpleLR.
*/
//====================================================================================================

class DigitalTrack {

public:

  // Note configuration parameters.
  DigitalTrack(const Settings* settings);
  // Dummy constructor
  DigitalTrack() {}
  // DigitalTrack();

  ~DigitalTrack(){}

  /// Initialize track with original, floating point coords
  void init(const string& fitterName, unsigned int nHelixParams,
	    unsigned int iPhiSec, unsigned int iEtaReg, int mbin, int cbin, int mBinhelix, int cBinhelix, 
	    unsigned int hitPattern,
	    float qOverPt_orig, float d0_orig, float phi0_orig, float tanLambda_orig, float z0_orig, float chisquaredRphi_orig, float chisquaredRz_orig, 
	    float qOverPt_bcon_orig, float phi0_bcon_orig, float chisquaredRphi_bcon_orig, // beam-spot constrained values. 
	    unsigned int nLayers, bool consistent, bool consistentSect, bool accepted, 
	    float tp_qOverPt, float tp_d0, float tp_phi0, float tp_tanLambda, float tp_z0, float tp_eta, 
	    int tp_index, bool tp_useForAlgEff, bool tp_useForEff, int tp_pdgId);

  // Digitize track
  void makeDigitalTrack();

  //--- The functions below return variables post-digitization.
  //--- Do not call any of the functions below, unless you have already called init() and make()!

  // Digits corresponding to track params.
  int          iDigi_oneOver2r()          const {this->ok(); return iDigi_oneOver2r_;} // half inverse curvature of track.
  int          iDigi_d0()                 const {this->ok(); return iDigi_d0_;} 
  int          iDigi_phi0rel()            const {this->ok(); return iDigi_phi0rel_;} // measured relative to centre of sector
  int          iDigi_z0()                 const {this->ok(); return iDigi_z0_;}
  int          iDigi_tanLambda()          const {this->ok(); return iDigi_tanLambda_;}
  unsigned int iDigi_chisquaredRphi()     const {this->ok(); return iDigi_chisquaredRphi_;}
  unsigned int iDigi_chisquaredRz()       const {this->ok(); return iDigi_chisquaredRz_;}

  // Digits corresponding to track params with post-fit beam-spot constraint.
  int          iDigi_oneOver2r_bcon()     const {this->ok(); return iDigi_oneOver2r_bcon_;} // half inverse curvature of track.
  int          iDigi_phi0rel_bcon()       const {this->ok(); return iDigi_phi0rel_bcon_;} // measured relative to centre of sector
  unsigned int iDigi_chisquaredRphi_bcon() const {this->ok(); return iDigi_chisquaredRphi_bcon_;}

  // Floating point track params derived from digitized info (so with degraded resolution).
  float        qOverPt()                  const {this->ok(); return qOverPt_;} 
  float        oneOver2r()                const {this->ok(); return oneOver2r_;} // half inverse curvature of track.
  float        d0()                       const {this->ok(); return d0_;}
  float        phi0()                     const {this->ok(); return phi0_;}
  float        phi0rel()                  const {this->ok(); return phi0rel_;} // measured relative to centre of sector
  float        z0()                       const {this->ok(); return z0_;}
  float        tanLambda()                const {this->ok(); return tanLambda_;}
  float        chisquaredRphi()           const {this->ok(); return chisquaredRphi_;}
  float        chisquaredRz()             const {this->ok(); return chisquaredRz_;}

  // Floating point track params derived from digitized track params with post-fit beam-spot constraint.
  float        qOverPt_bcon()             const {this->ok(); return qOverPt_bcon_;} 
  float        oneOver2r_bcon()           const {this->ok(); return oneOver2r_bcon_;} // half inverse curvature of track.
  float        phi0_bcon()                const {this->ok(); return phi0_bcon_;}
  float        phi0rel_bcon()             const {this->ok(); return phi0rel_bcon_;} // measured relative to centre of sector
  float        chisquaredRphi_bcon()      const {this->ok(); return chisquaredRphi_bcon_;}

  unsigned int iPhiSec()                  const {this->okin(); return iPhiSec_;}
  unsigned int iEtaReg()                  const {this->okin(); return iEtaReg_;}
  int          mBinhelix()                const {this->okin(); return mBinhelix_;}
  int          cBinhelix()                const {this->okin(); return cBinhelix_;}
  unsigned int nlayers()                  const {this->okin(); return nlayers_;}
  int          mBinHT()                   const {this->okin(); return mBin_;}
  int          cBinHT()                   const {this->okin(); return cBin_;}
  bool         accepted()                 const {this->okin(); return accepted_;}
  unsigned int hitPattern()               const {this->okin(); return hitPattern_;}

  //--- The functions below give access to the original variables prior to digitization.
  //%%% Those common to GP & HT input.
  float        orig_qOverPt()             const {this->okin(); return qOverPt_orig_;} 
  float        orig_oneOver2r()           const {this->okin(); return oneOver2r_orig_;} // half inverse curvature of track.
  float        orig_d0()                  const {this->okin(); return d0_orig_;}
  float        orig_phi0()                const {this->okin(); return phi0_orig_;}
  float        orig_phi0rel()             const {this->okin(); return phi0rel_orig_;} // measured relative to centre of sector
  float        orig_z0()                  const {this->okin(); return z0_orig_;}
  float        orig_tanLambda()           const {this->okin(); return tanLambda_orig_;}
  float        orig_chisquaredRphi()      const {this->okin(); return chisquaredRphi_orig_;}
  float        orig_chisquaredRz()        const {this->okin(); return chisquaredRz_orig_;}

  float        tp_pt()                    const {this->okin(); return tp_pt_;}
  float        tp_eta()                   const {this->okin(); return tp_eta_;}
  float        tp_d0()                    const {this->okin(); return tp_d0_;}
  float        tp_phi0()                  const {this->okin(); return tp_phi0_;}
  float        tp_tanLambda()             const {this->okin(); return tp_tanLambda_;}
  float        tp_z0()                    const {this->okin(); return tp_z0_;}
  float        tp_qoverpt()               const {this->okin(); return tp_qoverpt_;}
  int          tp_index()                 const {this->okin(); return tp_index_;}
  float        tp_useForAlgEff()          const {this->okin(); return tp_useForAlgEff_;}
  float        tp_useForEff()             const {this->okin(); return tp_useForEff_;}
  float        tp_pdgId()                 const {this->okin(); return tp_pdgId_;}

  //--- Utility: return phi nonant number corresponding to given phi sector number.
  unsigned int iGetNonant(unsigned int iPhiSec) const {return floor(iPhiSec*numPhiNonants_/numPhiSectors_);}

  bool         available()     const {return ranMake_;}

private:

  // Check DigitalTrack correctly initialized;
  void         okin()          const {if (! ranInit_) throw cms::Exception("DigitalTrack: You forgot to call init()!");}
  void         ok()            const {if (! ranMake_) throw cms::Exception("DigitalTrack: You forgot to call makeDigitalTrack()!");}

  // Get digitisation configuration parameters for the specific track fitter being used here.
  void         getDigiCfg(const string& fitterName);

  // Check that stub coords. are within assumed digitization range.
  void         checkInRange()  const;

  // Check that digitisation followed by undigitisation doesn't change significantly the track params.
  void         checkAccuracy() const;

private:

  // Check DigitalTrack correctly initialized.
  bool            ranInit_;
  bool            ranMake_;

  // Configuration params
  const Settings* settings_;

  string          fitterName_; 
  unsigned int    nHelixParams_;

  // Integer data after digitization (which doesn't degrade its resolution, but can recast it in a different form).
  unsigned int        nlayers_;
  unsigned int        iPhiSec_;
  unsigned int        iEtaReg_;
  int                 mBinhelix_;
  int                 cBinhelix_;
  bool                consistent_;
  bool                consistentSect_;
  int                 mBin_;
  int                 cBin_;
  bool                accepted_;
  
  float               tp_qoverpt_;
  float               tp_pt_;
  float               tp_eta_;
  float               tp_d0_;
  float               tp_phi0_;
  float               tp_tanLambda_;
  float               tp_z0_;
  int                 tp_index_;
  bool                tp_useForAlgEff_;
  bool                tp_useForEff_;
  int                 tp_pdgId_;

  // Digitization configuration
  bool                 skipTrackDigi_;
  unsigned int         oneOver2rBits_;
  float                oneOver2rRange_;
  unsigned int         d0Bits_;
  float                d0Range_;
  unsigned int         phi0Bits_;
  float                phi0Range_;
  unsigned int         z0Bits_;
  float                z0Range_;
  unsigned int         tanLambdaBits_;
  float                tanLambdaRange_;
  unsigned int         chisquaredBits_;
  float                chisquaredRange_;

  double               oneOver2rMult_;
  double               d0Mult_;
  double               phi0Mult_;
  double               z0Mult_;
  double               tanLambdaMult_;
  double               chisquaredMult_;

  // Number of phi sectors and phi nonants.
  unsigned int         numPhiSectors_;
  unsigned int         numPhiNonants_;   
  double               phiSectorWidth_;
  double               phiNonantWidth_;
  double               phiSectorCentre_;
  float                chosenRofPhi_;
  unsigned int         nbinsPt_;
  float                invPtToDPhi_;

  //--- Original floating point stub coords before digitization.

  unsigned int         hitPattern_;

  float                qOverPt_orig_;
  float                oneOver2r_orig_;
  float                d0_orig_;
  float                phi0_orig_;
  float                phi0rel_orig_;
  float                tanLambda_orig_;
  float                z0_orig_;
  float                chisquaredRphi_orig_;
  float                chisquaredRz_orig_;

  float                qOverPt_bcon_orig_;
  float                oneOver2r_bcon_orig_;
  float                phi0_bcon_orig_;
  float                phi0rel_bcon_orig_;
  float                chisquaredRphi_bcon_orig_;

  //--- Digits corresponding to track params.

  int                  iDigi_oneOver2r_;
  int                  iDigi_d0_;
  int                  iDigi_phi0rel_;
  int                  iDigi_z0_;
  int                  iDigi_tanLambda_;
  unsigned int         iDigi_chisquaredRphi_;
  unsigned int         iDigi_chisquaredRz_;

  int                  iDigi_oneOver2r_bcon_;
  int                  iDigi_phi0rel_bcon_;
  unsigned int         iDigi_chisquaredRphi_bcon_;

  //--- Floating point track coords derived from digitized info (so with degraded resolution). 

  float                qOverPt_;
  float                oneOver2r_;
  float                d0_;
  float                phi0_;
  float                phi0rel_;
  float                z0_;
  float                tanLambda_;
  float                chisquaredRphi_;
  float                chisquaredRz_;

  float                qOverPt_bcon_;
  float                oneOver2r_bcon_;
  float                phi0_bcon_;
  float                phi0rel_bcon_;
  float                chisquaredRphi_bcon_;
};

}
#endif

