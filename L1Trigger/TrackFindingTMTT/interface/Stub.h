#ifndef __STUB_H__
#define __STUB_H__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalStub.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <vector>
#include <set>
#include <array>
#include <map>

using namespace std;

class TrackerGeometry;
class TrackerTopology;
// class DetId;

namespace TMTT {

class TP;

// typedef edm::Ref< edm::DetSetVector< Phase2TrackerDigi >, Phase2TrackerDigi > Ref_Phase2TrackerDigi_;
typedef edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> > DetSetVec;
typedef edmNew::DetSet< TTStub<Ref_Phase2TrackerDigi_> >       DetSet;
typedef edm::Ref<DetSetVec, TTStub<Ref_Phase2TrackerDigi_> >   TTStubRef;
typedef edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > TTClusterRef;
typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_>           TTStubAssMap;
typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_>        TTClusterAssMap;

//=== Represents a Tracker stub (=pair of hits)

class Stub : public TTStubRef {

public:

  // Store useful info about the stub (for use with HYBRID code), with hard-wired constants to allow use outside CMSSW.
  Stub(double phi, double r, double z, double bend, int layerid, bool psModule, bool barrel, unsigned int iphi, double alpha, const Settings* settings, const TrackerTopology* trackerTopology, unsigned int ID, unsigned int iPhiSec);

  // Store useful info about stub (for use with TMTT code).
  Stub(const TTStubRef& ttStubRef, unsigned int index_in_vStubs, const Settings* settings, const TrackerGeometry*  trackerGeometry, const TrackerTopology*  trackerTopology);

  ~Stub(){}

  bool operator==(const Stub& stubOther) {return (this->index() == stubOther.index());}

  // Fill truth info with association from stub to tracking particles.
  // The 1st argument is a map relating TrackingParticles to TP.
  void fillTruth(const map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, const edm::Handle<TTStubAssMap>& mcTruthTTStubHandle, const edm::Handle<TTClusterAssMap>& mcTruthTTClusterHandle);

  // Calculate bin range along q/Pt axis of r-phi Hough transform array consistent with bend of this stub.
  void calcQoverPtrange();

  // Digitize stub for input to Geographic Processor, with digitized phi coord. measured relative to closest phi sector.
  // (This approximation is valid if their are an integer number of digitisation bins inside each phi nonant).
  // However, you should also call digitizeForHTinput() before accessing digitized stub data, even if you only care about that going into GP! Otherwise, you will not identify stubs assigned to more than one nonant.
  void digitizeForGPinput(unsigned int iPhiSec);

  // Digitize stub for input to Hough transform, with digitized phi coord. measured relative to specified phi sector.
  void digitizeForHTinput(unsigned int iPhiSec);

  // Digitize stub for input to r-z Seed Filter.
  // (Kept for backwards compatibility.)
  void digitizeForSFinput() {this->digitizeForSForTFinput("SeedFilter");}

  // Digitize stub for input to r-z Seed Filter or Track Fitter.
  // Argument is "SeedFilter" or name of Track Fitter.
  void digitizeForSForTFinput(string SForTF);

  // Digitize stub for input to Duplicate Removal .
  void digitizeForDRinput(unsigned int stubId);

  // Control warning messages about accessing non-digitized quantities.
  void setDigitizeWarningsOn(bool newVal) {digitizeWarningsOn_ = newVal;}

  // Restore stub to pre-digitized state. i.e. Undo what function digitize() did.

  void reset_digitize();

  // === Functions for returning info about reconstructed stubs ===

  // Location in InputData::vStubs_
  unsigned int                         index() const { return index_in_vStubs_; }

  //--- Stub data and quantities derived from it ---

  // Stub coordinates (optionally after digitisation, if digitisation requested via cfg).
  // N.B. Digitisation is not run when the stubs are created, but later, after stubs are assigned to sectors.
  // Until then, these functions return the original coordinates. 
  float                                  phi() const { return             phi_; }
  float                                    r() const { return               r_; }
  float                                    z() const { return               z_; }
  float                                theta() const { return    atan2(r_, z_); }
  float                                  eta() const { return     asinh(z_/r_); }
  // Access to digitized version of stub coords.
  const DigitalStub&             digitalStub() const { return      digitalStub_;}
  // Access to booleans indicating if the stub has been digitized.
  bool                   digitizedForGPinput() const { return digitizedForGPinput_;}
  bool                   digitizedForHTinput() const { return digitizedForHTinput_;}
  string             digitizedForSForTFinput() const { return digitizedForSForTFinput_;} // Returns which SF or TF digitisation was done for, if any.
  bool                   digitizedForDRinput() const { return digitizedForDRinput_;}

  // Get stub bend (i.e. displacement between two hits in stub in units of strip pitch) and its estimated resolution.
  float                                 bend() const { this->check1(); return bend_; } 
  // The bend resolution has a contribution from the sensor and a contribution from encoding the bend into
  // a reduced number of bits.
  float                              bendRes() const { return (settings_->bendResolution() + (numMergedBend_-1)*settings_->bendResolutionExtra()); }
  // Number of bend values which loss of bit to store bend resulted in being merged into this bend value.
  float                        numMergedBend() const { return numMergedBend_;}
  // Bend angle of track measured by stub and its estimated resolution.
  float                                 dphi() const { this->check2(); return (bend_ * dphiOverBend()); }
  float                              dphiRes() const { return (dphiOverBend() * this->bendRes()); }
  // Estimated track q/Pt based on stub bend info.
  float                              qOverPt() const { return (this->qOverPtOverBend() * this->bend()); }
  float                           qOverPtres() const { return (this->qOverPtOverBend() * this->bendRes()); }
  // Range in q/Pt bins in HT array compatible with stub bend.
  unsigned int               min_qOverPt_bin() const { return min_qOverPt_bin_; }
  unsigned int               max_qOverPt_bin() const { return max_qOverPt_bin_; }
  // Estimated phi0 of track at beam-line based on stub bend info.
  float                                 beta() const { return   (phi_ + dphi()); }
  // Estimated phi angle at which track intercepts a given radius rad, based on stub bend info. Also estimate uncertainty on this angle due to endcap 2S module strip length. 
  // This is identical to beta() if rad=0.
  pair <float, float>     trkPhiAtR(float rad) const;
  // Estimated resolution in trkPhiAtR(rad) based on nominal stub bend resolution.
  float                trkPhiAtRres(float rad) const { return this->dphiRes() * fabs(1 - rad / r_); }
  // Difference in phi between stub and angle at which track crosses given radius, assuming track has given Pt.
  float           phiDiff(float rad, float Pt) const { return fabs(r_ - rad)*(settings_->invPtToDphi())/Pt; }
  // -- conversion factors
  // Ratio of bend angle to bend, where bend is the displacement in strips between the two hits making up stub.
  float                         dphiOverBend() const { this->check2(); return dphiOverBend_; }
  // Correction factor that was used when calculating dPhiOverBend, due to tilt of module.
  float                         dphiOverBendCorrection() const { return dphiOverBendCorrection_; }
  // Approximation of dphiOverBendCorrection, used in firmware.
  float                         dphiOverBendCorrectionApprox() const { return dphiOverBendCorrection_approx_; }
  // Ratio of q/Pt to bend, where bend is the displacement in strips between the two hits making up stub.
  float                      qOverPtOverBend() const { return this->dphiOverBend() / (r_ * settings_->invPtToDphi()); }

  //--- Info about the two clusters that make up the stub.
  // Coordinates in frame of sensor, measured in units of strip pitch along two orthogonal axes running perpendicular and parallel to longer axis of pixels/strips (U & V).
  array<float, 2>             localU_cluster() const { return localU_cluster_;}
  array<float, 2>             localV_cluster() const { return localV_cluster_;}

  //--- Check if this stub will be output by front-end readout electronics,
  //--- (where we can reconfigure the stub window size and rapidity cut).
  //--- Don't use stubs failing this cut.
  bool                          frontendPass() const { return    frontendPass_; }
  // Indicates if stub would have passed front-end cuts, were it not for window size encoded in DegradeBend.h
  bool              stubFailedDegradeWindow() const { return    stubFailedDegradeWindow_;}

  //--- Quantities common to all stubs in a given module ---

  // Unique identifier for each stacked module, allowing one to check which stubs are on the same module.
  unsigned int                         idDet() const { return     idDet_.rawId();}
  // Uncertainty in stub coordinates due to strip length, assumed equal to 0.5*strip-or-pixel-length 
  float                                 rErr() const { return            rErr_;}
  float                                 zErr() const { return            zErr_;}
  // Coordinates of centre of two sensors in (r,phi,z)
  float                                 minR() const { return      moduleMinR_; }
  float                                 maxR() const { return      moduleMaxR_; }
  float                               minPhi() const { return    moduleMinPhi_; }
  float                               maxPhi() const { return    moduleMaxPhi_; }
  float                                 minZ() const { return      moduleMinZ_; }
  float                                 maxZ() const { return      moduleMaxZ_; }
  // Angle between normal to module and beam-line along +ve z axis. (In range -PI/2 to +PI/2).
  float                           moduleTilt() const { return      moduleTilt_; }
  // Which of two sensors in module is furthest from beam-line?
  bool                 outerModuleAtSmallerR() const { return   outerModuleAtSmallerR_; }
  // Sensor pitch over separation.
  float                         pitchOverSep() const { return    pitchOverSep_;}
  // Location of stub in module in units of strip number (or pixel number along finest granularity axis).
  // Range from 0 to (nStrips - 1) inclusive.
  unsigned int                          iphi() const { return            iphi_; }
  // alpha correction for non-radial strips in endcap 2S modules.
  // (If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR).
  // *** TO DO *** : Digitize this.
  float                                alpha() const { return           alpha_; }
  // Module type: PS or 2S?
  bool                              psModule() const { return        psModule_; }
  // Tracker layer ID number (1-6 = barrel layer; 11-15 = endcap A disk; 21-25 = endcap B disk)
  unsigned int                       layerId() const { return         layerId_; }
  // Reduced layer ID (in range 1-7). This encodes the layer ID in only 3 bits (to simplify firmware) by merging some barrel layer and endcap disk layer IDs into a single ID.
  unsigned int                layerIdReduced() const;
  // Endcap ring of module (returns zero in case of barrel)
  unsigned int                    endcapRing() const { return      endcapRing_; }
  bool                                barrel() const { return          barrel_; }
  // True if stub is in tilted barrel module.
  bool                          tiltedBarrel() const { return    tiltedBarrel_; }
  // Strip pitch (or pixel pitch along shortest axis).
  float                           stripPitch() const { return      stripPitch_; } 
  // Strip length (or pixel pitch along longest axis).
  float                          stripLength() const { return     stripLength_; } 
  // No. of strips in sensor.
  unsigned int                       nStrips() const { return         nStrips_; }
  // Width of sensitive region of sensor.
  float                          sensorWidth() const { return     sensorWidth_; }
  // Hit resolution perpendicular to strip (or to longest pixel axis) = pitch/sqrt(12). Measures phi.
  float                            sigmaPerp() const { return       sigmaPerp_; }
  // Hit resolution parallel to strip (or to longest pixel axis) = length/sqrt(12). Measures r or z.
  float                             sigmaPar() const { return        sigmaPar_; }

  // Clone a few of the above functions with the less helpful names expected by the track fitting code. (Try to phase these out with time ...)
  unsigned int                        nstrip() const { return this->nStrips(); }
  float                                width() const { return this->sensorWidth(); }
  float                               sigmaX() const { return this->sigmaPerp(); }
  float                               sigmaZ() const { return this->sigmaPar(); }

  //--- Truth info

  // Association of stub to tracking particles
  const set<const TP*>&             assocTPs() const { return        assocTPs_; } // Return TPs associated to this stub. (Whether only TPs contributing to both clusters are returned is determined by "StubMatchStrict" config param.)
  bool             genuine() const { return (assocTPs_.size() > 0); } // Did stub match at least one TP?
  const TP*                          assocTP() const { return         assocTP_; } // If only one TP contributed to both clusters, this tells you which TP it is. Returns nullptr if none.

  // Association of both clusters making up stub to tracking particles
  array<bool, 2>        genuineCluster() const { return array<bool, 2>{ {(assocTPofCluster_[0] != nullptr), (assocTPofCluster_[1] != nullptr)} }; } // Was cluster produced by a single TP?
  array<const TP*, 2>       assocTPofCluster() const { return       assocTPofCluster_; } // Which TP made each cluster. Warning: If cluster was not produced by a single TP, then returns nullptr! (P.S. If both clusters match same TP, then this will equal assocTP()).

  // Note if stub is a crazy distance from the tracking particle trajectory that produced it. (e.g. perhaps produced by delta ray)
  bool                             crazyStub() const;

  // Get stub bend and its resolution, as available within the front end chip (i.e. prior to loss of bits
  // or digitisation).
  float                       bendInFrontend() const { return bendInFrontend_; } 
  float                    bendResInFrontend() const { return settings_->bendResolution(); } 

  // Return tracker geometry (T3, T4, T5 ...)
  // See https://github.com/cms-sw/cmssw/blob/CMSSW_9_1_X/Configuration/Geometry/README.md
  string              trackerGeometryVersion() const { return trackerGeometryVersion_;}

private:

  // Degrade assumed stub bend resolution.
  // Also return boolean indicating if stub bend was outside assumed window, so stub should be rejected
  // and return an integer indicating how many values of bend are merged into this single one.
  void degradeResolution(float bend, 
			 float& degradedBend, bool& reject, unsigned int& num) const;

  // Set the frontendPass_ flag, indicating if frontend readout electronics will output this stub.  
  // Argument indicates if stub bend was outside window size encoded in DegradeBend.h
  void setFrontend(bool rejectStub);          

  // Set info about the module that this stub is in.
  void setModuleInfo(const TrackerGeometry* trackerGeometry, const TrackerTopology* trackerTopology, const DetId& detId);

  // Determine tracker geometry version by counting modules.
  void setTrackerGeometryVersion(const TrackerGeometry* trackerGeometry, const TrackerTopology* trackerTopology);

  // Function to calculate approximation for dphiOverBendCorrection aka B
  double getApproxB();

  // No HT firmware can access directly the stub bend info.
  void check1() const {if (digitizeWarningsOn_ && digitizedForHTinput_) throw cms::Exception("Stub: You can't access digitized bend variable within HT firmware!");}
  // If using daisy-chain firmware, then it makes no sense to access the digiitzed values of dphi within HT.
 void check2() const {if (digitizeWarningsOn_ && digitizedForHTinput_) throw cms::Exception("Stub: You can't access digitized dphi within the HT or KF!");}

private:

  const Settings* settings_; // configuration parameters.

  unsigned int                     index_in_vStubs_; // location of this stub in InputData::vStubs

  //--- Parameters passed along optical links from PP to MP (or equivalent ones if easier for analysis software to use).
  // WARNING: If you add any variables in this section, take care to ensure that they are digitized correctly by Stub::digitize().
  float                                        phi_; // stub coords, optionally after digitisation.
  float                                          r_;
  float                                          z_;
  float                                       bend_; // bend of stub.
  float                               dphiOverBend_; // related to rho parameter.
  float                               dphiOverBendCorrection_; // Correction from tilt of module
  float                               dphiOverBendCorrection_approx_; // Correction from tilt of module
  unsigned int                     min_qOverPt_bin_; // Range in q/Pt bins in HT array compatible with stub bend.
  unsigned int                     max_qOverPt_bin_; 

  //--- Info about the two clusters that make up the stub.
  array<float, 2>                   localU_cluster_;
  array<float, 2>                   localV_cluster_;

  //--- Parameters common to all stubs in a given module.
  DetId                                      idDet_; 
  float                                       rErr_;
  float                                       zErr_;
  float                                 moduleMinR_;
  float                                 moduleMaxR_;
  float                               moduleMinPhi_;
  float                               moduleMaxPhi_;
  float                                 moduleMinZ_;
  float                                 moduleMaxZ_;
  float                                 moduleTilt_;
  float                               pitchOverSep_;
  unsigned int                                iphi_;
  float                                      alpha_;
  bool                                    psModule_;
  unsigned int                             layerId_;
  unsigned int                          endcapRing_;
  bool                                      barrel_;
  bool                                tiltedBarrel_;
  float                                  sigmaPerp_;
  float                                   sigmaPar_;
  float                                 stripPitch_;
  float                                stripLength_;
  unsigned int                             nStrips_;
  float                                sensorWidth_;
  bool                       outerModuleAtSmallerR_;
  //--- Truth info about stub.
  const TP*                                assocTP_;
  set<const TP*>                          assocTPs_;
  //--- Truth info about the two clusters that make up the stub
  array<const TP*, 2>             assocTPofCluster_;

  // Would front-end electronics output this stub?
  bool                                frontendPass_;
  // Did stub fail window cuts assumed in DegradeBend.h?
  bool                     stubFailedDegradeWindow_;
  // Bend in front end chip (prior to degredation by loss of bits & digitization).
  float                             bendInFrontend_;
  // Used for stub bend resolution degrading.
  unsigned int                       numMergedBend_;

  DigitalStub                          digitalStub_; // Class used to digitize stub if required.
  bool                         digitizedForGPinput_; // Has this stub been digitized for GP input?
  bool                         digitizedForHTinput_; // Has this stub been digitized for HT input?
  string                   digitizedForSForTFinput_; // Has this stub been digitized for seed filter or track fitter input? If so, this was its name.
  bool                         digitizedForDRinput_; // Has this stub been digitized for seed filter input?                    
  bool                         digitizeWarningsOn_;  // Enable warnings about accessing non-digitized quantities.

  // Which tracker geometry is this?
  static thread_local string trackerGeometryVersion_;

  // Used to provide TMTT recommendations for stub window sizes that CMS should use.
  StubWindowSuggest              stubWindowSuggest_;

  // Used to degrade stub bend information.
  DegradeBend degradeBend_;

  //--- Utility to emulate dead modules.
  static bool stubKillerInit_;
  static StubKiller stubKiller_; 
};

}
#endif
