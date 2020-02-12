#ifndef __TP_H__
#define __TP_H__

#include "DataFormats/Math/interface/deltaPhi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include <vector>

using namespace std;

namespace TMTT {

class Stub;

typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;

class TP : public TrackingParticlePtr {

public:
  // Fill useful info about tracking particle.
  TP(const TrackingParticlePtr& tpPtr, unsigned int index_in_vTPs, const Settings* settings);
  ~TP(){}

  bool operator==(const TP& tpOther) {return (this->index() == tpOther.index());}

  // Fill truth info with association from tracking particle to stubs.
  void fillTruth(const vector<Stub>& vStubs);

  // == Functions for returning info about tracking particles ===

  // Location in InputData::vTPs_
  unsigned int                           index() const { return     index_in_vTPs_; }
  // Basic TP properties
  int                                    pdgId() const { return             pdgId_; }
  // Did TP come from in-time or out-of-time bunch crossing?
  bool                                inTimeBx() const { return          inTimeBx_; }
  // Did it come from the main physics collision or from pileup?
  bool                        physicsCollision() const { return  physicsCollision_; }
  int                                   charge() const { return            charge_; }
  float                                   mass() const { return              mass_; }
  float                                     pt() const { return                pt_; }
  float                                qOverPt() const { return  (pt_ > 0)  ?  charge_/pt_  :  9.9e9; }
  float                                    eta() const { return               eta_; }
  float                                  theta() const { return             theta_; }
  float                              tanLambda() const { return         tanLambda_; }
  float                                   phi0() const { return              phi0_; }
  // TP production vertex (x,y,z) coordinates.
  float                                     vx() const { return                vx_; }
  float                                     vy() const { return                vy_; }
  float                                     vz() const { return                vz_; }
  // d0 and z0 impact parameters with respect to (x,y) = (0,0).
  float                                     d0() const { return                d0_;}
  float                                     z0() const { return                z0_;}
  // Estimate track bend angle at a given radius, ignoring scattering.
  float                          dphi(float rad) const { return asin(settings_->invPtToDphi() * rad * charge_/pt_); }
  // Estimated phi angle at which TP trajectory crosses a given radius rad, ignoring scattering.
  float                     trkPhiAtR(float rad) const { return reco::deltaPhi(phi0_ - this->dphi(rad) - d0_/rad,  0.); }
  // Estimated z coord at which TP trajectory crosses a given radius rad, ignoring scattering.
  float                       trkZAtR(float rad) const { return (vz_ + rad * tanLambda_); }
  // Estimated phi angle at which TP trajectory crosses the module containing the given stub.
  float           trkPhiAtStub(const Stub* stub) const;
  // Estimated r coord at which TP trajectory crosses the module containing the given stub.
  float             trkRAtStub(const Stub* stub) const;
  // Estimated z coord at which TP trajectory crosses the module containing the given stub.
  float             trkZAtStub(const Stub* stub) const;

  // == Functions returning stubs produced by tracking particle.
  const vector<const Stub*>&        assocStubs() const { return        assocStubs_; } // associated stubs. (Includes those failing tightened front-end electronics cuts supplied by user). (Which stubs are returned is affected by "StubMatchStrict" config param.)
  unsigned int                   numAssocStubs() const { return assocStubs_.size(); }
  unsigned int                       numLayers() const { return  nLayersWithStubs_; }
  // TP is worth keeping (e.g. for fake rate measurement)
  bool                                     use() const { return               use_; }
  // TP can be used for efficiency measurement (good kinematics, in-time Bx, optionally specified PDG ID).
  bool                               useForEff() const { return         useForEff_; } 
  // TP can be used for algorithmic efficiency measurement (also requires stubs in enough layers).
  bool                            useForAlgEff() const { return      useForAlgEff_; } 

  void                            fillNearestJetInfo( const reco::GenJetCollection* genJets ); // Store info (deltaR, pt) with nearest jet

  float                             tpInJet() const { return tpInJet_ && nearestJetPt_ > 30; }
  float                             tpInHighPtJet() const { return tpInJet_ && nearestJetPt_ > 100; }
  float                             tpInVeryHighPtJet() const { return tpInJet_ && nearestJetPt_ > 200; }

  float                             nearestJetPt() const { return nearestJetPt_; }

private:

  void fillUse();          // Fill the use_ flag.
  void fillUseForEff();    // Fill the useForEff_ flag.
  void fillUseForAlgEff(); // Fill the useforAlgEff_ flag.

  // Calculate how many tracker layers this TP has stubs in.
  void calcNumLayers() { nLayersWithStubs_ = Utility::countLayers( settings_, assocStubs_, false); }

private:

  unsigned int                      index_in_vTPs_; // location of this TP in InputData::vTPs

  const Settings*                        settings_; // Configuration parameters

  int                                       pdgId_;
  bool                                   inTimeBx_; // TP came from in-time bunch crossing.
  bool                           physicsCollision_; // True if TP from physics collision rather than pileup.
  int                                      charge_;
  float                                      mass_;
  float                                        pt_; // TP kinematics
  float                                       eta_;
  float                                     theta_;
  float                                 tanLambda_;
  float                                      phi0_;
  float                                        vx_; // TP production point.
  float                                        vy_;
  float                                        vz_;
  float                                        d0_; // d0 impact parameter with respect to (x,y) = (0,0)
  float                                        z0_; // z0 impact parameter with respect to (x,y) = (0,0)

  bool                                        use_; // TP is worth keeping (e.g. for fake rate measurement)
  bool                                  useForEff_; // TP can be used for tracking efficiency measurement.
  bool                               useForAlgEff_; // TP can be used for tracking algorithmic efficiency measurement.

  vector<const Stub*>                  assocStubs_;
  unsigned int                   nLayersWithStubs_; // Number of tracker layers with stubs from this TP.

  bool                                tpInJet_;
  float                              nearestJetPt_;
};

}

#endif
