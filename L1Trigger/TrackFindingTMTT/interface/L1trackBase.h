#ifndef L1Trigger_TrackFindingTMTT_L1trackBase_h
#define L1Trigger_TrackFindingTMTT_L1trackBase_h

#include <vector>
#include <utility>

//=== L1 track base class
//=== This is a pure virtual class containing no implemented functions or data members.
//=== However, it declares functions that are common to the derived classes L1trackBase, L1track3D and L1fittedTrack,
//=== allowing software to analyse objects of all three types in the same way.

namespace tmtt {

  class Stub;
  class TP;

  class L1trackBase {
  public:
    L1trackBase() {}

    virtual ~L1trackBase() = default;

    //--- Get information about the reconstructed track.

    // Get stubs on track candidate.
    virtual const std::vector<const Stub*>& stubsConst() const = 0;
    virtual const std::vector<Stub*>& stubs() const = 0;
    // Get number of stubs on track candidate.
    virtual unsigned int numStubs() const = 0;
    // Get number of tracker layers these stubs are in.
    virtual unsigned int numLayers() const = 0;

    //--- User-friendly access to the helix parameters.

    virtual float qOverPt() const = 0;
    virtual float phi0() const = 0;
    //virtual float   z0()         const  = 0;
    //virtual float   tanLambda()  const  = 0;

    //--- Cell locations of the track candidate in the r-phi Hough transform array in units of bin number.
    virtual std::pair<unsigned int, unsigned int> cellLocationHT() const = 0;

    //--- Get phi sector and eta region used by track finding code that this track is in.
    virtual unsigned int iPhiSec() const = 0;
    virtual unsigned int iEtaReg() const = 0;

    //--- Opto-link ID used to send this track from HT to Track Fitter
    virtual unsigned int optoLinkID() const = 0;

    //--- Get information about its association (if any) to a truth Tracking Particle.

    // Get matching tracking particle (=nullptr if none).
    virtual const TP* matchedTP() const = 0;
    // Get the matched stubs.
    virtual const std::vector<const Stub*>& matchedStubs() const = 0;
    // Get number of matched stubs.
    virtual unsigned int numMatchedStubs() const = 0;
    // Get number of tracker layers with matched stubs.
    virtual unsigned int numMatchedLayers() const = 0;
  };

}  // namespace tmtt

#endif
