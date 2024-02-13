/*
Kalman Filter L1 Muon algorithm
Tyler Lam (UCLA)
Sep. 2021
*/
#ifndef L1Trigger_Phase2L1GMT_KMTFCore_h
#define L1Trigger_Phase2L1GMT_KMTFCore_h
#include "L1Trigger/Phase2L1GMT/interface/KMTFLUTs.h"
#include "DataFormats/L1TMuonPhase2/interface/KMTFTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cstdlib>
#include "ap_fixed.h"

namespace Phase2L1GMT {

  class KMTFCore {
  public:
    typedef ROOT::Math::SVector<double, 2> Vector2;
    typedef ROOT::Math::SMatrix<double, 2, 2, ROOT::Math::MatRepSym<double, 2> > CovarianceMatrix2;
    typedef ROOT::Math::SMatrix<double, 3, 2> Matrix32;
    typedef ROOT::Math::SMatrix<double, 2, 3> Matrix23;
    typedef ROOT::Math::SMatrix<double, 1, 3> Matrix13;
    typedef ROOT::Math::SMatrix<double, 3, 1> Matrix31;
    typedef ROOT::Math::SMatrix<double, 3, 3> Matrix33;

    KMTFCore(const edm::ParameterSet& settings);

    std::pair<l1t::KMTFTrack, l1t::KMTFTrack> chain(const l1t::MuonStubRef& seed, const l1t::MuonStubRefVector& stubs);

    std::vector<l1t::KMTFTrack> clean(const std::vector<l1t::KMTFTrack>& tracks, uint seed, bool vertex);

  private:
    KMTFLUTs* lutService_;

    std::pair<bool, uint> match(const l1t::MuonStubRef& seed, const l1t::MuonStubRefVector& stubs, int step);

    int correctedPhiB(const l1t::MuonStubRef& stub);
    void propagate(l1t::KMTFTrack& track);
    bool update(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub, int mask, int seedQual);
    bool updateOffline(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub);
    bool updateOffline1D(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub);
    bool updateLUT(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub, int mask, int seedQual);
    void vertexConstraint(l1t::KMTFTrack& track);
    void vertexConstraintOffline(l1t::KMTFTrack& track);
    void vertexConstraintLUT(l1t::KMTFTrack& track);
    int hitPattern(const l1t::KMTFTrack& track);
    int customBitmask(unsigned int bit1, unsigned int bit2, unsigned int bit3, unsigned int bit4);
    bool getBit(int bitmask, int pos);
    void setFourVectors(l1t::KMTFTrack& track);
    bool estimateChiSquare(l1t::KMTFTrack& track, bool vertex);
    void setRank(l1t::KMTFTrack& track, bool vertex);
    int wrapAround(int value, int maximum);
    int encode(bool ownwheel, int sector, int tag);
    std::pair<bool, uint> getByCode(const std::vector<l1t::KMTFTrack>& tracks, int mask);
    uint twosCompToBits(int q);
    uint etaStubRank(const l1t::MuonStubRef& stub);
    void calculateEta(l1t::KMTFTrack& track);
    uint matchAbs(std::map<uint, uint>& info, uint i, uint j);
    int ptLUT(int K);

    bool verbose_;

    //Initial Curvature
    std::vector<double> initK_;
    std::vector<double> initK2_;

    //propagation coefficients
    std::vector<double> eLoss_;
    std::vector<double> aPhi_;
    std::vector<double> aPhiB_;
    std::vector<double> aPhiBNLO_;
    std::vector<double> bPhi_;
    std::vector<double> bPhiB_;
    double phiAt2_;
    std::vector<double> etaLUT0_;
    std::vector<double> etaLUT1_;

    //Chi Square estimator input
    uint globalChi2Cut_;
    uint globalChi2CutLimit_;

    std::vector<double> chiSquareDisp1_;
    std::vector<double> chiSquareDisp2_;
    std::vector<double> chiSquareDisp3_;
    std::vector<int> chiSquareErrADisp1_;
    std::vector<int> chiSquareErrADisp2_;
    std::vector<int> chiSquareErrADisp3_;
    std::vector<double> chiSquareErrBDisp1_;
    std::vector<double> chiSquareErrBDisp2_;
    std::vector<double> chiSquareErrBDisp3_;

    std::vector<double> chiSquarePrompt1_;
    std::vector<double> chiSquarePrompt2_;
    std::vector<double> chiSquarePrompt3_;
    std::vector<int> chiSquareErrAPrompt1_;
    std::vector<int> chiSquareErrAPrompt2_;
    std::vector<int> chiSquareErrAPrompt3_;
    std::vector<double> chiSquareErrBPrompt1_;
    std::vector<double> chiSquareErrBPrompt2_;
    std::vector<double> chiSquareErrBPrompt3_;

    std::vector<int> chiSquareCutDispPattern_;
    std::vector<int> chiSquareCutOffDisp_;
    std::vector<int> chiSquareCutDisp_;
    std::vector<int> chiSquareCutPromptPattern_;
    std::vector<int> chiSquareCutOffPrompt_;
    std::vector<int> chiSquareCutPrompt_;

    //bitmasks to run== diferent combinations for a given seed in a given station
    std::vector<int> combos4_;
    std::vector<int> combos3_;
    std::vector<int> combos2_;
    std::vector<int> combos1_;

    //bits for fixed point precision
    static const int PHIBSCALE = 16;
    static const int PHIBSCALE_INT = 5;
    static const int BITSCURV = 16;
    static const int BITSPHI = 18;
    static const int BITSPHIB = 17;  // 12 bits *28 (+5 bits)
    static const int BITSPARAM = 14;
    static const int GAIN_0 = 9;
    static const int GAIN_0INT = 6;
    static const int GAIN_4 = 9;
    static const int GAIN_4INT = 4;
    static const int GAIN_V0 = 9;
    static const int GAIN_V0INT = 0;

    static const int GAIN2_0 = 12;
    static const int GAIN2_0INT = 6;
    static const int GAIN2_1 = 12;
    static const int GAIN2_1INT = 3;
    static const int GAIN2_4 = 12;
    static const int GAIN2_4INT = 4;
    static const int GAIN2_5 = 12;
    static const int GAIN2_5INT = 0;
    //STUFF NOT USED IN THE FIRMWARE BUT ONLY FOR DEBUGGING
    ///////////////////////////////////////////////////////

    bool useOfflineAlgo_;
    std::vector<double> mScatteringPhi_;
    std::vector<double> mScatteringPhiB_;
    //point resolution for phi
    double pointResolutionPhi_;
    //point resolution for phiB
    double pointResolutionPhiB_;
    std::vector<double> pointResolutionPhiBH_;
    std::vector<double> pointResolutionPhiBL_;
    //double pointResolutionPhiB_;
    //point resolution for vertex
    double pointResolutionVertex_;
    std::vector<double> curvResolution1_;
    std::vector<double> curvResolution2_;
    //Sorter
    class StubSorter {
    public:
      StubSorter(uint sector) { sec_ = sector; }

      bool operator()(const l1t::MuonStubRef& a, const l1t::MuonStubRef& b) {
        if (a->coord1() < b->coord1())
          return true;
        return false;
      }

    private:
      int sec_;
    };
  };

}  // namespace Phase2L1GMT
#endif
