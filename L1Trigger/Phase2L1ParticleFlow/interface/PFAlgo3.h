#ifndef L1Trigger_Phase2L1ParticleFlow_PFAlgo3_h
#define L1Trigger_Phase2L1ParticleFlow_PFAlgo3_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"

namespace l1tpf_impl {
  class PFAlgo3 : public PFAlgoBase {
  public:
    PFAlgo3(const edm::ParameterSet &);
    void runPF(Region &r) const override;

  protected:
    float drMatchMu_;
    enum class MuMatchMode { BoxBestByPtRatio, DrBestByPtRatio, DrBestByPtDiff } muMatchMode_;
    float drMatch_, ptMatchLow_, ptMatchHigh_, maxInvisiblePt_;
    bool useTrackCaloSigma_, rescaleUnmatchedTrack_, caloTrkWeightedAverage_;
    enum class TkCaloLinkMetric { BestByDR = 0, BestByDRPt = 1, BestByDR2Pt2 = 2 };
    float drMatchEm_, ptMinFracMatchEm_, drMatchEmHad_;
    float emHadSubtractionPtSlope_;
    TkCaloLinkMetric tkCaloLinkMetric_;
    bool caloReLinkStep_;
    float caloReLinkDr_, caloReLinkThreshold_;
    bool rescaleTracks_, sumTkCaloErr2_, ecalPriority_, trackEmUseAlsoTrackSigma_, trackEmMayUseCaloMomenta_,
        emCaloUseAlsoCaloSigma_;
    float tightTrackMaxInvisiblePt_;
    enum GoodTrackStatus { GoodTK_Calo_TkPt = 0, GoodTK_Calo_TkCaloPt = 1, GoodTk_Calo_CaloPt = 2, GoodTK_NoCalo = 3 };
    enum BadTrackStatus { BadTK_NoCalo = 1 };
    bool sortInputs_;

    /// do muon track linking (also sets track.muonLink)
    void link_tk2mu(Region &r, std::vector<int> &tk2mu, std::vector<int> &mu2tk) const;

    /// match all tracks to the closest EM cluster
    //  tk2em[itrack] = iem, or -1 if unmatched
    void link_tk2em(Region &r, std::vector<int> &tk2em) const;

    /// match all em to the closest had (can happen in parallel to the above)
    //  em2calo[iem] = icalo or -1
    void link_em2calo(Region &r, std::vector<int> &em2calo) const;

    /// for each EM cluster, count and add up the pt of all the corresponding tracks (skipping muons)
    void sum_tk2em(Region &r,
                   const std::vector<int> &tk2em,
                   std::vector<int> &em2ntk,
                   std::vector<float> &em2sumtkpt,
                   std::vector<float> &em2sumtkpterr) const;

    /// process ecal clusters after linking
    void emcalo_algo(Region &r,
                     const std::vector<int> &em2ntk,
                     const std::vector<float> &em2sumtkpt,
                     const std::vector<float> &em2sumtkpterr) const;

    /// promote all flagged tracks to electrons
    void emtk_algo(Region &r,
                   const std::vector<int> &tk2em,
                   const std::vector<int> &em2ntk,
                   const std::vector<float> &em2sumtkpterr) const;

    /// subtract EM component from Calo clusters for all photons and electrons (within tracker coverage)
    void sub_em2calo(Region &r, const std::vector<int> &em2calo) const;

    /// track to calo matching
    //  tk2calo[itk] = icalo or -1
    void link_tk2calo(Region &r, std::vector<int> &tk2calo) const;

    /// for each calo, compute the sum of the track pt
    void sum_tk2calo(Region &r,
                     const std::vector<int> &tk2calo,
                     std::vector<int> &calo2ntk,
                     std::vector<float> &calo2sumtkpt,
                     std::vector<float> &calo2sumtkpterr) const;

    /// promote unlinked low pt tracks to hadrons
    void unlinkedtk_algo(Region &r, const std::vector<int> &tk2calo) const;

    /// try to recover split hadron showers (v1.0):
    //  take hadrons that are not track matched, close by a hadron which has an excess of track pt vs calo pt
    //  add this pt to the calo pt of the other cluster
    //  off by default, as it seems to not do much in jets even if it helps remove tails in single-pion events
    void calo_relink(Region &r,
                     const std::vector<int> &calo2ntk,
                     const std::vector<float> &calo2sumtkpt,
                     const std::vector<float> &calo2sumtkpterr) const;

    /// process matched calo clusters, compare energy to sum track pt, compute track rescaling factor if needed
    //  alpha[icalo] = x < 1 if all tracks linked to icalo must have their pt rescaled by x
    void linkedcalo_algo(Region &r,
                         const std::vector<int> &calo2ntk,
                         const std::vector<float> &calo2sumtkpt,
                         const std::vector<float> &calo2sumtkpterr,
                         std::vector<float> &calo2alpha) const;

    /// process matched tracks, if necessary rescale or average
    void linkedtk_algo(Region &r,
                       const std::vector<int> &tk2calo,
                       const std::vector<int> &calo2ntk,
                       const std::vector<float> &calo2alpha) const;

    /// process unmatched calo clusters
    void unlinkedcalo_algo(Region &r) const;

    /// save muons in output list
    void save_muons(Region &r, const std::vector<int> &tk2mu) const;
  };

}  // namespace l1tpf_impl

#endif
