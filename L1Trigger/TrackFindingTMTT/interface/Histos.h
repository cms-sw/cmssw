#ifndef L1Trigger_TrackFindingTMTT_Histos_h
#define L1Trigger_TrackFindingTMTT_Histos_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/Array2D.h"

#include <vector>
#include <map>
#include <list>
#include <string>

class TH1F;
class TH2F;
class TH2Poly;
class TF1;
class TProfile;
class TGraphAsymmErrors;
class TGraph;
class TEfficiency;

namespace tmtt {

  class InputData;
  class TP;
  class Sector;
  class HTrphi;
  class Make3Dtracks;
  class L1fittedTrack;
  class L1fittedTrk4and5;

  class Histos {
  public:
    // Store cfg parameters.
    Histos(const Settings* settings);

    virtual ~Histos() = default;

    // Book & fill all histograms.
    virtual void book();
    virtual void fill(const InputData& inputData,
                      const Array2D<std::unique_ptr<Sector>>& mSectors,
                      const Array2D<std::unique_ptr<HTrphi>>& mHtPhis,
                      const Array2D<std::unique_ptr<Make3Dtracks>>& mGet3Dtrks,
                      const std::map<std::string, std::list<const L1fittedTrack*>>& mapFinalTracks);

    // Print tracking performance summary & make tracking efficiency histograms.
    virtual void endJobAnalysis(const HTrphi::ErrorMonitor* htRphiErrMon = nullptr);

    // Determine "B" parameter, used in GP firmware to allow for tilted modules.
    virtual void trackerGeometryAnalysis(const std::list<TrackerModule>& listTrackerModule);

    // Did user request output histograms via the TFileService in their cfg?
    virtual bool available() const { return fs_.isAvailable(); }

    // Should histograms be produced?
    virtual bool enabled() const { return (settings_->enableHistos() && available()); }

  protected:
    // Book histograms for specific topics.
    virtual TFileDirectory bookInputData();
    virtual TFileDirectory bookEtaPhiSectors();
    virtual TFileDirectory bookRphiHT();
    virtual TFileDirectory bookRZfilters();
    virtual TFileDirectory bookTrackCands(const std::string& tName);
    virtual std::map<std::string, TFileDirectory> bookTrackFitting();

    // Fill histograms for specific topics.
    virtual void fillInputData(const InputData& inputData);
    virtual void fillEtaPhiSectors(const InputData& inputData, const Array2D<std::unique_ptr<Sector>>& mSectors);
    virtual void fillRphiHT(const Array2D<std::unique_ptr<HTrphi>>& mHtRphis);
    virtual void fillRZfilters(const Array2D<std::unique_ptr<Make3Dtracks>>& mMake3Dtrks);
    virtual void fillTrackCands(const InputData& inputData,
                                const Array2D<std::unique_ptr<Make3Dtracks>>& mMake3Dtrks,
                                const std::string& tName);
    virtual void fillTrackCands(const InputData& inputData,
                                const std::vector<L1track3D>& tracks,
                                const std::string& tName);
    virtual void fillTrackFitting(const InputData& inputData,
                                  const std::map<std::string, std::list<const L1fittedTrack*>>& mapFinalTracks);

    // Produce plots of tracking efficiency after HZ or after r-z track filter (run at end of job)
    virtual TFileDirectory plotTrackEfficiency(const std::string& tName);
    // Produce plots of tracking efficiency after track fit (run at end of job).
    virtual TFileDirectory plotTrackEffAfterFit(const std::string& fitName);

    // For Hybrid tracking
    // Produce plots of tracklet seed finding efficiency before track reco
    virtual void plotTrackletSeedEfficiency(){};
    // Produce plots of hybrid duplicate removal efficiency after track reco
    virtual void plotHybridDupRemovalEfficiency(){};

    virtual void makeEfficiencyPlot(
        TFileDirectory& inputDir, TEfficiency* outputEfficiency, TH1F* pass, TH1F* all, TString name, TString title);

    // Print summary of track-finding performance after track pattern reco.
    virtual void printTrackPerformance(const std::string& tName);

    // Print summary of track-finding performance after helix fit for given track fitter.
    virtual void printFitTrackPerformance(const std::string& fitName);

    // For Hybrid tracking
    // Print summary of seed finding and extrapolation performance during track pattern reco.
    virtual void printTrackletSeedFindingPerformance(){};

    // Print summary of duplicate removal performance after track pattern reco.
    virtual void printHybridDupRemovalPerformance(){};

  protected:
    edm::Service<TFileService> fs_;

    // Configuration parameters.
    const Settings* settings_;
    unsigned int genMinStubLayers_;
    unsigned int numPhiSectors_;
    unsigned int numEtaRegions_;
    float houghMinPt_;
    unsigned int houghNbinsPt_;
    unsigned int houghNbinsPhi_;
    float chosenRofZ_;
    std::vector<std::string> trackFitters_;
    std::vector<std::string> useRZfilter_;
    bool ranRZfilter_;
    bool resPlotOpt_;

    bool oldSumW2opt_;

    // Histograms of input data.
    TH1F* hisStubsVsEta_;
    TH1F* hisStubsVsR_;

    TH1F* hisNumLayersPerTP_;
    TH1F* hisNumPSLayersPerTP_;

    TProfile* hisStubKillFE_;
    TProfile* hisStubIneffiVsInvPt_;
    TProfile* hisStubIneffiVsEta_;
    TH1F* hisBendStub_;
    TH1F* hisBendResStub_;

    // Histograms of B parameter for tilted modules.
    TGraph* graphBVsZoverR_;

    // Histograms checking that (eta,phi) sector definition is good.
    TH1F* hisNumEtaSecsPerStub_;
    TH1F* hisNumPhiSecsPerStub_;
    TH1F* hisNumStubsPerSec_;

    // Histograms studying 3D track candidates.
    std::map<std::string, TProfile*> profNumTrackCands_;
    std::map<std::string, TProfile*> profNumTracksVsEta_;
    std::map<std::string, TH1F*> hisNumTracksVsQoverPt_;
    std::map<std::string, TH1F*> hisNumTrksPerNon_;
    std::map<std::string, TProfile*> profStubsOnTracks_;
    std::map<std::string, TH1F*> hisStubsOnTracksPerNon_;
    std::map<std::string, TH1F*> hisStubsPerTrack_;
    std::map<std::string, TH1F*> hisLayersPerTrack_;

    std::map<std::string, TH1F*> hisNumStubsPerLink_;
    std::map<std::string, TProfile*> profMeanStubsPerLink_;
    std::map<std::string, TH1F*> hisFracMatchStubsOnTracks_;
    std::map<std::string, TProfile*> profDupTracksVsEta_;
    std::map<std::string, TProfile*> profDupTracksVsInvPt_;

    // Histos of track params after HT.
    std::map<std::string, TH1F*> hisQoverPt_;
    std::map<std::string, TH1F*> hisPhi0_;
    std::map<std::string, TH1F*> hisEta_;
    std::map<std::string, TH1F*> hisZ0_;

    // Histograms of track parameter resolution after HT transform.
    std::map<std::string, TH1F*> hisQoverPtRes_;
    std::map<std::string, TH1F*> hisPhi0Res_;
    std::map<std::string, TH1F*> hisEtaRes_;
    std::map<std::string, TH1F*> hisZ0Res_;

    // Histos used for denominator of tracking efficiency plots.
    TH1F* hisTPinvptForEff_;
    TH1F* hisTPetaForEff_;
    TH1F* hisTPphiForEff_;
    TH1F* hisTPd0ForEff_;
    TH1F* hisTPz0ForEff_;
    //
    TH1F* hisTPinvptForAlgEff_;
    TH1F* hisTPetaForAlgEff_;
    TH1F* hisTPphiForAlgEff_;
    TH1F* hisTPd0ForAlgEff_;
    TH1F* hisTPz0ForAlgEff_;

    // Histograms used to make efficiency plots with 3D track candidates prior to fit.
    std::map<std::string, TH1F*> hisRecoTPinvptForEff_;
    std::map<std::string, TH1F*> hisRecoTPetaForEff_;
    std::map<std::string, TH1F*> hisRecoTPphiForEff_;
    std::map<std::string, TH1F*> hisRecoTPd0ForEff_;
    std::map<std::string, TH1F*> hisRecoTPz0ForEff_;
    //
    std::map<std::string, TH1F*> hisPerfRecoTPinvptForEff_;
    std::map<std::string, TH1F*> hisPerfRecoTPetaForEff_;
    //
    std::map<std::string, TH1F*> hisRecoTPinvptForAlgEff_;
    std::map<std::string, TH1F*> hisRecoTPetaForAlgEff_;
    std::map<std::string, TH1F*> hisRecoTPphiForAlgEff_;
    std::map<std::string, TH1F*> hisRecoTPd0ForAlgEff_;
    std::map<std::string, TH1F*> hisRecoTPz0ForAlgEff_;
    //
    std::map<std::string, TH1F*> hisPerfRecoTPinvptForAlgEff_;
    std::map<std::string, TH1F*> hisPerfRecoTPetaForAlgEff_;

    // Histograms for track fitting evaluation, where std::map index specifies name of track fitting algorithm used.

    std::map<std::string, TProfile*> profNumFitTracks_;
    std::map<std::string, TH1F*> hisNumFitTrks_;
    std::map<std::string, TH1F*> hisNumFitTrksPerNon_;
    std::map<std::string, TH1F*> hisNumFitTrksPerSect_;
    std::map<std::string, TH1F*> hisStubsPerFitTrack_;
    std::map<std::string, TProfile*> profStubsOnFitTracks_;

    std::map<std::string, TH1F*> hisFitQinvPtMatched_;
    std::map<std::string, TH1F*> hisFitPhi0Matched_;
    std::map<std::string, TH1F*> hisFitD0Matched_;
    std::map<std::string, TH1F*> hisFitZ0Matched_;
    std::map<std::string, TH1F*> hisFitEtaMatched_;

    std::map<std::string, TH1F*> hisFitQinvPtUnmatched_;
    std::map<std::string, TH1F*> hisFitPhi0Unmatched_;
    std::map<std::string, TH1F*> hisFitD0Unmatched_;
    std::map<std::string, TH1F*> hisFitZ0Unmatched_;
    std::map<std::string, TH1F*> hisFitEtaUnmatched_;

    std::map<std::string, TH1F*> hisKalmanNumUpdateCalls_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay0Matched_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay1Matched_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay2Matched_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay0Unmatched_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay1Unmatched_;
    std::map<std::string, TH1F*> hisKalmanChi2DofSkipLay2Unmatched_;

    std::map<std::string, TH1F*> hisFitChi2DofRphiMatched_;
    std::map<std::string, TH1F*> hisFitChi2DofRzMatched_;
    std::map<std::string, TProfile*> profFitChi2DofRphiVsInvPtMatched_;

    std::map<std::string, TH1F*> hisFitChi2DofRphiUnmatched_;
    std::map<std::string, TH1F*> hisFitChi2DofRzUnmatched_;
    std::map<std::string, TProfile*> profFitChi2DofRphiVsInvPtUnmatched_;

    std::map<std::string, TProfile*> hisQoverPtResVsTrueEta_;
    std::map<std::string, TProfile*> hisPhi0ResVsTrueEta_;
    std::map<std::string, TProfile*> hisEtaResVsTrueEta_;
    std::map<std::string, TProfile*> hisZ0ResVsTrueEta_;
    std::map<std::string, TProfile*> hisD0ResVsTrueEta_;

    std::map<std::string, TProfile*> hisQoverPtResVsTrueInvPt_;
    std::map<std::string, TProfile*> hisPhi0ResVsTrueInvPt_;
    std::map<std::string, TProfile*> hisEtaResVsTrueInvPt_;
    std::map<std::string, TProfile*> hisZ0ResVsTrueInvPt_;
    std::map<std::string, TProfile*> hisD0ResVsTrueInvPt_;

    std::map<std::string, TProfile*> profDupFitTrksVsEta_;
    std::map<std::string, TProfile*> profDupFitTrksVsInvPt_;

    // Histograms used for efficiency plots made with fitted tracks.
    std::map<std::string, TH1F*> hisFitTPinvptForEff_;
    std::map<std::string, TH1F*> hisFitTPetaForEff_;
    std::map<std::string, TH1F*> hisFitTPphiForEff_;
    std::map<std::string, TH1F*> hisFitTPd0ForEff_;
    std::map<std::string, TH1F*> hisFitTPz0ForEff_;
    std::map<std::string, TH1F*> hisPerfFitTPinvptForEff_;
    std::map<std::string, TH1F*> hisPerfFitTPetaForEff_;
    std::map<std::string, TH1F*> hisFitTPinvptForAlgEff_;
    std::map<std::string, TH1F*> hisFitTPetaForAlgEff_;
    std::map<std::string, TH1F*> hisFitTPphiForAlgEff_;
    std::map<std::string, TH1F*> hisFitTPd0ForAlgEff_;
    std::map<std::string, TH1F*> hisFitTPz0ForAlgEff_;
    std::map<std::string, TH1F*> hisPerfFitTPinvptForAlgEff_;
    std::map<std::string, TH1F*> hisPerfFitTPetaForAlgEff_;

    // Histograms of tracking efficiency & fake rate after Hough transform or after r-z track filter.
    std::map<std::string, TEfficiency*> teffEffVsInvPt_;
    std::map<std::string, TEfficiency*> teffEffVsEta_;
    std::map<std::string, TEfficiency*> teffEffVsPhi_;
    std::map<std::string, TEfficiency*> teffEffVsD0_;
    std::map<std::string, TEfficiency*> teffEffVsZ0_;
    //
    std::map<std::string, TEfficiency*> teffPerfEffVsInvPt_;
    std::map<std::string, TEfficiency*> teffPerfEffVsEta_;
    std::map<std::string, TEfficiency*> teffAlgEffVsD0_;
    std::map<std::string, TEfficiency*> teffAlgEffVsZ0_;
    //
    std::map<std::string, TEfficiency*> teffAlgEffVsInvPt_;
    std::map<std::string, TEfficiency*> teffAlgEffVsEta_;
    std::map<std::string, TEfficiency*> teffAlgEffVsPhi_;
    //
    std::map<std::string, TEfficiency*> teffPerfAlgEffVsInvPt_;
    std::map<std::string, TEfficiency*> teffPerfAlgEffVsPt_;
    std::map<std::string, TEfficiency*> teffPerfAlgEffVsEta_;

    // Histograms of tracking efficiency & fake rate after Hough transform based on tracks after the track fit.
    std::map<std::string, TEfficiency*> teffEffFitVsInvPt_;
    std::map<std::string, TEfficiency*> teffEffFitVsEta_;
    std::map<std::string, TEfficiency*> teffEffFitVsPhi_;
    std::map<std::string, TEfficiency*> teffEffFitVsD0_;
    std::map<std::string, TEfficiency*> teffEffFitVsZ0_;
    //
    std::map<std::string, TEfficiency*> teffPerfEffFitVsInvPt_;
    std::map<std::string, TEfficiency*> teffPerfEffFitVsEta_;
    //
    std::map<std::string, TEfficiency*> teffAlgEffFitVsInvPt_;
    std::map<std::string, TEfficiency*> teffAlgEffFitVsEta_;
    std::map<std::string, TEfficiency*> teffAlgEffFitVsPhi_;
    std::map<std::string, TEfficiency*> teffAlgEffFitVsD0_;
    std::map<std::string, TEfficiency*> teffAlgEffFitVsZ0_;
    //
    std::map<std::string, TEfficiency*> teffPerfAlgEffFitVsInvPt_;
    std::map<std::string, TEfficiency*> teffPerfAlgEffFitVsEta_;

    // Number of genuine reconstructed and perfectly reconstructed tracks which were fitted.
    std::map<std::string, unsigned int> numFitAlgEff_;
    std::map<std::string, unsigned int> numFitPerfAlgEff_;

    // Number of genuine reconstructed and perfectly reconstructed tracks which were fitted post-cut.
    std::map<std::string, unsigned int> numFitAlgEffPass_;
    std::map<std::string, unsigned int> numFitPerfAlgEffPass_;

    // Range in r of each barrel layer.
    std::map<unsigned int, float> mapBarrelLayerMinR_;
    std::map<unsigned int, float> mapBarrelLayerMaxR_;
    // Range in z of each endcap wheel.
    std::map<unsigned int, float> mapEndcapWheelMinZ_;
    std::map<unsigned int, float> mapEndcapWheelMaxZ_;

    // Range in (r,z) of each module type.
    std::map<unsigned int, float> mapModuleTypeMinR_;
    std::map<unsigned int, float> mapModuleTypeMaxR_;
    std::map<unsigned int, float> mapModuleTypeMinZ_;
    std::map<unsigned int, float> mapModuleTypeMaxZ_;
    // Extra std::maps for wierd barrel layers 1-2 & endcap wheels 3-5.
    std::map<unsigned int, float> mapExtraAModuleTypeMinR_;
    std::map<unsigned int, float> mapExtraAModuleTypeMaxR_;
    std::map<unsigned int, float> mapExtraAModuleTypeMinZ_;
    std::map<unsigned int, float> mapExtraAModuleTypeMaxZ_;
    std::map<unsigned int, float> mapExtraBModuleTypeMinR_;
    std::map<unsigned int, float> mapExtraBModuleTypeMaxR_;
    std::map<unsigned int, float> mapExtraBModuleTypeMinZ_;
    std::map<unsigned int, float> mapExtraBModuleTypeMaxZ_;
    std::map<unsigned int, float> mapExtraCModuleTypeMinR_;
    std::map<unsigned int, float> mapExtraCModuleTypeMaxR_;
    std::map<unsigned int, float> mapExtraCModuleTypeMinZ_;
    std::map<unsigned int, float> mapExtraCModuleTypeMaxZ_;
    std::map<unsigned int, float> mapExtraDModuleTypeMinR_;
    std::map<unsigned int, float> mapExtraDModuleTypeMaxR_;
    std::map<unsigned int, float> mapExtraDModuleTypeMinZ_;
    std::map<unsigned int, float> mapExtraDModuleTypeMaxZ_;

    bool bApproxMistake_;

    bool printedGeomAnalysis_;
  };

}  // namespace tmtt
#endif
