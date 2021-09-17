#include "L1Trigger/TrackFindingTMTT/interface/Histos.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/Make3Dtracks.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrkRZfilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TH2Poly.h>
#include <TF1.h>
#include <TPad.h>
#include <TProfile.h>
#include <TGraphAsymmErrors.h>
#include <TGraph.h>
#include <TEfficiency.h>

#include <algorithm>
#include <array>
#include <unordered_set>
#include <list>
#include <sstream>
#include <memory>
#include <mutex>

using namespace std;

namespace tmtt {

  //=== Store cfg parameters.

  Histos::Histos(const Settings* settings) : settings_(settings), oldSumW2opt_(false), bApproxMistake_(false) {
    genMinStubLayers_ = settings->genMinStubLayers();
    numPhiSectors_ = settings->numPhiSectors();
    numEtaRegions_ = settings->numEtaRegions();
    houghMinPt_ = settings->houghMinPt();
    houghNbinsPt_ = settings->houghNbinsPt();
    houghNbinsPhi_ = settings->houghNbinsPhi();
    chosenRofZ_ = settings->chosenRofZ();
    trackFitters_ = settings->trackFitters();
    useRZfilter_ = settings->useRZfilter();
    ranRZfilter_ = (not useRZfilter_.empty());  // Was any r-z track filter run?
    resPlotOpt_ = settings->resPlotOpt();       // Only use signal events for helix resolution plots?
  }

  //=== Book all histograms

  void Histos::book() {
    // Don't bother booking histograms if user didn't request them via TFileService in their cfg.
    if (not this->enabled())
      return;

    oldSumW2opt_ = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);

    // Book histograms about input data.
    this->bookInputData();
    // Book histograms checking if (eta,phi) sector definition choices are good.
    this->bookEtaPhiSectors();
    // Book histograms checking filling of r-phi HT array.
    this->bookRphiHT();
    // Book histograms about r-z track filters.
    if (ranRZfilter_)
      this->bookRZfilters();
    // Book histograms studying 3D track candidates found after HT.
    this->bookTrackCands("HT");
    // Book histograms studying 3D track candidates found after r-z track filter.
    if (ranRZfilter_)
      this->bookTrackCands("RZ");
    // Book histograms studying track fitting performance
    this->bookTrackFitting();
  }

  //=== Fill all histograms

  void Histos::fill(const InputData& inputData,
                    const Array2D<unique_ptr<Sector>>& mSectors,
                    const Array2D<unique_ptr<HTrphi>>& mHtRphis,
                    const Array2D<unique_ptr<Make3Dtracks>>& mMake3Dtrks,
                    const std::map<std::string, std::list<const L1fittedTrack*>>& mapFinalTracks) {
    // Each function here protected by a mytex lock, so only one thread can run it at a time.

    // Don't bother filling histograms if user didn't request them via TFileService in their cfg.
    if (not this->enabled())
      return;

    // Fill histograms about input data.
    this->fillInputData(inputData);

    // Fill histograms checking if (eta,phi) sector definition choices are good.
    this->fillEtaPhiSectors(inputData, mSectors);

    // Fill histograms checking filling of r-phi HT array.
    this->fillRphiHT(mHtRphis);

    // Fill histograms about r-z track filters.
    if (ranRZfilter_)
      this->fillRZfilters(mMake3Dtrks);

    // Fill histograms studying 3D track candidates found after HT.
    this->fillTrackCands(inputData, mMake3Dtrks, "HT");

    // Fill histograms studying 3D track candidates found after r-z track filter.
    if (ranRZfilter_) {
      this->fillTrackCands(inputData, mMake3Dtrks, "RZ");
    }

    // Fill histograms studying track fitting performance
    this->fillTrackFitting(inputData, mapFinalTracks);
  }

  //=== Book histograms using input stubs and tracking particles.

  TFileDirectory Histos::bookInputData() {
    TFileDirectory inputDir = fs_->mkdir("InputData");

    hisStubsVsEta_ = inputDir.make<TH1F>("StubsVsEta", "; #eta; No. stubs in tracker", 30, -3.0, 3.0);
    hisStubsVsR_ = inputDir.make<TH1F>("StubsVsR", "; radius (cm); No. stubs in tracker", 1200, 0., 120.);

    hisNumLayersPerTP_ =
        inputDir.make<TH1F>("NumLayersPerTP", "; Number of layers per TP for alg. eff.", 20, -0.5, 19.5);
    hisNumPSLayersPerTP_ =
        inputDir.make<TH1F>("NumPSLayersPerTP", "; Number of PS layers per TP for alg. eff.", 20, -0.5, 19.5);

    // Study efficiency of tightened front end-electronics cuts.

    hisStubKillFE_ = inputDir.make<TProfile>(
        "StubKillFE", "; barrelLayer or 10+endcapRing; Stub fraction rejected by FE chip", 30, -0.5, 29.5);
    hisStubIneffiVsInvPt_ =
        inputDir.make<TProfile>("StubIneffiVsPt", "; 1/Pt; Inefficiency of FE chip for good stubs", 25, 0.0, 0.5);
    hisStubIneffiVsEta_ =
        inputDir.make<TProfile>("StubIneffiVsEta", "; |#eta|; Inefficiency of FE chip for good stubs", 15, 0.0, 3.0);

    // Study stub resolution.

    hisBendStub_ = inputDir.make<TH1F>("BendStub", "; Stub bend in units of strips", 59, -7.375, 7.375);
    hisBendResStub_ = inputDir.make<TH1F>("BendResStub", "; Stub bend minus TP bend in units of strips", 100, -5., 5.);

    // Histos for denominator of tracking efficiency
    hisTPinvptForEff_ = inputDir.make<TH1F>("TPinvptForEff", "; TP 1/Pt (for effi.);", 50, 0., 0.5);
    hisTPetaForEff_ = inputDir.make<TH1F>("TPetaForEff", "; TP #eta (for effi.);", 20, -3., 3.);
    hisTPphiForEff_ = inputDir.make<TH1F>("TPphiForEff", "; TP #phi (for effi.);", 20, -M_PI, M_PI);
    hisTPd0ForEff_ = inputDir.make<TH1F>("TPd0ForEff", "; TP d0 (for effi.);", 40, 0., 4.);
    hisTPz0ForEff_ = inputDir.make<TH1F>("TPz0ForEff", "; TP z0 (for effi.);", 50, 0., 25.);
    //
    hisTPinvptForAlgEff_ = inputDir.make<TH1F>("TPinvptForAlgEff", "; TP 1/Pt (for alg. effi.);", 50, 0., 0.5);
    hisTPetaForAlgEff_ = inputDir.make<TH1F>("TPetaForAlgEff", "; TP #eta (for alg. effi.);", 20, -3., 3.);
    hisTPphiForAlgEff_ = inputDir.make<TH1F>("TPphiForAlgEff", "; TP #phi (for alg. effi.);", 20, -M_PI, M_PI);
    hisTPd0ForAlgEff_ = inputDir.make<TH1F>("TPd0ForAlgEff", "; TP d0 (for alg. effi.);", 40, 0., 4.);
    hisTPz0ForAlgEff_ = inputDir.make<TH1F>("TPz0ForAlgEff", "; TP z0 (for alg. effi.);", 50, 0., 25.);

    return inputDir;
  }

  //=== Fill histograms using input stubs and tracking particles.

  void Histos::fillInputData(const InputData& inputData) {
    // Allow only one thread to run this function at a time
    static std::mutex myMutex;
    std::lock_guard<std::mutex> myGuard(myMutex);

    const list<const Stub*>& vStubs = inputData.stubsConst();
    const list<TP>& vTPs = inputData.getTPs();

    for (const Stub* stub : vStubs) {
      hisStubsVsEta_->Fill(stub->eta());
      hisStubsVsR_->Fill(stub->r());
    }

    // Study efficiency of stubs to pass front-end electronics cuts.

    const list<Stub>& vAllStubs = inputData.allStubs();  // Get all stubs prior to FE cuts to do this.
    for (const Stub& s : vAllStubs) {
      unsigned int layerOrTenPlusRing = s.barrel() ? s.layerId() : 10 + s.trackerModule()->endcapRing();
      // Fraction of all stubs (good and bad) failing tightened front-end electronics cuts.
      hisStubKillFE_->Fill(layerOrTenPlusRing, (!s.frontendPass()));
    }

    // Study efficiency for good stubs of tightened front end-electronics cuts.
    for (const TP& tp : vTPs) {
      if (tp.useForAlgEff()) {  // Only bother for stubs that are on TP that we have a chance of reconstructing.
        const vector<const Stub*>& stubs = tp.assocStubs();
        for (const Stub* s : stubs) {
          hisStubIneffiVsInvPt_->Fill(1. / tp.pt(), (!s->frontendPass()));
          hisStubIneffiVsEta_->Fill(std::abs(tp.eta()), (!s->frontendPass()));
        }
      }
    }

    // Plot stub bend-derived information.
    for (const Stub* stub : vStubs) {
      hisBendStub_->Fill(stub->bend());
    }

    // Look at stub resolution.
    for (const TP& tp : vTPs) {
      if (tp.useForAlgEff()) {
        const vector<const Stub*>& assStubs = tp.assocStubs();

        for (const Stub* stub : assStubs) {
          hisBendResStub_->Fill(stub->bend() - tp.dphi(stub->r()) / stub->dphiOverBend());
        }

        if (std::abs(tp.eta()) < 0.5) {
          double nLayersOnTP = Utility::countLayers(settings_, assStubs, true, false);
          double nPSLayersOnTP = Utility::countLayers(settings_, assStubs, true, true);
          hisNumLayersPerTP_->Fill(nLayersOnTP);
          hisNumPSLayersPerTP_->Fill(nPSLayersOnTP);
        }
      }
    }

    // Determine r (z) range of each barrel layer (endcap wheel).

    for (const Stub* stub : vStubs) {
      unsigned int layer = stub->layerId();
      if (stub->barrel()) {
        // Get range in r of each barrel layer.
        float r = stub->r();
        if (mapBarrelLayerMinR_.find(layer) == mapBarrelLayerMinR_.end()) {
          mapBarrelLayerMinR_[layer] = r;
          mapBarrelLayerMaxR_[layer] = r;
        } else {
          if (mapBarrelLayerMinR_[layer] > r)
            mapBarrelLayerMinR_[layer] = r;
          if (mapBarrelLayerMaxR_[layer] < r)
            mapBarrelLayerMaxR_[layer] = r;
        }
      } else {
        layer = layer % 10;
        // Range in |z| of each endcap wheel.
        float z = std::abs(stub->z());
        if (mapEndcapWheelMinZ_.find(layer) == mapEndcapWheelMinZ_.end()) {
          mapEndcapWheelMinZ_[layer] = z;
          mapEndcapWheelMaxZ_[layer] = z;
        } else {
          if (mapEndcapWheelMinZ_[layer] > z)
            mapEndcapWheelMinZ_[layer] = z;
          if (mapEndcapWheelMaxZ_[layer] < z)
            mapEndcapWheelMaxZ_[layer] = z;
        }
      }
    }

    // Determine Range in (r,|z|) of each module type.

    for (const Stub* stub : vStubs) {
      float r = stub->r();
      float z = std::abs(stub->z());
      unsigned int modType = stub->trackerModule()->moduleTypeID();
      // Do something ugly, as modules in 1-2nd & 3-4th endcap wheels are different to those in wheel 5 ...
      // And boundary between flat & tilted modules in barrel layers 1-3 varies in z.
      if (stub->barrel() && stub->layerId() == 1) {  // barrel layer 1
        if (mapExtraAModuleTypeMinR_.find(modType) == mapExtraAModuleTypeMinR_.end()) {
          mapExtraAModuleTypeMinR_[modType] = r;
          mapExtraAModuleTypeMaxR_[modType] = r;
          mapExtraAModuleTypeMinZ_[modType] = z;
          mapExtraAModuleTypeMaxZ_[modType] = z;
        } else {
          if (mapExtraAModuleTypeMinR_[modType] > r)
            mapExtraAModuleTypeMinR_[modType] = r;
          if (mapExtraAModuleTypeMaxR_[modType] < r)
            mapExtraAModuleTypeMaxR_[modType] = r;
          if (mapExtraAModuleTypeMinZ_[modType] > z)
            mapExtraAModuleTypeMinZ_[modType] = z;
          if (mapExtraAModuleTypeMaxZ_[modType] < z)
            mapExtraAModuleTypeMaxZ_[modType] = z;
        }
      } else if (stub->barrel() && stub->layerId() == 2) {  // barrel layer 2
        if (mapExtraBModuleTypeMinR_.find(modType) == mapExtraBModuleTypeMinR_.end()) {
          mapExtraBModuleTypeMinR_[modType] = r;
          mapExtraBModuleTypeMaxR_[modType] = r;
          mapExtraBModuleTypeMinZ_[modType] = z;
          mapExtraBModuleTypeMaxZ_[modType] = z;
        } else {
          if (mapExtraBModuleTypeMinR_[modType] > r)
            mapExtraBModuleTypeMinR_[modType] = r;
          if (mapExtraBModuleTypeMaxR_[modType] < r)
            mapExtraBModuleTypeMaxR_[modType] = r;
          if (mapExtraBModuleTypeMinZ_[modType] > z)
            mapExtraBModuleTypeMinZ_[modType] = z;
          if (mapExtraBModuleTypeMaxZ_[modType] < z)
            mapExtraBModuleTypeMaxZ_[modType] = z;
        }
      } else if (!stub->barrel() && (stub->layerId() % 10 == 1 || stub->layerId() % 10 == 2)) {  // endcap wheel 1-2
        if (mapExtraCModuleTypeMinR_.find(modType) == mapExtraCModuleTypeMinR_.end()) {
          mapExtraCModuleTypeMinR_[modType] = r;
          mapExtraCModuleTypeMaxR_[modType] = r;
          mapExtraCModuleTypeMinZ_[modType] = z;
          mapExtraCModuleTypeMaxZ_[modType] = z;
        } else {
          if (mapExtraCModuleTypeMinR_[modType] > r)
            mapExtraCModuleTypeMinR_[modType] = r;
          if (mapExtraCModuleTypeMaxR_[modType] < r)
            mapExtraCModuleTypeMaxR_[modType] = r;
          if (mapExtraCModuleTypeMinZ_[modType] > z)
            mapExtraCModuleTypeMinZ_[modType] = z;
          if (mapExtraCModuleTypeMaxZ_[modType] < z)
            mapExtraCModuleTypeMaxZ_[modType] = z;
        }
      } else if (!stub->barrel() && (stub->layerId() % 10 == 3 || stub->layerId() % 10 == 4)) {  // endcap wheel 3-4
        if (mapExtraDModuleTypeMinR_.find(modType) == mapExtraDModuleTypeMinR_.end()) {
          mapExtraDModuleTypeMinR_[modType] = r;
          mapExtraDModuleTypeMaxR_[modType] = r;
          mapExtraDModuleTypeMinZ_[modType] = z;
          mapExtraDModuleTypeMaxZ_[modType] = z;
        } else {
          if (mapExtraDModuleTypeMinR_[modType] > r)
            mapExtraDModuleTypeMinR_[modType] = r;
          if (mapExtraDModuleTypeMaxR_[modType] < r)
            mapExtraDModuleTypeMaxR_[modType] = r;
          if (mapExtraDModuleTypeMinZ_[modType] > z)
            mapExtraDModuleTypeMinZ_[modType] = z;
          if (mapExtraDModuleTypeMaxZ_[modType] < z)
            mapExtraDModuleTypeMaxZ_[modType] = z;
        }
      } else {  // barrel layer 3-6 or endcap wheel 5.
        if (mapModuleTypeMinR_.find(modType) == mapModuleTypeMinR_.end()) {
          mapModuleTypeMinR_[modType] = r;
          mapModuleTypeMaxR_[modType] = r;
          mapModuleTypeMinZ_[modType] = z;
          mapModuleTypeMaxZ_[modType] = z;
        } else {
          if (mapModuleTypeMinR_[modType] > r)
            mapModuleTypeMinR_[modType] = r;
          if (mapModuleTypeMaxR_[modType] < r)
            mapModuleTypeMaxR_[modType] = r;
          if (mapModuleTypeMinZ_[modType] > z)
            mapModuleTypeMinZ_[modType] = z;
          if (mapModuleTypeMaxZ_[modType] < z)
            mapModuleTypeMaxZ_[modType] = z;
        }
      }
    }

    //=== Make denominator of tracking efficiency plots

    for (const TP& tp : vTPs) {
      if (tp.useForEff()) {  // Check TP is good for efficiency measurement.
        // Plot kinematics of all good TP.
        hisTPinvptForEff_->Fill(1. / tp.pt());
        hisTPetaForEff_->Fill(tp.eta());
        hisTPphiForEff_->Fill(tp.phi0());
        // Plot also production point of all good TP.
        hisTPd0ForEff_->Fill(std::abs(tp.d0()));
        hisTPz0ForEff_->Fill(std::abs(tp.z0()));

        if (tp.useForAlgEff()) {  // Check TP is good for algorithmic efficiency measurement.
          hisTPinvptForAlgEff_->Fill(1. / tp.pt());
          hisTPetaForAlgEff_->Fill(tp.eta());
          hisTPphiForAlgEff_->Fill(tp.phi0());
          // Plot also production point of all good TP.
          hisTPd0ForAlgEff_->Fill(std::abs(tp.d0()));
          hisTPz0ForAlgEff_->Fill(std::abs(tp.z0()));
        }
      }
    }
  }

  //=== Book histograms checking if (eta,phi) sector defis(nition choices are good.

  TFileDirectory Histos::bookEtaPhiSectors() {
    TFileDirectory inputDir = fs_->mkdir("CheckSectors");

    // Check if stubs excessively duplicated between overlapping sectors.
    hisNumEtaSecsPerStub_ =
        inputDir.make<TH1F>("NumEtaSecPerStub", "; No. of #eta sectors each stub in", 20, -0.5, 19.5);
    hisNumPhiSecsPerStub_ =
        inputDir.make<TH1F>("NumPhiSecPerStub", "; No. of #phi sectors each stub in", 20, -0.5, 19.5);

    // Count stubs per (eta,phi) sector.
    hisNumStubsPerSec_ = inputDir.make<TH1F>("NumStubsPerSec", "; No. of stubs per sector", 150, -0.5, 299.5);

    return inputDir;
  }

  //=== Fill histograms checking if (eta,phi) sector definition choices are good.

  void Histos::fillEtaPhiSectors(const InputData& inputData, const Array2D<unique_ptr<Sector>>& mSectors) {
    // Allow only one thread to run this function at a time
    static std::mutex myMutex;
    std::lock_guard<std::mutex> myGuard(myMutex);

    const list<const Stub*>& vStubs = inputData.stubsConst();
    //const list<TP>& vTPs = inputData.getTPs();

    //=== Loop over all stubs, counting how many sectors each one appears in.

    for (const Stub* stub : vStubs) {
      // Number of (eta,phi), phi & eta sectors containing this stub.
      unsigned int nEtaSecs = 0;
      unsigned int nPhiSecs = 0;

      // Loop over (eta, phi) sectors.
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
        for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
          const Sector* sector = mSectors(iPhiSec, iEtaReg).get();

          // Check if sector contains stub stub, and if so count it.
          // Take care to just use one eta (phi) typical region when counting phi (eta) sectors.
          if (iPhiSec == 0 && sector->insideEta(stub))
            nEtaSecs++;
          if (iEtaReg == 0 && sector->insidePhi(stub))
            nPhiSecs++;
        }
      }

      // Plot number of sectors each stub appears in.
      hisNumEtaSecsPerStub_->Fill(nEtaSecs);
      hisNumPhiSecsPerStub_->Fill(nPhiSecs);
    }

    //=== Loop over all sectors, counting the stubs in each one.
    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
        const Sector* sector = mSectors(iPhiSec, iEtaReg).get();

        unsigned int nStubs = 0;
        for (const Stub* stub : vStubs) {
          if (sector->inside(stub))
            nStubs++;
        }
        hisNumStubsPerSec_->Fill(nStubs);
      }
    }
  }

  //=== Book histograms checking filling of r-phi HT array.

  TFileDirectory Histos::bookRphiHT() {
    TFileDirectory inputDir = fs_->mkdir("HTrphi");

    return inputDir;
  }

  //=== Fill histograms checking filling of r-phi HT array.

  void Histos::fillRphiHT(const Array2D<unique_ptr<HTrphi>>& mHtRphis) {
    //--- Loop over (eta,phi) sectors, counting the number of stubs in the HT array of each.

    // Allow only one thread to run this function at a time (UNCOMMENT IF YOU ADD HISTOS HERE)
    //static std::mutex myMutex;
    //std::lock_guard<std::mutex> myGuard(myMutex);
  }

  //=== Book histograms about r-z track filters (or other filters applied after r-phi HT array).

  TFileDirectory Histos::bookRZfilters() {
    TFileDirectory inputDir = fs_->mkdir("RZfilters");

    return inputDir;
  }

  //=== Fill histograms about r-z track filters.

  void Histos::fillRZfilters(const Array2D<unique_ptr<Make3Dtracks>>& mMake3Dtrks) {
    // Allow only one thread to run this function at a time (UNCOMMENT IF YOU ADD HISTOS HERE)
    //static std::mutex myMutex;
    //std::lock_guard<std::mutex> myGuard(myMutex);
  }

  //=== Book histograms studying track candidates found by Hough Transform.

  TFileDirectory Histos::bookTrackCands(const string& tName) {
    // Now book histograms for studying tracking in general.

    auto addn = [tName](const string& s) { return TString::Format("%s_%s", s.c_str(), tName.c_str()); };

    TFileDirectory inputDir = fs_->mkdir(addn("TrackCands").Data());

    bool TMTT = (tName == "HT" || tName == "RZ");

    // Count tracks in various ways (including/excluding duplicates, excluding fakes ...)
    profNumTrackCands_[tName] =
        inputDir.make<TProfile>(addn("NumTrackCands"), "; class; N. of tracks in tracker", 7, 0.5, 7.5);
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(7, "TP for eff recoed");
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(6, "TP recoed");
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(5, "TP recoed x #eta sector dups");
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(4, "TP recoed x sector dups");
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(2, "TP recoed x track dups");
    profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(1, "reco tracks including fakes");
    profNumTrackCands_[tName]->LabelsOption("d");

    hisNumTrksPerNon_[tName] = inputDir.make<TH1F>(addn("NumTrksPerNon"), "; No. tracks per nonant;", 100, -0.5, 399.5);

    unsigned int nEta = numEtaRegions_;
    hisNumTracksVsQoverPt_[tName] =
        inputDir.make<TH1F>(addn("NumTracksVsQoverPt"), "; Q/Pt; No. of tracks in tracker", 100, -0.5, 0.5);
    if (TMTT) {
      profNumTracksVsEta_[tName] = inputDir.make<TProfile>(
          addn("NumTracksVsEta"), "; #eta region; No. of tracks in tracker", nEta, -0.5, nEta - 0.5);
    }

    // Count stubs per event assigned to tracks (determines HT data output rate)

    profStubsOnTracks_[tName] =
        inputDir.make<TProfile>(addn("StubsOnTracks"), "; ; No. of stubs on tracks per event", 1, 0.5, 1.5);
    hisStubsOnTracksPerNon_[tName] =
        inputDir.make<TH1F>(addn("StubsOnTracksPerNon"), "; No. of stubs on tracks per nonant", 100, -0.5, 4999.5);

    hisStubsPerTrack_[tName] = inputDir.make<TH1F>(addn("StubsPerTrack"), ";No. of stubs per track;", 50, -0.5, 49.5);
    hisLayersPerTrack_[tName] =
        inputDir.make<TH1F>(addn("LayersPerTrack"), ";No. of layers with stubs per track;", 20, -0.5, 19.5);

    if (TMTT) {
      hisNumStubsPerLink_[tName] =
          inputDir.make<TH1F>(addn("NumStubsPerLink"), "; Mean #stubs per MHT output opto-link;", 50, -0.5, 249.5);
      profMeanStubsPerLink_[tName] =
          inputDir.make<TProfile>(addn("MeanStubsPerLink"), "; Mean #stubs per MHT output opto-link;", 36, -0.5, 35.5);
    }

    hisFracMatchStubsOnTracks_[tName] = inputDir.make<TH1F>(
        addn("FracMatchStubsOnTracks"), "; Frac. of stubs per trk matching best TP;", 101, -0.005, 1.005);

    if (TMTT) {
      // Study duplication of tracks within an individual HT array.
      profDupTracksVsEta_[tName] =
          inputDir.make<TProfile>(addn("DupTracksVsTPeta"), "; #eta; No. of dup. trks per TP;", 15, 0.0, 3.0);
      profDupTracksVsInvPt_[tName] =
          inputDir.make<TProfile>(addn("DupTracksVsInvPt"), "; 1/Pt; No. of dup. trks per TP", 25, 0., 0.5);
    }

    // Histos of track params.
    hisQoverPt_[tName] = inputDir.make<TH1F>(addn("QoverPt"), "; track q/Pt", 100, -0.5, 0.5);
    hisPhi0_[tName] = inputDir.make<TH1F>(addn("Phi0"), "; track #phi0", 70, -3.5, 3.5);
    hisEta_[tName] = inputDir.make<TH1F>(addn("Eta"), "; track #eta", 60, -3.0, 3.0);
    hisZ0_[tName] = inputDir.make<TH1F>(addn("Z0"), "; track z0", 100, -25.0, 25.0);

    // Histos of track parameter resolution
    hisQoverPtRes_[tName] = inputDir.make<TH1F>(addn("QoverPtRes"), "; track resolution in q/Pt", 100, -0.06, 0.06);
    hisPhi0Res_[tName] = inputDir.make<TH1F>(addn("Phi0Res"), "; track resolution in #phi0", 100, -0.04, 0.04);
    hisEtaRes_[tName] = inputDir.make<TH1F>(addn("EtaRes"), "; track resolution in #eta", 100, -1.0, 1.0);
    hisZ0Res_[tName] = inputDir.make<TH1F>(addn("Z0Res"), "; track resolution in z0", 100, -10.0, 10.0);

    // Histos for tracking efficiency vs. TP kinematics
    hisRecoTPinvptForEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPinvptForEff"), "; TP 1/Pt of recoed tracks (for effi.);", 50, 0., 0.5);
    hisRecoTPetaForEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPetaForEff"), "; TP #eta of recoed tracks (for effi.);", 20, -3., 3.);
    hisRecoTPphiForEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPphiForEff"), "; TP #phi of recoed tracks (for effi.);", 20, -M_PI, M_PI);

    // Histo for efficiency to reconstruct track perfectly (no incorrect hits).
    hisPerfRecoTPinvptForEff_[tName] = inputDir.make<TH1F>(
        addn("PerfRecoTPinvptForEff"), "; TP 1/Pt of recoed tracks (for perf. effi.);", 50, 0., 0.5);
    hisPerfRecoTPetaForEff_[tName] =
        inputDir.make<TH1F>(addn("PerfRecoTPetaForEff"), "; TP #eta of recoed tracks (for perf. effi.);", 20, -3., 3.);

    // Histos for  tracking efficiency vs. TP production point
    hisRecoTPd0ForEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPd0ForEff"), "; TP d0 of recoed tracks (for effi.);", 40, 0., 4.);
    hisRecoTPz0ForEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPz0ForEff"), "; TP z0 of recoed tracks (for effi.);", 50, 0., 25.);

    // Histos for algorithmic tracking efficiency vs. TP kinematics
    hisRecoTPinvptForAlgEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPinvptForAlgEff"), "; TP 1/Pt of recoed tracks (for alg. effi.);", 50, 0., 0.5);
    hisRecoTPetaForAlgEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPetaForAlgEff"), "; TP #eta of recoed tracks (for alg. effi.);", 20, -3., 3.);
    hisRecoTPphiForAlgEff_[tName] = inputDir.make<TH1F>(
        addn("RecoTPphiForAlgEff"), "; TP #phi of recoed tracks (for alg. effi.);", 20, -M_PI, M_PI);

    // Histo for efficiency to reconstruct track perfectly (no incorrect hits).
    hisPerfRecoTPinvptForAlgEff_[tName] = inputDir.make<TH1F>(
        addn("PerfRecoTPinvptForAlgEff"), "; TP 1/Pt of recoed tracks (for perf. alg. effi.);", 50, 0., 0.5);
    hisPerfRecoTPetaForAlgEff_[tName] =
        inputDir.make<TH1F>(addn("PerfRecoTPetaForAlgEff"), "; TP #eta (for perf. alg. effi.);", 20, -3., 3.);

    // Histos for algorithmic tracking efficiency vs. TP production point
    hisRecoTPd0ForAlgEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPd0ForAlgEff"), "; TP d0 of recoed tracks (for alg. effi.);", 40, 0., 4.);
    hisRecoTPz0ForAlgEff_[tName] =
        inputDir.make<TH1F>(addn("RecoTPz0ForAlgEff"), "; TP z0 of recoed tracks (for alg. effi.);", 50, 0., 25.);

    return inputDir;
  }

  //=== Fill histograms studying track candidates found before track fit is run.

  void Histos::fillTrackCands(const InputData& inputData,
                              const Array2D<std::unique_ptr<Make3Dtracks>>& mMake3Dtrks,
                              const string& tName) {
    // Allow only one thread to run this function at a time
    static std::mutex myMutex;
    std::lock_guard<std::mutex> myGuard(myMutex);

    vector<L1track3D> tracks;
    bool withRZfilter = (tName == "RZ") ? true : false;
    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
        const Make3Dtracks* make3Dtrk = mMake3Dtrks(iPhiSec, iEtaReg).get();
        const std::list<L1track3D>& tracksSec = make3Dtrk->trackCands3D(withRZfilter);
        tracks.insert(tracks.end(), tracksSec.begin(), tracksSec.end());
      }
    }
    this->fillTrackCands(inputData, tracks, tName);
  }

  //=== Fill histograms studying track candidates found before track fit is run.

  void Histos::fillTrackCands(const InputData& inputData, const vector<L1track3D>& tracks, const string& tName) {
    bool withRZfilter = (tName == "RZ");

    bool algoTMTT = (tName == "HT" || tName == "RZ");  // Check if running TMTT or Hybrid L1 tracking.

    const list<TP>& vTPs = inputData.getTPs();

    //=== Count track candidates found in the tracker.

    const unsigned int numPhiNonants = settings_->numPhiNonants();
    vector<unsigned int> nTrksPerEtaReg(numEtaRegions_, 0);
    vector<unsigned int> nTrksPerNonant(numPhiNonants, 0);
    for (const L1track3D& t : tracks) {
      unsigned int iNonant = floor((t.iPhiSec()) * numPhiNonants / (numPhiSectors_));  // phi nonant number
      nTrksPerEtaReg[t.iEtaReg()]++;
      nTrksPerNonant[iNonant]++;
    }

    profNumTrackCands_[tName]->Fill(1.0, tracks.size());  // Plot mean number of tracks/event.
    if (algoTMTT) {
      for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
        profNumTracksVsEta_[tName]->Fill(iEtaReg, nTrksPerEtaReg[iEtaReg]);
      }
    }
    for (unsigned int iNonant = 0; iNonant < numPhiNonants; iNonant++) {
      hisNumTrksPerNon_[tName]->Fill(nTrksPerNonant[iNonant]);
    }

    //=== Count stubs per event assigned to track candidates in the Tracker

    unsigned int nStubsOnTracks = 0;
    vector<unsigned int> nStubsOnTracksInNonant(numPhiNonants, 0);

    for (const L1track3D& t : tracks) {
      const vector<const Stub*>& stubs = t.stubsConst();
      unsigned int nStubs = stubs.size();
      unsigned int iNonant = floor((t.iPhiSec()) * numPhiNonants / (numPhiSectors_));  // phi nonant number
      // Count stubs on all tracks in this sector & nonant.
      nStubsOnTracks += nStubs;
      nStubsOnTracksInNonant[iNonant] += nStubs;
    }

    profStubsOnTracks_[tName]->Fill(1.0, nStubsOnTracks);

    for (unsigned int iNonant = 0; iNonant < numPhiNonants; iNonant++) {
      hisStubsOnTracksPerNon_[tName]->Fill(nStubsOnTracksInNonant[iNonant]);
    }

    // Plot number of tracks & number of stubs per output HT opto-link.

    if (algoTMTT && not withRZfilter) {
      //const unsigned int numPhiSecPerNon = numPhiSectors_ / numPhiNonants;
      // Hard-wired bodge
      const unsigned int nLinks = houghNbinsPt_ / 2;  // Hard-wired to number of course HT bins. Check.

      for (unsigned int iPhiNon = 0; iPhiNon < numPhiNonants; iPhiNon++) {
        // Each nonant has a separate set of links.
        vector<unsigned int> stubsToLinkCount(nLinks, 0);  // Must use vectors to count links with zero entries.
        for (const L1track3D& trk : tracks) {
          unsigned int iNonantTrk = floor((trk.iPhiSec()) * numPhiNonants / (numPhiSectors_));  // phi nonant number
          if (iPhiNon == iNonantTrk) {
            unsigned int link = trk.optoLinkID();
            if (link < nLinks) {
              stubsToLinkCount[link] += trk.numStubs();
            } else {
              std::stringstream text;
              text << "\n ===== HISTOS MESS UP: Increase size of nLinks ===== " << link << "\n";
              static std::once_flag printOnce;
              std::call_once(
                  printOnce, [](string t) { edm::LogWarning("L1track") << t; }, text.str());
            }
          }
        }

        for (unsigned int link = 0; link < nLinks; link++) {
          unsigned int nstbs = stubsToLinkCount[link];
          hisNumStubsPerLink_[tName]->Fill(nstbs);
          profMeanStubsPerLink_[tName]->Fill(link, nstbs);
        }
      }
    }

    // Plot q/pt spectrum of track candidates, and number of stubs/tracks
    for (const L1track3D& trk : tracks) {
      hisNumTracksVsQoverPt_[tName]->Fill(trk.qOverPt());  // Plot reconstructed q/Pt of track cands.
      hisStubsPerTrack_[tName]->Fill(trk.numStubs());      // Stubs per track.
      hisLayersPerTrack_[tName]->Fill(trk.numLayers());
    }

    // Count fraction of stubs on each track matched to a TP that are from same TP.

    for (const L1track3D& trk : tracks) {
      // Only consider tracks that match a tracking particle for the alg. efficiency measurement.
      const TP* tp = trk.matchedTP();
      if (tp != nullptr) {
        if (tp->useForAlgEff()) {
          hisFracMatchStubsOnTracks_[tName]->Fill(trk.purity());
        }
      }
    }

    // Count total number of tracking particles in the event that were reconstructed,
    // counting also how many of them were reconstructed multiple times (duplicate tracks).

    unsigned int nRecoedTPsForEff = 0;     // Total no. of TPs for effi measurement recoed as >= 1 track.
    unsigned int nRecoedTPs = 0;           // Total no. of TPs recoed as >= 1 one track.
    unsigned int nEtaSecsMatchingTPs = 0;  // Total no. of eta sectors that all TPs were reconstructed in
    unsigned int nSecsMatchingTPs = 0;     // Total no. of eta x phi sectors that all TPs were reconstructed in
    unsigned int nTrksMatchingTPs = 0;     // Total no. of tracks that all TPs were reconstructed as

    for (const TP& tp : vTPs) {
      vector<const L1track3D*> matchedTrks;
      for (const L1track3D& trk : tracks) {
        const TP* tpAssoc = trk.matchedTP();
        if (tpAssoc != nullptr) {
          if (tpAssoc->index() == tp.index())
            matchedTrks.push_back(&trk);
        }
      }
      unsigned int nTrk = matchedTrks.size();

      bool tpRecoed = false;

      if (nTrk > 0) {
        tpRecoed = true;           // This TP was reconstructed at least once in tracker.
        nTrksMatchingTPs += nTrk;  // Increment sum by no. of tracks this TP was reconstructed as

        set<unsigned int> iEtaRegRecoed;
        for (const L1track3D* trk : matchedTrks)
          iEtaRegRecoed.insert(trk->iEtaReg());
        nEtaSecsMatchingTPs = iEtaRegRecoed.size();

        set<pair<unsigned int, unsigned int>> iSecRecoed;
        for (const L1track3D* trk : matchedTrks)
          iSecRecoed.insert({trk->iPhiSec(), trk->iEtaReg()});
        nSecsMatchingTPs = iSecRecoed.size();

        if (algoTMTT) {
          for (const auto& p : iSecRecoed) {
            unsigned int nTrkInSec = 0;
            for (const L1track3D* trk : matchedTrks) {
              if (trk->iPhiSec() == p.first && trk->iEtaReg() == p.second)
                nTrkInSec++;
            }
            if (nTrkInSec > 0) {
              profDupTracksVsEta_[tName]->Fill(
                  std::abs(tp.eta()), nTrkInSec - 1);  // Study duplication of tracks within an individual HT array.
              profDupTracksVsInvPt_[tName]->Fill(
                  std::abs(tp.qOverPt()), nTrkInSec - 1);  // Study duplication of tracks within an individual HT array.
            }
          }
        }
      }

      if (tpRecoed) {
        // Increment sum each time a TP is reconstructed at least once inside Tracker
        if (tp.useForEff())
          nRecoedTPsForEff++;
        nRecoedTPs++;
      }
    }

    //--- Plot mean number of tracks/event, counting number due to different kinds of duplicates

    // Plot number of TPs for the efficiency measurement that are reconstructed.
    profNumTrackCands_[tName]->Fill(7.0, nRecoedTPsForEff);
    // Plot number of TPs that are reconstructed.
    profNumTrackCands_[tName]->Fill(6.0, nRecoedTPs);
    // Plot number of TPs that are reconstructed. Count +1 for each eta sector they are reconstructed in.
    profNumTrackCands_[tName]->Fill(5.0, nEtaSecsMatchingTPs);
    // Plot number of TPs that are reconstructed. Count +1 for each (eta,phi) sector they are reconstructed in.
    profNumTrackCands_[tName]->Fill(4.0, nSecsMatchingTPs);
    // Plot number of TP that are reconstructed. Count +1 for each track they are reconstructed as.
    profNumTrackCands_[tName]->Fill(2.0, nTrksMatchingTPs);

    // Histos of track helix params.
    for (const L1track3D& trk : tracks) {
      hisQoverPt_[tName]->Fill(trk.qOverPt());
      hisPhi0_[tName]->Fill(trk.phi0());
      hisEta_[tName]->Fill(trk.eta());
      hisZ0_[tName]->Fill(trk.z0());
    }

    // Histos of track parameter resolution

    for (const TP& tp : vTPs) {
      if ((resPlotOpt_ && tp.useForAlgEff()) ||
          (not resPlotOpt_)) {  // Check TP is good for efficiency measurement (& also comes from signal event if requested)

        // For each tracking particle, find the corresponding reconstructed track(s).
        for (const L1track3D& trk : tracks) {
          const TP* tpAssoc = trk.matchedTP();
          if (tpAssoc != nullptr) {
            if (tpAssoc->index() == tp.index()) {
              hisQoverPtRes_[tName]->Fill(trk.qOverPt() - tp.qOverPt());
              hisPhi0Res_[tName]->Fill(reco::deltaPhi(trk.phi0(), tp.phi0()));
              hisEtaRes_[tName]->Fill(trk.eta() - tp.eta());
              hisZ0Res_[tName]->Fill(trk.z0() - tp.z0());
            }
          }
        }
      }
    }

    //=== Study tracking efficiency by looping over tracking particles.

    for (const TP& tp : vTPs) {
      if (tp.useForEff()) {  // Check TP is good for efficiency measurement.

        // Check if this TP was reconstructed anywhere in the tracker..
        bool tpRecoed = false;
        bool tpRecoedPerfect = false;
        for (const L1track3D& trk : tracks) {
          const TP* tpAssoc = trk.matchedTP();
          if (tpAssoc != nullptr) {
            if (tpAssoc->index() == tp.index()) {
              tpRecoed = true;
              if (trk.purity() == 1.)
                tpRecoedPerfect = true;
            }
          }
        }

        // If TP was reconstucted by HT, then plot its kinematics.
        if (tpRecoed) {
          hisRecoTPinvptForEff_[tName]->Fill(1. / tp.pt());
          hisRecoTPetaForEff_[tName]->Fill(tp.eta());
          hisRecoTPphiForEff_[tName]->Fill(tp.phi0());
          // Plot also production point of all good reconstructed TP.
          hisRecoTPd0ForEff_[tName]->Fill(std::abs(tp.d0()));
          hisRecoTPz0ForEff_[tName]->Fill(std::abs(tp.z0()));
          // Also plot efficiency to perfectly reconstruct the track (no fake hits)
          if (tpRecoedPerfect) {
            hisPerfRecoTPinvptForEff_[tName]->Fill(1. / tp.pt());
            hisPerfRecoTPetaForEff_[tName]->Fill(tp.eta());
          }
          if (tp.useForAlgEff()) {  // Check TP is good for algorithmic efficiency measurement.
            hisRecoTPinvptForAlgEff_[tName]->Fill(1. / tp.pt());
            hisRecoTPetaForAlgEff_[tName]->Fill(tp.eta());
            hisRecoTPphiForAlgEff_[tName]->Fill(tp.phi0());
            // Plot also production point of all good reconstructed TP.
            hisRecoTPd0ForAlgEff_[tName]->Fill(std::abs(tp.d0()));
            hisRecoTPz0ForAlgEff_[tName]->Fill(std::abs(tp.z0()));

            // Also plot efficiency to perfectly reconstruct the track (no fake hits)
            if (tpRecoedPerfect) {
              hisPerfRecoTPinvptForAlgEff_[tName]->Fill(1. / tp.pt());
              hisPerfRecoTPetaForAlgEff_[tName]->Fill(tp.eta());
            }
          }
        }
      }
    }
  }

  //=== Book histograms for studying track fitting.

  map<string, TFileDirectory> Histos::bookTrackFitting() {
    map<string, TFileDirectory> inputDirMap;

    for (const string& fitName : trackFitters_) {
      // Define lambda function to facilitate adding "fitName" histogram names.
      auto addn = [fitName](const string& s) { return TString::Format("%s_%s", s.c_str(), fitName.c_str()); };

      TFileDirectory inputDir = fs_->mkdir(fitName);
      inputDirMap[fitName] = inputDir;

      profNumFitTracks_[fitName] =
          inputDir.make<TProfile>(addn("NumFitTracks"), "; class; # of fitted tracks", 11, 0.5, 11.5, -0.5, 9.9e6);
      profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(7, "TP for eff fitted");
      profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(6, "TP fitted");
      profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(2, "Fit tracks that are genuine");
      profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(1, "Fit tracks including fakes");
      profNumFitTracks_[fitName]->LabelsOption("d");

      hisNumFitTrks_[fitName] =
          inputDir.make<TH1F>(addn("NumFitTrks"), "; No. fitted tracks in tracker;", 200, -0.5, 399.5);
      hisNumFitTrksPerNon_[fitName] =
          inputDir.make<TH1F>(addn("NumFitTrksPerNon"), "; No. fitted tracks per nonant;", 200, -0.5, 199.5);

      hisStubsPerFitTrack_[fitName] =
          inputDir.make<TH1F>(addn("StubsPerFitTrack"), "; No. of stubs per fitted track", 20, -0.5, 19.5);
      profStubsOnFitTracks_[fitName] = inputDir.make<TProfile>(
          addn("StubsOnFitTracks"), "; ; No. of stubs on all fitted tracks per event", 1, 0.5, 1.5);

      hisFitQinvPtMatched_[fitName] =
          inputDir.make<TH1F>(addn("FitQinvPtMatched"), "Fitted q/p_{T} for matched tracks", 120, -0.6, 0.6);
      hisFitPhi0Matched_[fitName] =
          inputDir.make<TH1F>(addn("FitPhi0Matched"), "Fitted #phi_{0} for matched tracks", 70, -3.5, 3.5);
      hisFitD0Matched_[fitName] =
          inputDir.make<TH1F>(addn("FitD0Matched"), "Fitted d_{0} for matched tracks", 100, -2., 2.);
      hisFitZ0Matched_[fitName] =
          inputDir.make<TH1F>(addn("FitZ0Matched"), "Fitted z_{0} for matched tracks", 100, -25., 25.);
      hisFitEtaMatched_[fitName] =
          inputDir.make<TH1F>(addn("FitEtaMatched"), "Fitted #eta for matched tracks", 70, -3.5, 3.5);

      hisFitQinvPtUnmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitQinvPtUnmatched"), "Fitted q/p_{T} for unmatched tracks", 120, -0.6, 0.6);
      hisFitPhi0Unmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitPhi0Unmatched"), "Fitted #phi_{0} for unmatched tracks", 70, -3.5, 3.5);
      hisFitD0Unmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitD0Unmatched"), "Fitted d_{0} for unmatched tracks", 100, -2., 2.);
      hisFitZ0Unmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitZ0Unmatched"), "Fitted z_{0} for unmatched tracks", 100, -25., 25.);
      hisFitEtaUnmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitEtaUnmatched"), "Fitted #eta for unmatched tracks", 70, -3.5, 3.5);

      const unsigned int nBinsChi2 = 39;
      const float chi2dofBins[nBinsChi2 + 1] = {0.0,  0.2,  0.4,  0.6,   0.8,   1.0,   1.2,   1.4,   1.6,   1.8,
                                                2.0,  2.4,  2.8,  3.2,   3.6,   4.0,   4.5,   5.0,   6.0,   7.0,
                                                8.0,  9.0,  10.0, 12.0,  14.0,  16.0,  18.0,  20.0,  25.0,  30.0,
                                                40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0, 1000.0};

      hisFitChi2DofRphiMatched_[fitName] =
          inputDir.make<TH1F>(addn("FitChi2DofRphiMatched"), ";#chi^{2}rphi;", nBinsChi2, chi2dofBins);
      hisFitChi2DofRzMatched_[fitName] =
          inputDir.make<TH1F>(addn("FitChi2DofRzMatched"), ";#chi^{2}rz/DOF;", nBinsChi2, chi2dofBins);
      profFitChi2DofRphiVsInvPtMatched_[fitName] =
          inputDir.make<TProfile>(addn("FitChi2DofRphiVsInvPtMatched"), "; 1/p_{T}; Fit #chi^{2}rphi/dof", 25, 0., 0.5);

      hisFitChi2DofRphiUnmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitChi2DofRphiUnmatched"), ";#chi^{2}rphi/DOF;", nBinsChi2, chi2dofBins);
      hisFitChi2DofRzUnmatched_[fitName] =
          inputDir.make<TH1F>(addn("FitChi2DofRzUnmatched"), ";#chi^{2}rz/DOF;", nBinsChi2, chi2dofBins);
      profFitChi2DofRphiVsInvPtUnmatched_[fitName] = inputDir.make<TProfile>(
          addn("FitChi2DofRphiVsInvPtUnmatched"), "; 1/p_{T}; Fit #chi^{2}rphi/dof", 25, 0., 0.5);

      // Monitoring specific track fit algorithms.
      if (fitName.find("KF") != string::npos) {
        hisKalmanNumUpdateCalls_[fitName] =
            inputDir.make<TH1F>(addn("KalmanNumUpdateCalls"), "; Calls to KF updator;", 100, -0.5, 99.5);

        hisKalmanChi2DofSkipLay0Matched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay0Matched"), ";#chi^{2} for nSkippedLayers = 0;", nBinsChi2, chi2dofBins);
        hisKalmanChi2DofSkipLay1Matched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay1Matched"), ";#chi^{2} for nSkippedLayers = 1;", nBinsChi2, chi2dofBins);
        hisKalmanChi2DofSkipLay2Matched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay2Matched"), ";#chi^{2} for nSkippedLayers = 2;", nBinsChi2, chi2dofBins);
        hisKalmanChi2DofSkipLay0Unmatched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay0Unmatched"), ";#chi^{2} for nSkippedLayers = 0;", nBinsChi2, chi2dofBins);
        hisKalmanChi2DofSkipLay1Unmatched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay1Unmatched"), ";#chi^{2} for nSkippedLayers = 1;", nBinsChi2, chi2dofBins);
        hisKalmanChi2DofSkipLay2Unmatched_[fitName] = inputDir.make<TH1F>(
            addn("KalmanChi2DofSkipLay2Unmatched"), ";#chi^{2} for nSkippedLayers = 2;", nBinsChi2, chi2dofBins);
      }

      // Plots of helix param resolution.

      hisQoverPtResVsTrueEta_[fitName] = inputDir.make<TProfile>(
          addn("QoverPtResVsTrueEta"), "q/p_{T} resolution; |#eta|; q/p_{T} resolution", 30, 0.0, 3.0);
      hisPhi0ResVsTrueEta_[fitName] = inputDir.make<TProfile>(
          addn("PhiResVsTrueEta"), "#phi_{0} resolution; |#eta|; #phi_{0} resolution", 30, 0.0, 3.0);
      hisEtaResVsTrueEta_[fitName] =
          inputDir.make<TProfile>(addn("EtaResVsTrueEta"), "#eta resolution; |#eta|; #eta resolution", 30, 0.0, 3.0);
      hisZ0ResVsTrueEta_[fitName] =
          inputDir.make<TProfile>(addn("Z0ResVsTrueEta"), "z_{0} resolution; |#eta|; z_{0} resolution", 30, 0.0, 3.0);
      hisD0ResVsTrueEta_[fitName] =
          inputDir.make<TProfile>(addn("D0ResVsTrueEta"), "d_{0} resolution; |#eta|; d_{0} resolution", 30, 0.0, 3.0);

      hisQoverPtResVsTrueInvPt_[fitName] = inputDir.make<TProfile>(
          addn("QoverPtResVsTrueInvPt"), "q/p_{T} resolution; 1/p_{T}; q/p_{T} resolution", 25, 0.0, 0.5);
      hisPhi0ResVsTrueInvPt_[fitName] = inputDir.make<TProfile>(
          addn("PhiResVsTrueInvPt"), "#phi_{0} resolution; 1/p_{T}; #phi_{0} resolution", 25, 0.0, 0.5);
      hisEtaResVsTrueInvPt_[fitName] =
          inputDir.make<TProfile>(addn("EtaResVsTrueInvPt"), "#eta resolution; 1/p_{T}; #eta resolution", 25, 0.0, 0.5);
      hisZ0ResVsTrueInvPt_[fitName] = inputDir.make<TProfile>(
          addn("Z0ResVsTrueInvPt"), "z_{0} resolution; 1/p_{T}; z_{0} resolution", 25, 0.0, 0.5);
      hisD0ResVsTrueInvPt_[fitName] = inputDir.make<TProfile>(
          addn("D0ResVsTrueInvPt"), "d_{0} resolution; 1/p_{T}; d_{0} resolution", 25, 0.0, 0.5);

      // Duplicate track histos.
      profDupFitTrksVsEta_[fitName] =
          inputDir.make<TProfile>(addn("DupFitTrksVsEta"), "; #eta; No. of duplicate tracks per TP", 12, 0., 3.);
      profDupFitTrksVsInvPt_[fitName] =
          inputDir.make<TProfile>(addn("DupFitTrksVsInvPt"), "; 1/Pt; No. of duplicate tracks per TP", 25, 0., 0.5);

      // Histos for tracking efficiency vs. TP kinematics. (Binning must match similar histos in bookTrackCands()).
      hisFitTPinvptForEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPinvptForEff"), "; TP 1/Pt of fitted tracks (for effi.);", 50, 0., 0.5);
      hisFitTPetaForEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPetaForEff"), "; TP #eta of fitted tracks (for effi.);", 20, -3., 3.);
      hisFitTPphiForEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPphiForEff"), "; TP #phi of fitted tracks (for effi.);", 20, -M_PI, M_PI);

      // Histo for efficiency to reconstruct track perfectly (no incorrect hits). (Binning must match similar histos in bookTrackCands()).
      hisPerfFitTPinvptForEff_[fitName] = inputDir.make<TH1F>(
          addn("PerfFitTPinvptForEff"), "; TP 1/Pt of fitted tracks (for perf. effi.);", 50, 0., 0.5);
      hisPerfFitTPetaForEff_[fitName] = inputDir.make<TH1F>(
          addn("PerfFitTPetaForEff"), "; TP #eta of fitted tracks (for perfect effi.);", 20, -3., 3.);

      // Histos for tracking efficiency vs. TP production point. (Binning must match similar histos in bookTrackCands()).
      hisFitTPd0ForEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPd0ForEff"), "; TP d0 of fitted tracks (for effi.);", 40, 0., 4.);
      hisFitTPz0ForEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPz0ForEff"), "; TP z0 of fitted tracks (for effi.);", 50, 0., 25.);

      // Histos for algorithmic tracking efficiency vs. TP kinematics. (Binning must match similar histos in bookTrackCands()).
      hisFitTPinvptForAlgEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPinvptForAlgEff"), "; TP 1/Pt of fitted tracks (for alg. effi.);", 50, 0., 0.5);
      hisFitTPetaForAlgEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPetaForAlgEff"), "; TP #eta of fitted tracks (for alg. effi.);", 20, -3., 3.);
      hisFitTPphiForAlgEff_[fitName] = inputDir.make<TH1F>(
          addn("FitTPphiForAlgEff"), "; TP #phi of fitted tracks (for alg. effi.);", 20, -M_PI, M_PI);

      // Histo for efficiency to reconstruct track perfectly (no incorrect hits). (Binning must match similar histos in bookTrackCands()).
      hisPerfFitTPinvptForAlgEff_[fitName] = inputDir.make<TH1F>(
          addn("PerfFitTPinvptForAlgEff"), "; TP 1/Pt of fitted tracks (for perf. alg. effi.);", 50, 0., 0.5);
      hisPerfFitTPetaForAlgEff_[fitName] =
          inputDir.make<TH1F>(addn("PerfFitTPetaForAlgEff"), "; TP #eta (for perf. alg. effi.);", 20, -3., 3.);

      // Histos for algorithmic tracking efficiency vs. TP production point. (Binning must match similar histos in bookTrackCands()).
      hisFitTPd0ForAlgEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPd0ForAlgEff"), "; TP d0 of fitted tracks (for alg. effi.);", 40, 0., 4.);
      hisFitTPz0ForAlgEff_[fitName] =
          inputDir.make<TH1F>(addn("FitTPz0ForAlgEff"), "; TP z0 of fitted tracks (for alg. effi.);", 50, 0., 25.);
    }
    return inputDirMap;
  }

  //=== Fill histograms for studying track fitting.

  void Histos::fillTrackFitting(const InputData& inputData,
                                const map<string, list<const L1fittedTrack*>>& mapFinalTracks) {
    // Allow only one thread to run this function at a time
    static std::mutex myMutex;
    std::lock_guard<std::mutex> myGuard(myMutex);

    const list<TP>& vTPs = inputData.getTPs();

    // Loop over all the fitting algorithms we are trying.
    for (const string& fitName : trackFitters_) {
      const list<const L1fittedTrack*>& fittedTracks = mapFinalTracks.at(fitName);  // Get fitted tracks.

      // Count tracks
      unsigned int nFitTracks = 0;
      unsigned int nFitsMatchingTP = 0;

      const unsigned int numPhiNonants = settings_->numPhiNonants();
      vector<unsigned int> nFitTracksPerNonant(numPhiNonants, 0);

      for (const L1fittedTrack* fitTrk : fittedTracks) {
        nFitTracks++;

        // Get matched truth particle, if any.
        const TP* tp = fitTrk->matchedTP();
        if (tp != nullptr)
          nFitsMatchingTP++;
        // Count fitted tracks per nonant.
        unsigned int iNonant = (numPhiSectors_ > 0) ? floor(fitTrk->iPhiSec() * numPhiNonants / (numPhiSectors_))
                                                    : 0;  // phi nonant number
        nFitTracksPerNonant[iNonant]++;
      }

      profNumFitTracks_[fitName]->Fill(1, nFitTracks);
      profNumFitTracks_[fitName]->Fill(2, nFitsMatchingTP);

      hisNumFitTrks_[fitName]->Fill(nFitTracks);
      for (const unsigned int& num : nFitTracksPerNonant) {
        hisNumFitTrksPerNon_[fitName]->Fill(num);
      }

      // Count stubs assigned to fitted tracks.
      unsigned int nTotStubs = 0;
      for (const L1fittedTrack* fitTrk : fittedTracks) {
        unsigned int nStubs = fitTrk->numStubs();
        hisStubsPerFitTrack_[fitName]->Fill(nStubs);
        nTotStubs += nStubs;
      }
      profStubsOnFitTracks_[fitName]->Fill(1., nTotStubs);

      // Note truth particles that are successfully fitted. And which give rise to duplicate tracks.

      map<const TP*, bool> tpRecoedMap;  // Note which truth particles were successfully fitted.
      map<const TP*, bool>
          tpPerfRecoedMap;  // Note which truth particles were successfully fitted with no incorrect hits.
      map<const TP*, unsigned int> tpRecoedDup;  // Note that this TP gave rise to duplicate tracks.
      for (const TP& tp : vTPs) {
        tpRecoedMap[&tp] = false;
        tpPerfRecoedMap[&tp] = false;
        unsigned int nMatch = 0;
        for (const L1fittedTrack* fitTrk : fittedTracks) {
          const TP* assocTP = fitTrk->matchedTP();  // Get the TP the fitted track matches to, if any.
          if (assocTP == &tp) {
            tpRecoedMap[&tp] = true;
            if (fitTrk->purity() == 1.)
              tpPerfRecoedMap[&tp] = true;
            nMatch++;
          }
        }
        tpRecoedDup[&tp] = nMatch;
      }

      // Count truth particles that are successfully fitted.

      unsigned int nFittedTPs = 0;
      unsigned int nFittedTPsForEff = 0;
      for (const TP& tp : vTPs) {
        if (tpRecoedMap[&tp]) {  // Was this truth particle successfully fitted?
          nFittedTPs++;
          if (tp.useForEff())
            nFittedTPsForEff++;
        }
      }

      profNumFitTracks_[fitName]->Fill(6, nFittedTPs);
      profNumFitTracks_[fitName]->Fill(7, nFittedTPsForEff);

      // Loop over fitted tracks again.

      for (const L1fittedTrack* fitTrk : fittedTracks) {
        // Info for specific track fit algorithms.
        unsigned int nSkippedLayers = 0;
        unsigned int numUpdateCalls = 0;
        if (fitName.find("KF") != string::npos) {
          fitTrk->infoKF(nSkippedLayers, numUpdateCalls);
          hisKalmanNumUpdateCalls_[fitName]->Fill(numUpdateCalls);
        }

        //--- Compare fitted tracks that match truth particles to those that don't.

        // Get matched truth particle, if any.
        const TP* tp = fitTrk->matchedTP();

        if (tp != nullptr) {
          hisFitQinvPtMatched_[fitName]->Fill(fitTrk->qOverPt());
          hisFitPhi0Matched_[fitName]->Fill(fitTrk->phi0());
          hisFitD0Matched_[fitName]->Fill(fitTrk->d0());
          hisFitZ0Matched_[fitName]->Fill(fitTrk->z0());
          hisFitEtaMatched_[fitName]->Fill(fitTrk->eta());

          // Only plot matched chi2 for tracks with no incorrect stubs.
          if (fitTrk->purity() == 1.) {
            hisFitChi2DofRphiMatched_[fitName]->Fill(fitTrk->chi2rphi() / fitTrk->numDOFrphi());
            hisFitChi2DofRzMatched_[fitName]->Fill(fitTrk->chi2rz() / fitTrk->numDOFrz());
            profFitChi2DofRphiVsInvPtMatched_[fitName]->Fill(std::abs(fitTrk->qOverPt()),
                                                             (fitTrk->chi2rphi() / fitTrk->numDOFrphi()));

            if (fitName.find("KF") != string::npos) {
              // No. of skipped layers on track during Kalman track fit.
              if (nSkippedLayers == 0) {
                hisKalmanChi2DofSkipLay0Matched_[fitName]->Fill(fitTrk->chi2dof());
              } else if (nSkippedLayers == 1) {
                hisKalmanChi2DofSkipLay1Matched_[fitName]->Fill(fitTrk->chi2dof());
              } else if (nSkippedLayers >= 2) {
                hisKalmanChi2DofSkipLay2Matched_[fitName]->Fill(fitTrk->chi2dof());
              }
            }
          }

        } else {
          hisFitQinvPtUnmatched_[fitName]->Fill(fitTrk->qOverPt());
          hisFitPhi0Unmatched_[fitName]->Fill(fitTrk->phi0());
          hisFitD0Unmatched_[fitName]->Fill(fitTrk->d0());
          hisFitZ0Unmatched_[fitName]->Fill(fitTrk->z0());
          hisFitEtaUnmatched_[fitName]->Fill(fitTrk->eta());

          hisFitChi2DofRphiUnmatched_[fitName]->Fill(fitTrk->chi2rphi() / fitTrk->numDOFrphi());
          hisFitChi2DofRzUnmatched_[fitName]->Fill(fitTrk->chi2rz() / fitTrk->numDOFrz());
          profFitChi2DofRphiVsInvPtUnmatched_[fitName]->Fill(std::abs(fitTrk->qOverPt()),
                                                             (fitTrk->chi2rphi() / fitTrk->numDOFrphi()));

          if (fitName.find("KF") != string::npos) {
            // No. of skipped layers on track during Kalman track fit.
            if (nSkippedLayers == 0) {
              hisKalmanChi2DofSkipLay0Unmatched_[fitName]->Fill(fitTrk->chi2dof());
            } else if (nSkippedLayers == 1) {
              hisKalmanChi2DofSkipLay1Unmatched_[fitName]->Fill(fitTrk->chi2dof());
            } else if (nSkippedLayers >= 2) {
              hisKalmanChi2DofSkipLay2Unmatched_[fitName]->Fill(fitTrk->chi2dof());
            }
          }
        }
      }

      // Study helix param resolution.

      for (const L1fittedTrack* fitTrk : fittedTracks) {
        const TP* tp = fitTrk->matchedTP();
        if (tp != nullptr) {
          // IRT
          if ((resPlotOpt_ && tp->useForAlgEff()) ||
              (not resPlotOpt_)) {  // Check TP is good for efficiency measurement (& also comes from signal event if requested)

            // Plot helix parameter resolution against eta or Pt.
            hisQoverPtResVsTrueEta_[fitName]->Fill(std::abs(tp->eta()), std::abs(fitTrk->qOverPt() - tp->qOverPt()));
            hisPhi0ResVsTrueEta_[fitName]->Fill(std::abs(tp->eta()),
                                                std::abs(reco::deltaPhi(fitTrk->phi0(), tp->phi0())));
            hisEtaResVsTrueEta_[fitName]->Fill(std::abs(tp->eta()), std::abs(fitTrk->eta() - tp->eta()));
            hisZ0ResVsTrueEta_[fitName]->Fill(std::abs(tp->eta()), std::abs(fitTrk->z0() - tp->z0()));
            hisD0ResVsTrueEta_[fitName]->Fill(std::abs(tp->eta()), std::abs(fitTrk->d0() - tp->d0()));

            hisQoverPtResVsTrueInvPt_[fitName]->Fill(std::abs(tp->qOverPt()),
                                                     std::abs(fitTrk->qOverPt() - tp->qOverPt()));
            hisPhi0ResVsTrueInvPt_[fitName]->Fill(std::abs(tp->qOverPt()),
                                                  std::abs(reco::deltaPhi(fitTrk->phi0(), tp->phi0())));
            hisEtaResVsTrueInvPt_[fitName]->Fill(std::abs(tp->qOverPt()), std::abs(fitTrk->eta() - tp->eta()));
            hisZ0ResVsTrueInvPt_[fitName]->Fill(std::abs(tp->qOverPt()), std::abs(fitTrk->z0() - tp->z0()));
            hisD0ResVsTrueInvPt_[fitName]->Fill(std::abs(tp->qOverPt()), std::abs(fitTrk->d0() - tp->d0()));
          }
        }
      }

      //=== Study duplicate tracks.

      for (const TP& tp : vTPs) {
        if (tpRecoedMap[&tp]) {  // Was this truth particle successfully fitted?
          profDupFitTrksVsEta_[fitName]->Fill(std::abs(tp.eta()), tpRecoedDup[&tp] - 1);
          profDupFitTrksVsInvPt_[fitName]->Fill(std::abs(tp.qOverPt()), tpRecoedDup[&tp] - 1);
        }
      }

      //=== Study tracking efficiency by looping over tracking particles.

      for (const TP& tp : vTPs) {
        if (tp.useForEff()) {  // Check TP is good for efficiency measurement.

          // If TP was reconstucted by HT, then plot its kinematics.
          if (tpRecoedMap[&tp]) {  // This truth particle was successfully fitted.
            hisFitTPinvptForEff_[fitName]->Fill(1. / tp.pt());
            hisFitTPetaForEff_[fitName]->Fill(tp.eta());
            hisFitTPphiForEff_[fitName]->Fill(tp.phi0());
            // Plot also production point of all good reconstructed TP.
            hisFitTPd0ForEff_[fitName]->Fill(std::abs(tp.d0()));
            hisFitTPz0ForEff_[fitName]->Fill(std::abs(tp.z0()));
            // Also plot efficiency to perfectly reconstruct the track (no fake hits)
            if (tpPerfRecoedMap[&tp]) {  // This truth particle was successfully fitted with no incorrect hits.
              hisPerfFitTPinvptForEff_[fitName]->Fill(1. / tp.pt());
              hisPerfFitTPetaForEff_[fitName]->Fill(tp.eta());
            }
            if (tp.useForAlgEff()) {  // Check TP is good for algorithmic efficiency measurement.
              hisFitTPinvptForAlgEff_[fitName]->Fill(1. / tp.pt());
              hisFitTPetaForAlgEff_[fitName]->Fill(tp.eta());
              hisFitTPphiForAlgEff_[fitName]->Fill(tp.phi0());
              // Plot also production point of all good reconstructed TP.
              hisFitTPd0ForAlgEff_[fitName]->Fill(std::abs(tp.d0()));
              hisFitTPz0ForAlgEff_[fitName]->Fill(std::abs(tp.z0()));
              // Also plot efficiency to perfectly reconstruct the track (no fake hits)
              if (tpPerfRecoedMap[&tp]) {
                hisPerfFitTPinvptForAlgEff_[fitName]->Fill(1. / tp.pt());
                hisPerfFitTPetaForAlgEff_[fitName]->Fill(tp.eta());
              }
            }
          }
        }
      }
    }
  }

  //=== Produce plots of tracking efficiency after HT or after r-z track filter (run at end of job).

  TFileDirectory Histos::plotTrackEfficiency(const string& tName) {
    // Define lambda function to facilitate adding "tName" to directory & histogram names.
    auto addn = [tName](const string& s) { return TString::Format("%s_%s", s.c_str(), tName.c_str()); };

    TFileDirectory inputDir = fs_->mkdir(addn("Effi").Data());
    // Plot tracking efficiency
    makeEfficiencyPlot(inputDir,
                       teffEffVsInvPt_[tName],
                       hisRecoTPinvptForEff_[tName],
                       hisTPinvptForEff_,
                       addn("EffVsInvPt"),
                       "; 1/Pt; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffEffVsEta_[tName],
                       hisRecoTPetaForEff_[tName],
                       hisTPetaForEff_,
                       addn("EffVsEta"),
                       "; #eta; Tracking efficiency");

    makeEfficiencyPlot(inputDir,
                       teffEffVsPhi_[tName],
                       hisRecoTPphiForEff_[tName],
                       hisTPphiForEff_,
                       addn("EffVsPhi"),
                       "; #phi; Tracking efficiency");

    makeEfficiencyPlot(inputDir,
                       teffEffVsD0_[tName],
                       hisRecoTPd0ForEff_[tName],
                       hisTPd0ForEff_,
                       addn("EffVsD0"),
                       "; d0 (cm); Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffEffVsZ0_[tName],
                       hisRecoTPz0ForEff_[tName],
                       hisTPz0ForEff_,
                       addn("EffVsZ0"),
                       "; z0 (cm); Tracking efficiency");

    // Also plot efficiency to reconstruct track perfectly.
    makeEfficiencyPlot(inputDir,
                       teffPerfEffVsInvPt_[tName],
                       hisPerfRecoTPinvptForEff_[tName],
                       hisTPinvptForEff_,
                       addn("PerfEffVsInvPt"),
                       "; 1/Pt; Tracking perfect efficiency");
    makeEfficiencyPlot(inputDir,
                       teffPerfEffVsEta_[tName],
                       hisPerfRecoTPetaForEff_[tName],
                       hisTPetaForEff_,
                       addn("PerfEffVsEta"),
                       "; #eta; Tracking perfect efficiency");

    // Plot algorithmic tracking efficiency
    makeEfficiencyPlot(inputDir,
                       teffAlgEffVsInvPt_[tName],
                       hisRecoTPinvptForAlgEff_[tName],
                       hisTPinvptForAlgEff_,
                       addn("AlgEffVsInvPt"),
                       "; 1/Pt; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffVsEta_[tName],
                       hisRecoTPetaForAlgEff_[tName],
                       hisTPetaForAlgEff_,
                       addn("AlgEffVsEta"),
                       "; #eta; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffVsPhi_[tName],
                       hisRecoTPphiForAlgEff_[tName],
                       hisTPphiForAlgEff_,
                       addn("AlgEffVsPhi"),
                       "; #phi; Tracking efficiency");

    makeEfficiencyPlot(inputDir,
                       teffAlgEffVsD0_[tName],
                       hisRecoTPd0ForAlgEff_[tName],
                       hisTPd0ForAlgEff_,
                       addn("AlgEffVsD0"),
                       "; d0 (cm); Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffVsZ0_[tName],
                       hisRecoTPz0ForAlgEff_[tName],
                       hisTPz0ForAlgEff_,
                       addn("AlgEffVsZ0"),
                       "; z0 (cm); Tracking efficiency");

    // Also plot algorithmic efficiency to reconstruct track perfectly.
    makeEfficiencyPlot(inputDir,
                       teffPerfAlgEffVsInvPt_[tName],
                       hisPerfRecoTPinvptForAlgEff_[tName],
                       hisTPinvptForAlgEff_,
                       addn("PerfAlgEffVsInvPt"),
                       "; 1/Pt; Tracking perfect efficiency");
    makeEfficiencyPlot(inputDir,
                       teffPerfAlgEffVsEta_[tName],
                       hisPerfRecoTPetaForAlgEff_[tName],
                       hisTPetaForAlgEff_,
                       addn("PerfAlgEffVsEta"),
                       "; #eta; Tracking perfect efficiency");

    return inputDir;
  }

  //=== Produce plots of tracking efficiency after track fit (run at end of job).

  TFileDirectory Histos::plotTrackEffAfterFit(const string& fitName) {
    // Define lambda function to facilitate adding "fitName" to directory & histogram names.
    auto addn = [fitName](const string& s) { return TString::Format("%s_%s", s.c_str(), fitName.c_str()); };

    TFileDirectory inputDir = fs_->mkdir(addn("Effi").Data());
    // Plot tracking efficiency
    makeEfficiencyPlot(inputDir,
                       teffEffFitVsInvPt_[fitName],
                       hisFitTPinvptForEff_[fitName],
                       hisTPinvptForEff_,
                       addn("EffFitVsInvPt"),
                       "; 1/Pt; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffEffFitVsEta_[fitName],
                       hisFitTPetaForEff_[fitName],
                       hisTPetaForEff_,
                       addn("EffFitVsEta"),
                       "; #eta; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffEffFitVsPhi_[fitName],
                       hisFitTPphiForEff_[fitName],
                       hisTPphiForEff_,
                       addn("EffFitVsPhi"),
                       "; #phi; Tracking efficiency");

    makeEfficiencyPlot(inputDir,
                       teffEffFitVsD0_[fitName],
                       hisFitTPd0ForEff_[fitName],
                       hisTPd0ForEff_,
                       addn("EffFitVsD0"),
                       "; d0 (cm); Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffEffFitVsZ0_[fitName],
                       hisFitTPz0ForEff_[fitName],
                       hisTPz0ForEff_,
                       addn("EffFitVsZ0"),
                       "; z0 (cm); Tracking efficiency");

    // Also plot efficiency to reconstruct track perfectly.
    makeEfficiencyPlot(inputDir,
                       teffPerfEffFitVsInvPt_[fitName],
                       hisPerfFitTPinvptForEff_[fitName],
                       hisTPinvptForEff_,
                       addn("PerfEffFitVsInvPt"),
                       "; 1/Pt; Tracking perfect efficiency");
    makeEfficiencyPlot(inputDir,
                       teffPerfEffFitVsEta_[fitName],
                       hisPerfFitTPetaForEff_[fitName],
                       hisTPetaForEff_,
                       addn("PerfEffFitVsEta"),
                       "; #eta; Tracking perfect efficiency");

    // Plot algorithmic tracking efficiency
    makeEfficiencyPlot(inputDir,
                       teffAlgEffFitVsInvPt_[fitName],
                       hisFitTPinvptForAlgEff_[fitName],
                       hisTPinvptForAlgEff_,
                       addn("AlgEffFitVsInvPt"),
                       "; 1/Pt; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffFitVsEta_[fitName],
                       hisFitTPetaForAlgEff_[fitName],
                       hisTPetaForAlgEff_,
                       addn("AlgEffFitVsEta"),
                       "; #eta; Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffFitVsPhi_[fitName],
                       hisFitTPphiForAlgEff_[fitName],
                       hisTPphiForAlgEff_,
                       addn("AlgEffFitVsPhi"),
                       "; #phi; Tracking efficiency");

    makeEfficiencyPlot(inputDir,
                       teffAlgEffFitVsD0_[fitName],
                       hisFitTPd0ForAlgEff_[fitName],
                       hisTPd0ForAlgEff_,
                       addn("AlgEffFitVsD0"),
                       "; d0 (cm); Tracking efficiency");
    makeEfficiencyPlot(inputDir,
                       teffAlgEffFitVsZ0_[fitName],
                       hisFitTPz0ForAlgEff_[fitName],
                       hisTPz0ForAlgEff_,
                       addn("AlgEffFitVsZ0"),
                       "; z0 (cm); Tracking efficiency");

    // Also plot algorithmic efficiency to reconstruct track perfectly.
    makeEfficiencyPlot(inputDir,
                       teffPerfAlgEffFitVsInvPt_[fitName],
                       hisPerfFitTPinvptForAlgEff_[fitName],
                       hisTPinvptForAlgEff_,
                       addn("PerfAlgEffFitVsInvPt"),
                       "; 1/Pt; Tracking perfect efficiency");
    makeEfficiencyPlot(inputDir,
                       teffPerfAlgEffFitVsEta_[fitName],
                       hisPerfFitTPetaForAlgEff_[fitName],
                       hisTPetaForAlgEff_,
                       addn("Perf AlgEffFitVsEta"),
                       "; #eta; Tracking perfect efficiency");
    return inputDir;
  }

  void Histos::makeEfficiencyPlot(
      TFileDirectory& inputDir, TEfficiency* outputEfficiency, TH1F* pass, TH1F* all, TString name, TString title) {
    outputEfficiency = inputDir.make<TEfficiency>(*pass, *all);
    outputEfficiency->SetName(name);
    outputEfficiency->SetTitle(title);
  }

  //=== Print summary of track-finding performance after track pattern reco.

  void Histos::printTrackPerformance(const string& tName) {
    float numTrackCands = profNumTrackCands_[tName]->GetBinContent(1);   // No. of track cands
    float numTrackCandsErr = profNumTrackCands_[tName]->GetBinError(1);  // No. of track cands uncertainty
    float numMatchedTrackCandsIncDups =
        profNumTrackCands_[tName]->GetBinContent(2);  // Ditto, counting only those matched to TP
    float numMatchedTrackCandsExcDups = profNumTrackCands_[tName]->GetBinContent(6);  // Ditto, but excluding duplicates
    float numFakeTracks = numTrackCands - numMatchedTrackCandsIncDups;
    float numExtraDupTracks = numMatchedTrackCandsIncDups - numMatchedTrackCandsExcDups;
    float fracFake = numFakeTracks / (numTrackCands + 1.0e-6);
    float fracDup = numExtraDupTracks / (numTrackCands + 1.0e-6);

    float numStubsOnTracks = profStubsOnTracks_[tName]->GetBinContent(1);
    float meanStubsPerTrack =
        numStubsOnTracks / (numTrackCands + 1.0e-6);  //protection against demoninator equals zero.
    unsigned int numRecoTPforAlg = hisRecoTPinvptForAlgEff_[tName]->GetEntries();
    // Histograms of input truth particles (e.g. hisTPinvptForAlgEff_), used for denominator of efficiencies, are identical,
    // irrespective of whether made after HT or after r-z track filter, so always use the former.
    unsigned int numTPforAlg = hisTPinvptForAlgEff_->GetEntries();
    unsigned int numPerfRecoTPforAlg = hisPerfRecoTPinvptForAlgEff_[tName]->GetEntries();
    float algEff = float(numRecoTPforAlg) / (numTPforAlg + 1.0e-6);  //protection against demoninator equals zero.
    float algEffErr = sqrt(algEff * (1 - algEff) / (numTPforAlg + 1.0e-6));  // uncertainty
    float algPerfEff =
        float(numPerfRecoTPforAlg) / (numTPforAlg + 1.0e-6);  //protection against demoninator equals zero.
    float algPerfEffErr = sqrt(algPerfEff * (1 - algPerfEff) / (numTPforAlg + 1.0e-6));  // uncertainty

    PrintL1trk() << "=========================================================================";
    if (tName == "HT") {
      PrintL1trk() << "               TRACK-FINDING SUMMARY AFTER HOUGH TRANSFORM             ";
    } else if (tName == "RZ") {
      PrintL1trk() << "               TRACK-FINDING SUMMARY AFTER R-Z TRACK FILTER            ";
    } else if (tName == "TRACKLET") {
      PrintL1trk() << "               TRACK-FINDING SUMMARY AFTER TRACKLET PATTERN RECO       ";
    }
    PrintL1trk() << "Number of track candidates found per event = " << numTrackCands << " +- " << numTrackCandsErr;
    PrintL1trk() << "                     with mean stubs/track = " << meanStubsPerTrack;
    PrintL1trk() << "Fraction of track cands that are fake = " << fracFake;
    PrintL1trk() << "Fraction of track cands that are genuine, but extra duplicates = " << fracDup;

    PrintL1trk() << std::fixed << std::setprecision(4) << "Algorithmic tracking efficiency = " << numRecoTPforAlg << "/"
                 << numTPforAlg << " = " << algEff << " +- " << algEffErr;
    PrintL1trk() << "Perfect algorithmic tracking efficiency = " << numPerfRecoTPforAlg << "/" << numTPforAlg << " = "
                 << algPerfEff << " +- " << algPerfEffErr << " (no incorrect hits)";
  }

  //=== Print summary of track-finding performance after helix fit for given track fitter.

  void Histos::printFitTrackPerformance(const string& fitName) {
    float numFitTracks = profNumFitTracks_[fitName]->GetBinContent(1);   // No. of track cands
    float numFitTracksErr = profNumFitTracks_[fitName]->GetBinError(1);  // No. of track cands uncertainty
    float numMatchedFitTracksIncDups =
        profNumFitTracks_[fitName]->GetBinContent(2);  // Ditto, counting only those matched to TP
    float numMatchedFitTracksExcDups = profNumFitTracks_[fitName]->GetBinContent(6);  // Ditto, but excluding duplicates
    float numFakeFitTracks = numFitTracks - numMatchedFitTracksIncDups;
    float numExtraDupFitTracks = numMatchedFitTracksIncDups - numMatchedFitTracksExcDups;
    float fracFakeFit = numFakeFitTracks / (numFitTracks + 1.0e-6);
    float fracDupFit = numExtraDupFitTracks / (numFitTracks + 1.0e-6);

    float numStubsOnFitTracks = profStubsOnFitTracks_[fitName]->GetBinContent(1);
    float meanStubsPerFitTrack =
        numStubsOnFitTracks / (numFitTracks + 1.0e-6);  //protection against demoninator equals zero.
    unsigned int numFitTPforAlg = hisFitTPinvptForAlgEff_[fitName]->GetEntries();
    // Histograms of input truth particles (e.g. hisTPinvptForAlgEff_), used for denominator of efficiencies, are identical,
    // irrespective of whether made after HT or after r-z track filter, so always use the former.
    unsigned int numTPforAlg = hisTPinvptForAlgEff_->GetEntries();
    unsigned int numPerfFitTPforAlg = hisPerfFitTPinvptForAlgEff_[fitName]->GetEntries();
    float fitEff = float(numFitTPforAlg) / (numTPforAlg + 1.0e-6);  //protection against demoninator equals zero.
    float fitEffErr = sqrt(fitEff * (1 - fitEff) / (numTPforAlg + 1.0e-6));  // uncertainty
    float fitPerfEff =
        float(numPerfFitTPforAlg) / (numTPforAlg + 1.0e-6);  //protection against demoninator equals zero.
    float fitPerfEffErr = sqrt(fitPerfEff * (1 - fitPerfEff) / (numTPforAlg + 1.0e-6));  // uncertainty

    // Does this fitter require r-z track filter to be run before it?
    bool useRZfilt = (std::count(useRZfilter_.begin(), useRZfilter_.end(), fitName) > 0);

    PrintL1trk() << "=========================================================================";
    PrintL1trk() << "                    TRACK FIT SUMMARY FOR: " << fitName;
    PrintL1trk() << "Number of fitted track candidates found per event = " << numFitTracks << " +- " << numFitTracksErr;
    PrintL1trk() << "                     with mean stubs/track = " << meanStubsPerFitTrack;
    PrintL1trk() << "Fraction of fitted tracks that are fake = " << fracFakeFit;
    PrintL1trk() << "Fraction of fitted tracks that are genuine, but extra duplicates = " << fracDupFit;
    PrintL1trk() << "Algorithmic fitting efficiency = " << numFitTPforAlg << "/" << numTPforAlg << " = " << fitEff
                 << " +- " << fitEffErr;
    PrintL1trk() << "Perfect algorithmic fitting efficiency = " << numPerfFitTPforAlg << "/" << numTPforAlg << " = "
                 << fitPerfEff << " +- " << fitPerfEffErr << " (no incorrect hits)";
    if (useRZfilt)
      PrintL1trk() << "(The above fitter used the '" << settings_->rzFilterName() << "' r-z track filter.)";
  }

  //=== Print tracking performance summary & make tracking efficiency histograms.

  void Histos::endJobAnalysis(const HTrphi::ErrorMonitor* htRphiErrMon) {
    // Don't bother producing summary if user didn't request histograms via TFileService in their cfg.
    if (not this->enabled())
      return;

    // Protection when running in wierd mixed hybrid-TMTT modes.
    bool wierdMixedMode = (hisRecoTPinvptForEff_.find("TRACKLET") == hisRecoTPinvptForEff_.end());

    if (settings_->hybrid() && not wierdMixedMode) {
      // Produce plots of tracking efficieny after tracklet pattern reco.
      this->plotTrackletSeedEfficiency();
      this->plotTrackEfficiency("TRACKLET");
      this->plotHybridDupRemovalEfficiency();

    } else {
      // Produce plots of tracking efficiency using track candidates found after HT.
      this->plotTrackEfficiency("HT");

      // Optionally produce plots of tracking efficiency using track candidates found after r-z track filter.
      if (ranRZfilter_)
        this->plotTrackEfficiency("RZ");
    }

    // Produce more plots of tracking efficiency using track candidates after track fit.
    for (auto& fitName : trackFitters_) {
      this->plotTrackEffAfterFit(fitName);
    }

    PrintL1trk() << "=========================================================================";

    // Print r (z) range in which each barrel layer (endcap wheel) appears.
    // (Needed by firmware).
    PrintL1trk();
    PrintL1trk() << "--- r range in which stubs in each barrel layer appear ---";
    for (const auto& p : mapBarrelLayerMinR_) {
      unsigned int layer = p.first;
      PrintL1trk() << "   layer = " << layer << " : " << mapBarrelLayerMinR_[layer] << " < r < "
                   << mapBarrelLayerMaxR_[layer];
    }
    PrintL1trk() << "--- |z| range in which stubs in each endcap wheel appear ---";
    for (const auto& p : mapEndcapWheelMinZ_) {
      unsigned int layer = p.first;
      PrintL1trk() << "   wheel = " << layer << " : " << mapEndcapWheelMinZ_[layer] << " < |z| < "
                   << mapEndcapWheelMaxZ_[layer];
    }

    // Print (r,|z|) range in which each module type (defined in DigitalStub) appears.
    // (Needed by firmware).
    PrintL1trk();
    PrintL1trk() << "--- (r,|z|) range in which each module type (defined in DigitalStub) appears ---";
    for (const auto& p : mapModuleTypeMinR_) {
      unsigned int modType = p.first;
      PrintL1trk() << "   Module type = " << modType << setprecision(1) << " : r range = ("
                   << mapModuleTypeMinR_[modType] << "," << mapModuleTypeMaxR_[modType] << "); z range = ("
                   << mapModuleTypeMinZ_[modType] << "," << mapModuleTypeMaxZ_[modType] << ")";
    }
    // Ugly bodge to allow for modules in barrel layers 1-2 & endcap wheels 3-5 being different.
    PrintL1trk() << "and in addition";
    for (const auto& p : mapExtraAModuleTypeMinR_) {
      unsigned int modType = p.first;
      PrintL1trk() << "   Module type = " << modType << setprecision(1) << " : r range = ("
                   << mapExtraAModuleTypeMinR_[modType] << "," << mapExtraAModuleTypeMaxR_[modType] << "); z range = ("
                   << mapExtraAModuleTypeMinZ_[modType] << "," << mapExtraAModuleTypeMaxZ_[modType] << ")";
    }
    PrintL1trk() << "and in addition";
    for (const auto& p : mapExtraBModuleTypeMinR_) {
      unsigned int modType = p.first;
      PrintL1trk() << "   Module type = " << modType << setprecision(1) << " : r range = ("
                   << mapExtraBModuleTypeMinR_[modType] << "," << mapExtraBModuleTypeMaxR_[modType] << "); z range = ("
                   << mapExtraBModuleTypeMinZ_[modType] << "," << mapExtraBModuleTypeMaxZ_[modType] << ")";
    }
    PrintL1trk() << "and in addition";
    for (const auto& p : mapExtraCModuleTypeMinR_) {
      unsigned int modType = p.first;
      PrintL1trk() << "   Module type = " << modType << setprecision(1) << " : r range = ("
                   << mapExtraCModuleTypeMinR_[modType] << "," << mapExtraCModuleTypeMaxR_[modType] << "); z range = ("
                   << mapExtraCModuleTypeMinZ_[modType] << "," << mapExtraCModuleTypeMaxZ_[modType] << ")";
    }
    PrintL1trk() << "and in addition";
    for (const auto& p : mapExtraDModuleTypeMinR_) {
      unsigned int modType = p.first;
      PrintL1trk() << "   Module type = " << modType << setprecision(1) << " : r range = ("
                   << mapExtraDModuleTypeMinR_[modType] << "," << mapExtraDModuleTypeMaxR_[modType] << "); z range = ("
                   << mapExtraDModuleTypeMinZ_[modType] << "," << mapExtraDModuleTypeMaxZ_[modType] << ")";
    }
    PrintL1trk();

    if (settings_->hybrid() && not wierdMixedMode) {
      //--- Print summary of tracklet pattern reco
      this->printTrackletSeedFindingPerformance();
      this->printTrackPerformance("TRACKLET");
      this->printHybridDupRemovalPerformance();
    } else {
      //--- Print summary of track-finding performance after HT
      this->printTrackPerformance("HT");
      //--- Optionally print summary of track-finding performance after r-z track filter.
      if (ranRZfilter_)
        this->printTrackPerformance("RZ");
    }

    //--- Print summary of track-finding performance after helix fit, for each track fitting algorithm used.
    for (const string& fitName : trackFitters_) {
      this->printFitTrackPerformance(fitName);
    }
    PrintL1trk() << "=========================================================================";

    if (htRphiErrMon != nullptr && not settings_->hybrid()) {
      // Check that stub filling was consistent with known limitations of HT firmware design.

      PrintL1trk() << "Max. |gradients| of stub lines in HT array is: r-phi = " << htRphiErrMon->maxLineGradient;

      if (htRphiErrMon->maxLineGradient > 1.) {
        PrintL1trk()
            << "WARNING: Line |gradient| exceeds 1, which firmware will not be able to cope with! Please adjust HT "
               "array size to avoid this.";

      } else if (htRphiErrMon->numErrorsTypeA > 0.) {
        float frac = float(htRphiErrMon->numErrorsTypeA) / float(htRphiErrMon->numErrorsNorm);
        PrintL1trk()
            << "WARNING: Despite line gradients being less than one, some fraction of HT columns have filled cells "
               "with no filled neighbours in W, SW or NW direction. Firmware will object to this! ";
        PrintL1trk() << "This fraction = " << frac << " for r-phi HT";

      } else if (htRphiErrMon->numErrorsTypeB > 0.) {
        float frac = float(htRphiErrMon->numErrorsTypeB) / float(htRphiErrMon->numErrorsNorm);
        PrintL1trk()
            << "WARNING: Despite line gradients being less than one, some fraction of HT columns recorded individual "
               "stubs being added to more than two cells! Thomas firmware will object to this! ";
        PrintL1trk() << "This fraction = " << frac << " for r-phi HT";
      }
    }

    // Check if GP B approximation cfg params are inconsistent.
    if (bApproxMistake_)
      PrintL1trk() << "\n WARNING: BApprox cfg params are inconsistent - see printout above.";

    // Restore original ROOT default cfg.
    TH1::SetDefaultSumw2(oldSumW2opt_);
  }

  //=== Determine "B" parameter, used in GP firmware to allow for tilted modules.

  void Histos::trackerGeometryAnalysis(const list<TrackerModule>& listTrackerModule) {
    // Don't bother producing summary if user didn't request histograms via TFileService in their cfg.
    if (not this->enabled())
      return;

    map<float, float> dataForGraph;
    for (const TrackerModule& trackerModule : listTrackerModule) {
      if (trackerModule.tiltedBarrel()) {
        float paramB = trackerModule.paramB();
        float zOverR = std::abs(trackerModule.minZ()) / trackerModule.minR();
        dataForGraph[paramB] = zOverR;  // Only store each unique paramB once, to get better histo weights.
      }
    }

    const int nEntries = dataForGraph.size();
    vector<float> vecParamB(nEntries);
    vector<float> vecZoverR(nEntries);
    unsigned int i = 0;
    for (const auto& p : dataForGraph) {
      vecParamB[i] = p.first;
      vecZoverR[i] = p.second;
      i++;
    }

    PrintL1trk() << "=========================================================================";
    PrintL1trk() << "--- Fit to cfg params for FPGA-friendly approximation to B parameter in GP & KF ---";
    PrintL1trk() << "--- (used to allowed for tilted barrel modules)                                 ---";

    TFileDirectory inputDir = fs_->mkdir("InputDataB");
    graphBVsZoverR_ = inputDir.make<TGraph>(nEntries, &vecZoverR[0], &vecParamB[0]);
    graphBVsZoverR_->SetNameTitle("B vs module Z/R", "; Module Z/R; B");
    graphBVsZoverR_->Fit("pol1", "q");
    TF1* fittedFunction = graphBVsZoverR_->GetFunction("pol1");
    double gradient = fittedFunction->GetParameter(1);
    double intercept = fittedFunction->GetParameter(0);
    double gradient_err = fittedFunction->GetParError(1);
    double intercept_err = fittedFunction->GetParError(0);
    PrintL1trk() << "         BApprox_gradient (fitted)  = " << gradient << " +- " << gradient_err;
    PrintL1trk() << "         BApprox_intercept (fitted) = " << intercept << " +- " << intercept_err;
    PrintL1trk() << "=========================================================================";

    // Check fitted params consistent with those assumed in cfg file.
    if (settings_->useApproxB()) {
      double gradientDiff = std::abs(gradient - settings_->bApprox_gradient());
      double interceptDiff = std::abs(intercept - settings_->bApprox_intercept());
      constexpr unsigned int nSigma = 5;
      if (gradientDiff > nSigma * gradient_err ||
          interceptDiff > nSigma * intercept_err) {  // Uncertainty independent of number of events
        PrintL1trk() << "\n WARNING: fitted parameters inconsistent with those specified in cfg file:";
        PrintL1trk() << "         BApprox_gradient  (cfg) = " << settings_->bApprox_gradient();
        PrintL1trk() << "         BApprox_intercept (cfg) = " << settings_->bApprox_intercept();
        bApproxMistake_ = true;  // Note that problem has occurred.
      }
    }
  }

}  // namespace tmtt
