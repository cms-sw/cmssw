#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackTrigger/interface/L1TrackQuality.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace trackerTFP;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerKFout
   *  \brief  Converts KF output into tttrack collection and TFP output
   *  A bit accurate emulation of the track transformation, the 
   *  eta routing and splitting of the 96-bit track words into 64-bit 
   *  packets. Also run is a bit accurate emulation of the track quality
   *  BDT, whose output is also added to the track word.
   *  \author Christopher Brown
   *  \date   2021, Aug
   *  \update 2024, June by Claire Savard
   */
  class ProducerKFout : public stream::EDProducer<> {
  public:
    explicit ProducerKFout(const ParameterSet&);
    ~ProducerKFout() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}

    // ED input token of kf stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted kfout tracks
    EDPutTokenT<StreamsTrack> edPutTokenAccepted_;
    // ED output token for TTTracks
    EDPutTokenT<TTTracks> edPutTokenTTTracks_;
    // ED output token for truncated kfout tracks
    EDPutTokenT<StreamsTrack> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_;
    // Bins for dPhi/dZ use to create weight LUT
    vector<double> dPhiBins_;
    vector<double> dZBins_;

    std::unique_ptr<L1TrackQuality> trackQualityModel_;
    double tqTanlScale_;
    double tqZ0Scale_;
    static constexpr double ap_fixed_rescale = 32.0;

    // For convenience and keeping readable code, accessed many times
    int numWorkers_;
    int partialTrackWordBits_;

    // Helper function to convert floating value to bin
    template <typename T>
    unsigned int digitise(const T& bins, double value, double factor) {
      unsigned int bin = 0;
      for (unsigned int i = 0; i < bins.size() - 1; i++) {
        if (value * factor > bins[i] && value * factor <= bins[i + 1])
          break;
        bin++;
      }
      return bin;
    }
  };

  ProducerKFout::ProducerKFout(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& labelKF = iConfig.getParameter<string>("LabelKF");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchTTTracks = iConfig.getParameter<string>("BranchAcceptedTTTracks");
    const string& branchLost = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(labelKF, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(labelKF, branchTracks));
    edPutTokenAccepted_ = produces<StreamsTrack>(branchTracks);
    edPutTokenTTTracks_ = produces<TTTracks>(branchTTTracks);
    edPutTokenLost_ = produces<StreamsTrack>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    dataFormats_ = nullptr;

    trackQualityModel_ = std::make_unique<L1TrackQuality>(iConfig.getParameter<edm::ParameterSet>("TrackQualityPSet"));
    edm::ParameterSet trackQualityPSset = iConfig.getParameter<edm::ParameterSet>("TrackQualityPSet");
    tqTanlScale_ = trackQualityPSset.getParameter<double>("tqemu_TanlScale");
    tqZ0Scale_ = trackQualityPSset.getParameter<double>("tqemu_Z0Scale");
  }

  void ProducerKFout::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);

    // Calculate 1/dz**2 and 1/dphi**2 bins for v0 and v1 weightings
    float temp_dphi = 0.0;
    float temp_dz = 0.0;
    for (int i = 0;
         i < pow(2, dataFormats_->width(Variable::dPhi, Process::kfin)) / pow(2, setup_->weightBinFraction());
         i++) {
      temp_dphi =
          pow(dataFormats_->base(Variable::dPhi, Process::kfin) * (i + 1) * pow(2, setup_->weightBinFraction()), -2);
      temp_dphi = temp_dphi / setup_->dphiTruncation();
      temp_dphi = std::floor(temp_dphi);
      dPhiBins_.push_back(temp_dphi * setup_->dphiTruncation());
    }
    for (int i = 0; i < pow(2, dataFormats_->width(Variable::dZ, Process::kfin)) / pow(2, setup_->weightBinFraction());
         i++) {
      temp_dz =
          pow(dataFormats_->base(Variable::dZ, Process::kfin) * (i + 1) * pow(2, setup_->weightBinFraction()), -2);
      temp_dz = temp_dz * setup_->dzTruncation();
      temp_dz = std::ceil(temp_dz);
      dZBins_.push_back(temp_dz / setup_->dzTruncation());
    }
    numWorkers_ = setup_->kfNumWorker();
    partialTrackWordBits_ = TTBV::S_ / 2;
  }

  void ProducerKFout::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty KFout product
    StreamsTrack accepted(setup_->numRegions() * setup_->tfpNumChannel());
    StreamsTrack lost(setup_->numRegions() * setup_->tfpNumChannel());
    // read in KF Product and produce KFout product
    if (setup_->configurationSupported()) {
      Handle<StreamsStub> handleStubs;
      iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
      const StreamsStub& streamsStubs = *handleStubs.product();
      Handle<StreamsTrack> handleTracks;
      iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
      const StreamsTrack& streamsTracks = *handleTracks.product();

      // Setup KFout track collection
      TrackKFOutSAPtrCollection KFoutTracks;

      // Setup containers for track quality
      float tempTQMVAPreSig = 0.0;
      // Due to ap_fixed implementation in CMSSW this 10,5 must be specified at compile time, TODO make this a changeable parameter
      std::vector<ap_fixed<10, 5>> trackQuality_inputs = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // calculate track quality and fill TTTracks
      TTTracks ttTracks;
      int nTracks(0);
      for (const StreamTrack& stream : streamsTracks)
        nTracks += accumulate(stream.begin(), stream.end(), 0, [](int sum, const FrameTrack& frame) {
          return sum + (frame.first.isNonnull() ? 1 : 0);
        });
      ttTracks.reserve(nTracks);
      for (int iLink = 0; iLink < (int)streamsTracks.size(); iLink++) {
        for (int iTrack = 0; iTrack < (int)streamsTracks[iLink].size(); iTrack++) {
          const auto& track = streamsTracks[iLink].at(iTrack);
          TrackKF inTrack(track, dataFormats_);

          double temp_z0 = inTrack.zT() - ((inTrack.cot() * setup_->chosenRofZ()));
          // Correction to Phi calcuation depending if +ve/-ve phi sector
          const double baseSectorCorr = inTrack.sectorPhi() ? -setup_->baseSector() : setup_->baseSector();
          double temp_phi0 = inTrack.phiT() - ((inTrack.inv2R()) * setup_->hybridChosenRofPhi()) + baseSectorCorr;
          double temp_tanL = inTrack.cotGlobal();

          TTBV hitPattern(0, setup_->numLayers());
          double tempchi2rphi = 0;
          double tempchi2rz = 0;
          int temp_nstub = 0;
          int temp_ninterior = 0;
          bool counter = false;

          vector<StubKF> stubs;
          stubs.reserve(setup_->numLayers());
          for (int iStub = 0; iStub < setup_->numLayers(); iStub++) {
            const auto& stub = streamsStubs[setup_->numLayers() * iLink + iStub].at(iTrack);
            StubKF inStub(stub, dataFormats_, iStub);
            if (stub.first.isNonnull())
              stubs.emplace_back(stub, dataFormats_, iStub);

            if (!stub.first.isNonnull()) {
              if (counter)
                temp_ninterior += 1;
              continue;
            }

            counter = true;

            hitPattern.set(iStub);
            temp_nstub += 1;
            double phiSquared = pow(inStub.phi(), 2);
            double zSquared = pow(inStub.z(), 2);

            double tempv0 = dPhiBins_[(inStub.dPhi() / (dataFormats_->base(Variable::dPhi, Process::kfin) *
                                                        pow(2, setup_->weightBinFraction())))];
            double tempv1 = dZBins_[(
                inStub.dZ() / (dataFormats_->base(Variable::dZ, Process::kfin) * pow(2, setup_->weightBinFraction())))];

            double tempRphi = phiSquared * tempv0;
            double tempRz = zSquared * tempv1;

            tempchi2rphi += tempRphi;
            tempchi2rz += tempRz;
          }  // Iterate over track stubs

          // Create bit vectors for each output, including digitisation of chi2
          // TODO implement extraMVA, bendChi2, d0
          TTBV trackValid(1, TTTrack_TrackWord::TrackBitWidths::kValidSize, false);
          TTBV extraMVA(0, TTTrack_TrackWord::TrackBitWidths::kMVAOtherSize, false);
          TTBV bendChi2(0, TTTrack_TrackWord::TrackBitWidths::kBendChi2Size, false);
          TTBV chi2rphi(digitise(TTTrack_TrackWord::chi2RPhiBins, tempchi2rphi, (double)setup_->kfoutchi2rphiConv()),
                        TTTrack_TrackWord::TrackBitWidths::kChi2RPhiSize,
                        false);
          TTBV chi2rz(digitise(TTTrack_TrackWord::chi2RZBins, tempchi2rz, (double)setup_->kfoutchi2rzConv()),
                      TTTrack_TrackWord::TrackBitWidths::kChi2RZSize,
                      false);
          TTBV d0(0, TTTrack_TrackWord::TrackBitWidths::kD0Size, false);
          TTBV z0(
              temp_z0, dataFormats_->base(Variable::zT, Process::kf), TTTrack_TrackWord::TrackBitWidths::kZ0Size, true);
          TTBV tanL(temp_tanL,
                    dataFormats_->base(Variable::cot, Process::kf),
                    TTTrack_TrackWord::TrackBitWidths::kTanlSize,
                    true);
          TTBV phi0(temp_phi0,
                    dataFormats_->base(Variable::phiT, Process::kf),
                    TTTrack_TrackWord::TrackBitWidths::kPhiSize,
                    true);
          TTBV invR(-inTrack.inv2R(),
                    dataFormats_->base(Variable::inv2R, Process::kf),
                    TTTrack_TrackWord::TrackBitWidths::kRinvSize + 1,
                    true);
          invR.resize(TTTrack_TrackWord::TrackBitWidths::kRinvSize);

          // conversion to tttrack to calculate bendchi2
          // temporary fix for MVA1 while bendchi2 not implemented
          TTTrack temp_tttrack = inTrack.ttTrack(stubs);
          double tempbendchi2 = temp_tttrack.chi2BendRed();

          // Create input vector for BDT
          trackQuality_inputs = {
              (std::trunc(tanL.val() / tqTanlScale_)) / ap_fixed_rescale,
              (std::trunc(z0.val() / tqZ0Scale_)) / ap_fixed_rescale,
              digitise(TTTrack_TrackWord::bendChi2Bins, tempbendchi2, 1.),
              temp_nstub,
              temp_ninterior,
              digitise(TTTrack_TrackWord::chi2RPhiBins, tempchi2rphi, (double)setup_->kfoutchi2rphiConv()),
              digitise(TTTrack_TrackWord::chi2RZBins, tempchi2rz, (double)setup_->kfoutchi2rzConv())};

          // Run BDT emulation and package output into 3 bits
          // output needs sigmoid transformation applied
          tempTQMVAPreSig = trackQualityModel_->runEmulatedTQ(trackQuality_inputs);
          TTBV tqMVA(digitise(L1TrackQuality::getTqMVAPreSigBins(), tempTQMVAPreSig, 1.0),
                     TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize,
                     false);

          // Build 32 bit partial tracks for outputting in 64 bit packets
          //                  12 +  3       +  7         +  3    +  6
          TTBV partialTrack3((d0 + bendChi2 + hitPattern + tqMVA + extraMVA), partialTrackWordBits_, false);
          //                  16   + 12    + 4
          TTBV partialTrack2((tanL + z0 + chi2rz), partialTrackWordBits_, false);
          //                    1        + 15   +  12 +    4
          TTBV partialTrack1((trackValid + invR + phi0 + chi2rphi), partialTrackWordBits_, false);

          int sortKey = (inTrack.sectorEta() < (int)(setup_->numSectorsEta() / 2)) ? 0 : 1;
          int nonantId = iLink / setup_->kfNumWorker();
          // Set correct bit to valid for track valid
          TrackKFOut temp_track(partialTrack1.set((partialTrackWordBits_ - 1)),
                                partialTrack2,
                                partialTrack3,
                                sortKey,
                                nonantId,
                                track,
                                iTrack,
                                iLink,
                                true);
          KFoutTracks.push_back(std::make_shared<TrackKFOut>(temp_track));

          // add MVA to tttrack and add tttrack to collection
          temp_tttrack.settrkMVA1(1. / (1. + exp(tempTQMVAPreSig)));
          temp_tttrack.setTrackWordBits();
          ttTracks.emplace_back(temp_tttrack);
        }  // Iterate over Tracks
      }  // Iterate over Links
      const OrphanHandle<tt::TTTracks> orphanHandleTTTracks = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));

      // sort partial KFout tracks into 18 separate links (nonant idx * eta idx) with tttrack ref info
      // 0th index order: [nonant 0 + negative eta, nonant 0 + positive eta, nonant 1 + negative eta, ...]
      struct kfoTrack_info {
        TTBV partialBits;
        TTTrackRef trackRef;
      };
      vector<vector<kfoTrack_info>> sortedPartialTracks(setup_->numRegions() * setup_->tfpNumChannel(),
                                                        vector<kfoTrack_info>(0));
      for (int i = 0; i < (int)KFoutTracks.size(); i++) {
        auto& kfoTrack = KFoutTracks.at(i);
        if (kfoTrack->dataValid()) {
          sortedPartialTracks[kfoTrack->nonantId() * setup_->tfpNumChannel() + kfoTrack->sortKey()].push_back(
              {kfoTrack->PartialTrack1(), TTTrackRef(orphanHandleTTTracks, i)});
          sortedPartialTracks[kfoTrack->nonantId() * setup_->tfpNumChannel() + kfoTrack->sortKey()].push_back(
              {kfoTrack->PartialTrack2(), TTTrackRef(orphanHandleTTTracks, i)});
          sortedPartialTracks[kfoTrack->nonantId() * setup_->tfpNumChannel() + kfoTrack->sortKey()].push_back(
              {kfoTrack->PartialTrack3(), TTTrackRef(orphanHandleTTTracks, i)});
        }
      }
      // fill remaining tracks allowed on each link (setup_->numFramesIO()) with null info
      kfoTrack_info nullTrack_info;
      for (int i = 0; i < (int)sortedPartialTracks.size(); i++) {
        // will not fill if any additional tracks if already above limit
        while ((int)sortedPartialTracks.at(i).size() < setup_->numFramesIO() * 2)
          sortedPartialTracks.at(i).push_back(nullTrack_info);
      }

      // combine sorted partial tracks into proper format:
      // < TTTrackRef A, first 64 A bits >
      // < TTTrackRef B, last 32 A bits + first 32 B bits >
      // < TTTrackRef null, last 64 B bits >
      // ... repeat for next tracks
      const TTBV nullPartialBits(0, partialTrackWordBits_, false);
      const TTTrackRef nullTrackRef;
      int partialFactor = TTBV::S_ / partialTrackWordBits_;  //how many partial track words to combine in an output
      for (int iLink = 0; iLink < (int)sortedPartialTracks.size(); iLink++) {
        for (int iTrack = 0; iTrack < (int)sortedPartialTracks[iLink].size(); iTrack += partialFactor) {
          // if a partial track has no pair, pair it with null partial track
          if (iTrack + 1 == (int)sortedPartialTracks[iLink].size())
            sortedPartialTracks[iLink].push_back({nullPartialBits, nullTrackRef});
          // keep TTTrackRef null every third (96 bits / 32 partial bits) output packet
          TTTrackRef fillTrackRef;
          if ((iTrack / partialFactor + 1) % (TTTrack_TrackWord::kTrackWordSize / partialTrackWordBits_) != 0)
            fillTrackRef = sortedPartialTracks[iLink][iTrack + 1].trackRef;

          // if there are too many output packets, truncate and put remaining outputs in lost collection
          if (iTrack / partialFactor < setup_->numFramesIO())
            accepted[iLink].emplace_back(
                std::make_pair(fillTrackRef,
                               (sortedPartialTracks[iLink][iTrack].partialBits.slice(partialTrackWordBits_) +
                                sortedPartialTracks[iLink][iTrack + 1].partialBits.slice(partialTrackWordBits_))
                                   .bs()));
          else
            lost[iLink].emplace_back(
                std::make_pair(fillTrackRef,
                               (sortedPartialTracks[iLink][iTrack].partialBits.slice(partialTrackWordBits_) +
                                sortedPartialTracks[iLink][iTrack + 1].partialBits.slice(partialTrackWordBits_))
                                   .bs()));
        }
      }
    }  // Config Supported

    // store products
    iEvent.emplace(edPutTokenAccepted_, std::move(accepted));
    iEvent.emplace(edPutTokenLost_, std::move(lost));
  }
}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerKFout);
