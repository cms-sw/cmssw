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
#include "L1Trigger/TrackerTFP/interface/DistServer.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace trackerTFP;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerKFout
   *  \brief  Converts KF output into TFP output
   *  \author Christopher Brown
   *  \date   2021, Aug
   */
  class ProducerKFout : public stream::EDProducer<> {
  public:
    explicit ProducerKFout(const ParameterSet&);
    ~ProducerKFout() override {}
    template <typename T>
    int digitise(const vector<T> Bins, T Value, T factor = 1);

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}

    // ED input token of kf stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED input token of kf input to kf output TTTrack map
    EDGetTokenT<TTTrackRefMap> edGetTokenTTTrackRefMap_;
    // ED output token for accepted kfout tracks
    EDPutTokenT<StreamsTrack> edPutTokenAccepted_;
    // ED output token for truncated kfout tracks
    EDPutTokenT<StreamsTrack> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // Bins for dPhi/dZ use to create weight LUT
    vector<double> dPhiBins_;
    vector<double> dZBins_;

    // Constant used throughout for partial ttrack words
    int partialTrackWordBits_;

    // For convenience and keeping readable code, accessed many times
    int numWorkers_;
  };

  ProducerKFout::ProducerKFout(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& labelKF = iConfig.getParameter<string>("LabelKF");
    const string& labelAS = iConfig.getParameter<string>("LabelAS");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLost = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(labelKF, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(labelKF, branchTracks));
    edGetTokenTTTrackRefMap_ = consumes<TTTrackRefMap>(InputTag(labelAS, branchTracks));
    edPutTokenAccepted_ = produces<StreamsTrack>(branchTracks);
    edPutTokenLost_ = produces<StreamsTrack>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
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
    for (int i = 0;
         i < pow(2, dataFormats_->width(Variable::dPhi, Process::kfin)) / pow(2, setup_->weightBinFraction());
         i++)
      dPhiBins_.push_back(
          pow(dataFormats_->base(Variable::dPhi, Process::kfin) * (i + 1) * pow(2, setup_->weightBinFraction()), -2));

    for (int i = 0; i < pow(2, dataFormats_->width(Variable::dZ, Process::kfin)) / pow(2, setup_->weightBinFraction());
         i++)
      dZBins_.push_back(
          pow(dataFormats_->base(Variable::dZ, Process::kfin) * (i + 1) * pow(2, setup_->weightBinFraction()), -2));

    partialTrackWordBits_ = TTBV::S_ / 2;
    numWorkers_ = setup_->kfNumWorker();
  }

  // Helper function to convert floating chi2 to chi2 bin
  template <typename T>
  int ProducerKFout::digitise(const vector<T> Bins, T Value, T factor) {
    for (int i = 0; i < (int)Bins.size(); i++) {
      if (Value * factor > Bins[i] && Value * factor <= Bins[i + 1]) {
        return i;
      }
    }
    return -1;
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
      Handle<TTTrackRefMap> handleTTTrackRefMap;
      iEvent.getByToken<TTTrackRefMap>(edGetTokenTTTrackRefMap_, handleTTTrackRefMap);
      const TTTrackRefMap& ttTrackRefMap = *handleTTTrackRefMap.product();
      // 18 Output Links (First Vector) each has a vector of tracks per event (second vector) each track is 3 32 bit TTBV partial tracks
      vector<vector<TTBV>> SortedPartialTracks(setup_->numRegions() * setup_->tfpNumChannel(), vector<TTBV>(0));

      TrackKFOutSAPtrCollectionss InTrackStreams;
      TrackKFOutSAPtrCollectionss OutTrackStreams;

      // Setup empty collections for input tracks to distribution server
      for (int iRegion = 0; iRegion < setup_->numRegions(); iRegion++) {
        TrackKFOutSAPtrCollections temp_collection;
        for (int iLink = 0; iLink < setup_->tfpNumChannel(); iLink++) {
          TrackKFOutSAPtrCollection temp;
          for (int iTrack = 0; iTrack < setup_->numFramesIO(); iTrack++)
            temp.emplace_back(std::make_shared<TrackKFOut>());
          temp_collection.push_back(temp);
        }
        OutTrackStreams.push_back(temp_collection);
      }

      // Setup empty collections for oiutpu tracks from distribution server
      for (int iRegion = 0; iRegion < setup_->numRegions(); iRegion++) {
        TrackKFOutSAPtrCollections temp_collection;
        for (int iLink = 0; iLink < numWorkers_; iLink++) {
          TrackKFOutSAPtrCollection temp;
          for (int iTrack = 0; iTrack < setup_->numFramesIO(); iTrack++)
            temp.emplace_back(std::make_shared<TrackKFOut>());
          temp_collection.push_back(temp);
        }
        InTrackStreams.push_back(temp_collection);
      }

      StreamsTrack OutputStreamsTracks(setup_->numRegions() * setup_->tfpNumChannel());

      for (int iLink = 0; iLink < (int)streamsTracks.size(); iLink++) {
        for (int iTrack = 0; iTrack < (int)streamsTracks[iLink].size(); iTrack++) {
          const auto& track = streamsTracks[iLink].at(iTrack);
          TrackKF InTrack(track, dataFormats_);

          double temp_z0 = InTrack.zT() - ((InTrack.cot() * setup_->chosenRofZ()));

          // Correction to Phi calcuation depending if +ve/-ve phi sector
          const double baseSectorCorr = InTrack.sectorPhi() ? -setup_->baseSector() : setup_->baseSector();

          double temp_phi0 = InTrack.phiT() - ((InTrack.inv2R()) * setup_->hybridChosenRofPhi()) + baseSectorCorr;

          double temp_tanL = InTrack.cotGlobal();

          TTBV HitPattern(0, setup_->numLayers());

          double tempchi2rphi = 0;
          double tempchi2rz = 0;

          for (int iStub = 0; iStub < setup_->numLayers() - 1; iStub++) {
            const auto& stub = streamsStubs[setup_->numLayers() * iLink + iStub].at(iTrack);
            StubKF InStub(stub, dataFormats_, iStub);

            if (!stub.first.isNonnull())
              continue;

            HitPattern.set(iStub);
            double phiSquared = pow(InStub.phi(), 2);
            double zSquared = pow(InStub.z(), 2);

            double tempv0 = dPhiBins_[(InStub.dPhi() / (dataFormats_->base(Variable::dPhi, Process::kfin) *
                                                        pow(2, setup_->weightBinFraction())))];
            double tempv1 = dZBins_[(
                InStub.dZ() / (dataFormats_->base(Variable::dZ, Process::kfin) * pow(2, setup_->weightBinFraction())))];

            double tempRphi = phiSquared * tempv0;
            double tempRz = zSquared * tempv1;

            tempchi2rphi += tempRphi;
            tempchi2rz += tempRz;
          }  // Iterate over track stubs

          // TODO extract TTTrack bit widths from TTTrack word pending update to the TTTrack_word class
          TTBV TrackValid(1, 1, false);
          TTBV extraMVA(0, 6, false);
          TTBV TQMVA(0, 3, false);
          TTBV BendChi2(0, 3, false);
          TTBV Chi2rphi(
              digitise(setup_->kfoutchi2rphiBins(), tempchi2rphi, (double)setup_->kfoutchi2rphiConv()), 4, false);
          TTBV Chi2rz(digitise(setup_->kfoutchi2rzBins(), tempchi2rz, (double)setup_->kfoutchi2rzConv()), 4, false);
          TTBV D0(0, 13, false);
          TTBV z0(temp_z0, dataFormats_->base(Variable::zT, Process::kf), 12, true);
          TTBV TanL(temp_tanL, dataFormats_->base(Variable::cot, Process::kf), 16, true);
          TTBV phi0(temp_phi0, dataFormats_->base(Variable::phiT, Process::kf), 12, true);
          TTBV InvR(-InTrack.inv2R(), dataFormats_->base(Variable::inv2R, Process::kf), 16, true);
          InvR.resize(15);
          // 13   +   3   +   7        +  3       + 3
          TTBV PartialTrack3((D0 + BendChi2 + HitPattern + TQMVA + extraMVA), partialTrackWordBits_, false);
          // 16   + 12    + 4
          TTBV PartialTrack2((TanL + z0 + Chi2rz), partialTrackWordBits_, false);
          // 1        + 15   +  12 +    4
          TTBV PartialTrack1((TrackValid + InvR + phi0 + Chi2rphi), partialTrackWordBits_, false);

          int sortKey = (InTrack.sectorEta() < (int)(setup_->numSectorsEta() / 2)) ? 0 : 1;
          // Set correct bit to valid for track valid
          TrackKFOut Temp_track(
              PartialTrack1.set(31), PartialTrack2, PartialTrack3, sortKey, track, iTrack, iLink, true);

          InTrackStreams[iLink / numWorkers_][iLink % numWorkers_][iTrack] = (std::make_shared<TrackKFOut>(Temp_track));

        }  // Iterate over Tracks

        //Pad out input streams to Dist server with extra null track if odd number of tracks on a stream
        int iFinTrack = (int)streamsTracks[iLink].size();
        if (iFinTrack % numWorkers_ != 0) {
          TrackKFOut null_track(0, 0, 0, iLink % numWorkers_, tt::FrameTrack(), iFinTrack + 1, iLink, true);
          InTrackStreams[iLink / numWorkers_][iLink % numWorkers_][iFinTrack] =
              (std::make_shared<TrackKFOut>(null_track));
        }
      }  // Iterate over Links
      // Fill products and match up tracks

      // One distribution server for every region, num inputs = num KF workers, num outputs = num output links
      vector<DistServer> distServers(
          setup_->numRegions(),
          DistServer(numWorkers_, setup_->tfpNumChannel(), 2));  //Magic number for interleaving in dist server

      for (int iRegion = 0; iRegion < setup_->numRegions(); iRegion++) {
        for (int iTrack = 0; iTrack < setup_->numFramesIO() * ((double)TTBV::S_ / setup_->tttrackBits()); iTrack++) {
          TrackKFOutSAPtrCollection DistIn;
          for (int iWorker = 0; iWorker < numWorkers_; iWorker++)
            DistIn.push_back(InTrackStreams[iRegion][iWorker][iTrack]);  // Reorganise input to distribution server
          TrackKFOutSAPtrCollection DistOut = distServers[iRegion].clock(DistIn);  // Run dist server
          for (int iLink = 0; iLink < setup_->tfpNumChannel(); iLink++)
            OutTrackStreams[iRegion][iLink][iTrack] =
                DistOut[iLink];  // Reorganise output of distribution server in output streams
        }
      }

      // Pack output of distribution server onto each link, with correct partial tracks in correct places
      for (int iRegion = 0; iRegion < setup_->numRegions(); iRegion++) {
        for (int iLink = 0; iLink < setup_->tfpNumChannel(); iLink++) {
          for (int iTrack = 0; iTrack < (int)OutTrackStreams[iRegion][iLink].size(); iTrack++) {
            SortedPartialTracks[2 * iRegion + iLink].push_back(
                OutTrackStreams[iRegion][iLink][iTrack]->PartialTrack1());
            SortedPartialTracks[2 * iRegion + iLink].push_back(
                OutTrackStreams[iRegion][iLink][iTrack]->PartialTrack2());
            SortedPartialTracks[2 * iRegion + iLink].push_back(
                OutTrackStreams[iRegion][iLink][iTrack]->PartialTrack3());
            OutputStreamsTracks[2 * iRegion + iLink].emplace_back(OutTrackStreams[iRegion][iLink][iTrack]->track());
          }
        }
      }

      const TTBV NullBitTrack(0, partialTrackWordBits_, false);
      for (int iLink = 0; iLink < (int)OutputStreamsTracks.size(); iLink++) {
        // Iterate through partial tracks
        int numLinkTracks = (int)OutputStreamsTracks[iLink].size();
        if (numLinkTracks == 0)
          continue;  // Don't fill links if no tracks
        if ((numLinkTracks % 2 != 0)) {
          SortedPartialTracks[iLink].push_back(NullBitTrack);  //Pad out final set of bits
          OutputStreamsTracks[iLink].emplace_back(
              OutputStreamsTracks[iLink][numLinkTracks++]);  //Pad out with final repeated track
        }                                                    //If there is an odd number of tracks
        for (int iTrack = 0; iTrack < (int)(SortedPartialTracks[iLink].size()); iTrack++) {
          if (iTrack % 2 != 1)  // Write to links every other partial track, 3 partial tracks per full TTTrack
            continue;
          TTTrackRef TrackRef;
          for (auto& it : ttTrackRefMap) {  //Iterate through ttTrackRefMap to find TTTrackRef Key by a TTTrack Value
            if (it.second == OutputStreamsTracks[iLink][(int)(iTrack - 1) / 3].first)
              TrackRef = it.first;
          }
          if ((int)iTrack / 3 <= setup_->numFramesIO() * ((double)TTBV::S_ / setup_->tttrackBits()))
            accepted[iLink].emplace_back(
                std::make_pair(TrackRef,
                               (SortedPartialTracks[iLink][iTrack - 1].slice(partialTrackWordBits_) +
                                SortedPartialTracks[iLink][iTrack].slice(partialTrackWordBits_))
                                   .bs()));
          else
            lost[iLink].emplace_back(
                std::make_pair(TrackRef,
                               (SortedPartialTracks[iLink][iTrack - 1].slice(partialTrackWordBits_) +
                                SortedPartialTracks[iLink][iTrack].slice(partialTrackWordBits_))
                                   .bs()));
        }  //Iterate through sorted partial tracks
      }    // Iterate through links
    }      // Config Supported
    // store products
    iEvent.emplace(edPutTokenAccepted_, std::move(accepted));
    iEvent.emplace(edPutTokenLost_, std::move(lost));
  }
}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerKFout);
