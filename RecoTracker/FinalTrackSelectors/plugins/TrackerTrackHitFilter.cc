#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

//for S/N cut
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
//for angle cut
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TMath.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <boost/regex.hpp>
#include <map>
#include <optional>
//#include <math>

/**
 * Configurables:
 *
 *   Generic:
 *     tracks                  = InputTag of a collection of tracks to read from
 *     minimumHits             = Minimum hits that the output TrackCandidate must have to be saved
 *     replaceWithInactiveHits = instead of discarding hits, replace them with a invalid "inactive" hits,
 *                               so multiple scattering is accounted for correctly.
 *     stripFrontInvalidHits   = strip invalid hits at the beginning of the track
 *     stripBackInvalidHits    = strip invalid hits at the end of the track
 *     stripAllInvalidHits     = remove ALL invald hits (might be a problem for multiple scattering, use with care!)
 *
 *   Per structure:
 *      commands = list of commands, to be applied in order as they are written
 *      commands can be:
 *          "keep XYZ"  , "drop XYZ"    (XYZ = PXB, PXE, TIB, TID, TOB, TEC)
 *          "keep XYZ l", "drop XYZ n"  (XYZ as above, n = layer, wheel or disk = 1 .. 6 ; positive and negative are the same )
 *
 *   Individual modules:
 *     detsToIgnore        = individual list of detids on which hits must be discarded
 */
namespace reco {

  namespace modules {
    class TrackerTrackHitFilter : public edm::stream::EDProducer<> {
    public:
      TrackerTrackHitFilter(const edm::ParameterSet &iConfig);
      void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;
      int checkHit(const edm::EventSetup &iSetup, const DetId &detid, const TrackingRecHit *hit);
      void produceFromTrajectory(const edm::EventSetup &iSetup,
                                 const Trajectory *itt,
                                 std::vector<TrackingRecHit *> &hits);
      void produceFromTrack(const edm::EventSetup &iSetup, const Track *itt, std::vector<TrackingRecHit *> &hits);

      static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    private:
      class Rule {
      public:
        // parse a rule from a string
        Rule(const std::string &str);
        // check this DetId, update the verdict if the rule has anything to say about it
        void apply(DetId detid, const TrackerTopology *tTopo, bool &verdict) const {
          // check detector
          if (detid.subdetId() == subdet_) {
            // check layer
            if ((layer_ == 0) || (layer_ == layer(detid, tTopo))) {
              // override verdict
              verdict = keep_;
              //  std::cout<<"Verdict is "<<verdict<<" for subdet "<<subdet_<<", layer "<<layer_<<"! "<<std::endl;
            }
            // else std::cout<<"No, sorry, wrong layer.Retry..."<<std::endl;
          }
          //	else{ std::cout<<"No, sorry, wrong subdet.Retry..."<<std::endl;}
        }

      private:
        int subdet_;

        int layer_;
        bool keep_;
        int layer(DetId detid, const TrackerTopology *tTopo) const;
      };

      edm::InputTag src_;

      edm::RunNumber_t iRun;
      edm::EventNumber_t iEvt;

      size_t minimumHits_;

      bool replaceWithInactiveHits_;
      bool stripFrontInvalidHits_;
      bool stripBackInvalidHits_;
      bool stripAllInvalidHits_;

      bool isPhase2_;
      bool rejectBadStoNHits_;
      std::string CMNSubtractionMode_;
      std::vector<bool> subdetStoN_;           //(6); //,std::bool(false));
      std::vector<double> subdetStoNlowcut_;   //(6,-1.0);
      std::vector<double> subdetStoNhighcut_;  //(6,-1.0);
      bool checkStoN(const DetId &id, const TrackingRecHit *therechit);
      void parseStoN(const std::string &str);

      std::vector<uint32_t> detsToIgnore_;
      std::vector<Rule> rules_;

      bool useTrajectories_;
      bool rejectLowAngleHits_;
      double TrackAngleCut_;
      bool checkHitAngle(const TrajectoryMeasurement &meas);
      bool checkPXLQuality_;
      double pxlTPLProbXY_;
      double pxlTPLProbXYQ_;
      std::vector<int32_t> pxlTPLqBin_;

      bool checkPXLCorrClustCharge(const TrajectoryMeasurement &meas);
      double PXLcorrClusChargeCut_;

      edm::ESHandle<TrackerGeometry> theGeometry;
      edm::ESHandle<MagneticField> theMagField;

      edm::EDGetTokenT<reco::TrackCollection> tokenTracks;
      edm::EDGetTokenT<TrajTrackAssociationCollection> tokenTrajTrack;
      edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry;
      edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenMagField;
      edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTrackerTopo;

      std::optional<SiStripClusterInfo> siStripClusterInfo_;

      bool tagOverlaps_;
      int nOverlaps;
      int layerFromId(const DetId &id, const TrackerTopology *tTopo) const;
      int sideFromId(const DetId &id, const TrackerTopology *tTopo) const;
      // bool checkOverlapHit();

      TrackCandidate makeCandidate(const reco::Track &tk,
                                   std::vector<TrackingRecHit *>::iterator hitsBegin,
                                   std::vector<TrackingRecHit *>::iterator hitsEnd);
      //const TransientTrackingRecHitBuilder *RHBuilder;
    };  // class

    TrackerTrackHitFilter::Rule::Rule(const std::string &str) {
      static const boost::regex rule("(keep|drop)\\s+([A-Z]+)(\\s+(\\d+))?");
      boost::cmatch match;
      std::string match_1;
      std::string match_2;
      std::string match_3;
      // match and check it works
      if (!regex_match(str.c_str(), match, rule)) {
        throw cms::Exception("Configuration") << "Rule '" << str << "' not understood.\n";
      } else {
        edm::LogInfo("TrackerTrackHitFilter") << "*** Rule Command given to TrackerTrackHitFilter:\t" << str;
      }
      // Set up fields:
      //  rule type
      keep_ = (strncmp(match[1].first, "keep", 4) == 0);

      //  subdet
      subdet_ = -1;
      if (strncmp(match[2].first, "PXB", 3) == 0)
        subdet_ = PixelSubdetector::PixelBarrel;
      else if (strncmp(match[2].first, "PXE", 3) == 0)
        subdet_ = PixelSubdetector::PixelEndcap;
      else if (strncmp(match[2].first, "TIB", 3) == 0)
        subdet_ = StripSubdetector::TIB;
      else if (strncmp(match[2].first, "TID", 3) == 0)
        subdet_ = StripSubdetector::TID;
      else if (strncmp(match[2].first, "TOB", 3) == 0)
        subdet_ = StripSubdetector::TOB;
      else if (strncmp(match[2].first, "TEC", 3) == 0)
        subdet_ = StripSubdetector::TEC;
      if (subdet_ == -1) {
        throw cms::Exception("Configuration")
            << "Detector '" << match[2].first << "' not understood. Should be PXB, PXE, TIB, TID, TOB, TEC.\n";
      }
      //   layer (if present)
      if (match[4].first != match[4].second) {
        layer_ = atoi(match[4].first);
      } else {
        layer_ = 0;
      }
    }  //end Rule::Rule

    int TrackerTrackHitFilter::Rule::layer(DetId detid, const TrackerTopology *tTopo) const {
      return tTopo->layer(detid);
    }

    void TrackerTrackHitFilter::parseStoN(const std::string &str) {
      // match a set of capital case chars (preceded by an arbitrary number of leading blanks),
      //followed b an arbitrary number of blanks, one or more digits (not necessary, they cannot also be,
      // another set of blank spaces and, again another *eventual* digit
      // static boost::regex rule("\\s+([A-Z]+)(\\s+(\\d+)(\\.)?(\\d+))?(\\s+(\\d+)(\\.)?(\\d+))?");
      static const boost::regex rule(
          "([A-Z]+)"
          "\\s*(\\d+\\.*\\d*)?"
          "\\s*(\\d+\\.*\\d*)?");

      boost::cmatch match;
      std::string match_1;
      std::string match_2;
      std::string match_3;
      // match and check it works
      if (!regex_match(str.c_str(), match, rule)) {
        throw cms::Exception("Configuration") << "Rule for S to N cut '" << str << "' not understood.\n";
      } else {
        std::string match_0 = match[0].second;
        match_1 = match[1].second;
        match_2 = match[2].second;
        match_3 = match[3].second;
      }

      int cnt = 0;
      float subdet_ind[6];
      for (cnt = 0; cnt < 6; cnt++) {
        subdet_ind[cnt] = -1.0;
      }

      bool doALL = false;
      std::string match_1a(match[1].first, match[1].second);
      if (strncmp(match[1].first, "ALL", 3) == 0)
        doALL = true;
      if (doALL || strncmp(match[1].first, "PXB", 3) == 0)
        subdet_ind[0] = +1.0;
      if (doALL || strncmp(match[1].first, "PXE", 3) == 0)
        subdet_ind[1] = +1.0;
      if (doALL || strncmp(match[1].first, "TIB", 3) == 0)
        subdet_ind[2] = +1.0;
      if (doALL || strncmp(match[1].first, "TID", 3) == 0)
        subdet_ind[3] = +1.0;
      if (doALL || strncmp(match[1].first, "TOB", 3) == 0)
        subdet_ind[4] = +1.0;
      if (doALL || strncmp(match[1].first, "TEC", 3) == 0)
        subdet_ind[5] = +1.0;

      for (cnt = 0; cnt < 6; cnt++) {  //loop on subdets
        if (subdet_ind[cnt] > 0.0) {
          subdetStoN_[cnt] = true;
          if (match[2].first != match[2].second) {
            subdetStoNlowcut_[cnt] = atof(match[2].first);
          }
          if (match[3].first != match[3].second) {
            subdetStoNhighcut_[cnt] = atof(match[3].first);
          }
          edm::LogInfo("TrackerTrackHitFilter") << "Setting thresholds*&^ for subdet #" << cnt + 1 << " = "
                                                << subdetStoNlowcut_[cnt] << " - " << subdetStoNhighcut_[cnt];
        }
      }

      bool correct_regex = false;
      for (cnt = 0; cnt < 6; cnt++) {  //check that the regex was correct
        if (subdetStoN_[cnt])
          correct_regex = true;
      }

      if (!correct_regex) {
        throw cms::Exception("Configuration")
            << "Detector '" << match_1a << "' not understood in parseStoN. Should be PXB, PXE, TIB, TID, TOB, TEC.\n";
      }

      //std::cout<<"Reached end of parseStoN"<<std::endl;
    }  //end parseStoN

    TrackerTrackHitFilter::TrackerTrackHitFilter(const edm::ParameterSet &iConfig)
        : src_(iConfig.getParameter<edm::InputTag>("src")),
          minimumHits_(iConfig.getParameter<uint32_t>("minimumHits")),
          replaceWithInactiveHits_(iConfig.getParameter<bool>("replaceWithInactiveHits")),
          stripFrontInvalidHits_(iConfig.getParameter<bool>("stripFrontInvalidHits")),
          stripBackInvalidHits_(iConfig.getParameter<bool>("stripBackInvalidHits")),
          stripAllInvalidHits_(iConfig.getParameter<bool>("stripAllInvalidHits")),
          isPhase2_(iConfig.getParameter<bool>("isPhase2")),
          rejectBadStoNHits_(iConfig.getParameter<bool>("rejectBadStoNHits")),
          CMNSubtractionMode_(iConfig.getParameter<std::string>("CMNSubtractionMode")),
          detsToIgnore_(iConfig.getParameter<std::vector<uint32_t> >("detsToIgnore")),
          useTrajectories_(iConfig.getParameter<bool>("useTrajectories")),
          rejectLowAngleHits_(iConfig.getParameter<bool>("rejectLowAngleHits")),
          TrackAngleCut_(iConfig.getParameter<double>("TrackAngleCut")),
          checkPXLQuality_(iConfig.getParameter<bool>("usePixelQualityFlag")),
          pxlTPLProbXY_(iConfig.getParameter<double>("PxlTemplateProbXYCut")),
          pxlTPLProbXYQ_(iConfig.getParameter<double>("PxlTemplateProbXYChargeCut")),
          pxlTPLqBin_(iConfig.getParameter<std::vector<int32_t> >("PxlTemplateqBinCut")),
          PXLcorrClusChargeCut_(iConfig.getParameter<double>("PxlCorrClusterChargeCut")),
          tagOverlaps_(iConfig.getParameter<bool>("tagOverlaps")) {
      // construct the SiStripClusterInfo object only for Phase-0 / Phase-1
      // no Strip modules in Phase-2
      if (!isPhase2_) {
        siStripClusterInfo_ = consumesCollector();
      }

      tokenGeometry = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
      tokenMagField = esConsumes<MagneticField, IdealMagneticFieldRecord>();
      tokenTrackerTopo = esConsumes<TrackerTopology, TrackerTopologyRcd>();
      if (useTrajectories_)
        tokenTrajTrack = consumes<TrajTrackAssociationCollection>(src_);
      else
        tokenTracks = consumes<reco::TrackCollection>(src_);

      // sanity check
      if (stripAllInvalidHits_ && replaceWithInactiveHits_) {
        throw cms::Exception("Configuration") << "Inconsistent Configuration: you can't set both 'stripAllInvalidHits' "
                                                 "and 'replaceWithInactiveHits' to true\n";
      }
      if (rejectLowAngleHits_ && !useTrajectories_) {
        throw cms::Exception("Configuration") << "Wrong configuration of TrackerTrackHitFilter. You cannot apply the "
                                                 "cut on the track angle without using Trajectories!\n";
      }
      if (!useTrajectories_ && PXLcorrClusChargeCut_ > 0) {
        throw cms::Exception("Configuration")
            << "Wrong configuration of TrackerTrackHitFilter. You cannot apply the cut on the corrected pixel cluster "
               "charge without using Trajectories!\n";
      }

      if (pxlTPLqBin_.size() > 2) {
        edm::LogInfo("TrackerTrackHitFIlter") << "Warning from TrackerTrackHitFilter: vector with qBin cuts has size > "
                                                 "2. Additional items will be ignored.";
      }

      // read and parse commands
      std::vector<std::string> str_rules = iConfig.getParameter<std::vector<std::string> >("commands");
      rules_.reserve(str_rules.size());
      for (std::vector<std::string>::const_iterator it = str_rules.begin(), ed = str_rules.end(); it != ed; ++it) {
        rules_.push_back(Rule(*it));
      }

      if (rejectBadStoNHits_) {  //commands for S/N cut

        subdetStoN_.reserve(6);
        subdetStoNlowcut_.reserve(6);
        subdetStoNhighcut_.reserve(6);
        int cnt = 0;
        for (cnt = 0; cnt < 6; cnt++) {
          subdetStoN_[cnt] = false;
          subdetStoNlowcut_[cnt] = -1.0;
          subdetStoNhighcut_[cnt] = -1.0;
        }

        std::vector<std::string> str_StoNrules = iConfig.getParameter<std::vector<std::string> >("StoNcommands");
        for (std::vector<std::string>::const_iterator str_StoN = str_StoNrules.begin(); str_StoN != str_StoNrules.end();
             ++str_StoN) {
          parseStoN(*str_StoN);
        }
        ////edm::LogDebug("TrackerTrackHitFilter")
        edm::LogInfo("TrackerTrackHitFilter") << "Finished parsing S/N. Applying following cuts to subdets:";
        for (cnt = 0; cnt < 6; cnt++) {
          ////edm::LogDebug("TrackerTrackHitFilter")
          edm::LogVerbatim("TrackerTrackHitFilter")
              << "Subdet #" << cnt + 1 << " -> " << subdetStoNlowcut_[cnt] << " , " << subdetStoNhighcut_[cnt];
        }
      }  //end if rejectBadStoNHits_

      if (rejectLowAngleHits_)
        edm::LogInfo("TrackerTrackHitFilter") << "\nApplying cut on angle track = " << TrackAngleCut_;

      // sort detids to ignore
      std::sort(detsToIgnore_.begin(), detsToIgnore_.end());

      // issue the produce<>
      produces<TrackCandidateCollection>();
    }

    void TrackerTrackHitFilter::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
      //Dump Run and Event
      iRun = iEvent.id().run();
      iEvt = iEvent.id().event();

      // read with View, so we can read also a TrackRefVector
      edm::Handle<std::vector<reco::Track> > tracks;
      edm::Handle<TrajTrackAssociationCollection> assoMap;

      if (useTrajectories_)
        iEvent.getByToken(tokenTrajTrack, assoMap);
      else
        iEvent.getByToken(tokenTracks, tracks);

      // read from EventSetup
      theGeometry = iSetup.getHandle(tokenGeometry);
      theMagField = iSetup.getHandle(tokenMagField);
      if (!isPhase2_)
        siStripClusterInfo_->initEvent(iSetup);

      // prepare output collection
      size_t candcollsize;
      if (useTrajectories_)
        candcollsize = assoMap->size();
      else
        candcollsize = tracks->size();
      auto output = std::make_unique<TrackCandidateCollection>();

      output->reserve(candcollsize);

      // working area and tools
      std::vector<TrackingRecHit *> hits;

      if (useTrajectories_) {
        for (TrajTrackAssociationCollection::const_iterator itass = assoMap->begin(); itass != assoMap->end();
             ++itass) {
          const edm::Ref<std::vector<Trajectory> > traj = itass->key;  //trajectory in the collection
          const reco::TrackRef tkref = itass->val;                     //associated track track in the collection
          //std::cout<<"The hit collection has size "<<hits.size()<<" (should be 0) while the track contains initially "<< tkref->recHitsEnd() - tkref->recHitsBegin()<<std::endl;

          const Track *trk = &(*tkref);
          const Trajectory *myTrajectory = &(*traj);
          produceFromTrajectory(iSetup, myTrajectory, hits);

          std::vector<TrackingRecHit *>::iterator begin = hits.begin(), end = hits.end();

          // strip invalid hits at the beginning
          if (stripFrontInvalidHits_) {
            while ((begin != end) && ((*begin)->isValid() == false))
              ++begin;
          }
          // strip invalid hits at the end
          if (stripBackInvalidHits_ && (begin != end)) {
            --end;
            while ((begin != end) && ((*end)->isValid() == false))
              --end;
            ++end;
          }

          // if we still have some hits build the track candidate
          if (replaceWithInactiveHits_) {
            int nvalidhits = 0;
            for (std::vector<TrackingRecHit *>::iterator ithit = begin; ithit != end; ++ithit) {
              if ((*ithit)->isValid())
                nvalidhits++;
            }
            if (nvalidhits >= int(minimumHits_)) {
              output->push_back(makeCandidate(*trk, begin, end));
            }
          } else {  //all invalid hits have been already kicked out
            if ((end - begin) >= int(minimumHits_)) {
              output->push_back(makeCandidate(*trk, begin, end));
            }
          }

          // if we still have some hits build the track candidate
          //if ((end - begin) >= int(minimumHits_)) {
          //	output->push_back( makeCandidate ( *trk, begin, end ) );
          //}

          // now delete the hits not used by the candidate
          for (begin = hits.begin(), end = hits.end(); begin != end; ++begin) {
            if (*begin)
              delete *begin;
          }
          hits.clear();
        }  // loop on trajectories

      } else {  //use plain tracks

        // loop on tracks
        for (std::vector<reco::Track>::const_iterator ittrk = tracks->begin(), edtrk = tracks->end(); ittrk != edtrk;
             ++ittrk) {
          //    std::cout<<"The hit collection has size "<<hits.size()<<" (should be 0) while the track contains initially "<< ittrk->recHitsEnd() - ittrk->recHitsBegin()<<std::endl;

          const Track *trk = &(*ittrk);

          produceFromTrack(iSetup, trk, hits);
          //-----------------------
          /*
      std::cout<<"Hit collection in output has size "<<hits.size()<<". Dumping hit positions..."<<std::endl;
        for (std::vector<TrackingRecHit *>::iterator ith = hits.begin(), edh = hits.end(); ith != edh; ++ith) {
	  const TrackingRecHit *myhit = *(ith);
	    TransientTrackingRecHit::RecHitPointer ttrh;
	    float radius=0.0;
	    float xx=-999.0,yy=-999.0,zz=-999.0;
	    unsigned int myid=0;
	    if(myhit->isValid()){
	      ttrh = RHBuilder->build(myhit);
	      xx=ttrh->globalPosition().x();
	      yy=ttrh->globalPosition().y();
	      zz=ttrh->globalPosition().z();
	      radius = sqrt( pow(xx,2)+pow(yy,2) );
	      myid=myhit->geographicalId().rawId();
	    }
	    std::cout<<"-$-$ OUTPUT Hit position: ( "<<xx<<" , " <<yy<<" , " <<zz<<" ) , RADIUS = "  <<radius<<"  on DetID= "<< myid<<std::endl;
	}//end loop on hits
      */
          //-----------------------

          std::vector<TrackingRecHit *>::iterator begin = hits.begin(), end = hits.end();
          // std::cout << "Back in the main producer (TRK), the final hit collection has size " << hits.size() << std::endl;
          // strip invalid hits at the beginning
          if (stripFrontInvalidHits_) {
            while ((begin != end) && ((*begin)->isValid() == false))
              ++begin;
          }
          // strip invalid hits at the end
          if (stripBackInvalidHits_ && (begin != end)) {
            --end;
            while ((begin != end) && ((*end)->isValid() == false))
              --end;
            ++end;
          }

          // if we still have some hits build the track candidate
          if (replaceWithInactiveHits_) {
            int nvalidhits = 0;
            for (std::vector<TrackingRecHit *>::iterator ithit = begin; ithit != end; ++ithit) {
              if ((*ithit)->isValid())
                nvalidhits++;
            }
            if (nvalidhits >= int(minimumHits_)) {
              output->push_back(makeCandidate(*ittrk, begin, end));
            }

          } else {  //all invalid hits have been already kicked out
            if ((end - begin) >= int(minimumHits_)) {
              output->push_back(makeCandidate(*ittrk, begin, end));
            }
          }

          // now delete the hits not used by the candidate
          for (begin = hits.begin(), end = hits.end(); begin != end; ++begin) {
            if (*begin)
              delete *begin;
          }
          hits.clear();
        }  // loop on tracks
      }    //end else useTracks

      // std::cout<<"OUTPUT SIZE: "<<output->size()<<std::endl;

      iEvent.put(std::move(output));
    }

    TrackCandidate TrackerTrackHitFilter::makeCandidate(const reco::Track &tk,
                                                        std::vector<TrackingRecHit *>::iterator hitsBegin,
                                                        std::vector<TrackingRecHit *>::iterator hitsEnd) {
      PropagationDirection pdir = tk.seedDirection();
      PTrajectoryStateOnDet state;
      if (pdir == anyDirection)
        throw cms::Exception("UnimplementedFeature") << "Cannot work with tracks that have 'anyDirecton' \n";

      //  double innerP=sqrt( pow(tk.innerMomentum().X(),2)+pow(tk.innerMomentum().Y(),2)+pow(tk.innerMomentum().Z(),2) );
      //  if ( (pdir == alongMomentum) == ( innerP >= tk.outerP() ) ) {

      if ((pdir == alongMomentum) == ((tk.outerPosition() - tk.innerPosition()).Dot(tk.momentum()) >= 0)) {
        // use inner state
        TrajectoryStateOnSurface originalTsosIn(
            trajectoryStateTransform::innerStateOnSurface(tk, *theGeometry, &*theMagField));
        state = trajectoryStateTransform::persistentState(originalTsosIn, DetId(tk.innerDetId()));
      } else {
        // use outer state
        TrajectoryStateOnSurface originalTsosOut(
            trajectoryStateTransform::outerStateOnSurface(tk, *theGeometry, &*theMagField));
        state = trajectoryStateTransform::persistentState(originalTsosOut, DetId(tk.outerDetId()));
      }
      TrajectorySeed seed(state, TrackCandidate::RecHitContainer(), pdir);
      TrackCandidate::RecHitContainer ownHits;
      ownHits.reserve(hitsEnd - hitsBegin);
      for (; hitsBegin != hitsEnd; ++hitsBegin) {
        //if(! (*hitsBegin)->isValid() ) std::cout<<"Putting in the trackcandidate an INVALID HIT !"<<std::endl;
        ownHits.push_back(*hitsBegin);
      }

      TrackCandidate cand(ownHits, seed, state, tk.seedRef());

      return cand;
    }

    void TrackerTrackHitFilter::produceFromTrack(const edm::EventSetup &iSetup,
                                                 const Track *itt,
                                                 std::vector<TrackingRecHit *> &hits) {
      // loop on tracks
      hits.clear();  // extra safety

      for (trackingRecHit_iterator ith = itt->recHitsBegin(), edh = itt->recHitsEnd(); ith != edh; ++ith) {
        const TrackingRecHit *hit = (*ith);  // ith is an iterator on edm::Ref to rechit

        DetId detid = hit->geographicalId();

        //check that the hit is a real hit and not a constraint
        if (hit->isValid() && hit == nullptr && detid.rawId() == 0)
          continue;

        int verdict = checkHit(iSetup, detid, hit);
        if (verdict == 0) {
          // just copy the hit
          hits.push_back(hit->clone());
        } else if (verdict < -2) {  //hit rejected because did not pass the selections
                                    // still, if replaceWithInactiveHits is true we have to put a new hit
          if (replaceWithInactiveHits_) {
            hits.push_back(new InvalidTrackingRecHit(*(hit->det()), TrackingRecHit::inactive));
          }
        } else if (verdict == -2)
          hits.push_back(hit->clone());  //hit not in the tracker
        else if (verdict == -1) {        //hit not valid
          if (!stripAllInvalidHits_) {
            hits.push_back(hit->clone());
          }
        }
      }  // loop on hits

    }  //end TrackerTrackHitFilter::produceFromTrack

    void TrackerTrackHitFilter::produceFromTrajectory(const edm::EventSetup &iSetup,
                                                      const Trajectory *itt,
                                                      std::vector<TrackingRecHit *> &hits) {
      hits.clear();  // extra safety
      nOverlaps = 0;

      //Retrieve tracker topology from geometry
      edm::ESHandle<TrackerTopology> tTopoHand = iSetup.getHandle(tokenTrackerTopo);
      const TrackerTopology *tTopo = tTopoHand.product();

      std::vector<TrajectoryMeasurement> tmColl = itt->measurements();

      //---OverlapBegin needed eventually for overlaps, but I must create them here in any case
      const TrajectoryMeasurement *previousTM(nullptr);
      DetId previousId(0);
      //int previousLayer(-1);
      ///---OverlapEnd

      for (std::vector<TrajectoryMeasurement>::const_iterator itTrajMeas = tmColl.begin(); itTrajMeas != tmColl.end();
           itTrajMeas++) {
        TransientTrackingRecHit::ConstRecHitPointer hitpointer = itTrajMeas->recHit();

        //check that the hit is a real hit and not a constraint
        if (hitpointer->isValid() && hitpointer->hit() == nullptr) {
          continue;
        }

        const TrackingRecHit *hit = ((*hitpointer).hit());
        DetId detid = hit->geographicalId();
        int verdict = checkHit(iSetup, detid, hit);

        if (verdict == 0) {
          if (rejectLowAngleHits_ && !checkHitAngle(*itTrajMeas)) {  //check angle of track on module if requested
            verdict = -6;                                            //override previous verdicts
          }
        }

        /*
    //this has been included in checkHitAngle(*itTrajMeas)
    if (verdict == 0) {
    if( PXLcorrClusChargeCut_>0.0  && !checkPXLCorrClustCharge(*itTrajMeas) ){//check angle of track on module if requested
    verdict=-7;//override previous verdicts
    }
    }
    */

        if (verdict == 0) {  // Hit TAKEN !!!!

          if (tagOverlaps_) {  ///---OverlapBegin
            //std::cout<<"Looking for overlaps in Run="<<iRun<<" , Event ="<<iEvt<<std::flush;

            int side(sideFromId(detid, tTopo));    //side 0=barrel, 1=minus , 2=plus
            int layer(layerFromId(detid, tTopo));  //layer or disk
            int subDet = detid.subdetId();
            //std::cout  << "  Check Subdet #" <<subDet << ", layer = " <<layer<<" stereo: "<< ((subDet > 2)?(SiStripDetId(detid).stereo()):2);

            if ((previousTM != nullptr) && (layer != -1)) {
              //std::cout<<"A previous TM exists! "<<std::endl;
              for (std::vector<TrajectoryMeasurement>::const_iterator itmCompare = itTrajMeas - 1;
                   itmCompare >= tmColl.begin() && itmCompare > itTrajMeas - 4;
                   --itmCompare) {
                DetId compareId = itmCompare->recHit()->geographicalId();
                if (subDet != compareId.subdetId() || side != sideFromId(compareId, tTopo) ||
                    layer != layerFromId(compareId, tTopo))
                  break;
                if (!itmCompare->recHit()->isValid())
                  continue;
                if (GeomDetEnumerators::isTrackerPixel(theGeometry->geomDetSubDetector(detid.subdetId())) ||
                    (GeomDetEnumerators::isTrackerStrip(theGeometry->geomDetSubDetector(detid.subdetId())) &&
                     SiStripDetId(detid).stereo() ==
                         SiStripDetId(compareId).stereo())) {  //if either pixel or strip stereo module
                  //  overlapHits.push_back(std::make_pair(&(*itmCompare),&(*itm)));
                  //std::cout<< "Adding pair "<< ((subDet >2)?(SiStripDetId(detid).stereo()):2)
                  //     << " from SubDet = "<<subDet<<" , layer = " << layer<<"  Run:"<<iRun<<"\tEv: "<<iEvt<<"\tId1: "<<compareId.rawId()<<"\tId2: "<<detid.rawId()<<std::endl;
                  // if(abs(compareId.rawId()-detid.rawId())==1)std::cout<<"These two are from the same det! Id1= "<<detid.rawId()<<" has stereo type "<<SiStripDetId(detid).stereo() <<"\tId2: "<<compareId.rawId()<<" has stereo type "<<SiStripDetId(compareId).stereo()<<std::endl;
                  ///
                  // if(detid.rawId()<compareId.rawId()){
                  // std::cout<< "+++ "<< "\t"<<iRun<<"\t"<<iEvt<<"\t"<<detid.rawId()<<"\t"<<compareId.rawId()<<std::endl;
                  // }
                  //else  std::cout<< "+++ "<< "\t"<<iRun<<"\t"<<iEvt<<"\t"<<compareId.rawId()<<"\t"<<detid.rawId()<<std::endl;

                  nOverlaps++;
                  break;
                }
              }  //end second loop on TM for overlap tagging

            }  //end   if ( (layer!=-1 )&&(acceptLayer[subDet]))

            previousTM = &(*itTrajMeas);
            previousId = detid;
            //previousLayer = layer;
          }  //end if look for overlaps
          ///---OverlapEnd

          hits.push_back(hit->clone());  //just copy it
        }                                //end if HIT TAKEN
        else if (verdict < -2) {         //hit rejected because did not pass the selections
          // still, if replaceWithInactiveHits is true we have to put a new hit
          if (replaceWithInactiveHits_) {
            hits.push_back(new InvalidTrackingRecHit(*hit->det(), TrackingRecHit::inactive));
          }
        } else if (verdict == -2)
          hits.push_back(hit->clone());  //hit not in the tracker
        else if (verdict == -1) {        //hit not valid
          if (!stripAllInvalidHits_) {
            hits.push_back(hit->clone());
          }
        }
      }  // loop on hits

      std::reverse(hits.begin(), hits.end());
    }  //end TrackerTrackHitFilter::produceFromTrajectories

    int TrackerTrackHitFilter::checkHit(const edm::EventSetup &iSetup, const DetId &detid, const TrackingRecHit *hit) {
      //Retrieve tracker topology from geometry
      edm::ESHandle<TrackerTopology> tTopoHand = iSetup.getHandle(tokenTrackerTopo);
      const TrackerTopology *tTopo = tTopoHand.product();

      int hitresult = 0;
      if (hit->isValid()) {
        if (detid.det() == DetId::Tracker) {  // check for tracker hits
          bool verdict = true;
          // first check at structure level
          for (std::vector<Rule>::const_iterator itr = rules_.begin(), edr = rules_.end(); itr != edr; ++itr) {
            itr->apply(detid, tTopo, verdict);
          }

          // if the hit is good, check again at module level
          if (verdict) {
            if (std::binary_search(detsToIgnore_.begin(), detsToIgnore_.end(), detid.rawId())) {
              hitresult = -4;
            }
          } else
            hitresult = -3;
          //if the hit is in the desired part of the det, check other things
          if (hitresult == 0 && rejectBadStoNHits_) {
            if (!checkStoN(detid, hit))
              hitresult = -5;
          }  //end if S/N is ok
        }    //end hit in tracker
        else
          hitresult = -2;
      }  //end hit is valid
      else
        hitresult = -1;  //invalid hit
      return hitresult;
    }  //end  TrackerTrackHitFilter::checkHit()

    bool TrackerTrackHitFilter::checkStoN(const DetId &id, const TrackingRecHit *therechit) {
      bool keepthishit = true;
      // const uint32_t& recHitDetId = id.rawId();

      //check StoN only if subdet was set by the user
      //  int subdet_cnt=0;
      int subdet_cnt = id.subdetId();

      //  for(subdet_cnt=0;subdet_cnt<6; ++subdet_cnt){

      //  if( subdetStoN_[subdet_cnt-1]&& (id.subdetId()==subdet_cnt)  ){//check that hit is in a det belonging to a subdet where we decided to apply a S/N cut

      // for phase-2 OT placehold, do nothing
      if (GeomDetEnumerators::isOuterTracker(theGeometry->geomDetSubDetector(id.subdetId())) &&
          !GeomDetEnumerators::isTrackerStrip(theGeometry->geomDetSubDetector(id.subdetId()))) {
        return true;
      }

      if (GeomDetEnumerators::isTrackerStrip(theGeometry->geomDetSubDetector(id.subdetId()))) {
        if (subdetStoN_[subdet_cnt - 1]) {
          //check that hit is in a det belonging to a subdet where we decided to apply a S/N cut
          const std::type_info &type = typeid(*therechit);
          const SiStripCluster *cluster;
          if (type == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D *hit = dynamic_cast<const SiStripRecHit2D *>(therechit);
            if (hit != nullptr)
              cluster = &*(hit->cluster());
            else {
              edm::LogError("TrackerTrackHitFilter")
                  << "TrackerTrackHitFilter::checkStoN : Unknown valid tracker hit in subdet " << id.subdetId()
                  << "(detID=" << id.rawId() << ")\n ";
              keepthishit = false;
            }
          } else if (type == typeid(SiStripRecHit1D)) {
            const SiStripRecHit1D *hit = dynamic_cast<const SiStripRecHit1D *>(therechit);
            if (hit != nullptr)
              cluster = &*(hit->cluster());
            else {
              edm::LogError("TrackerTrackHitFilter")
                  << "TrackerTrackHitFilter::checkStoN : Unknown valid tracker hit in subdet " << id.subdetId()
                  << "(detID=" << id.rawId() << ")\n ";
              keepthishit = false;
            }
          }
          //the following two cases should not happen anymore since CMSSW > 2_0_X because of hit splitting in stereo modules
          //const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(therechit);
          //const ProjectedSiStripRecHit2D* unmatchedhit = dynamic_cast<const ProjectedSiStripRecHit2D*>(therechit);
          else {
            throw cms::Exception("Unknown RecHit Type")
                << "RecHit of type " << type.name() << " not supported. (use c++filt to demangle the name)";
          }

          if (keepthishit) {
            siStripClusterInfo_->setCluster(*cluster, id.rawId());
            if ((subdetStoNlowcut_[subdet_cnt - 1] > 0) &&
                (siStripClusterInfo_->signalOverNoise() < subdetStoNlowcut_[subdet_cnt - 1]))
              keepthishit = false;
            if ((subdetStoNhighcut_[subdet_cnt - 1] > 0) &&
                (siStripClusterInfo_->signalOverNoise() > subdetStoNhighcut_[subdet_cnt - 1]))
              keepthishit = false;
            //if(!keepthishit)std::cout<<"Hit rejected because of bad S/N: "<<siStripClusterInfo_->signalOverNoise()<<std::endl;
          }

        }  //end if  subdetStoN_[subdet_cnt]&&...

      }  //end if subdet_cnt >2
      else if (GeomDetEnumerators::isTrackerPixel(theGeometry->geomDetSubDetector(id.subdetId()))) {  //pixel
        //pixels have naturally a very low noise (because of their low capacitance). So the S/N cut is
        //irrelevant in this case. Leave it dummy
        keepthishit = true;

        /**************
       * Cut on cluster charge corr by angle embedded in the checkHitAngle() function
       *
       *************/

        if (checkPXLQuality_) {
          const SiPixelRecHit *pixelhit = dynamic_cast<const SiPixelRecHit *>(therechit);
          if (pixelhit != nullptr) {
            //std::cout << "ClusterCharge=" <<std::flush<<pixelhit->cluster()->charge() << std::flush;
            float xyprob = pixelhit->clusterProbability(0);        //x-y combined log_e probability of the pixel cluster
                                                                   //singl x- and y-prob not stored sicne CMSSW 3_9_0
            float xychargeprob = pixelhit->clusterProbability(1);  //xy-prob * charge prob
            //	float chargeprob   =pixelhit->clusterProbability(2);//charge prob
            bool haspassed_tplreco = pixelhit->hasFilledProb();  //the cluster was associted to a template
            int qbin = pixelhit->qBin();  //R==meas_charge/pred_charge:  Qbin=0 ->R>1.5 , =1->1<R<1.5 ,=2->0.85<R<1 ,
                                          // Qbin=3->0.95*Qminpred<R<0.85 ,=4->, =5->meas_charge<0.95*Qminpred

            //	if(haspassed_tplreco)	std::cout<<"  CLUSTPROB=\t"<<xprob<<"\t"<<yprob<<"\t"<<combprob<<"\t"<<qbin<<std::endl;
            //	else std::cout<<"CLUSTPROBNOTDEF=\t"<<xprob<<"\t"<<yprob<<"\t"<<combprob<<"\t"<<qbin<<std::endl;

            keepthishit = false;
            //	std::cout<<"yyyyy "<<qbin<<" "<<xprob<<"  "<<yprob<<std::endl;
            if (haspassed_tplreco && xyprob > pxlTPLProbXY_ && xychargeprob > pxlTPLProbXYQ_ && qbin > pxlTPLqBin_[0] &&
                qbin <= pxlTPLqBin_[1])
              keepthishit = true;

          } else {
            edm::LogInfo("TrackerTrackHitFilter") << "HIT IN PIXEL (" << subdet_cnt << ") but PixelRecHit is EMPTY!!!";
          }
        }  //end if check pixel quality flag
      }
      //    else  throw cms::Exception("TrackerTrackHitFilter") <<"Loop over subdetector out of range when applying the S/N cut: "<<subdet_cnt;

      //  }//end loop on subdets

      return keepthishit;
    }  //end CheckStoN

    bool TrackerTrackHitFilter::checkHitAngle(const TrajectoryMeasurement &meas) {
      bool angle_ok = false;
      bool corrcharge_ok = true;
      const TrajectoryStateOnSurface &tsos = meas.updatedState();
      /*
  edm::LogDebug("TrackerTrackHitFilter")<<"TSOS parameters: ";
  edm::LogDebug("TrackerTrackHitFilter") <<"Global momentum: "<<tsos.globalMomentum().x()<<"  "<<tsos.globalMomentum().y()<<"  "<<tsos.globalMomentum().z();
  edm::LogDebug("TrackerTrackHitFilter") <<"Local momentum: "<<tsos.localMomentum().x()<<"  "<<tsos.localMomentum().y()<<"  "<<tsos.localMomentum().z();
  edm::LogDebug("TrackerTrackHitFilter") <<"Track charge: "  <<tsos.charge();
  edm::LogDebug("TrackerTrackHitFilter")<<"Local position: "  <<tsos.localPosition().x()<<"  "<<tsos.localPosition().y()<<"  "<<tsos.localPosition().z();
  */
      if (tsos.isValid()) {
        //check the angle of this tsos
        float mom_x = tsos.localDirection().x();
        float mom_y = tsos.localDirection().y();
        float mom_z = tsos.localDirection().z();
        //we took LOCAL momentum, i.e. respect to surface. Thus the plane is z=0
        float angle = TMath::ASin(TMath::Abs(mom_z) / sqrt(pow(mom_x, 2) + pow(mom_y, 2) + pow(mom_z, 2)));
        if (!rejectLowAngleHits_ || angle >= TrackAngleCut_)
          angle_ok = true;  // keep this hit
        // else  std::cout<<"Hit rejected because angle is "<< angle<<" ( <"<<TrackAngleCut_<<" )"<<std::endl;

        if (angle_ok && PXLcorrClusChargeCut_ > 0.0) {
          //
          //get the hit from the TM and check that it is in the pixel
          const TransientTrackingRecHit::ConstRecHitPointer &hitpointer = meas.recHit();
          if (hitpointer->isValid()) {
            const TrackingRecHit *hit = (*hitpointer).hit();
            if (GeomDetEnumerators::isTrackerPixel(
                    theGeometry->geomDetSubDetector(hit->geographicalId().subdetId()))) {  //do it only for pixel hits
              corrcharge_ok = false;
              float clust_alpha = atan2(mom_z, mom_x);
              float clust_beta = atan2(mom_z, mom_y);

              //Now get the cluster charge

              const SiPixelRecHit *pixelhit = dynamic_cast<const SiPixelRecHit *>(hit);
              float clust_charge = pixelhit->cluster()->charge();
              float corr_clust_charge =
                  clust_charge * sqrt(1.0 / (1.0 / pow(tan(clust_alpha), 2) + 1.0 / pow(tan(clust_beta), 2) + 1.0));
              //std::cout<<"xxxxx "<<clust_charge<<" "<<corr_clust_charge<<"  " <<pixelhit->qBin()<<"  "<<pixelhit->clusterProbability(1)<<"  "<<pixelhit->clusterProbability(2)<< std::endl;
              if (corr_clust_charge > PXLcorrClusChargeCut_) {
                corrcharge_ok = true;
              }
            }  //end if hit is in pixel
          }    //end if hit is valid

        }  //check corr cluster charge for pixel hits

      }  //end if TSOS is valid
      else {
        edm::LogWarning("TrackerTrackHitFilter") << "TSOS not valid ! Impossible to calculate track angle.";
      }

      return angle_ok && corrcharge_ok;
    }  //end TrackerTrackHitFilter::checkHitAngle

    bool TrackerTrackHitFilter::checkPXLCorrClustCharge(const TrajectoryMeasurement &meas) {
      /*
    Code taken from DPGAnalysis/SiPixelTools/plugins/PixelNtuplizer_RealData.cc
  */

      bool corrcharge_ok = false;
      //get the hit from the TM and check that it is in the pixel
      const TransientTrackingRecHit::ConstRecHitPointer &hitpointer = meas.recHit();
      if (!hitpointer->isValid())
        return corrcharge_ok;
      const TrackingRecHit *hit = (*hitpointer).hit();
      if (GeomDetEnumerators::isTrackerStrip(
              theGeometry->geomDetSubDetector(hit->geographicalId().subdetId()))) {  //SiStrip hit, skip
        return corrcharge_ok;
      }

      const TrajectoryStateOnSurface &tsos = meas.updatedState();
      if (tsos.isValid()) {
        float mom_x = tsos.localDirection().x();
        float mom_y = tsos.localDirection().y();
        float mom_z = tsos.localDirection().z();
        float clust_alpha = atan2(mom_z, mom_x);
        float clust_beta = atan2(mom_z, mom_y);

        //Now get the cluster charge

        const SiPixelRecHit *pixelhit = dynamic_cast<const SiPixelRecHit *>(hit);
        float clust_charge = pixelhit->cluster()->charge();
        float corr_clust_charge =
            clust_charge * sqrt(1.0 / (1.0 / pow(tan(clust_alpha), 2) + 1.0 / pow(tan(clust_beta), 2) + 1.0));
        if (corr_clust_charge > PXLcorrClusChargeCut_)
          corrcharge_ok = true;

      }  //end if TSOS is valid
      return corrcharge_ok;

    }  //end TrackerTrackHitFilter::checkPXLCorrClustCharge

    int TrackerTrackHitFilter::layerFromId(const DetId &id, const TrackerTopology *tTopo) const {
      return tTopo->layer(id);
    }

    int TrackerTrackHitFilter::sideFromId(const DetId &id, const TrackerTopology *tTopo) const {
      return tTopo->side(id);
    }

    // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
    void TrackerTrackHitFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.setComment("");
      desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
      desc.add<uint32_t>("minimumHits", 3)->setComment("number of hits for refit");
      desc.add<bool>("replaceWithInactiveHits", false)
          ->setComment(
              " instead of removing hits replace them with inactive hits, so you still consider the multiple "
              "scattering");
      desc.add<bool>("stripFrontInvalidHits", false)
          ->setComment("strip invalid & inactive hits from any end of the track");
      desc.add<bool>("stripBackInvalidHits", false)
          ->setComment("strip invalid & inactive hits from any end of the track");
      desc.add<bool>("stripAllInvalidHits", false)->setComment("dangerous to turn on, you might forget about MS");
      desc.add<bool>("isPhase2", false);
      desc.add<bool>("rejectBadStoNHits", false);
      desc.add<std::string>("CMNSubtractionMode", std::string("Median"))->setComment("TT6");
      desc.add<std::vector<uint32_t> >("detsToIgnore", {});
      desc.add<bool>("useTrajectories", false);
      desc.add<bool>("rejectLowAngleHits", false);
      desc.add<double>("TrackAngleCut", 0.25)->setComment("rad");
      desc.add<bool>("usePixelQualityFlag", false);
      desc.add<double>("PxlTemplateProbXYCut", 0.000125);
      desc.add<double>("PxlTemplateProbXYChargeCut", -99.);
      desc.add<std::vector<int32_t> >("PxlTemplateqBinCut", {0, 3});
      desc.add<double>("PxlCorrClusterChargeCut", -999.0);
      desc.add<bool>("tagOverlaps", false);
      desc.add<std::vector<std::string> >("commands", {})->setComment("layers to remove");
      desc.add<std::vector<std::string> >("StoNcommands", {})->setComment("S/N cut per layer");
      descriptions.addWithDefaultLabel(desc);
    }

  }  // namespace modules
}  // namespace reco

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using reco::modules::TrackerTrackHitFilter;
DEFINE_FWK_MODULE(TrackerTrackHitFilter);
