#include "RecoTracker/MkFitCMS/standalone/MkStandaloneSeqs.h"
#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

#include "RecoTracker/MkFitCore/src/Debug.h"

#include "oneapi/tbb/parallel_for.h"

namespace mkfit {

  namespace StdSeq {

    //=========================================================================
    // Hit processing
    //=========================================================================

    void loadHitsAndBeamSpot(Event &ev, EventOfHits &eoh) {
      eoh.reset();

      // fill vector of hits in each layer
      // XXXXMT: Does it really makes sense to multi-thread this?
      tbb::parallel_for(tbb::blocked_range<int>(0, ev.layerHits_.size()), [&](const tbb::blocked_range<int> &layers) {
        for (int ilay = layers.begin(); ilay < layers.end(); ++ilay) {
          eoh.suckInHits(ilay, ev.layerHits_[ilay]);
        }
      });
      eoh.setBeamSpot(ev.beamSpot_);
    }

    void handle_duplicates(Event *) {
      /*
      // Mark tracks as duplicates; if within CMSSW, remove duplicate tracks from fit or candidate track collection
      if (Config::removeDuplicates) {
        if (Config::quality_val || Config::sim_val || Config::cmssw_val) {
          clean_duplicates(event->candidateTracks_);
          if (Config::backwardFit)
            clean_duplicates(event->fitTracks_);
        }
        // For the MEIF benchmarks and the stress tests, no validation flags are set so we will enter this block
        else {
          // Only care about the candidate tracks here; no need to run the duplicate removal on both candidate and fit tracks
          clean_duplicates(event->candidateTracks_);
        }
      }
      */
    }

    //=========================================================================
    // Random stuff
    //=========================================================================

    void dump_simtracks(Event *event) {
      // Ripped out of MkBuilder::begin_event, ifdefed under DEBUG

      std::vector<Track> &simtracks = event->simTracks_;

      for (int itrack = 0; itrack < (int)simtracks.size(); ++itrack) {
        // bool debug = true;
        Track track = simtracks[itrack];
        // simtracks are initially written with label = index; uncomment in case tracks were edited
        // if (track.label() != itrack) {
        //   dprintf("Bad label for simtrack %d -- %d\n", itrack, track.label());
        // }

        dprint("MX - simtrack with nHits=" << track.nFoundHits() << " chi2=" << track.chi2() << " pT=" << track.pT()
                                           << " phi=" << track.momPhi() << " eta=" << track.momEta());
      }

      for (int itrack = 0; itrack < (int)simtracks.size(); ++itrack) {
        for (int ihit = 0; ihit < simtracks[itrack].nFoundHits(); ++ihit) {
          dprint("track #" << itrack << " hit #" << ihit
                           << " hit pos=" << simtracks[itrack].hitsVector(event->layerHits_)[ihit].position()
                           << " phi=" << simtracks[itrack].hitsVector(event->layerHits_)[ihit].phi());
        }
      }
    }

    void track_print(Event *event, const Track &t, const char *pref) {
      printf("%s with q=%+i pT=%7.3f eta=% 7.3f nHits=%2d  label=%4d\nState:\n",
             pref,
             t.charge(),
             t.pT(),
             t.momEta(),
             t.nFoundHits(),
             t.label());

      print(t.state());

      printf("Hits:\n");
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        int lyr = t.getHitLyr(ih);
        int idx = t.getHitIdx(ih);
        if (idx >= 0) {
          const Hit &hit = event->layerHits_[lyr][idx];
          printf("    hit %2d lyr=%2d idx=%4d pos r=%7.3f z=% 8.3f   mc_hit=%4d mc_trk=%4d\n",
                 ih,
                 lyr,
                 idx,
                 hit.r(),
                 hit.z(),
                 hit.mcHitID(),
                 hit.mcTrackID(event->simHitsInfo_));
        } else
          printf("    hit %2d        idx=%i\n", ih, t.getHitIdx(ih));
      }
    }

    //------------------------------------------------------------------------------
    // Non-ROOT validation
    //------------------------------------------------------------------------------

    void Quality::quality_val(Event *event) {
      quality_reset();

      std::map<int, int> cmsswLabelToPos;
      if (Config::dumpForPlots && Config::readCmsswTracks) {
        for (size_t itrack = 0; itrack < event->cmsswTracks_.size(); itrack++) {
          cmsswLabelToPos[event->cmsswTracks_[itrack].label()] = itrack;
        }
      }

      for (size_t itrack = 0; itrack < event->candidateTracks_.size(); itrack++) {
        quality_process(event, event->candidateTracks_[itrack], itrack, cmsswLabelToPos);
      }

      quality_print();
    }

    void Quality::quality_reset() { m_cnt = m_cnt1 = m_cnt2 = m_cnt_8 = m_cnt1_8 = m_cnt2_8 = m_cnt_nomc = 0; }

    void Quality::quality_process(Event *event, Track &tkcand, const int itrack, std::map<int, int> &cmsswLabelToPos) {
      // KPM: Do not use this method for validating CMSSW tracks if we ever build a DumbCMSSW function for them to print out...
      // as we would need to access seeds through map of seed ids...

      // initialize track extra (input original seed label)
      const auto label = tkcand.label();
      TrackExtra extra(label);

      // track_print(event, tkcand, "quality_process -> track_print:");

      // access temp seed trk and set matching seed hits
      const auto &seed = event->seedTracks_[itrack];
      extra.findMatchingSeedHits(tkcand, seed, event->layerHits_);

      // set mcTrackID through 50% hit matching after seed
      extra.setMCTrackIDInfo(
          tkcand, event->layerHits_, event->simHitsInfo_, event->simTracks_, false, (Config::seedInput == simSeeds));
      const int mctrk = extra.mcTrackID();

      //  int mctrk = tkcand.label(); // assumes 100% "efficiency"

      const float pT = tkcand.pT();
      float pTmc = 0.f, etamc = 0.f, phimc = 0.f;
      float pTr = 0.f;
      int nfoundmc = -1;

      if (mctrk < 0 || static_cast<size_t>(mctrk) >= event->simTracks_.size()) {
        ++m_cnt_nomc;
        dprint("XX bad track idx " << mctrk << ", orig label was " << label);
      } else {
        auto &simtrack = event->simTracks_[mctrk];
        pTmc = simtrack.pT();
        etamc = simtrack.momEta();
        phimc = simtrack.momPhi();
        pTr = pT / pTmc;

        nfoundmc = simtrack.nUniqueLayers();

        ++m_cnt;
        if (pTr > 0.9 && pTr < 1.1)
          ++m_cnt1;
        if (pTr > 0.8 && pTr < 1.2)
          ++m_cnt2;

        if (tkcand.nFoundHits() >= 0.8f * nfoundmc) {
          ++m_cnt_8;
          if (pTr > 0.9 && pTr < 1.1)
            ++m_cnt1_8;
          if (pTr > 0.8 && pTr < 1.2)
            ++m_cnt2_8;
        }

        // perl -ne 'print if m/FOUND_LABEL\s+[-\d]+/o;' | sort -k2 -n
        // grep "FOUND_LABEL" | sort -n -k 8,8 -k 2,2
        // printf("FOUND_LABEL %6d  pT_mc= %8.2f eta_mc= %8.2f event= %d\n", label, pTmc, etamc, event->evtID());
      }

      float pTcmssw = 0.f, etacmssw = 0.f, phicmssw = 0.f;
      int nfoundcmssw = -1;
      if (Config::dumpForPlots && Config::readCmsswTracks) {
        if (cmsswLabelToPos.count(label)) {
          auto &cmsswtrack = event->cmsswTracks_[cmsswLabelToPos[label]];
          pTcmssw = cmsswtrack.pT();
          etacmssw = cmsswtrack.momEta();
          phicmssw = cmsswtrack.swimPhiToR(tkcand.x(), tkcand.y());  // to get rough estimate of diff in phi
          nfoundcmssw = cmsswtrack.nUniqueLayers();
        }
      }

      if (!Config::silent && Config::dumpForPlots) {
        std::lock_guard<std::mutex> printlock(Event::printmutex);
        printf(
            "MX - found track with chi2= %6.3f nFoundHits= %2d pT= %7.4f eta= %7.4f phi= %7.4f nfoundmc= %2d pTmc= "
            "%7.4f etamc= %7.4f phimc= %7.4f nfoundcmssw= %2d pTcmssw= %7.4f etacmssw= %7.4f phicmssw= %7.4f lab= %d\n",
            tkcand.chi2(),
            tkcand.nFoundHits(),
            pT,
            tkcand.momEta(),
            tkcand.momPhi(),
            nfoundmc,
            pTmc,
            etamc,
            phimc,
            nfoundcmssw,
            pTcmssw,
            etacmssw,
            phicmssw,
            label);
      }
    }

    void Quality::quality_print() {
      if (!Config::silent) {
        std::lock_guard<std::mutex> printlock(Event::printmutex);
        std::cout << "found tracks=" << m_cnt << "  in pT 10%=" << m_cnt1 << "  in pT 20%=" << m_cnt2
                  << "     no_mc_assoc=" << m_cnt_nomc << std::endl;
        std::cout << "  nH >= 80% =" << m_cnt_8 << "  in pT 10%=" << m_cnt1_8 << "  in pT 20%=" << m_cnt2_8
                  << std::endl;
      }
    }

    //------------------------------------------------------------------------------
    // Root validation
    //------------------------------------------------------------------------------

    void root_val_dumb_cmssw(Event *event) {
      // get labels correct first
      event->relabel_bad_seedtracks();
      event->relabel_cmsswtracks_from_seeds();

      //collection cleaning
      if (Config::nItersCMSSW > 0)
        event->select_tracks_iter(Config::nItersCMSSW);

      // set the track collections to each other
      event->candidateTracks_ = event->cmsswTracks_;
      event->fitTracks_ = event->candidateTracks_;

      // prep the tracks + extras
      prep_simtracks(event);
      prep_recotracks(event);

      // validate
      event->validate();
    }

    void root_val(Event *event) {
      // score the tracks
      score_tracks(event->seedTracks_);
      score_tracks(event->candidateTracks_);

      // deal with fit tracks
      if (Config::backwardFit) {
        score_tracks(event->fitTracks_);
      } else
        event->fitTracks_ = event->candidateTracks_;

      // sort hits + make extras, align if needed
      prep_recotracks(event);
      if (Config::cmssw_val)
        prep_cmsswtracks(event);

      // validate
      event->validate();
    }

    void prep_recotracks(Event *event) {
      // seed tracks extras always needed
      if (Config::sim_val || Config::sim_val_for_cmssw) {
        prep_tracks(event, event->seedTracks_, event->seedTracksExtra_, true);
      } else if (Config::cmssw_val)  // seed tracks are not validated, labels used for maps --> do NOT align index and labels!
      {
        prep_tracks(event, event->seedTracks_, event->seedTracksExtra_, false);
      }

      // make extras + align index == label() for candidate tracks
      prep_tracks(event, event->candidateTracks_, event->candidateTracksExtra_, true);
      prep_tracks(event, event->fitTracks_, event->fitTracksExtra_, true);
    }

    void prep_simtracks(Event *event) {
      // First prep sim tracks to have hits sorted, then mark unfindable if too short
      prep_reftracks(event, event->simTracks_, event->simTracksExtra_, false);

      // Now, make sure sim track shares at least four hits with a single cmssw seed.
      // This ensures we factor out any weakness from CMSSW

      // First, make a make a map of [lyr][hit idx].vector(seed trk labels)
      LayIdxIDVecMapMap seedHitIDMap;
      std::map<int, int> labelNHitsMap;
      std::map<int, int> labelAlgoMap;
      std::map<int, std::vector<int>> labelSeedHitsMap;
      for (const auto &seedtrack : event->seedTracks_) {
        for (int ihit = 0; ihit < seedtrack.nTotalHits(); ihit++) {
          const auto lyr = seedtrack.getHitLyr(ihit);
          const auto idx = seedtrack.getHitIdx(ihit);

          if (lyr < 0 || idx < 0)
            continue;  // standard check
          seedHitIDMap[lyr][idx].push_back(seedtrack.label());
          labelSeedHitsMap[seedtrack.label()].push_back(lyr);
        }
        labelNHitsMap[seedtrack.label()] = seedtrack.nTotalHits();
        labelAlgoMap[seedtrack.label()] = seedtrack.algoint();
      }

      // Then, loop over sim tracks, and add up how many lyrs they possess of a single seed track
      unsigned int count = 0;
      for (auto &simtrack : event->simTracks_) {
        if (simtrack.isNotFindable())
          continue;  // skip ones we already know are bad
        TrkIDLaySetMap seedIDMap;
        for (int ihit = 0; ihit < simtrack.nTotalHits(); ihit++) {
          const auto lyr = simtrack.getHitLyr(ihit);
          const auto idx = simtrack.getHitIdx(ihit);

          if (lyr < 0 || idx < 0)
            continue;  // standard check

          if (!seedHitIDMap.count(lyr))
            continue;  // ensure seed hit map has at least one entry for this layer
          if (!seedHitIDMap.at(lyr).count(idx))
            continue;  // ensure seed hit map has at least one entry for this idx

          for (const auto label : seedHitIDMap.at(lyr).at(idx)) {
            const auto &seedLayers = labelSeedHitsMap[label];
            if (std::find(seedLayers.begin(), seedLayers.end(), lyr) != seedLayers.end())  //seed check moved here
              seedIDMap[label].emplace(lyr);
          }
        }

        // now see if one of the seedIDs matched has at least 4 hits!
        bool isSimSeed = false;
        for (const auto &seedIDpair : seedIDMap) {
          if ((int)seedIDpair.second.size() == labelNHitsMap[seedIDpair.first]) {
            isSimSeed = true;
            if (Config::mtvRequireSeeds)
              simtrack.setAlgoint(labelAlgoMap[seedIDpair.first]);
            if (Config::mtvRequireSeeds)
              event->simTracksExtra_[count].addAlgo(labelAlgoMap[seedIDpair.first]);
            //break;
          }
        }
        if (Config::mtvLikeValidation) {
          // Apply MTV selection criteria and then return
          if (simtrack.prodType() != Track::ProdType::Signal || simtrack.charge() == 0 || simtrack.posR() > 2.5 ||
              std::abs(simtrack.z()) > 30 || std::abs(simtrack.momEta()) > 3.0)
            simtrack.setNotFindable();
          else if (Config::mtvRequireSeeds && !isSimSeed)
            simtrack.setNotFindable();
        } else {
          // set findability based on bool isSimSeed
          if (!isSimSeed)
            simtrack.setNotFindable();
        }
        count++;
      }
    }

    void prep_cmsswtracks(Event *event) { prep_reftracks(event, event->cmsswTracks_, event->cmsswTracksExtra_, true); }

    void prep_reftracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks) {
      prep_tracks(event, tracks, extras, realigntracks);

      // mark cmsswtracks as unfindable if too short
      for (auto &track : tracks) {
        const int nlyr = track.nUniqueLayers();
        if (nlyr < Config::cmsSelMinLayers)
          track.setNotFindable();
      }
    }

    void prep_tracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks) {
      for (size_t i = 0; i < tracks.size(); i++) {
        extras.emplace_back(tracks[i].label());
      }
      if (realigntracks)
        event->validation_.alignTracks(tracks, extras, false);
    }

    void score_tracks(TrackVec &tracks) {
      auto score_func = IterationConfig::get_track_scorer("default");
      for (auto &track : tracks) {
        track.setScore(getScoreCand(score_func, track));
      }
    }

  }  // namespace StdSeq

}  // namespace mkfit
