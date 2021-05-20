#include "L1Trigger/CSCTriggerPrimitives/interface/GEMCoPadProcessor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <set>

//----------------
// Constructors --
//----------------

GEMCoPadProcessor::GEMCoPadProcessor(int region, unsigned station, unsigned chamber, const edm::ParameterSet& conf)
    : theRegion(region), theStation(station), theChamber(chamber) {
  const edm::ParameterSet luts(conf.getParameter<edm::ParameterSet>("gemcscParams"));
  const edm::ParameterSet copad(station == 1 ? conf.getParameter<edm::ParameterSet>("copadParamGE11")
                                             : conf.getParameter<edm::ParameterSet>("copadParamGE21"));

  isEven_ = theChamber % 2 == 0;
  maxDeltaPad_ = copad.getParameter<unsigned int>("maxDeltaPad");
  maxDeltaRoll_ = copad.getParameter<unsigned int>("maxDeltaRoll");
  maxDeltaBX_ = copad.getParameter<unsigned int>("maxDeltaBX");

  padToHsME1aFiles_ = luts.getParameter<std::vector<std::string>>("padToHsME1aFiles");
  padToHsME1bFiles_ = luts.getParameter<std::vector<std::string>>("padToHsME1bFiles");
  padToHsME21Files_ = luts.getParameter<std::vector<std::string>>("padToHsME21Files");

  padToEsME1aFiles_ = luts.getParameter<std::vector<std::string>>("padToEsME1aFiles");
  padToEsME1bFiles_ = luts.getParameter<std::vector<std::string>>("padToEsME1bFiles");
  padToEsME21Files_ = luts.getParameter<std::vector<std::string>>("padToEsME21Files");

  rollToMaxWgME11Files_ = luts.getParameter<std::vector<std::string>>("rollToMaxWgME11Files");
  rollToMinWgME11Files_ = luts.getParameter<std::vector<std::string>>("rollToMinWgME11Files");
  rollToMaxWgME21Files_ = luts.getParameter<std::vector<std::string>>("rollToMaxWgME21Files");
  rollToMinWgME21Files_ = luts.getParameter<std::vector<std::string>>("rollToMinWgME21Files");

  GEMCSCLUT_pad_hs_ME1a_even_ = std::make_unique<CSCLUTReader>(padToHsME1aFiles_[0]);
  GEMCSCLUT_pad_hs_ME1a_odd_ = std::make_unique<CSCLUTReader>(padToHsME1aFiles_[1]);
  GEMCSCLUT_pad_hs_ME1b_even_ = std::make_unique<CSCLUTReader>(padToHsME1bFiles_[0]);
  GEMCSCLUT_pad_hs_ME1b_odd_ = std::make_unique<CSCLUTReader>(padToHsME1bFiles_[1]);
  GEMCSCLUT_pad_hs_ME21_even_ = std::make_unique<CSCLUTReader>(padToHsME21Files_[0]);
  GEMCSCLUT_pad_hs_ME21_odd_ = std::make_unique<CSCLUTReader>(padToHsME21Files_[1]);

  GEMCSCLUT_pad_es_ME1a_even_ = std::make_unique<CSCLUTReader>(padToEsME1aFiles_[0]);
  GEMCSCLUT_pad_es_ME1a_odd_ = std::make_unique<CSCLUTReader>(padToEsME1aFiles_[1]);
  GEMCSCLUT_pad_es_ME1b_even_ = std::make_unique<CSCLUTReader>(padToEsME1bFiles_[0]);
  GEMCSCLUT_pad_es_ME1b_odd_ = std::make_unique<CSCLUTReader>(padToEsME1bFiles_[1]);
  GEMCSCLUT_pad_es_ME21_even_ = std::make_unique<CSCLUTReader>(padToEsME21Files_[0]);
  GEMCSCLUT_pad_es_ME21_odd_ = std::make_unique<CSCLUTReader>(padToEsME21Files_[1]);

  GEMCSCLUT_roll_l1_min_wg_ME11_even_ = std::make_unique<CSCLUTReader>(rollToMinWgME11Files_[0]);
  GEMCSCLUT_roll_l1_min_wg_ME11_odd_ = std::make_unique<CSCLUTReader>(rollToMinWgME11Files_[1]);
  GEMCSCLUT_roll_l2_min_wg_ME11_even_ = std::make_unique<CSCLUTReader>(rollToMinWgME11Files_[2]);
  GEMCSCLUT_roll_l2_min_wg_ME11_odd_ = std::make_unique<CSCLUTReader>(rollToMinWgME11Files_[3]);

  GEMCSCLUT_roll_l1_max_wg_ME11_even_ = std::make_unique<CSCLUTReader>(rollToMaxWgME11Files_[0]);
  GEMCSCLUT_roll_l1_max_wg_ME11_odd_ = std::make_unique<CSCLUTReader>(rollToMaxWgME11Files_[1]);
  GEMCSCLUT_roll_l2_max_wg_ME11_even_ = std::make_unique<CSCLUTReader>(rollToMaxWgME11Files_[2]);
  GEMCSCLUT_roll_l2_max_wg_ME11_odd_ = std::make_unique<CSCLUTReader>(rollToMaxWgME11Files_[3]);

  GEMCSCLUT_roll_l1_min_wg_ME21_even_ = std::make_unique<CSCLUTReader>(rollToMinWgME21Files_[0]);
  GEMCSCLUT_roll_l1_min_wg_ME21_odd_ = std::make_unique<CSCLUTReader>(rollToMinWgME21Files_[1]);
  GEMCSCLUT_roll_l2_min_wg_ME21_even_ = std::make_unique<CSCLUTReader>(rollToMinWgME21Files_[2]);
  GEMCSCLUT_roll_l2_min_wg_ME21_odd_ = std::make_unique<CSCLUTReader>(rollToMinWgME21Files_[3]);

  GEMCSCLUT_roll_l1_max_wg_ME21_even_ = std::make_unique<CSCLUTReader>(rollToMaxWgME21Files_[0]);
  GEMCSCLUT_roll_l1_max_wg_ME21_odd_ = std::make_unique<CSCLUTReader>(rollToMaxWgME21Files_[1]);
  GEMCSCLUT_roll_l2_max_wg_ME21_even_ = std::make_unique<CSCLUTReader>(rollToMaxWgME21Files_[2]);
  GEMCSCLUT_roll_l2_max_wg_ME21_odd_ = std::make_unique<CSCLUTReader>(rollToMaxWgME21Files_[3]);
}

void GEMCoPadProcessor::clear() {
  gemCoPadV.clear();
  clusters_.clear();
}

std::vector<GEMCoPadDigi> GEMCoPadProcessor::run(const GEMPadDigiCollection* in_pads) {
  clear();

  // Build coincidences
  for (auto det_range = in_pads->begin(); det_range != in_pads->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;

    // coincidence pads are not built for ME0
    if (id.isME0())
      continue;

    // same chamber (no restriction on the roll number)
    if (id.region() != theRegion or id.station() != theStation or id.chamber() != theChamber)
      continue;

    // all coincidences detIDs will have layer=1
    if (id.layer() != 1)
      continue;

    // find all corresponding ids with layer 2 and same roll that differs at most maxDeltaRoll_
    for (unsigned int roll = id.roll() - maxDeltaRoll_; roll <= id.roll() + maxDeltaRoll_; ++roll) {
      GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), roll);

      auto co_pads_range = in_pads->get(co_id);

      // empty range = no possible coincidence pads
      if (co_pads_range.first == co_pads_range.second)
        continue;

      // now let's correlate the pads in two layers of this partition
      const auto& pads_range = (*det_range).second;
      for (auto p = pads_range.first; p != pads_range.second; ++p) {
        // ignore 16-partition GE2/1 pads
        if (id.isGE21() and p->nPartitions() == GEMPadDigi::GE21SplitStrip)
          continue;

        // only consider valid pads
        if (!p->isValid())
          continue;

        for (auto co_p = co_pads_range.first; co_p != co_pads_range.second; ++co_p) {
          // only consider valid pads
          if (!co_p->isValid())
            continue;

          // check the match in pad
          if ((unsigned)std::abs(p->pad() - co_p->pad()) > maxDeltaPad_)
            continue;

          // check the match in BX
          if ((unsigned)std::abs(p->bx() - co_p->bx()) > maxDeltaBX_)
            continue;

          // make a new coincidence pad digi
          gemCoPadV.push_back(GEMCoPadDigi(id.roll(), *p, *co_p));
        }
      }
    }
  }
  return gemCoPadV;
}

std::vector<GEMInternalCluster> GEMCoPadProcessor::run(const GEMPadDigiClusterCollection* in_clusters) {
  // Step 1: clear the GEMInternalCluster vector
  clear();

  // Step 2: put coincidence clusters in GEMInternalCluster vector
  addCoincidenceClusters(in_clusters);

  // Step 3: put single clusters in GEMInternalCluster vector who are not
  // part of any coincidence cluster
  addSingleClusters(in_clusters);

  // Step 4: translate the cluster central pad numbers into 1/8-strip number for matching with CSC trigger primitives
  doCoordinateConversion();

  // Step 5: return the vector with internal clusters
  return clusters_;
}

void GEMCoPadProcessor::addCoincidenceClusters(const GEMPadDigiClusterCollection* in_clusters) {
  // Build coincidences
  for (auto det_range = in_clusters->begin(); det_range != in_clusters->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;

    // coincidence pads are not built for ME0
    if (id.isME0())
      continue;

    // same chamber (no restriction on the roll number)
    if (id.region() != theRegion or id.station() != theStation or id.chamber() != theChamber)
      continue;

    // all coincidences detIDs will have layer=1
    if (id.layer() != 1)
      continue;

    // find all corresponding ids with layer 2 and same roll that differs at most maxDeltaRoll_
    for (unsigned int roll = id.roll() - maxDeltaRoll_; roll <= id.roll() + maxDeltaRoll_; ++roll) {
      GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), roll);

      auto co_clusters_range = in_clusters->get(co_id);

      // empty range = no possible coincidence pads
      if (co_clusters_range.first == co_clusters_range.second)
        continue;

      // now let's correlate the pads in two layers of this partition
      const auto& pads_range = (*det_range).second;
      for (auto p = pads_range.first; p != pads_range.second; ++p) {
        // ignore 16-partition GE2/1 pads
        if (id.isGE21() and p->nPartitions() == GEMPadDigi::GE21SplitStrip)
          continue;

        // only consider valid pads
        if (!p->isValid())
          continue;

        for (auto co_p = co_clusters_range.first; co_p != co_clusters_range.second; ++co_p) {
          // only consider valid clusters
          if (!co_p->isValid())
            continue;

          // check the match in BX
          if ((unsigned)std::abs(p->bx() - co_p->bx()) > maxDeltaBX_)
            continue;

          // check the match in pad
          int cl1f = p->pads().front();
          int cl1b = p->pads().back();
          int cl2f = co_p->pads().front();
          int cl2b = co_p->pads().back();

          unsigned deltaFF = std::abs(cl1f - cl2f);
          unsigned deltaFB = std::abs(cl1f - cl2b);
          unsigned deltaBF = std::abs(cl1b - cl2f);
          unsigned deltaBB = std::abs(cl1b - cl2b);

          std::vector<unsigned> v{deltaFF, deltaFB, deltaBF, deltaBB};

          if (*std::min_element(v.begin(), v.end()) > maxDeltaPad_)
            continue;

          // make a new coincidence
          clusters_.emplace_back(id, *p, *co_p);
        }
      }
    }
  }
}

void GEMCoPadProcessor::addSingleClusters(const GEMPadDigiClusterCollection* in_clusters) {
  // now start add single clusters
  for (auto det_range = in_clusters->begin(); det_range != in_clusters->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;

    // coincidence pads are not built for ME0
    if (id.isME0())
      continue;

    // same chamber (no restriction on the roll number)
    if (id.region() != theRegion or id.station() != theStation or id.chamber() != theChamber)
      continue;

    const auto& clusters_range = (*det_range).second;
    for (auto p = clusters_range.first; p != clusters_range.second; ++p) {
      // only consider valid clusters
      if (p->isValid())
        continue;

      // ignore 16-partition GE2/1 pads
      if (id.isGE21() and p->nPartitions() == GEMPadDigiCluster::GE21SplitStrip)
        continue;

      // ignore clusters already contained in a coincidence cluster
      if (std::find_if(std::begin(clusters_), std::end(clusters_), [p](const GEMInternalCluster& q) {
            return q.has_cluster(*p);
          }) != std::end(clusters_))
        continue;

      // put the single clusters into the collection
      if (id.layer() == 1)
        clusters_.emplace_back(id, *p, GEMPadDigiCluster());
      else
        clusters_.emplace_back(id, GEMPadDigiCluster(), *p);
    }
  }
}

void GEMCoPadProcessor::doCoordinateConversion() {
  // loop on clusters
  for (auto& cluster : clusters_) {
    // starting coordinates
    const int layer1_first_pad = cluster.layer1_pad();
    const int layer2_first_pad = cluster.layer2_pad();
    const int layer1_last_pad = layer1_first_pad + cluster.layer1_size() - 1;
    const int layer2_last_pad = layer2_first_pad + cluster.layer2_size() - 1;

    // calculate the  1/2-strip
    int layer1_pad_to_first_hs = -1;
    int layer1_pad_to_last_hs = -1;

    int layer2_pad_to_first_hs = -1;
    int layer2_pad_to_last_hs = -1;

    int layer1_pad_to_first_hs_me1a = -1;
    int layer1_pad_to_last_hs_me1a = -1;

    int layer2_pad_to_first_hs_me1a = -1;
    int layer2_pad_to_last_hs_me1a = -1;

    if (theStation == 1) {
      if (isEven_) {
        // ME1/b
        layer1_pad_to_first_hs = GEMCSCLUT_pad_hs_ME1b_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs = GEMCSCLUT_pad_hs_ME1b_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs = GEMCSCLUT_pad_hs_ME1b_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs = GEMCSCLUT_pad_hs_ME1b_even_->lookup(layer2_last_pad);
        // ME1/a
        layer1_pad_to_first_hs_me1a = GEMCSCLUT_pad_hs_ME1a_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs_me1a = GEMCSCLUT_pad_hs_ME1a_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs_me1a = GEMCSCLUT_pad_hs_ME1a_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs_me1a = GEMCSCLUT_pad_hs_ME1a_even_->lookup(layer2_last_pad);
      } else {
        // ME1/b
        layer1_pad_to_first_hs = GEMCSCLUT_pad_hs_ME1b_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs = GEMCSCLUT_pad_hs_ME1b_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs = GEMCSCLUT_pad_hs_ME1b_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs = GEMCSCLUT_pad_hs_ME1b_odd_->lookup(layer2_last_pad);
        // ME1/a
        layer1_pad_to_first_hs_me1a = GEMCSCLUT_pad_hs_ME1a_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs_me1a = GEMCSCLUT_pad_hs_ME1a_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs_me1a = GEMCSCLUT_pad_hs_ME1a_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs_me1a = GEMCSCLUT_pad_hs_ME1a_odd_->lookup(layer2_last_pad);
      }
    }

    // ME2/1
    if (theStation == 2) {
      if (isEven_) {
        layer1_pad_to_first_hs = GEMCSCLUT_pad_hs_ME21_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs = GEMCSCLUT_pad_hs_ME21_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs = GEMCSCLUT_pad_hs_ME21_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs = GEMCSCLUT_pad_hs_ME21_even_->lookup(layer2_last_pad);
      } else {
        layer1_pad_to_first_hs = GEMCSCLUT_pad_hs_ME21_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_hs = GEMCSCLUT_pad_hs_ME21_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_hs = GEMCSCLUT_pad_hs_ME21_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_hs = GEMCSCLUT_pad_hs_ME21_odd_->lookup(layer2_last_pad);
      }
    }

    // middle 1/2-strip
    int layer1_middle_hs = 0.5 * (layer1_pad_to_first_hs + layer1_pad_to_last_hs);
    int layer2_middle_hs = 0.5 * (layer2_pad_to_first_hs + layer2_pad_to_last_hs);
    int layer1_middle_hs_me1a = 0.5 * (layer1_pad_to_first_hs_me1a + layer1_pad_to_last_hs_me1a);
    int layer2_middle_hs_me1a = 0.5 * (layer2_pad_to_first_hs_me1a + layer2_pad_to_last_hs_me1a);

    // set the values
    cluster.set_layer1_first_hs(layer1_pad_to_first_hs);
    cluster.set_layer2_first_hs(layer2_pad_to_first_hs);
    cluster.set_layer1_last_hs(layer1_pad_to_last_hs);
    cluster.set_layer2_last_hs(layer2_pad_to_last_hs);

    // middle 1/2-strip
    cluster.set_layer1_middle_hs(layer1_middle_hs);
    cluster.set_layer2_middle_hs(layer2_middle_hs);

    if (theStation == 1) {
      cluster.set_layer1_first_hs_me1a(layer1_pad_to_first_hs_me1a);
      cluster.set_layer2_first_hs_me1a(layer2_pad_to_first_hs_me1a);
      cluster.set_layer1_last_hs_me1a(layer1_pad_to_last_hs_me1a);
      cluster.set_layer2_last_hs_me1a(layer2_pad_to_last_hs_me1a);

      // middle 1/2-strip
      cluster.set_layer1_middle_hs_me1a(layer1_middle_hs_me1a);
      cluster.set_layer2_middle_hs_me1a(layer2_middle_hs_me1a);
    }

    // calculate the 1/8-strips
    int layer1_pad_to_first_es = -1;
    int layer1_pad_to_last_es = -1;

    int layer2_pad_to_first_es = -1;
    int layer2_pad_to_last_es = -1;

    int layer1_pad_to_first_es_me1a = -1;
    int layer1_pad_to_last_es_me1a = -1;

    int layer2_pad_to_first_es_me1a = -1;
    int layer2_pad_to_last_es_me1a = -1;

    if (theStation == 1) {
      if (isEven_) {
        // ME1/b
        layer1_pad_to_first_es = GEMCSCLUT_pad_es_ME1b_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_es = GEMCSCLUT_pad_es_ME1b_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_es = GEMCSCLUT_pad_es_ME1b_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_es = GEMCSCLUT_pad_es_ME1b_even_->lookup(layer2_last_pad);
        // ME1/a
        layer1_pad_to_first_es_me1a = GEMCSCLUT_pad_es_ME1a_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_es_me1a = GEMCSCLUT_pad_es_ME1a_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_es_me1a = GEMCSCLUT_pad_es_ME1a_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_es_me1a = GEMCSCLUT_pad_es_ME1a_even_->lookup(layer2_last_pad);
      } else {
        // ME1/b
        layer1_pad_to_first_es = GEMCSCLUT_pad_es_ME1b_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_es = GEMCSCLUT_pad_es_ME1b_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_es = GEMCSCLUT_pad_es_ME1b_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_es = GEMCSCLUT_pad_es_ME1b_odd_->lookup(layer2_last_pad);
        // ME1/a
        layer1_pad_to_first_es_me1a = GEMCSCLUT_pad_es_ME1a_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_es_me1a = GEMCSCLUT_pad_es_ME1a_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_es_me1a = GEMCSCLUT_pad_es_ME1a_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_es_me1a = GEMCSCLUT_pad_es_ME1a_odd_->lookup(layer2_last_pad);
      }
    }

    // ME2/1
    if (theStation == 2) {
      if (isEven_) {
        layer1_pad_to_first_es = GEMCSCLUT_pad_es_ME21_even_->lookup(layer1_first_pad);
        layer2_pad_to_first_es = GEMCSCLUT_pad_es_ME21_even_->lookup(layer2_first_pad);
        layer1_pad_to_last_es = GEMCSCLUT_pad_es_ME21_even_->lookup(layer1_last_pad);
        layer2_pad_to_last_es = GEMCSCLUT_pad_es_ME21_even_->lookup(layer2_last_pad);
      } else {
        layer1_pad_to_first_es = GEMCSCLUT_pad_es_ME21_odd_->lookup(layer1_first_pad);
        layer2_pad_to_first_es = GEMCSCLUT_pad_es_ME21_odd_->lookup(layer2_first_pad);
        layer1_pad_to_last_es = GEMCSCLUT_pad_es_ME21_odd_->lookup(layer1_last_pad);
        layer2_pad_to_last_es = GEMCSCLUT_pad_es_ME21_odd_->lookup(layer2_last_pad);
      }
    }

    // middle 1/8-strip
    int layer1_middle_es = 0.5 * (layer1_pad_to_first_es + layer1_pad_to_last_es);
    int layer2_middle_es = 0.5 * (layer2_pad_to_first_es + layer2_pad_to_last_es);
    int layer1_middle_es_me1a = 0.5 * (layer1_pad_to_first_es_me1a + layer1_pad_to_last_es_me1a);
    int layer2_middle_es_me1a = 0.5 * (layer2_pad_to_first_es_me1a + layer2_pad_to_last_es_me1a);

    // set the values
    cluster.set_layer1_first_es(layer1_pad_to_first_es);
    cluster.set_layer2_first_es(layer2_pad_to_first_es);
    cluster.set_layer1_last_es(layer1_pad_to_last_es);
    cluster.set_layer2_last_es(layer2_pad_to_last_es);

    // middle 1/8-strip
    cluster.set_layer1_middle_es(layer1_middle_es);
    cluster.set_layer2_middle_es(layer2_middle_es);

    if (theStation == 1) {
      cluster.set_layer1_first_es_me1a(layer1_pad_to_first_es_me1a);
      cluster.set_layer2_first_es_me1a(layer2_pad_to_first_es_me1a);
      cluster.set_layer1_last_es_me1a(layer1_pad_to_last_es_me1a);
      cluster.set_layer2_last_es_me1a(layer2_pad_to_last_es_me1a);

      // middle 1/8-strip
      cluster.set_layer1_middle_es_me1a(layer1_middle_es_me1a);
      cluster.set_layer2_middle_es_me1a(layer2_middle_es_me1a);
    }

    // calculate the wiregroups
    const int roll = cluster.roll();

    int roll_l1_to_min_wg = -1;
    int roll_l1_to_max_wg = -1;
    int roll_l2_to_min_wg = -1;
    int roll_l2_to_max_wg = -1;

    // ME1/1
    if (theStation == 1) {
      if (isEven_) {
        roll_l1_to_min_wg = GEMCSCLUT_roll_l1_min_wg_ME11_even_->lookup(roll);
        roll_l1_to_max_wg = GEMCSCLUT_roll_l1_max_wg_ME11_even_->lookup(roll);
        roll_l2_to_min_wg = GEMCSCLUT_roll_l2_min_wg_ME11_even_->lookup(roll);
        roll_l2_to_max_wg = GEMCSCLUT_roll_l2_max_wg_ME11_even_->lookup(roll);
      } else {
        roll_l1_to_min_wg = GEMCSCLUT_roll_l1_min_wg_ME11_odd_->lookup(roll);
        roll_l1_to_max_wg = GEMCSCLUT_roll_l1_max_wg_ME11_odd_->lookup(roll);
        roll_l2_to_min_wg = GEMCSCLUT_roll_l2_min_wg_ME11_odd_->lookup(roll);
        roll_l2_to_max_wg = GEMCSCLUT_roll_l2_max_wg_ME11_odd_->lookup(roll);
      }
    }

    // ME2/1
    if (theStation == 2) {
      if (isEven_) {
        roll_l1_to_min_wg = GEMCSCLUT_roll_l1_min_wg_ME21_even_->lookup(roll);
        roll_l1_to_max_wg = GEMCSCLUT_roll_l1_max_wg_ME21_even_->lookup(roll);
        roll_l2_to_min_wg = GEMCSCLUT_roll_l2_min_wg_ME21_even_->lookup(roll);
        roll_l2_to_max_wg = GEMCSCLUT_roll_l2_max_wg_ME21_even_->lookup(roll);
      } else {
        roll_l1_to_min_wg = GEMCSCLUT_roll_l1_min_wg_ME21_odd_->lookup(roll);
        roll_l1_to_max_wg = GEMCSCLUT_roll_l1_max_wg_ME21_odd_->lookup(roll);
        roll_l2_to_min_wg = GEMCSCLUT_roll_l2_min_wg_ME21_odd_->lookup(roll);
        roll_l2_to_max_wg = GEMCSCLUT_roll_l2_max_wg_ME21_odd_->lookup(roll);
      }
    }

    // set the values
    cluster.set_layer1_min_wg(roll_l1_to_min_wg);
    cluster.set_layer1_max_wg(roll_l1_to_max_wg);
    cluster.set_layer2_min_wg(roll_l2_to_min_wg);
    cluster.set_layer2_max_wg(roll_l2_to_max_wg);
  }
}

const std::vector<GEMCoPadDigi>& GEMCoPadProcessor::readoutCoPads() const { return gemCoPadV; }

void GEMCoPadProcessor::declusterize(const GEMPadDigiClusterCollection* in_clusters,
                                     GEMPadDigiCollection& out_pads) const {
  for (auto detUnitIt = in_clusters->begin(); detUnitIt != in_clusters->end(); ++detUnitIt) {
    const GEMDetId& id = (*detUnitIt).first;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
      // coincidence pads are not built for ME0
      if (id.isME0())
        continue;

      // same chamber (no restriction on the roll number)
      if (id.region() != theRegion or id.station() != theStation or id.chamber() != theChamber)
        continue;

      // ignore 16-partition GE2/1 pads
      if (id.isGE21() and digiIt->nPartitions() == GEMPadDigiCluster::GE21SplitStrip)
        continue;

      // only consider valid clusters
      if (!digiIt->isValid())
        continue;

      for (const auto& p : digiIt->pads()) {
        out_pads.insertDigi(id, GEMPadDigi(p, digiIt->bx()));
      }
    }
  }
}
