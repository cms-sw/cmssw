/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemRPClusterProducerAlgorithm.h"

#include <iostream>

//----------------------------------------------------------------------------------------------------

TotemRPClusterProducerAlgorithm::TotemRPClusterProducerAlgorithm(const edm::ParameterSet &param) : param_(param) {
  verbosity_ = param_.getParameter<int>("verbosity");
}

//----------------------------------------------------------------------------------------------------

TotemRPClusterProducerAlgorithm::~TotemRPClusterProducerAlgorithm() {}

//----------------------------------------------------------------------------------------------------

int TotemRPClusterProducerAlgorithm::buildClusters(unsigned int detId,
                                                   const std::vector<TotemRPDigi> &digi,
                                                   std::vector<TotemRPCluster> &clusters) {
  clusters.clear();

  strip_digi_set_.clear();
  strip_digi_set_.insert(digi.begin(), digi.end());

  if (strip_digi_set_.empty())
    return 0;

  bool iter_beg = true;
  int cluster_beg = -16;
  int cluster_end;
  int prev_strip = -16;
  int cur_strip;

  for (TotemRPDigiSet::const_iterator i = strip_digi_set_.begin(); i != strip_digi_set_.end(); ++i) {
    cur_strip = i->stripNumber();
    bool non_continuity = (cur_strip != prev_strip + 1);

    if (iter_beg) {
      cluster_beg = cur_strip;
      iter_beg = false;
    } else if (non_continuity) {
      cluster_end = prev_strip;
      clusters.push_back(TotemRPCluster((uint16_t)cluster_beg, (uint16_t)cluster_end));

      cluster_beg = cur_strip;
    }

    prev_strip = cur_strip;
  }

  if (!iter_beg) {
    cluster_end = prev_strip;
    clusters.push_back(TotemRPCluster((uint16_t)cluster_beg, (uint16_t)cluster_end));
  }

  return clusters.size();
}
