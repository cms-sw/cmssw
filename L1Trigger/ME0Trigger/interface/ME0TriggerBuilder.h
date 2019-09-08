#ifndef L1Trigger_ME0Trigger_ME0TriggerBuilder_h
#define L1Trigger_ME0Trigger_ME0TriggerBuilder_h

/** \class ME0TriggerBuilder
 *
 * Builds ME0 trigger objects from ME0 pad clusters
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/ME0Trigger/interface/ME0Motherboard.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ME0TriggerBuilder {
public:
  /** Configure the algorithm via constructor.
   *  Receives ParameterSet percolated down from
   *  EDProducer which owns this Builder.
   */
  explicit ME0TriggerBuilder(const edm::ParameterSet&);

  ~ME0TriggerBuilder();

  /** Build Triggers from pads or clusters in each chamber and fill them into output collections. */
  template <class T>
  void build(const T* me0Pads, ME0TriggerDigiCollection& oc_trig);

  /** set geometry for the matching needs */
  void setME0Geometry(const ME0Geometry* g) { me0_g = g; }

  /** Max values of trigger labels for all ME0s;
   *  used to construct TMB processors.
   */
  enum trig_me0s { MAX_ENDCAPS = 2, MAX_CHAMBERS = 18 };

private:
  static const int min_endcap;
  static const int max_endcap;
  static const int min_chamber;
  static const int max_chamber;

  const ME0Geometry* me0_g;

  edm::ParameterSet config_;

  /** Pointers to TMB processors for all possible chambers. */
  std::unique_ptr<ME0Motherboard> tmb_[MAX_ENDCAPS][MAX_CHAMBERS];
};

template <class T>
void ME0TriggerBuilder::build(const T* me0Pads, ME0TriggerDigiCollection& oc_trig) {
  for (int endc = 0; endc < 2; endc++) {
    for (int cham = ME0DetId::minChamberId; cham < ME0DetId::maxChamberId; cham++) {
      ME0Motherboard* tmb = tmb_[endc][cham].get();
      tmb->setME0Geometry(me0_g);

      // 0th layer means whole chamber.
      const int region(endc == 0 ? -1 : 1);
      ME0DetId detid(region, 0, cham + 1, 0);

      // Run processors only if chamber exists in geometry.
      if (tmb == nullptr || me0_g->chamber(detid) == nullptr)
        continue;

      tmb->run(me0Pads);

      const std::vector<ME0TriggerDigi>& trigV = tmb->readoutTriggers();

      if (!trigV.empty()) {
        LogTrace("L1ME0Trigger") << "ME0TriggerBuilder got results in " << detid << std::endl
                                 << "Put " << trigV.size() << " Trigger digi" << ((trigV.size() > 1) ? "s " : " ")
                                 << "in collection\n";
        oc_trig.put(std::make_pair(trigV.begin(), trigV.end()), detid);
      }
    }
  }
}

#endif
