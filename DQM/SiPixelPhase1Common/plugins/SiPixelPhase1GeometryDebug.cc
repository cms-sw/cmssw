// -*- C++ -*-
// 
// Package:     SiPixelPhase1GeometryDebug
// Class  :     SiPixelPhase1GeometryDebug
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// This small plugin plots out varois geometry quantities against each other. 
// This is useful to see where a specific module or ROC ends up in a 2D map.
class SiPixelPhase1GeometryDebug : public SiPixelPhase1Base {
  enum {
    DETID,
    LADBLD,
    ROC,
    FED
  };

  public:
  explicit SiPixelPhase1GeometryDebug(const edm::ParameterSet& conf) 
    : SiPixelPhase1Base(conf) {
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup&) {
    auto& all = geometryInterface.allModules();
    GeometryInterface::Column ladderblade = {{geometryInterface.intern("PXLadder"), geometryInterface.intern("PXBlade")}};
    GeometryInterface::Column roc = {{geometryInterface.intern("ROC"), 0}};
    GeometryInterface::Column fed = {{geometryInterface.intern("FED"), 0}};

    for (auto iq : all) {
      auto ladbld = geometryInterface.extract(ladderblade, iq);
      auto rocno = geometryInterface.extract(roc, iq);
      auto fedno = geometryInterface.extract(fed, iq);
      auto detid = iq.sourceModule.rawId();

      histo[DETID ].fill((float) detid,         iq.sourceModule, &iEvent, iq.col, iq.row);
      histo[LADBLD].fill((float) ladbld.second, iq.sourceModule, &iEvent, iq.col, iq.row);
      histo[ROC   ].fill((float) rocno.second,  iq.sourceModule, &iEvent, iq.col, iq.row);
      histo[FED   ].fill((float) fedno.second,  iq.sourceModule, &iEvent, iq.col, iq.row);
    }
  }
};

DEFINE_FWK_MODULE(SiPixelPhase1GeometryDebug);

