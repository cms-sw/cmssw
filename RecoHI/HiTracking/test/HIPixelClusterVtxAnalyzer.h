#ifndef HIPixelClusterVtxAnalyzer_H
#define HIPixelClusterVtxAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ROOT includes
#include <TH1.h>

namespace edm { class Run; class Event; class EventSetup; }

class TrackerGeometry;

class HIPixelClusterVtxAnalyzer : public edm::EDAnalyzer
{
public:
  explicit HIPixelClusterVtxAnalyzer(const edm::ParameterSet& ps);
  ~HIPixelClusterVtxAnalyzer();
 
private:
  struct VertexHit
  {
    float z;
    float r;
    float w;
  };

  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  int getContainedHits(const std::vector<VertexHit> &hits, double z0, double &chi);

  edm::InputTag srcPixels_; //pixel rec hits

  double minZ_;
  double maxZ_;
  double zStep_;

  edm::Service<TFileService> fs;
  int maxHists_;
  int counter;

};
#endif
