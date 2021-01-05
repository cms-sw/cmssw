#ifndef Alignment_OfflineValidation_PixelBaryCentreAnalyzer_H
#define Alignment_OfflineValidation_PixelBaryCentreAnalyzer_H

/** \class PixelBaryCentreAnalyzer
 *  The analyer works as the following :
 *  - Read global tracker position from global tag
 *  - Read tracker alignment constants from different ESsource with different labels 
 *  - Calculate barycentres for different pixel substructures using global tracker position and alignment constants
 *  and store them in trees, one for each ESsource label.
 *
 *  Python script plotBaryCentre_VS_BeamSpot.py under script dir is used to plot barycentres from alignment constants used in Prompt-Reco, End-of-Year Rereco and so-called Run-2 (Ultra)Legacy Rereco. Options of the plotting script can be found from the helper in the script.
 *
 *  $Date: 2021/01/05 $ 
 *  $Revision: 1.0 $
 *  \author Tongguang Cheng - Beihang University <tongguang.cheng@cern.ch>
 */

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// Phase-1 Pixel
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// pixel quality
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
// global postion
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
// tracker alignment
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"
// beamspot
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

// TFileService
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ROOT
#include <TTree.h>
#include <TString.h>
#include <TVector3.h>

//
// class declaration
//

class PixelBaryCentreAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PixelBaryCentreAnalyzer(const edm::ParameterSet &);
  ~PixelBaryCentreAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  void initBC();
  void initBS();

  bool usePixelQuality_;
  int phase_;

  // ----------member data ---------------------------
  edm::ESWatcher<BeamSpotObjectsRcd> watcherBS_;
  edm::ESWatcher<TrackerAlignmentRcd> watcherTkAlign_;

  // labels of TkAlign tags
  std::vector<std::string> bcLabels_;
  // labels of beamspot tags
  std::vector<std::string> bsLabels_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> siPixelQualityToken_;

  // tree content
  int run_;
  int ls_;

  double BSx0_, BSy0_, BSz0_;
  TVector3 BS_;

  double PIXx0_, PIXy0_, PIXz0_;
  TVector3 PIX_, BPIX_, FPIX_;
  TVector3 BPIX_Flipped_, BPIX_NonFlipped_, BPIX_DiffFlippedNonFlipped_;

  TVector3 BPIXLayer_[4];
  TVector3 BPIXLayer_Flipped_[4];
  TVector3 BPIXLayer_NonFlipped_[4];
  TVector3 BPIXLayer_DiffFlippedNonFlipped_[4];
  //// number of modules for each BPIX layer : flipped and non-flipped separately
  //int nmodules_BPIXLayer_Flipped_[4];
  //int nmodules_BPIXLayer_NonFlipped_[4];

  edm::Service<TFileService> tFileService;
  std::map<std::string, TTree *> bcTrees_;
  std::map<std::string, TTree *> bsTrees_;
};

#endif
