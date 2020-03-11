// -*- C++ -*-

// Package:    PixelLumiDQM
// Class:      PixelLumiDQM

// Author: Amita Raval
// Based on Jeroen Hegeman's code for Pixel Cluster Count Luminosity

#include "PixelLumiDQM.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <sys/time.h>
#include <vector>

const unsigned int PixelLumiDQM::lastBunchCrossing;

// Constructors and destructor.
PixelLumiDQM::PixelLumiDQM(const edm::ParameterSet &iConfig)
    : fPixelClusterLabel(consumes<edmNew::DetSetVector<SiPixelCluster>>(
          iConfig.getUntrackedParameter<edm::InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters")))),
      fIncludePixelClusterInfo(iConfig.getUntrackedParameter<bool>("includePixelClusterInfo", true)),
      fIncludePixelQualCheckHistos(iConfig.getUntrackedParameter<bool>("includePixelQualCheckHistos", true)),
      fResetIntervalInLumiSections(iConfig.getUntrackedParameter<int>("resetEveryNLumiSections", 1)),
      fDeadModules(iConfig.getUntrackedParameter<std::vector<uint32_t>>("deadModules", std::vector<uint32_t>())),
      fMinPixelsPerCluster(iConfig.getUntrackedParameter<int>("minNumPixelsPerCluster", 0)),
      fMinClusterCharge(iConfig.getUntrackedParameter<double>("minChargePerCluster", 0)),
      bunchTriggerMask(lastBunchCrossing + 1, false),
      filledAndUnmaskedBunches(0),
      useInnerBarrelLayer(iConfig.getUntrackedParameter<bool>("useInnerBarrelLayer", false)),
      fLogFileName_(iConfig.getUntrackedParameter<std::string>("logFileName", "/tmp/pixel_lumi.txt")) {
  edm::LogInfo("Configuration") << "PixelLumiDQM looking for pixel clusters in '"
                                << iConfig.getUntrackedParameter<edm::InputTag>("pixelClusterLabel",
                                                                                edm::InputTag("siPixelClusters"))
                                << "'";
  edm::LogInfo("Configuration") << "PixelLumiDQM storing pixel cluster info? " << fIncludePixelClusterInfo;
  edm::LogInfo("Configuration") << "PixelLumiDQM storing pixel cluster quality check histograms? "
                                << fIncludePixelQualCheckHistos;

  if (fDeadModules.empty()) {
    edm::LogInfo("Configuration") << "No pixel modules specified to be ignored";
  } else {
    edm::LogInfo("Configuration") << fDeadModules.size() << " pixel modules specified to be ignored:";
    for (std::vector<uint32_t>::const_iterator it = fDeadModules.begin(); it != fDeadModules.end(); ++it) {
      edm::LogInfo("Configuration") << "  " << *it;
    }
  }
  edm::LogInfo("Configuration") << "Ignoring pixel clusters with less than " << fMinPixelsPerCluster << " pixels";
  edm::LogInfo("Configuration") << "Ignoring pixel clusters with charge less than " << fMinClusterCharge;
}

PixelLumiDQM::~PixelLumiDQM() {}

void PixelLumiDQM::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void PixelLumiDQM::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Collect all bookkeeping information.
  fRunNo = iEvent.id().run();
  fEvtNo = iEvent.id().event();
  fLSNo = iEvent.getLuminosityBlock().luminosityBlock();
  fBXNo = iEvent.bunchCrossing();
  fTimestamp = iEvent.time().unixTime();
  fHistBunchCrossings->Fill(float(fBXNo));
  fHistBunchCrossingsLastLumi->Fill(float(fBXNo));
  // This serves as event counter to compute luminosity from cluster counts.
  std::map<int, PixelClusterCount>::iterator it = fNumPixelClusters.find(fBXNo);
  if (it == fNumPixelClusters.end())
    fNumPixelClusters[fBXNo] = PixelClusterCount();

  if (fIncludePixelClusterInfo) {
    // Find tracker geometry.
    edm::ESHandle<TrackerGeometry> trackerGeo;
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeo);

    // Find pixel clusters.
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelClusters;
    iEvent.getByToken(fPixelClusterLabel, pixelClusters);

    // Loop over entire tracker geometry.
    for (TrackerGeometry::DetContainer::const_iterator i = trackerGeo->dets().begin(); i != trackerGeo->dets().end();
         ++i) {
      // See if this is a pixel unit(?).

      if (GeomDetEnumerators::isTrackerPixel((*i)->subDetector())) {
        DetId detId = (*i)->geographicalId();
        // Find all clusters on this detector module.
        edmNew::DetSetVector<SiPixelCluster>::const_iterator iSearch = pixelClusters->find(detId);
        if (iSearch != pixelClusters->end()) {
          // Count the number of clusters with at least a minimum
          // number of pixels per cluster and at least a minimum charge.
          size_t numClusters = 0;
          for (edmNew::DetSet<SiPixelCluster>::const_iterator itClus = iSearch->begin(); itClus != iSearch->end();
               ++itClus) {
            if ((itClus->size() >= fMinPixelsPerCluster) && (itClus->charge() >= fMinClusterCharge)) {
              ++numClusters;
            }
          }
          // DEBUG DEBUG DEBUG
          assert(numClusters <= iSearch->size());
          // DEBUG DEBUG DEBUG end

          // Add up the cluster count based on the position of this detector
          // element.
          if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
            PixelBarrelNameUpgrade detName = PixelBarrelNameUpgrade(detId);
            int layer = detName.layerName() - kOffsetLayers;
            fNumPixelClusters[fBXNo].numB.at(layer) += numClusters;
            fNumPixelClusters[fBXNo].dnumB.at(layer) += sqrt(numClusters);
          } else {
            // DEBUG DEBUG DEBUG
            assert(detId.subdetId() == PixelSubdetector::PixelEndcap);
            // DEBUG DEBUG DEBUG end

            PixelEndcapNameUpgrade detName = PixelEndcapNameUpgrade(detId);
            PixelEndcapNameUpgrade::HalfCylinder halfCylinder = detName.halfCylinder();
            int disk = detName.diskName() - kOffsetDisks;
            switch (halfCylinder) {
              case PixelEndcapNameUpgrade::mO:
              case PixelEndcapNameUpgrade::mI:
                fNumPixelClusters[fBXNo].numFM.at(disk) += numClusters;
                fNumPixelClusters[fBXNo].dnumFM.at(disk) += sqrt(numClusters);
                break;
              case PixelEndcapNameUpgrade::pO:
              case PixelEndcapNameUpgrade::pI:
                fNumPixelClusters[fBXNo].numFP.at(disk) += numClusters;
                fNumPixelClusters[fBXNo].dnumFP.at(disk) += sqrt(numClusters);
                break;
              default:
                assert(false);
                break;
            }
          }
        }
      }
    }
  }
  // ----------

  // Fill some pixel cluster quality check histograms if requested.
  if (fIncludePixelQualCheckHistos) {
    // Find tracker geometry.
    edm::ESHandle<TrackerGeometry> trackerGeo;
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeo);

    // Find pixel clusters.
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelClusters;
    iEvent.getByToken(fPixelClusterLabel, pixelClusters);

    bool filterDeadModules = (!fDeadModules.empty());
    std::vector<uint32_t>::const_iterator deadModulesBegin = fDeadModules.begin();
    std::vector<uint32_t>::const_iterator deadModulesEnd = fDeadModules.end();

    // Loop over entire tracker geometry.
    for (TrackerGeometry::DetContainer::const_iterator i = trackerGeo->dets().begin(); i != trackerGeo->dets().end();
         ++i) {
      // See if this is a pixel module.
      if (GeomDetEnumerators::isTrackerPixel((*i)->subDetector())) {
        DetId detId = (*i)->geographicalId();

        // Skip this module if it's on the list of modules to be ignored.
        if (filterDeadModules && find(deadModulesBegin, deadModulesEnd, detId()) != deadModulesEnd) {
          continue;
        }

        // Find all clusters in this module.
        edmNew::DetSetVector<SiPixelCluster>::const_iterator iSearch = pixelClusters->find(detId);

        // Loop over all clusters in this module.
        if (iSearch != pixelClusters->end()) {
          for (edmNew::DetSet<SiPixelCluster>::const_iterator clus = iSearch->begin(); clus != iSearch->end(); ++clus) {
            if ((clus->size() >= fMinPixelsPerCluster) && (clus->charge() >= fMinClusterCharge)) {
              PixelGeomDetUnit const *theGeomDet = dynamic_cast<PixelGeomDetUnit const *>(trackerGeo->idToDet(detId));
              PixelTopology const *topol = &(theGeomDet->specificTopology());
              double x = clus->x();
              double y = clus->y();
              LocalPoint clustLP = topol->localPosition(MeasurementPoint(x, y));
              GlobalPoint clustGP = theGeomDet->surface().toGlobal(clustLP);
              double charge = clus->charge() / 1.e3;
              int size = clus->size();

              if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
                PixelBarrelNameUpgrade detName = PixelBarrelNameUpgrade(detId);
                unsigned int layer = detName.layerName() - kOffsetLayers;
                if (layer < kNumLayers) {
                  std::string histName;
                  histName = "clusPosBarrel" + std::to_string(layer);
                  fHistContainerThisRun[histName]->Fill(clustGP.z(), clustGP.phi());
                  histName = "clusChargeBarrel" + std::to_string(layer);
                  fHistContainerThisRun[histName]->Fill(iEvent.bunchCrossing(), charge);
                  histName = "clusSizeBarrel" + std::to_string(layer);
                  fHistContainerThisRun[histName]->Fill(iEvent.bunchCrossing(), size);
                } else {
                  edm::LogWarning("pixelLumi") << "higher layer number, " << layer << ", than layers";
                }
              } else {
                // DEBUG DEBUG DEBUG
                assert(detId.subdetId() == PixelSubdetector::PixelEndcap);
                // DEBUG DEBUG DEBUG end

                PixelEndcapNameUpgrade detName = PixelEndcapNameUpgrade(detId);
                unsigned int disk = detName.diskName() - kOffsetDisks;
                if (disk < kNumDisks) {
                  std::string histName;
                  histName = "clusPosEndCap" + std::to_string(disk);
                  fHistContainerThisRun[histName]->Fill(clustGP.x(), clustGP.y());
                  histName = "clusChargeEndCap" + std::to_string(disk);
                  fHistContainerThisRun[histName]->Fill(iEvent.bunchCrossing(), charge);
                  histName = "clusSizeEndCap" + std::to_string(disk);
                  fHistContainerThisRun[histName]->Fill(iEvent.bunchCrossing(), size);
                } else {
                  edm::LogWarning("pixelLumi")
                      << "higher disk number, " << disk << ", than disks," << kNumDisks << std::endl;
                }
              }
            }
          }
        }
      }
    }
  }
}

void PixelLumiDQM::bookHistograms(DQMStore::IBooker &ibooker,
                                  edm::Run const &run,
                                  edm::EventSetup const & /* iSetup */) {
  edm::LogInfo("Status") << "Starting processing of run #" << run.id().run();

  // Top folder containing high-level information about pixel and HF lumi.
  std::string folder = "PixelLumi/";
  ibooker.setCurrentFolder(folder);

  fHistTotalRecordedLumiByLS = ibooker.book1D("totalPixelLumiByLS", "Pixel Lumi in nb vs LS", 8000, 0.5, 8000.5);
  fHistRecordedByBxCumulative = ibooker.book1D("PXLumiByBXsum",
                                               "Pixel Lumi in nb by BX Cumulative vs LS",
                                               lastBunchCrossing,
                                               0.5,
                                               float(lastBunchCrossing) + 0.5);

  std::string subfolder = folder + "lastLS/";
  ibooker.setCurrentFolder(subfolder);
  fHistRecordedByBxLastLumi = ibooker.book1D(
      "PXByBXLastLumi", "Pixel By BX Last Lumi", lastBunchCrossing + 1, -0.5, float(lastBunchCrossing) + 0.5);

  subfolder = folder + "ClusterCountingDetails/";
  ibooker.setCurrentFolder(subfolder);

  fHistnBClusVsLS[0] = ibooker.book1D("nBClusVsLS_0", "Fraction of Clusters vs LS Barrel layer 0", 8000, 0.5, 8000.5);
  fHistnBClusVsLS[1] = ibooker.book1D("nBClusVsLS_1", "Fraction of Clusters vs LS Barrel layer 1", 8000, 0.5, 8000.5);
  fHistnBClusVsLS[2] = ibooker.book1D("nBClusVsLS_2", "Fraction of Clusters vs LS Barrel layer 2", 8000, 0.5, 8000.5);
  fHistnFPClusVsLS[0] = ibooker.book1D("nFPClusVsLS_0", "Fraction of Clusters vs LS Barrel layer 0", 8000, 0.5, 8000.5);
  fHistnFPClusVsLS[1] = ibooker.book1D("nFPClusVsLS_1", "Fraction of Clusters vs LS Barrel layer 1", 8000, 0.5, 8000.5);
  fHistnFMClusVsLS[0] = ibooker.book1D("nFMClusVsLS_0", "Fraction of Clusters vs LS Barrel layer 0", 8000, 0.5, 8000.5);
  fHistnFMClusVsLS[1] = ibooker.book1D("nFMClusVsLS_1", "Fraction of Clusters vs LS Barrel layer 1", 8000, 0.5, 8000.5);
  fHistBunchCrossings = ibooker.book1D(
      "BunchCrossings", "Cumulative Bunch Crossings", lastBunchCrossing, 0.5, float(lastBunchCrossing) + 0.5);
  fHistBunchCrossingsLastLumi = ibooker.book1D(
      "BunchCrossingsLL", "Bunch Crossings Last Lumi", lastBunchCrossing, 0.5, float(lastBunchCrossing) + 0.5);
  fHistClusterCountByBxLastLumi = ibooker.book1D(
      "ClusterCountByBxLL", "Cluster Count by BX Last Lumi", lastBunchCrossing, 0.5, float(lastBunchCrossing) + 0.5);
  fHistClusterCountByBxCumulative = ibooker.book1D(
      "ClusterCountByBxSum", "Cluster Count by BX Cumulative", lastBunchCrossing, 0.5, float(lastBunchCrossing) + 0.5);
  fHistClusByLS = ibooker.book1D("totalClusByLS", "Number of Clusters all dets vs LS", 8000, 0.5, 8000.5);

  // Add some pixel cluster quality check histograms (in a subfolder).
  subfolder = folder + "qualityChecks/";
  ibooker.setCurrentFolder(subfolder);

  if (fIncludePixelQualCheckHistos) {
    // Create histograms for this run if not already present in our list.
    edm::LogInfo("Status") << "Creating histograms for run #" << run.id().run();

    // Pixel cluster positions in the barrel - (z, phi).
    for (size_t i = 0; i <= kNumLayers; ++i) {
      std::stringstream key;
      key << "clusPosBarrel" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster position - barrel layer " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 100, -30., 30., 64, -Geom::pi(), Geom::pi());
    }

    // Pixel cluster positions in the endcaps (x, y).
    for (size_t i = 0; i <= kNumDisks; ++i) {
      std::stringstream key;
      key << "clusPosEndCap" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster position - endcap disk " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 100, -20., 20., 100, -20., 20.);
    }

    // Pixel cluster charge in the barrel, per bx.
    for (size_t i = 0; i <= kNumLayers; ++i) {
      std::stringstream key;
      key << "clusChargeBarrel" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster charge - barrel layer " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 3564, .5, 3564.5, 100, 0., 100.);
    }

    // Pixel cluster charge in the endcaps, per bx.
    for (size_t i = 0; i <= kNumDisks; ++i) {
      std::stringstream key;
      key << "clusChargeEndCap" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster charge - endcap disk " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 3564, .5, 3564.5, 100, 0., 100.);
    }

    // Pixel cluster size in the barrel, per bx.
    for (size_t i = 0; i <= kNumLayers; ++i) {
      std::stringstream key;
      key << "clusSizeBarrel" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster size - barrel layer " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 3564, .5, 3564.5, 100, 0., 100.);
    }

    // Pixel cluster size in the endcaps, per bx.
    for (size_t i = 0; i <= kNumDisks; ++i) {
      std::stringstream key;
      key << "clusSizeEndCap" << i;
      std::stringstream name;
      name << key.str() << "_" << run.run();
      std::stringstream title;
      title << "Pixel cluster size - endcap disk " << i;
      fHistContainerThisRun[key.str()] =
          ibooker.book2D(name.str().c_str(), title.str().c_str(), 3564, .5, 3564.5, 100, 0., 100.);
    }
  }
}

// ------------ Method called when starting to process a luminosity block.
// ------------
void PixelLumiDQM::beginLuminosityBlock(edm::LuminosityBlock const &lumiBlock, edm::EventSetup const &) {
  // Only reset and fill every fResetIntervalInLumiSections (default is 1 LS)
  // Return unless the PREVIOUS LS was at the right modulo value
  // (e.g. is resetinterval = 5 the rest will only be executed at LS=6
  // NB: reset is done here so the histograms by LS are sent before resetting.
  // NB: not being used for now since default is 1 LS. There is a bug here.

  unsigned int ls = lumiBlock.luminosityBlockAuxiliary().luminosityBlock();

  if ((ls - 1) % fResetIntervalInLumiSections == 0) {
    fHistBunchCrossingsLastLumi->Reset();
    fHistClusterCountByBxLastLumi->Reset();
    fHistRecordedByBxLastLumi->Reset();
  }
}

// ------------ Method called when ending the processing of a luminosity block.
// ------------
void PixelLumiDQM::endLuminosityBlock(edm::LuminosityBlock const &lumiBlock, edm::EventSetup const &es) {
  unsigned int ls = lumiBlock.luminosityBlockAuxiliary().luminosityBlock();

  // Only fill every fResetIntervalInLumiSections (default is 1 LS)
  if (ls % fResetIntervalInLumiSections != 0)
    return;

  printf("Lumi Block = %d\n", ls);

  if ((ls - 1) % fResetIntervalInLumiSections == 0) {
  }

  unsigned int nBClus[3] = {0, 0, 0};
  unsigned int nFPClus[2] = {0, 0};
  unsigned int nFMClus[2] = {0, 0};

  double total_recorded = 0.;
  double total_recorded_unc_square = 0.;

  // Obtain bunch-by-bunch cluster counts and compute totals for lumi
  // calculation.
  double totalcounts = 0.0;
  double etotalcounts = 0.0;
  double totalevents = 0.0;
  double lumi_factor_per_bx = 0.0;
  if (useInnerBarrelLayer)
    lumi_factor_per_bx = FREQ_ORBIT * SECONDS_PER_LS * fResetIntervalInLumiSections / XSEC_PIXEL_CLUSTER;
  else
    lumi_factor_per_bx = FREQ_ORBIT * SECONDS_PER_LS * fResetIntervalInLumiSections / rXSEC_PIXEL_CLUSTER;

  for (std::map<int, PixelClusterCount>::iterator it = fNumPixelClusters.begin(); it != fNumPixelClusters.end(); it++) {
    // Sum all clusters for this BX.
    unsigned int total = (*it).second.numB.at(1) + (*it).second.numB.at(2) + (*it).second.numFP.at(0) +
                         (*it).second.numFP.at(1) + (*it).second.numFM.at(0) + (*it).second.numFM.at(1);
    if (useInnerBarrelLayer)
      total += (*it).second.numB.at(0);
    totalcounts += total;
    double etotal = (*it).second.dnumB.at(1) + (*it).second.dnumB.at(2) + (*it).second.dnumFP.at(0) +
                    (*it).second.dnumFP.at(1) + (*it).second.dnumFM.at(0) + (*it).second.dnumFM.at(1);
    if (useInnerBarrelLayer)
      etotal = (*it).second.dnumB.at(0);
    etotalcounts += etotal;
    etotal = sqrt(etotal);

    fHistClusterCountByBxLastLumi->setBinContent((*it).first, total);
    fHistClusterCountByBxLastLumi->setBinError((*it).first, etotal);
    fHistClusterCountByBxCumulative->setBinContent((*it).first,
                                                   fHistClusterCountByBxCumulative->getBinContent((*it).first) + total);

    unsigned int events_per_bx = fHistBunchCrossingsLastLumi->getBinContent((*it).first);
    totalevents += events_per_bx;
    double average_cluster_count = events_per_bx != 0 ? double(total) / events_per_bx : 0.;
    double average_cluster_count_unc = events_per_bx != 0 ? etotal / events_per_bx : 0.;
    double pixel_bx_lumi_per_ls = lumi_factor_per_bx * average_cluster_count / CM2_TO_NANOBARN;
    double pixel_bx_lumi_per_ls_unc = 0.0;
    if (useInnerBarrelLayer)
      pixel_bx_lumi_per_ls_unc = sqrt(lumi_factor_per_bx * lumi_factor_per_bx *
                                      (average_cluster_count_unc * average_cluster_count_unc +
                                       (average_cluster_count * XSEC_PIXEL_CLUSTER_UNC / XSEC_PIXEL_CLUSTER) *
                                           (average_cluster_count * XSEC_PIXEL_CLUSTER / XSEC_PIXEL_CLUSTER))) /
                                 CM2_TO_NANOBARN;
    else
      pixel_bx_lumi_per_ls_unc = sqrt(lumi_factor_per_bx * lumi_factor_per_bx *
                                      (average_cluster_count_unc * average_cluster_count_unc +
                                       (average_cluster_count * rXSEC_PIXEL_CLUSTER_UNC / rXSEC_PIXEL_CLUSTER) *
                                           (average_cluster_count * rXSEC_PIXEL_CLUSTER / rXSEC_PIXEL_CLUSTER))) /
                                 CM2_TO_NANOBARN;

    fHistRecordedByBxLastLumi->setBinContent((*it).first, pixel_bx_lumi_per_ls);
    fHistRecordedByBxLastLumi->setBinError((*it).first, pixel_bx_lumi_per_ls_unc);

    fHistRecordedByBxCumulative->setBinContent(
        (*it).first, fHistRecordedByBxCumulative->getBinContent((*it).first) + pixel_bx_lumi_per_ls);

    /*
      if(fHistRecordedByBxLastLumi->getBinContent((*it).first)!=0.)
      fHistRecordedByBxLastLumi->getBinContent((*it).first));
      if(fHistRecordedByBxCumulative->getBinContent((*it).first)!=0.)
      fHistRecordedByBxCumulative->getBinContent((*it).first));
    */

    nBClus[0] += (*it).second.numB.at(0);
    nBClus[1] += (*it).second.numB.at(1);
    nBClus[2] += (*it).second.numB.at(2);
    nFPClus[0] += (*it).second.numFP.at(0);
    nFPClus[1] += (*it).second.numFP.at(1);
    nFMClus[0] += (*it).second.numFM.at(0);
    nFMClus[1] += (*it).second.numFM.at(1);

    // Reset counters
    (*it).second.Reset();

    // edm::LogWarning("pixelLumi") << "bx="<< (*it).first << " clusters=" <<
    // (*it).second.numB.at(0));
  }

  if ((filledAndUnmaskedBunches = calculateBunchMask(fHistClusterCountByBxCumulative, bunchTriggerMask)) != 0) {
    for (unsigned int i = 0; i <= lastBunchCrossing; i++) {
      if (bunchTriggerMask[i]) {
        double err = fHistRecordedByBxLastLumi->getBinError(i);
        total_recorded += fHistRecordedByBxLastLumi->getBinContent(i);
        total_recorded_unc_square += err * err;
      }
    }

    // Replace the total obtained by summing over BXs with the average per BX
    // from the total cluster count and rescale

    if (totalevents > 10) {
      total_recorded = lumi_factor_per_bx * totalcounts / totalevents / CM2_TO_NANOBARN;
    } else
      total_recorded = 0.0;

    edm::LogWarning("pixelLumi") << " Total recorded " << total_recorded;
    fHistTotalRecordedLumiByLS->setBinContent(ls, total_recorded);
    fHistTotalRecordedLumiByLS->setBinError(ls, sqrt(total_recorded_unc_square));
  }
  // fill cluster counts by detector regions for sanity checks
  unsigned int all_detectors_counts = 0;
  for (unsigned int i = 0; i < 3; i++) {
    all_detectors_counts += nBClus[i];
  }
  for (unsigned int i = 0; i < 2; i++) {
    all_detectors_counts += nFPClus[i];
  }
  for (unsigned int i = 0; i < 2; i++) {
    all_detectors_counts += nFMClus[i];
  }

  fHistClusByLS->setBinContent(ls, all_detectors_counts);

  for (unsigned int i = 0; i < 3; i++) {
    fHistnBClusVsLS[i]->setBinContent(ls, float(nBClus[i]) / float(all_detectors_counts));
  }
  for (unsigned int i = 0; i < 2; i++) {
    fHistnFPClusVsLS[i]->setBinContent(ls, float(nFPClus[i]) / float(all_detectors_counts));
  }
  for (unsigned int i = 0; i < 2; i++) {
    fHistnFMClusVsLS[i]->setBinContent(ls, float(nFMClus[i]) / float(all_detectors_counts));
  }

  logFile_.open(fLogFileName_.c_str(), std::ios_base::trunc);

  timeval tv;
  gettimeofday(&tv, nullptr);
  tm *ts = gmtime(&tv.tv_sec);
  char datestring[256];
  strftime(datestring, sizeof(datestring), "%Y.%m.%d %T GMT %s", ts);
  logFile_ << "RunNumber " << fRunNo << std::endl;
  logFile_ << "EndTimeOfFit " << datestring << std::endl;
  logFile_ << "LumiRange " << ls << "-" << ls << std::endl;
  logFile_ << "Fill " << -99 << std::endl;
  logFile_ << "ActiveBunchCrossings " << filledAndUnmaskedBunches << std::endl;
  logFile_ << "PixelLumi " << fHistTotalRecordedLumiByLS->getBinContent(ls) * 0.98 << std::endl;
  logFile_ << "HFLumi " << -99 << std::endl;
  logFile_ << "Ratio " << -99 << std::endl;
  logFile_.close();
}

unsigned int PixelLumiDQM::calculateBunchMask(MonitorElement *e, std::vector<bool> &mask) {
  unsigned int nbins = e->getNbinsX();
  std::vector<float> ar(nbins, 0.);
  for (unsigned int i = 1; i <= nbins; i++) {
    ar[i] = e->getBinContent(i);
  }
  return calculateBunchMask(ar, nbins, mask);
}
unsigned int PixelLumiDQM::calculateBunchMask(std::vector<float> &e, unsigned int nbins, std::vector<bool> &mask) {
  // Take the cumulative cluster count histogram and find max and average of
  // non-empty bins.
  unsigned int active_count = 0;
  double maxc = 0.0;
  double ave = 0.0;  // Average of non-empty bins
  unsigned int non_empty_bins = 0;

  for (unsigned int i = 1; i <= nbins; i++) {
    double bin = e[i];
    if (bin != 0.0) {
      if (maxc < bin)
        maxc = bin;
      ave += bin;
      non_empty_bins++;
    }
  }

  ave /= non_empty_bins;
  edm::LogWarning("pixelLumi") << "Bunch mask finder - non empty bins " << non_empty_bins
                               << " average of non empty bins " << ave << " max content of one bin " << maxc;
  double mean = 0.;
  double sigma = 0.;
  if (non_empty_bins < 50) {
    mean = maxc;
    sigma = (maxc) / 20;
  } else {
    TH1F dist("dist", "dist", 500, 0., maxc + (maxc / 500.) * 20.);
    for (unsigned int i = 1; i <= nbins; i++) {
      double bin = e[i];
      dist.Fill(bin);
    }
    TF1 fit("plgaus", "gaus");
    dist.Fit(&fit, "", "", fmax(0., ave - (maxc - ave) / 5.), maxc);
    mean = fit.GetParameter("Mean");
    sigma = fit.GetParameter("Sigma");
  }
  edm::LogWarning("pixelLumi") << "Bunch mask will use mean" << mean << " sigma " << sigma;
  // Active BX defined as those which have nclus within fixed standard
  // deviations of peak.
  for (unsigned int i = 1; i <= nbins; i++) {
    double bin = e[i];
    if (bin > 0. && std::abs(bin - mean) < 5. * (sigma)) {
      mask[i] = true;
      active_count++;
    }
  }
  edm::LogWarning("pixelLumi") << "Bunch mask finds " << active_count << " active bunch crossings ";
  //   edm::LogWarning("pixelLumi") << "this is the full bx mask " ;
  //   for(unsigned int i = 1; i<= nbins; i++)
  //     edm::LogWarning("pixelLumi") << ((mask[i]) ? 1:0);
  return active_count;
}
// Define this as a CMSSW plug-in.
DEFINE_FWK_MODULE(PixelLumiDQM);
