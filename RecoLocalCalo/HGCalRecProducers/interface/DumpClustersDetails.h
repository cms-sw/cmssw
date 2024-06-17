#ifndef RecoLocalCalo_HGCalRecProducers_DumpClustersDetails_h
#define RecoLocalCalo_HGCalRecProducers_DumpClustersDetails_h

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <unistd.h>  // For getpid
#include <fmt/format.h>
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

namespace hgcalUtils {

  static bool sortByDetId(const std::pair<int, float>& pair1, const std::pair<int, float>& pair2) {
    return pair1.first < pair2.first;
  }

  class DumpClusters {
  public:
    DumpClusters() = default;

    template <typename T>
    void dumpInfos(const T& clusters,
                   const std::string& moduleLabel,
                   edm::RunNumber_t run,
                   edm::LuminosityBlockNumber_t lumi,
                   edm::EventNumber_t event,
                   bool dumpCellsDetId = false) const {
      // Get the process ID
      pid_t pid = getpid();

      // Create the filename using the PID
      std::ostringstream filename;
      filename << "CLUSTERS_" << pid << "_" << moduleLabel << "_" << run << "_" << lumi << "_" << event << ".txt";
      // Open the file
      std::ofstream outfile(filename.str());
      int count = 0;
      for (auto const& i : clusters) {
        outfile << fmt::format(
            "Seed: {}, Idx: {}, energy: {:.{}f}, x: {:.{}f}, y: {:.{}f}, z: {:.{}f}, eta: {:.{}f}, phi: {:.{}f}",
            i.seed().rawId(),
            count++,
            (float)i.energy(),
            std::numeric_limits<float>::max_digits10,
            i.x(),
            std::numeric_limits<float>::max_digits10,
            i.y(),
            std::numeric_limits<float>::max_digits10,
            i.z(),
            std::numeric_limits<float>::max_digits10,
            i.eta(),
            std::numeric_limits<float>::max_digits10,
            i.phi(),
            std::numeric_limits<float>::max_digits10);
        if (dumpCellsDetId) {
          auto sorted = i.hitsAndFractions();  // copy...
          std::stable_sort(std::begin(sorted), std::end(sorted), sortByDetId);
          for (auto const& c : sorted) {
            outfile << fmt::format(" ({}, {:.{}f})", c.first, c.second, std::numeric_limits<float>::max_digits10);
          }  // loop on hits and fractions
        } else {
          outfile << fmt::format(" ({} cells)", i.hitsAndFractions().size());
        }
        outfile << std::endl;
      }  // loop on clusters
      outfile.close();
    }
  };

  class DumpClustersSoA {
  public:
    DumpClustersSoA() = default;

    template <typename T>
    void dumpInfos(const T& clustersSoA,
                   const std::string& moduleLabel,
                   edm::RunNumber_t run,
                   edm::LuminosityBlockNumber_t lumi,
                   edm::EventNumber_t event) const {
      // Get the process ID
      pid_t pid = getpid();

      // Create the filename using the PID
      std::ostringstream filename;
      filename << "CLUSTERS_UTILS_SOA_" << pid << "_" << moduleLabel << "_" << run << "_" << lumi << "_" << event
               << ".txt";
      // Open the file
      std::ofstream outfile(filename.str());
      for (int i = 0; i < clustersSoA->metadata().size(); ++i) {
        auto clusterSoAV = clustersSoA.view()[i];
        outfile << fmt::format("Idx: {}, delta: {:.{}f}, rho: {:.{}f}, nearest: {}, clsIdx: {}, isSeed: {}",
                               i,
                               clusterSoAV.delta(),
                               std::numeric_limits<float>::max_digits10,
                               clusterSoAV.rho(),
                               std::numeric_limits<float>::max_digits10,
                               clusterSoAV.nearestHigher(),
                               clusterSoAV.clusterIndex(),
                               clusterSoAV.isSeed())
                << std::endl;
      }
      outfile.close();
    }
  };

  class DumpCellsSoA {
  public:
    DumpCellsSoA() = default;

    template <typename T>
    void dumpInfos(const T& cells,
                   const std::string& moduleLabel,
                   edm::RunNumber_t run,
                   edm::LuminosityBlockNumber_t lumi,
                   edm::EventNumber_t event) const {
      // Get the process ID
      pid_t pid = getpid();

      // Create the filename using the PID
      std::ostringstream filename;
      filename << "RECHITS_SOA_" << pid << "_" << moduleLabel << "_" << run << "_" << lumi << "_" << event << ".txt";
      // Open the file
      std::ofstream outfile(filename.str());
      for (int i = 0; i < cells->metadata().size(); ++i) {
        auto cellSoAV = cells.view()[i];
        outfile
            << fmt::format(
                   "Idx Cell: {}, x: {:.{}f}, y: {:.{}f}, layer: {}, weight: {:.{}f}, sigmaNoise: {:.{}f}, detid: {}",
                   i,
                   (cellSoAV.dim1()),
                   std::numeric_limits<float>::max_digits10,
                   (cellSoAV.dim2()),
                   std::numeric_limits<float>::max_digits10,
                   cellSoAV.layer(),
                   (cellSoAV.weight()),
                   std::numeric_limits<float>::max_digits10,
                   (cellSoAV.sigmaNoise()),
                   std::numeric_limits<float>::max_digits10,
                   cellSoAV.detid())
            << std::endl;
      }
      outfile.close();
    }
  };

  class DumpLegacySoA {
  public:
    DumpLegacySoA() = default;

    template <typename T>
    void dumpInfos(const T& cells, const std::string& moduleType) const {
      // Get the process ID
      pid_t pid = getpid();

      // Create the filename using the PID
      std::ostringstream filename;
      // Seed the random number generator
      srand(time(0));
      // Generate a random number between 100 and 999
      int random_number = 100 + rand() % 900;

      filename << "RECHITS_LEGACY_" << pid << "_" << moduleType << "_" << random_number << ".txt";
      // Open the file
      std::ofstream outfile(filename.str());
      for (unsigned int l = 0; l < cells.size(); l++) {
        for (unsigned int i = 0; i < cells.at(l).dim1.size(); ++i) {
          outfile << fmt::format(
                         "Idx Cell: {}, x: {:.{}f}, y: {:.{}f}, layer: {}, weight: {:.{}f}, sigmaNoise: {:.{}f}, "
                         "delta: {:.{}f}, rho: {:.{}f}, nearestHigher: {}, clsIdx: {}, isSeed: {}, detid: {}",
                         i,
                         (cells.at(l).dim1.at(i)),
                         std::numeric_limits<float>::max_digits10,
                         (cells.at(l).dim2.at(i)),
                         std::numeric_limits<float>::max_digits10,
                         l,
                         (cells.at(l).weight.at(i)),
                         std::numeric_limits<float>::max_digits10,
                         (cells.at(l).sigmaNoise.at(i)),
                         std::numeric_limits<float>::max_digits10,
                         cells.at(l).delta.at(i),
                         std::numeric_limits<float>::max_digits10,
                         cells.at(l).rho.at(i),
                         std::numeric_limits<float>::max_digits10,
                         cells.at(l).nearestHigher.at(i),
                         cells.at(l).clusterIndex.at(i),
                         cells.at(l).isSeed.at(i),
                         cells.at(l).detid.at(i))
                  << std::endl;
        }
      }
      outfile.close();
    }
  };
}  // namespace hgcalUtils

#endif
