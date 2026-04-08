#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>

#include "RecoTracker/LSTGeometry/interface/IO.h"

namespace lstgeometry {

  void writeSensorCentroids(Sensors const& sensors, std::string const& base_filename, bool binary) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto const& [detid, sensor] : sensors) {
        float x = sensor.centerX;
        float y = sensor.centerY;
        float z = sensor.centerZ;
        unsigned int moduleType = static_cast<unsigned int>(sensor.moduleType);
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        file.write(reinterpret_cast<const char*>(&x), sizeof(x));
        file.write(reinterpret_cast<const char*>(&y), sizeof(y));
        file.write(reinterpret_cast<const char*>(&z), sizeof(z));
        file.write(reinterpret_cast<const char*>(&moduleType), sizeof(moduleType));
      }
    } else {
      for (auto const& [detid, sensor] : sensors) {
        file << detid << "," << sensor.centerX << "," << sensor.centerY << "," << sensor.centerZ << ","
             << static_cast<unsigned int>(sensor.moduleType) << std::endl;
      }
    }
  }

  void writeSlopes(Slopes const& slopes, Sensors const& sensors, std::string const& base_filename, bool binary) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto const& [detid, slope] : slopes) {
        float drdz_slope = slope.drdz;
        float dxdy_slope = slope.dxdy;
        float phi = sensors.at(detid).centerPhi;
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        if (drdz_slope != kDefaultSlope) {
          file.write(reinterpret_cast<const char*>(&drdz_slope), sizeof(drdz_slope));
          file.write(reinterpret_cast<const char*>(&dxdy_slope), sizeof(dxdy_slope));
        } else {
          file.write(reinterpret_cast<const char*>(&dxdy_slope), sizeof(dxdy_slope));
          file.write(reinterpret_cast<const char*>(&phi), sizeof(phi));
        }
      }
    } else {
      for (auto const& [detid, slope] : slopes) {
        float drdz_slope = slope.drdz;
        float dxdy_slope = slope.dxdy;
        float phi = sensors.at(detid).centerPhi;
        file << detid << ",";
        if (drdz_slope != kDefaultSlope) {
          file << drdz_slope << "," << dxdy_slope << std::endl;
        } else {
          file << dxdy_slope << "," << phi << std::endl;
        }
      }
    }
  }

  void writeModuleConnections(ModuleMap const& connections, std::string const& base_filename, bool binary) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto const& [detid, list] : connections) {
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        unsigned int length = list.size();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        for (unsigned int i : list) {
          file.write(reinterpret_cast<const char*>(&i), sizeof(i));
        }
      }
    } else {
      for (auto const& [detid, list] : connections) {
        file << detid << "," << list.size();
        for (unsigned int i : list) {
          file << "," << i;
        }
        file << std::endl;
      }
    }
  }

  void writePixelMaps(PixelMap const& maps, std::string const& base_filename, bool binary) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    for (auto const& [layersubdetcharge, vec_of_vecs] : maps) {
      auto const& [layer, subdet, charge] = layersubdetcharge;

      std::string charge_str = charge > 0 ? "_pos" : (charge < 0 ? "_neg" : "");
      std::string filename =
          std::format("{}{}_layer{}_subdet{}.{}", base_filename, charge_str, layer, subdet, binary ? "bin" : "txt");

      std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

      for (unsigned int isuperbin = 0; isuperbin < vec_of_vecs.size(); isuperbin++) {
        auto const& vec = vec_of_vecs.at(isuperbin);
        if (vec.empty())
          continue;

        unsigned int length = vec.size();
        if (binary) {
          file.write(reinterpret_cast<const char*>(&isuperbin), sizeof(isuperbin));
          file.write(reinterpret_cast<const char*>(&length), sizeof(length));
          for (unsigned int i : vec) {
            file.write(reinterpret_cast<const char*>(&i), sizeof(i));
          }
        } else {
          file << isuperbin << "," << length;
          for (unsigned int i : vec) {
            file << "," << i;
          }
          file << std::endl;
        }
      }
    }
  }

}  // namespace lstgeometry
