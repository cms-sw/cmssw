#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <unordered_map>

using namespace trklet;
using namespace std;

ProcessBase::ProcessBase(string name, Settings const& settings, Globals* global)
    : name_(name), settings_(settings), globals_(global) {}

unsigned int ProcessBase::nbits(unsigned int power) {
  if (power == 2)
    return 1;
  if (power == 4)
    return 2;
  if (power == 8)
    return 3;
  if (power == 16)
    return 4;
  if (power == 32)
    return 5;

  throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << "nbits: power = " << power;
  return 0;
}

void ProcessBase::initLayerDisk(unsigned int pos, int& layer, int& disk) {
  string subname = name_.substr(pos, 2);
  layer = 0;
  disk = 0;
  if (subname.substr(0, 1) == "L")
    layer = stoi(subname.substr(1, 1));
  else if (subname.substr(0, 1) == "D")
    disk = stoi(subname.substr(1, 1));
  else
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " " << name_ << " subname = " << subname << " "
                                      << layer << " " << disk;
}

void ProcessBase::initLayerDisk(unsigned int pos, int& layer, int& disk, int& layerdisk) {
  initLayerDisk(pos, layer, disk);

  layerdisk = layer - 1;
  if (disk > 0)
    layerdisk = N_LAYER + disk - 1;
}

unsigned int ProcessBase::initLayerDisk(unsigned int pos) {
  int layer, disk;
  initLayerDisk(pos, layer, disk);

  if (disk > 0)
    return N_LAYER + disk - 1;
  return layer - 1;
}

void ProcessBase::initLayerDisksandISeed(unsigned int& layerdisk1, unsigned int& layerdisk2, unsigned int& iSeed) {
  layerdisk1 = 99;
  layerdisk2 = 99;

  if (name_.substr(0, 3) == "TE_") {
    if (name_[3] == 'L') {
      layerdisk1 = name_[4] - '1';
    } else if (name_[3] == 'D') {
      layerdisk1 = N_LAYER + name_[4] - '1';
    }
    if (name_[11] == 'L') {
      layerdisk2 = name_[12] - '1';
    } else if (name_[11] == 'D') {
      layerdisk2 = N_LAYER + name_[12] - '1';
    } else if (name_[12] == 'L') {
      layerdisk2 = name_[13] - '1';
    } else if (name_[12] == 'D') {
      layerdisk2 = N_LAYER + name_[13] - '1';
    }
  }

  if ((name_.substr(0, 3) == "TC_") || (name_.substr(0, 3) == "TP_")) {
    if (name_[3] == 'L') {
      layerdisk1 = name_[4] - '1';
    } else if (name_[3] == 'D') {
      layerdisk1 = N_LAYER + name_[4] - '1';
    }
    if (name_[5] == 'L') {
      layerdisk2 = name_[6] - '1';
    } else if (name_[5] == 'D') {
      layerdisk2 = N_LAYER + name_[6] - '1';
    }
  }

  if (layerdisk1 == LayerDisk::L1 && layerdisk2 == LayerDisk::L2)
    iSeed = Seed::L1L2;
  else if (layerdisk1 == LayerDisk::L2 && layerdisk2 == LayerDisk::L3)
    iSeed = Seed::L2L3;
  else if (layerdisk1 == LayerDisk::L3 && layerdisk2 == LayerDisk::L4)
    iSeed = Seed::L3L4;
  else if (layerdisk1 == LayerDisk::L5 && layerdisk2 == LayerDisk::L6)
    iSeed = Seed::L5L6;
  else if (layerdisk1 == LayerDisk::D1 && layerdisk2 == LayerDisk::D2)
    iSeed = Seed::D1D2;
  else if (layerdisk1 == LayerDisk::D3 && layerdisk2 == LayerDisk::D4)
    iSeed = Seed::D3D4;
  else if (layerdisk1 == LayerDisk::L1 && layerdisk2 == LayerDisk::D1)
    iSeed = Seed::L1D1;
  else if (layerdisk1 == LayerDisk::L2 && layerdisk2 == LayerDisk::D1)
    iSeed = Seed::L2D1;
  else {
    throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " layerdisk1 " << layerdisk1 << " layerdisk2 "
                                       << layerdisk2;
  }
}

unsigned int ProcessBase::getISeed(const std::string& name) {
  std::size_t pos = name.find('_');
  std::string name1 = name.substr(pos + 1);
  pos = name1.find('_');
  std::string name2 = name1.substr(0, pos);

  unordered_map<string, unsigned int> seedmap = {
      {"L1L2", 0},   {"L2L3", 1},   {"L3L4", 2},   {"L5L6", 3},   {"D1D2", 4},    {"D3D4", 5},   {"L1D1", 6},
      {"L2D1", 7},   {"L1L2XX", 0}, {"L2L3XX", 1}, {"L3L4XX", 2}, {"L5L6XX", 3},  {"D1D2XX", 4}, {"D3D4XX", 5},
      {"L1D1XX", 6}, {"L2D1XX", 7}, {"L3L4L2", 8}, {"L5L6L4", 9}, {"L2L3D1", 10}, {"D1D2L2", 11}};
  auto found = seedmap.find(name2);
  if (found != seedmap.end())
    return found->second;

  throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " " << getName() << " name name1 name2 " << name
                                     << " - " << name1 << " - " << name2;
  return 0;
}
