#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <set>
#include <filesystem>

using namespace trklet;
using namespace std;

MemoryBase::MemoryBase(string name, Settings const& settings) : name_(name), settings_(settings) {
  iSector_ = 0;
  bx_ = 0;
  event_ = 0;
}

void MemoryBase::initLayerDisk(unsigned int pos, int& layer, int& disk) {
  string subname = name_.substr(pos, 2);
  layer = 0;
  disk = 0;

  if (subname.substr(0, 1) == "L")
    layer = stoi(subname.substr(1, 1));
  else if (subname.substr(0, 1) == "D")
    disk = stoi(subname.substr(1, 1));
  else
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " name = " << name_ << " subname = " << subname
                                      << " " << layer << " " << disk;
}

unsigned int MemoryBase::initLayerDisk(unsigned int pos) {
  int layer, disk;
  initLayerDisk(pos, layer, disk);

  if (disk > 0)
    return N_DISK + disk;
  return layer - 1;
}

void MemoryBase::initSpecialSeeding(unsigned int pos, bool& overlap, bool& extra, bool& extended) {
  overlap = false;
  extra = false;
  extended = false;

  char subname = name_[pos];

  static const std::set<char> overlapset = {
      'X', 'Y', 'W', 'Q', 'R', 'S', 'T', 'Z', 'x', 'y', 'w', 'q', 'r', 's', 't', 'z'};
  overlap = overlapset.find(subname) != overlapset.end();

  static const std::set<char> extraset = {'I', 'J', 'K', 'L'};
  extra = extraset.find(subname) != extraset.end();

  static const std::set<char> extendedset = {
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x', 'y', 'z', 'w', 'q', 'r', 's', 't'};
  extended = extendedset.find(subname) != extendedset.end();
}

void MemoryBase::findAndReplaceAll(std::string& data, std::string toSearch, std::string replaceStr) {
  // Get the first occurrence
  size_t pos = data.find(toSearch);

  // Repeat till end is reached
  while (pos != std::string::npos) {
    // Replace this occurrence of Sub String
    data.replace(pos, toSearch.size(), replaceStr);
    // Get the next occurrence from the current position
    pos = data.find(toSearch, pos + replaceStr.size());
  }
}

void MemoryBase::openFile(bool first, std::string dirName, std::string filebase) {
  std::string fname = filebase + getName();

  findAndReplaceAll(fname, "PHIa", "PHIaa");
  findAndReplaceAll(fname, "PHIb", "PHIbb");
  findAndReplaceAll(fname, "PHIc", "PHIcc");
  findAndReplaceAll(fname, "PHId", "PHIdd");

  findAndReplaceAll(fname, "PHIx", "PHIxx");
  findAndReplaceAll(fname, "PHIy", "PHIyy");
  findAndReplaceAll(fname, "PHIz", "PHIzz");
  findAndReplaceAll(fname, "PHIw", "PHIww");

  fname += "_";
  if (iSector_ + 1 < 10)
    fname += "0";
  fname += std::to_string(iSector_ + 1);
  fname += ".dat";

  openfile(out_, first, dirName, dirName + fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}

size_t MemoryBase::find_nth(const string& haystack, size_t pos, const string& needle, size_t nth) {
  size_t found_pos = haystack.find(needle, pos);
  if (0 == nth || string::npos == found_pos)
    return found_pos;
  return find_nth(haystack, found_pos + 1, needle, nth - 1);
}
