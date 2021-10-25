#ifndef L1Trigger_TrackFindingTracklet_interface_MemoryBase_h
#define L1Trigger_TrackFindingTracklet_interface_MemoryBase_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <fstream>
#include <sstream>
#include <cassert>
#include <bitset>

namespace trklet {

  class MemoryBase {
  public:
    MemoryBase(std::string name, Settings const& settings);

    virtual ~MemoryBase() = default;

    std::string const& getName() const { return name_; }
    std::string getLastPartOfName() const { return name_.substr(name_.find_last_of('_') + 1); }

    virtual void clean() = 0;

    //method sets the layer and disk based on the name. pos is the position in the memory name where the layer or disk is specified
    void initLayerDisk(unsigned int pos, int& layer, int& disk);

    unsigned int initLayerDisk(unsigned int pos);

    // Based on memory name check if this memory is used for special seeding:
    // overlap is layer-disk seeding
    // extra is the L2L3 seeding
    // extended is the seeding for displaced tracks
    void initSpecialSeeding(unsigned int pos, bool& overlap, bool& extra, bool& extended);

    //Used for a hack below due to MAC OS case sensitiviy problem for files
    void findAndReplaceAll(std::string& data, std::string toSearch, std::string replaceStr);

    void openFile(bool first, std::string dirName, std::string filebase);

    static size_t find_nth(const std::string& haystack, size_t pos, const std::string& needle, size_t nth);

  protected:
    std::string name_;
    unsigned int iSector_;

    std::ofstream out_;
    int bx_;
    int event_;

    Settings const& settings_;
  };
};  // namespace trklet
#endif
