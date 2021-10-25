#ifndef L1Trigger_TrackFindingTracklet_interface_ProcessBase_h
#define L1Trigger_TrackFindingTracklet_interface_ProcessBase_h

#include <string>

namespace trklet {

  class MemoryBase;
  class Settings;
  class Globals;

  class ProcessBase {
  public:
    ProcessBase(std::string name, Settings const& settings, Globals* global);

    virtual ~ProcessBase() = default;

    // Add wire from pin "output" or "input" this proc module to memory instance "memory".
    virtual void addOutput(MemoryBase* memory, std::string output) = 0;
    virtual void addInput(MemoryBase* memory, std::string input) = 0;

    std::string const& getName() const { return name_; }

    unsigned int nbits(unsigned int power);

    //method sets the layer and disk based on the name. pos is the position in the memory name where the layer or disk is specified
    void initLayerDisk(unsigned int pos, int& layer, int& disk);
    void initLayerDisk(unsigned int pos, int& layer, int& disk, int& layerdisk);

    unsigned int initLayerDisk(unsigned int pos);

    //This function processes the name of a TE module to determine the layerdisks and iseed
    void initLayerDisksandISeed(unsigned int& layerdisk1, unsigned int& layerdisk2, unsigned int& iSeed);

    unsigned int getISeed(const std::string& name);

  protected:
    std::string name_;

    Settings const& settings_;
    Globals* globals_;
  };
};  // namespace trklet
#endif
