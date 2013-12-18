#ifndef CondFormats_SiStripObjects_Phase2TrackerCabling_H
#define CondFormats_SiStripObjects_Phase2TrackerCabling_H

#include <CondFormats/SiStripObjects/interface/Phase2TrackerModule.h>
#include <vector>
#include <algorithm>

class Phase2TrackerCabling
{
  public:
    // Constructor taking FED channel connection objects as input.
    Phase2TrackerCabling( const std::vector<Phase2TrackerModule>& cons ):connections_(cons) { mode_ = 0; }

    // Copy ocnstructor
    Phase2TrackerCabling( const Phase2TrackerCabling& src ) { connections_ = src.connections_; mode_ = 0; }

    // Default constructor
    Phase2TrackerCabling() { mode_ = 0; }

    // Default destructor
    virtual ~Phase2TrackerCabling() {}

    // change the mode
    Phase2TrackerCabling& fedCabling();
    Phase2TrackerCabling& detCabling();
    Phase2TrackerCabling& gbtCabling();

    // get the list of modules
    const std::vector<Phase2TrackerModule>& connections() const { return connections_; }

    // find a connection for a given fed channel
    const Phase2TrackerModule& findFedCh(std::pair<unsigned int, unsigned int> fedch);

    // find a connection for a given detid
    const Phase2TrackerModule& findDetid(uint32_t detid);

    // find a connection for a given gbtid
    const Phase2TrackerModule& findGbtid(uint32_t gbtid);

    // return all the modules connected to a given cooling line
    Phase2TrackerCabling filterByCoolingLine(uint32_t coolingLine);

    // return all the modules connected to a given HV group
    Phase2TrackerCabling filterByPowerGroup(uint32_t powerGroup);

  protected:
    void checkMode(const char* funcname, int mode);

  private:
    std::vector<Phase2TrackerModule> connections_;
    int mode_;

};

#endif
