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
    const Phase2TrackerCabling& fedCabling() const;
    const Phase2TrackerCabling& detCabling() const;
    const Phase2TrackerCabling& gbtCabling() const;

    // get the list of modules
    const std::vector<Phase2TrackerModule>& connections() const { return connections_; }

    // find a connection for a given fed channel
    const Phase2TrackerModule& findFedCh(std::pair<unsigned int, unsigned int> fedch) const;

    // find a connection for a given detid
    const Phase2TrackerModule& findDetid(uint32_t detid) const;

    // find a connection for a given gbtid
    const Phase2TrackerModule& findGbtid(uint32_t gbtid) const;

    // return all the modules connected to a given cooling line
    Phase2TrackerCabling filterByCoolingLine(uint32_t coolingLine) const;

    // return all the modules connected to a given HV group
    Phase2TrackerCabling filterByPowerGroup(uint32_t powerGroup) const;

    // print a summary of the content
    std::string summaryDescription() const;

    // print the details of the content
    std::string description(bool compact=false) const;

  protected:
    // check the proper mode and sort if needed
    void checkMode(const char* funcname, int mode) const;

  private:
    // data members: declared as mutable to allow easy sorting. 
    // for the vector, it is a bit an overkill, as we just want to sort it.
    // since it is private with no way to add/remove elements, it has no impact.

    // the connections
    mutable std::vector<Phase2TrackerModule> connections_;
    // the mode: sorted state
    mutable int mode_;

};

#endif
