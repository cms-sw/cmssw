#ifndef EventFilter_Phase2TrackerRawToDigi_CRACKMapping_h
#define EventFilter_Phase2TrackerRawToDigi_CRACKMapping_h

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

/**
 * @brief: CRACKMapping Object, used to impose a hardware map between
 * FED offsets and real detectors in the CRACK Geometry. 
 */

namespace crack {

class CRACKMapping {
public:
    // Constructor
    CRACKMapping() = default;
    
    // Construct from ParameterSet (VPSet format)
    explicit CRACKMapping(const edm::ParameterSet& pset);
    
    // Construct from file
    explicit CRACKMapping(const std::string& filename);
    
    // Destructor
    ~CRACKMapping() = default;
    
    // ----- Load methods -----
    void loadFromVPSet(const std::vector<edm::ParameterSet>& vpset);  // <-- ADD THIS
    void loadFromPSet(const edm::ParameterSet& pset);
    void loadFromFile(const std::string& filename);
    void addMapping(int dtc, int core, int offset, int gbtID);
    
    // ----- Lookup methods -----
    
    // Primary lookup: [dtc][core][offset] -> gbtID
    int getGbtID(int dtc, int core, int offset) const;
    
    // Check if mapping exists
    bool hasMapping(int dtc, int core, int offset) const;
    
    // Get all offsets for a given DTC and Core
    std::vector<int> getOffsets(int dtc, int core) const;
    
    // Get all cores for a given DTC
    std::vector<int> getCores(int dtc) const;
    
    // Get all DTCs
    std::vector<int> getDTCs() const;
    
    // Check if DTC exists
    bool hasDTC(int dtc) const;
    
    // Check if DTC and Core exists
    bool hasCore(int dtc, int core) const;
    
    // ----- Utility methods -----
    void clear();
    size_t size() const;
    void print() const;
    void printDTC(int dtc) const;
    void printCore(int dtc, int core) const;
    
private:
    // Data structure: [dtc][core][offset] -> gbtID
    std::map<int, std::map<int, std::map<int, int>>> mapping_;
};

} // namespace crack

#endif // EventFilter_Phase2TrackerRawToDigi_CRACKMapping_h