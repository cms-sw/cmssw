#include "EventFilter/Phase2TrackerRawToDigi/interface/CRACKMapping.h"

namespace crack {

// ----- Constructors -----

CRACKMapping::CRACKMapping(const edm::ParameterSet& pset) {
    loadFromPSet(pset);
}

CRACKMapping::CRACKMapping(const std::string& filename) {
    loadFromFile(filename);
}

// ----- Load methods -----

void CRACKMapping::loadFromVPSet(const std::vector<edm::ParameterSet>& vpset) {
    for (const auto& chanPSet : vpset) {
        int dtc = chanPSet.getParameter<int>("dtc");
        int core = chanPSet.getParameter<int>("coreID");
        int offset = chanPSet.getParameter<int>("offset");
        int gbtID = chanPSet.getParameter<int>("gbtID");
        
        addMapping(dtc, core, offset, gbtID);
    }
}

void CRACKMapping::loadFromPSet(const edm::ParameterSet& pset) {
  auto channelPSets = pset.getParameter<std::vector<edm::ParameterSet>>("crackMapping");

  for (const auto& chanPSet : channelPSets) {
    int dtc = chanPSet.getParameter<int>("dtc");
    int core = chanPSet.getParameter<int>("coreID");
    int offset = chanPSet.getParameter<int>("offset");
    int gbtID = chanPSet.getParameter<int>("gbtID");

    addMapping(dtc, core, offset, gbtID);
  }
}

void CRACKMapping::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw cms::Exception("FileNotFound") 
            << "CRACK mapping file not found: " << filename;
    }
    
    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        int dtc, core, offset, gbtID;
        char comma;
        
        ss >> dtc >> comma >> core >> comma >> offset >> comma >> gbtID;
        
        if (ss.fail()) {
            throw cms::Exception("ParseError") 
                << "Failed to parse line " << lineNum << ": " << line 
                << " (expected format: dtc,core,offset,gbtID)";
        }
        
        addMapping(dtc, core, offset, gbtID);
    }
    
    file.close();
}

void CRACKMapping::addMapping(int dtc, int core, int offset, int gbtID) {
    mapping_[dtc][core][offset] = gbtID;
}

// ----- Lookup methods -----

int CRACKMapping::getGbtID(int dtc, int core, int offset) const {
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) {
        throw cms::Exception("MappingNotFound") 
            << "DTC " << dtc << " not found in CRACK mapping";
    }
    
    auto coreIt = dtcIt->second.find(core);
    if (coreIt == dtcIt->second.end()) {
        throw cms::Exception("MappingNotFound") 
            << "Core " << core << " not found for DTC " << dtc;
    }
    
    auto offsetIt = coreIt->second.find(offset);
    if (offsetIt == coreIt->second.end()) {
        throw cms::Exception("MappingNotFound") 
            << "Offset " << offset << " not found for DTC " << dtc 
            << ", Core " << core;
    }
    
    return offsetIt->second;
}

bool CRACKMapping::hasMapping(int dtc, int core, int offset) const {
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) return false;
    
    auto coreIt = dtcIt->second.find(core);
    if (coreIt == dtcIt->second.end()) return false;
    
    return coreIt->second.find(offset) != coreIt->second.end();
}

std::vector<int> CRACKMapping::getOffsets(int dtc, int core) const {
    std::vector<int> offsets;
    
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) return offsets;
    
    auto coreIt = dtcIt->second.find(core);
    if (coreIt == dtcIt->second.end()) return offsets;
    
    for (const auto& pair : coreIt->second) {
        offsets.push_back(pair.first);
    }
    
    return offsets;
}

std::vector<int> CRACKMapping::getCores(int dtc) const {
    std::vector<int> cores;
    
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) return cores;
    
    for (const auto& pair : dtcIt->second) {
        cores.push_back(pair.first);
    }
    
    return cores;
}

std::vector<int> CRACKMapping::getDTCs() const {
    std::vector<int> dtcs;
    for (const auto& pair : mapping_) {
        dtcs.push_back(pair.first);
    }
    return dtcs;
}

bool CRACKMapping::hasDTC(int dtc) const {
    return mapping_.find(dtc) != mapping_.end();
}

bool CRACKMapping::hasCore(int dtc, int core) const {
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) return false;
    return dtcIt->second.find(core) != dtcIt->second.end();
}

// ----- Utility methods -----

void CRACKMapping::clear() {
    mapping_.clear();
}

size_t CRACKMapping::size() const {
    size_t count = 0;
    for (const auto& dtcPair : mapping_) {
        for (const auto& corePair : dtcPair.second) {
            count += corePair.second.size();
        }
    }
    return count;
}

void CRACKMapping::print() const {
    std::cout << "=== CRACK Mapping ===" << std::endl;
    std::cout << "Total entries: " << size() << std::endl;
    std::cout << "Number of DTCs: " << mapping_.size() << std::endl;
    std::cout << std::endl;
    
    for (const auto& dtcPair : mapping_) {
        printDTC(dtcPair.first);
    }
}

void CRACKMapping::printDTC(int dtc) const {
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) {
        std::cout << "DTC " << dtc << ": No mappings found" << std::endl;
        return;
    }
    
    std::cout << "DTC " << dtc << ":" << std::endl;
    std::cout << "  Cores: " << dtcIt->second.size() << std::endl;
    
    for (const auto& corePair : dtcIt->second) {
        printCore(dtc, corePair.first);
    }
    std::cout << std::endl;
}

void CRACKMapping::printCore(int dtc, int core) const {
    auto dtcIt = mapping_.find(dtc);
    if (dtcIt == mapping_.end()) {
        std::cout << "  DTC " << dtc << " not found" << std::endl;
        return;
    }
    
    auto coreIt = dtcIt->second.find(core);
    if (coreIt == dtcIt->second.end()) {
        std::cout << "  Core " << core << " not found for DTC " << dtc << std::endl;
        return;
    }
    
    std::cout << "  Core " << core << ":" << std::endl;
    std::cout << "    Offset | GBT_ID" << std::endl;
    std::cout << "    -------+--------" << std::endl;
    
    for (const auto& offsetPair : coreIt->second) {
        std::cout << "    " << std::setw(7) << offsetPair.first 
                  << " | " << std::setw(6) << offsetPair.second 
                  << std::endl;
    }
}

} // namespace crack