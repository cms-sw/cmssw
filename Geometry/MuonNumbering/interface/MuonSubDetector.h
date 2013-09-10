#ifndef MuonNumbering_MuonSubDetector_h
#define MuonNumbering_MuonSubDetector_h

/** \class MuonSubDetector
 *
 * class to handle muon sensitive detectors,
 * possible arguments for constructor:
 * "MuonDTHits", "MuonCSCHits", "MuonRPCHits"
 *
 * the function suIdName() returns the detector SuId
 * for the ROU factory
 *  
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include<string>

class MuonSubDetector {
 public:

  /*  
   * possible arguments for constructor:
   * "MuonDTHits", "MuonCSCHits", "MuonRPCHits", "MuonGEMHits"
   */

  MuonSubDetector(std::string name);
  ~MuonSubDetector(){};

  bool isBarrel();
  bool isEndcap();
  bool isRpc();
  bool isGem();
  std::string name();
  std::string suIdName();
      
 private:
  enum subDetector {barrel,endcap,rpc,gem,nodef};
  subDetector detector;
  std::string detectorName;
};

#endif
