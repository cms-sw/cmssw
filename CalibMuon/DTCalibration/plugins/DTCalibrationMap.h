#ifndef DTCalibration_DTCalibrationMap_H
#define DTCalibration_DTCalibrationMap_H

/** \class DTCalibrationMap
 *  Allow saving and retrieving of calibration constants to/from txt file.
 *  This is mainly provided for backward compatibility with the ORCA MuBarDigiParameters file.
 *  Can be used to save an arbitrary number of constants with the
 *  needed granularity and to retrieve them back using the wireId.
 *  The first 5 fields for each key are allocated to ttri, sigma_ttrig, kfactor, vdrift and sigma_vdrift.
 *
 *  $Date: 2008/12/09 22:43:38 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTWireId.h"


#include <string>
#include <map>
#include <vector>



namespace edm {
  class ParameterSet;
}

class DTCalibrationMap {
public:
  /// Constructor
  DTCalibrationMap(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTCalibrationMap();

  // Operations
  
  /// Return the t_trig (ns) for a particular wire
  float tTrig(DTWireId wireId) const;

  /// Return the sigma of the t_trig (ns) for a particular wire
  float sigma_tTrig(DTWireId wireId) const;

  /// Return the kfactor for a particular wire
  float kFactor(DTWireId wireId) const;

  /// Return the mean drift velocity for a particular wire (cm/ns)
  float meanVDrift(DTWireId wireId) const;

  /// Return the sigma of the mean drift velocity for a particular wire (cm/ns)
  float sigma_meanVDrift(DTWireId wireId) const;

  typedef std::vector<float> CalibConsts;
  typedef DTWireId Key;
  typedef std::map<Key, CalibConsts>::const_iterator const_iterator;

  // Clean the map
  void cleanTheConsts() {
    theMap.clear();
  }

  // Get a particular number (field) between all the calibration
  // constants available for a particluar wire
  float getField(DTWireId wireId, int field) const;

  // Get from the map the calibration constants for a particular wire
  const CalibConsts* getConsts(DTWireId wireId) const;


  // Add to the map the calibration consts for a given key 
  void addCell(Key wireId, const CalibConsts& calibConst);
  
  // Write the calibration consts to a file 
  void writeConsts(const std::string& outputFileName) const;

  // Get a key to read calibration constants for a particular wire
  // with the given granularity
  Key getKey(DTWireId wireId) const;

  const_iterator keyAndConsts_begin() const {
    return theMap.begin();
  }

  const_iterator keyAndConsts_end() const {
    return theMap.end();
  }


protected:

private:

  // Specify the granularity for the calibration constants
  enum CalibGranularity {byChamber,bySL,byLayer,byWire};
  CalibGranularity theGranularity;


  // Read the calibration consts from a file 
  void readConsts(const std::string& inputFileName);


  // Check the consistency of a given key with the selected granularity
  bool checkGranularity(Key aKey) const;


  // The number of fields (calibration numbers) to be read from file
  unsigned int nFields;

  // The name of the file containing the calibration constants
  std::string calibConstFileName;

  // Define the granularity to be used for t0
  std::string calibConstGranularity;

  // The map between the Key and the calibration constants
  std::map<Key, CalibConsts> theMap;
  
};

#endif

