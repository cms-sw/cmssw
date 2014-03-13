#ifndef CondFormats_DTObjects_DTRecoUncertainties_H
#define CondFormats_DTObjects_DTRecoUncertainties_H

/** \class DTRecoUncertainties
 *  DB object for storing DT hit uncertatinties.
 *
 *  \author G. Cerminara - CERN
 */


#include <map>
#include <vector>
#include <string>
#include <stdint.h>

class DTWireId;

class DTRecoUncertainties {
public:
  /// Constructor
  DTRecoUncertainties();

  /// Destructor
  virtual ~DTRecoUncertainties();

  void setVersion(int version) {
    theVersion = version;
  }
  
  /// Label specifying the structure of the payload; currently supported:
  /// "uniformPerStep" (uniform uncertainties per SL and step; index 0-3 = uncertainties for steps 1-4 in cm)
  int version() const {
    return theVersion;
  }

  /// get the uncertainties for the SL correspoding to the given WireId and for the correct step as defined by the algorithm
  float get(const DTWireId& wireid, unsigned int index) const;
  
  /// fills the map
  void set(const DTWireId& wireid, const std::vector<float>& values);
  


  /// Access methods to data
  typedef std::map<uint32_t, std::vector<float> >::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

private:
  
  // map of uncertainties per SL Id. The position in the vector depends on 
  // version() as it depends on the Reco algorithm being used.
  std::map<uint32_t, std::vector<float> > payload;
  
  int theVersion;

};
#endif

