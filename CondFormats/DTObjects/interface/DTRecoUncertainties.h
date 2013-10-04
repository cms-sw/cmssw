#ifndef CondFormats_DTObjects_DTRecoUncertainties_H
#define CondFormats_DTObjects_DTRecoUncertainties_H

/** \class DTRecoUncertainties
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
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

  void setType(const std::string& tt) {
    theType = tt;
  };
  
  std::string type() const {
    return theType;
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
  
  // map of uncertainties per SL Id. The position in the vector is determined by the 
  // DTRecoUncertainties::Type as it depends on the Reco algorithm e=being used.
  std::map<uint32_t, std::vector<float> > payload;
  
  std::string theType;

};
#endif

