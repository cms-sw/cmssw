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
#include <stdint.h>

class DTWireId;

class DTRecoUncertainties {
public:
  /// Constructor
  DTRecoUncertainties();

  /// Destructor
  virtual ~DTRecoUncertainties();

  // Operations
  enum Type {
    ldStep1 = 0,
    ldStep3 = 1
  };



  /// get the uncertainties for the SL correspoding to the given WireId and for the correct step as defined by the algorithm
  float get(const DTWireId& wireid, DTRecoUncertainties::Type) const;
  
  /// fills the map
  void set(const DTWireId& wireid, DTRecoUncertainties::Type type, float value);

  

private:


  
  // map of uncertainties per SL Id. The position in the vector is determined by the 
  // DTRecoUncertainties::Type as it depends on the Reco algorithm e=being used.
  std::map<uint32_t, std::vector<float> > payload;
  

};
#endif

