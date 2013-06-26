#ifndef DTTMax_H
#define DTTMax_H

/** \class DTTMax
 *  Class to calculate the different TMax values according to
 *  the track path
 *
 *  $Date: 2013/05/23 15:28:44 $
 *  $Revision: 1.2 $

 *  \author Marina Giunta
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <string>
#include <vector>

class DTSuperLayer;
class DTSuperLayerId;
class DTTTrigBaseSync;

namespace dttmaxenums{
  enum TMaxCells {c123, c124, c134, c234, notInit};
  enum SigmaFactor{r32, r72, r78, noR};
  enum SegDir {L, R};
}


class DTTMax {
 public:
  typedef dttmaxenums::TMaxCells TMaxCells;
  typedef dttmaxenums::SegDir SegDir;
  typedef dttmaxenums::SigmaFactor SigmaFactor;
  
  /// Constructor
  DTTMax(const std::vector<DTRecHit1D> & hits, const DTSuperLayer & isl, GlobalVector dir, 
	 GlobalPoint pos, DTTTrigBaseSync* sync);
  
  /// Destructor
  virtual ~DTTMax();
  
  /// Information on each of the four TMax values in a SL
  struct TMax{
    TMax(float t_, TMaxCells cells_, std::string type_, SigmaFactor sigma_, 
	 unsigned t0Factor_,unsigned hSubGroup_) :
    t(t_), cells(cells_), type(type_), sigma(sigma_), t0Factor(t0Factor_), hSubGroup(hSubGroup_) {}
    
    float t;
    TMaxCells cells;
    std::string type;       // LLR, LRL,...
    SigmaFactor sigma; // factor relating the width of the Tmax distribution 
                       // and the cell resolution
    unsigned t0Factor; // "quantity" of Delta(t0) included in the tmax formula
    unsigned hSubGroup;//different t0 hists (one hit within a given distance from the wire)
  };

  // All information on one of the layers crossed by the segment
  struct InfoLayer {
    InfoLayer(const DTRecHit1D& rh_, const DTSuperLayer & isl, GlobalVector dir, 
	      GlobalPoint pos, DTTTrigBaseSync* sync);
    DTRecHit1D rh;
    DTWireId idWire;
    DTEnums::DTCellSide lr;
    float wireX;
    float time;
  };

  // Return the three TMax for a given cell
  std::vector<const TMax*> getTMax(const DTWireId & idWire);

  // Return the four TMaxes of the SL
  std::vector<const TMax*> getTMax(const DTSuperLayerId & isl);

  // Return one of the four TMaxes of the SL
  const TMax* getTMax(TMaxCells cCase);

  // Get InfoLayer (r/w) from layer number
  InfoLayer*& getInfoLayer(int layer) {return theInfoLayers[layer-1];}

 private:
  DTTMax(){}; // Hide default constructor

  //debug flag 
  bool debug;

  std::vector<InfoLayer*> theInfoLayers;
  std::vector<TMax*> theTMaxes;
  SegDir theSegDir;
  std::string theSegType; // LRLR, LRLL, ....

};
#endif

