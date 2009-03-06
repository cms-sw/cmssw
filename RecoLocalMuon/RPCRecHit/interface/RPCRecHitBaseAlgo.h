#ifndef RecoLocalMuon_RPCRecHitBaseAlgo_H
#define RecoLocalMuon_RPCRecHitBaseAlgo_H

/** \class RPCRecHitBaseAlgo
 *  Abstract algorithmic class to compute Rec Hit
 *  form a RPC digi
 *
 *  $Date: 2006/04/18 16:28:00 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

class RPCCluster;
class RPCRoll;
class RPCDetId;

namespace edm {
  class ParameterSet;
  class EventSetup;
}


class RPCRecHitBaseAlgo {

 public:
  
  /// Constructor
  RPCRecHitBaseAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~RPCRecHitBaseAlgo();  

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup) = 0;

  /// Build all hits in the range associated to the rpcId, at the 1st step.
  virtual edm::OwnVector<RPCRecHit> reconstruct(const RPCRoll& roll,
						const RPCDetId& rpcId,
						const RPCDigiCollection::Range& digiRange);

  /// standard local recHit computation
  virtual bool compute(const RPCRoll& roll,
                       const RPCCluster& cl,
                       LocalPoint& Point,
                       LocalError& error) const = 0;


  /// local recHit computation accounting for track direction and 
  /// absolute position
  virtual bool compute(const RPCRoll& roll,
		       const RPCCluster& cl,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& Point,
                       LocalError& error) const = 0;
};
#endif
