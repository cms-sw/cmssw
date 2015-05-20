#ifndef RecoLocalMuon_ME0RecHitBaseAlgo_H
#define RecoLocalMuon_ME0RecHitBaseAlgo_H

/** \class ME0RecHitBaseAlgo
 *  Abstract algorithmic class to compute Rec Hit
 *  form a ME0 digi
 *
 *  $Date: 2014/02/04 10:16:32 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

class ME0DetId;

namespace edm {
  class ParameterSet;
  class EventSetup;
}


class ME0RecHitBaseAlgo {

 public:

  /// Constructor
  ME0RecHitBaseAlgo(const edm::ParameterSet& config);

 /// Destructor
  virtual ~ME0RecHitBaseAlgo();

 /// Pass the Event Setup to the algo at each event
 virtual void setES(const edm::EventSetup& setup) = 0;
 
 /// Build all hits in the range associated to the me0Id, at the 1st step.
 virtual edm::OwnVector<ME0RecHit> reconstruct(const ME0DetId& me0Id,
                                                  const ME0DigiPreRecoCollection::Range& digiRange);

 /// standard local recHit computation
  virtual bool compute(const ME0DigiPreReco& digi,
                             LocalPoint& Point,
                             LocalError& error) const = 0;

};
#endif

