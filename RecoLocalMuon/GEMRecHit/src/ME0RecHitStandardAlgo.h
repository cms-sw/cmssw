#ifndef RecoLocalMuon_ME0RecHitStandardAlgo_H
#define RecoLocalMuon_ME0RecHitStandardAlgo_H

/** \class ME0RecHitStandardAlgo
 *  Concrete implementation of ME0RecHitBaseAlgo.
 *
 *  $Date: 2014/02/04 10:16:36 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */
#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitBaseAlgo.h"

class ME0RecHitStandardAlgo : public ME0RecHitBaseAlgo {
 public:
  /// Constructor
  ME0RecHitStandardAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~ME0RecHitStandardAlgo();

  // Operations

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);


  virtual bool compute(const ME0DigiPreReco& digi,
                       LocalPoint& point,
                       LocalError& error) const;


};
#endif


