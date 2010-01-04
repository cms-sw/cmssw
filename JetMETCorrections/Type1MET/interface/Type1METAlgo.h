#ifndef Type1METAlgo_h
#define Type1METAlgo_h

/** \class Type1METAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author M. Schmitt, R. Cavanaugh, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

class JetCorrector;

class Type1METAlgo 
{
 public:
  Type1METAlgo();
  virtual ~Type1METAlgo();
  virtual void run(const reco::METCollection&, 
		   const JetCorrector&,
		   const reco::CaloJetCollection&, 
		   double, double, 
		   reco::METCollection *);
  virtual void run(const reco::CaloMETCollection&, 
		   const JetCorrector&,
		   const reco::CaloJetCollection&, 
		   double, double,
		   reco::CaloMETCollection*);
};

#endif // Type1METAlgo_h

/*  LocalWords:  Type1METAlgo
 */
