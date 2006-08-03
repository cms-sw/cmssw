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

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

class Type1METAlgo 
{
 public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  typedef std::vector<const CaloJet*> JetInputColl;
  Type1METAlgo();
  virtual ~Type1METAlgo();
  virtual void run(const CaloMETCollection*, JetInputColl, JetInputColl, METCollection &);
};

#endif // Type1METAlgo_h

/*  LocalWords:  Type1METAlgo
 */
