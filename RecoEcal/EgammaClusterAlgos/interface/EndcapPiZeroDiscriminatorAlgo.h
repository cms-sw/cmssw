#ifndef RecoEcal_EgammaClusterAlgos_EndcapPiZeroDiscriminatorAlgo_h
#define RecoEcal_EgammaClusterAlgos_EndcapPiZeroDiscriminatorAlgo_h
//
// $Id: EndcapPiZeroDiscriminatorAlgo.h,v 1.1 2006/09/11 12:17:30 futyand Exp $
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"


// C/C++ headers
#include <string>
#include <vector>
#include <set>
#include <fstream>

class EndcapPiZeroDiscriminatorAlgo {

 public:
 
   enum DebugLevel_pi0 { pDEBUG = 0, pINFO = 1, pERROR = 2 };

   typedef math::XYZPoint Point;

   typedef std::map<DetId, EcalRecHit> RecHitsMap;

   EndcapPiZeroDiscriminatorAlgo() : 
   preshStripEnergyCut_(0.), preshSeededNstr_(5), debugLevel_(pINFO)
   {}

   EndcapPiZeroDiscriminatorAlgo(double stripEnergyCut, int nStripCut, DebugLevel_pi0 debugLevel = pINFO ) :
   preshStripEnergyCut_(stripEnergyCut),  preshSeededNstr_(nStripCut), debugLevel_(debugLevel)
   {}

   ~EndcapPiZeroDiscriminatorAlgo() {};
   
// Aris 10/7/2006
// ---------------
   std::vector<float>  findPreshVector(ESDetId strip, RecHitsMap *rechits_map,
					 CaloSubdetectorTopology *topology_p);
//   void findPreshVector(ESDetId strip, RecHitsMap *rechits_map,
//					 CaloSubdetectorTopology *topology_p, std::vector<float>& vout_stripE);

   void findPi0Road(ESDetId strip, EcalPreshowerNavigator theESNav, int plane, std::vector<ESDetId>& vout);

   bool goodPi0Strip(RecHitsMap::iterator candidate_it, ESDetId lastID);

   void readWeightFile(char *WFile);

   float Activation_fun(float SUM);

   float getNNoutput(float *input);

   void calculateNNInputVariables(std::vector<float>& vph1, std::vector<float>& vph2,
                                                                      float pS1_max, float pS9_max, float pS25_max, 
                                                                      float *nn_invar);
   float GetNNOutput(float Et_SE, float *input_var);
 private:
  
   double preshStripEnergyCut_;
   int preshSeededNstr_;
   int debugLevel_;

   int inp_var;
   int Layers, Indim, Hidden, Outdim;

   float* I_H_Weight;
   float* H_O_Weight;
   float* H_Thresh;
   float* O_Thresh;

//   std::vector<ESDetId> road_2d;

   // The map of hits
   RecHitsMap *rechits_map;

};
#endif

