#ifndef DTSegment_DTLPPatternReco_h
#define DTSegment_DTLPPatternReco_h

/** \class DTLPPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/14 13:22:56 $
 * $Revision: 1.1 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

// Base Class Headers 
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"


//CMSSW Headers
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//#include <DataFormats/GeometrySurface/interface/LocalError.h>

/* Collaborating Class Declarations */
namespace edm {
  class ParameterSet;
  class EventSetup;
}


/* Class DTLPPatternReco Interface */

class DTLPPatternReco : public DTRecSegment2DBaseAlgo {

  public:

    // Constructor
    DTLPPatternReco(const edm::ParameterSet& pset) ;

    // Destructor
    virtual ~DTLPPatternReco() ;

    /* Operations */

    /// this function is called in the producer
    virtual edm::OwnVector<DTSLRecSegment2D>
      reconstruct(const DTSuperLayer* sl,
                  const std::vector<DTRecHit1DPair>& hits);

    /// return the algo name
    virtual std::string algoName() const { return "DTLPPatternReco"; }

    /// Through this function the EventSetup is percolated to the
    /// objs which request it
    virtual void setES(const edm::EventSetup& setup);

private:

  enum ReconstructInSLOrChamber { ReconstructInSL, ReconstructInChamber };

  class ResultLPAlgo{
  public:
      ResultLPAlgo(){ Chi2var =0;};
      ~ResultLPAlgo(){lambdas.clear();};
      double mVar;
      double qVar;
      double chi2Var;
      std::vector<int> lambdas;  
  };

  edm::OwnVector<DTChamberRecSegment2D>
  DTLPPatternReco::reconstructSupersegment(const DTChamber* chamber,
					   const std::vector<DTRecHit1DPair>& pairs);
  
  void reconstructSegmentOrSupersegment(edm::OwnVector<DTSLRecSegment2D>& ResultsSegment,
					edm::OwnVector<DTChamberRecSegment2D>& ResultsSuperSegments,
					const std::vector<DTRecHit1DPair>& pairs,
					const DTSuperLayer* sl,
					const DTChamber* chamber,
					ReconstructInSLOrChamber sl_chamber )

  void populateCoordinatesLists(std::list<double>& pz,
				  std::list<double>& px,
				  std::list<double>& pex,
				  const DTSuperLayer* sl,
				  const DTChamber* chamber,
				  const std::vector<DTRecHit1DPair>& pairs,
				  ReconstructInSLOrChamber sl_chamber);

  bool lpAlgorithm(ResultLPAlgo& theAlgoResults, //This function implements the LP algorithm
		 const std::list<double>& pz,
		 const std::list<double>& px,
		 const std::list<double>& pex,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double theBigM);

  void removeUsedHhits(ResultPerformFit& theAlgoResults,
			std::list<double>& pz,
			std::list<double>& px,
			std::list<double>& pex);

  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry
  double theDeltaFactor, theMinimumM, theMaximumM, theMinimumQ, theMaximumQ, theBigM;
   
};

#endif // DTSegment_DTLPPatternReco_h
