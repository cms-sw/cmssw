/** \class DTLPPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/14 13:22:56 $
 * $Revision: 1.1 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

/* C++ Headers*/
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <math.h>
#include <iostream>

/* Collaborating class headers*/
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

// Linear Programming Header
extern "C" {
#include "glpk.h"
}

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTLPPatternReco.h"

//Constructor
DTLPPatternReco::DTLPPatternReco(const edm::ParameterSet& pset): DTRecSegment2DBaseAlgo(pset){
  std::cout << "DTLPPatternReco Constructor Called" << std::endl; 
  //getting parameters from python
  theDeltaFactor = pset.getParameter<double>("DeltaFactor");//multiplier of sigmas in LP algo
  theMinimumM = pset.getParameter<double>("min_m");
  theMaximumM = pset.getParameter<double>("max_m");
  theMinimumQ = pset.getParameter<double>("min_q");
  theMaximumQ = pset.getParameter<double>("max_q");
  theBigM = pset.getParameter<double>("bigM");//"Big M" used in LP
}


//Destructor
DTLPPatternReco::~DTLPPatternReco() {
std::cout << "DTLPPatternReco Destructor Called" << std::endl;
}

void DTLPPatternReco::setES(const edm::EventSetup& setup){
   // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  }

edm::OwnVector<DTSLRecSegment2D> DTLPPatternReco::reconstruct(const DTSuperLayer* sl, const std::vector<DTRecHit1DPair>& pairs){
  edm::OwnVector<DTSLRecSegment2D> theResults;
  reconstructSegmentOrSupersegment(theResults, NULL, pairs, sl, NULL, ReconstructInSL);
  return theResults
}

edm::OwnVector<DTChamberRecSegment2D> DTLPPatternReco::reconstructSupersegment(const DTChamber* chamber, const std::vector<DTRecHit1DPair>& pairs){
  edm::OwnVector<DTChamberRecSegment2D> theResults;
  reconstructSegmentOrSupersegment(NULL, theResults, pairs, NULL, chamber,  ReconstructInChamber);
  return theResults
}

void DTLPPatternReco::reconstructSegmentOrSupersegment(edm::OwnVector<DTSLRecSegment2D>& ResultsSegment, edm::OwnVector<DTChamberRecSegment2D>& ResultsSuperSegments, const std::vector<DTRecHit1DPair>& pairs, const DTSuperLayer* sl, const DTChamber* chamber, ReconstructInSLOrChamber sl_chamber ){
  if (sl_chamber == ReconstructInSL && sl == NULL) 
    sl = theDTGeometry -> SuperLayer( pairs.begin()->WireId().SLId() );
  if (sl_chamber == ReconstructInChamber && chamber == NULL)
    chamber = theDTGeometry -> Chamber( pairs.begin()->WireId().ChamberId() );
  std::list<double> pz; //positions of hits along z
  std::list<double> px; //along x (left and right the wires)
  std::list<double> pex; //errors (sigmas) on x
  ResultLPAlgo theAlgoResults;//datastructure containing all useful results from perform_fit
  populateCoordinatesLists(pz, px, pex, pairs, sl, chamber, sl_chamber);//in SL or chamber ref frame
  while(lpAlgo(theAlgoResults, pz, px, pex, theMinimumM, theMaximumM, theMinimumQ, theMaximumQ, theBigM) ){
    std::cout << "Creating 2Dsegment" << std::endl;
    LocalPoint seg2Dposition( (float)theAlgoResults.Qvar, 0. , 0. );
    LocalVector seg2DDirection ((float) theAlgoResults.Mvar,  0. , 1.  );//don't know if I need to normalize the vector
    AlgebraicSymMatrix seg2DCovMatrix;
    std::vector<DTRecHit1D> hits1D;
    for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
      if(theAlgoResults.lambdas[i]%2) hits1D.push_back(pairs[(theAlgoResults.lambdas[i] - 1)/2].componentRecHit(DTEnums::Right));
      else hits1D.push_back(pairs[theAlgoResults.lambdas[i]/2].componentRecHit(DTEnums::Left));
    }
  if(sl_chamber == ReconstructInSL) ResultsSegments.push_back(new DTSLRecSegment2D(sl->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.Chi2var, hits1D));
  if(sl_chamber == ReconstructInChamber) ResultsSuperSegments.push_back(new DTChamberRecSegment2D(chamber->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.Chi2var, hits1D) );
  std::cout << "2D segment created, removing used hits" << std::endl;
  remove_used_hits(theAlgoResults, pz, px, pex);
  std::cout << "Used hits removed" << std::endl;
  theAlgoResults.lambdas.clear();
  std::cout << "Checking if I need to reiterate perform_fit" << std::endl;
  }
  px.clear();
  pz.clear();
  pex.clear();
}

void DTLPPatternReco::populateCoordinatesLists(std::list<double>& pz,  std::list<double>& px,  std::list<double>& pex, const DTSuperLayer* sl, const DTChamber* chamber, const std::vector<DTRecHit1DPair>& pairs, ReconstructInSLOrChamber sl_chamber) {
  //populate the arrays with positions in the SuperLayer (or chamber) reference frame, iterating on pairs
  for (std::vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it!=pairs.end(); ++it){
    DTWireId theWireId = it->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    //transform the Local position in Layer-rf in a SL local position, or chamber
    std::pair<const DTRecHit1D*, const DTRecHit1D*> theRecHitPair = it -> componentRecHits();
    LocalPoint thePosition;
    //left hit
    if(!ChamberCoord) thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.first-> localPosition()));
    else thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.first -> localPosition()));
    pex.push_back( (double)std::sqrt(theRecHitPair.first -> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    std::cout << "DTLPPatternReco::reconstruct left : px =  = " << px.back() << " pz = " << pz.back() << " pex = "<< pex.back() << std::endl;
    //and right hit
    if(!ChamberCoord) thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.second->localPosition()));
    else thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.second -> localPosition()));
    pex.push_back( (double)std::sqrt(theRecHitPair.second-> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    std::cout << "DTLPPatternReco::reconstruct right : px = "  << px.back() << " pz = " << pz.back() 
	      << " pex = "<< pex.back() << std::endl;
    }
  std::cout << "DTLPPatternReco:: : px, pz and pex lists populated " << std::endl;
}


void DTLPPatternReco::removeUsedHits(ResultLPAlgo& theAlgoResults, std::list<double>& pz,  std::list<double>& px,  std::list<double>& pex){
  int jz =0;
  int jx =0;
  int jex =0;  
  std::list<double>::iterator itz = pz.begin();
  std::list<double>::iterator itx = px.begin();
  std::list<double>::iterator itex = pex.begin();
  for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
    for (; itz != pz.end(); ++++itz) {
      if (theAlgoResults.lambdas[i]/2  == jz/2 ){
	itz = pz.erase(itz);
	itz = pz.erase(itz);
	break;
      }
      jz += 2;
    }
    for (; itx != px.end(); ++++itx) {
      if (theAlgoResults.lambdas[i]/2 == jx/2){
	itx = px.erase(itx);
	itx = px.erase(itx);
	break;
      }
      jx += 2;
    }
    for (; itex != pex.end(); ++++itex) {
      if (theAlgoResults.lambdas[i]/2 == jex/2){
	itex = pex.erase(itex);
	itex = pex.erase(itex);
	break;
      }
      jex += 2;
    }
  }
}

bool DTLPPatternReco::lpAlgorithm(ResultLPAlgo& theAlgoResults, const std::list<double>& pz, const std::list<double>& px, const std::list<double>& pex, const double m_min, const double m_max, const double q_min, const double q_max, const double BIG_M)
    {
    //the data struct representing the problem
	LPX *lp;
    //the vectors that will contain indices over rows and columns to load in the problem matrix
    std::vector<int> ia, ja;
    //the vector that will contain coefficients to load in the problem matrix
    std::vector<double> ar;
    lp = lpx_create_prob();
	lpx_set_class(lp, LPX_MIP);//mixed integer prog.
	lpx_set_prob_name(lp, "trackFit");
	//setting a minimization problem
	lpx_set_obj_dir(lp, LPX_MIN);
    
    const unsigned int n_points = pz.size();
    if (n_points <= 4) return false;
	

    /*******************COLUMNS DEFINITION********************************
    *                                                                    */ 
    //columns are the structural variables
    //for each point there is sigma and lambda
    //than we have the m and q (m+, m- ...)
    const int n_cols = 2 * n_points + 4;
    lpx_add_cols(lp, n_cols);
    // set columns boundaries	
    lpx_set_col_bnds(lp, 1, LPX_LO, 0.0, 0.0); // m+ >=0
    lpx_set_col_bnds(lp, 2, LPX_LO, 0.0, 0.0); // m- >=0
    lpx_set_col_bnds(lp, 3, LPX_LO, 0.0, 0.0); // q+ >=0
    lpx_set_col_bnds(lp, 4, LPX_LO, 0.0, 0.0); // m- >=0
    for (unsigned int i=0; i<n_points; i++) {
        // sigmas >= 0
        lpx_set_col_bnds(lp, 4 + i + 1, LPX_LO, 0.0, 0.0);
        // 0 <= lambdas <= 1 
        lpx_set_col_bnds(lp, n_points + 4 + i + 1, LPX_DB, 0., 1.);
        // lambdas are integer (binary)
        lpx_set_col_kind(lp, n_points + 4 + i + 1, LPX_IV);
        }     
     
    /*******************ROWS DEFINITION*****************************
     *                                                             */		
	//rows are auxiliary variables (those not appearing in the objective func)
	// 4 are needed by the inequalities for each point (at page 3 of the note)
	// n/2 are the constraints over each pair of points, i.e. only one at most can be used 
	// 2 are m and q
	const int n_rows = 4 * n_points + n_points/2 + 2; 
	lpx_add_rows(lp, n_rows);
	//set rows boundaries
	//constraints over the first four inequalites
	int i = 0;
	for (std::list<double>::const_iterator it = px.begin(); it != px.end(); it++) {
	  lpx_set_row_bnds(lp, i + 1, LPX_UP, 0., *it); //a1 <= Xi
	  lpx_set_row_bnds(lp, n_points + i + 1, LPX_LO, *it, 0.); //a2 >= Xi
	  i++;
	}
	i =0;
	for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); it++) {
        lpx_set_row_bnds(lp, 2 * n_points + i + 1, LPX_UP, 0., (*it) * theDeltaFactor ); //a3 <= Di
        lpx_set_row_bnds(lp, 3 * n_points + i + 1, LPX_UP, 0., BIG_M - ((*it) * theDeltaFactor)); //a4 <= M-Di
	i ++;
	}
        // Constraints on lambda pairs
	for (unsigned int i=0; i<n_points; i++) if (i%2==0) lpx_set_row_bnds(lp, 4 * n_points + i/2 + 1, LPX_LO, 1., 0.);			
	// Constraints on m and q
	lpx_set_row_bnds(lp, n_rows-1, LPX_DB, m_min, m_max);// m_min <= m <= m_max
	lpx_set_row_bnds(lp, n_rows, LPX_DB, q_min, q_max);// q_min <= q <= q_max
	
    /******************OBJECTIVE FUNCTION*********************************
     *                                                                   */
    // set the objective function coefficients
    //first four are 0 because the m+ m- q+ q- do not appear in the objective function
    lpx_set_obj_coef(lp, 1, 0.);//m+
    lpx_set_obj_coef(lp, 2, 0.);//m-
    lpx_set_obj_coef(lp, 3, 0.);//q+
    lpx_set_obj_coef(lp, 4, 0.);//q-
    // for the next structural variables (sigmas) the coefficient is 1/delta
    i = 0;
    for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); it++){
      lpx_set_obj_coef(lp, i+5, 1./( theDeltaFactor * (*it))); // sigma_i / Delta_i
      i++;
    }
    // than we have the coefficients for lambda: M (to maximize number of points, 
    //because we are minimizing the objective function)
    for (unsigned int i=0; i<n_points; i++) lpx_set_obj_coef(lp, n_points + 5 + i,  BIG_M );

    /******************SETTING MATRIX COEFFICIENTS*********************************
     *                                                                            */
    // a good thing to know is that only non-zero elements need to be passed to the loader (we can safely skip the zeros)
    //ia indexes rows, ja columns ar contains the coefficient for given row and col.
    i = 0;
    for (std::list<double>::const_iterator it = pz.begin(); it != pz.end(); it++) {//FIXME use static arrays
   
        //first equation(row): m * Zi + q - sigma_i - lambda_i * M
        ia.push_back(i + 1), ja.push_back(1), ar.push_back(*it);//m+ * Zi
        ia.push_back(i + 1), ja.push_back(2), ar.push_back(-(*it));//m- * -Zi
        ia.push_back(i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
        ia.push_back(i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
        ia.push_back(i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
        ia.push_back(i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(-BIG_M);// -M * lambda_i       
        
        //second equation: m * Zi + q + sigma_i + lambda_i * M
        ia.push_back(n_points + i + 1), ja.push_back(1), ar.push_back(*it);//m+ * Zi
        ia.push_back(n_points + i + 1), ja.push_back(2), ar.push_back(-(*it));//m- * -Zi
        ia.push_back(n_points + i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
        ia.push_back(n_points + i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
        ia.push_back(n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(1);// +1 * sigma_i
        ia.push_back(n_points + i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(BIG_M);// M * lambda_i        
        
        //third equation: sigma_i - M * lambda_i
        ia.push_back(2 * n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(1);// +1 * sigma_i
        ia.push_back(2 * n_points + i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(-BIG_M);// -M * lambda_i        
        
        //fourth equation: -sigma_i + M * lambda_i  
        ia.push_back(3 * n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
        ia.push_back(3 * n_points + i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(BIG_M);// M * lambda_i        
        
        // exclusivity of pairs
        if (i%2==0) {
            ia.push_back(4 * n_points + i/2 + 1), ja.push_back( n_points + 4 + i + 1 ), ar.push_back(1);// +1 * lambda_i
            ia.push_back(4 * n_points + i/2 + 1), ja.push_back( n_points + 4 + i + 1 + 1 ), ar.push_back(1);// +1 * lambda_i+1
            }
	i++;
        }
    // constraints on m: m = m+ - m-
    ia.push_back(4 * n_points + n_points/2 + 1), ja.push_back(1 ), ar.push_back(1);// +1 * m+
    ia.push_back(4 * n_points + n_points/2 + 1), ja.push_back(2 ), ar.push_back(-1);// -1 * m-
    // constraints on q: q = q+ - q-
    ia.push_back(4 * n_points + n_points/2 + 1), ja.push_back(3 ), ar.push_back(1);// +1 * q+
    ia.push_back(4 * n_points + n_points/2 + 1), ja.push_back(4 ), ar.push_back(-1);// -1 * m-	

    lpx_load_matrix(lp, ia.size(), (int*)(&ia[0]-1), (int*)(&ja[0]-1), (double*)(&ar[0]-1));
    ia.clear();
    ja.clear();
    ar.clear();

    /******************SOLUTION*********************************
     *                                                         */
    
    if(!lpx_simplex(lp)) return false;
    lpx_integer(lp);    
    //lpx_intopt(lp);
    //for (int i=0; i<n_cols; i++) printf("%d --> %f\n",i,lpx_mip_col_val(lp, i+1));
    //I must return the m and q values found, and all values af lambdas (to know which point has been used)
    theAlgoResults.Mvar = lpx_mip_col_val(lp, 1) - lpx_mip_col_val(lp, 2);//push back m
    theAlgoResults.Qvar =  lpx_mip_col_val(lp, 3) - lpx_mip_col_val(lp, 4);//push back q
    theAlgoResults.lambdas.clear();
    i = 0;
    for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); ++it){
      // theAlgoResults.lambdas.push_back( lpx_mip_col_val(lp, n_points + 4 + i + 1) );//push back lambdas
      if (!lpx_mip_col_val(lp, n_points + 4 + i + 1)) {
	//std::cout << "found a lambda = 0" << std::endl;
	theAlgoResults.lambdas.push_back(i);
	theAlgoResults.Chi2var += ((lpx_mip_col_val(lp, 4 + i + 1) - *it))*((lpx_mip_col_val(lp, 4 + i + 1) - *it)) / ((*it) * (*it));
      }
      i++;
    }
    if (theAlgoResults.lambdas.size() < 3) return false;
    std::cout << "We have used "<< theAlgoResults.lambdas.size()<< " hits"  <<std::endl;
    //printf("y =  %f * x  + %f\n", res[0], res[1]);
    //printf("obj func = %f\n",lpx_mip_obj_val(lp));
    lpx_delete_prob(lp);

    std::cout << "DTLPPatternReco::perform_fit : m = " << theAlgoResults.Mvar << " q = " << theAlgoResults.Qvar << std::endl;
    return true;
}



