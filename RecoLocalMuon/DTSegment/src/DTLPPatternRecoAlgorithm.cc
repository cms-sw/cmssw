/** DTLPPatternReco Algorithm
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/25 18:20:55 $
 * $Revision: 1.3 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

#include "RecoLocalMuon/DTSegment/src/DTLPPatternRecoAlgorithm.h"

#include <list>
#include <iostream>

// Linear Programming Header
extern "C" {
#include "glpk.h"
}

using namespace std;

bool lpAlgorithm(lpAlgo::ResultLPAlgo& theAlgoResults,
		 const list<double>& pz,
		 const list<double>& px,
		 const list<double>& pex,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double BIG_M, const double theDeltaFactor)
    {
      bool debug = false;
    const unsigned int n_points = pz.size();
    if (n_points <= 4) return false;
    //the data struct representing the problem
    glp_prob *lp = 0;
    //the vectors that will contain indices over rows and columns to load in the problem matrix
    vector<int> ia, ja;
    //the vector that will contain coefficients to load in the problem matrix
    vector<double> ar;
    lp = glp_create_prob();

    // glp_set_class(lp, GLP_MIP);//mixed integer prog.
    glp_set_prob_name(lp, "trackFit");
    //setting a minimization problem
    glp_set_obj_dir(lp, GLP_MIN);
    
    if(debug)
    cout << "[lpAlgorithm] LP-MIP Problem Created, # points = " << pz.size()<< " "  << px.size() << " "  << pex.size() << endl
	 << "              m_min= " << m_min <<  " m_max= " << m_max
	 << " q_min= " << q_min <<  " q_max= " << q_max << endl
	 << "              big_m= " << BIG_M << " delta_fact = " << theDeltaFactor << endl;
    

    /*******************COLUMNS DEFINITION********************************
    *                                                                    */ 
    //columns are the structural variables
    //for each point there is sigma and lambda
    //than we have the m and q (m+, m- ...)
    const int n_cols = 2 * n_points + 4;
    glp_add_cols(lp, n_cols);
    // set columns boundaries	
    glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0); // m+ >=0
    glp_set_col_bnds(lp, 2, GLP_LO, 0.0, 0.0); // m- >=0
    glp_set_col_bnds(lp, 3, GLP_LO, 0.0, 0.0); // q+ >=0
    glp_set_col_bnds(lp, 4, GLP_LO, 0.0, 0.0); // m- >=0
    for (unsigned int i=0; i<n_points; i++) {
        // sigmas >= 0
        glp_set_col_bnds(lp, 4 + i + 1, GLP_LO, 0.0, 0.0);
        // 0 <= lambdas <= 1 
        glp_set_col_bnds(lp, n_points + 4 + i + 1, GLP_DB, 0., 1.);
        // lambdas are integer (binary)
        glp_set_col_kind(lp, n_points + 4 + i + 1, GLP_IV);
        }     
     
    if(debug) cout << "Columns defined" << endl;
    /*******************ROWS DEFINITION*****************************
     *                                                             */		
	//rows are auxiliary variables (those not appearing in the objective func)
	// 4 are needed by the inequalities for each point (at page 3 of the note)
	// n/2 are the constraints over each pair of points, i.e. only one at most can be used 
	// 2 are m and q
	const int n_rows = 4 * n_points + n_points/2 + 2; 
	glp_add_rows(lp, n_rows);
	//set rows boundaries
	//constraints over the first four inequalites
	int i = 0;
	for (list<double>::const_iterator it = px.begin(); it != px.end(); it++) {
	  glp_set_row_bnds(lp, i + 1, GLP_UP, 0., *it); //a1 <= Xi
	  glp_set_row_bnds(lp, n_points + i + 1, GLP_LO, *it, 0.); //a2 >= Xi
	  i++;
	}
	i =0;
	for (list<double>::const_iterator it = pex.begin(); it != pex.end(); it++) {
        glp_set_row_bnds(lp, 2 * n_points + i + 1, GLP_UP, 0., (*it) * theDeltaFactor ); //a3 <= Di
        glp_set_row_bnds(lp, 3 * n_points + i + 1, GLP_UP, 0., BIG_M - ((*it) * theDeltaFactor)); //a4 <= M-Di
	i ++;
	}
        // Constraints on lambda pairs
	for (unsigned int i=0; i<n_points; i++) if (i%2==0) glp_set_row_bnds(lp, 4 * n_points + i/2 + 1, GLP_LO, 1., 0.);			
	// Constraints on m and q
	glp_set_row_bnds(lp, n_rows-1, GLP_DB, m_min, m_max);// m_min <= m <= m_max
	glp_set_row_bnds(lp, n_rows, GLP_DB, q_min, q_max);// q_min <= q <= q_max
    
	if(debug) cout << "Rows defined" << endl;
	
    /******************OBJECTIVE FUNCTION*********************************
     *                                                                   */
    // set the objective function coefficients
    //first four are 0 because the m+ m- q+ q- do not appear in the objective function
    glp_set_obj_coef(lp, 1, 0.);//m+
    glp_set_obj_coef(lp, 2, 0.);//m-
    glp_set_obj_coef(lp, 3, 0.);//q+
    glp_set_obj_coef(lp, 4, 0.);//q-
    // for the next structural variables (sigmas) the coefficient is 1/delta
    i = 0;
    for (list<double>::const_iterator it = pex.begin(); it != pex.end(); it++){
      glp_set_obj_coef(lp, i+5, 1./( theDeltaFactor * (*it))); // sigma_i / Delta_i
      i++;
    }
    // than we have the coefficients for lambda: M (to maximize number of points, 
    //because we are minimizing the objective function)
    for (unsigned int i=0; i<n_points; i++) glp_set_obj_coef(lp, n_points + 5 + i,  BIG_M );
    
    if(debug) cout << "Objective funct. defined" << endl;

    /******************SETTING MATRIX COEFFICIENTS*********************************
     *                                                                            */
    // a good thing to know is that only non-zero elements need to be passed to the loader (we can safely skip the zeros)
    //ia indexes rows, ja columns ar contains the coefficient for given row and col.
    i = 0;
    for (list<double>::const_iterator it = pz.begin(); it != pz.end(); it++) {//FIXME use static arrays
   
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
    ia.push_back(4 * n_points + n_points/2 + 2), ja.push_back(3 ), ar.push_back(1);// +1 * q+
    ia.push_back(4 * n_points + n_points/2 + 2), ja.push_back(4 ), ar.push_back(-1);// -1 * m-	

    glp_load_matrix(lp, ia.size(), (int*)(&ia[0]-1), (int*)(&ja[0]-1), (double*)(&ar[0]-1));
    ia.clear();
    ja.clear();
    ar.clear();
    
    if(debug) cout << "Matrix coeff. set" << endl;

    /******************SOLUTION*********************************
     *                                                         */

    // define the control parameters
    glp_iocp parm;
    glp_smcp parm1;
    glp_init_iocp(&parm);
    glp_init_smcp(&parm1);
    if(debug) { // set the GLPK verbosity level 
      parm.msg_lev = GLP_MSG_ALL; 
      parm1.msg_lev = GLP_MSG_ALL; 
    } else {
      parm.msg_lev = GLP_MSG_OFF; 
      parm1.msg_lev = GLP_MSG_OFF; 
    }
    parm1.it_lim = 1000; // set the max # of iterations // FIXME: this is arbitrary
    //if(!glp_simplex(lp, &parm1)) return false;
    int retSimplex = glp_simplex(lp, &parm1);
//     if(debug) cout << "simplex returned: " << retSimplex <<endl;
    // check the return value of the algo
    if(retSimplex != 0) {
      cout << "[lpAlgorithm]***Warning: glp_simplex return code" << endl;
      if(debug) printGLPReturnCode(retSimplex);
      return false;
    }

    // Check the status of the simplex solution
    int statusSimplSol = glp_get_status(lp);
    if(statusSimplSol != GLP_OPT) {
      cout << "[lpAlgorithm]***Warning: wrong simplex solution status" << endl;
      if(debug) printGLPSolutionStatus(statusSimplSol);
      return false;
    }

    //if(!glp_intopt(lp, &parm)) return false;
    int retGlpIntopt = glp_intopt(lp, &parm);
//     if(debug) cout << "intopt returned: " << retGlpIntopt <<endl;
    // check the return value of the algo
    if(retGlpIntopt != 0) {
      cout << "[lpAlgorithm]***Warning: glp_intopt return code " << endl;
      if(debug) printGLPReturnCode(retGlpIntopt);
      int retLpxIntopt = lpx_intopt(lp); // FIXME: experimental, maybe we can move to this one by default
      if(debug) printLPXReturnCode(retLpxIntopt);
      if(retLpxIntopt != LPX_E_OK) return false;
    }

    // check the status of the MIP solution 
    int statusMIPSol = glp_mip_status(lp);
    if(statusMIPSol != GLP_OPT) {
      cout << "[lpAlgorithm]***Warning: wrong MIP solution status" << endl;
      if(debug) printGLPSolutionStatus(statusMIPSol);
      return false;
    }

    //glp_integer(lp);    
    //lpx_intopt(lp);
    //for (int i=0; i<n_cols; i++) printf("%d --> %f\n",i,lpx_mip_col_val(lp, i+1));
    //I must return the m and q values found, and all values af lambdas (to know which point has been used)
    theAlgoResults.mVar = glp_mip_col_val(lp, 1) - glp_mip_col_val(lp, 2);//push back m
    theAlgoResults.qVar =  glp_mip_col_val(lp, 3) - glp_mip_col_val(lp, 4);//push back q
    theAlgoResults.lambdas.clear();
    i = 0;
    int control = 0;
    for (list<double>::const_iterator it = pex.begin(); it != pex.end(); ++it){
      // theAlgoResults.lambdas.push_back( lpx_mip_col_val(lp, n_points + 4 + i + 1) );//push back lambdas
      if (!glp_mip_col_val(lp, n_points + 4 + i + 1)) {
// 	if(debug) cout << "found a lambda = 0" << endl;
	theAlgoResults.lambdas.push_back(0);
	control ++;
	theAlgoResults.chi2Var += ((glp_mip_col_val(lp, 4 + i + 1) - *it))*((glp_mip_col_val(lp, 4 + i + 1) - *it)) / ((*it) * (*it));
      }
      else theAlgoResults.lambdas.push_back(1);
      i++;
    }
    // check that we used more than 2 hits 
    if (control < 3) { 
      glp_delete_prob(lp);
      return false;
    }


    // delete the problem and free the memory  
    glp_delete_prob(lp);

    if(debug) {
      cout << "[lpAlgorithm] # of points used: " << control << " hits"  << endl;
      cout << "              m = " << theAlgoResults.mVar << " q = " << theAlgoResults.qVar << endl;
    }
    return true;
}



void printGLPReturnCode(int returnCode) {
  cout << "   GLP return code is: ";
  switch(returnCode) {
  case GLP_EBADB: {
    cout << " invalid basis" << endl;
    break;
  }
  case GLP_ESING: {
    cout << " singular matrix" << endl;
    break;
  }
  case GLP_ECOND: {
    cout << " ill-conditioned matrix" << endl;
    break;
  }
  case GLP_EBOUND: {
    cout << " invalid bounds" << endl;
    break;
  }
  case GLP_EFAIL: {
    cout << " solver failed" << endl;
    break;
  }
  case GLP_EOBJLL: {
    cout << " objective lower limit reached" << endl;
    break;
  }
  case GLP_EOBJUL: {
    cout << " objective upper limit reached" << endl;
    break;
  }
  case GLP_EITLIM: {
    cout << " iteration limit exceeded" << endl;
    break;
  }
  case GLP_ETMLIM: {
    cout << " time limit exceeded" << endl;
    break;
  }
  case GLP_ENOPFS: {
    cout << " no primal feasible solution" << endl;
    break;
  }
  case GLP_ENODFS: {
    cout << " no dual feasible solution" << endl;
    break;
  }
  case GLP_EROOT: {
    cout << " root LP optimum not provided" << endl;
    break;
  }
  case GLP_ESTOP: {
    cout << " search terminated by application" << endl;
    break;
  }
  case GLP_EMIPGAP: {
    cout << " relative mip gap tolerance reached" << endl;
    break;
  }
  case GLP_ENOFEAS: {
    cout << " no primal/dual feasible solution" << endl;
    break;
  }
  case GLP_ENOCVG: {
    cout << " no convergence" << endl;
    break;
  }
  case GLP_EINSTAB: {
    cout << " numerical instability" << endl;
    break;
  }
  case GLP_EDATA: {
    cout << " invalid data" << endl;
    break;
  }
  case GLP_ERANGE: {
    cout << " result out of range" << endl;
    break;
  }
  default: {
    cout << "!!! unknown return code " << returnCode << " !!!" << endl;
  }
  }
}


void printGLPSolutionStatus(int status) {
  cout << "   GLP status code is: ";
  switch(status) {
  case GLP_UNDEF: {
    cout << " solution is undefined" << endl;
    break;
  }
  case GLP_FEAS: {
    cout << " solution is feasible" << endl;
    break;
  }
  case GLP_INFEAS: {
    cout << " solution is infeasible" << endl;
    break;
  }
  case GLP_NOFEAS: {
    cout << " no feasible solution exists" << endl;
    break;
  }
  case GLP_OPT: {
    cout << " solution is optimal" << endl;
    break;
  }
  case GLP_UNBND: {
    cout << " solution is unbounded" << endl;
    break;
  }
  default: {
    cout << " !!! unknown status value: " << status << " !!!" << endl;
    break;
  }
  }
}


void printLPXReturnCode(int returnCode) {
  cout << "   LPX return code is: ";
  switch(returnCode) {
  case LPX_E_OK: {
    cout << "success" << endl;
    break;
  }
  case LPX_E_EMPTY: {
    cout << "empty problem" << endl;
    break;
  }
  case LPX_E_BADB: {
    cout << "invalid initial basis" << endl;
    break;
  }
  case LPX_E_INFEAS: {
    cout << "infeasible initial solution" << endl;
    break;
  }
  case LPX_E_FAULT: {
    cout << "unable to start the search" << endl;
    break;
  }
  case LPX_E_OBJLL: {
    cout << "objective lower limit reached" << endl;
    break;
  }
  case LPX_E_OBJUL: {
    cout << "objective upper limit reached" << endl;
    break;
  }
  case LPX_E_ITLIM: {
    cout << "iterations limit exhausted" << endl;
    break;
  }
  case LPX_E_TMLIM: {
    cout << "time limit exhausted" << endl;
    break;
  }
  case LPX_E_NOFEAS: {
    cout << "no feasible solution" << endl;
    break;
  }
  case LPX_E_INSTAB: {
    cout << "numerical instability" << endl;
    break;
  }
  case LPX_E_SING: {
    cout << "problems with basis matrix" << endl;
    break;
  }
  case LPX_E_NOCONV: {
    cout << "no convergence (interior)" << endl;
    break;
  }
  case LPX_E_NOPFS: {
    cout << "no primal feas. sol. (LP presolver)" << endl;
    break;
  }
  case LPX_E_NODFS: {
    cout << "no dual feas. sol. (LP presolver)" << endl;
    break;
  }
  case LPX_E_MIPGAP: {
    cout << "relative mip gap tolerance reached" << endl;
    break;
  }
  default: {
    cout << "!!! unknown return code " << returnCode << " !!!" << endl;
  }
  }
}
