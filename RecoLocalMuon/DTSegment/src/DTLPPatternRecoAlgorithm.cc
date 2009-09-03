/** DTLPPatternReco Algorithm
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/09/01 09:03:55 $
 * $Revision: 1.5 $
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
		 const list<double>& pzl,
		 const list<double>& pxl,
		 const list<double>& pexl,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double BIG_M, const double theDeltaFactor)
{
  bool debug =false;
  //FIXME
  vector<double> pz, px, pex;
  pz.resize(pzl.size());
  copy(pzl.begin(), pzl.end(), pz.begin());
  px.resize(pxl.size());
  copy(pxl.begin(), pxl.end(), px.begin());
  pex.resize(pexl.size());
  copy(pexl.begin(), pexl.end(), pex.begin());
  //
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
        // 0 <= sigmas <= Di
        glp_set_col_bnds(lp, 4 + i + 1, GLP_DB, 0.0, pex[i] * theDeltaFactor);
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
    
    const int n_rows = 3 * n_points + n_points/2 + 2; 
    glp_add_rows(lp, n_rows);
    //set rows boundaries
    //constraints over the first three inequalites
    
    for (unsigned int i =0; i<n_points; i++){
      glp_set_row_bnds(lp, i + 1, GLP_UP, 0., px[i]); //a1 <= Xi
      glp_set_row_bnds(lp, n_points + i + 1, GLP_LO, px[i], 0.); //a2 >= Xi
      glp_set_row_bnds(lp, 2 * n_points + i + 1, GLP_UP, 0., 0.); //a4 <= 0
    }
    // Constraints on lambda pairs
    for (unsigned int i=0; i<n_points; i++) if (i%2==0) glp_set_row_bnds(lp, 3 * n_points + i/2 + 1, GLP_LO, 1., 0.);			
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
    // than we have the coefficients for lambda: M (to maximize number of points, 
    //because we are minimizing the objective function)
    for (unsigned int i=0; i<n_points; i++){
      glp_set_obj_coef(lp, i+5, 1./( theDeltaFactor * pex[i])); // sigma_i / Delta_i
      glp_set_obj_coef(lp, n_points + 5 + i,  BIG_M );//coefficient for lambdas
    }
    
    if(debug) cout << "Objective funct. defined" << endl;

    /******************SETTING MATRIX COEFFICIENTS*********************************
     *                                                                            */
    // a good thing to know is that only non-zero elements need to be passed to the loader (we can safely skip the zeros)
    //ia indexes rows, ja columns ar contains the coefficient for given row and col.
    for(unsigned int i=0; i < n_points; i++)
    {
      //first equation(row): m * Zi + q - sigma_i - lambda_i * M
      ia.push_back(i + 1), ja.push_back(1), ar.push_back(pz[i]);//m+ * Zi
      ia.push_back(i + 1), ja.push_back(2), ar.push_back(-pz[i]);//m- * -Zi
      ia.push_back(i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
      ia.push_back(i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
      ia.push_back(i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
      ia.push_back(i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(-BIG_M);// -M * lambda_i       
        
      //second equation: m * Zi + q + sigma_i + lambda_i * M
      ia.push_back(n_points + i + 1), ja.push_back(1), ar.push_back(pz[i]);//m+ * Zi
      ia.push_back(n_points + i + 1), ja.push_back(2), ar.push_back(-pz[i]);//m- * -Zi
      ia.push_back(n_points + i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
      ia.push_back(n_points + i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
      ia.push_back(n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(1);// +1 * sigma_i
      ia.push_back(n_points + i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(BIG_M);// M * lambda_i        
        
      //third equation:  lambda_i * Di - sigma_i //ENZO
      ia.push_back(2 * n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
      ia.push_back(2 * n_points + i + 1), ja.push_back(n_points + 4 + i + 1), ar.push_back(pex[i]*theDeltaFactor);// Di * lambda_i        
        
      // exclusivity of pairs
      if (i%2==0) {
	ia.push_back(3 * n_points + i/2 + 1), ja.push_back( n_points + 4 + i + 1 ), ar.push_back(1);// +1 * lambda_i
	ia.push_back(3 * n_points + i/2 + 1), ja.push_back( n_points + 4 + i + 1 + 1 ), ar.push_back(1);// +1 * lambda_i+1
            }
    }
    // constraints on m: m = m+ - m-
    ia.push_back(n_rows - 1), ja.push_back(1 ), ar.push_back(1);// +1 * m+
    ia.push_back(n_rows - 1), ja.push_back(2 ), ar.push_back(-1);// -1 * m-
    // constraints on q: q = q+ - q-
    ia.push_back(n_rows), ja.push_back(3 ), ar.push_back(1);// +1 * q+
    ia.push_back(n_rows), ja.push_back(4 ), ar.push_back(-1);// -1 * m-	

    glp_load_matrix(lp, ia.size(), (int*)(&ia[0]-1), (int*)(&ja[0]-1), (double*)(&ar[0]-1));
    ia.clear();
    ja.clear();
    ar.clear();
    
    if(debug) cout << "Matrix coeff. set" << endl;

    /******************SOLUTION*********************************
     *                                                         */

    // routine to save an mps file with the problem
    // glp_write_mps(lp, GLP_MPS_FILE, NULL, "mps.txt");
    
    // define the control parameters
    glp_iocp paramIocp;
    glp_smcp paramSmplx;
    glp_init_iocp(&paramIocp);
    glp_init_smcp(&paramSmplx);

    if(debug) { // set the GLPK verbosity level 
      paramIocp.msg_lev = GLP_MSG_ALL; 
      paramSmplx.msg_lev = GLP_MSG_ALL; 
    } else {
      paramIocp.msg_lev = GLP_MSG_OFF; 
      paramSmplx.msg_lev = GLP_MSG_OFF; 
    }
    //    paramSmplx.it_lim = 1000; // set the max # of iterations // FIXME: this is arbitrary
    paramIocp.tm_lim = 60000; // set the max # of iterations // FIXME: this is arbitrary
    // try optimization
    // paramSmplx.presolve = GLP_ON;
    // paramSmplx.meth = GLP_DUALP;
    paramIocp.presolve = GLP_ON;
    paramIocp.br_tech = GLP_BR_MFV;
//     paramIocp.gmi_cuts = GLP_ON;
//      paramIocp.clq_cuts = GLP_ON;
//    paramIocp.cov_cuts = GLP_ON;


    //experimental
    // _glp_order_matrix(lp);

    int retGlpIntopt = glp_intopt(lp, &paramIocp);
    if(retGlpIntopt != 0) {
      cout << "[lpAlgorithm]***Warning: glp_intopt return code " << endl;
      if(debug) printGLPReturnCode(retGlpIntopt);
      glp_delete_prob(lp);
      return false;
    }

    // check the status of the MIP solution 
    int statusMIPSol = glp_mip_status(lp);
    if(statusMIPSol != GLP_OPT) {
      cout << "[lpAlgorithm]***Warning: wrong MIP solution status" << endl;
      printGLPSolutionStatus(statusMIPSol);
      glp_delete_prob(lp);
      return false;
    }
    //I must return the m and q values found, and all values af lambdas (to know which point has been used)
    theAlgoResults.mVar = glp_mip_col_val(lp, 1) - glp_mip_col_val(lp, 2);//push back m
    theAlgoResults.qVar =  glp_mip_col_val(lp, 3) - glp_mip_col_val(lp, 4);//push back q
    theAlgoResults.lambdas.clear();
    int control = 0;
    for (unsigned int i =0; i < n_points; i++){
      // theAlgoResults.lambdas.push_back( lpx_mip_col_val(lp, n_points + 4 + i + 1) );//push back lambdas
      if (!glp_mip_col_val(lp, n_points + 4 + i + 1)) {
// 	if(debug) cout << "found a lambda = 0" << endl;
	theAlgoResults.lambdas.push_back(0);
	control ++;
	theAlgoResults.chi2Var += (glp_mip_col_val(lp, 4 + i + 1) - pex[i])*(glp_mip_col_val(lp, 4 + i + 1) - pex[i]) / (pex[i]*pex[i]);
      }
      else theAlgoResults.lambdas.push_back(1);
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
