/** DTLPPatternReco Algorithm
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/09/03 13:30:39 $
 * $Revision: 1.7 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

#include "RecoLocalMuon/DTSegment/src/DTLPPatternRecoAlgorithm.h"

#include <list>
#include <iostream>
#include <set>
#include <cmath>

// Linear Programming Header
extern "C" {
#include "/afs/cern.ch/user/e/ebusseti/public/glpk/src/glpapi.h"
#include "glpk.h"
}

using namespace std;

/*This is the function called by the MIP solver (branch-and-cut
  method) each time a branching is needed.  This custom implementation
  forces the algo to branch first on lambdas.
 */
void callback_func(glp_tree *tree, void * info){
  if(glp_ios_reason(tree) == GLP_IBRANCH){//reason for which the function is called
    glp_prob * lp = glp_ios_get_prob(tree);//we get the subproblem
    int n = glp_get_num_int(lp), j, next;//integer variables
      double beta;
      //if(debug) for(i=0; i<n; i++)
      //	printf("Value of column %d is %f\n", i , lpx_get_col_prim(lp, i+1) );
      for (j = 0; j <= n/2; j++){
	if (glp_ios_can_branch(tree, j*2 + 2)){//even ones are lambdas
	  beta = lpx_get_col_prim(lp, j*2 + 2);
	  if (beta - floor(beta) < ceil(beta) - beta)//decide on which branch keep on searchingk
	    next = GLP_DN_BRNCH;
	  else
	    next = GLP_UP_BRNCH;
	  glp_ios_branch_upon(tree, j*2 + 2, next);//actual branching
	  break;
	}
      }
    }
  }

bool lpAlgorithm(lpAlgo::ResultLPAlgo& theAlgoResults,
		 const vector<double>& pz,
		 const vector<double>& px,
		 const vector<double>& pex,
		 const vector<int>& layers,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double BIG_M, const double theDeltaFactor)
{
  bool debug =true ;

  /*I want to know how many different layers we have */
  set<int> layerSet(layers.begin(), layers.end());
  vector<int> layerSetVector(layerSet.begin(), layerSet.end());
  const unsigned int n_layers = layerSet.size();//how many different layers we have
  if(debug) cout << "There are hits in " << n_layers << " different(s) layer(s)" << endl;
  
  const unsigned int n_points = pz.size();//how many hits we have
  if (n_points <= 4) return false;//too few hits, we won't reconstruct anything
  glp_prob *lp = 0;  //the data struct representing the problem
  vector<int> ia, ja; //the vectors that will contain indices over rows and columns to load in the problem matrix
  vector<double> ar; //the vector that will contain coefficients to load in the problem matrix

  lp = glp_create_prob();
  
  glp_set_prob_name(lp, "trackFit");
  glp_set_obj_dir(lp, GLP_MIN);//setting a minimization problem
  
  if(debug)
    cout << "[lpAlgorithm] LP-MIP Problem Created, # points = " << pz.size()<< " "  << px.size() << " "  << pex.size() << endl
	 << "              m_min= " << m_min <<  " m_max= " << m_max
	 << " q_min= " << q_min <<  " q_max= " << q_max << endl
	 << "              big_m= " << BIG_M << " delta_fact = " << theDeltaFactor << endl;

  vector<double> maxAllowedTolerances;
  for (unsigned int i=0; i < n_points; ++i) maxAllowedTolerances.push_back(pex[i] * theDeltaFactor);

  //experiment of 9/07
  /*for (unsigned int i=0; i < n_points; ++i){
    if(i > 0 && i < n_points && layers[i] == layers[i-1] && layers[i] == layers[i+1])
      maxAllowedTolerances.push_back( min(pex[i] * theDeltaFactor, min( abs(px[i]-px[i+1]) , abs(px[i]-px[i-i]) ) ) );
    else if(i > 0 && layers[i] == layers[i-1])
      maxAllowedTolerances.push_back( min(pex[i] * theDeltaFactor, abs(px[i]-px[i-1])) );
    else if(i < n_points && layers[i] == layers[i+1])
      maxAllowedTolerances.push_back( min(pex[i] * theDeltaFactor, abs(px[i+1]-px[i])));
    cout << "max allowed tol. " << i << "has been set to " << maxAllowedTolerances.back() << endl;
    }*/
  
    

  /*******************COLUMNS DEFINITION********************************
   *                                                                     
   *Columns are the structural variables for each point there is
   *sigma, for each pair we have lambda and omega. Then we have the m
   *and q (m+, m- ...)*/

  const int n_cols = 2 * n_points + 4;
  glp_add_cols(lp, n_cols);
  // set columns boundaries
  glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0); // m+ >=0
  glp_set_col_bnds(lp, 2, GLP_LO, 0.0, 0.0); // m- >=0
  glp_set_col_bnds(lp, 3, GLP_LO, 0.0, 0.0); // q+ >=0
  glp_set_col_bnds(lp, 4, GLP_LO, 0.0, 0.0); // m- >=0

  for (unsigned int i=0; i<n_points; i++) {
    glp_set_col_bnds(lp, 4 + i + 1, GLP_DB, 0.0, maxAllowedTolerances[i]);// 0 <= sigmas <= Di
    glp_set_col_bnds(lp, n_points + 4 + i + 1, GLP_DB, 0., 1.);   // 0 <= lambdas and omegas <= 1 
    if(i%2 && px[i-1] == px[i]) glp_set_col_bnds(lp, n_points + 4 + i + 1, GLP_DB, 1., 1.);//adding rule to fix omega == 0 if the pair consists of two coincident hits: FIXME not tested much
    glp_set_col_kind(lp, n_points + 4 + i + 1, GLP_IV);    // lambdas and omegas are integer (binary)
  }     
  
  if(debug) cout << "Columns defined" << endl;

  /*******************ROWS DEFINITION*****************************
   *                                                             		
   * Rows are auxiliary variables (those not appearing in the objective func)
   * 4 are needed by the inequalities for each point (see the report in documentation)
   * n_layers are the constraints over each layers, i.e. only one at most can be used 
   * 2 are m and q*/
  
  const int n_rows = 3 * n_points + n_layers +2; 
  glp_add_rows(lp, n_rows);
  
  //set rows boundaries has been moved below, with the matrix

  /* Constraints on m and q */
  glp_set_row_bnds(lp, n_rows-1, GLP_DB, m_min, m_max);// m_min <= m <= m_max
  glp_set_row_bnds(lp, n_rows, GLP_DB, q_min, q_max);// q_min <= q <= q_max
			
  if(debug) cout << "Rows defined" << endl;
  
  /******************OBJECTIVE_FUNCTION*****************************
  * Set the objective function coefficients: first four are 0 because
  * the m+ m- q+ q- do not appear in the objective function, than we
  * have 1/Delta for sigmas, M for lambdas (to maximize number of
  * used hits, and 0 again for omegas*/
  glp_set_obj_coef(lp, 1, 0.);//m+
  glp_set_obj_coef(lp, 2, 0.);//m-
  glp_set_obj_coef(lp, 3, 0.);//q+
  glp_set_obj_coef(lp, 4, 0.);//q-
 
  for (unsigned int i=0; i<n_points; i++){//set sigmas and lambdas
    glp_set_obj_coef(lp, i+5, 1./maxAllowedTolerances[i]); // sigma_i / Delta_i
    if (i%2 == 0) glp_set_obj_coef(lp, n_points + 5 + i,  BIG_M );//coefficient for lambdas
    else glp_set_obj_coef(lp, n_points + 5 + i,  0. );//coeff for omegas
  }
  
  if(debug) cout << "Objective funct. defined" << endl;
  
  /******************SETTING_MATRIX_COEFFICIENTS********************************
   * A good thing to know is that only non-zero elements need to be
   * passed to the loader (we can safely skip the zeros) ia indexes
   * rows, ja columns, and ar contains the coefficient for given row and
   * col. Then we load it with a single funct*/
  
  double bigM=BIG_M; //redefined so it's simpler to change value of M to experiment different behaviours 

  for(unsigned int i=0; i < n_points; i++) {
    
    /* Definition of first equation(row): m * Zi + q - sigma_i -
       lambda_i * M */
    ia.push_back(i + 1), ja.push_back(1), ar.push_back(pz[i]);//m+ * Zi
    ia.push_back(i + 1), ja.push_back(2), ar.push_back(-pz[i]);//m- * -Zi
    ia.push_back(i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
    ia.push_back(i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
    ia.push_back(i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
    ia.push_back(i + 1), ja.push_back(n_points + 4 + (i - i%2) + 1), ar.push_back(-bigM);// -M * lambda_i
    
    if(i%2 == 0){//left hit branch      
      ia.push_back(i + 1), ja.push_back(n_points + 4 + (i - i%2)  + 2), ar.push_back(- (px[i+1] - px[i]));// -(X_i+1 - Xi) * omega_i
      glp_set_row_bnds(lp, i + 1, GLP_UP, 0., px[i]); //a1 <= Xi
    }
    else{//right hit branch
      glp_set_row_bnds(lp, i + 1, GLP_UP, 0., px[i]);//a1 <= Xi
    }

    /*second equation: m * Zi + q + sigma_i + lambda_i * M */
    ia.push_back(n_points + i + 1), ja.push_back(1), ar.push_back(pz[i]);//m+ * Zi
    ia.push_back(n_points + i + 1), ja.push_back(2), ar.push_back(-pz[i]);//m- * -Zi
    ia.push_back(n_points + i + 1), ja.push_back(3), ar.push_back(1);//+1 * q+
    ia.push_back(n_points + i + 1), ja.push_back(4), ar.push_back(-1);//-1 * q-
    ia.push_back(n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(1);// +1 * sigma_i
    ia.push_back(n_points + i + 1), ja.push_back(n_points + 4 + (i - i%2) + 1), ar.push_back(bigM);// M * lambda_i
    if(i%2==0){//left hit branch
      glp_set_row_bnds(lp, n_points + i + 1, GLP_LO, px[i], 0.); //a2 >= Xi
    }
    else{//right hit branch
      ia.push_back(n_points + i + 1), ja.push_back(n_points + 4 + (i - i%2) + 2), ar.push_back(px[i-1] - px[i]);// -L * omega_i
      glp_set_row_bnds(lp, n_points + i + 1, GLP_LO, px[i-1], 0.); //a2 >= Xi - L
    }

    /* Third equation:  omega_i * Di - sigma_i */
    if(i%2 ==0){//left hit branch
      ia.push_back(2 * n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(-1);// -1 * sigma_i
      ia.push_back(2 * n_points + i + 1), ja.push_back(n_points + 4 + (i - i%2) + 2), ar.push_back(maxAllowedTolerances[i]);      //omega_i * Di
      glp_set_row_bnds(lp, 2 * n_points + i + 1, GLP_UP, 0., 0.); //a3 <= 0
    }
    else{//right hit branch
      ia.push_back(2 * n_points + i + 1), ja.push_back(4 + i + 1), ar.push_back(1);// 1 * sigma_i
      ia.push_back(2 * n_points + i + 1), ja.push_back(n_points + 4 + (i - i%2) + 2), ar.push_back(maxAllowedTolerances[i]);//omega_i * Di
      glp_set_row_bnds(lp, 2 * n_points + i + 1, GLP_LO, maxAllowedTolerances[i], 0.); //a3 >= Di
    }

  }
  /*Exclusivity of hits from same layer */
   for (unsigned int i =0; i < n_layers; ++i){
    if (debug) cout << "Defining constraints for layer " << layerSetVector[i] << endl;
    float num_hits_this_layer = 0;
    for(unsigned int j = 0; j < n_points/2; ++j){
      if (layers[j*2] == layerSetVector[i]){
        ia.push_back(3 * n_points + i + 1), ja.push_back( n_points + 4 + j*2 + 1 ), ar.push_back(1);// +1 * lambda_j
  	if(debug) cout<<"Added lambda " << (int)j << " to the inequality." <<endl;
   	num_hits_this_layer ++;
      }
    }
    if (debug) cout << "We have stored " <<  num_hits_this_layer << " hits." << endl;
    glp_set_row_bnds(lp, 3 * n_points + i + 1, GLP_LO,num_hits_this_layer - 1, 0.);//sum(lambdas) >= n-1
  }
 
   
   /* constraint on m: m = m+ - m */
   ia.push_back(n_rows - 1), ja.push_back(1 ), ar.push_back(1);// +1 * m+
   ia.push_back(n_rows - 1), ja.push_back(2 ), ar.push_back(-1);// -1 * m-
   /* constraints on q: q = q+ - q- */
   ia.push_back(n_rows), ja.push_back(3 ), ar.push_back(1);// +1 * q+
   ia.push_back(n_rows), ja.push_back(4 ), ar.push_back(-1);// -1 * m-	

   /*Load all coefficients in the matrix*/
   glp_load_matrix(lp, ia.size(), (int*)(&ia[0]-1), (int*)(&ja[0]-1), (double*)(&ar[0]-1));
   ia.clear();
   ja.clear();
   ar.clear();
  
   if(debug) cout << "Matrix coeff. set" << endl;
  
  /******************SOLUTION*********************************
   *                                                         */
  
  // routine to save an mps file with the problem
  // glp_write_mps(lp, GLP_MPS_FILE, NULL, "mps.txt");
  
  /*Define the control parameters*/
  glp_iocp paramIocp;
  glp_init_iocp(&paramIocp);
  paramIocp.cb_func = callback_func;//use the callback func. defined above

  
  if(debug) { // set the GLPK verbosity level 
    paramIocp.msg_lev = GLP_MSG_ALL; 
  } else {
    paramIocp.msg_lev = GLP_MSG_OFF; 
  }
  paramIocp.tm_lim = 60000; // set the time limit in millisecond // FIXME: this is arbitrary
 
  paramIocp.presolve = GLP_ON;//necessary
					   							   
  lpx_order_matrix(lp);//this is really needed?
  


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
  for (unsigned int i =0; i < n_points/2; i++){
    if (!glp_mip_col_val(lp, n_points + 4 + i*2 + 1)) {
       if(debug) cout << "found a lambda = 0" << endl
		      << "its layer is: " << layers[i*2] << endl
		      << "it's the hit number " << i*2 + glp_mip_col_val(lp, n_points + 4 + i*2 + 2) << endl;
      if(!glp_mip_col_val(lp, n_points + 4 + i*2 + 2)){
	if(debug) cout << "we used a left hit" << endl;
	theAlgoResults.lambdas.push_back(0);
	theAlgoResults.lambdas.push_back(1);
	theAlgoResults.chi2Var += (glp_mip_col_val(lp, 4 + i*2 + 1) - pex[i*2])*(glp_mip_col_val(lp, 4 + i*2 + 1) - pex[i*2]) / (pex[i*2]*pex[i*2]);
      }
      else{
	if(debug) cout << "we used a right hit" << endl;
	theAlgoResults.lambdas.push_back(1);
	theAlgoResults.lambdas.push_back(0);
	theAlgoResults.chi2Var += (glp_mip_col_val(lp, 4 + i*2 + 2) - pex[i*2])*(glp_mip_col_val(lp, 4 + i*2 +2) - pex[i*2]) / (pex[i*2]*pex[i*2]);
      }
      control ++;
      
    }
    else {
      theAlgoResults.lambdas.push_back(1);
      theAlgoResults.lambdas.push_back(1);
    }
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
