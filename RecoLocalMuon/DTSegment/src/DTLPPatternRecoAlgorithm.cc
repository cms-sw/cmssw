/** DTLPPatternReco Algorithm
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/18 11:41:52 $
 * $Revision: 1.2 $
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

bool lpAlgorithm(DTLPPatternReco::ResultLPAlgo& theAlgoResults,
		 const std::list<double>& pz,
		 const std::list<double>& px,
		 const std::list<double>& pex,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double BIG_M, const double theDeltaFactor)
    {
    const unsigned int n_points = pz.size();
    if (n_points <= 4) return false;
    //the data struct representing the problem
    glp_prob *lp;
    //the vectors that will contain indices over rows and columns to load in the problem matrix
    std::vector<int> ia, ja;
    //the vector that will contain coefficients to load in the problem matrix
    std::vector<double> ar;
    lp = glp_create_prob();

    // glp_set_class(lp, GLP_MIP);//mixed integer prog.
    glp_set_prob_name(lp, "trackFit");
    //setting a minimization problem
    glp_set_obj_dir(lp, GLP_MIN);
    
    std::cout << "Problem Created, n_points = " << pz.size()<< " "  << px.size() << " "  << pex.size() << std::endl;	

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
     
    std::cout << "Columns defined" << std::endl;
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
	for (std::list<double>::const_iterator it = px.begin(); it != px.end(); it++) {
	  glp_set_row_bnds(lp, i + 1, GLP_UP, 0., *it); //a1 <= Xi
	  glp_set_row_bnds(lp, n_points + i + 1, GLP_LO, *it, 0.); //a2 >= Xi
	  i++;
	}
	i =0;
	for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); it++) {
        glp_set_row_bnds(lp, 2 * n_points + i + 1, GLP_UP, 0., (*it) * theDeltaFactor ); //a3 <= Di
        glp_set_row_bnds(lp, 3 * n_points + i + 1, GLP_UP, 0., BIG_M - ((*it) * theDeltaFactor)); //a4 <= M-Di
	i ++;
	}
        // Constraints on lambda pairs
	for (unsigned int i=0; i<n_points; i++) if (i%2==0) glp_set_row_bnds(lp, 4 * n_points + i/2 + 1, GLP_LO, 1., 0.);			
	// Constraints on m and q
	glp_set_row_bnds(lp, n_rows-1, GLP_DB, m_min, m_max);// m_min <= m <= m_max
	glp_set_row_bnds(lp, n_rows, GLP_DB, q_min, q_max);// q_min <= q <= q_max
    
	std::cout << "Rows defined" << std::endl;
	
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
    for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); it++){
      glp_set_obj_coef(lp, i+5, 1./( theDeltaFactor * (*it))); // sigma_i / Delta_i
      i++;
    }
    // than we have the coefficients for lambda: M (to maximize number of points, 
    //because we are minimizing the objective function)
    for (unsigned int i=0; i<n_points; i++) glp_set_obj_coef(lp, n_points + 5 + i,  BIG_M );
    
    std::cout << "Objective funct. defined" << std::endl;

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
    ia.push_back(4 * n_points + n_points/2 + 2), ja.push_back(3 ), ar.push_back(1);// +1 * q+
    ia.push_back(4 * n_points + n_points/2 + 2), ja.push_back(4 ), ar.push_back(-1);// -1 * m-	

    glp_load_matrix(lp, ia.size(), (int*)(&ia[0]-1), (int*)(&ja[0]-1), (double*)(&ar[0]-1));
    ia.clear();
    ja.clear();
    ar.clear();
    
    std::cout << "Matrix coeff. set" << std::endl;

    /******************SOLUTION*********************************
     *                                                         */
    glp_iocp parm;
    glp_smcp parm1;
    glp_init_iocp(&parm);
    glp_init_smcp(&parm1);
    parm.msg_lev = GLP_MSG_ALL; 
    parm1.msg_lev = GLP_MSG_ALL;
    //if(!glp_simplex(lp, &parm1)) return false;
    glp_simplex(lp, &parm1);
    std::cout << "simplex returned" <<std::endl;
    //if(!glp_intopt(lp, &parm)) return false;
    glp_intopt(lp, &parm);
    std::cout << "intopt returned" <<std::endl;
    //glp_integer(lp);    
    //lpx_intopt(lp);
    //for (int i=0; i<n_cols; i++) printf("%d --> %f\n",i,lpx_mip_col_val(lp, i+1));
    //I must return the m and q values found, and all values af lambdas (to know which point has been used)
    theAlgoResults.mVar = glp_mip_col_val(lp, 1) - glp_mip_col_val(lp, 2);//push back m
    theAlgoResults.qVar =  glp_mip_col_val(lp, 3) - glp_mip_col_val(lp, 4);//push back q
    theAlgoResults.lambdas.clear();
    i = 0;
    int control = 0;
    for (std::list<double>::const_iterator it = pex.begin(); it != pex.end(); ++it){
      // theAlgoResults.lambdas.push_back( lpx_mip_col_val(lp, n_points + 4 + i + 1) );//push back lambdas
      if (!glp_mip_col_val(lp, n_points + 4 + i + 1)) {
	std::cout << "found a lambda = 0" << std::endl;
	theAlgoResults.lambdas.push_back(0);
	control ++;
	theAlgoResults.chi2Var += ((glp_mip_col_val(lp, 4 + i + 1) - *it))*((glp_mip_col_val(lp, 4 + i + 1) - *it)) / ((*it) * (*it));
      }
      else theAlgoResults.lambdas.push_back(1);
      i++;
    }
    if (control < 3) {glp_delete_prob(lp);  return false;}
     std::cout << "We have used "<< control << " hits"  <<std::endl;
    //printf("y =  %f * x  + %f\n", res[0], res[1]);
    //printf("obj func = %f\n",lpx_mip_obj_val(lp));
    glp_delete_prob(lp);


    std::cout << "DTLPPatternReco::perform_fit : m = " << theAlgoResults.mVar << " q = " << theAlgoResults.qVar << std::endl;
    return true;
}



