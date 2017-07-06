
#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

class HIPUserVariables : public AlignmentUserVariables {

  public:

  /** constructor */
  HIPUserVariables(int npar) :
    jtvj(npar,0) , 
    jtve(npar,0) ,
    alichi2(0.0),
    alindof(0),
    nhit(0),
    alipar(npar,0),
    alierr(npar,0)
    //iterpar(maxiter,npar,0),
    //iterpos(maxiter,3,0),
    //iterrot(maxiter,9,0),
    //iterrpos(maxiter,3,0),
    //iterrrot(maxiter,9,0),
    //niter(0)  
  {}

  /** destructor */
  virtual ~HIPUserVariables() {};

  /** data members */

  //static const int maxiter = 9;

  AlgebraicSymMatrix jtvj;
  AlgebraicVector jtve;
  double alichi2;
  int alindof;
  int nhit;
  AlgebraicVector alipar;
  AlgebraicVector alierr;
  //AlgebraicMatrix iterpar;
  //AlgebraicMatrix iterpos,iterrot;
  //AlgebraicMatrix iterrpos,iterrrot;
  //int niter;

 /** clone method (copy constructor) */
  HIPUserVariables* clone(void) const { 
    return new HIPUserVariables(*this);
  }

};
