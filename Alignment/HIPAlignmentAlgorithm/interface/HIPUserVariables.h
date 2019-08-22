#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class HIPUserVariables : public AlignmentUserVariables {
public:
  /** data members */
  AlgebraicSymMatrix jtvj;
  AlgebraicVector jtve;
  double alichi2;
  int alindof;
  int nhit;
  int datatype;
  AlgebraicVector alipar;
  AlgebraicVector alierr;

  /** constructors */
  HIPUserVariables(int npar)
      : jtvj(npar, 0),
        jtve(npar, 0),
        alichi2(0.0),
        alindof(0),
        nhit(0),
        datatype(-2),
        alipar(npar, 0),
        alierr(npar, 0) {}

  HIPUserVariables(const HIPUserVariables& other)
      : jtvj(other.jtvj),
        jtve(other.jtve),
        alichi2(other.alichi2),
        alindof(other.alindof),
        nhit(other.nhit),
        datatype(other.datatype),
        alipar(other.alipar),
        alierr(other.alierr) {}

  /** destructor */
  ~HIPUserVariables() override{};

  /** clone method (copy constructor) */
  HIPUserVariables* clone(void) const override { return new HIPUserVariables(*this); }
};
