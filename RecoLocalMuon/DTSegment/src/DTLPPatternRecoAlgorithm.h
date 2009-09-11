#include <vector>
#include <list>

namespace lpAlgo {
  class ResultLPAlgo{
  public:
    ResultLPAlgo(){ chi2Var =0;};
    ~ResultLPAlgo(){lambdas.clear();};
    double mVar;
    double qVar;
    double chi2Var;
    std::vector<int> lambdas;  
  };
}

bool lpAlgorithm(lpAlgo::ResultLPAlgo& theAlgoResults,
		 const std::vector<double>& pz,
		 const std::vector<double>& px,
		 const std::vector<double>& pex,
		 const std::vector<int>& layers,
		 const double m_min, const double m_max,
		 const double q_min, const double q_max,
		 const double BIG_M, const double theDeltaFactor);

void printGLPReturnCode(int returnCode);

void printGLPSolutionStatus(int status);

void printLPXReturnCode(int returnCode);
