#ifndef PhysicsTools_MVATrainer_LeastSquares_h
#define PhysicsTools_MVATrainer_LeastSquares_h

#include <string>
#include <vector>

#include <TMatrixD.h>
#include <TVectorD.h>

#include <xercesc/dom/DOM.hpp>

namespace PhysicsTools {

class LeastSquares
{
    public:
	LeastSquares(unsigned int n);
	virtual ~LeastSquares();

	void add(const std::vector<double> &values, double dest,
	         double weight = 1.0);
	void add(const LeastSquares &other, double weight = 1.0);
	void calculate();

	std::vector<double> getWeights() const;
	std::vector<double> getMeans() const;
	double getConstant() const;

	inline unsigned int getSize() const { return n; }
	inline const TMatrixDSym &getCoefficients() const { return coeffs; }
	inline const TMatrixDSym &getCovariance() const { return covar; }
	inline const TMatrixDSym &getCorrelations() const { return corr; }
	inline const TMatrixD &getRotation() { return rotation; }

	void load(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem);
	XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *save(
		XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *doc) const;

	static TVectorD solveFisher(const TMatrixDSym &coeffs);
	static TMatrixD solveRotation(const TMatrixDSym &covar,
	                              TVectorD &trace);

    private:
	TMatrixDSym		coeffs;
	TMatrixDSym		covar;
	TMatrixDSym		corr;
	TMatrixD		rotation;
	TVectorD		weights;
	TVectorD		variance;
	TVectorD		trace;
	const unsigned int	n;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_LeastSquares_h
