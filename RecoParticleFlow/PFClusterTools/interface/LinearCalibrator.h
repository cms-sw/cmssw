#ifndef LINEARCALIBRATOR_HH_
#define LINEARCALIBRATOR_HH_

#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"


#include "TMatrixD.h"
#include "TVectorD.h"

/**
 * \class LinearCalibrator Class
 * 
 * \brief This class implements the simple "linear" calibration for the "a,b,c" coefficients.
 * It extends Calibrator.
 * 
 * Calibrations are given i.t.o,
 * 		E_calib = a + b * det_1 + c * det_2 + ...
 * 
 * 
 \author Jamie Ballin
 \date   April 2008
 */
namespace pftools {
class LinearCalibrator : public Calibrator {
public:
	LinearCalibrator();
	virtual ~LinearCalibrator();


	/*
	 * Note: covariant return type w.r.t. Calibrator class: the overloading has changed 
	 * the return type but this IS allowed with modern compilers.
	 * See documentation in Calibrator.h
	 */
	LinearCalibrator* clone() const;
	LinearCalibrator* create() const;

protected:
	
	virtual std::map<DetectorElementPtr, double>
			getCalibrationCoefficientsCore() throw(PFToolsException&);


	LinearCalibrator(const LinearCalibrator& lc);
	/*
	 * Converts the particle deposits into a useful matrix formulation.
	 */
	virtual void initEijMatrix(TMatrixD& eij, TVectorD& truthE);

	/*
	 * Utility method to extract the unique number of detected elements.
	 */
	virtual void populateDetElIndex();

	virtual TVectorD& getProjections(const TMatrixD& eij, TVectorD& proj,
			const TVectorD& truthE) const;

	virtual TMatrixD& getHessian(const TMatrixD& eij, TMatrixD& hess,
			const TVectorD& truthE) const;

	/*
	 * Map to convert detector element to array row/column index.
	 */
	std::map<DetectorElementPtr, unsigned> myDetElIndex;

};
}

#endif /*LINEARCALIBRATOR_HH_*/
