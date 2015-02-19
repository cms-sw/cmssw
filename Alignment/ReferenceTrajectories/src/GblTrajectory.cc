/*
 * GblTrajectory.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 *    Revision: 113
 */

/** \file GBL general information
 *
 *  \section intro_sec Introduction
 *
 *  For a track with an initial trajectory from a prefit of the
 *  measurements (internal seed) or an external prediction
 *  (external seed) the description of multiple scattering is
 *  added by offsets in a local system. Along the initial
 *  trajectory points are defined with can describe a measurement
 *  or a (thin) scatterer or both. Measurements are arbitrary
 *  functions of the local track parameters at a point (e.g. 2D:
 *  position, 4D: direction+position). The refit provides corrections
 *  to the local track parameters (in the local system) and the
 *  corresponding covariance matrix at any of those points.
 *  Non-diagonal covariance matrices will be diagonalized internally.
 *  Outliers can be down-weighted by use of M-estimators.
 *
 *  The broken lines trajectory is defined by (2D) offsets at the
 *  first and last point and all points with a scatterer. The
 *  prediction for a measurement is obtained by interpolation of
 *  the enclosing offsets and for triplets of adjacent offsets
 *  kink angles are determined. This requires for all points the
 *  jacobians for propagation to the previous and next offset.
 *  These are calculated from the point-to-point jacobians along
 *  the initial trajectory. The sequence of points has to be
 *  strictly monotonic in arc-length.
 *
 *  Additional local or global parameters can be added and the
 *  trajectories can be written to special binary files for
 *  calibration and alignment with Millepede-II.
 *  (V. Blobel, NIM A, 566 (2006), pp. 5-13).
 *
 *  Besides simple trajectories describing the path of a single
 *  particle composed trajectories are supported. These are
 *  constructed from the trajectories of multiple particles and
 *  some external parameters (like those describing a decay)
 *  and transformations at the first points from the external
 *  to the local track parameters.
 *
 *  The conventions for the coordinate systems follow:
 *  Derivation of Jacobians for the propagation of covariance
 *  matrices of track parameters in homogeneous magnetic fields
 *  A. Strandlie, W. Wittek, NIM A, 566 (2006) 687-698.
 *
 *  \section call_sec Calling sequence
 *
 *    -# Create list of points on initial trajectory:\n
 *            <tt>std::vector<GblPoint> list</tt>
 *    -# For all points on initial trajectory:
 *        - Create points and add appropriate attributes:\n
 *           - <tt>point = gbl::GblPoint(..)</tt>
 *           - <tt>point.addMeasurement(..)</tt>
 *           - Add additional local or global parameters to measurement:\n
 *             - <tt>point.addLocals(..)</tt>
 *             - <tt>point.addGlobals(..)</tt>
 *           - <tt>point.addScatterer(..)</tt>
 *        - Add point (ordered by arc length) to list:\n
 *            <tt>list.push_back(point)</tt>
 *    -# Create (simple) trajectory from list of points:\n
 *            <tt>traj = gbl::GblTrajectory (list)</tt>
 *    -# Optionally with external seed:\n
 *            <tt>traj = gbl::GblTrajectory (list,seed)</tt>
 *    -# Optionally check validity of trajectory:\n
 *            <tt>if (not traj.isValid()) .. //abort</tt>
 *    -# Fit trajectory, return error code,
 *       get Chi2, Ndf (and weight lost by M-estimators):\n
 *            <tt>ierr = traj.fit(..)</tt>
 *    -# For any point on initial trajectory:
 *        - Get corrections and covariance matrix for track parameters:\n
 *            <tt>[..] = traj.getResults(label)</tt>
 *    -# Optionally write trajectory to MP binary file (doesn't needs to be fitted):\n
 *            <tt>traj.milleOut(..)</tt>
 *
 *  \section loc_sec Local system and local parameters
 *  At each point on the trajectory a local coordinate system with local track
 *  parameters has to be defined. The first of the five parameters describes
 *  the bending, the next two the direction and the last two the position (offsets).
 *  The curvilinear system (T,U,V) with parameters (q/p, lambda, phi, x_t, y_t)
 *  is well suited.
 *
 *  \section impl_sec Implementation
 *
 *  Matrices are implemented with ROOT (root.cern.ch). User input or output is in the
 *  form of TMatrices. Internally SMatrices are used for fixes sized and simple matrices
 *  based on std::vector<> for variable sized matrices.
 *
 *  \section ref_sec References
 *    - V. Blobel, C. Kleinwort, F. Meier,
 *      Fast alignment of a complex tracking detector using advanced track models,
 *      Computer Phys. Communications (2011), doi:10.1016/j.cpc.2011.03.017
 *    - C. Kleinwort, General Broken Lines as advanced track fitting method,
 *      NIM A, 673 (2012), 107-110, doi:10.1016/j.nima.2012.01.024
 */

#include "Alignment/ReferenceTrajectories/interface/GblTrajectory.h"

//! Namespace for the general broken lines package
namespace gbl {

/// Create new (simple) trajectory from list of points.
/**
 * Curved trajectory in space (default) or without curvature (q/p) or in one
 * plane (u-direction) only.
 * \param [in] aPointList List of points
 * \param [in] flagCurv Use q/p
 * \param [in] flagU1dir Use in u1 direction
 * \param [in] flagU2dir Use in u2 direction
 */
GblTrajectory::GblTrajectory(const std::vector<GblPoint> &aPointList,
		bool flagCurv, bool flagU1dir, bool flagU2dir) :
		numAllPoints(aPointList.size()), numPoints(), numOffsets(0), numInnerTrans(
				0), numCurvature(flagCurv ? 1 : 0), numParameters(0), numLocals(
				0), numMeasurements(0), externalPoint(0), theDimension(0), thePoints(), theData(), measDataIndex(), scatDataIndex(), externalSeed(), innerTransformations(), externalDerivatives(), externalMeasurements(), externalPrecisions() {

	if (flagU1dir)
		theDimension.push_back(0);
	if (flagU2dir)
		theDimension.push_back(1);
	// simple (single) trajectory
	thePoints.push_back(aPointList);
	numPoints.push_back(numAllPoints);
	construct(); // construct trajectory
}

/// Create new (simple) trajectory from list of points with external seed.
/**
 * Curved trajectory in space (default) or without curvature (q/p) or in one
 * plane (u-direction) only.
 * \param [in] aPointList List of points
 * \param [in] aLabel (Signed) label of point for external seed
 * (<0: in front, >0: after point, slope changes at scatterer!)
 * \param [in] aSeed Precision matrix of external seed
 * \param [in] flagCurv Use q/p
 * \param [in] flagU1dir Use in u1 direction
 * \param [in] flagU2dir Use in u2 direction
 */
GblTrajectory::GblTrajectory(const std::vector<GblPoint> &aPointList,
		unsigned int aLabel, const TMatrixDSym &aSeed, bool flagCurv,
		bool flagU1dir, bool flagU2dir) :
		numAllPoints(aPointList.size()), numPoints(), numOffsets(0), numInnerTrans(
				0), numCurvature(flagCurv ? 1 : 0), numParameters(0), numLocals(
				0), numMeasurements(0), externalPoint(aLabel), theDimension(0), thePoints(), theData(), measDataIndex(), scatDataIndex(), externalSeed(
				aSeed), innerTransformations(), externalDerivatives(), externalMeasurements(), externalPrecisions() {

	if (flagU1dir)
		theDimension.push_back(0);
	if (flagU2dir)
		theDimension.push_back(1);
	// simple (single) trajectory
	thePoints.push_back(aPointList);
	numPoints.push_back(numAllPoints);
	construct(); // construct trajectory
}

/// Create new composed trajectory from list of points and transformations.
/**
 * Composed of curved trajectory in space.
 * \param [in] aPointsAndTransList List containing pairs with list of points and transformation (at inner (first) point)
 */
GblTrajectory::GblTrajectory(
		const std::vector<std::pair<std::vector<GblPoint>, TMatrixD> > &aPointsAndTransList) :
		numAllPoints(), numPoints(), numOffsets(0), numInnerTrans(
				aPointsAndTransList.size()), numParameters(0), numLocals(0), numMeasurements(
				0), externalPoint(0), theDimension(0), thePoints(), theData(), measDataIndex(), scatDataIndex(), externalSeed(), innerTransformations(), externalDerivatives(), externalMeasurements(), externalPrecisions() {

	for (unsigned int iTraj = 0; iTraj < aPointsAndTransList.size(); ++iTraj) {
		thePoints.push_back(aPointsAndTransList[iTraj].first);
		numPoints.push_back(thePoints.back().size());
		numAllPoints += numPoints.back();
		innerTransformations.push_back(aPointsAndTransList[iTraj].second);
	}
	theDimension.push_back(0);
	theDimension.push_back(1);
	numCurvature = innerTransformations[0].GetNcols();
	construct(); // construct (composed) trajectory
}

/// Create new composed trajectory from list of points and transformations with (independent) external measurements.
/**
 * Composed of curved trajectory in space.
 * \param [in] aPointsAndTransList List containing pairs with list of points and transformation (at inner (first) point)
 * \param [in] extDerivatives Derivatives of external measurements vs external parameters
 * \param [in] extMeasurements External measurements (residuals)
 * \param [in] extPrecisions Precision of external measurements
 */
GblTrajectory::GblTrajectory(
		const std::vector<std::pair<std::vector<GblPoint>, TMatrixD> > &aPointsAndTransList,
		const TMatrixD &extDerivatives, const TVectorD &extMeasurements,
		const TVectorD &extPrecisions) :
		numAllPoints(), numPoints(), numOffsets(0), numInnerTrans(
				aPointsAndTransList.size()), numParameters(0), numLocals(0), numMeasurements(
				0), externalPoint(0), theDimension(0), thePoints(), theData(), measDataIndex(), scatDataIndex(), externalSeed(), innerTransformations(), externalDerivatives(
				extDerivatives), externalMeasurements(extMeasurements), externalPrecisions(
				extPrecisions) {

	for (unsigned int iTraj = 0; iTraj < aPointsAndTransList.size(); ++iTraj) {
		thePoints.push_back(aPointsAndTransList[iTraj].first);
		numPoints.push_back(thePoints.back().size());
		numAllPoints += numPoints.back();
		innerTransformations.push_back(aPointsAndTransList[iTraj].second);
	}
	theDimension.push_back(0);
	theDimension.push_back(1);
	numCurvature = innerTransformations[0].GetNcols();
	construct(); // construct (composed) trajectory
}

GblTrajectory::~GblTrajectory() {
}

/// Retrieve validity of trajectory
bool GblTrajectory::isValid() const {
	return constructOK;
}

/// Retrieve number of point from trajectory
unsigned int GblTrajectory::getNumPoints() const {
	return numAllPoints;
}

/// Construct trajectory from list of points.
/**
 * Trajectory is prepared for fit or output to binary file, may consists of sub-trajectories.
 */
void GblTrajectory::construct() {

	constructOK = false;
	fitOK = false;
	unsigned int aLabel = 0;
	if (numAllPoints < 2) {
		std::cout << " GblTrajectory construction failed: too few GblPoints "
				<< std::endl;
		return;
	}
	// loop over trajectories
	numTrajectories = thePoints.size();
	//std::cout << " numTrajectories: " << numTrajectories << ", " << innerTransformations.size() << std::endl;
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		std::vector<GblPoint>::iterator itPoint;
		for (itPoint = thePoints[iTraj].begin();
				itPoint < thePoints[iTraj].end(); ++itPoint) {
			numLocals = std::max(numLocals, itPoint->getNumLocals());
			numMeasurements += itPoint->hasMeasurement();
			itPoint->setLabel(++aLabel);
		}
	}
	defineOffsets();
	calcJacobians();
	try {
		prepare();
	} catch (std::overflow_error &e) {
		std::cout << " GblTrajectory construction failed: " << e.what()
				<< std::endl;
		return;
	}
	constructOK = true;
	// number of fit parameters
	numParameters = (numOffsets - 2 * numInnerTrans) * theDimension.size()
			+ numCurvature + numLocals;
}

/// Define offsets from list of points.
/**
 * Define offsets at points with scatterers and first and last point.
 * All other points need interpolation from adjacent points with offsets.
 */
void GblTrajectory::defineOffsets() {

	// loop over trajectories
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		// first point is offset
		thePoints[iTraj].front().setOffset(numOffsets++);
		// intermediate scatterers are offsets
		std::vector<GblPoint>::iterator itPoint;
		for (itPoint = thePoints[iTraj].begin() + 1;
				itPoint < thePoints[iTraj].end() - 1; ++itPoint) {
			if (itPoint->hasScatterer()) {
				itPoint->setOffset(numOffsets++);
			} else {
				itPoint->setOffset(-numOffsets);
			}
		}
		// last point is offset
		thePoints[iTraj].back().setOffset(numOffsets++);
	}
}

/// Calculate Jacobians to previous/next scatterer from point to point ones.
void GblTrajectory::calcJacobians() {

	SMatrix55 scatJacobian;
	// loop over trajectories
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		// forward propagation (all)
		GblPoint* previousPoint = &thePoints[iTraj].front();
		unsigned int numStep = 0;
		std::vector<GblPoint>::iterator itPoint;
		for (itPoint = thePoints[iTraj].begin() + 1;
				itPoint < thePoints[iTraj].end(); ++itPoint) {
			if (numStep == 0) {
				scatJacobian = itPoint->getP2pJacobian();
			} else {
				scatJacobian = itPoint->getP2pJacobian() * scatJacobian;
			}
			numStep++;
			itPoint->addPrevJacobian(scatJacobian); // iPoint -> previous scatterer
			if (itPoint->getOffset() >= 0) {
				previousPoint->addNextJacobian(scatJacobian); // lastPoint -> next scatterer
				numStep = 0;
				previousPoint = &(*itPoint);
			}
		}
		// backward propagation (without scatterers)
		for (itPoint = thePoints[iTraj].end() - 1;
				itPoint > thePoints[iTraj].begin(); --itPoint) {
			if (itPoint->getOffset() >= 0) {
				scatJacobian = itPoint->getP2pJacobian();
				continue; // skip offsets
			}
			itPoint->addNextJacobian(scatJacobian); // iPoint -> next scatterer
			scatJacobian = scatJacobian * itPoint->getP2pJacobian();
		}
	}
}

/// Get jacobian for transformation from fit to track parameters at point.
/**
 * Jacobian broken lines (q/p,..,u_i,u_i+1..) to track (q/p,u',u) parameters
 * including additional local parameters.
 * \param [in] aSignedLabel (Signed) label of point for external seed
 * (<0: in front, >0: after point, slope changes at scatterer!)
 * \return List of fit parameters with non zero derivatives and
 * corresponding transformation matrix
 */
std::pair<std::vector<unsigned int>, TMatrixD> GblTrajectory::getJacobian(
		int aSignedLabel) const {

	unsigned int nDim = theDimension.size();
	unsigned int nCurv = numCurvature;
	unsigned int nLocals = numLocals;
	unsigned int nBorder = nCurv + nLocals;
	unsigned int nParBRL = nBorder + 2 * nDim;
	unsigned int nParLoc = nLocals + 5;
	std::vector<unsigned int> anIndex;
	anIndex.reserve(nParBRL);
	TMatrixD aJacobian(nParLoc, nParBRL);
	aJacobian.Zero();

	unsigned int aLabel = abs(aSignedLabel);
	unsigned int firstLabel = 1;
	unsigned int lastLabel = 0;
	unsigned int aTrajectory = 0;
	// loop over trajectories
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		aTrajectory = iTraj;
		lastLabel += numPoints[iTraj];
		if (aLabel <= lastLabel)
			break;
		if (iTraj < numTrajectories - 1)
			firstLabel += numPoints[iTraj];
	}
	int nJacobian; // 0: prev, 1: next
	// check consistency of (index, direction)
	if (aSignedLabel > 0) {
		nJacobian = 1;
		if (aLabel >= lastLabel) {
			aLabel = lastLabel;
			nJacobian = 0;
		}
	} else {
		nJacobian = 0;
		if (aLabel <= firstLabel) {
			aLabel = firstLabel;
			nJacobian = 1;
		}
	}
	const GblPoint aPoint = thePoints[aTrajectory][aLabel - firstLabel];
	std::vector<unsigned int> labDer(5);
	SMatrix55 matDer;
	getFitToLocalJacobian(labDer, matDer, aPoint, 5, nJacobian);

	// from local parameters
	for (unsigned int i = 0; i < nLocals; ++i) {
		aJacobian(i + 5, i) = 1.0;
		anIndex.push_back(i + 1);
	}
	// from trajectory parameters
	unsigned int iCol = nLocals;
	for (unsigned int i = 0; i < 5; ++i) {
		if (labDer[i] > 0) {
			anIndex.push_back(labDer[i]);
			for (unsigned int j = 0; j < 5; ++j) {
				aJacobian(j, iCol) = matDer(j, i);
			}
			++iCol;
		}
	}
	return std::make_pair(anIndex, aJacobian);
}

/// Get (part of) jacobian for transformation from (trajectory) fit to track parameters at point.
/**
 * Jacobian broken lines (q/p,..,u_i,u_i+1..) to local (q/p,u',u) parameters.
 * \param [out] anIndex List of fit parameters with non zero derivatives
 * \param [out] aJacobian Corresponding transformation matrix
 * \param [in] aPoint Point to use
 * \param [in] measDim Dimension of 'measurement'
 * (<=2: calculate only offset part, >2: complete matrix)
 * \param [in] nJacobian Direction (0: to previous offset, 1: to next offset)
 */
void GblTrajectory::getFitToLocalJacobian(std::vector<unsigned int> &anIndex,
		SMatrix55 &aJacobian, const GblPoint &aPoint, unsigned int measDim,
		unsigned int nJacobian) const {

	unsigned int nDim = theDimension.size();
	unsigned int nCurv = numCurvature;
	unsigned int nLocals = numLocals;

	int nOffset = aPoint.getOffset();

	if (nOffset < 0) // need interpolation
			{
		SMatrix22 prevW, prevWJ, nextW, nextWJ, matN;
		SVector2 prevWd, nextWd;
		int ierr;
		aPoint.getDerivatives(0, prevW, prevWJ, prevWd); // W-, W- * J-, W- * d-
		aPoint.getDerivatives(1, nextW, nextWJ, nextWd); // W-, W- * J-, W- * d-
		const SMatrix22 sumWJ(prevWJ + nextWJ);
		matN = sumWJ.Inverse(ierr); // N = (W- * J- + W+ * J+)^-1
		// derivatives for u_int
		const SMatrix22 prevNW(matN * prevW); // N * W-
		const SMatrix22 nextNW(matN * nextW); // N * W+
		const SVector2 prevNd(matN * prevWd); // N * W- * d-
		const SVector2 nextNd(matN * nextWd); // N * W+ * d+

		unsigned int iOff = nDim * (-nOffset - 1) + nLocals + nCurv + 1; // first offset ('i' in u_i)

		// local offset
		if (nCurv > 0) {
			aJacobian.Place_in_col(-prevNd - nextNd, 3, 0); // from curvature
			anIndex[0] = nLocals + 1;
		}
		aJacobian.Place_at(prevNW, 3, 1); // from 1st Offset
		aJacobian.Place_at(nextNW, 3, 3); // from 2nd Offset
		for (unsigned int i = 0; i < nDim; ++i) {
			anIndex[1 + theDimension[i]] = iOff + i;
			anIndex[3 + theDimension[i]] = iOff + nDim + i;
		}

		// local slope and curvature
		if (measDim > 2) {
			// derivatives for u'_int
			const SMatrix22 prevWPN(nextWJ * prevNW); // W+ * J+ * N * W-
			const SMatrix22 nextWPN(prevWJ * nextNW); // W- * J- * N * W+
			const SVector2 prevWNd(nextWJ * prevNd); // W+ * J+ * N * W- * d-
			const SVector2 nextWNd(prevWJ * nextNd); // W- * J- * N * W+ * d+
			if (nCurv > 0) {
				aJacobian(0, 0) = 1.0;
				aJacobian.Place_in_col(prevWNd - nextWNd, 1, 0); // from curvature
			}
			aJacobian.Place_at(-prevWPN, 1, 1); // from 1st Offset
			aJacobian.Place_at(nextWPN, 1, 3); // from 2nd Offset
		}
	} else { // at point
		// anIndex must be sorted
		// forward : iOff2 = iOff1 + nDim, index1 = 1, index2 = 3
		// backward: iOff2 = iOff1 - nDim, index1 = 3, index2 = 1
		unsigned int iOff1 = nDim * nOffset + nCurv + nLocals + 1; // first offset ('i' in u_i)
		unsigned int index1 = 3 - 2 * nJacobian; // index of first offset
		unsigned int iOff2 = iOff1 + nDim * (nJacobian * 2 - 1); // second offset ('i' in u_i)
		unsigned int index2 = 1 + 2 * nJacobian; // index of second offset
		// local offset
		aJacobian(3, index1) = 1.0; // from 1st Offset
		aJacobian(4, index1 + 1) = 1.0;
		for (unsigned int i = 0; i < nDim; ++i) {
			anIndex[index1 + theDimension[i]] = iOff1 + i;
		}

		// local slope and curvature
		if (measDim > 2) {
			SMatrix22 matW, matWJ;
			SVector2 vecWd;
			aPoint.getDerivatives(nJacobian, matW, matWJ, vecWd); // W, W * J, W * d
			double sign = (nJacobian > 0) ? 1. : -1.;
			if (nCurv > 0) {
				aJacobian(0, 0) = 1.0;
				aJacobian.Place_in_col(-sign * vecWd, 1, 0); // from curvature
				anIndex[0] = nLocals + 1;
			}
			aJacobian.Place_at(-sign * matWJ, 1, index1); // from 1st Offset
			aJacobian.Place_at(sign * matW, 1, index2); // from 2nd Offset
			for (unsigned int i = 0; i < nDim; ++i) {
				anIndex[index2 + theDimension[i]] = iOff2 + i;
			}
		}
	}
}

/// Get jacobian for transformation from (trajectory) fit to kink parameters at point.
/**
 * Jacobian broken lines (q/p,..,u_i-1,u_i,u_i+1..) to kink (du') parameters.
 * \param [out] anIndex List of fit parameters with non zero derivatives
 * \param [out] aJacobian Corresponding transformation matrix
 * \param [in] aPoint Point to use
 */
void GblTrajectory::getFitToKinkJacobian(std::vector<unsigned int> &anIndex,
		SMatrix27 &aJacobian, const GblPoint &aPoint) const {

	unsigned int nDim = theDimension.size();
	unsigned int nCurv = numCurvature;
	unsigned int nLocals = numLocals;

	int nOffset = aPoint.getOffset();

	SMatrix22 prevW, prevWJ, nextW, nextWJ;
	SVector2 prevWd, nextWd;
	aPoint.getDerivatives(0, prevW, prevWJ, prevWd); // W-, W- * J-, W- * d-
	aPoint.getDerivatives(1, nextW, nextWJ, nextWd); // W-, W- * J-, W- * d-
	const SMatrix22 sumWJ(prevWJ + nextWJ); // W- * J- + W+ * J+
	const SVector2 sumWd(prevWd + nextWd); // W+ * d+ + W- * d-

	unsigned int iOff = (nOffset - 1) * nDim + nCurv + nLocals + 1; // first offset ('i' in u_i)

	// local offset
	if (nCurv > 0) {
		aJacobian.Place_in_col(-sumWd, 0, 0); // from curvature
		anIndex[0] = nLocals + 1;
	}
	aJacobian.Place_at(prevW, 0, 1); // from 1st Offset
	aJacobian.Place_at(-sumWJ, 0, 3); // from 2nd Offset
	aJacobian.Place_at(nextW, 0, 5); // from 1st Offset
	for (unsigned int i = 0; i < nDim; ++i) {
		anIndex[1 + theDimension[i]] = iOff + i;
		anIndex[3 + theDimension[i]] = iOff + nDim + i;
		anIndex[5 + theDimension[i]] = iOff + nDim * 2 + i;
	}
}

/// Get fit results at point.
/**
 * Get corrections and covariance matrix for local track and additional parameters
 * in forward or backward direction.
 * \param [in] aSignedLabel (Signed) label of point on trajectory
 * (<0: in front, >0: after point, slope changes at scatterer!)
 * \param [out] localPar Corrections for local parameters
 * \param [out] localCov Covariance for local parameters
 * \return error code (non-zero if trajectory not fitted successfully)
 */
unsigned int GblTrajectory::getResults(int aSignedLabel, TVectorD &localPar,
		TMatrixDSym &localCov) const {
	if (not fitOK)
		return 1;
	std::pair<std::vector<unsigned int>, TMatrixD> indexAndJacobian =
			getJacobian(aSignedLabel);
	unsigned int nParBrl = indexAndJacobian.first.size();
	TVectorD aVec(nParBrl); // compressed vector
	for (unsigned int i = 0; i < nParBrl; ++i) {
		aVec[i] = theVector(indexAndJacobian.first[i] - 1);
	}
	TMatrixDSym aMat = theMatrix.getBlockMatrix(indexAndJacobian.first); // compressed matrix
	localPar = indexAndJacobian.second * aVec;
	localCov = aMat.Similarity(indexAndJacobian.second);
	return 0;
}

/// Get residuals at point from measurement.
/**
 * Get (diagonalized) residual, error of measurement and residual and down-weighting
 * factor for measurement at point
 *
 * \param [in]  aLabel Label of point on trajectory
 * \param [out] numData Number of data blocks from measurement at point
 * \param [out] aResiduals Measurements-Predictions
 * \param [out] aMeasErrors Errors of Measurements
 * \param [out] aResErrors Errors of Residuals (including correlations from track fit)
 * \param [out] aDownWeights Down-Weighting factors
 * \return error code (non-zero if trajectory not fitted successfully)
 */
unsigned int GblTrajectory::getMeasResults(unsigned int aLabel,
		unsigned int &numData, TVectorD &aResiduals, TVectorD &aMeasErrors,
		TVectorD &aResErrors, TVectorD &aDownWeights) {
	numData = 0;
	if (not fitOK)
		return 1;

	unsigned int firstData = measDataIndex[aLabel - 1]; // first data block with measurement
	numData = measDataIndex[aLabel] - firstData; // number of data blocks
	for (unsigned int i = 0; i < numData; ++i) {
		getResAndErr(firstData + i, aResiduals[i], aMeasErrors[i],
				aResErrors[i], aDownWeights[i]);
	}
	return 0;
}

/// Get (kink) residuals at point from scatterer.
/**
 * Get (diagonalized) residual, error of measurement and residual and down-weighting
 * factor for scatterering kinks at point
 *
 * \param [in]  aLabel Label of point on trajectory
 * \param [out] numData Number of data blocks from scatterer at point
 * \param [out] aResiduals (kink)Measurements-(kink)Predictions
 * \param [out] aMeasErrors Errors of (kink)Measurements
 * \param [out] aResErrors Errors of Residuals (including correlations from track fit)
 * \param [out] aDownWeights Down-Weighting factors
 * \return error code (non-zero if trajectory not fitted successfully)
 */
unsigned int GblTrajectory::getScatResults(unsigned int aLabel,
		unsigned int &numData, TVectorD &aResiduals, TVectorD &aMeasErrors,
		TVectorD &aResErrors, TVectorD &aDownWeights) {
	numData = 0;
	if (not fitOK)
		return 1;

	unsigned int firstData = scatDataIndex[aLabel - 1]; // first data block with scatterer
	numData = scatDataIndex[aLabel] - firstData; // number of data blocks
	for (unsigned int i = 0; i < numData; ++i) {
		getResAndErr(firstData + i, aResiduals[i], aMeasErrors[i],
				aResErrors[i], aDownWeights[i]);
	}
	return 0;
}

/// Get (list of) labels of points on (simple) trajectory
/**
 * \param [out] aLabelList List of labels (aLabelList[i] = i+1)
 */
void GblTrajectory::getLabels(std::vector<unsigned int> &aLabelList) {
	unsigned int aLabel = 0;
	unsigned int nPoint = thePoints[0].size();
	aLabelList.resize(nPoint);
	for (unsigned i = 0; i < nPoint; ++i) {
		aLabelList[i] = ++aLabel;
	}
}

/// Get (list of lists of) labels of points on (composed) trajectory
/**
 * \param [out] aLabelList List of of lists of labels
 */
void GblTrajectory::getLabels(
		std::vector<std::vector<unsigned int> > &aLabelList) {
	unsigned int aLabel = 0;
	aLabelList.resize(numTrajectories);
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		unsigned int nPoint = thePoints[iTraj].size();
		aLabelList[iTraj].resize(nPoint);
		for (unsigned i = 0; i < nPoint; ++i) {
			aLabelList[iTraj][i] = ++aLabel;
		}
	}
}

/// Get residual and errors from data block.
/**
 * Get residual, error of measurement and residual and down-weighting
 * factor for (single) data block
 * \param [in]  aData Label of data block
 * \param [out] aResidual Measurement-Prediction
 * \param [out] aMeasError Error of Measurement
 * \param [out] aResError Error of Residual (including correlations from track fit)
 * \param [out] aDownWeight Down-Weighting factor
 */
void GblTrajectory::getResAndErr(unsigned int aData, double &aResidual,
		double &aMeasError, double &aResError, double &aDownWeight) {

	double aMeasVar;
	std::vector<unsigned int>* indLocal;
	std::vector<double>* derLocal;
	theData[aData].getResidual(aResidual, aMeasVar, aDownWeight, indLocal,
			derLocal);
	unsigned int nParBrl = (*indLocal).size();
	TVectorD aVec(nParBrl); // compressed vector of derivatives
	for (unsigned int j = 0; j < nParBrl; ++j) {
		aVec[j] = (*derLocal)[j];
	}
	TMatrixDSym aMat = theMatrix.getBlockMatrix(*indLocal); // compressed (covariance) matrix
	double aFitVar = aMat.Similarity(aVec); // variance from track fit
	aMeasError = sqrt(aMeasVar); // error of measurement
	aResError = (aFitVar < aMeasVar ? sqrt(aMeasVar - aFitVar) : 0.); // error of residual
}

/// Build linear equation system from data (blocks).
void GblTrajectory::buildLinearEquationSystem() {
	unsigned int nBorder = numCurvature + numLocals;
	theVector.resize(numParameters);
	theMatrix.resize(numParameters, nBorder);
	double aValue, aWeight;
	std::vector<unsigned int>* indLocal;
	std::vector<double>* derLocal;
	std::vector<GblData>::iterator itData;
	for (itData = theData.begin(); itData < theData.end(); ++itData) {
		itData->getLocalData(aValue, aWeight, indLocal, derLocal);
		for (unsigned int j = 0; j < indLocal->size(); ++j) {
			theVector((*indLocal)[j] - 1) += (*derLocal)[j] * aWeight * aValue;
		}
		theMatrix.addBlockMatrix(aWeight, indLocal, derLocal);
	}
}

/// Prepare fit for simple or composed trajectory
/**
 * Generate data (blocks) from measurements, kinks, external seed and measurements.
 */
void GblTrajectory::prepare() {
	unsigned int nDim = theDimension.size();
	// upper limit
	unsigned int maxData = numMeasurements + nDim * (numOffsets - 2)
			+ externalSeed.GetNrows();
	theData.reserve(maxData);
	measDataIndex.resize(numAllPoints + 3); // include external seed and measurements
	scatDataIndex.resize(numAllPoints + 1);
	unsigned int nData = 0;
	std::vector<TMatrixD> innerTransDer;
	std::vector<std::vector<unsigned int> > innerTransLab;
	// composed trajectory ?
	if (numInnerTrans > 0) {
		//std::cout << "composed trajectory" << std::endl;
		for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
			// innermost point
			GblPoint* innerPoint = &thePoints[iTraj].front();
			// transformation fit to local track parameters
			std::vector<unsigned int> firstLabels(5);
			SMatrix55 matFitToLocal, matLocalToFit;
			getFitToLocalJacobian(firstLabels, matFitToLocal, *innerPoint, 5);
			// transformation local track to fit parameters
			int ierr;
			matLocalToFit = matFitToLocal.Inverse(ierr);
			TMatrixD localToFit(5, 5);
			for (unsigned int i = 0; i < 5; ++i) {
				for (unsigned int j = 0; j < 5; ++j) {
					localToFit(i, j) = matLocalToFit(i, j);
				}
			}
			// transformation external to fit parameters at inner (first) point
			innerTransDer.push_back(localToFit * innerTransformations[iTraj]);
			innerTransLab.push_back(firstLabels);
		}
	}
	// measurements
	SMatrix55 matP;
	// loop over trajectories
	std::vector<GblPoint>::iterator itPoint;
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		for (itPoint = thePoints[iTraj].begin();
				itPoint < thePoints[iTraj].end(); ++itPoint) {
			SVector5 aMeas, aPrec;
			unsigned int nLabel = itPoint->getLabel();
			unsigned int measDim = itPoint->hasMeasurement();
			if (measDim) {
				const TMatrixD localDer = itPoint->getLocalDerivatives();
				const std::vector<int> globalLab = itPoint->getGlobalLabels();
				const TMatrixD globalDer = itPoint->getGlobalDerivatives();
				TMatrixD transDer;
				itPoint->getMeasurement(matP, aMeas, aPrec);
				unsigned int iOff = 5 - measDim; // first active component
				std::vector<unsigned int> labDer(5);
				SMatrix55 matDer, matPDer;
				unsigned int nJacobian =
						(itPoint < thePoints[iTraj].end() - 1) ? 1 : 0; // last point needs backward propagation
				getFitToLocalJacobian(labDer, matDer, *itPoint, measDim,
						nJacobian);
				if (measDim > 2) {
					matPDer = matP * matDer;
				} else { // 'shortcut' for position measurements
					matPDer.Place_at(
							matP.Sub<SMatrix22>(3, 3)
									* matDer.Sub<SMatrix25>(3, 0), 3, 0);
				}

				if (numInnerTrans > 0) {
					// transform for external parameters
					TMatrixD proDer(measDim, 5);
					// match parameters
					unsigned int ifirst = 0;
					unsigned int ilabel = 0;
					while (ilabel < 5) {
						if (labDer[ilabel] > 0) {
							while (innerTransLab[iTraj][ifirst]
									!= labDer[ilabel] and ifirst < 5) {
								++ifirst;
							}
							if (ifirst >= 5) {
								labDer[ilabel] -= 2 * nDim * (iTraj + 1); // adjust label
							} else {
								// match
								labDer[ilabel] = 0; // mark as related to external parameters
								for (unsigned int k = iOff; k < 5; ++k) {
									proDer(k - iOff, ifirst) = matPDer(k,
											ilabel);
								}
							}
						}
						++ilabel;
					}
					transDer.ResizeTo(measDim, numCurvature);
					transDer = proDer * innerTransDer[iTraj];
				}
				for (unsigned int i = iOff; i < 5; ++i) {
					if (aPrec(i) > 0.) {
						GblData aData(nLabel, aMeas(i), aPrec(i));
						aData.addDerivatives(i, labDer, matPDer, iOff, localDer,
								globalLab, globalDer, numLocals, transDer);
						theData.push_back(aData);
						nData++;
					}
				}

			}
			measDataIndex[nLabel] = nData;
		}
	}

	// pseudo measurements from kinks
	SMatrix22 matT;
	scatDataIndex[0] = nData;
	scatDataIndex[1] = nData;
	// loop over trajectories
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		for (itPoint = thePoints[iTraj].begin() + 1;
				itPoint < thePoints[iTraj].end() - 1; ++itPoint) {
			SVector2 aMeas, aPrec;
			unsigned int nLabel = itPoint->getLabel();
			if (itPoint->hasScatterer()) {
				itPoint->getScatterer(matT, aMeas, aPrec);
				TMatrixD transDer;
				std::vector<unsigned int> labDer(7);
				SMatrix27 matDer, matTDer;
				getFitToKinkJacobian(labDer, matDer, *itPoint);
				matTDer = matT * matDer;
				if (numInnerTrans > 0) {
					// transform for external parameters
					TMatrixD proDer(nDim, 5);
					// match parameters
					unsigned int ifirst = 0;
					unsigned int ilabel = 0;
					while (ilabel < 7) {
						if (labDer[ilabel] > 0) {
							while (innerTransLab[iTraj][ifirst]
									!= labDer[ilabel] and ifirst < 5) {
								++ifirst;
							}
							if (ifirst >= 5) {
								labDer[ilabel] -= 2 * nDim * (iTraj + 1); // adjust label
							} else {
								// match
								labDer[ilabel] = 0; // mark as related to external parameters
								for (unsigned int k = 0; k < nDim; ++k) {
									proDer(k, ifirst) = matTDer(k, ilabel);
								}
							}
						}
						++ilabel;
					}
					transDer.ResizeTo(nDim, numCurvature);
					transDer = proDer * innerTransDer[iTraj];
				}
				for (unsigned int i = 0; i < nDim; ++i) {
					unsigned int iDim = theDimension[i];
					if (aPrec(iDim) > 0.) {
						GblData aData(nLabel, aMeas(iDim), aPrec(iDim));
						aData.addDerivatives(iDim, labDer, matTDer, numLocals,
								transDer);
						theData.push_back(aData);
						nData++;
					}
				}
			}
			scatDataIndex[nLabel] = nData;
		}
		scatDataIndex[thePoints[iTraj].back().getLabel()] = nData;
	}

	// external seed
	if (externalPoint > 0) {
		std::pair<std::vector<unsigned int>, TMatrixD> indexAndJacobian =
				getJacobian(externalPoint);
		std::vector<unsigned int> externalSeedIndex = indexAndJacobian.first;
		std::vector<double> externalSeedDerivatives(externalSeedIndex.size());
		const TMatrixDSymEigen externalSeedEigen(externalSeed);
		const TVectorD valEigen = externalSeedEigen.GetEigenValues();
		TMatrixD vecEigen = externalSeedEigen.GetEigenVectors();
		vecEigen = vecEigen.T() * indexAndJacobian.second;
		for (int i = 0; i < externalSeed.GetNrows(); ++i) {
			if (valEigen(i) > 0.) {
				for (int j = 0; j < externalSeed.GetNcols(); ++j) {
					externalSeedDerivatives[j] = vecEigen(i, j);
				}
				GblData aData(externalPoint, 0., valEigen(i));
				aData.addDerivatives(externalSeedIndex, externalSeedDerivatives);
				theData.push_back(aData);
				nData++;
			}
		}
	}
	measDataIndex[numAllPoints + 1] = nData;
	// external measurements
	unsigned int nExt = externalMeasurements.GetNrows();
	if (nExt > 0) {
		std::vector<unsigned int> index(numCurvature);
		std::vector<double> derivatives(numCurvature);
		for (unsigned int iExt = 0; iExt < nExt; ++iExt) {
			for (unsigned int iCol = 0; iCol < numCurvature; ++iCol) {
				index[iCol] = iCol + 1;
				derivatives[iCol] = externalDerivatives(iExt, iCol);
			}
			GblData aData(1U, externalMeasurements(iExt),
					externalPrecisions(iExt));
			aData.addDerivatives(index, derivatives);
			theData.push_back(aData);
			nData++;
		}
	}
	measDataIndex[numAllPoints + 2] = nData;
}

/// Calculate predictions for all points.
void GblTrajectory::predict() {
	std::vector<GblData>::iterator itData;
	for (itData = theData.begin(); itData < theData.end(); ++itData) {
		itData->setPrediction(theVector);
	}
}

/// Down-weight all points.
/**
 * \param [in] aMethod M-estimator (1: Tukey, 2:Huber, 3:Cauchy)
 */
double GblTrajectory::downWeight(unsigned int aMethod) {
	double aLoss = 0.;
	std::vector<GblData>::iterator itData;
	for (itData = theData.begin(); itData < theData.end(); ++itData) {
		aLoss += (1. - itData->setDownWeighting(aMethod));
	}
	return aLoss;
}

/// Perform fit of trajectory.
/**
 * Optionally iterate for outlier down-weighting.
 * \param [out] Chi2 Chi2 sum (corrected for down-weighting)
 * \param [out] Ndf  Number of degrees of freedom
 * \param [out] lostWeight Sum of weights lost due to down-weighting
 * \param [in] optionList Iterations for down-weighting
 * (One character per iteration: t,h,c (or T,H,C) for Tukey, Huber or Cauchy function)
 * \return Error code (non zero value indicates failure of fit)
 */
unsigned int GblTrajectory::fit(double &Chi2, int &Ndf, double &lostWeight,
		std::string optionList) {
	const double normChi2[4] = { 1.0, 0.8737, 0.9326, 0.8228 };
	const std::string methodList = "TtHhCc";

	Chi2 = 0.;
	Ndf = -1;
	lostWeight = 0.;
	if (not constructOK)
		return 10;

	unsigned int aMethod = 0;

	buildLinearEquationSystem();
	lostWeight = 0.;
	unsigned int ierr = 0;
	try {

		theMatrix.solveAndInvertBorderedBand(theVector, theVector);
		predict();

		for (unsigned int i = 0; i < optionList.size(); ++i) // down weighting iterations
				{
			size_t aPosition = methodList.find(optionList[i]);
			if (aPosition != std::string::npos) {
				aMethod = aPosition / 2 + 1;
				lostWeight = downWeight(aMethod);
				buildLinearEquationSystem();
				theMatrix.solveAndInvertBorderedBand(theVector, theVector);
				predict();
			}
		}
		Ndf = theData.size() - numParameters;
		Chi2 = 0.;
		for (unsigned int i = 0; i < theData.size(); ++i) {
			Chi2 += theData[i].getChi2();
		}
		Chi2 /= normChi2[aMethod];
		fitOK = true;

	} catch (int e) {
		std::cout << " GblTrajectory::fit exception " << e << std::endl;
		ierr = e;
	}
	return ierr;
}

/// Write valid trajectory to Millepede-II binary file.
void GblTrajectory::milleOut(MilleBinary &aMille) {
	double aValue;
	double aErr;
	std::vector<unsigned int>* indLocal;
	std::vector<double>* derLocal;
	std::vector<int>* labGlobal;
	std::vector<double>* derGlobal;

	if (not constructOK)
		return;

//   data: measurements, kinks and external seed
	std::vector<GblData>::iterator itData;
	for (itData = theData.begin(); itData != theData.end(); ++itData) {
		itData->getAllData(aValue, aErr, indLocal, derLocal, labGlobal,
				derGlobal);
		aMille.addData(aValue, aErr, *indLocal, *derLocal, *labGlobal,
				*derGlobal);
	}
	aMille.writeRecord();
}

/// Print GblTrajectory
/**
 * \param [in] level print level (0: minimum, >0: more)
 */
void GblTrajectory::printTrajectory(unsigned int level) {
	if (numInnerTrans) {
		std::cout << "Composed GblTrajectory, " << numInnerTrans
				<< " subtrajectories" << std::endl;
	} else {
		std::cout << "Simple GblTrajectory" << std::endl;
	}
	if (theDimension.size() < 2) {
		std::cout << " 2D-trajectory" << std::endl;
	}
	std::cout << " Number of GblPoints          : " << numAllPoints
			<< std::endl;
	std::cout << " Number of points with offsets: " << numOffsets << std::endl;
	std::cout << " Number of fit parameters     : " << numParameters
			<< std::endl;
	std::cout << " Number of measurements       : " << numMeasurements
			<< std::endl;
	if (externalMeasurements.GetNrows()) {
		std::cout << " Number of ext. measurements  : "
				<< externalMeasurements.GetNrows() << std::endl;
	}
	if (externalPoint) {
		std::cout << " Label of point with ext. seed: " << externalPoint
				<< std::endl;
	}
	if (constructOK) {
		std::cout << " Constructed OK " << std::endl;
	}
	if (fitOK) {
		std::cout << " Fitted OK " << std::endl;
	}
	if (level > 0) {
		if (numInnerTrans) {
			std::cout << " Inner transformations" << std::endl;
			for (unsigned int i = 0; i < numInnerTrans; ++i) {
				innerTransformations[i].Print();
			}
		}
		if (externalMeasurements.GetNrows()) {
			std::cout << " External measurements" << std::endl;
			std::cout << "  Measurements:" << std::endl;
			externalMeasurements.Print();
			std::cout << "  Precisions:" << std::endl;
			externalPrecisions.Print();
			std::cout << "  Derivatives:" << std::endl;
			externalDerivatives.Print();
		}
		if (externalPoint) {
			std::cout << " External seed:" << std::endl;
			externalSeed.Print();
		}
		if (fitOK) {
			std::cout << " Fit results" << std::endl;
			std::cout << "  Parameters:" << std::endl;
			theVector.print();
			std::cout << "  Covariance matrix (bordered band part):"
					<< std::endl;
			theMatrix.printMatrix();
		}
	}
}

/// Print \link GblPoint GblPoints \endlink on trajectory
/**
 * \param [in] level print level (0: minimum, >0: more)
 */
void GblTrajectory::printPoints(unsigned int level) {
	std::cout << "GblPoints " << std::endl;
	for (unsigned int iTraj = 0; iTraj < numTrajectories; ++iTraj) {
		std::vector<GblPoint>::iterator itPoint;
		for (itPoint = thePoints[iTraj].begin();
				itPoint < thePoints[iTraj].end(); ++itPoint) {
			itPoint->printPoint(level);
		}
	}
}

/// Print GblData blocks for trajectory
void GblTrajectory::printData() {
	std::cout << "GblData blocks " << std::endl;
	std::vector<GblData>::iterator itData;
	for (itData = theData.begin(); itData < theData.end(); ++itData) {
		itData->printData();
	}
}

}
