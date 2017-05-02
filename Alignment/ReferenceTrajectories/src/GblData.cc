/*
 * GblData.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

/** \file
 *  GblData methods.
 *
 *  \author Claus Kleinwort, DESY, 2011 (Claus.Kleinwort@desy.de)
 *
 *  \copyright
 *  Copyright (c) 2011 - 2016 Deutsches Elektronen-Synchroton,
 *  Member of the Helmholtz Association, (DESY), HAMBURG, GERMANY \n\n
 *  This library is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Library General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version. \n\n
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Library General Public License for more details. \n\n
 *  You should have received a copy of the GNU Library General Public
 *  License along with this program (see the file COPYING.LIB for more
 *  details); if not, write to the Free Software Foundation, Inc.,
 *  675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "Alignment/ReferenceTrajectories/interface/GblData.h"
using namespace Eigen;

//! Namespace for the general broken lines package
namespace gbl {

  /// Create data block.
  /**
   * \param [in] aLabel Label of corresponding point
   * \param [in] aType Type of (scalar) measurement
   * \param [in] aValue Value of (scalar) measurement
   * \param [in] aPrec Precision of (scalar) measurement
   * \param [in] aTraj Trajectory number
   * \param [in] aPoint Point number
   */
  GblData::GblData(unsigned int aLabel, dataBlockType aType, double aValue,
                   double aPrec, unsigned int aTraj, unsigned int aPoint) :
    theLabel(aLabel), theRow(0), theType(aType), theValue(aValue),
    thePrecision(aPrec), theTrajectory(aTraj), thePoint(aPoint),
    theDownWeight(1.), thePrediction(0.), theNumLocal(0), moreParameters(),
    moreDerivatives()
  {
  }

  GblData::~GblData() {
  }

  /// Add derivatives from external seed.
  /**
   * Add (non-zero) derivatives to data block. Fill list of labels of used fit parameters.
   * \param [in] index Labels for derivatives
   * \param [in] derivatives Derivatives (vector)
   */
  void GblData::addDerivatives(const std::vector<unsigned int> &index,
                               const std::vector<double> &derivatives) {
    for (unsigned int i = 0; i < derivatives.size(); ++i) // any derivatives
      {
        if (derivatives[i]) {
          moreParameters.push_back(index[i]);
          moreDerivatives.push_back(derivatives[i]);
        }
      }
  }

  /// Calculate prediction for data from fit (by GblTrajectory::fit).
  void GblData::setPrediction(const VVector &aVector) {

    thePrediction = 0.;
    if (theNumLocal > 0) {
      for (unsigned int i = 0; i < theNumLocal; ++i) {
        thePrediction += theDerivatives[i] * aVector(theParameters[i] - 1);
      }
    } else {
      for (unsigned int i = 0; i < moreDerivatives.size(); ++i) {
        thePrediction += moreDerivatives[i]
          * aVector(moreParameters[i] - 1);
      }
    }
  }

  /// Outlier down weighting with M-estimators (by GblTrajectory::fit).
  /**
   * \param [in] aMethod M-estimator (1: Tukey, 2:Huber, 3:Cauchy)
   */
  double GblData::setDownWeighting(unsigned int aMethod) {

    double aWeight = 1.;
    double scaledResidual = fabs(theValue - thePrediction) * sqrt(thePrecision);
    if (aMethod == 1) // Tukey
      {
        if (scaledResidual < 4.6851) {
          aWeight = (1.0 - 0.045558 * scaledResidual * scaledResidual);
          aWeight *= aWeight;
        } else {
          aWeight = 0.;
        }
      } else if (aMethod == 2) //Huber
      {
        if (scaledResidual >= 1.345) {
          aWeight = 1.345 / scaledResidual;
        }
      } else if (aMethod == 3) //Cauchy
      {
        aWeight = 1.0 / (1.0 + (scaledResidual * scaledResidual / 5.6877));
      }
    theDownWeight = aWeight;
    return aWeight;
  }

  /// Calculate Chi2 contribution.
  /**
   * \return (down-weighted) Chi2
   */
  double GblData::getChi2() const {
    double aDiff = theValue - thePrediction;
    return aDiff * aDiff * thePrecision * theDownWeight;
  }

  /// Print data block.
  void GblData::printData() const {

    std::cout << " measurement at label " << theLabel << " of type " << theType
              << " from row " << theRow << ": " << theValue << ", "
              << thePrecision << std::endl;
    std::cout << "  param " << moreParameters.size() + theNumLocal << ":";
    for (unsigned int i = 0; i < moreParameters.size(); ++i) {
      std::cout << " " << moreParameters[i];
    }
    for (unsigned int i = 0; i < theNumLocal; ++i) {
      std::cout << " " << theParameters[i];
    }
    std::cout << std::endl;
    std::cout << "  deriv " << moreDerivatives.size() + theNumLocal << ":";
    for (unsigned int i = 0; i < moreDerivatives.size(); ++i) {
      std::cout << " " << moreDerivatives[i];
    }
    for (unsigned int i = 0; i < theNumLocal; ++i) {
      std::cout << " " << theDerivatives[i];
    }
    std::cout << std::endl;
  }

  /// Get label.
  /**
   * \return label of corresponding point
   */
  unsigned int GblData::getLabel() const {
    return theLabel;
  }

  /// Get type.
  /**
   * \return type
   */
  dataBlockType GblData::getType() const {
    return theType;
  }

  /// Get Data for local fit.
  /**
   * \param [out] aValue Value
   * \param [out] aWeight Weight
   * \param [out] numLocal Number of local labels/derivatives
   * \param [out] indLocal Array of labels of used (local) fit parameters
   * \param [out] derLocal Array of derivatives for used (local) fit parameters
   */
  void GblData::getLocalData(double &aValue, double &aWeight,
                             unsigned int &numLocal, unsigned int* &indLocal, double* &derLocal) {

    aValue = theValue;
    aWeight = thePrecision * theDownWeight;
    if (theNumLocal > 0) {
      numLocal = theNumLocal;
      indLocal = theParameters;
      derLocal = theDerivatives;
    } else {
      numLocal = moreParameters.size();
      indLocal = &moreParameters[0];
      derLocal = &moreDerivatives[0];
    }
  }

  /// Get all Data for MP-II binary record.
  /**
   * \param [out] aValue Value
   * \param [out] aErr Error
   * \param [out] numLocal Number of local labels/derivatives
   * \param [out] indLocal Array of labels of used (local) fit parameters
   * \param [out] derLocal Array of derivatives for used (local) fit parameters
   * \param [out] aTraj Trajectory number
   * \param [out] aPoint Point number
   * \param [out] aRow Row number
   */
  void GblData::getAllData(double &aValue, double &aErr, unsigned int &numLocal,
                           unsigned int* &indLocal, double* &derLocal, unsigned int &aTraj,
                           unsigned int &aPoint, unsigned int &aRow) {
    aValue = theValue;
    aErr = 1.0 / sqrt(thePrecision);
    if (theNumLocal > 0) {
      numLocal = theNumLocal;
      indLocal = theParameters;
      derLocal = theDerivatives;
    } else {
      numLocal = moreParameters.size();
      indLocal = &moreParameters[0];
      derLocal = &moreDerivatives[0];
    }
    aTraj = theTrajectory;
    aPoint = thePoint;
    aRow = theRow;
  }

  /// Get data for residual (and errors).
  /**
   * \param [out] aResidual Measurement-Prediction
   * \param [out] aVariance Variance (of measurement)
   * \param [out] aDownWeight Down-weighting factor
   * \param [out] numLocal Number of local labels/derivatives
   * \param [out] indLocal Array of labels of used (local) fit parameters
   * \param [out] derLocal Array of derivatives for used (local) fit parameters
   */
  void GblData::getResidual(double &aResidual, double &aVariance,
                            double &aDownWeight, unsigned int &numLocal, unsigned int* &indLocal,
                            double* &derLocal) {
    aResidual = theValue - thePrediction;
    aVariance = 1.0 / thePrecision;
    aDownWeight = theDownWeight;
    if (theNumLocal > 0) {
      numLocal = theNumLocal;
      indLocal = theParameters;
      derLocal = theDerivatives;
    } else {
      numLocal = moreParameters.size();
      indLocal = &moreParameters[0];
      derLocal = &moreDerivatives[0];
    }
  }
}
