/*
 * MilleBinary.cpp
 *
 *  Created on: Aug 31, 2011
 *      Author: kleinwrt
 */

/** \file
 *  MilleBinary methods.
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

#include "Alignment/ReferenceTrajectories/interface/MilleBinary.h"

//! Namespace for the general broken lines package
namespace gbl {

  /// Create binary file.
  /**
   * \param [in] fileName File name
   * \param [in] doublePrec Flag for storage as double values
   * \param [in] aSize Buffer size
   */
  MilleBinary::MilleBinary(const std::string& fileName, bool doublePrec,
                           unsigned int aSize) :
    binaryFile(fileName.c_str(), std::ios::binary | std::ios::out), intBuffer(),
    floatBuffer(), doubleBuffer(), doublePrecision(doublePrec)
  {
    intBuffer.reserve(aSize);
    intBuffer.push_back(0); // first word is error counter
    if (doublePrecision) {
      doubleBuffer.reserve(aSize);
      doubleBuffer.push_back(0.);

    } else {
      floatBuffer.reserve(aSize);
      floatBuffer.push_back(0.);
    }
  }

  MilleBinary::~MilleBinary() {
    binaryFile.close();
  }

  /// Add data block to (end of) record.
  /**
   * \param [in] aMeas Value
   * \param [in] aErr Error
   * \param [in] numLocal Number of local labels/derivatives
   * \param [in] indLocal Array of labels of local parameters
   * \param [in] derLocal Array of derivatives for local parameters
   * \param [in] labGlobal List of labels of global parameters
   * \param [in] derGlobal List of derivatives for global parameters
   */
  void MilleBinary::addData(double aMeas, double aErr, unsigned int numLocal,
                            unsigned int* indLocal, double* derLocal,
                            const std::vector<int> &labGlobal,
                            const std::vector<double> &derGlobal) {

    if (doublePrecision) {
      // double values
      intBuffer.push_back(0);
      doubleBuffer.push_back(aMeas);
      for (unsigned int i = 0; i < numLocal; ++i) {
        intBuffer.push_back(indLocal[i]);
        doubleBuffer.push_back(derLocal[i]);
      }
      intBuffer.push_back(0);
      doubleBuffer.push_back(aErr);
      for (unsigned int i = 0; i < labGlobal.size(); ++i) {
        if (derGlobal[i]) {
          intBuffer.push_back(labGlobal[i]);
          doubleBuffer.push_back(derGlobal[i]);
        }
      }
    } else {
      // float values
      intBuffer.push_back(0);
      floatBuffer.push_back(aMeas);
      for (unsigned int i = 0; i < numLocal; ++i) {
        intBuffer.push_back(indLocal[i]);
        floatBuffer.push_back(derLocal[i]);
      }
      intBuffer.push_back(0);
      floatBuffer.push_back(aErr);
      for (unsigned int i = 0; i < labGlobal.size(); ++i) {
        if (derGlobal[i]) {
          intBuffer.push_back(labGlobal[i]);
          floatBuffer.push_back(derGlobal[i]);
        }
      }
    }
  }

  /// Write record to file.
  void MilleBinary::writeRecord() {

    const int recordLength =
      (doublePrecision) ? -intBuffer.size() * 2 : intBuffer.size() * 2;
    binaryFile.write(reinterpret_cast<const char*>(&recordLength),
                     sizeof(recordLength));
    if (doublePrecision)
      binaryFile.write(reinterpret_cast<char*>(&doubleBuffer[0]),
                       doubleBuffer.size() * sizeof(doubleBuffer[0]));
    else
      binaryFile.write(reinterpret_cast<char*>(&floatBuffer[0]),
                       floatBuffer.size() * sizeof(floatBuffer[0]));
    binaryFile.write(reinterpret_cast<char*>(&intBuffer[0]),
                     intBuffer.size() * sizeof(intBuffer[0]));
    // start with new record
    intBuffer.resize(1);
    if (doublePrecision)
      doubleBuffer.resize(1);
    else
      floatBuffer.resize(1);
  }
}
