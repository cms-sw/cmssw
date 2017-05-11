/*
 * MilleBinary.h
 *
 *  Created on: Aug 31, 2011
 *      Author: kleinwrt
 */

/** \file
 *  MilleBinary definition.
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

#ifndef MILLEBINARY_H_
#define MILLEBINARY_H_

#include<fstream>
#include<vector>

//! Namespace for the general broken lines package
namespace gbl {

  ///  Millepede-II (binary) record.
  /**
   *  Containing information for local (track) and global fit.
   *
   *  The data blocks are collected in two arrays, a real array
   *  (containing float or double values) and integer array, of same length.
   *  A positive record length indicate _float_ and a negative one _double_ values.
   *  The content of the record is:
   *\verbatim
   *         real array              integer array
   *     0   0.0                     error count (this record)
   *     1   RMEAS, measured value   0                            -+
   *     2   local derivative        index of local derivative     |
   *     3   local derivative        index of local derivative     |
   *     4    ...                                                  | block
   *         SIGMA, error (>0)       0                             |
   *         global derivative       label of global derivative    |
   *         global derivative       label of global derivative   -+
   *         RMEAS, measured value   0
   *         local derivative        index of local derivative
   *         local derivative        index of local derivative
   *         ...
   *         SIGMA, error            0
   *         global derivative       label of global derivative
   *         global derivative       label of global derivative
   *         ...
   *         global derivative       label of global derivative
   *\endverbatim
   */
  class MilleBinary {
  public:
    MilleBinary(const std::string& fileName = "milleBinaryISN.dat",
                bool doublePrec = false, unsigned int aSize = 2000);
    virtual ~MilleBinary();
    void addData(double aMeas, double aErr, unsigned int numLocal,
                 unsigned int* indLocal, double* derLocal,
                 const std::vector<int> &labGlobal,
                 const std::vector<double> &derGlobal);
    void writeRecord();

  private:
    std::ofstream binaryFile; ///< Binary File
    std::vector<int> intBuffer; ///< Integer buffer
    std::vector<float> floatBuffer; ///< Float buffer
    std::vector<double> doubleBuffer; ///< Double buffer
    bool doublePrecision; ///< Flag for storage in as *double* values
  };
}
#endif /* MILLEBINARY_H_ */
