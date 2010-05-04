#ifndef Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H
#define Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H

/** \class CSCPairConstraint
 *  $Date: Fri Mar 26 10:47:07 CDT 2010 $
 *  $Revision: 1.0 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCPairConstraint {
public:
  CSCPairConstraint(int i, int j, double value, double error)
    : m_i(i), m_j(j), m_value(value), m_error(error) {};
  virtual ~CSCPairConstraint() {};

  virtual int i() const { return m_i; };
  virtual int j() const { return m_j; };
  virtual double value() const { return m_value; };
  virtual double error() const { return m_error; };

protected:
  int m_i, m_j;
  double m_value, m_error;
};

#endif // Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H
