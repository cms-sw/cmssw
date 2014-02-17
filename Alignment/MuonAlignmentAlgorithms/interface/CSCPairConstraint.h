#ifndef Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H
#define Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H

/** \class CSCPairConstraint
 *  $Date: 2010/05/05 04:00:38 $
 *  $Revision: 1.2 $
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
  virtual bool valid() const { return true; };

protected:
  int m_i, m_j;
  double m_value, m_error;
};

#endif // Alignment_MuonAlignmentAlgorithms_CSCPairConstraint_H
