#ifndef Alignment_MuonAlignmentAlgorithms_CSCChamberFitter_H
#define Alignment_MuonAlignmentAlgorithms_CSCChamberFitter_H

/** \class CSCChamberFitter
 *  $Date: Fri Mar 26 11:14:26 CDT 2010 $
 *  $Revision: 1.0 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/CSCPairConstraint.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/CSCPairResidualsConstraint.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/CSCAlignmentCorrections.h"

class CSCChamberFitter {
public:
  CSCChamberFitter(const edm::ParameterSet &iConfig, std::vector<CSCPairResidualsConstraint*> &residualsConstraints);
  virtual ~CSCChamberFitter() {};

  bool fit(std::vector<CSCAlignmentCorrections*> &corrections) const;

protected:
  int index(std::string alignable) const;
  void walk(std::map<int,bool> &touched, int alignable) const;
  long alignableId(std::string alignable) const;
  bool isFrame(int i) const;
  double chi2(AlgebraicVector A, double lambda) const;
  double lhsVector(int k) const;
  double hessian(int k, int l, double lambda) const;

  std::string m_name;
  std::vector<std::string> m_alignables;
  std::vector<int> m_frames;
  int m_fixed;
  std::vector<CSCPairConstraint*> m_constraints;
};

#endif // Alignment_MuonAlignmentAlgorithms_CSCChamberFitter_H

