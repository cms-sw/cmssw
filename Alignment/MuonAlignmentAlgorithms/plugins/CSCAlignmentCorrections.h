#ifndef Alignment_MuonAlignmentAlgorithms_CSCAlignmentCorrections_H
#define Alignment_MuonAlignmentAlgorithms_CSCAlignmentCorrections_H

/** \class CSCAlignmentCorrections
 *  $Date: 2010/05/27 19:40:03 $
 *  $Revision: 1.1 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include <fstream>

#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"  
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

class CSCAlignmentCorrections {
public:
  CSCAlignmentCorrections(std::string fitterName, double oldchi2, double newchi2): m_fitterName(fitterName), m_oldchi2(oldchi2), m_newchi2(newchi2) {};
  virtual ~CSCAlignmentCorrections() {};

  void insertCorrection(std::string name, CSCDetId id, double value) {
    m_name.push_back(name);
    m_id.push_back(id);
    m_value.push_back(value);
  };

  void insertMode(std::vector<double> coefficient, std::vector<std::string> modename, std::vector<long> modeid, double error) {
    m_coefficient.push_back(coefficient);
    m_modename.push_back(modename);
    m_modeid.push_back(modeid);
    m_error.push_back(error);
  };

  void insertResidual(std::string i, std::string j, double before, double uncert, double residual, double pull) {
    m_i.push_back(i);
    m_j.push_back(j);
    m_before.push_back(before);
    m_uncert.push_back(uncert);
    m_residual.push_back(residual);
    m_pull.push_back(pull);
  };

  void applyAlignment(AlignableNavigator *alignableNavigator, AlignmentParameterStore *alignmentParameterStore, int mode, bool combineME11);
  void plot();
  void report(std::ofstream &report);

protected:
  std::string m_fitterName;
  double m_oldchi2, m_newchi2;

  // there's one of these for each chamber
  std::vector<std::string> m_name;
  std::vector<CSCDetId> m_id;
  std::vector<double> m_value;

  // there's one of these for each error mode
  std::vector<std::vector<double> > m_coefficient;
  std::vector<std::vector<std::string> > m_modename;
  std::vector<std::vector<long> > m_modeid;
  std::vector<double> m_error;

  // there's one of these for each constraint
  std::vector<std::string> m_i;
  std::vector<std::string> m_j;
  std::vector<double> m_before;
  std::vector<double> m_uncert;
  std::vector<double> m_residual;
  std::vector<double> m_pull;

  std::vector<TH1F*> th1f_modes;
};

#endif // Alignment_MuonAlignmentAlgorithms_CSCAlignmentCorrections_H
