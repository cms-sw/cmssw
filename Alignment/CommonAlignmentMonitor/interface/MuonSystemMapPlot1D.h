#ifndef Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
#define Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H

/** \class MuonSystemMapPlot1D
 *  $Date: 2010/01/06 15:23:09 $
 *  $Revision: 1.3 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals1DOFFitter.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include <string>
#include <sstream>

class AlignmentMonitorMuonSystemMap1D;

class MuonSystemMapPlot1D {
public:
  MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool xy, bool add_1d);

  void fill_x_1d(double residx, double chi2, int dof);
  void fill_x(char charge, double abscissa, double residx, double chi2, int dof);
  void fill_y(char charge, double abscissa, double residy, double chi2, int dof);
  void fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof);
  void fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof);

private:
  std::string m_name;
  int m_bins;
  bool m_xy;
  bool m_1d;

  TH1F *m_x_1d;
  TH2F *m_x_2d, *m_y_2d, *m_dxdz_2d, *m_dydz_2d;
};

#endif // Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
