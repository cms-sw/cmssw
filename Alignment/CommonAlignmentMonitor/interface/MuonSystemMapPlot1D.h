#ifndef Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
#define Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H

/** \class MuonSystemMapPlot1D
 *  $Date: 2009/08/29 18:18:07 $
 *  $Revision: 1.2 $
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
  MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool twodimensional);

  void fill_x(char charge, double abscissa, double residx, double chi2, int dof);
  void fill_y(char charge, double abscissa, double residy, double chi2, int dof);
  void fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof);
  void fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof);

private:
  std::string m_name;
  int m_bins;
  bool m_twodimensional;

  TProfile *m_x_prof, *m_y_prof, *m_dxdz_prof, *m_dydz_prof;
  TProfile *m_x_profPos, *m_y_profPos, *m_dxdz_profPos, *m_dydz_profPos;
  TProfile *m_x_profNeg, *m_y_profNeg, *m_dxdz_profNeg, *m_dydz_profNeg;
  TH2F *m_x_2d, *m_y_2d, *m_dxdz_2d, *m_dydz_2d;
  TH2F *m_x_2dweight, *m_y_2dweight, *m_dxdz_2dweight, *m_dydz_2dweight;
  TH1F *m_x_hist, *m_y_hist, *m_dxdz_hist, *m_dydz_hist;
  TH1F *m_x_weights, *m_y_weights, *m_dxdz_weights, *m_dydz_weights;
  TH1F *m_x_valweights, *m_y_valweights, *m_dxdz_valweights, *m_dydz_valweights;
};

#endif // Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
