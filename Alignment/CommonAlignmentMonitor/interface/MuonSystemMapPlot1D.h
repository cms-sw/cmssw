#ifndef Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
#define Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H

/** \class MuonSystemMapPlot1D
 *  $Date: Fri Apr 17 18:08:24 CDT 2009 $
 *  $Revision: 1.0 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals1DOFFitter.h"

#include "TH1F.h"
#include "TProfile.h"

#include <string>
#include <sstream>

class AlignmentMonitorMuonSystemMap1D;

class MuonSystemMapPlot1D {
public:
  MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, int minHits);

  void fill_x(char charge, double abscissa, double residx, double redchi2);
  void fill_y(char charge, double abscissa, double residy, double redchi2);
  void fill_dxdz(char charge, double abscissa, double slopex, double redchi2);
  void fill_dydz(char charge, double abscissa, double slopey, double redchi2);

  void fit();
  void fill_profs();

  void write(FILE *file, int &which);
  void read(FILE *file, int &which);

private:
  std::string m_name;
  int m_bins;

  TH1F *m_x, *m_y, *m_dxdz, *m_dydz;
  TH1F *m_x_anti, *m_y_anti, *m_dxdz_anti, *m_dydz_anti;
  TProfile *m_x_prof, *m_y_prof, *m_dxdz_prof, *m_dydz_prof;
  TH1F *m_x_hist, *m_y_hist, *m_dxdz_hist, *m_dydz_hist;

  std::vector<MuonResidualsTwoBin*> m_x_fitter, m_y_fitter, m_dxdz_fitter, m_dydz_fitter;
};

#endif // Alignment_CommonAlignmentMonitor_MuonSystemMapPlot1D_H
