// utility class for bulk-handling mapplot hitstograms
// $Id$

#include "Alignment/CommonAlignmentMonitor/interface/MuonSystemMapPlot1D.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorMuonSystemMap1D.h"
#include "TMath.h"

MuonSystemMapPlot1D::MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool xy, bool add_1d)
   : m_name(name), m_bins(bins), m_xy(xy), m_1d(add_1d)
{
  m_x_2d = m_y_2d = m_dxdz_2d = m_dydz_2d = NULL;
  std::stringstream name_x_2d, name_y_2d, name_dxdz_2d, name_dydz_2d;
  name_x_2d << m_name << "_x_2d";
  name_y_2d << m_name << "_y_2d";
  name_dxdz_2d << m_name << "_dxdz_2d";
  name_dydz_2d << m_name << "_dydz_2d";

  const int nbins = 200;
  const double window = 100.;

  m_x_2d = module->book2D("/iterN/", name_x_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  if (m_xy) m_y_2d = module->book2D("/iterN/", name_y_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  m_dxdz_2d = module->book2D("/iterN/", name_dxdz_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  if (m_xy) m_dydz_2d = module->book2D("/iterN/", name_dydz_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);

  m_x_1d = NULL;//m_y_1d = m_dxdz_1d = m_dydz_1d = NULL;
  if (m_1d) {
    std::stringstream name_x_1d;//, name_y_1d, name_dxdz_1d, name_dydz_1d;
    name_x_1d << m_name << "_x_1d";
    //name_y_1d << m_name << "_y_2d";
    //name_dxdz_1d << m_name << "_dxdz_1d";
    //name_dydz_1d << m_name << "_dydz_1d";

    m_x_1d = module->book1D("/iterN/", name_x_1d.str().c_str(), "", nbins, -window, window);
    //if (m_xy) m_y_1d = module->book1D("/iterN/", name_y_1d.str().c_str(), "", nbins, -window, window);
    //m_dxdz_1d = module->book1D("/iterN/", name_dxdz_1d.str().c_str(), "", nbins, -window, window);
    //if (m_xy) m_dydz_1d = module->book1D("/iterN/", name_dydz_1d.str().c_str(), "", -window, window);
  }
}

void MuonSystemMapPlot1D::fill_x_1d(double residx, double chi2, int dof)
{
  if (m_1d && chi2 > 0.) {
  //  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
    // assume that residx was in radians
    double residual = residx * 1000.;
    m_x_1d->Fill(residual);
  }
}

void MuonSystemMapPlot1D::fill_x(char charge, double abscissa, double residx, double chi2, int dof)
{
  if (chi2 > 0.) {
  // &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
    double residual = residx * 10.;
    //double weight = dof / chi2;
    m_x_2d->Fill(abscissa, residual);
  }
}

void MuonSystemMapPlot1D::fill_y(char charge, double abscissa, double residy, double chi2, int dof)
{
  if (m_xy  &&  chi2 > 0.) {
  // &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
    double residual = residy * 10.;
    //double weight = dof / chi2;
    m_y_2d->Fill(abscissa, residual);
  }
}

void MuonSystemMapPlot1D::fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof)
{
  if (chi2 > 0.) {
  //  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
    double residual = slopex * 1000.;
    //double weight = dof / chi2;
    m_dxdz_2d->Fill(abscissa, residual);
  }
}

void MuonSystemMapPlot1D::fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof)
{
  if (m_xy  &&  chi2 > 0.) {
  //  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
    double residual = slopey * 1000.;
    //double weight = dof / chi2;
    m_dydz_2d->Fill(abscissa, residual);
  }
}
