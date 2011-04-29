#include "Alignment/CommonAlignmentMonitor/interface/MuonSystemMapPlot1D.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorMuonSystemMap1D.h"
#include "TMath.h"

const double MuonSystemMapPlot1D_xrange = 30.;
const double MuonSystemMapPlot1D_yrange = 50.;
const double MuonSystemMapPlot1D_dxdzrange = 50.;
const double MuonSystemMapPlot1D_dydzrange = 200.;

MuonSystemMapPlot1D::MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool twodimensional)
   : m_name(name), m_bins(bins), m_twodimensional(twodimensional)
{
  m_x_prof = m_y_prof = m_dxdz_prof = m_dydz_prof = m_x_profPos = m_y_profPos = m_dxdz_profPos = m_dydz_profPos = m_x_profNeg = m_y_profNeg = m_dxdz_profNeg = m_dydz_profNeg = NULL;
  m_x_2d = m_y_2d = m_dxdz_2d = m_dydz_2d = m_x_2dweight = m_y_2dweight = m_dxdz_2dweight = m_dydz_2dweight = NULL;
  m_x_hist = m_y_hist = m_dxdz_hist = m_dydz_hist = m_x_weights = m_y_weights = m_dxdz_weights = m_dydz_weights = m_x_valweights = m_y_valweights = m_dxdz_valweights = m_dydz_valweights = NULL;

  std::stringstream name_x_prof, name_y_prof, name_dxdz_prof, name_dydz_prof;
  std::stringstream name_x_profPos, name_y_profPos, name_dxdz_profPos, name_dydz_profPos;
  std::stringstream name_x_profNeg, name_y_profNeg, name_dxdz_profNeg, name_dydz_profNeg;
  std::stringstream name_x_2d, name_y_2d, name_dxdz_2d, name_dydz_2d;
  std::stringstream name_x_2dweight, name_y_2dweight, name_dxdz_2dweight, name_dydz_2dweight;
  std::stringstream name_x_hist, name_y_hist, name_dxdz_hist, name_dydz_hist;
  std::stringstream name_x_weights, name_y_weights, name_dxdz_weights, name_dydz_weights;
  std::stringstream name_x_valweights, name_y_valweights, name_dxdz_valweights, name_dydz_valweights;

  name_x_prof << m_name << "_x_prof";
  name_y_prof << m_name << "_y_prof";
  name_dxdz_prof << m_name << "_dxdz_prof";
  name_dydz_prof << m_name << "_dydz_prof";
  name_x_profPos << m_name << "_x_profPos";
  name_y_profPos << m_name << "_y_profPos";
  name_dxdz_profPos << m_name << "_dxdz_profPos";
  name_dydz_profPos << m_name << "_dydz_profPos";
  name_x_profNeg << m_name << "_x_profNeg";
  name_y_profNeg << m_name << "_y_profNeg";
  name_dxdz_profNeg << m_name << "_dxdz_profNeg";
  name_dydz_profNeg << m_name << "_dydz_profNeg";
  name_x_2d << m_name << "_x_2d";
  name_y_2d << m_name << "_y_2d";
  name_dxdz_2d << m_name << "_dxdz_2d";
  name_dydz_2d << m_name << "_dydz_2d";
  name_x_2dweight << m_name << "_x_2dweight";
  name_y_2dweight << m_name << "_y_2dweight";
  name_dxdz_2dweight << m_name << "_dxdz_2dweight";
  name_dydz_2dweight << m_name << "_dydz_2dweight";
  name_x_hist << m_name << "_x_hist";
  name_y_hist << m_name << "_y_hist";
  name_dxdz_hist << m_name << "_dxdz_hist";
  name_dydz_hist << m_name << "_dydz_hist";
  name_x_weights << m_name << "_x_weights";
  name_y_weights << m_name << "_y_weights";
  name_dxdz_weights << m_name << "_dxdz_weights";
  name_dydz_weights << m_name << "_dydz_weights";
  name_x_valweights << m_name << "_x_valweights";
  name_y_valweights << m_name << "_y_valweights";
  name_dxdz_valweights << m_name << "_dxdz_valweights";
  name_dydz_valweights << m_name << "_dydz_valweights";

  m_x_prof = module->bookProfile("/iterN/", name_x_prof.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_y_prof = module->bookProfile("/iterN/", name_y_prof.str().c_str(), "", m_bins, low, high);
  m_dxdz_prof = module->bookProfile("/iterN/", name_dxdz_prof.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_dydz_prof = module->bookProfile("/iterN/", name_dydz_prof.str().c_str(), "", m_bins, low, high);
  m_x_profPos = module->bookProfile("/iterN/", name_x_profPos.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_y_profPos = module->bookProfile("/iterN/", name_y_profPos.str().c_str(), "", m_bins, low, high);
  m_dxdz_profPos = module->bookProfile("/iterN/", name_dxdz_profPos.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_dydz_profPos = module->bookProfile("/iterN/", name_dydz_profPos.str().c_str(), "", m_bins, low, high);
  m_x_profNeg = module->bookProfile("/iterN/", name_x_profNeg.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_y_profNeg = module->bookProfile("/iterN/", name_y_profNeg.str().c_str(), "", m_bins, low, high);
  m_dxdz_profNeg = module->bookProfile("/iterN/", name_dxdz_profNeg.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_dydz_profNeg = module->bookProfile("/iterN/", name_dydz_profNeg.str().c_str(), "", m_bins, low, high);
  m_x_2d = module->book2D("/iterN/", name_x_2d.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  if (m_twodimensional) m_y_2d = module->book2D("/iterN/", name_y_2d.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  m_dxdz_2d = module->book2D("/iterN/", name_dxdz_2d.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  if (m_twodimensional) m_dydz_2d = module->book2D("/iterN/", name_dydz_2d.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  m_x_2dweight = module->book2D("/iterN/", name_x_2dweight.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  if (m_twodimensional) m_y_2dweight = module->book2D("/iterN/", name_y_2dweight.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  m_dxdz_2dweight = module->book2D("/iterN/", name_dxdz_2dweight.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  if (m_twodimensional) m_dydz_2dweight = module->book2D("/iterN/", name_dydz_2dweight.str().c_str(), "", m_bins, low, high, 80, -40., 40.);
  m_x_hist = module->book1D("/iterN/", name_x_hist.str().c_str(), "", m_bins, -MuonSystemMapPlot1D_xrange, MuonSystemMapPlot1D_xrange);
  if (m_twodimensional) m_y_hist = module->book1D("/iterN/", name_y_hist.str().c_str(), "", m_bins, -MuonSystemMapPlot1D_yrange, MuonSystemMapPlot1D_yrange);
  m_dxdz_hist = module->book1D("/iterN/", name_dxdz_hist.str().c_str(), "", m_bins, -MuonSystemMapPlot1D_dxdzrange, MuonSystemMapPlot1D_dxdzrange);
  if (m_twodimensional) m_dydz_hist = module->book1D("/iterN/", name_dydz_hist.str().c_str(), "", m_bins, -MuonSystemMapPlot1D_dydzrange, MuonSystemMapPlot1D_dydzrange);
  m_x_weights = module->book1D("/iterN/", name_x_weights.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_y_weights = module->book1D("/iterN/", name_y_weights.str().c_str(), "", m_bins, low, high);
  m_dxdz_weights = module->book1D("/iterN/", name_dxdz_weights.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_dydz_weights = module->book1D("/iterN/", name_dydz_weights.str().c_str(), "", m_bins, low, high);
  m_x_valweights = module->book1D("/iterN/", name_x_valweights.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_y_valweights = module->book1D("/iterN/", name_y_valweights.str().c_str(), "", m_bins, low, high);
  m_dxdz_valweights = module->book1D("/iterN/", name_dxdz_valweights.str().c_str(), "", m_bins, low, high);
  if (m_twodimensional) m_dydz_valweights = module->book1D("/iterN/", name_dydz_valweights.str().c_str(), "", m_bins, low, high);

  m_x_prof->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_y_prof->SetAxisRange(-10., 10., "Y");
  m_dxdz_prof->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_dydz_prof->SetAxisRange(-10., 10., "Y");

  m_x_profPos->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_y_profPos->SetAxisRange(-10., 10., "Y");
  m_dxdz_profPos->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_dydz_profPos->SetAxisRange(-10., 10., "Y");

  m_x_profNeg->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_y_profNeg->SetAxisRange(-10., 10., "Y");
  m_dxdz_profNeg->SetAxisRange(-10., 10., "Y");
  if (m_twodimensional) m_dydz_profNeg->SetAxisRange(-10., 10., "Y");
}

void MuonSystemMapPlot1D::fill_x(char charge, double abscissa, double residx, double chi2, int dof) {
   if (chi2 > 0.  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
      double residual = residx * 10.;
      double weight = dof / chi2;

      if (fabs(residual) < MuonSystemMapPlot1D_xrange) {
	 m_x_prof->Fill(abscissa, residual);
	 if (charge > 0) m_x_profPos->Fill(abscissa, residual);
	 else m_x_profNeg->Fill(abscissa, residual);
	 int i = m_x_weights->FindBin(abscissa);
	 m_x_weights->SetBinContent(i, m_x_weights->GetBinContent(i) + weight);
	 m_x_valweights->SetBinContent(i, m_x_valweights->GetBinContent(i) + residual * weight);
      }
      m_x_2d->Fill(abscissa, residual);
      m_x_2dweight->Fill(abscissa, residual, weight);
      m_x_hist->Fill(residual, weight);
   }
}

void MuonSystemMapPlot1D::fill_y(char charge, double abscissa, double residy, double chi2, int dof) {
   if (m_twodimensional  &&  chi2 > 0.  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
      double residual = residy * 10.;
      double weight = dof / chi2;
      
      if (fabs(residual) < MuonSystemMapPlot1D_yrange) {
	 m_y_prof->Fill(abscissa, residual);
	 if (charge > 0) m_y_profPos->Fill(abscissa, residual);
	 else m_y_profNeg->Fill(abscissa, residual);
	 int i = m_y_weights->FindBin(abscissa);
	 m_y_weights->SetBinContent(i, m_y_weights->GetBinContent(i) + weight);
	 m_y_valweights->SetBinContent(i, m_y_valweights->GetBinContent(i) + residual * weight);
      }
      m_y_2d->Fill(abscissa, residual);
      m_y_2dweight->Fill(abscissa, residual, weight);
      m_y_hist->Fill(residual, weight);
   }
}

void MuonSystemMapPlot1D::fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof) {
   if (chi2 > 0.  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
      double residual = slopex * 1000.;
      double weight = dof / chi2;
      
      if (fabs(residual) < MuonSystemMapPlot1D_dxdzrange) {
	 m_dxdz_prof->Fill(abscissa, residual);
	 if (charge > 0) m_dxdz_profPos->Fill(abscissa, residual);
	 else m_dxdz_profNeg->Fill(abscissa, residual);
	 int i = m_dxdz_weights->FindBin(abscissa);
	 m_dxdz_weights->SetBinContent(i, m_dxdz_weights->GetBinContent(i) + weight);
	 m_dxdz_valweights->SetBinContent(i, m_dxdz_valweights->GetBinContent(i) + residual * weight);
      }
      m_dxdz_2d->Fill(abscissa, residual);
      m_dxdz_2dweight->Fill(abscissa, residual, weight);
      m_dxdz_hist->Fill(residual, weight);
   }
}

void MuonSystemMapPlot1D::fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof) {
   if (m_twodimensional  &&  chi2 > 0.  &&  TMath::Prob(chi2, dof) < 0.95) {  // no spikes allowed
      double residual = slopey * 1000.;
      double weight = dof / chi2;
      
      if (fabs(residual) < MuonSystemMapPlot1D_dydzrange) {
	 m_dydz_prof->Fill(abscissa, residual);
	 if (charge > 0) m_dydz_profPos->Fill(abscissa, residual);
	 else m_dydz_profNeg->Fill(abscissa, residual);
	 int i = m_dydz_weights->FindBin(abscissa);
	 m_dydz_weights->SetBinContent(i, m_dydz_weights->GetBinContent(i) + weight);
	 m_dydz_valweights->SetBinContent(i, m_dydz_valweights->GetBinContent(i) + residual * weight);
      }
      m_dydz_2d->Fill(abscissa, residual);
      m_dydz_2dweight->Fill(abscissa, residual, weight);
      m_dydz_hist->Fill(residual, weight);
   }
}
