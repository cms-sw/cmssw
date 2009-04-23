#include "Alignment/CommonAlignmentMonitor/interface/MuonSystemMapPlot1D.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorMuonSystemMap1D.h"

MuonSystemMapPlot1D::MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, int minHits)
  : m_name(name), m_bins(bins)
{
  std::stringstream name_x, name_y, name_dxdz, name_dydz;
  std::stringstream name_x_anti, name_y_anti, name_dxdz_anti, name_dydz_anti;
  std::stringstream name_x_prof, name_y_prof, name_dxdz_prof, name_dydz_prof;
  std::stringstream name_x_hist, name_y_hist, name_dxdz_hist, name_dydz_hist;

  name_x << m_name << "_x";
  name_y << m_name << "_y";
  name_dxdz << m_name << "_dxdz";
  name_dydz << m_name << "_dydz";
  name_x_anti << m_name << "_x_anti";
  name_y_anti << m_name << "_y_anti";
  name_dxdz_anti << m_name << "_dxdz_anti";
  name_dydz_anti << m_name << "_dydz_anti";
  name_x_prof << m_name << "_x_prof";
  name_y_prof << m_name << "_y_prof";
  name_dxdz_prof << m_name << "_dxdz_prof";
  name_dydz_prof << m_name << "_dydz_prof";
  name_x_hist << m_name << "_x_hist";
  name_y_hist << m_name << "_y_hist";
  name_dxdz_hist << m_name << "_dxdz_hist";
  name_dydz_hist << m_name << "_dydz_hist";

  m_x = module->book1D("/iterN/", name_x.str().c_str(), "", m_bins, low, high);
  m_y = module->book1D("/iterN/", name_y.str().c_str(), "", m_bins, low, high);
  m_dxdz = module->book1D("/iterN/", name_dxdz.str().c_str(), "", m_bins, low, high);
  m_dydz = module->book1D("/iterN/", name_dydz.str().c_str(), "", m_bins, low, high);
  m_x_anti = module->book1D("/iterN/", name_x_anti.str().c_str(), "", m_bins, low, high);
  m_y_anti = module->book1D("/iterN/", name_y_anti.str().c_str(), "", m_bins, low, high);
  m_dxdz_anti = module->book1D("/iterN/", name_dxdz_anti.str().c_str(), "", m_bins, low, high);
  m_dydz_anti = module->book1D("/iterN/", name_dydz_anti.str().c_str(), "", m_bins, low, high);
  m_x_prof = module->bookProfile("/iterN/", name_x_prof.str().c_str(), "", m_bins, low, high);
  m_y_prof = module->bookProfile("/iterN/", name_y_prof.str().c_str(), "", m_bins, low, high);
  m_dxdz_prof = module->bookProfile("/iterN/", name_dxdz_prof.str().c_str(), "", m_bins, low, high);
  m_dydz_prof = module->bookProfile("/iterN/", name_dydz_prof.str().c_str(), "", m_bins, low, high);
  m_x_hist = module->book1D("/iterN/", name_x_hist.str().c_str(), "", m_bins, -100., 100.);
  m_y_hist = module->book1D("/iterN/", name_y_hist.str().c_str(), "", m_bins, -100., 100.);
  m_dxdz_hist = module->book1D("/iterN/", name_dxdz_hist.str().c_str(), "", m_bins, -100., 100.);
  m_dydz_hist = module->book1D("/iterN/", name_dydz_hist.str().c_str(), "", m_bins, -100., 100.);

  m_x->SetAxisRange(-10., 10., "Y");
  m_y->SetAxisRange(-10., 10., "Y");
  m_dxdz->SetAxisRange(-10., 10., "Y");
  m_dydz->SetAxisRange(-10., 10., "Y");
  m_x_anti->SetAxisRange(-10., 10., "Y");
  m_y_anti->SetAxisRange(-10., 10., "Y");
  m_dxdz_anti->SetAxisRange(-10., 10., "Y");
  m_dydz_anti->SetAxisRange(-10., 10., "Y");
  m_x_prof->SetAxisRange(-10., 10., "Y");
  m_y_prof->SetAxisRange(-10., 10., "Y");
  m_dxdz_prof->SetAxisRange(-10., 10., "Y");
  m_dydz_prof->SetAxisRange(-10., 10., "Y");

  for (int i = 0;  i < m_bins;  i++) {
    MuonResidualsTwoBin *fitter = NULL;

    fitter = new MuonResidualsTwoBin(true, new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits), new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits));
    fitter->setPrintLevel(-1);
    fitter->setStrategy(0);
    m_x_fitter.push_back(fitter);

    fitter = new MuonResidualsTwoBin(true, new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits), new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits));
    fitter->setPrintLevel(-1);
    fitter->setStrategy(0);
    m_y_fitter.push_back(fitter);

    fitter = new MuonResidualsTwoBin(true, new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits), new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits));
    fitter->setPrintLevel(-1);
    fitter->setStrategy(0);
    m_dxdz_fitter.push_back(fitter);

    fitter = new MuonResidualsTwoBin(true, new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits), new MuonResiduals1DOFFitter(MuonResidualsFitter::kROOTVoigt, minHits));
    fitter->setPrintLevel(-1);
    fitter->setStrategy(0);
    m_dydz_fitter.push_back(fitter);
  }
}

void MuonSystemMapPlot1D::fill_x(char charge, double abscissa, double residx, double redchi2) {
  int i = m_x->FindBin(abscissa);
  if (i < 1  ||  i > m_bins) return;
  i -= 1;

  double *resblock = new double[MuonResiduals1DOFFitter::kNData];
  resblock[MuonResiduals1DOFFitter::kResid] = residx;
  resblock[MuonResiduals1DOFFitter::kRedChi2] = redchi2;
  m_x_fitter[i]->fill(charge, resblock);
}

void MuonSystemMapPlot1D::fill_y(char charge, double abscissa, double residy, double redchi2) {
  int i = m_y->FindBin(abscissa);
  if (i < 1  ||  i > m_bins) return;
  i -= 1;

  double *resblock = new double[MuonResiduals1DOFFitter::kNData];
  resblock[MuonResiduals1DOFFitter::kResid] = residy;
  resblock[MuonResiduals1DOFFitter::kRedChi2] = redchi2;
  m_y_fitter[i]->fill(charge, resblock);
}

void MuonSystemMapPlot1D::fill_dxdz(char charge, double abscissa, double slopex, double redchi2) {
  int i = m_dxdz->FindBin(abscissa);
  if (i < 1  ||  i > m_bins) return;
  i -= 1;

  double *resblock = new double[MuonResiduals1DOFFitter::kNData];
  resblock[MuonResiduals1DOFFitter::kResid] = slopex;
  resblock[MuonResiduals1DOFFitter::kRedChi2] = redchi2;
  m_dxdz_fitter[i]->fill(charge, resblock);
}

void MuonSystemMapPlot1D::fill_dydz(char charge, double abscissa, double slopey, double redchi2) {
  int i = m_dydz->FindBin(abscissa);
  if (i < 1  ||  i > m_bins) return;
  i -= 1;

  double *resblock = new double[MuonResiduals1DOFFitter::kNData];
  resblock[MuonResiduals1DOFFitter::kResid] = slopey;
  resblock[MuonResiduals1DOFFitter::kRedChi2] = redchi2;
  m_dydz_fitter[i]->fill(charge, resblock);
}

void MuonSystemMapPlot1D::fit() {
  for (int i = 0;  i < m_bins;  i++) {
    if (m_x_fitter[i]->fit(NULL)) {
      m_x->SetBinContent(i+1, 10.*m_x_fitter[i]->value(MuonResiduals1DOFFitter::kResid));
      m_x->SetBinError(i+1, 10.*m_x_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
      m_x_anti->SetBinContent(i+1, 10.*m_x_fitter[i]->antisym(MuonResiduals1DOFFitter::kResid));
      m_x_anti->SetBinError(i+1, 10.*m_x_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
    }
    else {
      m_x->SetBinContent(i+1, 20000.);
      m_x->SetBinError(i+1, 10000.);
      m_x_anti->SetBinContent(i+1, 20000.);
      m_x_anti->SetBinError(i+1, 10000.);
    }

    if (m_y_fitter[i]->fit(NULL)) {
      m_y->SetBinContent(i+1, 10.*m_y_fitter[i]->value(MuonResiduals1DOFFitter::kResid));
      m_y->SetBinError(i+1, 10.*m_y_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
      m_y_anti->SetBinContent(i+1, 10.*m_y_fitter[i]->antisym(MuonResiduals1DOFFitter::kResid));
      m_y_anti->SetBinError(i+1, 10.*m_y_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
    }
    else {
      m_y->SetBinContent(i+1, 20000.);
      m_y->SetBinError(i+1, 10000.);
      m_y_anti->SetBinContent(i+1, 20000.);
      m_y_anti->SetBinError(i+1, 10000.);
    }

    if (m_dxdz_fitter[i]->fit(NULL)) {
      m_dxdz->SetBinContent(i+1, 1000.*m_dxdz_fitter[i]->value(MuonResiduals1DOFFitter::kResid));
      m_dxdz->SetBinError(i+1, 1000.*m_dxdz_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
      m_dxdz_anti->SetBinContent(i+1, 1000.*m_dxdz_fitter[i]->antisym(MuonResiduals1DOFFitter::kResid));
      m_dxdz_anti->SetBinError(i+1, 1000.*m_dxdz_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
    }
    else {
      m_dxdz->SetBinContent(i+1, 20000.);
      m_dxdz->SetBinError(i+1, 10000.);
      m_dxdz_anti->SetBinContent(i+1, 20000.);
      m_dxdz_anti->SetBinError(i+1, 10000.);
    }

    if (m_dydz_fitter[i]->fit(NULL)) {
      m_dydz->SetBinContent(i+1, 1000.*m_dydz_fitter[i]->value(MuonResiduals1DOFFitter::kResid));
      m_dydz->SetBinError(i+1, 1000.*m_dydz_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
      m_dydz_anti->SetBinContent(i+1, 1000.*m_dydz_fitter[i]->antisym(MuonResiduals1DOFFitter::kResid));
      m_dydz_anti->SetBinError(i+1, 1000.*m_dydz_fitter[i]->error(MuonResiduals1DOFFitter::kResid));
    }
    else {
      m_dydz->SetBinContent(i+1, 20000.);
      m_dydz->SetBinError(i+1, 10000.);
      m_dydz_anti->SetBinContent(i+1, 20000.);
      m_dydz_anti->SetBinError(i+1, 10000.);
    }
  }
}

void MuonSystemMapPlot1D::fill_profs() {
  for (int i = 0;  i < m_bins;  i++) {
    for (std::vector<double*>::const_iterator resblock = m_x_fitter[i]->residualsPos_begin();  resblock != m_x_fitter[i]->residualsPos_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_x_prof->Fill(m_x_prof->GetBinCenter(i), 10.*resid, 1./redchi2);
      m_x_hist->Fill(10.*resid, 1./redchi2);
    }
    for (std::vector<double*>::const_iterator resblock = m_x_fitter[i]->residualsNeg_begin();  resblock != m_x_fitter[i]->residualsNeg_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_x_prof->Fill(m_x_prof->GetBinCenter(i), 10.*resid, 1./redchi2);
      m_x_hist->Fill(10.*resid, 1./redchi2);
    }
  }

  for (int i = 0;  i < m_bins;  i++) {
    for (std::vector<double*>::const_iterator resblock = m_y_fitter[i]->residualsPos_begin();  resblock != m_y_fitter[i]->residualsPos_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_y_prof->Fill(m_y_prof->GetBinCenter(i), 10.*resid, 1./redchi2);
      m_y_hist->Fill(10.*resid, 1./redchi2);
    }
    for (std::vector<double*>::const_iterator resblock = m_y_fitter[i]->residualsNeg_begin();  resblock != m_y_fitter[i]->residualsNeg_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_y_prof->Fill(m_y_prof->GetBinCenter(i), 10.*resid, 1./redchi2);
      m_y_hist->Fill(10.*resid, 1./redchi2);
    }
  }

  for (int i = 0;  i < m_bins;  i++) {
    for (std::vector<double*>::const_iterator resblock = m_dxdz_fitter[i]->residualsPos_begin();  resblock != m_dxdz_fitter[i]->residualsPos_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_dxdz_prof->Fill(m_dxdz_prof->GetBinCenter(i), 1000.*resid, 1./redchi2);
      m_dxdz_hist->Fill(1000.*resid, 1./redchi2);
    }
    for (std::vector<double*>::const_iterator resblock = m_dxdz_fitter[i]->residualsNeg_begin();  resblock != m_dxdz_fitter[i]->residualsNeg_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_dxdz_prof->Fill(m_dxdz_prof->GetBinCenter(i), 1000.*resid, 1./redchi2);
      m_dxdz_hist->Fill(1000.*resid, 1./redchi2);
    }
  }

  for (int i = 0;  i < m_bins;  i++) {
    for (std::vector<double*>::const_iterator resblock = m_dydz_fitter[i]->residualsPos_begin();  resblock != m_dydz_fitter[i]->residualsPos_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_dydz_prof->Fill(m_dydz_prof->GetBinCenter(i), 1000.*resid, 1./redchi2);
      m_dydz_hist->Fill(1000.*resid, 1./redchi2);
    }
    for (std::vector<double*>::const_iterator resblock = m_dydz_fitter[i]->residualsNeg_begin();  resblock != m_dydz_fitter[i]->residualsNeg_end();  ++resblock) {
      double resid = (*resblock)[MuonResiduals1DOFFitter::kResid];
      double redchi2 = (*resblock)[MuonResiduals1DOFFitter::kRedChi2];
      m_dydz_prof->Fill(m_dydz_prof->GetBinCenter(i), 1000.*resid, 1./redchi2);
      m_dydz_hist->Fill(1000.*resid, 1./redchi2);
    }
  }
}

void MuonSystemMapPlot1D::write(FILE *file, int &which) {
  for (int i = 0;  i < m_bins;  i++) {
    m_x_fitter[i]->write(file, which);
    which++;

    m_y_fitter[i]->write(file, which);
    which++;

    m_dxdz_fitter[i]->write(file, which);
    which++;

    m_dydz_fitter[i]->write(file, which);
    which++;
  }
}

void MuonSystemMapPlot1D::read(FILE *file, int &which) {
  for (int i = 0;  i < m_bins;  i++) {
    m_x_fitter[i]->read(file, which);
    which++;

    m_y_fitter[i]->read(file, which);
    which++;

    m_dxdz_fitter[i]->read(file, which);
    which++;

    m_dydz_fitter[i]->read(file, which);
    which++;
  }
}
