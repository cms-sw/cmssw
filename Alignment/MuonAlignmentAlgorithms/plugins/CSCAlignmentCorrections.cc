#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCAlignmentCorrections.h"
#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCPairResidualsConstraint.h"

void CSCAlignmentCorrections::plot() {
  edm::Service<TFileService> tFileService;

  for (unsigned int i = 0;  i < m_coefficient.size();  i++) {
    std::string modifiedName = m_fitterName;
    if (modifiedName[0] == 'M'  &&  modifiedName[1] == 'E') {
       if (modifiedName[2] == '-') modifiedName[2] = 'm';
       else if (modifiedName[2] == '+') modifiedName[2] = 'p';
       if (modifiedName[4] == '/') modifiedName[4] = '_';
       if (modifiedName[6] == '/') modifiedName[6] = '_';
    }
    else if (modifiedName[0] == 'Y'  &&  modifiedName[1] == 'E') {
       if (modifiedName[2] == '-') modifiedName[2] = 'm';
       else if (modifiedName[2] == '+') modifiedName[2] = 'p';
    }

    std::stringstream histname, histtitle;
    histname << modifiedName << "_mode_" << i;
    histtitle << m_error[i];

    TH1F *hist = tFileService->make<TH1F>(histname.str().c_str(), histtitle.str().c_str(), m_coefficient[i].size(), 0.5, m_coefficient[i].size() + 0.5);

    bool showed_full_name = false;
    for (unsigned int j = 0;  j < m_coefficient[i].size();  j++) {
      hist->SetBinContent(j+1, m_coefficient[i][j]);

      if (m_modeid[i][j] == -1  ||  !showed_full_name) {
	hist->GetXaxis()->SetBinLabel(j+1, m_modename[i][j].c_str());
      }
      else {
	std::stringstream shortname;
	shortname << m_modename[i][j][7] << m_modename[i][j][8];
	hist->GetXaxis()->SetBinLabel(j+1, shortname.str().c_str());
      }
      if (m_modeid[i][j] != -1) showed_full_name = true;
    }

    th1f_modes.push_back(hist);
  }
}

void CSCAlignmentCorrections::report(std::ofstream &report) {
  report << "cscReports.append(CSCFitterReport(\"" << m_fitterName << "\", " << m_oldchi2 << ", " << m_newchi2 << "))" << std::endl;
  
  for (unsigned int i = 0;  i < m_name.size();  i++) {
    report << "cscReports[-1].addChamberCorrection(\"" << m_name[i] << "\", " << m_id[i].rawId() << ", " << m_value[i] << ")" << std::endl;
  }

  for (unsigned int i = 0;  i < m_coefficient.size();  i++) {
    report << "cscReports[-1].addErrorMode(" << m_error[i] << ")" << std::endl;

    for (unsigned int j = 0;  j < m_coefficient[i].size();  j++) {
      report << "cscReports[-1].addErrorModeTerm(\"" << m_modename[i][j] << "\", " << m_modeid[i][j] << ", " << m_coefficient[i][j] << ")" << std::endl;
    }
  }

  for (unsigned int i = 0;  i < m_i.size();  i++) {
    report << "cscReports[-1].addCSCConstraintResidual(\"" << m_i[i] << "\", \"" << m_j[i] << "\", " << m_before[i] << ", " << m_uncert[i] << ", " << m_residual[i] << ", " << m_pull[i] << ")" << std::endl;
  }

  report << std::endl;
}

void CSCAlignmentCorrections::applyAlignment(AlignableNavigator *alignableNavigator, AlignmentParameterStore *alignmentParameterStore, int mode, bool combineME11) {
  for (unsigned int i = 0;  i < m_name.size();  i++) {
    // CSC sign conventions
    bool backward = ((m_id[i].endcap() == 1  &&  m_id[i].station() >= 3)  ||  (m_id[i].endcap() == 2  &&  m_id[i].station() < 3));

    // get the alignable (or two alignables if in ME1/1)
    const DetId id(m_id[i]);
    Alignable *alignable = alignableNavigator->alignableFromDetId(id).alignable();
    Alignable *also = NULL;
    if (combineME11  &&  m_id[i].station() == 1  &&  m_id[i].ring() == 1) {
      CSCDetId alsoid(m_id[i].endcap(), 1, 4, m_id[i].chamber(), 0);
      const DetId alsoid2(alsoid);
      also = alignableNavigator->alignableFromDetId(alsoid2).alignable();
    }

    AlgebraicVector params(6);
    AlgebraicSymMatrix cov(6);

    if (mode == CSCPairResidualsConstraint::kModePhiy) {
      params[4] = m_value[i];
      cov[4][4] = 1e-6;
    }
    else if (mode == CSCPairResidualsConstraint::kModePhiPos) {
      GlobalPoint center = alignable->surface().toGlobal(LocalPoint(0., 0., 0.));
      double radius = sqrt(center.x()*center.x() + center.y()*center.y());

      double phi_correction = m_value[i];
      params[0] = -radius * sin(phi_correction) * (backward ? -1. : 1.);
      params[1] = radius * (cos(phi_correction) - 1.);
      params[5] = phi_correction * (backward ? -1. : 1.);

      cov[0][0] = 1e-6;
      cov[1][1] = 1e-6;
      cov[5][5] = 1e-6;
    }
    else if (mode == CSCPairResidualsConstraint::kModePhiz) {
      params[5] = m_value[i] * (backward ? -1. : 1.);
      cov[5][5] = 1e-6;
    }
    else assert(false);

    AlignmentParameters *parnew = alignable->alignmentParameters()->cloneFromSelected(params, cov);
    alignable->setAlignmentParameters(parnew);
    alignmentParameterStore->applyParameters(alignable);
    alignable->alignmentParameters()->setValid(true);
    if (also != NULL) {
      AlignmentParameters *parnew2 = also->alignmentParameters()->cloneFromSelected(params, cov);
      also->setAlignmentParameters(parnew2);
      alignmentParameterStore->applyParameters(also);
      also->alignmentParameters()->setValid(true);
    }
  }
}
