#include "HLTriggerOffline/Btag/interface/HLTBTagHarvestingAnalyzer.h"

HLTBTagHarvestingAnalyzer::HLTBTagHarvestingAnalyzer(const edm::ParameterSet &iConfig) {
  // getParameter
  mainFolder_ = iConfig.getParameter<std::string>("mainFolder");
  hltPathNames_ = iConfig.getParameter<std::vector<std::string>>("HLTPathNames");
  edm::ParameterSet mc = iConfig.getParameter<edm::ParameterSet>("mcFlavours");
  m_mcLabels = mc.getParameterNamesForType<std::vector<unsigned int>>();
  m_histoName = iConfig.getParameter<std::vector<std::string>>("histoName");
  m_minTag = iConfig.getParameter<double>("minTag");

  HCALSpecialsNames[HEP17] = "HEP17";
  HCALSpecialsNames[HEP18] = "HEP18";
  HCALSpecialsNames[HEM17] = "HEM17";
}

HLTBTagHarvestingAnalyzer::~HLTBTagHarvestingAnalyzer() {}

// ------------ method called once each job just after ending the event loop
// ------------
void HLTBTagHarvestingAnalyzer::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  using namespace edm;
  Exception excp(errors::LogicError);
  std::string dqmFolder_hist;

  // for each hltPath and for each flavour, do the "b-tag efficiency vs jet pt"
  // and "b-tag efficiency vs mistag rate" plots
  for (unsigned int ind = 0; ind < hltPathNames_.size(); ind++) {
    dqmFolder_hist = Form("%s/Discriminator/%s", mainFolder_.c_str(), hltPathNames_[ind].c_str());
    std::string effDir = Form("%s/Discriminator/%s/efficiency", mainFolder_.c_str(), hltPathNames_[ind].c_str());
    std::string relationsDir = Form("%s/Discriminator/%s/HEP17_HEM17", mainFolder_.c_str(), hltPathNames_[ind].c_str());
    ibooker.setCurrentFolder(effDir);
    TH1 *den = nullptr;
    TH1 *num = nullptr;
    std::map<std::string, TH1F> effics;
    std::map<std::string, bool> efficsOK;
    std::map<std::string, std::map<HCALSpecials, TH1F>> efficsmod;
    std::map<std::string, std::map<HCALSpecials, bool>> efficsmodOK;
    for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
      bool isOK = false;
      std::string label = m_histoName.at(ind) + "__";  //"JetTag__";
      std::string flavour = m_mcLabels[i];
      label += flavour;
      isOK =
          GetNumDenumerators(ibooker, igetter, dqmFolder_hist + "/" + label, dqmFolder_hist + "/" + label, num, den, 0);
      if (isOK) {
        // do the 'b-tag efficiency vs discr' plot
        effics[flavour] = calculateEfficiency1D(ibooker, igetter, *num, *den, label + "_efficiency_vs_disc");
        efficsOK[flavour] = isOK;
      }
      // for modules (HEP17 etc.)
      for (const auto& j : HCALSpecialsNames) {
        ibooker.setCurrentFolder(dqmFolder_hist + "/" + j.second + "/efficiency");
        isOK = GetNumDenumerators(ibooker,
                                  igetter,
                                  dqmFolder_hist + "/" + j.second + "/" + label,
                                  dqmFolder_hist + "/" + j.second + "/" + label,
                                  num,
                                  den,
                                  0);
        if (isOK) {
          // do the 'b-tag efficiency vs discr' plot
          efficsmod[flavour][j.first] =
              calculateEfficiency1D(ibooker, igetter, *num, *den, label + "_efficiency_vs_disc");
          efficsmodOK[flavour][j.first] = isOK;
        }
      }
      ibooker.setCurrentFolder(effDir);
      label = m_histoName.at(ind) + "___";
      std::string labelEta = label;
      std::string labelPhi = label;
      label += flavour + "_disc_pT";
      labelEta += flavour + "_disc_eta";
      labelPhi += flavour + "_disc_phi";
      isOK =
          GetNumDenumerators(ibooker, igetter, dqmFolder_hist + "/" + label, dqmFolder_hist + "/" + label, num, den, 1);
      if (isOK) {
        // do the 'b-tag efficiency vs pT' plot
        TH1F eff = calculateEfficiency1D(ibooker, igetter, *num, *den, label + "_efficiency_vs_pT");
      }
      isOK = GetNumDenumerators(
          ibooker, igetter, dqmFolder_hist + "/" + labelEta, dqmFolder_hist + "/" + labelEta, num, den, 2);
      if (isOK) {
        // do the 'b-tag efficiency vs Eta' plot
        TH1F eff = calculateEfficiency1D(ibooker, igetter, *num, *den, labelEta + "_efficiency_vs_eta");
      }
      isOK = GetNumDenumerators(
          ibooker, igetter, dqmFolder_hist + "/" + labelPhi, dqmFolder_hist + "/" + labelPhi, num, den, 2);
      if (isOK) {
        // do the 'b-tag efficiency vs Phi' plot
        TH1F eff = calculateEfficiency1D(ibooker, igetter, *num, *den, labelPhi + "_efficiency_vs_phi");
      }

      /// save efficiency_vs_disc_HEP17 / efficiency_vs_disc_HEM17 plots
      ibooker.setCurrentFolder(relationsDir);
      if (efficsmodOK[flavour][HEP17] && efficsmodOK[flavour][HEM17])
        modulesrate(ibooker,
                    igetter,
                    &efficsmod[flavour][HEP17],
                    &efficsmod[flavour][HEM17],
                    m_histoName.at(ind) + "_" + flavour + "_HEP17_HEM17_effs_vs_disc_rate");
      ibooker.setCurrentFolder(effDir);

    }  /// for mc labels

    /// save mistagrate vs b-eff plots
    if (efficsOK["b"] && efficsOK["c"])
      mistagrate(ibooker, igetter, &effics["b"], &effics["c"], m_histoName.at(ind) + "_b_c_mistagrate");
    if (efficsOK["b"] && efficsOK["light"])
      mistagrate(ibooker, igetter, &effics["b"], &effics["light"], m_histoName.at(ind) + "_b_light_mistagrate");
    if (efficsOK["b"] && efficsOK["g"])
      mistagrate(ibooker, igetter, &effics["b"], &effics["g"], m_histoName.at(ind) + "_b_g_mistagrate");

    /// save mistagrate vs b-eff plots for modules (HEP17 etc.)
    for (const auto& j : HCALSpecialsNames) {
      ibooker.setCurrentFolder(dqmFolder_hist + "/" + j.second + "/efficiency");
      if (efficsmodOK["b"][j.first] && efficsmodOK["c"][j.first])
        mistagrate(ibooker,
                   igetter,
                   &efficsmod["b"][j.first],
                   &efficsmod["c"][j.first],
                   m_histoName.at(ind) + "_b_c_mistagrate");
      if (efficsmodOK["b"][j.first] && efficsmodOK["light"][j.first])
        mistagrate(ibooker,
                   igetter,
                   &efficsmod["b"][j.first],
                   &efficsmod["light"][j.first],
                   m_histoName.at(ind) + "_b_light_mistagrate");
      if (efficsmodOK["b"][j.first] && efficsmodOK["g"][j.first])
        mistagrate(ibooker,
                   igetter,
                   &efficsmod["b"][j.first],
                   &efficsmod["g"][j.first],
                   m_histoName.at(ind) + "_b_g_mistagrate");
    }

    /// save mistagrate_HEP17 / mistagrate_HEM17 plots
    ibooker.setCurrentFolder(relationsDir);
    bool isOK = false;
    isOK = GetNumDenumerators(ibooker,
                              igetter,
                              dqmFolder_hist + "/HEP17/efficiency/" + m_histoName.at(ind) + "_b_c_mistagrate",
                              dqmFolder_hist + "/HEM17/efficiency/" + m_histoName.at(ind) + "_b_c_mistagrate",
                              num,
                              den,
                              3);
    if (isOK)
      modulesrate(ibooker, igetter, (TH1F *)num, (TH1F *)den, m_histoName.at(ind) + "_HEP17_HEM17_b_c_mistagrate");
    isOK = GetNumDenumerators(ibooker,
                              igetter,
                              dqmFolder_hist + "/HEP17/efficiency/" + m_histoName.at(ind) + "_b_light_mistagrate",
                              dqmFolder_hist + "/HEM17/efficiency/" + m_histoName.at(ind) + "_b_light_mistagrate",
                              num,
                              den,
                              3);
    if (isOK)
      modulesrate(ibooker, igetter, (TH1F *)num, (TH1F *)den, m_histoName.at(ind) + "_HEP17_HEM17_b_light_mistagrate");
    isOK = GetNumDenumerators(ibooker,
                              igetter,
                              dqmFolder_hist + "/HEP17/efficiency/" + m_histoName.at(ind) + "_b_g_mistagrate",
                              dqmFolder_hist + "/HEM17/efficiency/" + m_histoName.at(ind) + "_b_g_mistagrate",
                              num,
                              den,
                              3);
    if (isOK)
      modulesrate(ibooker, igetter, (TH1F *)num, (TH1F *)den, m_histoName.at(ind) + "_HEP17_HEM17_b_g_mistagrate");
  }  /// for triggers
}

bool HLTBTagHarvestingAnalyzer::GetNumDenumerators(DQMStore::IBooker &ibooker,
                                                   DQMStore::IGetter &igetter,
                                                   std::string num,
                                                   std::string den,
                                                   TH1 *&ptrnum,
                                                   TH1 *&ptrden,
                                                   int type) {
  using namespace edm;
  /*
     possible types:
     type =0 for eff_vs_discriminator
     type =1 for eff_vs_pT
     type =2 for eff_vs_eta or eff_vs_phi
     type =3 for HEP17 / HEM17 mistagrate relation
   */
  MonitorElement *denME = nullptr;
  MonitorElement *numME = nullptr;
  denME = igetter.get(den);
  numME = igetter.get(num);
  Exception excp(errors::LogicError);

  if (denME == nullptr || numME == nullptr) {
    excp << "Plots not found:\n";
    if (denME == nullptr)
      excp << den << "\n";
    if (numME == nullptr)
      excp << num << "\n";
    excp.raise();
  }

  if (type == 0)  // efficiency_vs_discr: fill "ptrnum" with the cumulative function of
                  // the DQM plots contained in "num" and "ptrden" with a flat function
  {
    TH1 *numH1 = numME->getTH1();
    TH1 *denH1 = denME->getTH1();
    ptrden = (TH1 *)denH1->Clone("denominator");
    ptrnum = (TH1 *)numH1->Clone("numerator");

    ptrnum->SetBinContent(1, numH1->Integral());
    ptrden->SetBinContent(1, numH1->Integral());
    for (int j = 2; j <= numH1->GetNbinsX(); j++) {
      ptrnum->SetBinContent(j, numH1->Integral() - numH1->Integral(1, j - 1));
      ptrden->SetBinContent(j, numH1->Integral());
    }
  }

  if (type == 1)  // efficiency_vs_pT: fill "ptrden" with projection of the plots
                  // contained in "den" and fill "ptrnum" with projection of the
                  // plots contained in "num", having btag>m_minTag
  {
    TH2F *numH2 = numME->getTH2F();
    TH2F *denH2 = denME->getTH2F();

    /// numerator preparing
    TCutG *cutg_num = new TCutG("cutg_num", 4);
    cutg_num->SetPoint(0, m_minTag, 0);
    cutg_num->SetPoint(1, m_minTag, 9999);
    cutg_num->SetPoint(2, 1.1, 9999);
    cutg_num->SetPoint(3, 1.1, 0);
    ptrnum = numH2->ProjectionY("numerator", 0, -1, "[cutg_num]");

    /// denominator preparing
    TCutG *cutg_den = new TCutG("cutg_den", 4);
    cutg_den->SetPoint(0, -10.1, 0);
    cutg_den->SetPoint(1, -10.1, 9999);
    cutg_den->SetPoint(2, 1.1, 9999);
    cutg_den->SetPoint(3, 1.1, 0);
    ptrden = denH2->ProjectionY("denumerator", 0, -1, "[cutg_den]");
    delete cutg_num;
    delete cutg_den;
  }

  if (type == 2)  // efficiency_vs_eta: fill "ptrden" with projection of the
                  // plots contained in "den" and fill "ptrnum" with projection
                  // of the plots contained in "num", having btag>m_minTag
  {
    TH2F *numH2 = numME->getTH2F();
    TH2F *denH2 = denME->getTH2F();

    /// numerator preparing
    TCutG *cutg_num = new TCutG("cutg_num", 4);
    cutg_num->SetPoint(0, m_minTag, -10);
    cutg_num->SetPoint(1, m_minTag, 10);
    cutg_num->SetPoint(2, 1.1, 10);
    cutg_num->SetPoint(3, 1.1, -10);
    ptrnum = numH2->ProjectionY("numerator", 0, -1, "[cutg_num]");

    /// denominator preparing
    TCutG *cutg_den = new TCutG("cutg_den", 4);
    cutg_den->SetPoint(0, -10.1, -10);
    cutg_den->SetPoint(1, -10.1, 10);
    cutg_den->SetPoint(2, 1.1, 10);
    cutg_den->SetPoint(3, 1.1, -10);
    ptrden = denH2->ProjectionY("denumerator", 0, -1, "[cutg_den]");
    delete cutg_num;
    delete cutg_den;
  }

  if (type == 3)  // mistagrate HEP17 / HEM17 relation: fill "ptrnum" with HEP17
                  // mistagrate and "ptrden" with HEM17 mistagrate
  {
    ptrden = denME->getTH1();
    ptrnum = numME->getTH1();
  }
  return true;
}

void HLTBTagHarvestingAnalyzer::mistagrate(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, TH1F *num, TH1F *den, std::string effName) {
  // do the efficiency_vs_mistag_rate plot
  TH1F *eff;
  eff = new TH1F(effName.c_str(), effName.c_str(), 100, 0, 1);
  eff->SetTitle(effName.c_str());
  eff->SetXTitle("b-effficiency");
  eff->SetYTitle("mistag rate");
  eff->SetOption("E");
  eff->SetLineColor(2);
  eff->SetLineWidth(2);
  eff->SetMarkerStyle(20);
  eff->SetMarkerSize(0.8);
  eff->GetYaxis()->SetRangeUser(0.001, 1.001);
  eff->GetXaxis()->SetRangeUser(-0.001, 1.001);
  eff->SetStats(kFALSE);

  // for each bin in the discr -> find efficiency and mistagrate -> put them in
  // a plot
  for (int i = 1; i <= num->GetNbinsX(); i++) {
    double beff = num->GetBinContent(i);
    double miseff = den->GetBinContent(i);
    double miseffErr = den->GetBinError(i);
    int binX = eff->GetXaxis()->FindBin(beff);
    if (eff->GetBinContent(binX) != 0)
      continue;
    eff->SetBinContent(binX, miseff);
    eff->SetBinError(binX, miseffErr);
  }
  MonitorElement *me;
  me = ibooker.book1D(effName, eff);
  me->setEfficiencyFlag();

  delete eff;
  return;
}

void HLTBTagHarvestingAnalyzer::modulesrate(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, TH1F *num, TH1F *den, std::string effName) {
  // do the eff_vs_disc_HEP17 / eff_vs_disc_HEM17 plot
  TH1F *eff = new TH1F(*num);
  // eff = new TH1F(effName.c_str(),effName.c_str(),100,0,1);
  eff->Divide(den);
  eff->SetTitle(effName.c_str());
  eff->SetXTitle(num->GetXaxis()->GetTitle());
  eff->SetYTitle("");
  eff->SetOption("E");
  eff->SetLineColor(2);
  eff->SetLineWidth(2);
  eff->SetMarkerStyle(20);
  eff->SetMarkerSize(0.8);
  eff->GetYaxis()->SetRangeUser(0.001, 2.001);
  // eff->GetXaxis()->SetRangeUser(-0.001,1.001);
  eff->SetStats(kFALSE);

  MonitorElement *me;
  me = ibooker.book1D(effName, eff);
  me->setEfficiencyFlag();

  delete eff;
  return;
}

TH1F HLTBTagHarvestingAnalyzer::calculateEfficiency1D(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, TH1 &num, TH1 &den, std::string effName) {
  // calculate the efficiency as num/den ratio
  TH1F eff;
  if (num.GetXaxis()->GetXbins()->GetSize() == 0) {
    eff = TH1F(effName.c_str(),
               effName.c_str(),
               num.GetXaxis()->GetNbins(),
               num.GetXaxis()->GetXmin(),
               num.GetXaxis()->GetXmax());
  } else {
    eff = TH1F(effName.c_str(), effName.c_str(), num.GetXaxis()->GetNbins(), num.GetXaxis()->GetXbins()->GetArray());
  }
  eff.SetTitle(effName.c_str());
  eff.SetXTitle(num.GetXaxis()->GetTitle());
  eff.SetYTitle("Efficiency");
  eff.SetOption("PE");
  eff.SetLineColor(2);
  eff.SetLineWidth(2);
  eff.SetMarkerStyle(20);
  eff.SetMarkerSize(0.8);
  eff.GetYaxis()->SetRangeUser(-0.001, 1.001);
  for (int i = 1; i <= num.GetNbinsX(); i++) {
    double d, n, err;
    d = den.GetBinContent(i);
    n = num.GetBinContent(i);
    double e;
    if (d != 0) {
      e = n / d;
      err = std::max(e - TEfficiency::ClopperPearson(d, n, 0.683, false),
                     TEfficiency::ClopperPearson(d, n, 0.683, true) - e);
      // err =  sqrt(e*(1-e)/d); //from binomial standard deviation
    } else {
      e = 0;
      err = 0;
    }
    eff.SetBinContent(i, e);
    eff.SetBinError(i, err);
  }

  MonitorElement *me;
  me = ibooker.book1D(effName, &eff);
  me->setEfficiencyFlag();

  return eff;
}

// define this as a plug-in
DEFINE_FWK_MODULE(HLTBTagHarvestingAnalyzer);
