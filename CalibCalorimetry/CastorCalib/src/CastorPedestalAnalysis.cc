#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorPedestalAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TFile.h>
#include <cmath>

CastorPedestalAnalysis::CastorPedestalAnalysis(const edm::ParameterSet& ps)
    : fRefPedestals(nullptr),
      fRefPedestalWidths(nullptr),
      fRawPedestals(nullptr),
      fRawPedestalWidths(nullptr),
      fValPedestals(nullptr),
      fValPedestalWidths(nullptr) {
  evt = 0;
  sample = 0;
  m_file = nullptr;
  m_AllPedsOK = 0;
  for (int i = 0; i < 4; i++)
    m_stat[i] = 0;
  for (int k = 0; k < 4; k++)
    state.push_back(true);

  // user cfg parameters
  m_outputFileMean = ps.getUntrackedParameter<std::string>("outputFileMeans", "");
  if (!m_outputFileMean.empty()) {
    edm::LogWarning("Castor") << "Castor pedestal means will be saved to " << m_outputFileMean.c_str();
  }
  m_outputFileWidth = ps.getUntrackedParameter<std::string>("outputFileWidths", "");
  if (!m_outputFileWidth.empty()) {
    edm::LogWarning("Castor") << "Castor pedestal widths will be saved to " << m_outputFileWidth.c_str();
  }
  m_outputFileROOT = ps.getUntrackedParameter<std::string>("outputFileHist", "");
  if (!m_outputFileROOT.empty()) {
    edm::LogWarning("Castor") << "Castor pedestal histograms will be saved to " << m_outputFileROOT.c_str();
  }
  m_nevtsample = ps.getUntrackedParameter<int>("nevtsample", 0);
  // for compatibility with previous versions
  if (m_nevtsample == 9999999)
    m_nevtsample = 0;
  m_pedsinADC = ps.getUntrackedParameter<int>("pedsinADC", 0);
  m_hiSaveflag = ps.getUntrackedParameter<int>("hiSaveflag", 0);
  m_pedValflag = ps.getUntrackedParameter<int>("pedValflag", 0);
  if (m_pedValflag < 0)
    m_pedValflag = 0;
  if (m_nevtsample > 0 && m_pedValflag > 0) {
    edm::LogWarning("Castor") << "WARNING - incompatible cfg options: nevtsample = " << m_nevtsample
                              << ", pedValflag = " << m_pedValflag;
    edm::LogWarning("Castor") << "Setting pedValflag = 0";
    m_pedValflag = 0;
  }
  if (m_pedValflag > 1)
    m_pedValflag = 1;
  m_startTS = ps.getUntrackedParameter<int>("firstTS", 0);
  if (m_startTS < 0)
    m_startTS = 0;
  m_endTS = ps.getUntrackedParameter<int>("lastTS", 9);

  //  m_logFile.open("CastorPedestalAnalysis.log");

  castorHists.ALLPEDS = new TH1F("Castor All Pedestals", "HF All Peds", 10, 0, 9);
  castorHists.PEDRMS = new TH1F("Castor All Pedestal Widths", "HF All Pedestal RMS", 100, 0, 3);
  castorHists.PEDMEAN = new TH1F("Castor All Pedestal Means", "HF All Pedestal Means", 100, 0, 9);
  castorHists.CHI2 = new TH1F("Castor Chi2/ndf for whole range Gauss fit", "HF Chi2/ndf Gauss", 200, 0., 50.);
}

//-----------------------------------------------------------------------------
CastorPedestalAnalysis::~CastorPedestalAnalysis() {
  for (_meot = castorHists.PEDTRENDS.begin(); _meot != castorHists.PEDTRENDS.end(); _meot++) {
    for (int i = 0; i < 16; i++)
      _meot->second[i].first->Delete();
  }

  castorHists.ALLPEDS->Delete();
  castorHists.PEDRMS->Delete();
  castorHists.PEDMEAN->Delete();
  castorHists.CHI2->Delete();
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::setup(const std::string& m_outputFileROOT) {
  // open the histogram file, create directories within
  m_file = new TFile(m_outputFileROOT.c_str(), "RECREATE");
  m_file->mkdir("Castor");
  m_file->cd();
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::processEvent(const CastorDigiCollection& castor, const CastorDbService& cond) {
  evt++;
  sample = 1;
  evt_curr = evt;
  if (m_nevtsample > 0) {
    sample = (evt - 1) / m_nevtsample + 1;
    evt_curr = evt % m_nevtsample;
    if (evt_curr == 0)
      evt_curr = m_nevtsample;
  }

  m_shape = cond.getCastorShape();
  // HF
  try {
    if (castor.empty())
      throw(int) castor.size();
    for (CastorDigiCollection::const_iterator j = castor.begin(); j != castor.end(); ++j) {
      const CastorDataFrame digi = (const CastorDataFrame)(*j);
      m_coder = cond.getCastorCoder(digi.id());
      for (int i = m_startTS; i < digi.size() && i <= m_endTS; i++) {
        for (int flag = 0; flag < 4; flag++) {
          if (i + flag < digi.size() && i + flag <= m_endTS) {
            per2CapsHists(flag, 2, digi.id(), digi.sample(i), digi.sample(i + flag), castorHists.PEDTRENDS, cond);
          }
        }
      }
      if (m_startTS == 0 && m_endTS > 4) {
        AllChanHists(digi.id(),
                     digi.sample(0),
                     digi.sample(1),
                     digi.sample(2),
                     digi.sample(3),
                     digi.sample(4),
                     digi.sample(5),
                     castorHists.PEDTRENDS);
      }
    }
  } catch (int i) {
    //    m_logFile << "Event with " << i<<" Castor Digis passed." << std::endl;
  }
  // Call the function every m_nevtsample events
  if (m_nevtsample > 0) {
    if (evt % m_nevtsample == 0)
      SampleAnalysis();
  }
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::per2CapsHists(int flag,
                                           int id,
                                           const HcalDetId detid,
                                           const HcalQIESample& qie1,
                                           const HcalQIESample& qie2,
                                           std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT,
                                           const CastorDbService& cond) {
  // this function is due to be called for every time slice, it fills either a charge
  // histo for a single capID (flag=0) or a product histo for two capIDs (flag>0)

  static const int bins = 10;
  static const int bins2 = 100;
  float lo = -0.5;
  float hi = 9.5;
  std::map<int, PEDBUNCH> _mei;
  static std::map<HcalDetId, std::map<int, float> > QieCalibMap;
  std::string type = "Castor";

  /*
  if(id==0){
    if(detid.ieta()<16) type = "HB";
    if(detid.ieta()>16) type = "HE";
    if(detid.ieta()==16){
      if(detid.depth()<3) type = "HB";
      if(detid.depth()==3) type = "HE";
    }
  } 
  else if(id==1) type = "HO";
  else if(id==2) type = "HF"; 
  */

  _meot = toolT.find(detid);

  // if histos for the current channel do not exist, first create them,
  if (_meot == toolT.end()) {
    std::map<int, PEDBUNCH> insert;
    std::map<int, float> qiecalib;
    char name[1024];
    for (int i = 0; i < 4; i++) {
      lo = -0.5;
      // fix from Andy: if you convert to fC and then bin in units of 1, you may 'skip' a bin while
      // filling, since the ADCs are quantized
      if (m_pedsinADC)
        hi = 9.5;
      else
        hi = 11.5;
      sprintf(
          name, "%s Pedestal, eta=%d phi=%d d=%d cap=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth(), i);
      insert[i].first = new TH1F(name, name, bins, lo, hi);
      sprintf(name,
              "%s Product, eta=%d phi=%d d=%d caps=%d*%d",
              type.c_str(),
              detid.ieta(),
              detid.iphi(),
              detid.depth(),
              i,
              (i + 1) % 4);
      insert[4 + i].first = new TH1F(name, name, bins2, 0., 100.);
      sprintf(name,
              "%s Product, eta=%d phi=%d d=%d caps=%d*%d",
              type.c_str(),
              detid.ieta(),
              detid.iphi(),
              detid.depth(),
              i,
              (i + 2) % 4);
      insert[8 + i].first = new TH1F(name, name, bins2, 0., 100.);
      sprintf(name,
              "%s Product, eta=%d phi=%d d=%d caps=%d*%d",
              type.c_str(),
              detid.ieta(),
              detid.iphi(),
              detid.depth(),
              i,
              (i + 3) % 4);
      insert[12 + i].first = new TH1F(name, name, bins2, 0., 100.);
    }
    sprintf(name, "%s Signal in TS 4+5, eta=%d phi=%d d=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[16].first = new TH1F(name, name, 21, -0.5, 20.5);
    sprintf(
        name, "%s Signal in TS 4+5-2-3, eta=%d phi=%d d=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[17].first = new TH1F(name, name, 21, -10.5, 10.5);
    sprintf(name,
            "%s Signal in TS 4+5-(0+1+2+3)/2., eta=%d phi=%d d=%d",
            type.c_str(),
            detid.ieta(),
            detid.iphi(),
            detid.depth());
    insert[18].first = new TH1F(name, name, 21, -10.5, 10.5);
    toolT[detid] = insert;
    _meot = toolT.find(detid);
    // store QIE calibrations in a map for later reuse
    QieCalibMap[detid] = qiecalib;
  }

  _mei = _meot->second;

  const CastorQIECoder* coder = cond.getCastorCoder(detid);
  const CastorQIEShape* shape = cond.getCastorShape();
  float charge1 = coder->charge(*shape, qie1.adc(), qie1.capid());
  float charge2 = coder->charge(*shape, qie2.adc(), qie2.capid());

  // fill single capID histo
  if (flag == 0) {
    if (m_nevtsample > 0) {
      if ((evt - 1) % m_nevtsample == 0 && state[qie1.capid()]) {
        state[qie1.capid()] = false;
        _mei[qie1.capid()].first->Reset();
        _mei[qie1.capid() + 4].first->Reset();
        _mei[qie1.capid() + 8].first->Reset();
        _mei[qie1.capid() + 12].first->Reset();
      }
    }
    if (qie1.adc() < bins) {
      if (m_pedsinADC)
        _mei[qie1.capid()].first->Fill(qie1.adc());
      else
        _mei[qie1.capid()].first->Fill(charge1);
    } else if (qie1.adc() >= bins) {
      _mei[qie1.capid()].first->AddBinContent(bins + 1, 1);
    }
  }

  // fill 2 capID histo
  if (flag > 0) {
    std::map<int, float> qiecalib = QieCalibMap[detid];
    //float charge1=(qie1.adc()-qiecalib[qie1.capid()+4])/qiecalib[qie1.capid()];
    //float charge2=(qie2.adc()-qiecalib[qie2.capid()+4])/qiecalib[qie2.capid()];
    if (charge1 * charge2 < bins2) {
      _mei[qie1.capid() + 4 * flag].first->Fill(charge1 * charge2);
    } else {
      _mei[qie1.capid() + 4 * flag].first->Fill(bins2);
    }
  }

  if (flag == 0) {
    //    if(id==0) hbHists.ALLPEDS->Fill(qie1.adc());
    //   else if(id==1) hoHists.ALLPEDS->Fill(qie1.adc());
    //   else if(id==2) castorHists.ALLPEDS->Fill(qie1.adc());
    castorHists.ALLPEDS->Fill(qie1.adc());
  }
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::AllChanHists(const HcalDetId detid,
                                          const HcalQIESample& qie0,
                                          const HcalQIESample& qie1,
                                          const HcalQIESample& qie2,
                                          const HcalQIESample& qie3,
                                          const HcalQIESample& qie4,
                                          const HcalQIESample& qie5,
                                          std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT) {
  // this function is due to be called for every channel

  _meot = toolT.find(detid);
  std::map<int, PEDBUNCH> _mei = _meot->second;
  _mei[16].first->Fill(qie4.adc() + qie5.adc() - 1.);
  _mei[17].first->Fill(qie4.adc() + qie5.adc() - qie2.adc() - qie3.adc());
  _mei[18].first->Fill(qie4.adc() + qie5.adc() - (qie0.adc() + qie1.adc() + qie2.adc() + qie3.adc()) / 2.);
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::SampleAnalysis() {
  // it is called every m_nevtsample events (a sample) and the end of run
  char PedSampleNum[20];

  // Compute pedestal constants for each HBHE, HO, HF
  sprintf(PedSampleNum, "Castor_Sample%d", sample);
  m_file->cd();
  m_file->mkdir(PedSampleNum);
  m_file->cd(PedSampleNum);
  GetPedConst(castorHists.PEDTRENDS, castorHists.PEDMEAN, castorHists.PEDRMS);
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::GetPedConst(std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT,
                                         TH1F* PedMeans,
                                         TH1F* PedWidths) {
  // Completely rewritten version oct 2006
  // Compute pedestal constants and fill into CastorPedestals and CastorPedestalWidths objects
  float cap[4];
  float sig[4][4];
  float dcap[4];
  float dsig[4][4];
  float chi2[4];

  for (_meot = toolT.begin(); _meot != toolT.end(); _meot++) {
    HcalDetId detid = _meot->first;

    // take mean and width from a Gaussian fit or directly from the histo
    if (fitflag > 0) {
      for (int i = 0; i < 4; i++) {
        TF1* fit = _meot->second[i].first->GetFunction("gaus");
        chi2[i] = 0;
        if (fit->GetNDF() != 0)
          chi2[i] = fit->GetChisquare() / fit->GetNDF();
        cap[i] = fit->GetParameter(1);
        sig[i][i] = fit->GetParameter(2);
        dcap[i] = fit->GetParError(1);
        dsig[i][i] = fit->GetParError(2);
      }
    } else {
      for (int i = 0; i < 4; i++) {
        cap[i] = _meot->second[i].first->GetMean();
        sig[i][i] = _meot->second[i].first->GetRMS();
        m_stat[i] = 0;

        for (int j = m_startTS; j < m_endTS + 1; j++) {
          m_stat[i] += _meot->second[i].first->GetBinContent(j + 1);
        }
        dcap[i] = sig[i][i] / sqrt(m_stat[i]);
        //        dsig[i][i] = dcap[i]*sig[i][i]/cap[i];
        dsig[i][i] = sig[i][i] / sqrt(2. * m_stat[i]);
        chi2[i] = 0.;
      }
    }

    for (int i = 0; i < 4; i++) {
      if (m_hiSaveflag > 0) {
        if (m_pedsinADC)
          _meot->second[i].first->GetXaxis()->SetTitle("ADC");
        else
          _meot->second[i].first->GetXaxis()->SetTitle("Charge, fC");
        _meot->second[i].first->GetYaxis()->SetTitle("CapID samplings");
        _meot->second[i].first->Write();
      }
      if (m_nevtsample > 0) {
        _meot->second[i].second.first[0].push_back(cap[i]);
        _meot->second[i].second.first[1].push_back(dcap[i]);
        _meot->second[i].second.first[2].push_back(sig[i][i]);
        _meot->second[i].second.first[3].push_back(dsig[i][i]);
        _meot->second[i].second.first[4].push_back(chi2[i]);
      }
      PedMeans->Fill(cap[i]);
      PedWidths->Fill(sig[i][i]);
    }

    // special histos for Shuichi
    if (m_hiSaveflag == -100) {
      for (int i = 16; i < 19; i++) {
        if (m_pedsinADC)
          _meot->second[i].first->GetXaxis()->SetTitle("ADC");
        else
          _meot->second[i].first->GetXaxis()->SetTitle("Charge, fC");
        _meot->second[i].first->GetYaxis()->SetTitle("Events");
        _meot->second[i].first->Write();
      }
    }

    // diagonal sigma is width squared
    sig[0][0] = sig[0][0] * sig[0][0];
    sig[1][1] = sig[1][1] * sig[1][1];
    sig[2][2] = sig[2][2] * sig[2][2];
    sig[3][3] = sig[3][3] * sig[3][3];

    // off diagonal sigmas (correlations) are computed from 3 histograms
    // here we still have all 4*3=12 combinations
    sig[0][1] = _meot->second[4].first->GetMean() - cap[0] * cap[1];
    sig[0][2] = _meot->second[8].first->GetMean() - cap[0] * cap[2];
    sig[1][2] = _meot->second[5].first->GetMean() - cap[1] * cap[2];
    sig[1][3] = _meot->second[9].first->GetMean() - cap[1] * cap[3];
    sig[2][3] = _meot->second[6].first->GetMean() - cap[2] * cap[3];
    sig[0][3] = _meot->second[12].first->GetMean() - cap[0] * cap[3];
    sig[1][0] = _meot->second[13].first->GetMean() - cap[1] * cap[0];
    sig[2][0] = _meot->second[10].first->GetMean() - cap[2] * cap[0];
    sig[2][1] = _meot->second[14].first->GetMean() - cap[2] * cap[1];
    sig[3][1] = _meot->second[11].first->GetMean() - cap[3] * cap[1];
    sig[3][2] = _meot->second[15].first->GetMean() - cap[3] * cap[2];
    sig[3][0] = _meot->second[7].first->GetMean() - cap[3] * cap[0];

    // there is no proper error calculation for the correlation coefficients
    for (int i = 0; i < 4; i++) {
      if (m_nevtsample > 0) {
        _meot->second[i].second.first[5].push_back(sig[i][(i + 1) % 4]);
        _meot->second[i].second.first[6].push_back(2 * sig[i][i] * dsig[i][i]);
        _meot->second[i].second.first[7].push_back(sig[i][(i + 2) % 4]);
        _meot->second[i].second.first[8].push_back(2 * sig[i][i] * dsig[i][i]);
        _meot->second[i].second.first[9].push_back(sig[i][(i + 3) % 4]);
        _meot->second[i].second.first[10].push_back(2 * sig[i][i] * dsig[i][i]);
      }
      // save product histos if desired
      if (m_hiSaveflag > 10) {
        if (m_pedsinADC)
          _meot->second[i + 4].first->GetXaxis()->SetTitle("ADC^2");
        else
          _meot->second[i + 4].first->GetXaxis()->SetTitle("Charge^2, fC^2");
        _meot->second[i + 4].first->GetYaxis()->SetTitle("2-CapID samplings");
        _meot->second[i + 4].first->Write();
        if (m_pedsinADC)
          _meot->second[i + 8].first->GetXaxis()->SetTitle("ADC^2");
        else
          _meot->second[i + 8].first->GetXaxis()->SetTitle("Charge^2, fC^2");
        _meot->second[i + 8].first->GetYaxis()->SetTitle("2-CapID samplings");
        _meot->second[i + 8].first->Write();
        if (m_pedsinADC)
          _meot->second[i + 12].first->GetXaxis()->SetTitle("ADC^2");
        else
          _meot->second[i + 12].first->GetXaxis()->SetTitle("Charge^2, fC^2");
        _meot->second[i + 12].first->GetYaxis()->SetTitle("2-CapID samplings");
        _meot->second[i + 12].first->Write();
      }
    }

    // fill the objects - at this point only close and medium correlations are stored
    // and the matrix is assumed symmetric
    if (m_nevtsample < 1) {
      sig[1][0] = sig[0][1];
      sig[2][0] = sig[0][2];
      sig[2][1] = sig[1][2];
      sig[3][1] = sig[1][3];
      sig[3][2] = sig[2][3];
      sig[0][3] = sig[3][0];
      if (fRawPedestals) {
        CastorPedestal item(detid, cap[0], cap[1], cap[2], cap[3]);
        fRawPedestals->addValues(item);
      }
      if (fRawPedestalWidths) {
        CastorPedestalWidth widthsp(detid);
        widthsp.setSigma(0, 0, sig[0][0]);
        widthsp.setSigma(0, 1, sig[0][1]);
        widthsp.setSigma(0, 2, sig[0][2]);
        widthsp.setSigma(1, 1, sig[1][1]);
        widthsp.setSigma(1, 2, sig[1][2]);
        widthsp.setSigma(1, 3, sig[1][3]);
        widthsp.setSigma(2, 2, sig[2][2]);
        widthsp.setSigma(2, 3, sig[2][3]);
        widthsp.setSigma(3, 3, sig[3][3]);
        widthsp.setSigma(3, 0, sig[0][3]);
        fRawPedestalWidths->addValues(widthsp);
      }
    }
  }
}

//-----------------------------------------------------------------------------
int CastorPedestalAnalysis::done(const CastorPedestals* fInputPedestals,
                                 const CastorPedestalWidths* fInputPedestalWidths,
                                 CastorPedestals* fOutputPedestals,
                                 CastorPedestalWidths* fOutputPedestalWidths) {
  int nstat[4];

  // Pedestal objects
  // inputs...
  fRefPedestals = fInputPedestals;
  fRefPedestalWidths = fInputPedestalWidths;

  // outputs...
  if (m_pedValflag > 0) {
    fValPedestals = fOutputPedestals;
    fValPedestalWidths = fOutputPedestalWidths;
    fRawPedestals = new CastorPedestals();
    fRawPedestalWidths = new CastorPedestalWidths();
  } else {
    fRawPedestals = fOutputPedestals;
    fRawPedestalWidths = fOutputPedestalWidths;
    fValPedestals = new CastorPedestals();
    fValPedestalWidths = new CastorPedestalWidths();
  }

  // compute pedestal constants
  if (m_nevtsample < 1)
    SampleAnalysis();
  if (m_nevtsample > 0) {
    if (evt % m_nevtsample != 0)
      SampleAnalysis();
  }

  // trending histos
  if (m_nevtsample > 0) {
    m_file->cd();
    m_file->cd("Castor");
    Trendings(castorHists.PEDTRENDS, castorHists.CHI2, castorHists.CAPID_AVERAGE, castorHists.CAPID_CHI2);
  }

  if (m_nevtsample < 1) {
    // pedestal validation: m_AllPedsOK=-1 means not validated,
    //                                   0 everything OK,
    //                                   N>0 : mod(N,100000) drifts + width changes
    //                                         int(N/100000) missing channels
    m_AllPedsOK = -1;
    if (m_pedValflag > 0) {
      for (int i = 0; i < 4; i++)
        nstat[i] = (int)m_stat[i];
      int NPedErrors = CastorPedVal(nstat,
                                    fRefPedestals,
                                    fRefPedestalWidths,
                                    fRawPedestals,
                                    fRawPedestalWidths,
                                    fValPedestals,
                                    fValPedestalWidths);
      m_AllPedsOK = NPedErrors;
    }
    // setting m_AllPedsOK=-2 will inhibit writing pedestals out
    //    if(m_pedValflag==1){
    //      if(evt<100)m_AllPedsOK=-2;
    //    }
  }

  // Write other histograms.

  // Castor
  m_file->cd();
  m_file->cd("Castor");
  castorHists.ALLPEDS->Write();
  castorHists.PEDRMS->Write();
  castorHists.PEDMEAN->Write();

  m_file->Close();
  edm::LogWarning("Castor") << "Hcal/Castor histograms written to " << m_outputFileROOT.c_str();
  return (int)m_AllPedsOK;
}

//-----------------------------------------------------------------------------
void CastorPedestalAnalysis::Trendings(std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT,
                                       TH1F* Chi2,
                                       TH1F* CapidAverage,
                                       TH1F* CapidChi2) {
  // check stability of pedestal constants in a single long run

  std::map<int, std::vector<double> > AverageValues;

  for (_meot = toolT.begin(); _meot != toolT.end(); _meot++) {
    for (int i = 0; i < 4; i++) {
      char name[1024];
      HcalDetId detid = _meot->first;
      sprintf(name, "Pedestal trend, eta=%d phi=%d d=%d cap=%d", detid.ieta(), detid.iphi(), detid.depth(), i);
      int bins = _meot->second[i].second.first[0].size();
      float lo = 0.5;
      float hi = (float)bins + 0.5;
      _meot->second[i].second.second.push_back(new TH1F(name, name, bins, lo, hi));
      sprintf(name, "Width trend, eta=%d phi=%d d=%d cap=%d", detid.ieta(), detid.iphi(), detid.depth(), i);
      bins = _meot->second[i].second.first[2].size();
      hi = (float)bins + 0.5;
      _meot->second[i].second.second.push_back(new TH1F(name, name, bins, lo, hi));
      sprintf(name,
              "Correlation trend, eta=%d phi=%d d=%d caps=%d*%d",
              detid.ieta(),
              detid.iphi(),
              detid.depth(),
              i,
              (i + 1) % 4);
      bins = _meot->second[i].second.first[5].size();
      hi = (float)bins + 0.5;
      _meot->second[i].second.second.push_back(new TH1F(name, name, bins, lo, hi));
      /*      sprintf(name,"Correlation trend, eta=%d phi=%d d=%d caps=%d*%d",detid.ieta(),detid.iphi(),detid.depth(),i,(i+2)%4);
      bins = _meot->second[i].second.first[7].size();
      hi = (float)bins+0.5;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
      sprintf(name,"Correlation trend, eta=%d phi=%d d=%d caps=%d*%d",detid.ieta(),detid.iphi(),detid.depth(),i,(i+3)%4);
      bins = _meot->second[i].second.first[9].size();
      hi = (float)bins+0.5;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi)); */

      std::vector<double>::iterator sample_it;
      // Pedestal mean - put content and errors
      int j = 0;
      for (sample_it = _meot->second[i].second.first[0].begin(); sample_it != _meot->second[i].second.first[0].end();
           ++sample_it) {
        _meot->second[i].second.second[0]->SetBinContent(++j, *sample_it);
      }
      j = 0;
      for (sample_it = _meot->second[i].second.first[1].begin(); sample_it != _meot->second[i].second.first[1].end();
           ++sample_it) {
        _meot->second[i].second.second[0]->SetBinError(++j, *sample_it);
      }
      // fit with a constant - extract parameters
      _meot->second[i].second.second[0]->Fit("pol0", "Q");
      TF1* fit = _meot->second[i].second.second[0]->GetFunction("pol0");
      AverageValues[0].push_back(fit->GetParameter(0));
      AverageValues[1].push_back(fit->GetParError(0));
      if (sample > 1)
        AverageValues[2].push_back(fit->GetChisquare() / fit->GetNDF());
      else
        AverageValues[2].push_back(fit->GetChisquare());
      sprintf(name, "Sample (%d events)", m_nevtsample);
      _meot->second[i].second.second[0]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[0]->GetYaxis()->SetTitle("Pedestal value");
      _meot->second[i].second.second[0]->Write();
      // Pedestal width - put content and errors
      j = 0;
      for (sample_it = _meot->second[i].second.first[2].begin(); sample_it != _meot->second[i].second.first[2].end();
           ++sample_it) {
        _meot->second[i].second.second[1]->SetBinContent(++j, *sample_it);
      }
      j = 0;
      for (sample_it = _meot->second[i].second.first[3].begin(); sample_it != _meot->second[i].second.first[3].end();
           ++sample_it) {
        _meot->second[i].second.second[1]->SetBinError(++j, *sample_it);
      }
      _meot->second[i].second.second[1]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[1]->GetYaxis()->SetTitle("Pedestal width");
      _meot->second[i].second.second[1]->Write();
      // Correlation coeffs - put contents and errors
      j = 0;
      for (sample_it = _meot->second[i].second.first[5].begin(); sample_it != _meot->second[i].second.first[5].end();
           ++sample_it) {
        _meot->second[i].second.second[2]->SetBinContent(++j, *sample_it);
      }
      j = 0;
      for (sample_it = _meot->second[i].second.first[6].begin(); sample_it != _meot->second[i].second.first[6].end();
           ++sample_it) {
        _meot->second[i].second.second[2]->SetBinError(++j, *sample_it);
      }
      _meot->second[i].second.second[2]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[2]->GetYaxis()->SetTitle("Close correlation");
      _meot->second[i].second.second[2]->Write();
      /*     j=0;
      for(sample_it=_meot->second[i].second.first[7].begin();
          sample_it!=_meot->second[i].second.first[7].end();sample_it++){
        _meot->second[i].second.second[3]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[8].begin();
          sample_it!=_meot->second[i].second.first[8].end();sample_it++){
        _meot->second[i].second.second[3]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[3]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[3]->GetYaxis()->SetTitle("Intermediate correlation");
      _meot->second[i].second.second[3]->Write();
      j=0;
      for(sample_it=_meot->second[i].second.first[9].begin();
          sample_it!=_meot->second[i].second.first[9].end();sample_it++){
        _meot->second[i].second.second[4]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[10].begin();
          sample_it!=_meot->second[i].second.first[10].end();sample_it++){
        _meot->second[i].second.second[4]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[4]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[4]->GetYaxis()->SetTitle("Distant correlation");
      _meot->second[i].second.second[4]->Write(); */
      // chi2
      j = 0;
      for (sample_it = _meot->second[i].second.first[4].begin(); sample_it != _meot->second[i].second.first[4].end();
           ++sample_it) {
        Chi2->Fill(*sample_it);
      }
    }
  }
  CapidAverage = new TH1F("Constant fit: Pedestal Values",
                          "Constant fit: Pedestal Values",
                          AverageValues[0].size(),
                          0.,
                          AverageValues[0].size());
  std::vector<double>::iterator sample_it;
  int j = 0;
  for (sample_it = AverageValues[0].begin(); sample_it != AverageValues[0].end(); ++sample_it) {
    CapidAverage->SetBinContent(++j, *sample_it);
  }
  j = 0;
  for (sample_it = AverageValues[1].begin(); sample_it != AverageValues[1].end(); ++sample_it) {
    CapidAverage->SetBinError(++j, *sample_it);
  }
  CapidChi2 = new TH1F(
      "Constant fit: Chi2/ndf", "Constant fit: Chi2/ndf", AverageValues[2].size(), 0., AverageValues[2].size());
  j = 0;
  for (sample_it = AverageValues[2].begin(); sample_it != AverageValues[2].end(); ++sample_it) {
    CapidChi2->SetBinContent(++j, *sample_it);
    //CapidChi2->SetBinError(++j,0);
  }
  Chi2->GetXaxis()->SetTitle("Chi2/ndf");
  Chi2->GetYaxis()->SetTitle("50 x [(16+2) x 4 x 4] `events`");
  Chi2->Write();
  CapidAverage->GetYaxis()->SetTitle("Pedestal value");
  CapidAverage->GetXaxis()->SetTitle("(16+2) x 4 x 4 `events`");
  CapidAverage->Write();
  CapidChi2->GetYaxis()->SetTitle("Chi2/ndf");
  CapidChi2->GetXaxis()->SetTitle("(16+2) x 4 x 4 `events`");
  CapidChi2->Write();
}

//-----------------------------------------------------------------------------
int CastorPedestalAnalysis::CastorPedVal(int nstat[4],
                                         const CastorPedestals* fRefPedestals,
                                         const CastorPedestalWidths* fRefPedestalWidths,
                                         CastorPedestals* fRawPedestals,
                                         CastorPedestalWidths* fRawPedestalWidths,
                                         CastorPedestals* fValPedestals,
                                         CastorPedestalWidths* fValPedestalWidths) {
  // new version of pedestal validation - it is designed to be as independent of
  // all the rest as possible - you only need to provide valid pedestal objects
  // and a vector of statistics per capID to use this as standalone code
  HcalDetId detid;
  float RefPedVals[4];
  float RefPedSigs[4][4];
  float RawPedVals[4];
  float RawPedSigs[4][4];
  std::map<HcalDetId, bool> isinRaw;
  std::map<HcalDetId, bool> isinRef;
  std::vector<DetId> RefChanns = fRefPedestals->getAllChannels();
  std::vector<DetId> RawChanns = fRawPedestals->getAllChannels();
  std::ofstream PedValLog;
  PedValLog.open("CastorPedVal.log");

  if (nstat[0] + nstat[1] + nstat[2] + nstat[3] < 2500)
    PedValLog << "CastorPedVal: warning - low statistics" << std::endl;
  // find complete list of channels in current data and reference
  for (int i = 0; i < (int)RawChanns.size(); i++) {
    isinRef[HcalDetId(RawChanns[i])] = false;
  }
  for (int i = 0; i < (int)RefChanns.size(); i++) {
    detid = HcalDetId(RefChanns[i]);
    isinRaw[detid] = false;
    isinRef[detid] = true;
  }
  for (int i = 0; i < (int)RawChanns.size(); i++) {
    detid = HcalDetId(RawChanns[i]);
    isinRaw[detid] = true;
    if (isinRef[detid] == false) {
      PedValLog << "CastorPedVal: channel " << detid << " not found in reference set" << std::endl;
      std::cerr << "CastorPedVal: channel " << detid << " not found in reference set" << std::endl;
    }
  }

  // main loop over channels
  int erflag = 0;
  for (int i = 0; i < (int)RefChanns.size(); i++) {
    detid = HcalDetId(RefChanns[i]);
    for (int icap = 0; icap < 4; icap++) {
      RefPedVals[icap] = fRefPedestals->getValues(detid)->getValue(icap);
      for (int icap2 = icap; icap2 < 4; icap2++) {
        RefPedSigs[icap][icap2] = fRefPedestalWidths->getValues(detid)->getSigma(icap, icap2);
        if (icap2 != icap)
          RefPedSigs[icap2][icap] = RefPedSigs[icap][icap2];
      }
    }

    // read new raw values
    if (isinRaw[detid]) {
      for (int icap = 0; icap < 4; icap++) {
        RawPedVals[icap] = fRawPedestals->getValues(detid)->getValue(icap);
        for (int icap2 = icap; icap2 < 4; icap2++) {
          RawPedSigs[icap][icap2] = fRawPedestalWidths->getValues(detid)->getSigma(icap, icap2);
          if (icap2 != icap)
            RawPedSigs[icap2][icap] = RawPedSigs[icap][icap2];
        }
      }

      // first quick check if raw values make sense: if not, the channel is treated like absent
      for (int icap = 0; icap < 4; icap++) {
        if (RawPedVals[icap] < 1. || RawPedSigs[icap][icap] < 0.01)
          isinRaw[detid] = false;
        for (int icap2 = icap; icap2 < 4; icap2++) {
          if (fabs(RawPedSigs[icap][icap2] / sqrt(RawPedSigs[icap][icap] * RawPedSigs[icap2][icap2])) > 1.)
            isinRaw[detid] = false;
        }
      }
    }

    // check raw values against reference
    if (isinRaw[detid]) {
      for (int icap = 0; icap < 4; icap++) {
        int icap2 = (icap + 1) % 4;
        float width = sqrt(RawPedSigs[icap][icap]);
        float erof1 = width / sqrt((float)nstat[icap]);
        float erof2 = sqrt(erof1 * erof1 + RawPedSigs[icap][icap] / (float)nstat[icap]);
        float erofwidth = width / sqrt(2. * nstat[icap]);
        float diffof1 = RawPedVals[icap] - RefPedVals[icap];
        float diffof2 = RawPedVals[icap] + RawPedVals[icap2] - RefPedVals[icap] - RefPedVals[icap2];
        float diffofw = width - sqrt(RefPedSigs[icap][icap]);

        // validation in 2 TS for HB, HE, HO, in 1 TS for HF
        int nTS = 2;
        if (detid.subdet() == HcalForward)
          nTS = 1;
        if (nTS == 1 && fabs(diffof1) > 0.5 + erof1) {
          erflag += 1;
          PedValLog << "HcalPedVal: drift in channel " << detid << " cap " << icap << ": " << RawPedVals[icap] << " - "
                    << RefPedVals[icap] << " = " << diffof1 << std::endl;
        }
        if (nTS == 2 && fabs(diffof2) > 0.5 + erof2) {
          erflag += 1;
          PedValLog << "HcalPedVal: drift in channel " << detid << " caps " << icap << "+" << icap2 << ": "
                    << RawPedVals[icap] << "+" << RawPedVals[icap2] << " - " << RefPedVals[icap] << "+"
                    << RefPedVals[icap2] << " = " << diffof2 << std::endl;
        }
        if (fabs(diffofw) > 0.15 * width + erofwidth) {
          erflag += 1;
          PedValLog << "HcalPedVal: width changed in channel " << detid << " cap " << icap << ": " << width << " - "
                    << sqrt(RefPedSigs[icap][icap]) << " = " << diffofw << std::endl;
        }
      }
    }

    // for disconnected/bad channels restore reference values
    else {
      PedValLog << "HcalPedVal: no valid data from channel " << detid << std::endl;
      erflag += 100000;
      CastorPedestal item(detid, RefPedVals[0], RefPedVals[1], RefPedVals[2], RefPedVals[3]);
      fValPedestals->addValues(item);
      CastorPedestalWidth widthsp(detid);
      for (int icap = 0; icap < 4; icap++) {
        for (int icap2 = icap; icap2 < 4; icap2++)
          widthsp.setSigma(icap2, icap, RefPedSigs[icap2][icap]);
      }
      fValPedestalWidths->addValues(widthsp);
    }

    // end of channel loop
  }

  if (erflag == 0)
    PedValLog << "HcalPedVal: all pedestals checked OK" << std::endl;

  // now construct the remaining part of the validated objects
  // if nothing changed outside tolerance, validated set = reference set
  if (erflag % 100000 == 0) {
    for (int i = 0; i < (int)RefChanns.size(); i++) {
      detid = HcalDetId(RefChanns[i]);
      if (isinRaw[detid]) {
        CastorPedestalWidth widthsp(detid);
        for (int icap = 0; icap < 4; icap++) {
          RefPedVals[icap] = fRefPedestals->getValues(detid)->getValue(icap);
          for (int icap2 = icap; icap2 < 4; icap2++) {
            RefPedSigs[icap][icap2] = fRefPedestalWidths->getValues(detid)->getSigma(icap, icap2);
            if (icap2 != icap)
              RefPedSigs[icap2][icap] = RefPedSigs[icap][icap2];
            widthsp.setSigma(icap2, icap, RefPedSigs[icap2][icap]);
          }
        }
        fValPedestalWidths->addValues(widthsp);
        CastorPedestal item(detid, RefPedVals[0], RefPedVals[1], RefPedVals[2], RefPedVals[3]);
        fValPedestals->addValues(item);
      }
    }
  }

  // if anything changed, validated set = raw set + reference for missing/bad channels
  else {
    for (int i = 0; i < (int)RawChanns.size(); i++) {
      detid = HcalDetId(RawChanns[i]);
      if (isinRaw[detid]) {
        CastorPedestalWidth widthsp(detid);
        for (int icap = 0; icap < 4; icap++) {
          RawPedVals[icap] = fRawPedestals->getValues(detid)->getValue(icap);
          for (int icap2 = icap; icap2 < 4; icap2++) {
            RawPedSigs[icap][icap2] = fRawPedestalWidths->getValues(detid)->getSigma(icap, icap2);
            if (icap2 != icap)
              RawPedSigs[icap2][icap] = RawPedSigs[icap][icap2];
            widthsp.setSigma(icap2, icap, RawPedSigs[icap2][icap]);
          }
        }
        fValPedestalWidths->addValues(widthsp);
        CastorPedestal item(detid, RawPedVals[0], RawPedVals[1], RawPedVals[2], RawPedVals[3]);
        fValPedestals->addValues(item);
      }
    }
  }
  return erflag;
}
