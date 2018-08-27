//****************************************************//
//********** CastorDigiMonitor: ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 29.08.2008 (first version) ******//
////---- digi values in Castor r/o channels
//// last revision: 31.05.2011 (Panos Katsas) to remove selecting N events for
///filling the histograms
//****************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//     add rms check, DB   15.04.2015 (Vladimir Popov)
//==================================================================//

#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include <string>
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

vector<std::string> HltPaths_;
int StatusBadChannel = 1;
int ChannelStatus[14][16];
int N_GoodChannels = 224;
int EtowerLastModule = 5;
int TrigIndexMax = 0;

CastorDigiMonitor::CastorDigiMonitor(const edm::ParameterSet& ps) {
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);
  subsystemname_ =
      ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor");
  EtowerLastModule = ps.getUntrackedParameter<int>("towerLastModule", 6);
  RatioThresh1 = ps.getUntrackedParameter<double>("ratioThreshold", 0.9);
  Qrms_DEAD = ps.getUntrackedParameter<double>("QrmsDead", 0.01);  // fC
  HltPaths_ = ps.getParameter<vector<string> >("HltPaths");

  Qrms_DEAD = Qrms_DEAD * Qrms_DEAD;
  TS_MAX = ps.getUntrackedParameter<double>("qieTSmax", 6);
  StatusBadChannel = CastorChannelStatus::StatusBit::BAD;
  if (fVerbosity > 0)
    LogPrint("CastorDigi") << "enum CastorChannelStatus::StatusBit::BAD="
                           << StatusBadChannel
                           << "EtowerLastModule = " << EtowerLastModule << endl;
}

CastorDigiMonitor::~CastorDigiMonitor() {}

void CastorDigiMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                       const edm::Run& iRun,
                                       const edm::EventSetup& iSetup) {
  char s[60];
  string st;
  if (fVerbosity > 0) LogPrint("CastorMonitorModule") << "Digi bookHist(start)";

  getDbData(iSetup);

  char sTileIndex[50];
  sprintf(sTileIndex, "Cell(=moduleZ*16+sector#phi)");

  ievt_ = 0;

  ibooker.setCurrentFolder(subsystemname_);
  hBX = ibooker.bookProfile("average E(digi) in BX",
                            "Castor average E (digi);Event.BX;fC", 3601, -0.5,
                            3600.5, 0., 1.e10, "");
  hBX->getTProfile()->SetOption("hist");

  string trname = HltPaths_[0];
  hpBXtrig = ibooker.bookProfile(
      "average E(digi) in BXtrig",
      "Castor average E (digi) trigger:'" + trname + "';Event.BX;fC", 3601,
      -0.5, 3600.5, 0., 1.e10, "");
  hpBXtrig->getTProfile()->SetOption("hist");

  hpTrigRes = ibooker.bookProfile(
      "E(digi)vsTriggerIndex",
      "Castor average E(digi) by triggerIndex;triggerIndex;fC", 512, 0., 512,
      0., 1.e10, "");
  hpTrigRes->getTProfile()->SetOption("hist");

  ibooker.setCurrentFolder(subsystemname_ + "/CastorDigiMonitor");

  std::string s2 = "CASTOR QIE_capID+er+dv";
  h2digierr = ibooker.bookProfile2D(s2, s2, 14, 0., 14., 16, 0., 16., 100, 0,
                                    1.e10, "");
  h2digierr->getTProfile2D()->GetXaxis()->SetTitle("Module Z");
  h2digierr->getTProfile2D()->GetYaxis()->SetTitle("Sector #phi");
  h2digierr->getTProfile2D()->SetMaximum(1.);
  h2digierr->getTProfile2D()->SetMinimum(QIEerrThreshold);
  h2digierr->getTProfile2D()->SetOption("colz");

  sprintf(s, "CASTORreportSummaryMap");
  h2repsum =
      ibooker.bookProfile2D(s, s, 14, 0., 14., 16, 0., 16., 100, 0, 1.e10, "");
  h2repsum->getTProfile2D()->GetXaxis()->SetTitle("Module Z");
  h2repsum->getTProfile2D()->GetYaxis()->SetTitle("Sector #phi");
  h2repsum->getTProfile2D()->SetMaximum(1.);
  h2repsum->getTProfile2D()->SetMinimum(QIEerrThreshold);
  h2repsum->getTProfile2D()->SetOption("colz");

  sprintf(s, "CASTOR BadChannelsMap");
  h2status = ibooker.book2D(s, s, 14, 0., 14., 16, 0., 16.);
  h2status->getTH2F()->GetXaxis()->SetTitle("Module Z");
  h2status->getTH2F()->GetYaxis()->SetTitle("Sector #phi");
  h2status->getTH2F()->SetOption("colz");

  sprintf(s, "CASTOR TSmax Significance Map");
  h2TSratio = ibooker.book2D(s, s, 14, 0., 14., 16, 0., 16.);
  h2TSratio->getTH2F()->GetXaxis()->SetTitle("Module Z");
  h2TSratio->getTH2F()->GetYaxis()->SetTitle("Sector #phi");
  h2TSratio->getTH2F()->SetOption("colz");

  sprintf(s, "CASTOR TSmax Significance All chan");
  hTSratio = ibooker.book1D(s, s, 105, 0., 1.05);

  sprintf(s, "DigiSize");
  hdigisize = ibooker.book1DD(s, s, 20, 0., 20.);
  sprintf(s, "ModuleZ(fC)_allTS");
  hModule = ibooker.book1D(s, s, 14, 0., 14.);
  hModule->getTH1F()->GetXaxis()->SetTitle("ModuleZ");
  hModule->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");
  sprintf(s, "Sector #phi(fC)_allTS");
  hSector = ibooker.book1D(s, s, 16, 0., 16.);
  hSector->getTH1F()->GetXaxis()->SetTitle("Sector #phi");
  hSector->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");

  st = "Castor cells avr digi(fC) per event Map TS vs Channel";
  h2QmeantsvsCh =
      ibooker.bookProfile2D(st, st + ";" + string(sTileIndex) + ";TS", 224, 0.,
                            224., 10, 0., 10., 0., 1.e10, "");
  h2QmeantsvsCh->getTProfile2D()->SetOption("colz");

  st = "Castor cells avr digiRMS(fC) per event Map TS vs Channel";
  h2QrmsTSvsCh = ibooker.book2D(st, st + ";" + string(sTileIndex) + ";TS", 224,
                                0., 224., 10, 0., 10.);
  h2QrmsTSvsCh->getTH2F()->SetOption("colz");

  sprintf(s, "CASTOR data quality");
  h2qualityMap = ibooker.book2D(s, s, 14, 0, 14, 16, 0, 16);
  h2qualityMap->getTH2F()->GetXaxis()->SetTitle("module Z");
  h2qualityMap->getTH2F()->GetYaxis()->SetTitle("Sector #phi");
  h2qualityMap->getTH2F()->SetOption("colz");

  hReport = ibooker.bookFloat("CASTOR reportSummary");

  sprintf(s, "QmeanfC_map(allTS)");
  h2QmeanMap = ibooker.book2D(s, s, 14, 0., 14., 16, 0., 16.);
  h2QmeanMap->getTH2F()->GetXaxis()->SetTitle("Module Z");
  h2QmeanMap->getTH2F()->GetYaxis()->SetTitle("Sector #phi");
  h2QmeanMap->getTH2F()->SetOption("textcolz");

  const int NEtow = 20;
  float EhadTow[NEtow + 1];
  float EMTow[NEtow + 1];
  float ETower[NEtow + 2];
  double E0tow = 500. / 1024.;
  EMTow[0] = 0.;
  EMTow[1] = E0tow;
  EhadTow[0] = 0.;
  EhadTow[1] = E0tow;
  ETower[0] = 0.;
  ETower[1] = E0tow;
  double lnBtow = log(1.8);  // 2.
  for (int j = 1; j < NEtow; j++) EMTow[j + 1] = E0tow * exp(j * lnBtow);
  for (int j = 1; j < NEtow; j++) EhadTow[j + 1] = E0tow * exp(j * lnBtow);
  for (int j = 1; j <= NEtow; j++) ETower[j + 1] = E0tow * exp(j * lnBtow);

  sprintf(s, "CASTOR_Tower_EMvsEhad(fC)");
  h2towEMvsHAD = ibooker.book2D(s, s, NEtow, EhadTow, NEtow, EMTow);
  h2towEMvsHAD->getTH2F()->GetXaxis()->SetTitle("Ehad [fC]");
  h2towEMvsHAD->getTH2F()->GetYaxis()->SetTitle("EM [fC]");
  h2towEMvsHAD->getTH2F()->SetOption("colz");

  sprintf(s, "CASTOR_TowerTotalEnergy(fC)");
  htowE = ibooker.book1D(s, s, NEtow + 1, ETower);
  htowE->getTH1F()->GetXaxis()->SetTitle("fC");

  for (int ts = 0; ts <= 1; ts++) {
    sprintf(s, "QIErms_TS=%d", ts);
    hQIErms[ts] = ibooker.book1D(s, s, 1000, 0., 100.);
    hQIErms[ts]->getTH1F()->GetXaxis()->SetTitle("QIErms(fC)");
  }

  for (int ind = 0; ind < 224; ind++)
    for (int ts = 0; ts < 10; ts++) QrmsTS[ind][ts] = QmeanTS[ind][ts] = 0.;

  return;
}

void CastorDigiMonitor::processEvent(edm::Event const& event,
                                     const CastorDigiCollection& castorDigis,
                                     const edm::TriggerResults& TrigResults,
                                     const CastorDbService& cond) {
  if (fVerbosity > 1) LogPrint("CastorDigiMonitor") << "processEvent(begin)";

  if (castorDigis.empty()) {
    for (int mod = 0; mod < 14; mod++)
      for (int sec = 0; sec < 16; sec++) h2repsum->Fill(mod, sec, 0.);
    hBX->Fill(event.bunchCrossing(), 0.);
    fillTrigRes(event, TrigResults, 0.);
    return;
  }

  float Ecell[14][16];
  for (CastorDigiCollection::const_iterator j = castorDigis.begin();
       j != castorDigis.end(); j++) {
    const CastorDataFrame digi = (const CastorDataFrame)(*j);

    int module = digi.id().module() - 1;
    int sector = digi.id().sector() - 1;
    if (ChannelStatus[module][sector] == StatusBadChannel) continue;

    int capid1 = digi.sample(0).capid();
    hdigisize->Fill(digi.size());
    double sum = 0.;
    int err = 0, err2 = 0;
    for (int i = 0; i < digi.size(); i++) {
      int capid = digi.sample(i).capid();
      int dv = digi.sample(i).dv();
      int er = digi.sample(i).er();
      int rawd = digi.sample(i).adc();
      rawd = rawd & 0x7F;
      err |= (capid != capid1) | er << 1 | (!dv) << 2;  // =0
      err2 += (capid != capid1) | er | (!dv);           // =0
      //     if(err !=0) continue;
      int ind = ModSecToIndex(module, sector);
      h2QmeantsvsCh->Fill(ind, i, LedMonAdc2fc[rawd]);
      float q = LedMonAdc2fc[rawd];
      Ecell[module][sector] = q;
      sum += q;  //     sum += LedMonAdc2fc[rawd];
      QrmsTS[ind][i] += (q * q);
      QmeanTS[ind][i] += q;
      if (err != 0 && fVerbosity > 0)
        LogPrint("CastorDigiMonitor")
            << "event/idigi=" << ievt_ << "/" << i
            << " cap=cap1_dv_er_err: " << capid << "=" << capid1 << " " << dv
            << " " << er << " " << err;
      if (capid1 < 3)
        capid1 = capid + 1;
      else
        capid1 = 0;
    }
    h2digierr->Fill(module, sector, err);
    h2repsum->Fill(module, sector, 1. - err2 / digi.size());
  }  // end for(CastorDigiCollection::const_iterator ...

  ievt_++;

  double Etotal = 0.;
  for (int sec = 0; sec < 16; sec++)
    for (int mod = 0; mod < 14; mod++) Etotal = Ecell[mod][sec];
  hBX->Fill(event.bunchCrossing(), Etotal);
  fillTrigRes(event, TrigResults, Etotal);

  for (int sec = 0; sec < 16; sec++) {
    float em = Ecell[0][sec] + Ecell[1][sec];
    double ehad = 0.;
    for (int mod = 2; mod < EtowerLastModule; mod++) ehad += Ecell[mod][sec];
    h2towEMvsHAD->Fill(em, ehad);
    htowE->Fill(em + ehad);
  }

  const float repChanBAD = 0.9;
  const float repChanWarning = 0.95;
  if (ievt_ % 100 != 0) return;

  float ModuleSum[14], SectorSum[16];
  for (int m = 0; m < 14; m++) ModuleSum[m] = 0.;
  for (int s = 0; s < 16; s++) SectorSum[s] = 0.;
  for (int mod = 0; mod < 14; mod++)
    for (int sec = 0; sec < 16; sec++) {
      for (int ts = 0; ts <= 1; ts++) {
        int ind = ModSecToIndex(mod, sec);
        double Qmean = QmeanTS[ind][ts] / ievt_;
        double Qrms = sqrt(QrmsTS[ind][ts] / ievt_ - Qmean * Qmean);
        hQIErms[ts]->Fill(Qrms);
      }

      double sum = 0.;
      for (int ts = 1; ts <= TS_MAX; ts++) {
        int ind = ModSecToIndex(mod, sec) + 1;
        double a =  //(1) h2QtsvsCh->getTH2D()->GetBinContent(ind,ts);
            h2QmeantsvsCh->getTProfile2D()->GetBinContent(ind, ts);
        sum += a;
        double Qmean = QmeanTS[ind - 1][ts - 1] / ievt_;
        double Qrms = QrmsTS[ind - 1][ts - 1] / ievt_ - Qmean * Qmean;
        h2QrmsTSvsCh->getTH2F()->SetBinContent(ind, ts, sqrt(Qrms));
      }
      ModuleSum[mod] += sum;
      SectorSum[sec] += sum;
      float isum = float(int(sum * 10. + 0.5)) / 10.;
      if (ChannelStatus[mod][sec] != StatusBadChannel)
        h2QmeanMap->getTH2F()->SetBinContent(mod + 1, sec + 1, isum);
    }  // end for(int mod=0; mod<14; mod++) for(int sec=0;...

  for (int mod = 0; mod < 14; mod++)
    hModule->getTH1F()->SetBinContent(mod + 1, ModuleSum[mod]);
  for (int sec = 0; sec < 16; sec++)
    hSector->getTH1F()->SetBinContent(sec + 1, SectorSum[sec]);

  int nGoodCh = 0;
  hTSratio->Reset();
  for (int mod = 0; mod < 14; mod++)
    for (int sec = 0; sec < 16; sec++) {
      if (ChannelStatus[mod][sec] == StatusBadChannel) continue;
      int ind = ModSecToIndex(mod, sec);
      double Qmean = QmeanTS[ind][TSped] / ievt_;
      double Qrms = QrmsTS[ind][TSped] / ievt_ - Qmean * Qmean;
      float ChanStatus = 0.;
      if (Qrms < Qrms_DEAD) ChanStatus = 1.;
      h2status->getTH2F()->SetBinContent(mod + 1, sec + 1, ChanStatus);

      float am = 0.;
      for (int ts = 0; ts < TS_MAX - 1; ts++) {
        float a =
            h2QmeantsvsCh->getTProfile2D()->GetBinContent(ind + 1, ts + 1) +
            h2QmeantsvsCh->getTProfile2D()->GetBinContent(ind + 1, ts + 2);
        if (am < a) am = a;
      }

      double sum = 0.;
      for (int ts = 0; ts < TS_MAX; ts++)
        sum += h2QmeantsvsCh->getTProfile2D()->GetBinContent(ind + 1, ts + 1);

      float r = 0.;  // worth case - no peak
      if (am > 0.) r = 1. - (sum - am) / (TS_MAX - 2) / am * 2.;
      // if(r<0.|| r>1.) cout<<"ievt="<<ievt<<" r="<<r<<" amax= "<<am<<"
      // sum="<<sum<<endl;
      h2TSratio->getTH2F()->SetBinContent(mod + 1, sec + 1, r);
      hTSratio->Fill(r);

      float statusTS = 1.0;
      if (r > RatioThresh1)
        statusTS = repChanWarning;
      else if (r > 0.99)
        statusTS = repChanBAD;
      float gChanStatus = statusTS;
      if (ChanStatus > 0.) gChanStatus = repChanBAD;  // RMS
      h2qualityMap->getTH2F()->SetBinContent(mod + 1, sec + 1, gChanStatus);
      if (gChanStatus > repChanBAD) ++nGoodCh;
    }
  hReport->Fill(float(nGoodCh) / N_GoodChannels);
  return;
}

void CastorDigiMonitor::endRun() {
  if (fVerbosity > 0)
    LogPrint("CastorDigiMonitor")
        << "DigiMonitor::endRun: trigger max index = " << TrigIndexMax
        << "  TriggerIndexies(N):" << endl;
  for (int i = 1; i < hpTrigRes->getTProfile()->GetNbinsX(); i++)
    if (hpTrigRes->getTProfile()->GetBinContent(i) > 0)
      LogPrint("CastorDigiMonitor")
          << i - 1 << "(" << hpTrigRes->getTProfile()->GetBinContent(i) << ") ";
}

void CastorDigiMonitor::fillTrigRes(edm::Event const& event,
                                    const edm::TriggerResults& TrigResults,
                                    double Etotal) {
  int nTriggers = TrigResults.size();
  const edm::TriggerNames& trigName = event.triggerNames(TrigResults);
  bool event_triggered = false;
  if (nTriggers > 0)
    for (int iTrig = 0; iTrig < nTriggers; ++iTrig) {
      if (TrigResults.accept(iTrig)) {
        int index = trigName.triggerIndex(trigName.triggerName(iTrig));
        if (TrigIndexMax < index) TrigIndexMax = index;
        if (fVerbosity > 0)
          LogPrint("CastorDigi")
              << "trigger[" << iTrig << "] name:" << trigName.triggerName(iTrig)
              << " index= " << index << endl;
        hpTrigRes->Fill(index, Etotal);
        for (int n = 0; n < int(HltPaths_.size()); n++) {
          if (trigName.triggerName(iTrig).find(HltPaths_[n]) !=
              std::string::npos)
            event_triggered = true;
        }
      }  // end if(TrigResults.accept(iTrig)
    }

  if (event_triggered) hpBXtrig->Fill(event.bunchCrossing(), Etotal);
  return;
}

void CastorDigiMonitor::getDbData(const edm::EventSetup& iSetup) {
  edm::ESHandle<CastorChannelQuality> dbChQuality;
  iSetup.get<CastorChannelQualityRcd>().get(dbChQuality);
  if (fVerbosity > 0) {
    LogPrint("CastorDigiMonitor")
        << " CastorChQuality in CondDB=" << dbChQuality.isValid() << endl;
  }

  int chInd = 0;
  for (int mod = 0; mod < 14; mod++)
    for (int sec = 0; sec < 16; sec++) ChannelStatus[mod][sec] = 0;
  std::vector<DetId> channels = dbChQuality->getAllChannels();
  N_GoodChannels = 224 - channels.size();
  if (fVerbosity > 0)
    LogPrint("CastorDigiMonitor")
        << "CastorDigiMonitor::getDBData: QualityRcdSize=" << channels.size();
  for (std::vector<DetId>::iterator ch = channels.begin(); ch != channels.end();
       ch++) {
    const CastorChannelStatus* quality = dbChQuality->getValues(*ch);
    int value = quality->getValue();
    int rawId = quality->rawId();
    chInd++;
    int mod = HcalCastorDetId(*ch).module() - 1;
    int sec = HcalCastorDetId(*ch).sector() - 1;
    if (mod > 0 && mod < 16 && sec > 0 && sec < 16)
      ChannelStatus[mod][sec] = value;
    if (fVerbosity > 0)
      LogPrint("CastorDigiMonitor")
          << chInd << " module=" << mod << " sec=" << sec << " rawId=" << rawId
          << " value=" << value << endl;
  }  // end for(std::vector<DetId>::it...
  return;
}

int CastorDigiMonitor::ModSecToIndex(int module, int sector) {
  int ind = sector + module * 16;
  if (ind > 223) ind = 223;
  return (ind);
}
