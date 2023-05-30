
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorLedAnalysis.h"
#include <TFile.h>
#include <cmath>

using namespace std;

CastorLedAnalysis::CastorLedAnalysis(const edm::ParameterSet& ps) {
  // init
  evt = 0;
  sample = 0;
  m_file = nullptr;
  char output[100]{0};
  // output files
  for (int k = 0; k < 4; k++)
    state.push_back(true);  // 4 cap-ids (do we care?)
  m_outputFileText = ps.getUntrackedParameter<string>("outputFileText", "");
  m_outputFileX = ps.getUntrackedParameter<string>("outputFileXML", "");
  if (!m_outputFileText.empty()) {
    cout << "Castor LED results will be saved to " << m_outputFileText.c_str() << endl;
    m_outFile.open(m_outputFileText.c_str());
  }
  m_outputFileROOT = ps.getUntrackedParameter<string>("outputFileHist", "");
  if (!m_outputFileROOT.empty()) {
    cout << "Castor LED histograms will be saved to " << m_outputFileROOT.c_str() << endl;
  }

  m_nevtsample = ps.getUntrackedParameter<int>("nevtsample", 9999999);
  if (m_nevtsample < 1)
    m_nevtsample = 9999999;
  m_hiSaveflag = ps.getUntrackedParameter<int>("hiSaveflag", 0);
  if (m_hiSaveflag < 0)
    m_hiSaveflag = 0;
  if (m_hiSaveflag > 0)
    m_hiSaveflag = 1;
  m_fitflag = ps.getUntrackedParameter<int>("analysisflag", 2);
  if (m_fitflag < 0)
    m_fitflag = 0;
  if (m_fitflag > 4)
    m_fitflag = 4;
  m_startTS = ps.getUntrackedParameter<int>("firstTS", 0);
  if (m_startTS < 0)
    m_startTS = 0;
  m_endTS = ps.getUntrackedParameter<int>("lastTS", 9);
  m_usecalib = ps.getUntrackedParameter<bool>("usecalib", false);
  m_logFile.open("CastorLedAnalysis.log");

  int runNum = ps.getUntrackedParameter<int>("runNumber", 999999);

  // histogram booking
  castorHists.ALLLEDS = new TH1F("Castor All LEDs", "HF All Leds", 10, 0, 9);
  castorHists.LEDRMS = new TH1F("Castor All LED RMS", "HF All LED RMS", 100, 0, 3);
  castorHists.LEDMEAN = new TH1F("Castor All LED Means", "HF All LED Means", 100, 0, 9);
  castorHists.CHI2 = new TH1F("Castor Chi2 by ndf for Landau fit", "HF Chi2/ndf Landau", 200, 0., 50.);

  //XML file header
  m_outputFileXML.open(m_outputFileX.c_str());
  snprintf(output, sizeof output, "<?xml version='1.0' encoding='UTF-8'?>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "<ROOT>");
  m_outputFileXML << output << endl << endl;
  snprintf(output, sizeof output, "  <HEADER>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "    <TYPE>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <EXTENSION_TABLE_NAME>HCAL_LED_TIMING</EXTENSION_TABLE_NAME>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <NAME>HCAL LED Timing</NAME>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "    </TYPE>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "    <RUN>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <RUN_TYPE>hcal-led-timing-test</RUN_TYPE>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <RUN_NUMBER>%06i</RUN_NUMBER>", runNum);
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <RUN_BEGIN_TIMESTAMP>2007-07-09 00:00:00.0</RUN_BEGIN_TIMESTAMP>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <COMMENT_DESCRIPTION></COMMENT_DESCRIPTION>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "    </RUN>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "  </HEADER>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "<!-- Tags secton -->");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "  <ELEMENTS>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "    <DATA_SET id='-1'/>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <IOV id='1'>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <INTERVAL_OF_VALIDITY_BEGIN>2147483647</INTERVAL_OF_VALIDITY_BEGIN>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <INTERVAL_OF_VALIDITY_END>0</INTERVAL_OF_VALIDITY_END>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      </IOV>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <TAG id='2' mode='auto'>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <TAG_NAME>laser_led_%06i<TAG_NAME>", runNum);
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <DETECTOR_NAME>HCAL</DETECTOR_NAME>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <COMMENT_DESCRIPTION></COMMENT_DESCRIPTION>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      </TAG>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "  </ELEMENTS>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "  <MAPS>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      <TAG idref ='2'>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        <IOV idref='1'>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "          <DATA_SET idref='-1' />");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "        </IOV>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "      </TAG>");
  m_outputFileXML << output << endl;
  snprintf(output, sizeof output, "  </MAPS>");
  m_outputFileXML << output << endl;
}

//-----------------------------------------------------------------------------
CastorLedAnalysis::~CastorLedAnalysis() {
  ///All done, clean up!!

  for (_meol = castorHists.LEDTRENDS.begin(); _meol != castorHists.LEDTRENDS.end(); _meol++) {
    for (int i = 0; i < 15; i++)
      _meol->second[i].first->Delete();
  }

  castorHists.ALLLEDS->Delete();
  castorHists.LEDRMS->Delete();
  castorHists.LEDMEAN->Delete();
  castorHists.CHI2->Delete();
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::LedSetup(const std::string& m_outputFileROOT) {
  // open the histogram file, create directories within
  m_file = new TFile(m_outputFileROOT.c_str(), "RECREATE");
  m_file->mkdir("Castor");
  m_file->cd();
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::GetLedConst(map<HcalDetId, map<int, LEDBUNCH> >& toolT) {
  double time2 = 0;
  double time1 = 0;
  double time3 = 0;
  double time4 = 0;
  double dtime2 = 0;
  double dtime1 = 0;
  double dtime3 = 0;
  double dtime4 = 0;
  char output[100]{0};

  if (!m_outputFileText.empty()) {
    if (m_fitflag == 0 || m_fitflag == 2)
      m_outFile << "Det Eta,Phi,D   Mean    Error" << std::endl;
    else if (m_fitflag == 1 || m_fitflag == 3)
      m_outFile << "Det Eta,Phi,D   Peak    Error" << std::endl;
    else if (m_fitflag == 4)
      m_outFile << "Det Eta,Phi,D   Mean    Error      Peak    Error       MeanEv  Error       PeakEv  Error"
                << std::endl;
  }
  for (_meol = toolT.begin(); _meol != toolT.end(); _meol++) {
    // scale the LED pulse to 1 event
    _meol->second[10].first->Scale(1. / evt_curr);
    if (m_fitflag == 0 || m_fitflag == 4) {
      time1 = _meol->second[10].first->GetMean();
      dtime1 = _meol->second[10].first->GetRMS() / sqrt((float)evt_curr * (m_endTS - m_startTS + 1));
    }
    if (m_fitflag == 1 || m_fitflag == 4) {
      // put proper errors
      for (int j = 0; j < 10; j++)
        _meol->second[10].first->SetBinError(j + 1, _meol->second[j].first->GetRMS() / sqrt((float)evt_curr));
    }
    if (m_fitflag == 1 || m_fitflag == 3 || m_fitflag == 4) {
      _meol->second[10].first->Fit("landau", "Q");
      //      _meol->second[10].first->Fit("gaus","Q");
      TF1* fit = _meol->second[10].first->GetFunction("landau");
      //      TF1 *fit = _meol->second[10].first->GetFunction("gaus");
      time2 = fit->GetParameter(1);
      dtime2 = fit->GetParError(1);
    }
    if (m_fitflag == 2 || m_fitflag == 4) {
      time3 = _meol->second[12].first->GetMean();
      dtime3 = _meol->second[12].first->GetRMS() / sqrt((float)_meol->second[12].first->GetEntries());
    }
    if (m_fitflag == 3 || m_fitflag == 4) {
      time4 = _meol->second[13].first->GetMean();
      dtime4 = _meol->second[13].first->GetRMS() / sqrt((float)_meol->second[13].first->GetEntries());
    }
    for (int i = 0; i < 10; i++) {
      _meol->second[i].first->GetXaxis()->SetTitle("Pulse height (fC)");
      _meol->second[i].first->GetYaxis()->SetTitle("Counts");
      //      if(m_hiSaveflag>0)_meol->second[i].first->Write();
    }
    _meol->second[10].first->GetXaxis()->SetTitle("Time slice");
    _meol->second[10].first->GetYaxis()->SetTitle("Averaged pulse (fC)");
    if (m_hiSaveflag > 0)
      _meol->second[10].first->Write();
    _meol->second[10].second.first[0].push_back(time1);
    _meol->second[10].second.first[1].push_back(dtime1);
    _meol->second[11].second.first[0].push_back(time2);
    _meol->second[11].second.first[1].push_back(dtime2);
    _meol->second[12].first->GetXaxis()->SetTitle("Mean TS");
    _meol->second[12].first->GetYaxis()->SetTitle("Counts");
    if (m_fitflag == 2 && m_hiSaveflag > 0)
      _meol->second[12].first->Write();
    _meol->second[12].second.first[0].push_back(time3);
    _meol->second[12].second.first[1].push_back(dtime3);
    _meol->second[13].first->GetXaxis()->SetTitle("Peak TS");
    _meol->second[13].first->GetYaxis()->SetTitle("Counts");
    if (m_fitflag > 2 && m_hiSaveflag > 0)
      _meol->second[13].first->Write();
    _meol->second[13].second.first[0].push_back(time4);
    _meol->second[13].second.first[1].push_back(dtime4);
    _meol->second[14].first->GetXaxis()->SetTitle("Peak TS error");
    _meol->second[14].first->GetYaxis()->SetTitle("Counts");
    if (m_fitflag > 2 && m_hiSaveflag > 0)
      _meol->second[14].first->Write();
    _meol->second[15].first->GetXaxis()->SetTitle("Chi2/NDF");
    _meol->second[15].first->GetYaxis()->SetTitle("Counts");
    if (m_fitflag > 2 && m_hiSaveflag > 0)
      _meol->second[15].first->Write();
    _meol->second[16].first->GetXaxis()->SetTitle("Integrated Signal");
    _meol->second[16].first->Write();

    // Ascii printout (need to modify to include new info)
    HcalDetId detid = _meol->first;

    if (!m_outputFileText.empty()) {
      if (m_fitflag == 0) {
        m_outFile << detid << "   " << time1 << " " << dtime1 << std::endl;
        snprintf(output, sizeof output, "  <DATA_SET>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <VERSION>version:1</VERSION>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ETA>%2i</ETA>", detid.ietaAbs());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <PHI>%2i</PHI>", detid.iphi());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <DEPTH>%2i</DEPTH>", detid.depth());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <Z>%2i</Z>", detid.zside());
        m_outputFileXML << output << endl;
        if (detid.subdet() == 1)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HB</DETECTOR_NAME>");
        if (detid.subdet() == 2)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HE</DETECTOR_NAME>");
        if (detid.subdet() == 3)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HO</DETECTOR_NAME>");
        if (detid.subdet() == 4)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HF</DETECTOR_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <MEAN_TIME>%7f</MEAN_TIME>", time1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <OFFSET_TIME> 0</OFFSET_TIME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag + 1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <STATUS_WORD>  0</STATUS_WORD>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "  </DATA_SET>");
        m_outputFileXML << output << endl;

      } else if (m_fitflag == 1) {
        m_outFile << detid << "   " << time2 << " " << dtime2 << std::endl;
        snprintf(output, sizeof output, "  <DATA_SET>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <VERSION>version:1</VERSION>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ETA>%2i</ETA>", detid.ietaAbs());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <PHI>%2i</PHI>", detid.iphi());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <DEPTH>%2i</DEPTH>", detid.depth());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <Z>%2i</Z>", detid.zside());
        m_outputFileXML << output << endl;
        if (detid.subdet() == 1)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HB</DETECTOR_NAME>");
        if (detid.subdet() == 2)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HE</DETECTOR_NAME>");
        if (detid.subdet() == 3)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HO</DETECTOR_NAME>");
        if (detid.subdet() == 4)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HF</DETECTOR_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <MEAN_TIME>%7f</MEAN_TIME>", time2);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <OFFSET_TIME> 0</OFFSET_TIME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime2);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag + 1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <STATUS_WORD>  0</STATUS_WORD>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "  </DATA_SET>");
        m_outputFileXML << output << endl;
      }

      else if (m_fitflag == 2) {
        m_outFile << detid << "   " << time3 << " " << dtime3 << std::endl;
        snprintf(output, sizeof output, "  <DATA_SET>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <VERSION>version:1</VERSION>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ETA>%2i</ETA>", detid.ietaAbs());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <PHI>%2i</PHI>", detid.iphi());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <DEPTH>%2i</DEPTH>", detid.depth());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <Z>%2i</Z>", detid.zside());
        m_outputFileXML << output << endl;
        if (detid.subdet() == 1)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HB</DETECTOR_NAME>");
        if (detid.subdet() == 2)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HE</DETECTOR_NAME>");
        if (detid.subdet() == 3)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HO</DETECTOR_NAME>");
        if (detid.subdet() == 4)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HF</DETECTOR_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <MEAN_TIME>%7f</MEAN_TIME>", time3);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <OFFSET_TIME> 0</OFFSET_TIME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime3);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag + 1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <STATUS_WORD>  0</STATUS_WORD>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "  </DATA_SET>");
        m_outputFileXML << output << endl;
      } else if (m_fitflag == 3) {
        m_outFile << detid << "   " << time4 << " " << dtime4 << std::endl;
        snprintf(output, sizeof output, "  <DATA_SET>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <VERSION>version:1</VERSION>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ETA>%2i</ETA>", detid.ietaAbs());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <PHI>%2i</PHI>", detid.iphi());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <DEPTH>%2i</DEPTH>", detid.depth());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <Z>%2i</Z>", detid.zside());
        m_outputFileXML << output << endl;
        if (detid.subdet() == 1)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HB</DETECTOR_NAME>");
        if (detid.subdet() == 2)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HE</DETECTOR_NAME>");
        if (detid.subdet() == 3)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HO</DETECTOR_NAME>");
        if (detid.subdet() == 4)
          snprintf(output, sizeof output, "      <DETECTOR_NAME>HF</DETECTOR_NAME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId());
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </CHANNEL>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    <DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <MEAN_TIME>%7f</MEAN_TIME>", time4);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <OFFSET_TIME> 0</OFFSET_TIME>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime4);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag + 1);
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "      <STATUS_WORD>  0</STATUS_WORD>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "    </DATA>");
        m_outputFileXML << output << endl;
        snprintf(output, sizeof output, "  </DATA_SET>");
        m_outputFileXML << output << endl;
      }

      else if (m_fitflag == 4) {
        m_outFile << detid << "   " << time1 << " " << dtime1 << "   " << time2 << " " << dtime2 << "   " << time3
                  << " " << dtime3 << "   " << time4 << " " << dtime4 << std::endl;
      }
    }
  }
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::LedSampleAnalysis() {
  // it is called every m_nevtsample events (a sample) and the end of run
  char LedSampleNum[20];

  sprintf(LedSampleNum, "LedSample_%d", sample);
  m_file->cd();
  m_file->mkdir(LedSampleNum);
  m_file->cd(LedSampleNum);

  // Compute LED constants
  GetLedConst(castorHists.LEDTRENDS);
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::LedTrendings(map<HcalDetId, map<int, LEDBUNCH> >& toolT) {
  for (_meol = toolT.begin(); _meol != toolT.end(); _meol++) {
    char name[1024];
    HcalDetId detid = _meol->first;
    sprintf(name, "LED timing trend, eta=%d phi=%d depth=%d", detid.ieta(), detid.iphi(), detid.depth());
    int bins = _meol->second[10 + m_fitflag].second.first[0].size();
    float lo = 0.5;
    float hi = (float)bins + 0.5;
    _meol->second[10 + m_fitflag].second.second.push_back(new TH1F(name, name, bins, lo, hi));

    std::vector<double>::iterator sample_it;
    // LED timing - put content and errors
    int j = 0;
    for (sample_it = _meol->second[10 + m_fitflag].second.first[0].begin();
         sample_it != _meol->second[10 + m_fitflag].second.first[0].end();
         ++sample_it) {
      _meol->second[10 + m_fitflag].second.second[0]->SetBinContent(++j, *sample_it);
    }
    j = 0;
    for (sample_it = _meol->second[10 + m_fitflag].second.first[1].begin();
         sample_it != _meol->second[10 + m_fitflag].second.first[1].end();
         ++sample_it) {
      _meol->second[10 + m_fitflag].second.second[0]->SetBinError(++j, *sample_it);
    }
    sprintf(name, "Sample (%d events)", m_nevtsample);
    _meol->second[10 + m_fitflag].second.second[0]->GetXaxis()->SetTitle(name);
    _meol->second[10 + m_fitflag].second.second[0]->GetYaxis()->SetTitle("Peak position");
    _meol->second[10 + m_fitflag].second.second[0]->Write();
  }
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::LedDone() {
  // First process the last sample (remaining events).
  if (evt % m_nevtsample != 0)
    LedSampleAnalysis();

  // Now do the end of run analysis: trending histos
  if (sample > 1 && m_fitflag != 4) {
    m_file->cd();
    m_file->cd("Castor");
    LedTrendings(castorHists.LEDTRENDS);
  }

  // Write other histograms.
  m_file->cd();
  m_file->cd("Castor");
  castorHists.ALLLEDS->Write();
  castorHists.LEDRMS->Write();
  castorHists.LEDMEAN->Write();

  // Write the histo file and close it
  //  m_file->Write();
  m_file->Close();
  cout << "Castor histograms written to " << m_outputFileROOT.c_str() << endl;
}

//-----------------------------------------------------------------------------
void CastorLedAnalysis::processLedEvent(const CastorDigiCollection& castor, const CastorDbService& cond) {
  evt++;
  sample = (evt - 1) / m_nevtsample + 1;
  evt_curr = evt % m_nevtsample;
  if (evt_curr == 0)
    evt_curr = m_nevtsample;

  // HF/Castor
  try {
    if (castor.empty())
      throw (int)castor.size();
    for (CastorDigiCollection::const_iterator j = castor.begin(); j != castor.end(); ++j) {
      const CastorDataFrame digi = (const CastorDataFrame)(*j);
      _meol = castorHists.LEDTRENDS.find(digi.id());
      if (_meol == castorHists.LEDTRENDS.end()) {
        SetupLEDHists(2, digi.id(), castorHists.LEDTRENDS);
      }
      LedCastorHists(digi.id(), digi, castorHists.LEDTRENDS, cond);
    }
  } catch (int i) {
    //    m_logFile << "Event with " << i<<" Castor Digis passed." << std::endl;
  }

  // Call the function every m_nevtsample events
  if (evt % m_nevtsample == 0)
    LedSampleAnalysis();
}
//----------------------------------------------------------------------------
void CastorLedAnalysis::SetupLEDHists(int id, const HcalDetId detid, map<HcalDetId, map<int, LEDBUNCH> >& toolT) {
  string type = "HBHE";
  if (id == 1)
    type = "HO";
  if (id == 2)
    type = "HF";

  _meol = toolT.find(detid);
  if (_meol == toolT.end()) {
    // if histos for this channel do not exist, create them
    map<int, LEDBUNCH> insert;
    char name[1024];
    for (int i = 0; i < 10; i++) {
      sprintf(name,
              "%s Pulse height, eta=%d phi=%d depth=%d TS=%d",
              type.c_str(),
              detid.ieta(),
              detid.iphi(),
              detid.depth(),
              i);
      insert[i].first = new TH1F(name, name, 200, 0., 2000.);
    }
    sprintf(name, "%s LED Mean pulse, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[10].first = new TH1F(name, name, 10, -0.5, 9.5);
    sprintf(name, "%s LED Pulse, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[11].first = new TH1F(name, name, 10, -0.5, 9.5);
    sprintf(name, "%s Mean TS, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[12].first = new TH1F(name, name, 200, 0., 10.);
    sprintf(name, "%s Peak TS, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[13].first = new TH1F(name, name, 200, 0., 10.);
    sprintf(name, "%s Peak TS error, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[14].first = new TH1F(name, name, 200, 0., 0.05);
    sprintf(name, "%s Fit chi2, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[15].first = new TH1F(name, name, 100, 0., 50.);
    sprintf(
        name, "%s Integrated Signal, eta=%d phi=%d depth=%d", type.c_str(), detid.ieta(), detid.iphi(), detid.depth());
    insert[16].first = new TH1F(name, name, 500, 0., 5000.);

    toolT[detid] = insert;
  }
}
//-----------------------------------------------------------------------------
void CastorLedAnalysis::LedCastorHists(const HcalDetId& detid,
                                       const CastorDataFrame& ledDigi,
                                       map<HcalDetId, map<int, LEDBUNCH> >& toolT,
                                       const CastorDbService& cond) {
  map<int, LEDBUNCH> _mei;
  _meol = toolT.find(detid);
  _mei = _meol->second;
  // Rest the histos if we're at the end of a 'bunch'
  if ((evt - 1) % m_nevtsample == 0 && state[0]) {
    for (int k = 0; k < (int)state.size(); k++)
      state[k] = false;
    for (int i = 0; i < 16; i++)
      _mei[i].first->Reset();
  }

  // now we have the signal in fC, let's get rid of that darn pedestal
  // Most of this is borrowed from CastorSimpleReconstructor, so thanks Jeremy/Phil

  float max_fC = 0;
  float ta = 0;
  m_coder = cond.getCastorCoder(detid);
  m_ped = cond.getPedestal(detid);
  m_shape = cond.getCastorShape();
  //cout << "New Digi!!!!!!!!!!!!!!!!!!!!!!" << endl;
  for (int TS = m_startTS; TS < m_endTS && TS < ledDigi.size(); TS++) {
    int capid = ledDigi[TS].capid();
    // BE CAREFUL: this is assuming peds are stored in ADCs
    int adc = (int)(ledDigi[TS].adc() - m_ped->getValue(capid));
    if (adc < 0) {
      adc = 0;
    }  // to prevent negative adcs after ped subtraction, which should really only happen
       // if you're using the wrong peds.
    double fC = m_coder->charge(*m_shape, adc, capid);
    //ta = (fC - m_ped->getValue(capid));
    ta = fC;
    //cout << "DetID: " << detid << "  CapID: " << capid << "  ADC: " << adc << "  Ped: " << m_ped->getValue(capid) << "  fC: " << fC << endl;
    _mei[TS].first->Fill(ta);
    _mei[10].first->AddBinContent(TS + 1, ta);  // This is average pulse, could probably do better (Profile?)
    if (m_fitflag > 1) {
      if (TS == m_startTS)
        _mei[11].first->Reset();
      _mei[11].first->SetBinContent(TS + 1, ta);
    }

    // keep track of max TS and max amplitude (in fC)
    if (ta > max_fC) {
      max_fC = ta;
    }
  }

  // Now we have a sample with pedestals subtracted and in units of fC
  // If we are using a weighted mean (m_fitflag = 2) to extraxt timing
  // we now want to use Phil's timing correction.  This is not necessary
  // if we are performing a Landau fit (m_fitflag = 3)

  float sum = 0.;
  for (int i = 0; i < 10; i++)
    sum = sum + _mei[11].first->GetBinContent(i + 1);
  if (sum > 100) {
    if (m_fitflag == 2 || m_fitflag == 4) {
      float timmean = _mei[11].first->GetMean();  // let's use Phil's way instead
      float timmeancorr = BinsizeCorr(timmean);
      _mei[12].first->Fill(timmeancorr);
    }
    _mei[16].first->Fill(
        _mei[11].first->Integral());  // Integrated charge (may be more usfull to convert to Energy first?)
    if (m_fitflag == 3 || m_fitflag == 4) {
      _mei[11].first->Fit("landau", "Q");
      TF1* fit = _mei[11].first->GetFunction("landau");
      _mei[13].first->Fill(fit->GetParameter(1));
      _mei[14].first->Fill(fit->GetParError(1));
      _mei[15].first->Fill(fit->GetChisquare() / fit->GetNDF());
    }
  }
}

//-----------------------------------------------------------------------------
float CastorLedAnalysis::BinsizeCorr(float time) {
  // this is the bin size correction to be applied for laser data (from Andy),
  // it comes from a pulse shape measured from TB04 data (from Jordan)
  // This should eventually be replaced with the more thorough treatment from Phil

  float corrtime = 0.;
  static const float tstrue[32] = {0.003, 0.03425, 0.06548, 0.09675, 0.128, 0.15925, 0.1905, 0.22175,
                                   0.253, 0.28425, 0.3155,  0.34675, 0.378, 0.40925, 0.4405, 0.47175,
                                   0.503, 0.53425, 0.5655,  0.59675, 0.628, 0.65925, 0.6905, 0.72175,
                                   0.753, 0.78425, 0.8155,  0.84675, 0.878, 0.90925, 0.9405, 0.97175};
  static const float tsreco[32] = {-0.00422, 0.01815, 0.04409, 0.07346, 0.09799, 0.12192, 0.15072, 0.18158,
                                   0.21397,  0.24865, 0.28448, 0.31973, 0.35449, 0.39208, 0.43282, 0.47244,
                                   0.5105,   0.55008, 0.58827, 0.62828, 0.6717,  0.70966, 0.74086, 0.77496,
                                   0.80843,  0.83472, 0.86044, 0.8843,  0.90674, 0.92982, 0.95072, 0.9726};

  int inttime = (int)time;
  float restime = time - inttime;
  for (int i = 0; i <= 32; i++) {
    float lolim = 0.;
    float uplim = 1.;
    float tsdown;
    float tsup;
    if (i > 0) {
      lolim = tsreco[i - 1];
      tsdown = tstrue[i - 1];
    } else
      tsdown = tstrue[31] - 1.;
    if (i < 32) {
      uplim = tsreco[i];
      tsup = tstrue[i];
    } else
      tsup = tstrue[0] + 1.;
    if (restime >= lolim && restime < uplim) {
      corrtime = (tsdown * (uplim - restime) + tsup * (restime - lolim)) / (uplim - lolim);
    }
  }
  corrtime += inttime;

  return corrtime;
}
//-----------------------------------------------------------------------------
