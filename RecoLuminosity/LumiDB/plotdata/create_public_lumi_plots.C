#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>

#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TGaxis.h"
#include "TTimeStamp.h"
#include "TColor.h"
#include "TH1F.h"
#include "TImage.h"
#include "TPaveLabel.h"

// NOTE: All the below colors have been tweaked based on the PNG
// versions of the CMS logos present in this CVS area.

//------------------------------
// Color scheme 'Greg'.
//------------------------------

Int_t const kCMSBlue = 1756;
Int_t const kCMSOrange = 1757;
Int_t const kCMSBlueD = 1759;
Int_t const kCMSOrangeD = 1760;

// This is the light blue of the CMS logo.
TColor cmsBlue(kCMSBlue, 0./255., 152./255., 212./255.);

// This is the orange from the CMS logo.
TColor cmsOrange(kCMSOrange, 241./255., 194./255., 40./255.);

// Slightly darker versions of the above colors for the lines.
TColor cmsBlueD(kCMSBlueD, 102./255., 153./255., 204./255.);
TColor cmsOrangeD(kCMSOrangeD, 255./255., 153./255., 0./255.);

//------------------------------
// Color scheme 'Joe'.
//------------------------------

// Several colors from the alternative CMS logo, with their darker
// line variants.

Int_t const kCMSRed = 1700;
Int_t const kCMSYellow = 1701;
Int_t const kCMSPurple = 1702;
Int_t const kCMSGreen = 1703;
Int_t const kCMSOrange2 = 1704;

TColor cmsRed(kCMSRed, 208./255., 0./255., 37./255.);
TColor cmsYellow(kCMSYellow, 255./255., 248./255., 0./255.);
TColor cmsPurple(kCMSPurple, 125./255., 16./255., 123./255.);
TColor cmsGreen(kCMSGreen, 60./255., 177./255., 110./255.);
TColor cmsOrange2(kCMSOrange2, 227./255., 136./255., 36./255.);

//------------------------------

std::string replaceInString(std::string const stringIn,
                            std::string const stringSearch,
                            std::string const stringReplace) {
  std::string stringOut(stringIn);
  if (stringSearch != stringReplace) {
    string::size_type pos = 0;
    while ((pos = stringOut.find(stringSearch, pos)) != string::npos) {
      stringOut.replace(pos, stringSearch.size(), stringReplace);
      ++pos;
    }
  }
  return stringOut;
}

void rootInit() {
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetHistFillStyle(1001);
  gStyle->SetHistFillStyle(1001);
  gStyle->SetHistFillColor(51);
  gStyle->SetHistLineWidth(2);
  gStyle->SetFrameFillColor(0);
  gStyle->SetTitleW(0.65);
  gStyle->SetTitleH(0.08);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleAlign(23);
  gStyle->SetStatW(0.25);
  gStyle->SetStatH(0.2);
  gStyle->SetStatColor(0);
  gStyle->SetHistFillStyle(5101);
  gStyle->SetEndErrorSize(0);
  gStyle->SetPalette(1);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetGridStyle(0);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFillColor(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetFillStyle(4000);
}

TCanvas* createCanvas() {
  return new TCanvas("canvas", "Canvas", 10, 10, 1800, 1400);
}

TLegend* createLegend(float const shift_x=0., float const shift_y=0.) {
  float width = .5;
  float height = .1;
  float min_x = .2 + shift_x;
  float max_x = min_x + width;
  float min_y = .78 + shift_y;
  float max_y = min_y + height;
  return new TLegend(min_x, min_y, max_x, max_y);
}

void drawLogo(TCanvas* canvas, std::string const logoName) {
  TImage* logo = TImage::Open(logoName.c_str());
  float aspectRatio = 1. * canvas->GetWh() / canvas->GetWw();
  float min_x = .1005;
  float max_x = .2;
  float max_y = .899;
  float min_y = max_y - (max_x - min_x) / aspectRatio;
  TPad* p = new TPad("p", "p", min_x, min_y, max_x, max_y);
  p->SetMargin(.02, .01, .01, .02);
  p->SetBorderSize(0.);
  p->SetFillColor(0);
  canvas->cd();
  p->Draw();
  p->cd();
  logo->Draw();
}

void drawDateLabel(TCanvas* canvas,
                   std::string const start_time,
                   std::string const end_time) {
  std::string date_str("Data included from ");
  date_str += start_time;
  date_str += std::string(" to ");
  date_str += end_time;
  date_str += std::string(" UTC");
  float x_offset = .4;
  float y_lo = .82;
  float height = .2;
  TPaveLabel* label = new TPaveLabel(.5 - x_offset, y_lo,
                                     .5 + x_offset, y_lo + height,
                                     date_str.c_str(), "NDC");
  label->SetBorderSize(0.);
  label->SetFillColor(0);
  label->Draw();
}

void duplicateYAxis(TCanvas* const canvas,
                    TAxis const* const axOri) {
  canvas->Update();
  TGaxis* secAxis = new TGaxis(canvas->GetUxmax(), canvas->GetUymin(),
                               canvas->GetUxmax(), canvas->GetUymax(),
                               canvas->GetUymin(), canvas->GetUymax(),
                               axOri->GetNdivisions(), "+L");
  secAxis->SetLineColor(axOri->GetAxisColor());
  secAxis->SetLabelColor(axOri->GetLabelColor());
  secAxis->SetLabelFont(axOri->GetLabelFont());
  secAxis->SetLabelOffset(axOri->GetLabelOffset());
  secAxis->SetLabelSize(axOri->GetLabelSize());
  secAxis->Draw();
}

TTimeStamp timestampFromString(std::string const& input) {
  // NOTE: This does assume a certain date/time format of the string!
  std::string tmpStr = input.c_str();
  TTimeStamp timestamp(atoi(tmpStr.substr(0, 4).c_str()),
                       atoi(tmpStr.substr(5, 2).c_str()),
                       atoi(tmpStr.substr(8, 2).c_str()),
                       atoi(tmpStr.substr(11, 2).c_str()),
                       atoi(tmpStr.substr(14, 2).c_str()),
                       atoi(tmpStr.substr(17, 2).c_str()),
                       0, true, 0);
  return timestamp;
}

TTimeStamp zeroTimeInTimestamp(TTimeStamp const& input) {
  std::string tmpStr(input.AsString("s"));
  TTimeStamp timestamp(atoi(tmpStr.substr(0, 4).c_str()),
                       atoi(tmpStr.substr(5, 2).c_str()),
                       atoi(tmpStr.substr(8, 2).c_str()),
                       0, 0, 0,
                       0, true, 0);
  return timestamp;
}

bool is_leap(int year) {
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

static int _days_in_month[] = {
  0, /* unused; this vector uses 1-based indexing */
  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

static int _days_before_month[] = {
  0, /* unused; this vector uses 1-based indexing */
  0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
};

int days_in_month(int year, int month) {
  assert(month >= 1);
  assert(month <= 12);
  if (month == 2 && is_leap(year))
    return 29;
  else
    return _days_in_month[month];
}

int days_before_month(int year, int month) {
  int days;
  assert(month >= 1);
  assert(month <= 12);
  days = _days_before_month[month];
  if (month > 2 && is_leap(year))
    ++days;
  return days;
}

TTimeStamp timestampFromOrdinal(int ordinal) {
  // This is tuned to do one thing: reconstruct the date from an
  // integer stored by Python's datetime.toordinal() method.

  // NOTE: This is basically a stripped-down version of the CPython
  // implementation of datetime.fromordinal().

  // DEBUG DEBUG DEBUG
  assert(ordinal >= 1);
  // DEBUG DEBUG DEBUG end

#define DI4Y      1461
#define DI100Y   36524
#define DI400Y  146097

  int n, n1, n4, n100, n400, leapyear, preceding;
  int year, month, day;

  --ordinal;
  n400 = ordinal / DI400Y;
  n = ordinal % DI400Y;
  year = n400 * 400 + 1;

  n100 = n / DI100Y;
  n = n % DI100Y;

  n4 = n / DI4Y;
  n = n % DI4Y;

  n1 = n / 365;
  n = n % 365;

  year += n100 * 100 + n4 * 4 + n1;
  if (n1 == 4 || n100 == 4) {
    // DEBUG DEBUG DEBUG
    assert(n == 0);
    // DEBUG DEBUG DEBUG end
    year -= 1;
    month = 12;
    day = 31;
  } else {

    leapyear = n1 == 3 && (n4 != 24 || n100 == 3);
    // DEBUG DEBUG DEBUG
    assert(leapyear == is_leap(year));
    // DEBUG DEBUG DEBUG end
    month = (n + 50) >> 5;
    preceding = (_days_before_month[month] + (month > 2 && leapyear));
    if (preceding > n) {
      month -= 1;
      preceding -= days_in_month(year, month);
    }
    n -= preceding;
    // DEBUG DEBUG DEBUG
    assert(0 <= n);
    assert(n < days_in_month(year, month));
    // DEBUG DEBUG DEBUG end

    day = n + 1;
  }

  TTimeStamp timestamp(year, month, day,
                       0, 0, 0, 0, true);
  return timestamp;
}

void readInputFileIntLumi(std::string& fileName,
                          std::vector<int>& runV,
                          std::vector<std::string>& start_timeV,
                          std::vector<std::string>& end_timeV,
                          std::vector<float>& delivered_lumiV,
                          std::vector<float>& recorded_lumiV) {

  runV.clear();
  start_timeV.clear();
  end_timeV.clear();
  delivered_lumiV.clear();
  recorded_lumiV.clear();

  char line[200];
  std::string lineS;
  size_t const i = 50;
  char runC[i], start_timeC[i], end_timeC[i],
    delivered_lumiC[i], recorded_lumiC[i];
  FILE* file = fopen(fileName.c_str(), "rt");
  if (!file) {
    std::cerr << "ERROR Could not open file '" << fileName << "'" << std::endl;
  } else {
    int runI;
    float delivered_lumiF, recorded_lumiF;
    while (fgets(line, 100, file)) {
      if (line[0] != '#') {
        sscanf(line, "%[^','],%[^','],%[^','],%[^','],%[^',']",
               runC, start_timeC, end_timeC, delivered_lumiC, recorded_lumiC);
        runI = atoi(runC);
        std::string start_timeS(start_timeC);
        std::string end_timeS(end_timeC);
        delivered_lumiF = atof(delivered_lumiC);
        recorded_lumiF = atof(recorded_lumiC);
        runV.push_back(runI);
        start_timeV.push_back(start_timeS);
        end_timeV.push_back(end_timeS);
        delivered_lumiV.push_back(delivered_lumiF);
        recorded_lumiV.push_back(recorded_lumiF);
      }
    }
    fclose(file);
  }
}

void readInputFilePeakLumi(std::string const& fileName,
                           std::vector<int>& dayV,
                           std::vector<int>& runV,
                           std::vector<int>& lsV,
                           std::vector<float>& lumiV) {

  dayV.clear();
  runV.clear();
  lsV.clear();
  lumiV.clear();

  char line[200];
  size_t const i = 50;
  char dayC[i], runC[i], lsC[i], lumiC[i];
  int dayI;
  int runI;
  int lsI;
  float lumiF;
  FILE* file = fopen(fileName.c_str(), "rt");
  if (!file) {
    std::cerr << "ERROR Could not open file '" << fileName << "'" << std::endl;
  } else {
    while (fgets(line, 100, file)) {
      if (line[0] != '#') {
        sscanf(line, "%[^','],%[^','],%[^','],%[^',']",
               dayC, runC, lsC, lumiC);
        dayI = atoi(dayC);
        runI = atoi(runC);
        lsI = atoi(lsC);
        lumiF = atof(lumiC);
        TTimeStamp t = timestampFromOrdinal(dayI);
        dayV.push_back(timestampFromOrdinal(dayI).GetSec());
        runV.push_back(runI);
        lsV.push_back(lsI);
        lumiV.push_back(lumiF);
      }
    }
    fclose(file);
  }
}

void mapIntLumiToDays(std::vector<int>& runV,
                      std::vector<std::string> start_timeV,
                      std::vector<std::string> end_timeV,
                      std::vector<float>& delivered_lumiV,
                      std::vector<float>& recorded_lumiV,
                      float* delivered_lumiA, float* recorded_lumiA) {

  // Figure out the time span of the data in days.
  TTimeStamp tmpLo = timestampFromString(start_timeV.front());
  TTimeStamp tmpHi = timestampFromString(end_timeV.back());
  TTimeStamp dateLo = zeroTimeInTimestamp(tmpLo);
  TTimeStamp dateHi = zeroTimeInTimestamp(tmpHi);
  int numDays = ((dateHi.GetSec() - dateLo.GetSec()) / (24 * 60 * 60)) + 1;

  // Map luminosities on to days.
  for(size_t ind = 0; ind < runV.size(); ++ind) {
    TTimeStamp timeStart = timestampFromString(start_timeV.at(ind));
    TTimeStamp timeEnd = timestampFromString(end_timeV.at(ind).c_str());

    Int_t dayDiff = (timeEnd.GetSec() - timeStart.GetSec()) / (60 * 60 * 24);
    //Int_t dayDiff = timeEnd.GetDate() - timeStart.GetDate();
    // DEBUG DEBUG DEBUG
    // This assumes we don't take runs longer than 24 hours.
    assert(dayDiff >= 0);
    assert(dayDiff < 2);
    // DEBUG DEBUG DEBUG end

    time_t start = dateLo.GetSec();
    time_t dayIndex = (timeStart.GetSec() - start) / (24 * 60 * 60);

    if (dayDiff == 0) {
      // Whole run is contained in a single day.
      delivered_lumiA[dayIndex] += delivered_lumiV.at(ind);
      recorded_lumiA[dayIndex] += recorded_lumiV.at(ind);
    } else {
      // Run runs across midnight, need to split the lumi across two
      // days.
      // DEBUG DEBUG DEBUG
      assert((dayIndex + 1) < numDays);
      // DEBUG DEBUG DEBUG end
      TTimeStamp timeMid(timeStart);
      timeMid = zeroTimeInTimestamp(timeMid);
      timeMid.SetSec(timeMid.GetSec() + (24 * 60 * 60));
      float frac1 = 1. * (timeMid.GetSec() - timeStart.GetSec()) /
        (timeEnd.GetSec() - timeStart.GetSec());
      float frac2 = 1. * (timeEnd.GetSec() - timeMid.GetSec()) /
        (timeEnd.GetSec() - timeStart.GetSec());
      // DEBUG DEBUG DEBUG
      assert(abs(frac1 + frac2 - 1.) < 1.e-9);
      // DEBUG DEBUG DEBUG end
      float tmpDel = delivered_lumiV.at(ind);
      float tmpRec = recorded_lumiV.at(ind);
      delivered_lumiA[dayIndex] += frac1 * tmpDel;
      recorded_lumiA[dayIndex] += frac1 * tmpRec;
      delivered_lumiA[dayIndex + 1] += frac2 * tmpDel;
      recorded_lumiA[dayIndex + 1] += frac2 * tmpRec;
    }
  }
}

void create_plots(std::string const colorScheme="Greg", int const year=2012,
                  std::string const dataPath="/afs/cern.ch/cms/lumi/www/publicplots/",
                  std::string const beamType="PROTPHYS") {

  std::string partType = "?";
  std::string partTypeStr = "?";
  if (beamType == "PROTPHYS") {
    partType = "pp";
    partTypeStr = "p-p";
  } else if (beamType == "IONPHYS") {
    partType = "pbpb";
    partTypeStr = "Pb-Pb";
  } else {
    std::cerr << "ERROR Unknown beam type: " << beamType << std::endl;
  }

  std::string eBeam = "?";
  std::string units = "?";
  std::string units2 = "?";
  std::string nucleon = "";
  // Conversion factor to go to inverse femtobarn or inverse picobarn
  // depending on the beam type.
  float conversionFactor = 0;
  float scaleFactor = 0;
  float scaleFactor2 = 0;
  switch (year) {
  case 2010:
    if (partType == "pp") {
      eBeam = "7 TeV";
      units = "fb";
      units2 = "pb";
      conversionFactor = 1.e6;
      scaleFactor = 1.e-3;
      scaleFactor2 = 1.e-3;
    } else {
      std::cerr << "ERROR Unknown beam type for 2010: " << beamType << std::endl;
    }
    break;
  case 2011:
    if (partType == "pp") {
      eBeam = "7 TeV";
      units = "fb";
      units2 = "pb";
      conversionFactor = 1.e6;
      scaleFactor = 1.e-3;
      scaleFactor2 = 1.e-3;
    } else if (partType == "pbpb") {
      eBeam = "2.76 TeV";
      units = "#mub";
      units2 = "#mub";
      conversionFactor = 1.;
      scaleFactor = 1.;
      scaleFactor2 = 1.e3;
      nucleon = "/nucleon";
    } else {
      std::cerr << "ERROR Unknown beam type for 2011: " << beamType << std::endl;
    }
    break;
  case 2012:
    if (partType == "pp") {
      eBeam = "8 TeV";
      units = "fb";
      units2 = "pb";
      conversionFactor = 1.e6;
      scaleFactor = 1.e-3;
      scaleFactor2 = 1.e-3;
    } else {
      std::cerr << "ERROR Unknown beam type for 2012: " << beamType << std::endl;
    }
    break;
  default:
    std::cerr << "ERROR Unknown year: " << year << std::endl;
  }

  // Overall title and axis titles for everything. One version for the
  // per-day plot, one for the cumulative plot, one for the peak lumi
  // plot.
  std::string titlePerDay =
    std::string(Form("CMS Integrated Luminosity Per Day, %d, %s, #sqrt{s} = %s%s;",
                     year, partTypeStr.c_str(), eBeam.c_str(), nucleon.c_str())) +
    std::string("Date;") +
    std::string(Form("Integrated Luminosity (%s^{-1}/day)", units2.c_str()));
  std::string titleCumulative =
    std::string(Form("CMS Total Integrated Luminosity, %d, %s, #sqrt{s} = %s%s;",
                     year, partTypeStr.c_str(), eBeam.c_str(), nucleon.c_str())) +
    std::string("Date;") +
    std::string(Form("Total Integrated Luminosity (%s^{-1})", units.c_str()));
  std::string titlePeak =
    std::string(Form("CMS Peak Luminosity Per Day, %d, %s, #sqrt{s} = %s%s;",
                     year, partTypeStr.c_str(), eBeam.c_str(), nucleon.c_str())) +
    std::string("Date;") +
    std::string("Peak Delivered Luminosity (Hz/nb)");
  std::string titleCumulativeYears =
    std::string(Form("CMS Total Integrated Luminosity, %s;",
                     partTypeStr.c_str(), nucleon.c_str())) +
    std::string("Time in year;") +
    std::string(Form("Total Integrated Luminosity (%s^{-1})", units.c_str()));

  // This is the intermediate CSV file with the integrated lumi data
  // from the lumi DB
  std::string fileNameIntLumi = Form("%s/totallumivstime-%s-%d.csv",
                                     dataPath.c_str(), partType.c_str(), year);
  // Same for the peak lumi file.
  std::string fileNamePeakLumi = Form("%s/lumipeak-%s-%d.csv",
                                      dataPath.c_str(), partType.c_str(), year);

  // Basic style settings.
  rootInit();

  //------------------------------

  // Color scheme settings.
  Int_t kFillColorDelivered = 0;
  Int_t kFillColorRecorded = 0;
  Int_t kFillColorPeak = 0;
  Int_t kLineColorDelivered = 0;
  Int_t kLineColorRecorded = 0;
  Int_t kLineColorPeak = 0;
  std::string fileSuffix = "";
  std::string logoName = "cms_logo_1.png";

  if (colorScheme == "Greg") {
    // Color scheme 'Greg'.
    kFillColorDelivered = kCMSBlue;
    kFillColorRecorded = kCMSOrange;
    kFillColorPeak = kCMSOrange;
    kLineColorDelivered = TColor::GetColorDark(kFillColorDelivered);
    kLineColorRecorded = TColor::GetColorDark(kFillColorRecorded);
    kLineColorPeak = TColor::GetColorDark(kFillColorPeak);
    logoName = "cms_logo_2.png";
    fileSuffix = "";
  } else if (colorScheme == "Joe") {
    // Color scheme 'Joe'.
    kFillColorDelivered = kCMSYellow;
    kFillColorRecorded = kCMSRed;
    kFillColorPeak = kCMSRed;
    kLineColorDelivered = TColor::GetColorDark(kFillColorDelivered);
    kLineColorRecorded = TColor::GetColorDark(kFillColorRecorded);
    kLineColorPeak = TColor::GetColorDark(kFillColorPeak);
    logoName = "cms_logo_3.png";
    fileSuffix = "_alt";
  } else {
    std::cerr << "ERROR Unknown color scheme '"
              << colorScheme << "' --> using the default ('Greg')" << std::endl;
  }

  //------------------------------

  // Read in the integrated lumi data.

  std::vector<int> runV;
  std::vector<std::string> start_timeV;
  std::vector<std::string> end_timeV;
  std::vector<float> delivered_lumiV;
  std::vector<float> recorded_lumiV;

  readInputFileIntLumi(fileNameIntLumi, runV, start_timeV, end_timeV,
                       delivered_lumiV, recorded_lumiV);

  // Scale everything.
  // DEBUG DEBUG DEBUG
  assert(delivered_lumiV.size() == recorded_lumiV.size());
  // DEBUG DEBUG DEBUG end
  for (size_t ijk = 0; ijk != delivered_lumiV.size(); ++ijk) {
    delivered_lumiV.at(ijk) = delivered_lumiV.at(ijk) / conversionFactor;
    recorded_lumiV.at(ijk) = recorded_lumiV.at(ijk) / conversionFactor;
  }

  // Figure out the time span of the data in days.
  TTimeStamp tmpLo = timestampFromString(start_timeV.front());
  TTimeStamp tmpHi = timestampFromString(end_timeV.back());
  TTimeStamp dateLo = zeroTimeInTimestamp(tmpLo);
  TTimeStamp dateHi = zeroTimeInTimestamp(tmpHi);
  int numDays = ((dateHi.GetSec() - dateLo.GetSec()) / (24 * 60 * 60)) + 1;

  int const nBins = numDays;
  float delivered_lumiA[nBins];
  float recorded_lumiA[nBins];

  // Zero the arrays before use!
  for (int iTmp = 0; iTmp < numDays; ++iTmp) {
    delivered_lumiA[iTmp] = 0.;
    recorded_lumiA[iTmp] = 0.;
  }

  mapIntLumiToDays(runV, start_timeV, end_timeV,
                   delivered_lumiV, recorded_lumiV,
                   delivered_lumiA, recorded_lumiA);

  // Figure out the maxima.
  float maxDel = -1;
  float maxRec = -1;
  for (int day = 0; day < numDays; ++day) {
    maxDel = max(delivered_lumiA[day], maxDel);
    maxRec = max(recorded_lumiA[day], maxRec);
  }

  TTimeStamp a(dateLo);
  TTimeStamp b(dateHi);
  // NOTE: Watch out with this magic. It centers the bins on the days.
  b.SetSec(b.GetSec() + (24 * 60 * 60));
  a.SetSec(a.GetSec() - (12 * 60 * 60) - (60 * 60));
  b.SetSec(b.GetSec() - (12 * 60 * 60) - (60 * 60));
  TH1F h_delLum("", "", numDays, a.GetSec(), b.GetSec());
  TH1F h_recLum("", "", numDays, a.GetSec(), b.GetSec());

  for (int i = 0; i != numDays; ++i) {
    h_delLum.SetBinContent(i + 1, delivered_lumiA[i]);
    h_recLum.SetBinContent(i + 1, recorded_lumiA[i]);
  }

  //------------------------------

  // Read in the peak lumi data.

  std::vector<int> dayV;
  std::vector<int> dummy1, dummy2;
  std::vector<float> peakLumiV;
  readInputFilePeakLumi(fileNamePeakLumi, dayV, dummy1, dummy2, peakLumiV);

  int peakDayStart = dayV.front();
  int peakDayEnd = dayV.back();
  int numDaysPeakLumi = (peakDayEnd - peakDayStart) / (24 * 60 * 60) + 1;

  // DEBUG DEBUG DEBUG
  assert(numDaysPeakLumi == numDays);
  // DEBUG DEBUG DEBUG end

  peakDayStart -= (12 * 60 * 60);
  peakDayEnd += (12 * 60 * 60);

  TH1F hPeakLum("", "", numDaysPeakLumi, peakDayStart, peakDayEnd);
  for (size_t j = 0; j < dayV.size(); ++j) {
    hPeakLum.Fill(dayV.at(j), peakLumiV.at(j));
  }

  //------------------------------

  // Now we can move on to the plotting.

  //------------------------------
  // Create the lumi-per-day plot.
  //------------------------------

  h_delLum.SetLineColor(kLineColorDelivered);
  h_delLum.SetMarkerColor(kLineColorDelivered);
  h_delLum.SetFillColor(kFillColorDelivered);

  h_recLum.SetLineColor(kLineColorRecorded);
  h_recLum.SetMarkerColor(kLineColorRecorded);
  h_recLum.SetFillColor(kFillColorRecorded);

  h_delLum.SetLineWidth(2);
  h_recLum.SetLineWidth(2);

  // Titles etc.
  h_delLum.SetTitle(titlePerDay.c_str());
  h_delLum.GetXaxis()->SetTimeDisplay(1);
  h_delLum.GetXaxis()->SetTimeFormat("%d/%m");
  h_delLum.GetXaxis()->SetTimeOffset(0, "gmt");
  h_delLum.GetXaxis()->SetLabelOffset(0.01);
  h_delLum.GetYaxis()->SetTitleOffset(1.2);
  h_delLum.GetXaxis()->SetTitleFont(62);
  h_delLum.GetYaxis()->SetTitleFont(62);
  h_delLum.GetXaxis()->SetNdivisions(705);

  // Tweak the axes ranges a bit to create a bit more 'air.'
  float airSpace = .2;
  float min_y = 0.;
  // Round to next multiple of ten.
  float tmp = (1. + airSpace) * max(maxDel, maxRec);
  float max_y = ceil(tmp / 10.) * 10;
  TCanvas* canvas = createCanvas();
  h_delLum.Draw();
  h_recLum.Draw("SAME");
  h_delLum.GetYaxis()->SetRangeUser(min_y, max_y);

  // Add legend to the top left.
  TLegend* legend = createLegend();
  float marginOld = legend->GetMargin();
  legend->SetX2NDC(legend->GetX2NDC() +
                   1.01 * (legend->GetX2NDC() - legend->GetX1NDC()));
  legend->SetMargin(marginOld / 1.01);
  legend->AddEntry(&h_delLum,
                   Form("LHC Delivered, max: %6.1f %s^{-1}/day",
                        units2.c_str(), maxDel), "F");
  legend->AddEntry(&h_recLum,
                   Form("CMS Recorded, max: %6.1f %s^{-1}/day",
                        units2.c_str(), maxRec), "F");
  legend->Draw();

  // Duplicate the vertical axis on the right-hand side.
  duplicateYAxis(canvas, h_delLum.GetYaxis());

  // Add a label specifying up until when data taken was included in
  // this plot.
  drawDateLabel(canvas, end_timeV.front(), end_timeV.back());

  // Redraw the axes. This way the graphs don't overshoot on top of
  // the axes any more.
  canvas->RedrawAxis();

  // Add the CMS logo in the top right corner. This has to be the last
  // action so the logo sits on top of the axes.
  drawLogo(canvas, logoName);

  canvas->Print(Form("int_lumi_per_day_%s_%d%s.png",
                     partType.c_str(), year, fileSuffix.c_str()));

  delete legend;
  delete canvas;

  //------------------------------
  // Create the cumulative lumi plot.
  //------------------------------

  TH1F* h_delLumCum = dynamic_cast<TH1F*>(h_delLum.Clone());
  TH1F* h_recLumCum = dynamic_cast<TH1F*>(h_recLum.Clone());
  double cumDel = 0.;
  double cumRec = 0.;
  for (int bin = 1; bin != h_delLum.GetNbinsX() + 1; ++bin) {
    cumDel += h_delLum.GetBinContent(bin);
    h_delLumCum->SetBinContent(bin, cumDel);
    cumRec += h_recLum.GetBinContent(bin);
    h_recLumCum->SetBinContent(bin, cumRec);
  }

  // Scale to reduce the vertical labels a bit.
  h_delLumCum->Scale(scaleFactor);
  h_recLumCum->Scale(scaleFactor);

  h_delLumCum->SetTitle(titleCumulative.c_str());

  canvas = createCanvas();
  h_delLumCum->Draw();
  h_recLumCum->Draw("SAME");

  float sumDel = h_delLumCum->GetBinContent(h_delLumCum->GetNbinsX());
  float sumRec = h_recLumCum->GetBinContent(h_recLumCum->GetNbinsX());

  // Tweak the axes ranges a bit to create a bit more 'air.'
  airSpace = .2;
  min_y = 0.;
  tmp = (1. + airSpace) * sumDel;
  max_y = tmp;
  h_delLumCum->Draw();
  h_recLumCum->Draw("SAME");
  h_delLumCum->GetYaxis()->SetRangeUser(min_y, max_y);

  // Add legend to the top left.
  legend = createLegend();
  legend->AddEntry(h_delLumCum,
                   Form("LHC Delivered: %.2f %s^{-1}", units.c_str(), sumDel),
                   "F");
  legend->AddEntry(h_recLumCum,
                   Form("CMS Recorded: %.2f %s^{-1}", units.c_str(), sumRec),
                   "F");
  legend->Draw();

  // Duplicate the vertical axis on the right-hand side.
  duplicateYAxis(canvas, h_delLumCum->GetYaxis());

  // Add a label specifying up until when data taken was included in
  // this plot.
  drawDateLabel(canvas, end_timeV.front(), end_timeV.back());

  // Redraw the axes. This way the graphs don't overshoot on top of
  // the axes any more.
  canvas->RedrawAxis();

  // Add the CMS logo in the top right corner. This has to be the last
  // action so the logo sits on top of the axes.
  drawLogo(canvas, logoName);

  canvas->Print(Form("int_lumi_cumulative_%s_%d%s.png",
                     partType.c_str(), year, fileSuffix.c_str()));

  delete h_delLumCum;
  delete h_recLumCum;
  delete canvas;
  delete legend;

  //------------------------------
  // Create the peak-lumi-per-day plot.
  //------------------------------

  // Scale the vertical axis to reduce the size of the labels a bit.
  hPeakLum.Scale(scaleFactor2);

  hPeakLum.SetTitle(titlePeak.c_str());
  hPeakLum.GetXaxis()->SetTimeDisplay(1);
  hPeakLum.GetXaxis()->SetTimeFormat("%d/%m");
  hPeakLum.GetXaxis()->SetTimeOffset(0, "gmt");
  hPeakLum.GetXaxis()->SetLabelOffset(0.01);
  hPeakLum.GetXaxis()->SetTitleFont(62);
  hPeakLum.GetYaxis()->SetTitleFont(62);
  hPeakLum.GetXaxis()->SetNdivisions(705);

  hPeakLum.SetLineColor(kLineColorPeak);
  hPeakLum.SetMarkerColor(kLineColorPeak);
  hPeakLum.SetFillColor(kFillColorPeak);

  double maxPeak = 0.;
  for (int i = 1; i != hPeakLum.GetNbinsX() + 1; ++i) {
    maxPeak = max(hPeakLum.GetBinContent(i), maxPeak);
  }
  airSpace = .2;
  min_y = 0.;
  tmp = (1. + airSpace) * maxPeak;
  max_y = tmp;
  canvas = createCanvas();
  hPeakLum.Draw();
  hPeakLum.GetYaxis()->SetRangeUser(min_y, max_y);
  duplicateYAxis(canvas, hPeakLum.GetYaxis());

  // Add a label specifying up until when data taken was included in
  // this plot.
  drawDateLabel(canvas, end_timeV.front(), end_timeV.back());

  legend = createLegend();
  legend->SetHeader(Form("Max. inst. lumi.: %.2f Hz/nb", maxPeak));
  legend->Draw();

  canvas->RedrawAxis();

  // Add the CMS logo in the top right corner. This has to be the last
  // action so the logo sits on top of the axes.
  drawLogo(canvas, logoName);

  canvas->Print(Form("peak_lumi_per_day_%s_%d%s.png", partType.c_str(), year, fileSuffix.c_str()));

  delete legend;
  delete canvas;

  //------------------------------
  // Create the integrated lumi for 2010, 2011, 2012 plot.
  //------------------------------

  if (beamType == "PROTPHYS") {

    // Read 2010.
    std::string fileNameIntLumi_2010 = replaceInString(fileNameIntLumi,
                                                       Form("%d", year),
                                                       "2010");
    std::vector<float> delivered_lumiV_2010;
    std::vector<float> recorded_lumiV_2010;
    readInputFileIntLumi(fileNameIntLumi_2010, runV, start_timeV, end_timeV,
                         delivered_lumiV_2010, recorded_lumiV_2010);
    // DEBUG DEBUG DEBUG
    assert(delivered_lumiV_2010.size() == recorded_lumiV_2010.size());
    // DEBUG DEBUG DEBUG end
    for (size_t ijk = 0; ijk != delivered_lumiV_2010.size(); ++ijk) {
      delivered_lumiV_2010.at(ijk) = delivered_lumiV_2010.at(ijk) / conversionFactor;
    }
    tmpLo = timestampFromString(start_timeV.front());
    tmpHi = timestampFromString(end_timeV.back());
    dateLo = zeroTimeInTimestamp(tmpLo);
    dateHi = zeroTimeInTimestamp(tmpHi);
    numDays = ((dateHi.GetSec() - dateLo.GetSec()) / (24 * 60 * 60)) + 1;
    int const nBins = numDays;
    float delivered_lumiA_2010[nBins];
    float recorded_lumiA_2010[nBins];
    for (int iTmp = 0; iTmp < numDays; ++iTmp) {
      delivered_lumiA_2010[iTmp] = 0.;
      recorded_lumiA_2010[iTmp] = 0.;
    }
    mapIntLumiToDays(runV, start_timeV, end_timeV,
                     delivered_lumiV_2010, recorded_lumiV_2010,
                     delivered_lumiA_2010, recorded_lumiA_2010);
    a = dateLo;
    b = dateHi;
    // NOTE: Watch out with this magic. It centers the bins on the days.
    b.SetSec(b.GetSec() + (24 * 60 * 60));
    a.SetSec(a.GetSec() - (12 * 60 * 60) - (60 * 60));
    b.SetSec(b.GetSec() - (12 * 60 * 60) - (60 * 60));
    TH1F h_delLum_2010("", "", numDays, a.GetSec(), b.GetSec());
    for (int i = 0; i != numDays; ++i) {
      h_delLum_2010.SetBinContent(i + 1, delivered_lumiA_2010[i]);
    }
    TH1F* h_delLumCum_2010 = dynamic_cast<TH1F*>(h_delLum_2010.Clone());
    double cumDel = 0.;
    for (int bin = 1; bin != h_delLum_2010.GetNbinsX() + 1; ++bin) {
      cumDel += h_delLum_2010.GetBinContent(bin);
      h_delLumCum_2010->SetBinContent(bin, cumDel);
    }

    // Read 2011.
    std::string fileNameIntLumi_2011 = replaceInString(fileNameIntLumi,
                                                       Form("%d", year),
                                                       "2011");
    std::vector<float> delivered_lumiV_2011;
    std::vector<float> recorded_lumiV_2011;
    readInputFileIntLumi(fileNameIntLumi_2011, runV, start_timeV, end_timeV,
                         delivered_lumiV_2011, recorded_lumiV_2011);
    // DEBUG DEBUG DEBUG
    assert(delivered_lumiV_2011.size() == recorded_lumiV_2011.size());
    // DEBUG DEBUG DEBUG end
    for (size_t ijk = 0; ijk != delivered_lumiV_2011.size(); ++ijk) {
      delivered_lumiV_2011.at(ijk) = delivered_lumiV_2011.at(ijk) / conversionFactor;
    }
    tmpLo = timestampFromString(start_timeV.front());
    tmpHi = timestampFromString(end_timeV.back());
    dateLo = zeroTimeInTimestamp(tmpLo);
    dateHi = zeroTimeInTimestamp(tmpHi);
    numDays = ((dateHi.GetSec() - dateLo.GetSec()) / (24 * 60 * 60)) + 1;
    int const nBins = numDays;
    float delivered_lumiA_2011[nBins];
    float recorded_lumiA_2011[nBins];
    for (int iTmp = 0; iTmp < numDays; ++iTmp) {
      delivered_lumiA_2011[iTmp] = 0.;
      recorded_lumiA_2011[iTmp] = 0.;
    }
    mapIntLumiToDays(runV, start_timeV, end_timeV,
                     delivered_lumiV_2011, recorded_lumiV_2011,
                     delivered_lumiA_2011, recorded_lumiA_2011);
    a = dateLo;
    b = dateHi;
    // Shift the whole data a year back.
    a.SetSec(a.GetSec() - 365 * 24 * 60 * 60);
    b.SetSec(b.GetSec() - 365 * 24 * 60 * 60);
    // NOTE: Watch out with this magic. It centers the bins on the days.
    b.SetSec(b.GetSec() + (24 * 60 * 60));
    a.SetSec(a.GetSec() - (12 * 60 * 60) - (60 * 60));
    b.SetSec(b.GetSec() - (12 * 60 * 60) - (60 * 60));
    TH1F h_delLum_2011("", "", numDays, a.GetSec(), b.GetSec());
    for (int i = 0; i != numDays; ++i) {
      h_delLum_2011.SetBinContent(i + 1, delivered_lumiA_2011[i]);
    }
    TH1F* h_delLumCum_2011 = dynamic_cast<TH1F*>(h_delLum_2011.Clone());
    double cumDel = 0.;
    for (int bin = 1; bin != h_delLum_2011.GetNbinsX() + 1; ++bin) {
      cumDel += h_delLum_2011.GetBinContent(bin);
      h_delLumCum_2011->SetBinContent(bin, cumDel);
    }

    // Read 2012.
    std::string fileNameIntLumi_2012 = replaceInString(fileNameIntLumi,
                                                       Form("%d", year),
                                                       "2012");
    std::vector<float> delivered_lumiV_2012;
    std::vector<float> recorded_lumiV_2012;
    readInputFileIntLumi(fileNameIntLumi_2012, runV, start_timeV, end_timeV,
                         delivered_lumiV_2012, recorded_lumiV_2012);
    // DEBUG DEBUG DEBUG
    assert(delivered_lumiV_2012.size() == recorded_lumiV_2012.size());
    // DEBUG DEBUG DEBUG end
    for (size_t ijk = 0; ijk != delivered_lumiV_2012.size(); ++ijk) {
      delivered_lumiV_2012.at(ijk) = delivered_lumiV_2012.at(ijk) / conversionFactor;
    }
    tmpLo = timestampFromString(start_timeV.front());
    tmpHi = timestampFromString(end_timeV.back());
    dateLo = zeroTimeInTimestamp(tmpLo);
    dateHi = zeroTimeInTimestamp(tmpHi);
    numDays = ((dateHi.GetSec() - dateLo.GetSec()) / (24 * 60 * 60)) + 1;
    int const nBins = numDays;
    float delivered_lumiA_2012[nBins];
    float recorded_lumiA_2012[nBins];
    for (int iTmp = 0; iTmp < numDays; ++iTmp) {
      delivered_lumiA_2012[iTmp] = 0.;
      recorded_lumiA_2012[iTmp] = 0.;
    }
    mapIntLumiToDays(runV, start_timeV, end_timeV,
                     delivered_lumiV_2012, recorded_lumiV_2012,
                     delivered_lumiA_2012, recorded_lumiA_2012);
    a = dateLo;
    b = dateHi;
    // Shift the whole data two years back.
    a.SetSec(a.GetSec() - (365 + 366) * 24 * 60 * 60);
    b.SetSec(b.GetSec() - (365 + 366) * 24 * 60 * 60);
    // NOTE: Watch out with this magic. It centers the bins on the days.
    b.SetSec(b.GetSec() + (24 * 60 * 60));
    a.SetSec(a.GetSec() - (12 * 60 * 60) - (60 * 60));
    b.SetSec(b.GetSec() - (12 * 60 * 60) - (60 * 60));
    TH1F h_delLum_2012("", "", numDays, a.GetSec(), b.GetSec());
    for (int i = 0; i != numDays; ++i) {
      h_delLum_2012.SetBinContent(i + 1, delivered_lumiA_2012[i]);
    }
    TH1F* h_delLumCum_2012 = dynamic_cast<TH1F*>(h_delLum_2012.Clone());
    double cumDel = 0.;
    for (int bin = 1; bin != h_delLum_2012.GetNbinsX() + 1; ++bin) {
      cumDel += h_delLum_2012.GetBinContent(bin);
      h_delLumCum_2012->SetBinContent(bin, cumDel);
    }

    // Tweak.
    h_delLumCum_2010->SetFillColor(kWhite);
    h_delLumCum_2011->SetFillColor(kWhite);
    h_delLumCum_2012->SetFillColor(kWhite);
    h_delLumCum_2010->SetLineColor(kGreen);
    h_delLumCum_2011->SetLineColor(kRed);
    h_delLumCum_2012->SetLineColor(kBlue);
    h_delLumCum_2010->SetLineWidth(5.);
    h_delLumCum_2011->SetLineWidth(5.);
    h_delLumCum_2012->SetLineWidth(5.);
    h_delLumCum_2010->Scale(scaleFactor);
    h_delLumCum_2011->Scale(scaleFactor);
    h_delLumCum_2012->Scale(scaleFactor);

    h_delLumCum_2011->SetTitle(titleCumulativeYears.c_str());
    h_delLumCum_2011->GetXaxis()->SetTimeDisplay(1);
    h_delLumCum_2011->GetXaxis()->SetTimeFormat("%d/%m");
    h_delLumCum_2011->GetXaxis()->SetTimeOffset(0, "gmt");
    h_delLumCum_2011->GetXaxis()->SetLabelOffset(0.01);
    h_delLumCum_2011->GetYaxis()->SetTitleOffset(1.2);
    h_delLumCum_2011->GetXaxis()->SetTitleFont(62);
    h_delLumCum_2011->GetYaxis()->SetTitleFont(62);
    h_delLumCum_2011->GetXaxis()->SetNdivisions(705);

    // And plot.
    TCanvas* canvas = createCanvas();
    // NOTE: The order here is a bit interesting but it sets the scale...
    h_delLumCum_2011->Draw("][");
    h_delLumCum_2010->Draw("SAME ][");
    h_delLumCum_2012->Draw("SAME ][");

    min_y = 0.;
    airSpace = .2;
    max_y = (1. + airSpace) * h_delLumCum_2012->GetMaximum();
    h_delLumCum_2011->GetYaxis()->SetRangeUser(min_y, max_y);

    legend = createLegend();
    legend->AddEntry(h_delLumCum_2010, "2010, #sqrt{s} = 7 TeV");
    legend->AddEntry(h_delLumCum_2011, "2011, #sqrt{s} = 7 TeV");
    legend->AddEntry(h_delLumCum_2012, "2012, #sqrt{s} = 8 TeV");
    legend->Draw();
    duplicateYAxis(canvas, h_delLumCum_2011->GetYaxis());
    canvas->RedrawAxis();
    drawLogo(canvas, logoName);
    canvas->Print(Form("int_lumi_cumulative_%s%s.png",
                       partType.c_str(), fileSuffix.c_str()));
    delete h_delLumCum_2010;
    delete h_delLumCum_2011;
    delete h_delLumCum_2012;
    delete legend;
    delete canvas;
  }
}

void create_public_lumi_plots(int const year=2012,
                              std::string const dataPath="/afs/cern.ch/cms/lumi/www/publicplots/",
                              std::string const beamType="PROTPHYS") {
  std::vector<std::string> colorSchemes;
  colorSchemes.push_back("Greg");
  colorSchemes.push_back("Joe");
  for (size_t i = 0; i < colorSchemes.size(); ++i) {
    create_plots(colorSchemes[i], year, dataPath, beamType);
  }
}
