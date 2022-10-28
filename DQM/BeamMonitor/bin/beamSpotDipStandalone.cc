#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <bitset>
#include <unistd.h>

//
#include "Dip.h"
#include "DipFactory.h"
#include "DipPublication.h"
#include "DipTimestamp.h"

using namespace std;

// constants
const char* qualities[3] = {"Uncertain", "Bad", "Good"};
const bool publishStatErrors = true;

const int secPerLS = 23;
const int rad2urad = 1000000;
const int cm2um = 10000;
const int cm2mm = 10;
const int intLS = 1;  // for CMS scaler

// variables
long lastFitTime = 0;
long lastModTime = 0;
std::bitset<8> alive;
int lsCount = 0;
int currentLS = 0;

// DIP objects
DipFactory* dip;
DipData* messageCMS;
DipData* messageLHC;
DipData* messagePV;
DipPublication* publicationCMS;
DipPublication* publicationLHC;
DipPublication* publicationPV;

// initial values of beamspot object
int runnum;
string startTime;
string endTime;
time_t startTimeStamp = 0;
time_t endTimeStamp = 0;
string lumiRange = "0 - 0";
string quality = "Uncertain";
int type = -1;
float x = 0;
float y = 0;
float z = 0;
float dxdz = 0;
float dydz = 0;
float err_x = 0;
float err_y = 0;
float err_z = 0;
float err_dxdz = 0;
float err_dydz = 0;
float width_x = 0;
float width_y = 0;
float sigma_z = 0;
float err_width_x = 0;
float err_width_y = 0;
float err_sigma_z = 0;

//
int events = 0;
float meanPV = 0;
float err_meanPV = 0;
float rmsPV = 0;
float err_rmsPV = 0;
int maxPV = 0;
int nPV = 0;

//
float Size[3];
float Centroid[3];
float Tilt[2];

//
bool verbose = false;
bool testing = false;

const string subjectCMS = "dip/CMS/Tracker/BeamSpot";
const string subjectLHC = "dip/CMS/LHC/LuminousRegion";
const string subjectPV = "dip/CMS/Tracker/PrimaryVertices";

string sourceFile = "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResultsForDIP.txt";
string sourceFile1 = "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResultsOld_TkStatus.txt";

const int timeoutLS[2] = {1, 2};

//

/*****************************************************************************/
string getDateTime(time_t t) {
  char mbstr[100];
  strftime(mbstr, sizeof(mbstr), "%Y.%m.%d %H:%M:%S %z", std::localtime(&t));

  return mbstr;
}

string getDateTime() {
  time_t t = time(nullptr);

  return getDateTime(t);
}

/*****************************************************************************/
class ErrHandler : public DipPublicationErrorHandler {
public:
  virtual ~ErrHandler() = default;

private:
  void handleException(DipPublication* publication, DipException& e) override {
    cerr << "exception (create): " << e.what() << endl;
  }
};

/*****************************************************************************/
long getFileSize(string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return (rc == 0 ? stat_buf.st_size : -1);
}

/*****************************************************************************/
time_t getLastTime(string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return (rc == 0 ? stat_buf.st_mtime : -1);
}

/*****************************************************************************/
vector<string> parse(string line, const string& delimiter) {
  vector<string> list;

  size_t pos = 0;
  while ((pos = line.find(delimiter)) != string::npos) {
    string token = line.substr(0, pos);

    list.push_back(token);

    line.erase(0, pos + delimiter.length());
  }

  list.push_back(line);  // remainder

  return list;
}

/*****************************************************************************/
void CMS2LHCRF_POS(float x, float y, float z) {
  if (x != 0) {  // Rotation + Translation + Inversion + Scaling
    double tmpx = x;
    // x*rotY[0]*rotZ[0] + y*rotY[0]*rotZ[1] - z*rotY[1] + trans[0];
    Centroid[0] = tmpx;
    Centroid[0] *= -1.0 * cm2um;
  } else
    Centroid[0] = x;

  if (y != 0) {  // Rotation + Translation + Scaling
    double tmpy = y;
    // x*(rotX[1]*rotY[1]*rotZ[0] - rotX[0]*rotZ[1]) +
    // y*(rotX[0]*rotZ[0] + rotX[1]*rotY[1]*rotZ[1]) +
    // z*rotX[1]*rotY[0] + trans[1];
    Centroid[1] = tmpy;
    Centroid[1] *= cm2um;
  } else
    Centroid[1] = y;

  if (z != 0) {  // Rotation + Translation + Inversion + Scaling
    double tmpz = z;
    // x*(rotX[0]*rotY[1]*rotZ[0] + rotX[1]*rotZ[1]) +
    // y*(rotX[0]*rotY[1]*rotZ[1] - rotX[1]*rotZ[0]) +
    // z*rotX[0]*rotY[0] + trans[2];
    Centroid[2] = tmpz;
    Centroid[2] *= -1.0 * cm2mm;
  } else
    Centroid[2] = z;
}

/*****************************************************************************/
void trueRcd() {
  try {
    // CMS to LHC RF
    CMS2LHCRF_POS(x, y, z);

    Tilt[0] = dxdz * rad2urad;
    Tilt[1] = (dydz != 0 ? (dydz * -1 * rad2urad) : 0);

    Size[0] = width_x * cm2um;
    Size[1] = width_y * cm2um;
    Size[2] = sigma_z * cm2mm;

    // CMS
    messageCMS->insert(runnum, "runnum");
    messageCMS->insert(startTime, "startTime");
    messageCMS->insert(endTime, "endTime");
    messageCMS->insert(startTimeStamp, "startTimeStamp");
    messageCMS->insert(endTimeStamp, "endTimeStamp");
    messageCMS->insert(lumiRange, "lumiRange");
    messageCMS->insert(quality, "quality");
    messageCMS->insert(type, "type");  // Unknown=-1, Fake=0, Tracker=2(Good)
    messageCMS->insert(x, "x");
    messageCMS->insert(y, "y");
    messageCMS->insert(z, "z");
    messageCMS->insert(dxdz, "dxdz");
    messageCMS->insert(dydz, "dydz");
    messageCMS->insert(width_x, "width_x");
    messageCMS->insert(width_y, "width_y");
    messageCMS->insert(sigma_z, "sigma_z");

    if (publishStatErrors) {
      messageCMS->insert(err_x, "err_x");
      messageCMS->insert(err_y, "err_y");
      messageCMS->insert(err_z, "err_z");
      messageCMS->insert(err_dxdz, "err_dxdz");
      messageCMS->insert(err_dydz, "err_dydz");
      messageCMS->insert(err_width_x, "err_width_x");
      messageCMS->insert(err_width_y, "err_width_y");
      messageCMS->insert(err_sigma_z, "err_sigma_z");
    }

    // LHC
    messageLHC->insert(Size, 3, "Size");
    messageLHC->insert(Centroid, 3, "Centroid");
    messageLHC->insert(Tilt, 2, "Tilt");

    // PV
    messagePV->insert(runnum, "runnum");
    messagePV->insert(startTime, "startTime");
    messagePV->insert(endTime, "endTime");
    messagePV->insert(startTimeStamp, "startTimeStamp");
    messagePV->insert(endTimeStamp, "endTimeStamp");
    messagePV->insert(lumiRange, "lumiRange");
    messagePV->insert(events, "events");
    messagePV->insert(meanPV, "meanPV");
    messagePV->insert(err_meanPV, "err_meanPV");
    messagePV->insert(rmsPV, "rmsPV");
    messagePV->insert(err_rmsPV, "err_rmsPV");
    messagePV->insert(maxPV, "maxPV");
    messagePV->insert(nPV, "nPV");
  } catch (exception& e) {
    cerr << "exception (trueRcd): " << e.what() << endl;
  }
}

/*****************************************************************************/
void fakeRcd() {
  try {
    Centroid[0] = 0;
    Centroid[1] = 0;
    Centroid[2] = 0;

    Size[0] = 0;
    Size[1] = 0;
    Size[2] = 0;

    Tilt[0] = 0;
    Tilt[1] = 0;

    messageLHC->insert(Size, 3, "Size");
    messageLHC->insert(Centroid, 3, "Centroid");
    messageLHC->insert(Tilt, 2, "Tilt");
  } catch (exception& e) {
    cerr << "exception (fakeRcd): " << e.what() << endl;
  }
}

/*****************************************************************************/
void publishRcd(string qlty, string err, bool pubCMS, bool fitTime) {
  try {
    bool updateCMS = pubCMS && (currentLS % intLS == 0);

    if (verbose) {
      cerr << "sending (" << qlty << " | " << err << ")";

      if (alive.test(7)) {
        if (updateCMS)
          cerr << " to CCC and CMS";
        else if (!alive.test(1) && !alive.test(2))
          cerr << " to CCC only";
      }

      cerr << endl;
    }

    DipTimestamp zeit;
    if (fitTime) {
      long epoch;
      epoch = endTimeStamp * 1000;  // convert to ms
      zeit = DipTimestamp(epoch);
    } else
      zeit = DipTimestamp();

    // send
    if (updateCMS)
      publicationCMS->send(*messageCMS, zeit);

    publicationLHC->send(*messageLHC, zeit);
    publicationPV->send(*messagePV, zeit);

    // set qualities

    if (qlty == qualities[0]) {  // Uncertain
      if (updateCMS)
        publicationCMS->setQualityUncertain(err.c_str());

      publicationLHC->setQualityUncertain(err.c_str());
    } else if (qlty == qualities[1]) {  // Bad
      if (updateCMS)
        publicationCMS->setQualityBad(err.c_str());

      publicationLHC->setQualityBad(err.c_str());
    } else if (qlty == "UNINITIALIZED") {
      if (updateCMS)
        publicationCMS->setQualityBad("UNINITIALIZED");

      publicationLHC->setQualityBad("UNINITIALIZED");
    }
  } catch (exception& e) {
    cerr << "exception (publishRcd): " << e.what() << endl;
  }
}

/*****************************************************************************/
bool readRcd(ifstream& file) {
  int nthLnInRcd = 0;
  bool rcdQlty = false;

  try {
    string record;
    while (getline(file, record)) {
      nthLnInRcd++;

      vector<string> tmp = parse(record, " ");

      switch (nthLnInRcd) {
        case 1:
          if (record.rfind("Run", 0) != 0) {
            cerr << "Reading of results text file interrupted. " + getDateTime() << endl;
            return false;
          }
          runnum = stoi(tmp[1]);
          break;
        case 2:
          startTime = tmp[1] + " " + tmp[2] + " " + tmp[3];
          startTimeStamp = stol(tmp[4]);
          break;
        case 3:
          endTime = tmp[1] + " " + tmp[2] + " " + tmp[3];
          endTimeStamp = stol(tmp[4]);
          break;
        case 4:
          lumiRange = record.substr(10);
          if (verbose)
            cerr << "lumisection range: " + lumiRange << endl;
          currentLS = stoi(tmp[3]);
          break;
        case 5:
          type = stoi(tmp[1]);
          if (testing)
            quality = qualities[0];  // Uncertain
          else if (type >= 2)
            quality = qualities[2];  // Good
          else
            quality = qualities[1];  // Bad
          break;

        case 6:
          x = stof(tmp[1]);
          break;
        case 7:
          y = stof(tmp[1]);
          break;
        case 8:
          z = stof(tmp[1]);
          break;

        case 9:
          sigma_z = stof(tmp[1]);
          break;
        case 10:
          dxdz = stof(tmp[1]);
          break;
        case 11:
          dydz = stof(tmp[1]);
          break;
        case 12:
          width_x = stof(tmp[1]);
          break;
        case 13:
          width_y = stof(tmp[1]);
          break;

        case 14:
          err_x = sqrt(stof(tmp[1]));
          break;
        case 15:
          err_y = sqrt(stof(tmp[2]));
          break;
        case 16:
          err_z = sqrt(stof(tmp[3]));
          break;
        case 17:
          err_sigma_z = sqrt(stof(tmp[4]));
          break;
        case 18:
          err_dxdz = sqrt(stof(tmp[5]));
          break;
        case 19:
          err_dydz = sqrt(stof(tmp[6]));
          break;
        case 20:
          err_width_x = sqrt(stof(tmp[7]));
          err_width_y = err_width_x;
          break;
        case 21:
          break;
        case 22:
          break;
        case 23:
          break;
        case 24:
          events = stoi(tmp[1]);
          break;

        case 25:
          meanPV = stof(tmp[1]);
          break;
        case 26:
          err_meanPV = stof(tmp[1]);
          break;
        case 27:
          rmsPV = stof(tmp[1]);
          break;
        case 28:
          err_rmsPV = stof(tmp[1]);
          break;
        case 29:
          maxPV = stoi(tmp[1]);
          break;
        case 30:
          nPV = stoi(tmp[1]);
          rcdQlty = true;
          break;

        default:
          break;
      }
    }

    file.close();
  } catch (exception& e) {
    cerr << "io exception (readRcd): " << e.what() << endl;
  }

  return rcdQlty;
}

/*****************************************************************************/
string tkStatus() {
  string outstr;

  ifstream logfile(sourceFile1);

  if (!logfile.good() || getFileSize(sourceFile1) == 0) {
    // file does not exist or has zero size
    outstr = "No CMS Tracker status available. No DAQ/DQM.";
  } else {
    int nthLnInRcd = 0;
    string record;

    try {
      string record;

      while (getline(logfile, record)) {
        nthLnInRcd++;
        vector<string> tmp = parse(record, " ");

        switch (nthLnInRcd) {
          case 7:
            if (tmp[1].find("Yes") == string::npos)
              outstr = "CMS Tracker OFF.";
            else
              outstr = "CMS not taking data or no beam.";
            break;
          case 8:
            runnum = stoi(tmp[1]);
            break;
          default:
            break;
        }
      }
    } catch (exception& e) {
      cerr << "exception (tkStatus): " << e.what() << endl;
    }
  }

  logfile.close();

  return outstr;
}

/*****************************************************************************/
void problem() {
  if (verbose)
    cerr << "no update | alive = " << alive << endl;

  lsCount++;

  if ((lsCount % (timeoutLS[0] * secPerLS) == 0) && (lsCount % (timeoutLS[1] * secPerLS) != 0))  // first timeout
  {
    if (!alive.test(1))
      alive.flip(1);
    if (!alive.test(2)) {
      if (!alive.test(7))
        fakeRcd();
      else
        trueRcd();

      stringstream warnMsg;
      warnMsg << "No new data for " << lsCount << " LS";
      publishRcd("Uncertain", warnMsg.str(), false, false);
    } else {
      fakeRcd();

      stringstream warnMsg;
      warnMsg << "No new data for " << lsCount << " LS: " << tkStatus();
      publishRcd("Bad", warnMsg.str(), false, false);
    }
  } else if (lsCount % (timeoutLS[1] * secPerLS) == 0) {  // second timeout
    if (!alive.test(2))
      alive.flip(2);
    fakeRcd();

    stringstream warnMsg;
    warnMsg << "No new data for " << lsCount << " LS: " << tkStatus();
    publishRcd("Bad", warnMsg.str(), false, false);
  }
}

/*****************************************************************************/
void polling() {
  try {
    ifstream logFile(sourceFile);

    if (!logFile.good()) {
      cerr << "Source File: " + sourceFile + " doesn't exist!" << endl;
      problem();
    } else {
      lastModTime = getLastTime(sourceFile);

      if (lastFitTime == 0)
        lastFitTime = lastModTime;

      if (getFileSize(sourceFile) == 0) {
        // source file has zero length
        if (lastModTime > lastFitTime) {
          string tmp = tkStatus();
          cerr << "New run starts. Run number: " << runnum << endl;
          if (verbose)
            cerr << "Initial lastModTime = " + getDateTime(lastModTime) << endl;
        }
        lastFitTime = lastModTime;
      }

      if (lastModTime > lastFitTime) {
        // source file modified
        if (verbose) {
          cerr << "time of last fit    = " + getDateTime(lastFitTime) << endl;
          cerr << "time of current fit = " + getDateTime(lastModTime) << endl;
        }
        lastFitTime = lastModTime;

        // source file length > 0
        if (getFileSize(sourceFile) > 0) {
          if (verbose)
            cerr << "reading record from " + sourceFile << endl;

          if (readRcd(logFile)) {
            if (verbose)
              cerr << "got new record from file" << endl;

            trueRcd();
            alive.reset();
            alive.flip(7);
          } else {
            if (verbose)
              cerr << "problem with new record" << endl;
            fakeRcd();
          }

          lsCount = 0;
        }
      } else {
        // source file not touched
        problem();
      }
    }

    logFile.close();

  } catch (exception& e) {
    cerr << "io exception (end of lumi): " << e.what() << endl;
  };
}

/*****************************************************************************/
void beginServer() {
  try {
    ErrHandler errHandler;

    cerr << "server started at " + getDateTime() << endl;

    cerr << "creating publication " + subjectCMS << endl;
    publicationCMS = dip->createDipPublication(subjectCMS.c_str(), &errHandler);
    messageCMS = dip->createDipData();

    cerr << "creating publication " + subjectLHC << endl;
    publicationLHC = dip->createDipPublication(subjectLHC.c_str(), &errHandler);
    messageLHC = dip->createDipData();

    cerr << "creating publication " + subjectPV << endl;
    publicationPV = dip->createDipPublication(subjectPV.c_str(), &errHandler);
    messagePV = dip->createDipData();

    trueRcd();  // starts with all 0
    publishRcd("UNINITIALIZED", "", true, false);
  } catch (exception& e) {
    cerr << "exception (start up): " << e.what() << endl;
  }

  quality = qualities[0];  // start with Uncertain
}

/*****************************************************************************/
void endServer() {
  // destroy publications and data
  cerr << "destroying publication " + subjectCMS << endl;
  dip->destroyDipPublication(publicationCMS);
  delete messageCMS;

  cerr << "destroying publication " + subjectLHC << endl;
  dip->destroyDipPublication(publicationLHC);
  delete messageLHC;

  cerr << "destroying publication " + subjectPV << endl;
  dip->destroyDipPublication(publicationPV);
  delete messagePV;
}

/*****************************************************************************/
int main(int narg, char* args[]) {
  // options
  verbose = strcmp(args[1], "true");
  testing = strcmp(args[2], "true");

  sourceFile = args[3];
  sourceFile1 = args[4];

  //
  startTime = getDateTime();
  endTime = getDateTime();

  dip = Dip::create("CmsBeamSpotServer");
  // Use both CMS-based DIM DNS server (https://its.cern.ch/jira/browse/CMSOMS-280)
  dip->setDNSNode("cmsdimns1.cern.ch,cmsdimns2.cern.ch");

  cerr << "reading from file (NFS)" << endl;

  //
  beginServer();

  cerr << "entering polling loop" << endl;

  while (true) {
    polling();
    sleep(1);
  }

  cerr << "[done]" << endl;

  //
  endServer();

  return 0;
}
