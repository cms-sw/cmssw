#include "DQM/BeamMonitor/plugins/BeamSpotDipServer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <fstream>
#include <vector>
#include <ctime>
#include <sys/stat.h>

#include "Dip.h"
#include "DipFactory.h"
#include "DipPublication.h"
#include "DipTimestamp.h"

using namespace std;

/*****************************************************************************/
class ErrHandler : public DipPublicationErrorHandler {
public:
  virtual ~ErrHandler() = default;

private:
  void handleException(DipPublication* publication, DipException& e) override {
    edm::LogError("BeamSpotDipServer") << "exception (create): " << e.what();
  }
};

/*****************************************************************************/
BeamSpotDipServer::BeamSpotDipServer(const edm::ParameterSet& ps) {
  //
  verbose = ps.getUntrackedParameter<bool>("verbose");
  testing = ps.getUntrackedParameter<bool>("testing");

  subjectCMS = ps.getUntrackedParameter<string>("subjectCMS");
  subjectLHC = ps.getUntrackedParameter<string>("subjectLHC");
  subjectPV = ps.getUntrackedParameter<string>("subjectPV");

  readFromNFS = ps.getUntrackedParameter<bool>("readFromNFS");
  // only if readFromNFS = true
  sourceFile = ps.getUntrackedParameter<string>("sourceFile");    // beamspot
  sourceFile1 = ps.getUntrackedParameter<string>("sourceFile1");  // tk status

  timeoutLS = ps.getUntrackedParameter<vector<int>>("timeoutLS");

  //
  bsLegacyToken_ = esConsumes<edm::Transition::EndLuminosityBlock>();

  dcsRecordInputTag_ = ps.getUntrackedParameter<edm::InputTag>("dcsRecordInputTag");
  dcsRecordToken_ = consumes<DCSRecord>(dcsRecordInputTag_);

  //
  dip = Dip::create("CmsBeamSpotServer");

  // Use both CMS-based DIM DNS server (https://its.cern.ch/jira/browse/CMSOMS-280)
  dip->setDNSNode("cmsdimns1.cern.ch,cmsdimns2.cern.ch");

  edm::LogInfo("BeamSpotDipServer") << "reading from " << (readFromNFS ? "file (NFS)" : "database");
}

/*****************************************************************************/
void BeamSpotDipServer::bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) {
  // do nothing
}

/*****************************************************************************/
void BeamSpotDipServer::dqmBeginRun(const edm::Run& r, const edm::EventSetup&) {
  edm::LogInfo("BeamSpotDipServer") << "begin run " << r.run();

  try {
    ErrHandler errHandler;

    edm::LogInfo("BeamSpotDipServer") << "server started at " + getDateTime();

    edm::LogInfo("BeamSpotDipServer") << "creating publication " + subjectCMS;
    publicationCMS = dip->createDipPublication(subjectCMS.c_str(), &errHandler);
    messageCMS = dip->createDipData();

    edm::LogInfo("BeamSpotDipServer") << "creating publication " + subjectLHC;
    publicationLHC = dip->createDipPublication(subjectLHC.c_str(), &errHandler);
    messageLHC = dip->createDipData();

    edm::LogInfo("BeamSpotDipServer") << "creating publication " + subjectPV;
    publicationPV = dip->createDipPublication(subjectPV.c_str(), &errHandler);
    messagePV = dip->createDipData();

    trueRcd();  // starts with all 0
    publishRcd("UNINITIALIZED", "", true, false);
  } catch (exception& e) {
    edm::LogError("BeamSpotDipServer") << "exception (start up): " << e.what();
  }

  quality = qualities[0];  // start with Uncertain
}

/*****************************************************************************/
void BeamSpotDipServer::dqmBeginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {
  // do nothing
}

/*****************************************************************************/
void BeamSpotDipServer::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  if (!readFromNFS) {
    // get runnumber
    runnum = iEvent.run();

    // get tracker status if in a new lumisection
    int nthlumi = iEvent.luminosityBlock();

    if (nthlumi > lastlumi) {  // check every LS
      lastlumi = nthlumi;

      edm::Handle<DCSRecord> dcsRecord;
      iEvent.getByToken(dcsRecordToken_, dcsRecord);

      wholeTrackerOn =
          (*dcsRecord).highVoltageReady(DCSRecord::BPIX) && (*dcsRecord).highVoltageReady(DCSRecord::FPIX) &&
          (*dcsRecord).highVoltageReady(DCSRecord::TIBTID) && (*dcsRecord).highVoltageReady(DCSRecord::TOB) &&
          (*dcsRecord).highVoltageReady(DCSRecord::TECp) && (*dcsRecord).highVoltageReady(DCSRecord::TECm);

      if (verbose)
        edm::LogInfo("BeamSpotDipServer") << "whole tracker on? " << (wholeTrackerOn ? "yes" : "no");
    }
  }
}

/*****************************************************************************/
void BeamSpotDipServer::dqmEndLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {
  edm::LogInfo("BeamSpotDipServer") << "--------------------- end of LS " << lumiSeg.luminosityBlock();

  try {
    if (readFromNFS) {
      ifstream logFile(sourceFile);

      if (!logFile.good()) {
        edm::LogWarning("BeamSpotDipServer") << "Source File: " + sourceFile + " doesn't exist!";
        problem();
      } else {
        lastModTime = getLastTime(sourceFile);

        if (lastFitTime == 0)
          lastFitTime = lastModTime;

        if (getFileSize(sourceFile) == 0) {
          // source file has zero length
          if (lastModTime > lastFitTime) {
            string tmp = tkStatus();
            edm::LogInfo("BeamSpotDipServer") << "New run starts. Run number: " << runnum;
            if (verbose)
              edm::LogInfo("BeamSpotDipServer") << "Initial lastModTime = " + getDateTime(lastModTime);
          }
          lastFitTime = lastModTime;
        }

        if (lastModTime > lastFitTime) {
          // source file modified
          if (verbose) {
            edm::LogInfo("BeamSpotDipServer") << "time of last fit    = " + getDateTime(lastFitTime);
            edm::LogInfo("BeamSpotDipServer") << "time of current fit = " + getDateTime(lastModTime);
          }
          lastFitTime = lastModTime;

          // source file length > 0
          if (getFileSize(sourceFile) > 0) {
            if (verbose)
              edm::LogInfo("BeamSpotDipServer") << "reading record from " + sourceFile;

            if (readRcd(logFile)) {
              if (verbose)
                edm::LogInfo("BeamSpotDipServer") << "got new record from file";

              trueRcd();
              alive.reset();
              alive.flip(7);
            } else {
              if (verbose)
                edm::LogInfo("BeamSpotDipServer") << "problem with new record";
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
    } else {
      edm::ESHandle<BeamSpotOnlineObjects> bsLegacyHandle = iSetup.getHandle(bsLegacyToken_);
      auto const& bs = *bsLegacyHandle;

      // from database
      if (readRcd(bs)) {
        if (verbose)
          edm::LogInfo("BeamSpotDipServer") << "got new record from database";
        trueRcd();
        alive.reset();
        alive.flip(7);
      } else {
        if (verbose)
          edm::LogInfo("BeamSpotDipServer") << "problem with new record";
        fakeRcd();
      }

      lsCount = 0;
    }

    // quality of the publish results
    if (testing)
      publishRcd(qualities[0], "Testing", true, true);  // Uncertain
    else if (quality == qualities[1])                   // Bad
      publishRcd(quality, "No fit or fit fails", true, true);
    else
      publishRcd(quality, "", true, true);  // Good
  } catch (exception& e) {
    edm::LogWarning("BeamSpotDipServer") << "io exception (end of lumi): " << e.what();
  };
}

/*****************************************************************************/
void BeamSpotDipServer::dqmEndRun(const edm::Run&, const edm::EventSetup&) {
  // destroy publications and data
  edm::LogInfo("BeamSpotDipServer") << "destroying publication " + subjectCMS;
  dip->destroyDipPublication(publicationCMS);
  delete messageCMS;

  edm::LogInfo("BeamSpotDipServer") << "destroying publication " + subjectLHC;
  dip->destroyDipPublication(publicationLHC);
  delete messageLHC;

  edm::LogInfo("BeamSpotDipServer") << "destroying publication " + subjectPV;
  dip->destroyDipPublication(publicationPV);
  delete messagePV;
}

/*****************************************************************************/
long BeamSpotDipServer::getFileSize(string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return (rc == 0 ? stat_buf.st_size : -1);
}

/*****************************************************************************/
time_t BeamSpotDipServer::getLastTime(string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return (rc == 0 ? stat_buf.st_mtime : -1);
}

/*****************************************************************************/
vector<string> BeamSpotDipServer::parse(string line, const string& delimiter) {
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
string BeamSpotDipServer::tkStatus() {
  string outstr;

  if (readFromNFS) {  // get from file on /nfs
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
        edm::LogWarning("BeamSpotDipServer") << "exception (tkStatus): " << e.what();
      }
    }

    logfile.close();
  } else {
    // get from DCS
    if (wholeTrackerOn)
      outstr = "CMS not taking data or no beam.";
    else
      outstr = "CMS Tracker OFF.";
  }

  return outstr;
}

/*****************************************************************************/
void BeamSpotDipServer::problem() {
  if (verbose)
    edm::LogInfo("BeamSpotDipServer") << "no update | alive = " << alive;

  lsCount++;

  if ((lsCount % timeoutLS[0] == 0) && (lsCount % timeoutLS[1] != 0))  // first time out
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
  } else if (lsCount % timeoutLS[1] == 0)  // second time out
  {
    if (!alive.test(2))
      alive.flip(2);
    fakeRcd();

    stringstream warnMsg;
    warnMsg << "No new data for " << lsCount << " LS: " << tkStatus();
    publishRcd("Bad", warnMsg.str(), false, false);
  }
}

/*****************************************************************************/
bool BeamSpotDipServer::readRcd(const BeamSpotOnlineObjects& bs)
// read from database
{
  runnum = bs.lastAnalyzedRun();

  // get from BeamSpotOnlineObject

  try {
    startTime = bs.startTime();
    startTimeStamp = bs.startTimeStamp();
    endTime = bs.endTime();
    endTimeStamp = bs.endTimeStamp();
  } catch (exception& e) {
    edm::LogWarning("BeamSpotDipServer") << "time variables are not available (readRcd): " << e.what();

    startTime = bs.creationTime();
    startTimeStamp = bs.creationTime();
    endTime = bs.creationTime();
    endTimeStamp = bs.creationTime();
  }

  try {
    lumiRange = bs.lumiRange();
  } catch (exception& e) {
    edm::LogWarning("BeamSpotDipServer") << "lumirange variable not avaialble (readRcd): " << e.what();

    lumiRange = to_string(bs.lastAnalyzedLumi());
  }

  currentLS = bs.lastAnalyzedLumi();

  type = bs.beamType();

  if (verbose)
    edm::LogInfo("BeamSpotDipServer") << "run: " << runnum << ", LS: " << currentLS << ", time: " << startTime << " "
                                      << startTimeStamp << ", type: " << type;

  if (testing)
    quality = qualities[0];  // Uncertain
  else if (type >= 2)
    quality = qualities[2];  // Good
  else
    quality = qualities[1];  // Bad

  x = bs.x();
  y = bs.y();
  z = bs.z();

  sigma_z = bs.sigmaZ();
  dxdz = bs.dxdz();
  dydz = bs.dydz();
  width_x = bs.beamWidthX();
  width_y = bs.beamWidthX();

  err_x = bs.xError();
  err_y = bs.yError();
  err_z = bs.zError();
  err_sigma_z = bs.sigmaZError();
  err_dxdz = bs.dxdzError();
  err_dydz = bs.dydzError();
  err_width_x = bs.beamWidthXError();
  err_width_y = bs.beamWidthYError();

  try {
    events = bs.usedEvents();
    meanPV = bs.meanPV();
    err_meanPV = bs.meanErrorPV();
    rmsPV = bs.rmsPV();
    err_rmsPV = bs.rmsErrorPV();
    maxPV = bs.maxPVs();
  } catch (exception& e) {
    edm::LogWarning("BeamSpotDipServer") << "PV variables are not available (readRcd): " << e.what();

    events = 0.;
    meanPV = 0.;
    err_meanPV = 0.;
    rmsPV = 0.;
    err_rmsPV = 0.;
    maxPV = 0.;
  }

  nPV = bs.numPVs();

  if (verbose)
    edm::LogInfo("BeamSpotDipServer") << "pos: (" << x << "," << y << "," << z << ")"
                                      << " nPV: " << nPV;

  return true;
}

/*****************************************************************************/
bool BeamSpotDipServer::readRcd(ifstream& file)  // readFromNFS
{
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
            edm::LogError("BeamSpotDipServer") << "Reading of results text file interrupted. " + getDateTime();
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
            edm::LogInfo("BeamSpotDipServer") << "lumisection range: " + lumiRange;
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
    edm::LogWarning("BeamSpotDipServer") << "io exception (readRcd): " << e.what();
  }

  return rcdQlty;
}

/*****************************************************************************/
void BeamSpotDipServer::CMS2LHCRF_POS(float x, float y, float z) {
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
void BeamSpotDipServer::trueRcd() {
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
    edm::LogWarning("BeamSpotDipServer") << "exception (trueRcd): " << e.what();
  }
}

/*****************************************************************************/
void BeamSpotDipServer::fakeRcd() {
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
    edm::LogWarning("BeamSpotDipServer") << "exception (fakeRcd): " << e.what();
  }
}

/*****************************************************************************/
void BeamSpotDipServer::publishRcd(string qlty, string err, bool pubCMS, bool fitTime) {
  try {
    bool updateCMS = pubCMS && (currentLS % intLS == 0);

    if (verbose) {
      edm::LogInfo("BeamSpotDipServer") << "sending (" << qlty << " | " << err << ")";

      if (alive.test(7)) {
        if (updateCMS)
          edm::LogInfo("BeamSpotDipServer") << " to CCC and CMS";
        else if (!alive.test(1) && !alive.test(2))
          edm::LogInfo("BeamSpotDipServer") << " to CCC only";
      }
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
    edm::LogWarning("BeamSpotDipServer") << "exception (publishRcd): " << e.what();
  }
}

/*****************************************************************************/
string BeamSpotDipServer::getDateTime(time_t t) {
  char mbstr[100];
  strftime(mbstr, sizeof(mbstr), "%Y.%m.%d %H:%M:%S %z", std::localtime(&t));

  return mbstr;
}

//
string BeamSpotDipServer::getDateTime() {
  time_t t = time(nullptr);

  return getDateTime(t);
}

DEFINE_FWK_MODULE(BeamSpotDipServer);
