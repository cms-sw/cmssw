//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 * $Id: EcalLaserCondTools.cc,v 1.2 2010/06/14 10:45:17 pgras Exp $
 *
 * author: Ph Gras. June, 2010
 */

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalLaserCondTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "hdf5.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <memory>
#include <cmath>

EcalLaserCondTools::EcalLaserCondTools(const edm::ParameterSet& ps)
    : fout_(nullptr),
      eventList_(nullptr),
      eventListFileName_(ps.getParameter<std::string>("eventListFile")),
      verb_(ps.getParameter<int>("verbosity")),
      mode_(ps.getParameter<std::string>("mode")),
      fnames_(ps.getParameter<std::vector<std::string> >("inputFiles")),
      skipIov_(ps.getParameter<int>("skipIov")),
      nIovs_(ps.getParameter<int>("nIovs")),
      fromTime_(ps.getParameter<int>("fromTime")),
      toTime_(ps.getParameter<int>("toTime")),
      minP_(ps.getParameter<double>("transparencyMin")),
      maxP_(ps.getParameter<double>("transparencyMax")) {
  if (mode_ == "db_to_ascii_file") {
    laserAPDPNRatiosToken_ = esConsumes();
  }

  ferr_ = fopen("corr_errors.txt", "w");
  fprintf(ferr_, "#t1\tdetid\tp1\tp2\tp3");

  if (!eventListFileName_.empty()) {
    eventList_ = fopen(eventListFileName_.c_str(), "r");
    if (eventList_ == nullptr)
      throw cms::Exception("User") << "Failed to open file " << eventListFileName_ << "\n";
  }
}

EcalLaserCondTools::~EcalLaserCondTools() {
  if (ferr_)
    fclose(ferr_);
  if (fout_)
    fclose(fout_);
}

void EcalLaserCondTools::analyze(const edm::Event& event, const edm::EventSetup& es) {
  if (mode_ == "ascii_file_to_db") {
    if (verb_ > 2)
      edm::LogPrint("EcalLaserCondTools") << "ascii_file_to_db mode\n";

    if (!db_.isAvailable()) {
      throw cms::Exception("CondDBAccess") << "Failed to connect to PoolDBOutputService\n";
    }
    FileReader corrReader(fnames_);
    corrReader.setVerbosity(verb_);
    fillDb(corrReader);
  } else if (mode_ == "hdf_file_to_db") {
    from_hdf_to_db();
  } else if (mode_ == "db_to_ascii_file") {
    dbToAscii(es);
  } else {
    cms::Exception("InvalidParam") << "Value of parameter mode is not valid. Expecting ascii_file_to_db or read";
  }
}

void EcalLaserCondTools::from_hdf_to_db() {
  cond::Time_t iovStart = 0;

  hid_t file, space, memspace;
  hid_t dset_rawid, dset_t2, dset;

  hsize_t dims[2] = {};

  for (unsigned int ifile = 0; ifile < fnames_.size(); ++ifile) {
    if (verb_) {
      edm::LogPrint("EcalLaserCondTools") << " - converting file: " << fnames_[ifile] << "\n";
    }

    file = H5Fopen(fnames_[ifile].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dset_rawid = H5Dopen(file, "cmssw_id", H5P_DEFAULT);
    space = H5Dget_space(dset_rawid);
    H5Sget_simple_extent_dims(space, dims, nullptr);

    unsigned int nCrystals = dims[0];
    int rawid[nCrystals];
    herr_t status;

    status = H5Dread(dset_rawid, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawid);
    if (status < 0)
      throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

    H5Dclose(dset_rawid);
    H5Sclose(space);

    dset_t2 = H5Dopen(file, "t2", H5P_DEFAULT);
    space = H5Dget_space(dset_t2);
    H5Sget_simple_extent_dims(space, dims, nullptr);

    unsigned int nIovs = dims[0];
    unsigned int nLME = dims[1];

    if (verb_) {
      edm::LogPrint("EcalLaserCondTools") << "Number of crystals: " << nCrystals << "\n";
      edm::LogPrint("EcalLaserCondTools") << "Number of IOVs: " << nIovs << "\n";
      edm::LogPrint("EcalLaserCondTools") << "Number of Monitoring regions: " << nLME << "\n";
    }

    int t1[nIovs], t3[nIovs], t2[nIovs][nLME];

    // -- reading data (cmsswid, t2, t1, t3, p2, p1, p3
    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << " * reading t2 table "
                                          << "\n";
    status = H5Dread(dset_t2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, t2[0]);
    if (status < 0)
      throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

    H5Dclose(dset_t2);
    //H5Sclose(space);

    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << " * reading t1 table "
                                          << "\n";
    dset = H5Dopen(file, "t1", H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, t1);
    if (status < 0)
      throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

    H5Dclose(dset);

    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << " * reading t3 table "
                                          << "\n";
    dset = H5Dopen(file, "t3", H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, t3);
    if (status < 0)
      throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

    H5Dclose(dset);

    assert(EcalLaserCondTools::nLmes == nLME);

    // read crystal info IOV by IOV (otherwise too large)
    float p1[nCrystals], p2[nCrystals], p3[nCrystals];
    hsize_t iov_dim[1] = {nCrystals};
    memspace = H5Screate_simple(1, iov_dim, nullptr);

    EcalLaserAPDPNRatios corrSet;
    for (unsigned int iIov = skipIov_; iIov < nIovs && iIov < unsigned(nIovs_); ++iIov) {
      EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
      iovStart = uint64_t(t1[iIov]) << 32;
      for (size_t iLme = 0; iLme < EcalLaserCondTools::nLmes; ++iLme) {
        t.t1 = edm::Timestamp(uint64_t(t1[iIov]) << 32);
        t.t2 = edm::Timestamp(uint64_t(t2[iIov][iLme]) << 32);
        t.t3 = edm::Timestamp(uint64_t(t3[iIov]) << 32);
        corrSet.setTime(iLme, t);
      }

      hsize_t offset[2] = {iIov, 0};      // shift rows: iIov, columns: 0
      hsize_t count[2] = {1, nCrystals};  // 1 row, nXtal columns

      dset = H5Dopen(file, "p1", H5P_DEFAULT);
      space = H5Dget_space(dset);
      status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, nullptr, count, nullptr);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

      status = H5Dread(dset, H5T_NATIVE_FLOAT, memspace, space, H5P_DEFAULT, p1);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

      H5Dclose(dset);
      //H5Sclose(space);

      dset = H5Dopen(file, "p2", H5P_DEFAULT);
      space = H5Dget_space(dset);
      status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, nullptr, count, nullptr);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

      status = H5Dread(dset, H5T_NATIVE_FLOAT, memspace, space, H5P_DEFAULT, p2);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

      H5Dclose(dset);
      //      H5Sclose(space);

      dset = H5Dopen(file, "p3", H5P_DEFAULT);
      space = H5Dget_space(dset);
      status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, nullptr, count, nullptr);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";

      status = H5Dread(dset, H5T_NATIVE_FLOAT, memspace, space, H5P_DEFAULT, p3);
      if (status < 0)
        throw cms::Exception("EcalLaserCondTool:HDF") << "Error while reading HD file.";
      H5Dclose(dset);
      H5Sclose(space);

      for (size_t iXtal = 0; iXtal < nCrystals; ++iXtal) {
        DetId detid = rawid[iXtal];

        EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = EcalLaserAPDPNRatios::EcalLaserAPDPNpair();
        corr.p1 = p1[iXtal];
        corr.p2 = p2[iXtal];
        corr.p3 = p3[iXtal];

        if (!std::isfinite(corr.p1) || !std::isfinite(corr.p2) || !std::isfinite(corr.p3) || corr.p1 < minP_ ||
            corr.p1 > maxP_ || corr.p2 < minP_ || corr.p2 > maxP_ || corr.p3 < minP_ || corr.p3 > maxP_) {
          fprintf(ferr_, "%d %d %f %f %f\n", t1[iIov], (int)detid, corr.p1, corr.p2, corr.p3);
          corr.p1 = corr.p2 = corr.p3 = 1;
        }
        corrSet.setValue((int)detid, corr);
      }

      try {
        //Write correction set in DB (one IOV):
        //if (db_->isNewTagRequest("EcalLaserAPDPNRatiosRcd")) {
        //  if (verb_)
        //      edm::LogPrint("EcalLaserCondTools") << "First IOV, extending starting time.\n";
        //  iovStart = db_->beginOfTime();
        //}
        timeval t;
        gettimeofday(&t, nullptr);
        if (verb_ > 1)
          edm::LogPrint("EcalLaserCondTools")
              << "[" << timeToString(t.tv_sec) << "] "
              << "Write IOV " << iIov << " starting from " << timeToString(iovStart >> 32) << "... ";
        db_->writeOneIOV(corrSet, iovStart, "EcalLaserAPDPNRatiosRcd");
      } catch (const cms::Exception& e) {
        if (verb_ > 1)
          edm::LogPrint("EcalLaserCondTools") << "Failed. ";
        edm::LogPrint("EcalLaserCondTools") << "Exception catched while writting to cond DB" << e.what() << "\n";
      }
      if (verb_ > 1)
        edm::LogPrint("EcalLaserCondTools") << "Suceeded.\n";

    }  // loop over IOVs

    H5Sclose(memspace);
    H5Fclose(file);
  }  // loop over input files
}

void EcalLaserCondTools::fillDb(CorrReader& r) {
  int iIov = 0;
  int processedIovs = 0;
  if (verb_ > 2)
    edm::LogPrint("EcalLaserCondTools") << "Starting filling DB...\n";
  int t1 = 0;
  int t3 = 0;
  int t2[nLmes];

  int prevT1 = 0;
  int prevT3 = 0;
  int prevT = 0;
  int t = 0;
  if (eventList_) {
    int iline = 0;
    while (!feof(eventList_)) {
      //skips comment lines:
      char c[2];
      while (fscanf(eventList_, " %1[#]%*[^\n]\n", &c[0]) == 1)
        ++iline;

      int n = fscanf(eventList_, "%*d %*d %*d %d%*[^\n]\n", &t);
      if (verb_ > 1)
        edm::LogPrint("EcalLaserCondTools") << "Event time: t = " << t << ", " << timeToString(t) << "\n";
      ++iline;
      if (n != 1)
        throw cms::Exception("User") << "Syntax error in event list file " << eventListFileName_ << " at line " << iline
                                     << " " << n << " "
                                     << ".\n";
      if (t < prevT)
        throw cms::Exception("User") << "Events in the event list file " << eventListFileName_
                                     << " are not ordered in increased time as required! See line " << iline << "\n";
      if (t == 0)
        throw cms::Exception("User") << "Found an unexpected t = 0 time value "
                                        "in the event list file"
                                     << eventListFileName_ << " at line " << iline << "!\n";
      //Look for IOV:
      bool iovFound = true;
      if (t <= t3) {  //IOV already inserted for previous event.
        if (verb_ > 1)
          edm::LogPrint("EcalLaserCondTools") << "Event in same IOV than previous one.\n";
        continue;
      }

      while ((iovFound = r.readTime(t1, t2, t3)) && t3 < t) /*NOP*/
        ;

      if (iovFound) {
        if (t1 < prevT1 && t3 < prevT3)
          throw cms::Exception("User")
              << "IOVs in the correction ascii file are not ordered in increased time as required!\n";
        else if (t1 < prevT1 || t3 < prevT3)
          throw cms::Exception("User") << "Found interleaved IOVs in the correction ascii file!\n";
        processIov(r, t1, t2, t3);
      } else {
        edm::LogPrint("EcalLaserCondTools") << "Warning: event beyond last IOV t3. Event time: " << timeToString(t)
                                            << ". Last IOV t3: " << timeToString(t3) << "\n";
      }
    }
  } else
    while (r.readTime(t1, t2, t3)) {
      ++iIov;
      if (iIov <= skipIov_) {
        edm::LogPrint("EcalLaserCondTools") << "Skipping IOV " << iIov << "\n";
        continue;
      } else if (processedIovs >= nIovs_ && nIovs_ >= 0) {
        edm::LogPrint("EcalLaserCondTools") << "Requested number of IOVs, " << nIovs_ << ", processed.\n";
        return;
      } else {
        processIov(r, t1, t2, t3);
        ++processedIovs;
      }
    }
}

void EcalLaserCondTools::processIov(CorrReader& r, int t1, int t2[EcalLaserCondTools::nLmes], int t3) {
  static int iIov = 0;
  ++iIov;

  //   FILE* fdebug = 0;
  //   if(i==) fdebug = fopen("debug.txt", "w");

  //  if(iIov <= skipIov_) { std::cout << "Skipping IOV " << iIov << "\n"; return; }

  cond::Time_t iovStart = 0;

  if (verb_ > 1) {
    edm::LogPrint("EcalLaserCondTools") << "t1:" << t1 << "(" << timeToString(t1) << ") \n"
                                        << "t3: " << t3 << "(" << timeToString(t3) << ")\nt2-t1: ";
    for (int i = 0; i < EcalLaserCondTools::nLmes; ++i) {
      edm::LogPrint("EcalLaserCondTools") << t2[i] - t1 << "\t";
    }

    edm::LogPrint("EcalLaserCondTools") << "\n";
  }
  if (t1 < fromTime_) {
    edm::LogPrint("EcalLaserCondTools") << "Skipping IOV " << iIov << ", "
                                        << ", which is before 'fromTime'," << timeToString(fromTime_) << "("
                                        << fromTime_ << ").\n";
    return;
  }

  if (toTime_ != -1 && t3 < toTime_) {
    edm::LogPrint("EcalLaserCondTools") << "Skipping IOV " << iIov << ", "
                                        << ", which is beyond 'toTime'," << timeToString(toTime_) << "(" << toTime_
                                        << ").\n";
    return;
  }

  if (t1 == 0) {
    edm::LogPrint("EcalLaserCondTools") << "Skipping IOV with t1 = 0"
                                        << "\n";
    return;
  }

  EcalLaserAPDPNRatios corrSet;

  EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
  iovStart = uint64_t(t1) << 32;
  for (size_t i = 0; i < EcalLaserCondTools::nLmes; ++i) {
    t.t1 = edm::Timestamp(uint64_t(t1) << 32);
    t.t2 = edm::Timestamp(uint64_t(t2[i]) << 32);
    t.t3 = edm::Timestamp(uint64_t(t3) << 32);
    corrSet.setTime(i, t);
  }

  constexpr int ncrystals = 75848;
  std::set<int> detidList;
  for (int i = 0; i < ncrystals; ++i) {
    DetId detid;
    //EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = {0, 0, 0};
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = EcalLaserAPDPNRatios::EcalLaserAPDPNpair();
    if (verb_ > 2)
      edm::LogPrint("EcalLaserCondTools") << "Reading " << toNth(i + 1) << " crystal\n";
    if (!r.readPs(detid, corr)) {
      throw cms::Exception("LasCor") << "Failed to read " << toNth(i + 1) << " crystal correction.\n";
    }

    std::pair<std::set<int>::iterator, bool> res = detidList.insert(int(detid));

    if (!res.second) {  //detid already processed
      edm::LogPrint("EcalLaserCondTools")
          << "Duplicate det id, for IOV " << iIov << " t1 = " << t1 << " detid = " << int(detid) << "\n";
    }

    if (!std::isfinite(corr.p1) || !std::isfinite(corr.p2) || !std::isfinite(corr.p3) || corr.p1 < minP_ ||
        corr.p1 > maxP_ || corr.p2 < minP_ || corr.p2 > maxP_ || corr.p3 < minP_ || corr.p3 > maxP_) {
      fprintf(ferr_, "%d %d %f %f %f\n", t1, (int)detid, corr.p1, corr.p2, corr.p3);
      corr.p1 = corr.p2 = corr.p3 = 1;
    }

    if (verb_ > 2) {
      if (detid.subdetId() == EcalBarrel) {
        edm::LogPrint("EcalLaserCondTools") << EBDetId(detid);
      } else if (detid.subdetId() == EcalEndcap) {
        edm::LogPrint("EcalLaserCondTools") << EEDetId(detid);
      } else {
        edm::LogPrint("EcalLaserCondTools") << (int)detid;
      }
      edm::LogPrint("EcalLaserCondTools") << ": "
                                          << "p1 = " << corr.p1 << "\t"
                                          << "p2 = " << corr.p2 << "\t"
                                          << "p3 = " << corr.p3 << "\n";
    }

    corrSet.setValue((int)detid, corr);
  }

  try {
    //Write correction set in DB (one IOV):
    if (db_->isNewTagRequest("EcalLaserAPDPNRatiosRcd")) {
      if (verb_)
        edm::LogPrint("EcalLaserCondTools") << "First IOV, extending starting time.\n";

      iovStart = db_->beginOfTime();
    }
    timeval t;
    gettimeofday(&t, nullptr);
    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools")
          << "[" << timeToString(t.tv_sec) << "] "
          << "Write IOV " << iIov << " starting from " << timeToString(iovStart >> 32) << "... ";
    db_->writeOneIOV(corrSet, iovStart, "EcalLaserAPDPNRatiosRcd");
  } catch (const cms::Exception& e) {
    edm::LogPrint("EcalLaserCondTools") << "Failed.\nException cathed while writting to cond DB" << e.what() << "\n";
  }
  edm::LogPrint("EcalLaserCondTools") << "Suceeded.\n";
}

bool EcalLaserCondTools::FileReader::nextFile() {
  for (;;) {
    ++ifile_;
    if (ifile_ >= fnames_.size()) {
      if (verb_ > 1)
        edm::LogPrint("EcalLaserCondTools") << "No more correction files.\n";

      return false;
    }
    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << "Opening file " << fnames_[ifile_] << "\n";

    f_ = fopen(fnames_[ifile_].c_str(), "r");
    iline_ = 0;
    if (f_ == nullptr) {
      std::cerr << "Failed to open file " << fnames_[ifile_] << ". File skipped!\n";
    } else {
      return true;
    }
  }
}

bool EcalLaserCondTools::FileReader::readTime(int& t1, int t2[EcalLaserCondTools::nLmes], int& t3) {
  trim();
  if ((f_ == nullptr || feof(f_)) && !nextFile()) {
    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << "No more record\n";

    return false;
  }
  int i;
  char* buf = nullptr;
  size_t s = 0;
  while ((i = fgetc(f_)) != 'T' && i != 'L' && i >= 0)
    getline(&buf, &s, f_);
  if (buf)
    free(buf);
  buf = nullptr;

  if (i == 'L') {  //last record put 3 consecutive times starting from end of prev. IOV
    t1 = t3;
    for (int i = 0; i < EcalLaserCondTools::nLmes; ++i)
      t2[i] = t1 + 1;
    t3 = t1 + 2;
    return true;
  }

  if (i != 'T') {
    if (verb_ > 1)
      edm::LogPrint("EcalLaserCondTools") << "No more record or bad line type/marker (getc returned " << i << ")\n";

    return false;
  }

  EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
  int n = fscanf(f_, "%d %d", &t1, &t3);
  for (int i = 0; i < EcalLaserCondTools::nLmes; ++i) {
    int nn = fscanf(f_, "%d", &t2[i]);
    if (nn != 1)
      break;
    n += nn;
  }

  int nnn = fscanf(f_, " ");

  if (n != (2 + EcalLaserCondTools::nLmes) || nnn != 0)
    throw cms::Exception("LasCorFile") << "File " << fnames_[ifile_] << " line " << iline_
                                       << ": syntax error. Expecting 'T' marker followed by 94 values: "
                                       << "t1 t2 t3(lme 1) t3(lme 2) ... t3(lme " << EcalLaserCondTools::nLmes << ")\n";

  return true;
}

bool EcalLaserCondTools::FileReader::readPs(DetId& detid, EcalLaserAPDPNRatios::EcalLaserAPDPNpair& corr) {
  if (f_ == nullptr) {
    if (verb_)
      edm::LogPrint("EcalLaserCondTools") << "Requested to read p1..p3 parameter line while no file is closed.\n";

    return false;
  }

  trim();
  int i = fgetc(f_);

  if (i != 'P') {
    if (verb_ && i >= 0)
      edm::LogPrint("EcalLaserCondTools") << "File " << fnames_[ifile_] << " line " << iline_
                                          << ": unexpected line type, '" << (char)i << "' while expecting 'P'\n";

    if (verb_ && i < 0)
      edm::LogPrint("EcalLaserCondTools") << "Failed to read p1..p3 parameter line\n";

    return false;
  }

  int rawdetid;
  int n = fscanf(f_, "%d %f %f %f\n", &rawdetid, &corr.p1, &corr.p2, &corr.p3);
  ++iline_;

  if (n != 4) {
    //    corr.p2=corr.p1;
    //    corr.p3=corr.p1;
    throw cms::Exception("I/O") << "Syntax error at line " << iline_ << "of file " << fnames_[ifile_] << " read " << n
                                << " values,"
                                << " raw id" << rawdetid << ": " << corr.p1 << ", " << corr.p2;
  }
  detid = rawdetid;
  constexpr int ECALID = 3;
  if (detid.det() != ECALID)
    throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                         << " contains an invalid det ID (detector code is not ECAL!)\n";
  if (detid.subdetId() == EcalBarrel) {
    EBDetId ebDetId(detid);
    if (!EBDetId::validDetId(ebDetId.ietaAbs(), ebDetId.iphi()))
      throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                           << " contains an invalid det ID (detector code is not ECAL!)\n";
  }
  if (detid.subdetId() == EcalEndcap) {
    EEDetId eeDetId(detid);
    if (!EEDetId::validDetId(eeDetId.ix(), eeDetId.iy(), eeDetId.zside()))
      throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                           << " contains an invalid det ID (detector code is not ECAL!)\n";
  }
  ++iline_;
  return true;
}

void EcalLaserCondTools::FileReader::trim() {
  if (f_ == nullptr)
    return;
  bool skipLine = false;
  int c;
  while ((c = fgetc(f_)) >= 0 && (c == ' ' || c == '\t' || c == '\n' || c == '#' || skipLine)) {
    if (c == '#')
      skipLine = true;
    if (c == '\n') {
      ++iline_;
      skipLine = false;
    }
  }
  ungetc(c, f_);
}

std::string EcalLaserCondTools::toNth(int n) {
  std::stringstream s;
  s << n;
  if (n % 100 < 10 || n % 100 > 20) {
    switch (n % 10) {
      case 1:
        s << "st";
        break;
      case 2:
        s << "nd";
        break;
      case 3:
        s << "rd";
        break;
      default:
        s << "th";
    }
  } else {
    s << "th";
  }
  return s.str();
}

std::string EcalLaserCondTools::timeToString(time_t t) {
  char buf[256];
  struct tm lt;
  localtime_r(&t, &lt);
  strftime(buf, sizeof(buf), "%F %R:%S", &lt);
  buf[sizeof(buf) - 1] = 0;
  return std::string(buf);
}

void EcalLaserCondTools::dbToAscii(const edm::EventSetup& es) {
  const auto& laserAPDPNRatios = es.getData(laserAPDPNRatiosToken_);

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& p = laserAPDPNRatios.getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& t = laserAPDPNRatios.getTimeMap();

  if (t.size() != EcalLaserCondTools::nLmes)
    throw cms::Exception("LasCor") << "Unexpected number time parameter triplets\n";

  if (fout_ == nullptr) {
    fout_ = fopen("corr_dump.txt", "w");
    if (fout_ == nullptr)
      throw cms::Exception("LasCor") << "Failed to create file corr_dump.txt\n";
  }

  unsigned t1 = t[0].t1.unixTime();
  unsigned t3 = t[0].t3.unixTime();
  fprintf(fout_, "T %d\t%d", t1, t3);

  if (verb_)
    edm::LogPrint("EcalLaserCondTools") << "Processing IOV " << t1 << " - " << t3 << "(" << timeToString(t1) << " - "
                                        << timeToString(t3) << "\n";

  for (unsigned i = 0; i < t.size(); ++i) {
    if (t[i].t1.unixTime() != t1 || t[i].t3.unixTime() != t3) {
      throw cms::Exception("LasCor") << "Inconsitency in t1, t3: "
                                     << "t1(lme 1) =" << t1 << ", t1(lme " << (i + 1) << ") = " << t[i].t1.unixTime()
                                     << ", t3(lme 1) =" << t3 << ", t3(lme " << (i + 1) << ") = " << t[i].t3.unixTime()
                                     << "\n";
    }
    fprintf(fout_, "\t%d", t[i].t2.unixTime());
  }
  fputc('\n', fout_);
  fflush(fout_);

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId detId(ieta, iphi);
        EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = p.barrel(detId.hashedIndex());
        fprintf(fout_, "P %d\t%f\t%f\t%f\n", (int)detId, corr.p1, corr.p2, corr.p3);
      }
    }
  }

  for (int iZ = 1; iZ >= -1; --iZ) {
    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        if (EEDetId::validDetId(iX, iY, iZ)) {
          EEDetId detId(iX, iY, iZ);
          EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = p.endcap(detId.hashedIndex());
          fprintf(fout_, "P %d\t%f\t%f\t%f\n", (int)detId, corr.p1, corr.p2, corr.p3);
        }
      }
    }
  }
}
//DEFINE_FWK_MODULE(EcalLaserCondTools);
