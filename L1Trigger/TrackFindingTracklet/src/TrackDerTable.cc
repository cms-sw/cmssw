#include "L1Trigger/TrackFindingTracklet/interface/TrackDerTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

TrackDerTable::TrackDerTable(Settings const& settings) : settings_(settings) {
  Nlay_ = N_LAYER;
  Ndisk_ = N_DISK;

  LayerMemBits_ = 6;
  DiskMemBits_ = 7;
  LayerDiskMemBits_ = 18;

  alphaBits_ = settings_.alphaBitsTable();

  nextLayerValue_ = 0;
  nextDiskValue_ = 0;
  nextLayerDiskValue_ = 0;
  lastMultiplicity_ = (1 << (3 * alphaBits_));

  for (int i = 0; i < (1 << Nlay_); i++) {
    LayerMem_.push_back(-1);
  }

  for (int i = 0; i < (1 << (2 * Ndisk_)); i++) {
    DiskMem_.push_back(-1);
  }

  for (int i = 0; i < (1 << (LayerMemBits_ + DiskMemBits_)); i++) {
    LayerDiskMem_.push_back(-1);
  }
}

const TrackDer* TrackDerTable::getDerivatives(unsigned int layermask,
                                              unsigned int diskmask,
                                              unsigned int alphaindex,
                                              unsigned int rinvindex) const {
  int index = getIndex(layermask, diskmask);
  if (index < 0) {
    return nullptr;
  }
  return &derivatives_[index + alphaindex * (1 << settings_.nrinvBitsTable()) + rinvindex];
}

int TrackDerTable::getIndex(unsigned int layermask, unsigned int diskmask) const {
  assert(layermask < LayerMem_.size());

  assert(diskmask < DiskMem_.size());

  int layercode = LayerMem_[layermask];
  int diskcode = DiskMem_[diskmask];

  if (diskcode < 0 || layercode < 0) {
    if (settings_.warnNoDer()) {
      edm::LogPrint("Tracklet") << "layermask diskmask : " << layermask << " " << diskmask;
    }
    return -1;
  }

  assert(layercode >= 0);
  assert(layercode < (1 << LayerMemBits_));
  assert(diskcode >= 0);
  assert(diskcode < (1 << DiskMemBits_));

  int layerdiskaddress = layercode + (diskcode << LayerMemBits_);

  assert(layerdiskaddress >= 0);
  assert(layerdiskaddress < (1 << (LayerMemBits_ + DiskMemBits_)));

  int address = LayerDiskMem_[layerdiskaddress];

  if (address < 0) {
    if (settings_.warnNoDer()) {
      edm::LogVerbatim("Tracklet") << "layermask diskmask : " << layermask << " " << diskmask;
    }
    return -1;
  }

  assert(address >= 0);
  assert(address < (1 << LayerDiskMemBits_));

  return address;
}

void TrackDerTable::addEntry(unsigned int layermask, unsigned int diskmask, int multiplicity, int nrinv) {
  assert(multiplicity <= (1 << (3 * alphaBits_)));

  assert(layermask < (unsigned int)(1 << Nlay_));

  assert(diskmask < (unsigned int)(1 << (2 * Ndisk_)));

  if (LayerMem_[layermask] == -1) {
    LayerMem_[layermask] = nextLayerValue_++;
  }
  if (DiskMem_[diskmask] == -1) {
    DiskMem_[diskmask] = nextDiskValue_++;
  }

  int layercode = LayerMem_[layermask];
  int diskcode = DiskMem_[diskmask];

  assert(layercode >= 0);
  assert(layercode < (1 << LayerMemBits_));
  assert(diskcode >= 0);
  assert(diskcode < (1 << DiskMemBits_));

  int layerdiskaddress = layercode + (diskcode << LayerMemBits_);

  assert(layerdiskaddress >= 0);
  assert(layerdiskaddress < (1 << (LayerMemBits_ + DiskMemBits_)));

  int address = LayerDiskMem_[layerdiskaddress];

  if (address != -1) {
    edm::LogPrint("Tracklet") << "Duplicate entry:  layermask=" << layermask << " diskmaks=" << diskmask;
  }

  assert(address == -1);

  LayerDiskMem_[layerdiskaddress] = nextLayerDiskValue_;

  nextLayerDiskValue_ += multiplicity * nrinv;

  lastMultiplicity_ = multiplicity * nrinv;

  for (int i = 0; i < multiplicity; i++) {
    for (int irinv = 0; irinv < nrinv; irinv++) {
      TrackDer tmp;
      tmp.setIndex(layermask, diskmask, i, irinv);
      derivatives_.push_back(tmp);
    }
  }
}

void TrackDerTable::readPatternFile(std::string fileName) {
  ifstream in(fileName.c_str());
  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "reading fit pattern file " << fileName;
    edm::LogVerbatim("Tracklet") << "  flags (good/eof/fail/bad): " << in.good() << " " << in.eof() << " " << in.fail()
                                 << " " << in.bad();
  }

  while (in.good()) {
    std::string layerstr, diskstr;
    int multiplicity;

    in >> layerstr >> diskstr >> multiplicity;

    //correct multiplicity if you dont want 3 bits of alpha.
    if (alphaBits_ == 2) {
      if (multiplicity == 8)
        multiplicity = 4;
      if (multiplicity == 64)
        multiplicity = 16;
      if (multiplicity == 512)
        multiplicity = 64;
    }

    if (alphaBits_ == 1) {
      if (multiplicity == 8)
        multiplicity = 2;
      if (multiplicity == 64)
        multiplicity = 4;
      if (multiplicity == 512)
        multiplicity = 8;
    }

    if (!in.good())
      continue;

    char** tmpptr = nullptr;

    int layers = strtol(layerstr.c_str(), tmpptr, 2);
    int disks = strtol(diskstr.c_str(), tmpptr, 2);

    addEntry(layers, disks, multiplicity, (1 << settings_.nrinvBitsTable()));
  }
}

void TrackDerTable::fillTable() {
  int nentries = getEntries();

  for (int i = 0; i < nentries; i++) {
    TrackDer& der = derivatives_[i];
    int layermask = der.layerMask();
    int diskmask = der.diskMask();
    int alphamask = der.alphaMask();
    int irinv = der.irinv();

    double rinv = (irinv - ((1 << (settings_.nrinvBitsTable() - 1)) - 0.5)) * settings_.rinvmax() /
                  (1 << (settings_.nrinvBitsTable() - 1));

    bool print = false;

    if (print) {
      edm::LogVerbatim("Tracklet") << "PRINT i " << i << " " << layermask << " " << diskmask << " " << alphamask << " "
                                   << print;
    }

    int nlayers = 0;
    double r[N_LAYER];

    for (unsigned l = 0; l < N_LAYER; l++) {
      if (layermask & (1 << (N_LAYER - 1 - l))) {
        r[nlayers] = settings_.rmean(l);
        nlayers++;
      }
    }

    int ndisks = 0;
    double z[N_DISK];
    double alpha[N_DISK];

    double t = tpar(settings_, diskmask, layermask);

    for (unsigned d = 0; d < N_DISK; d++) {
      if (diskmask & (3 << (2 * (N_DISK - 1 - d)))) {
        z[ndisks] = settings_.zmean(d);
        alpha[ndisks] = 0.0;
        double r = settings_.zmean(d) / t;
        double r2 = r * r;
        if (diskmask & (1 << (2 * (N_DISK - 1 - d)))) {
          if (alphaBits_ == 3) {
            int ialpha = alphamask & 7;
            alphamask = alphamask >> 3;
            alpha[ndisks] = settings_.half2SmoduleWidth() * (ialpha - 3.5) / 4.0 / r2;
            if (print)
              edm::LogVerbatim("Tracklet") << "PRINT 3 alpha ialpha : " << alpha[ndisks] << " " << ialpha;
          }
          if (alphaBits_ == 2) {
            int ialpha = alphamask & 3;
            alphamask = alphamask >> 2;
            alpha[ndisks] = settings_.half2SmoduleWidth() * (ialpha - 1.5) / 2.0 / r2;
          }
          if (alphaBits_ == 1) {
            int ialpha = alphamask & 1;
            alphamask = alphamask >> 1;
            alpha[ndisks] = settings_.half2SmoduleWidth() * (ialpha - 0.5) / r2;
            if (print)
              edm::LogVerbatim("Tracklet") << "PRINT 1 alpha ialpha : " << alpha[ndisks] << " " << ialpha;
          }
        }
        ndisks++;
      }
    }

    double D[N_FITPARAM][N_FITSTUB * 2];
    int iD[N_FITPARAM][N_FITSTUB * 2];
    double MinvDt[N_FITPARAM][N_FITSTUB * 2];
    double MinvDtDelta[N_FITPARAM][N_FITSTUB * 2];
    int iMinvDt[N_FITPARAM][N_FITSTUB * 2];
    double sigma[N_FITSTUB * 2];
    double kfactor[N_FITSTUB * 2];

    if (print) {
      edm::LogVerbatim("Tracklet") << "PRINT ndisks alpha[0] z[0] t: " << ndisks << " " << alpha[0] << " " << z[0]
                                   << " " << t;
      for (int iii = 0; iii < nlayers; iii++) {
        edm::LogVerbatim("Tracklet") << "PRINT iii r: " << iii << " " << r[iii];
      }
    }

    calculateDerivatives(settings_, nlayers, r, ndisks, z, alpha, t, rinv, D, iD, MinvDt, iMinvDt, sigma, kfactor);

    double delta = 0.1;

    for (int i = 0; i < nlayers; i++) {
      if (r[i] > settings_.rPS2S())
        continue;

      r[i] += delta;

      calculateDerivatives(
          settings_, nlayers, r, ndisks, z, alpha, t, rinv, D, iD, MinvDtDelta, iMinvDt, sigma, kfactor);

      for (int ii = 0; ii < nlayers; ii++) {
        if (r[ii] > settings_.rPS2S())
          continue;
        double tder = (MinvDtDelta[2][2 * ii + 1] - MinvDt[2][2 * ii + 1]) / delta;
        int itder = (1 << (settings_.fittbitshift() + settings_.rcorrbits())) * tder * settings_.kr() * settings_.kz() /
                    settings_.ktpars();
        double zder = (MinvDtDelta[3][2 * ii + 1] - MinvDt[3][2 * ii + 1]) / delta;
        int izder = (1 << (settings_.fitz0bitshift() + settings_.rcorrbits())) * zder * settings_.kr() *
                    settings_.kz() / settings_.kz0pars();
        der.settdzcorr(i, ii, tder);
        der.setz0dzcorr(i, ii, zder);
        der.setitdzcorr(i, ii, itder);
        der.setiz0dzcorr(i, ii, izder);
      }

      r[i] -= delta;
    }

    if (print) {
      edm::LogVerbatim("Tracklet") << "iMinvDt table build : " << iMinvDt[0][10] << " " << iMinvDt[1][10] << " "
                                   << iMinvDt[2][10] << " " << iMinvDt[3][10] << " " << t << " " << nlayers << " "
                                   << ndisks;

      std::string oss = "alpha :";
      for (int iii = 0; iii < ndisks; iii++) {
        oss += " ";
        oss += std::to_string(alpha[iii]);
      }
      edm::LogVerbatim("Tracklet") << oss;
      oss = "z :";
      for (int iii = 0; iii < ndisks; iii++) {
        oss += " ";
        oss += std::to_string(z[iii]);
      }
      edm::LogVerbatim("Tracklet") << oss;
    }

    if (print) {
      edm::LogVerbatim("Tracklet") << "PRINT nlayers ndisks : " << nlayers << " " << ndisks;
    }

    for (int j = 0; j < nlayers + ndisks; j++) {
      der.settpar(t);

      //integer
      assert(std::abs(iMinvDt[0][2 * j]) < (1 << 23));
      assert(std::abs(iMinvDt[0][2 * j + 1]) < (1 << 23));
      assert(std::abs(iMinvDt[1][2 * j]) < (1 << 23));
      assert(std::abs(iMinvDt[1][2 * j + 1]) < (1 << 23));
      assert(std::abs(iMinvDt[2][2 * j]) < (1 << 19));
      assert(std::abs(iMinvDt[2][2 * j + 1]) < (1 << 19));
      assert(std::abs(iMinvDt[3][2 * j]) < (1 << 19));
      assert(std::abs(iMinvDt[3][2 * j + 1]) < (1 << 19));

      if (print) {
        edm::LogVerbatim("Tracklet") << "PRINT i " << i << " " << j << " " << iMinvDt[1][2 * j] << " "
                                     << std::abs(iMinvDt[1][2 * j]);
      }

      der.setirinvdphi(j, iMinvDt[0][2 * j]);
      der.setirinvdzordr(j, iMinvDt[0][2 * j + 1]);
      der.setiphi0dphi(j, iMinvDt[1][2 * j]);
      der.setiphi0dzordr(j, iMinvDt[1][2 * j + 1]);
      der.setitdphi(j, iMinvDt[2][2 * j]);
      der.setitdzordr(j, iMinvDt[2][2 * j + 1]);
      der.setiz0dphi(j, iMinvDt[3][2 * j]);
      der.setiz0dzordr(j, iMinvDt[3][2 * j + 1]);
      //floating point
      der.setrinvdphi(j, MinvDt[0][2 * j]);
      der.setrinvdzordr(j, MinvDt[0][2 * j + 1]);
      der.setphi0dphi(j, MinvDt[1][2 * j]);
      der.setphi0dzordr(j, MinvDt[1][2 * j + 1]);
      der.settdphi(j, MinvDt[2][2 * j]);
      der.settdzordr(j, MinvDt[2][2 * j + 1]);
      der.setz0dphi(j, MinvDt[3][2 * j]);
      der.setz0dzordr(j, MinvDt[3][2 * j + 1]);
    }
  }

  if (settings_.writeTable()) {
    ofstream outL = openfile(settings_.tablePath(), "FitDerTableNew_LayerMem.tab", __FILE__, __LINE__);

    int nbits = 6;
    for (unsigned int i = 0; i < LayerMem_.size(); i++) {
      FPGAWord tmp;
      int tmp1 = LayerMem_[i];
      if (tmp1 < 0)
        tmp1 = (1 << nbits) - 1;
      tmp.set(tmp1, nbits, true, __LINE__, __FILE__);
      outL << tmp.str() << endl;
    }
    outL.close();

    ofstream outD = openfile(settings_.tablePath(), "FitDerTableNew_DiskMem.tab", __FILE__, __LINE__);

    nbits = 7;
    for (int tmp1 : DiskMem_) {
      if (tmp1 < 0)
        tmp1 = (1 << nbits) - 1;
      FPGAWord tmp;
      tmp.set(tmp1, nbits, true, __LINE__, __FILE__);
      outD << tmp.str() << endl;
    }
    outD.close();

    ofstream outLD = openfile(settings_.tablePath(), "FitDerTableNew_LayerDiskMem.tab", __FILE__, __LINE__);

    nbits = 15;
    for (int tmp1 : LayerDiskMem_) {
      if (tmp1 < 0)
        tmp1 = (1 << nbits) - 1;
      FPGAWord tmp;
      tmp.set(tmp1, nbits, true, __LINE__, __FILE__);
      outLD << tmp.str() << endl;
    }
    outLD.close();

    const std::array<string, N_TRKLSEED> seedings = {{"L1L2", "L3L4", "L5L6", "D1D2", "D3D4", "D1L1", "D1L2"}};
    const string prefix = settings_.tablePath() + "FitDerTableNew_";

    // open files for derivative tables

    ofstream outrinvdphi[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Rinvdphi_" + seedings[i] + ".tab";
      outrinvdphi[i].open(fname);
      if (outrinvdphi[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outrinvdzordr[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Rinvdzordr_" + seedings[i] + ".tab";
      outrinvdzordr[i].open(fname);
      if (outrinvdzordr[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outphi0dphi[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Phi0dphi_" + seedings[i] + ".tab";
      outphi0dphi[i].open(fname);
      if (outphi0dphi[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outphi0dzordr[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Phi0dzordr_" + seedings[i] + ".tab";
      outphi0dzordr[i].open(fname);
      if (outphi0dzordr[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outtdphi[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Tdphi_" + seedings[i] + ".tab";
      outtdphi[i].open(fname);
      if (outtdphi[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outtdzordr[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Tdzordr_" + seedings[i] + ".tab";
      outtdzordr[i].open(fname);
      if (outtdzordr[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outz0dphi[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      const string fname = prefix + "Z0dphi_" + seedings[i] + ".tab";
      outz0dphi[i].open(fname);
      if (outz0dphi[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    ofstream outz0dzordr[N_TRKLSEED];
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      string fname = prefix + "Z0dzordr_" + seedings[i] + ".tab";
      outz0dzordr[i].open(fname);
      if (outz0dzordr[i].fail())
        throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;
    }

    for (auto& der : derivatives_) {
      unsigned int layerhits = der.layerMask();  // 6 bits layer hit pattern
      unsigned int diskmask = der.diskMask();    // 10 bits disk hit pattern
      unsigned int diskhits = 0;
      if (diskmask & (3 << 8))
        diskhits += 16;
      if (diskmask & (3 << 6))
        diskhits += 8;
      if (diskmask & (3 << 4))
        diskhits += 4;
      if (diskmask & (3 << 2))
        diskhits += 2;
      if (diskmask & (3 << 0))
        diskhits += 1;
      assert(diskhits < 32);                            // 5 bits
      unsigned int hits = (layerhits << 5) + diskhits;  // 11 bits hit pattern
      assert(hits < 4096);

      // loop over all seedings
      int i = 0;  // seeding index
      for (const string& seed : seedings) {
        unsigned int iseed1 = 0;
        unsigned int iseed2 = 0;
        // check if the seeding is good for the current hit pattern
        if (seed == "L1L2") {
          iseed1 = 1;
          iseed2 = 2;
        }
        if (seed == "L3L4") {
          iseed1 = 3;
          iseed2 = 4;
        }
        if (seed == "L5L6") {
          iseed1 = 5;
          iseed2 = 6;
        }
        if (seed == "D1D2") {
          iseed1 = 7;
          iseed2 = 8;
        }
        if (seed == "D3D4") {
          iseed1 = 9;
          iseed2 = 10;
        }
        if (seed == "D1L1") {
          iseed1 = 7;
          iseed2 = 1;
        }
        if (seed == "D1L2") {
          iseed1 = 7;
          iseed2 = 2;
        }

        bool goodseed = (hits & (1 << (11 - iseed1))) and (hits & (1 << (11 - iseed2)));

        int itmprinvdphi[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmprinvdzordr[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmpphi0dphi[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmpphi0dzordr[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmptdphi[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmptdzordr[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmpz0dphi[N_PROJ] = {9999999, 9999999, 9999999, 9999999};
        int itmpz0dzordr[N_PROJ] = {9999999, 9999999, 9999999, 9999999};

        // loop over bits in hit pattern
        int ider = 0;
        if (goodseed) {
          for (unsigned int ihit = 1; ihit < N_FITSTUB * 2; ++ihit) {
            // skip seeding layers
            if (ihit == iseed1 or ihit == iseed2) {
              ider++;
              continue;
            }
            // skip if no hit
            if (not(hits & (1 << (11 - ihit))))
              continue;

            int inputI = -1;
            if (seed == "L1L2") {
              if (ihit == 3 or ihit == 10)
                inputI = 0;  // L3 or D4
              if (ihit == 4 or ihit == 9)
                inputI = 1;  // L4 or D3
              if (ihit == 5 or ihit == 8)
                inputI = 2;  // L5 or D2
              if (ihit == 6 or ihit == 7)
                inputI = 3;  // L6 or D1
            } else if (seed == "L3L4") {
              if (ihit == 1)
                inputI = 0;  // L1
              if (ihit == 2)
                inputI = 1;  // L2
              if (ihit == 5 or ihit == 8)
                inputI = 2;  // L5 or D2
              if (ihit == 6 or ihit == 7)
                inputI = 3;  // L6 or D1
            } else if (seed == "L5L6") {
              if (ihit == 1)
                inputI = 0;  // L1
              if (ihit == 2)
                inputI = 1;  // L2
              if (ihit == 3)
                inputI = 2;  // L3
              if (ihit == 4)
                inputI = 3;  // L4
            } else if (seed == "D1D2") {
              if (ihit == 1)
                inputI = 0;  // L1
              if (ihit == 9)
                inputI = 1;  // D3
              if (ihit == 10)
                inputI = 2;  // D4
              if (ihit == 2 or ihit == 11)
                inputI = 3;  // L2 or D5
            } else if (seed == "D3D4") {
              if (ihit == 1)
                inputI = 0;  // L1
              if (ihit == 7)
                inputI = 1;  // D1
              if (ihit == 8)
                inputI = 2;  // D2
              if (ihit == 2 or ihit == 11)
                inputI = 3;  // L2 or D5
            } else if (seed == "D1L1" or "D1L2") {
              if (ihit == 8)
                inputI = 0;  // D2
              if (ihit == 9)
                inputI = 1;  // D3
              if (ihit == 10)
                inputI = 2;  // D4
              if (ihit == 11)
                inputI = 3;  // D5
            }
            if (inputI >= 0 and inputI < (int)N_PROJ) {
              itmprinvdphi[inputI] = der.irinvdphi(ider);
              itmprinvdzordr[inputI] = der.irinvdzordr(ider);
              itmpphi0dphi[inputI] = der.iphi0dphi(ider);
              itmpphi0dzordr[inputI] = der.iphi0dzordr(ider);
              itmptdphi[inputI] = der.itdphi(ider);
              itmptdzordr[inputI] = der.itdzordr(ider);
              itmpz0dphi[inputI] = der.iz0dphi(ider);
              itmpz0dzordr[inputI] = der.iz0dzordr(ider);
            }

            ider++;

          }  // for (unsigned int ihit = 1; ihit < 12; ++ihit)
        }    // if (goodseed)

        FPGAWord tmprinvdphi[N_PROJ];
        int nbits = 16;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmprinvdphi[j] > (1 << nbits))
            itmprinvdphi[j] = (1 << nbits) - 1;
          tmprinvdphi[j].set(itmprinvdphi[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outrinvdphi[i] << tmprinvdphi[0].str() << tmprinvdphi[1].str() << tmprinvdphi[2].str() << tmprinvdphi[3].str()
                       << endl;

        FPGAWord tmprinvdzordr[N_PROJ];
        nbits = 15;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmprinvdzordr[j] > (1 << nbits))
            itmprinvdzordr[j] = (1 << nbits) - 1;
          tmprinvdzordr[j].set(itmprinvdzordr[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outrinvdzordr[i] << tmprinvdzordr[0].str() << tmprinvdzordr[1].str() << tmprinvdzordr[2].str()
                         << tmprinvdzordr[3].str() << endl;

        FPGAWord tmpphi0dphi[N_PROJ];
        nbits = 13;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmpphi0dphi[j] > (1 << nbits))
            itmpphi0dphi[j] = (1 << nbits) - 1;
          tmpphi0dphi[j].set(itmpphi0dphi[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outphi0dphi[i] << tmpphi0dphi[0].str() << tmpphi0dphi[1].str() << tmpphi0dphi[2].str() << tmpphi0dphi[3].str()
                       << endl;

        FPGAWord tmpphi0dzordr[N_PROJ];
        nbits = 15;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmpphi0dzordr[j] > (1 << nbits))
            itmpphi0dzordr[j] = (1 << nbits) - 1;
          tmpphi0dzordr[j].set(itmpphi0dzordr[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outphi0dzordr[i] << tmpphi0dzordr[0].str() << tmpphi0dzordr[1].str() << tmpphi0dzordr[2].str()
                         << tmpphi0dzordr[3].str() << endl;

        FPGAWord tmptdphi[N_PROJ];
        nbits = 14;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmptdphi[j] > (1 << nbits))
            itmptdphi[j] = (1 << nbits) - 1;
          tmptdphi[j].set(itmptdphi[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outtdphi[i] << tmptdphi[0].str() << tmptdphi[1].str() << tmptdphi[2].str() << tmptdphi[3].str() << endl;

        FPGAWord tmptdzordr[N_PROJ];
        nbits = 15;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmptdzordr[j] > (1 << nbits))
            itmptdzordr[j] = (1 << nbits) - 1;
          tmptdzordr[j].set(itmptdzordr[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outtdzordr[i] << tmptdzordr[0].str() << tmptdzordr[1].str() << tmptdzordr[2].str() << tmptdzordr[3].str()
                      << endl;

        FPGAWord tmpz0dphi[N_PROJ];
        nbits = 13;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmpz0dphi[j] > (1 << nbits))
            itmpz0dphi[j] = (1 << nbits) - 1;
          tmpz0dphi[j].set(itmpz0dphi[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outz0dphi[i] << tmpz0dphi[0].str() << tmpz0dphi[1].str() << tmpz0dphi[2].str() << tmpz0dphi[3].str() << endl;

        FPGAWord tmpz0dzordr[N_PROJ];
        nbits = 15;
        for (unsigned int j = 0; j < N_PROJ; ++j) {
          if (itmpz0dzordr[j] > (1 << nbits))
            itmpz0dzordr[j] = (1 << nbits) - 1;
          tmpz0dzordr[j].set(itmpz0dzordr[j], nbits + 1, false, __LINE__, __FILE__);
        }
        outz0dzordr[i] << tmpz0dzordr[0].str() << tmpz0dzordr[1].str() << tmpz0dzordr[2].str() << tmpz0dzordr[3].str()
                       << endl;

        i++;
      }  // for (const string & seed : seedings)

    }  // for (auto & der : derivatives_)

    // close files
    for (unsigned int i = 0; i < N_TRKLSEED; ++i) {
      outrinvdphi[i].close();
      outrinvdzordr[i].close();
      outphi0dphi[i].close();
      outphi0dzordr[i].close();
      outtdphi[i].close();
      outtdzordr[i].close();
      outz0dphi[i].close();
      outz0dzordr[i].close();
    }

  }  // if (writeFitDerTable)
}

void TrackDerTable::invert(double M[4][8], unsigned int n) {
  assert(n <= 4);

  unsigned int i, j, k;
  double ratio, a;

  for (i = 0; i < n; i++) {
    for (j = n; j < 2 * n; j++) {
      if (i == (j - n))
        M[i][j] = 1.0;
      else
        M[i][j] = 0.0;
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i != j) {
        ratio = M[j][i] / M[i][i];
        for (k = 0; k < 2 * n; k++) {
          M[j][k] -= ratio * M[i][k];
        }
      }
    }
  }

  for (i = 0; i < n; i++) {
    a = M[i][i];
    for (j = 0; j < 2 * n; j++) {
      M[i][j] /= a;
    }
  }
}

void TrackDerTable::invert(std::vector<std::vector<double> >& M, unsigned int n) {
  assert(M.size() == n);
  assert(M[0].size() == 2 * n);

  unsigned int i, j, k;
  double ratio, a;

  for (i = 0; i < n; i++) {
    for (j = n; j < 2 * n; j++) {
      if (i == (j - n))
        M[i][j] = 1.0;
      else
        M[i][j] = 0.0;
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i != j) {
        ratio = M[j][i] / M[i][i];
        for (k = 0; k < 2 * n; k++) {
          M[j][k] -= ratio * M[i][k];
        }
      }
    }
  }

  for (i = 0; i < n; i++) {
    a = M[i][i];
    for (j = 0; j < 2 * n; j++) {
      M[i][j] /= a;
    }
  }
}

void TrackDerTable::calculateDerivatives(Settings const& settings,
                                         unsigned int nlayers,
                                         double r[N_LAYER],
                                         unsigned int ndisks,
                                         double z[N_DISK],
                                         double alpha[N_DISK],
                                         double t,
                                         double rinv,
                                         double D[N_FITPARAM][N_FITSTUB * 2],
                                         int iD[N_FITPARAM][N_FITSTUB * 2],
                                         double MinvDt[N_FITPARAM][N_FITSTUB * 2],
                                         int iMinvDt[N_FITPARAM][N_FITSTUB * 2],
                                         double sigma[N_FITSTUB * 2],
                                         double kfactor[N_FITSTUB * 2]) {
  double sigmax = settings.stripPitch(true) / sqrt(12.0);
  double sigmaz = settings.stripLength(true) / sqrt(12.0);
  double sigmaz2 = settings.stripLength(false) / sqrt(12.0);

  double sigmazpsbarrel = sigmaz;  //This is a bit of a hack - these weights should be properly determined
  if (std::abs(t) > 2.0)
    sigmazpsbarrel = sigmaz * std::abs(t) / 2.0;
  if (std::abs(t) > 3.8)
    sigmazpsbarrel = sigmaz * std::abs(t);

  double sigmax2sdisk = settings.stripPitch(false) / sqrt(12.0);
  double sigmaz2sdisk = settings.stripLength(false) / sqrt(12.0);

  double sigmaxpsdisk = settings.stripPitch(true) / sqrt(12.0);
  double sigmazpsdisk = settings.stripLength(true) / sqrt(12.0);

  unsigned int n = nlayers + ndisks;

  assert(n <= N_FITSTUB);

  double rnew[N_FITSTUB];

  int j = 0;

  //here we handle a barrel hit
  for (unsigned int i = 0; i < nlayers; i++) {
    double ri = r[i];

    rnew[i] = ri;

    //first we have the phi position
    D[0][j] = -0.5 * ri * ri / sqrt(1 - 0.25 * ri * ri * rinv * rinv) / sigmax;
    D[1][j] = ri / sigmax;
    D[2][j] = 0.0;
    D[3][j] = 0.0;
    sigma[j] = sigmax;
    kfactor[j] = settings.kphi1();
    j++;
    //second the z position
    D[0][j] = 0.0;
    D[1][j] = 0.0;
    if (ri < settings.rPS2S()) {
      D[2][j] = (2 / rinv) * asin(0.5 * ri * rinv) / sigmazpsbarrel;
      D[3][j] = 1.0 / sigmazpsbarrel;
      sigma[j] = sigmazpsbarrel;
      kfactor[j] = settings.kz();
    } else {
      D[2][j] = (2 / rinv) * asin(0.5 * ri * rinv) / sigmaz2;
      D[3][j] = 1.0 / sigmaz2;
      sigma[j] = sigmaz2;
      kfactor[j] = settings.kz();
    }

    j++;
  }

  for (unsigned int i = 0; i < ndisks; i++) {
    double zi = z[i];

    double z0 = 0.0;

    double rmultiplier = alpha[i] * zi / t;

    double phimultiplier = zi / t;

    double drdrinv = -2.0 * sin(0.5 * rinv * (zi - z0) / t) / (rinv * rinv) +
                     (zi - z0) * cos(0.5 * rinv * (zi - z0) / t) / (rinv * t);
    double drdphi0 = 0;
    double drdt = -(zi - z0) * cos(0.5 * rinv * (zi - z0) / t) / (t * t);
    double drdz0 = -cos(0.5 * rinv * (zi - z0) / t) / t;

    double dphidrinv = -0.5 * (zi - z0) / t;
    double dphidphi0 = 1.0;
    double dphidt = 0.5 * rinv * (zi - z0) / (t * t);
    double dphidz0 = 0.5 * rinv / t;

    double r = (zi - z0) / t;

    rnew[i + nlayers] = r;

    sigma[j] = sigmax2sdisk;
    if (std::abs(alpha[i]) < 1e-10) {
      sigma[j] = sigmaxpsdisk;
    }

    D[0][j] = (phimultiplier * dphidrinv + rmultiplier * drdrinv) / sigma[j];
    D[1][j] = (phimultiplier * dphidphi0 + rmultiplier * drdphi0) / sigma[j];
    D[2][j] = (phimultiplier * dphidt + rmultiplier * drdt) / sigma[j];
    D[3][j] = (phimultiplier * dphidz0 + rmultiplier * drdz0) / sigma[j];
    kfactor[j] = settings.kphi();

    j++;

    if (std::abs(alpha[i]) < 1e-10) {
      D[0][j] = drdrinv / sigmazpsdisk;
      D[1][j] = drdphi0 / sigmazpsdisk;
      D[2][j] = drdt / sigmazpsdisk;
      D[3][j] = drdz0 / sigmazpsdisk;
      sigma[j] = sigmazpsdisk;
      kfactor[j] = settings.kr();
    } else {
      D[0][j] = drdrinv / sigmaz2sdisk;
      D[1][j] = drdphi0 / sigmaz2sdisk;
      D[2][j] = drdt / sigmaz2sdisk;
      D[3][j] = drdz0 / sigmaz2sdisk;
      sigma[j] = sigmaz2sdisk;
      kfactor[j] = settings.kr();
    }

    j++;
  }

  double M[4][8];

  for (unsigned int i1 = 0; i1 < 4; i1++) {
    for (unsigned int i2 = 0; i2 < 4; i2++) {
      M[i1][i2] = 0.0;
      for (unsigned int j = 0; j < 2 * n; j++) {
        M[i1][i2] += D[i1][j] * D[i2][j];
      }
    }
  }

  invert(M, 4);

  for (unsigned int j = 0; j < N_FITSTUB * 2; j++) {
    for (unsigned int i1 = 0; i1 < N_FITPARAM; i1++) {
      MinvDt[i1][j] = 0.0;
      iMinvDt[i1][j] = 0;
    }
  }

  for (unsigned int j = 0; j < 2 * n; j++) {
    for (unsigned int i1 = 0; i1 < 4; i1++) {
      for (unsigned int i2 = 0; i2 < 4; i2++) {
        MinvDt[i1][j] += M[i1][i2 + 4] * D[i2][j];
      }
    }
  }

  for (unsigned int i = 0; i < n; i++) {
    iD[0][2 * i] =
        D[0][2 * i] * (1 << settings.chisqphifactbits()) * settings.krinvpars() / (1 << settings.fitrinvbitshift());
    iD[1][2 * i] =
        D[1][2 * i] * (1 << settings.chisqphifactbits()) * settings.kphi0pars() / (1 << settings.fitphi0bitshift());
    iD[2][2 * i] =
        D[2][2 * i] * (1 << settings.chisqphifactbits()) * settings.ktpars() / (1 << settings.fittbitshift());
    iD[3][2 * i] =
        D[3][2 * i] * (1 << settings.chisqphifactbits()) * settings.kz0pars() / (1 << settings.fitz0bitshift());

    iD[0][2 * i + 1] =
        D[0][2 * i + 1] * (1 << settings.chisqzfactbits()) * settings.krinvpars() / (1 << settings.fitrinvbitshift());
    iD[1][2 * i + 1] =
        D[1][2 * i + 1] * (1 << settings.chisqzfactbits()) * settings.kphi0pars() / (1 << settings.fitphi0bitshift());
    iD[2][2 * i + 1] =
        D[2][2 * i + 1] * (1 << settings.chisqzfactbits()) * settings.ktpars() / (1 << settings.fittbitshift());
    iD[3][2 * i + 1] =
        D[3][2 * i + 1] * (1 << settings.chisqzfactbits()) * settings.kz0pars() / (1 << settings.fitz0bitshift());

    //First the barrel
    if (i < nlayers) {
      MinvDt[0][2 * i] *= rnew[i] / sigmax;
      MinvDt[1][2 * i] *= rnew[i] / sigmax;
      MinvDt[2][2 * i] *= rnew[i] / sigmax;
      MinvDt[3][2 * i] *= rnew[i] / sigmax;

      iMinvDt[0][2 * i] =
          (1 << settings.fitrinvbitshift()) * MinvDt[0][2 * i] * settings.kphi1() / settings.krinvpars();
      iMinvDt[1][2 * i] =
          (1 << settings.fitphi0bitshift()) * MinvDt[1][2 * i] * settings.kphi1() / settings.kphi0pars();
      iMinvDt[2][2 * i] = (1 << settings.fittbitshift()) * MinvDt[2][2 * i] * settings.kphi1() / settings.ktpars();
      iMinvDt[3][2 * i] = (1 << settings.fitz0bitshift()) * MinvDt[3][2 * i] * settings.kphi1() / settings.kz0pars();

      if (rnew[i] < settings.rPS2S()) {
        MinvDt[0][2 * i + 1] /= sigmazpsbarrel;
        MinvDt[1][2 * i + 1] /= sigmazpsbarrel;
        MinvDt[2][2 * i + 1] /= sigmazpsbarrel;
        MinvDt[3][2 * i + 1] /= sigmazpsbarrel;

        iMinvDt[0][2 * i + 1] =
            (1 << settings.fitrinvbitshift()) * MinvDt[0][2 * i + 1] * settings.kz() / settings.krinvpars();
        iMinvDt[1][2 * i + 1] =
            (1 << settings.fitphi0bitshift()) * MinvDt[1][2 * i + 1] * settings.kz() / settings.kphi0pars();
        iMinvDt[2][2 * i + 1] =
            (1 << settings.fittbitshift()) * MinvDt[2][2 * i + 1] * settings.kz() / settings.ktpars();
        iMinvDt[3][2 * i + 1] =
            (1 << settings.fitz0bitshift()) * MinvDt[3][2 * i + 1] * settings.kz() / settings.kz0pars();
      } else {
        MinvDt[0][2 * i + 1] /= sigmaz2;
        MinvDt[1][2 * i + 1] /= sigmaz2;
        MinvDt[2][2 * i + 1] /= sigmaz2;
        MinvDt[3][2 * i + 1] /= sigmaz2;

        int fact = (1 << (settings.nzbitsstub(0) - settings.nzbitsstub(5)));

        iMinvDt[0][2 * i + 1] =
            (1 << settings.fitrinvbitshift()) * MinvDt[0][2 * i + 1] * fact * settings.kz() / settings.krinvpars();
        iMinvDt[1][2 * i + 1] =
            (1 << settings.fitphi0bitshift()) * MinvDt[1][2 * i + 1] * fact * settings.kz() / settings.kphi0pars();
        iMinvDt[2][2 * i + 1] =
            (1 << settings.fittbitshift()) * MinvDt[2][2 * i + 1] * fact * settings.kz() / settings.ktpars();
        iMinvDt[3][2 * i + 1] =
            (1 << settings.fitz0bitshift()) * MinvDt[3][2 * i + 1] * fact * settings.kz() / settings.kz0pars();
      }
    }

    //Secondly the disks
    else {
      double denom = (std::abs(alpha[i - nlayers]) < 1e-10) ? sigmaxpsdisk : sigmax2sdisk;

      MinvDt[0][2 * i] *= (rnew[i] / denom);
      MinvDt[1][2 * i] *= (rnew[i] / denom);
      MinvDt[2][2 * i] *= (rnew[i] / denom);
      MinvDt[3][2 * i] *= (rnew[i] / denom);

      assert(MinvDt[0][2 * i] == MinvDt[0][2 * i]);

      iMinvDt[0][2 * i] = (1 << settings.fitrinvbitshift()) * MinvDt[0][2 * i] * settings.kphi() / settings.krinvpars();
      iMinvDt[1][2 * i] = (1 << settings.fitphi0bitshift()) * MinvDt[1][2 * i] * settings.kphi() / settings.kphi0pars();
      iMinvDt[2][2 * i] = (1 << settings.fittbitshift()) * MinvDt[2][2 * i] * settings.kphi() / settings.ktpars();
      iMinvDt[3][2 * i] = (1 << settings.fitz0bitshift()) * MinvDt[3][2 * i] * settings.kphi() / settings.kz();

      denom = (std::abs(alpha[i - nlayers]) < 1e-10) ? sigmazpsdisk : sigmaz2sdisk;

      MinvDt[0][2 * i + 1] /= denom;
      MinvDt[1][2 * i + 1] /= denom;
      MinvDt[2][2 * i + 1] /= denom;
      MinvDt[3][2 * i + 1] /= denom;

      iMinvDt[0][2 * i + 1] =
          (1 << settings.fitrinvbitshift()) * MinvDt[0][2 * i + 1] * settings.krprojshiftdisk() / settings.krinvpars();
      iMinvDt[1][2 * i + 1] =
          (1 << settings.fitphi0bitshift()) * MinvDt[1][2 * i + 1] * settings.krprojshiftdisk() / settings.kphi0pars();
      iMinvDt[2][2 * i + 1] =
          (1 << settings.fittbitshift()) * MinvDt[2][2 * i + 1] * settings.krprojshiftdisk() / settings.ktpars();
      iMinvDt[3][2 * i + 1] =
          (1 << settings.fitz0bitshift()) * MinvDt[3][2 * i + 1] * settings.krprojshiftdisk() / settings.kz();
    }
  }
}

double TrackDerTable::tpar(Settings const& settings, int diskmask, int layermask) {
  if (diskmask == 0)
    return 0.0;

  double tmax = 1000.0;
  double tmin = 0.0;

  for (int d = 1; d <= (int)N_DISK; d++) {
    if (diskmask & (1 << (2 * (5 - d) + 1))) {  //PS hit
      double dmax = settings.zmean(d - 1) / 22.0;
      if (dmax > sinh(2.4))
        dmax = sinh(2.4);
      double dmin = settings.zmean(d - 1) / 65.0;
      if (dmax < tmax)
        tmax = dmax;
      if (dmin > tmin)
        tmin = dmin;
    }

    if (diskmask & (1 << (2 * (5 - d)))) {  //2S hit
      double dmax = settings.zmean(d - 1) / 65.0;
      double dmin = settings.zmean(d - 1) / 105.0;
      if (dmax < tmax)
        tmax = dmax;
      if (dmin > tmin)
        tmin = dmin;
    }
  }

  for (int l = 1; l <= (int)N_LAYER; l++) {
    if (layermask & (1 << (6 - l))) {
      double lmax = settings.zlength() / settings.rmean(l - 1);
      if (lmax < tmax)
        tmax = lmax;
    }
  }

  return 0.5 * (tmax + tmin) * 1.07;
}
