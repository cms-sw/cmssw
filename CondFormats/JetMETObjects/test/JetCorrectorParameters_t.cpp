#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Utilities/interface/FileInPath.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>

#include "TBenchmark.h"

using std::cout;
using std::endl;
using std::flush;
using std::ofstream;
using std::setw;
using std::string;
using std::vector;

class testJetCorrectorParameters : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testJetCorrectorParameters);
  CPPUNIT_TEST(benchmarkBinIndex1D);
  CPPUNIT_TEST(benchmarkBinIndex3D);
  CPPUNIT_TEST(compareBinIndex1D);
  CPPUNIT_TEST(compareBinIndex3D);
  CPPUNIT_TEST_SUITE_END();

public:
  testJetCorrectorParameters() {}
  ~testJetCorrectorParameters() {}
  void setUp();
  void tearDown();
  void setupCorrector(bool is3D);
  void destroyCorrector();
  void generateFiles();

  void benchmarkBinIndex(bool is3D);
  void benchmarkBinIndex1D() { benchmarkBinIndex(false); }
  void benchmarkBinIndex3D() { benchmarkBinIndex(true); }
  void compareBinIndex1D();
  void compareBinIndex3D();

  inline void loadbar3(
      unsigned int x, unsigned int n, unsigned int w = 50, unsigned int freq = 100, string prefix = "") {
    if ((x != n) && (x % (n / freq) != 0))
      return;
    float ratio = x / (float)n;
    int c = ratio * w;

    cout << prefix << std::fixed << setw(8) << std::setprecision(0) << (ratio * 100) << "% [";
    for (int x = 0; x < c; x++)
      cout << "=";
    for (unsigned int x = c; x < w; x++)
      cout << " ";
    cout << "] (" << x << "/" << n << ")\r" << flush;
  }

private:
  string filename;
  TBenchmark* m_benchmark;
  JetCorrectorParameters* L1JetPar;
  vector<JetCorrectorParameters> vPar;
  FactorizedJetCorrector* jetCorrector;
  vector<float> fX;
  vector<float> veta;
  vector<float> vrho;
  vector<float> vpt;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testJetCorrectorParameters);

void testJetCorrectorParameters::setUp() {
  m_benchmark = new TBenchmark();
  veta = {-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139,
          -2.964, -2.853, -2.65,  -2.5,   -2.322, -2.172, -2.043, -1.93,  -1.83,  -1.74,  -1.653, -1.566,
          -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609, -0.522,
          -0.435, -0.348, -0.261, -0.174, -0.087, 0,      0.087,  0.174,  0.261,  0.348,  0.435,  0.522,
          0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,  1.305,  1.392,  1.479,  1.566,
          1.653,  1.74,   1.83,   1.93,   2.043,  2.172,  2.322,  2.5,    2.65,   2.853,  2.964,  3.139,
          3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,  4.889,  5.191};
  for (unsigned int irho = 1; irho <= 50; irho++) {
    vrho.push_back(irho);
  }
  vpt = {1,   2,   3,   4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,
         17,  20,  23,  27,   30,   35,   40,   45,   57,   72,   90,   120,  150,  200,  300,
         400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500};

  generateFiles();
}

void testJetCorrectorParameters::tearDown() {
  if (m_benchmark != nullptr)
    delete m_benchmark;
}

void testJetCorrectorParameters::setupCorrector(bool is3D) {
  string path = "CondFormats/JetMETObjects/data/";
  try {
    string to_try = path + ((is3D) ? "testJetCorrectorParameters_3D_L1FastJet_AK4PFchs.txt"
                                   : "testJetCorrectorParameters_1D_L1FastJet_AK4PFchs.txt");
    edm::FileInPath strFIP(to_try);
    filename = strFIP.fullPath();
  } catch (edm::Exception const& ex) {
    throw ex;
  }

  L1JetPar = new JetCorrectorParameters(filename);
  vPar.push_back(*L1JetPar);
  jetCorrector = new FactorizedJetCorrector(vPar);
}

void testJetCorrectorParameters::destroyCorrector() {
  delete L1JetPar;
  delete jetCorrector;
}

void testJetCorrectorParameters::generateFiles() {
  string path = std::getenv("CMSSW_BASE");
  path += "/src/CondFormats/JetMETObjects/data/";
  string name1D = "testJetCorrectorParameters_1D_L1FastJet_AK4PFchs.txt";
  string name3D = "testJetCorrectorParameters_3D_L1FastJet_AK4PFchs.txt";

  ofstream txtFile1D;
  txtFile1D.open((path + name1D).c_str());
  CPPUNIT_ASSERT(txtFile1D.is_open());
  txtFile1D << "{1 JetEta 3 Rho JetPt JetA 1 Correction L1FastJet}" << endl;
  for (unsigned int ieta = 0; ieta < veta.size() - 1; ieta++) {
    txtFile1D << std::setw(15) << veta[ieta] << std::setw(15) << veta[ieta + 1] << std::setw(15) << "6" << std::setw(15)
              << "0" << std::setw(15) << "200" << std::setw(15) << "1" << std::setw(15) << "6500" << std::setw(15)
              << "0" << std::setw(15) << "10" << std::setw(15) << endl;
  }
  txtFile1D.close();

  vector<float> ptbins = {4,  8,  10,  13,  16,  19,  22,  25,  30,  35,   40,  45,
                          60, 75, 100, 160, 230, 330, 440, 600, 820, 1100, 1400};
  ofstream txtFile3D;
  txtFile3D.open((path + name3D).c_str());
  CPPUNIT_ASSERT(txtFile3D.is_open());
  txtFile3D << "{3 JetEta Rho JetPt 1 JetPt 1 Correction L1FastJet}" << endl;
  for (unsigned int ieta = 0; ieta < veta.size() - 1; ieta++) {
    for (unsigned int irho = 0; irho < vrho.size() - 1; irho++) {
      for (unsigned int ipt = 0; ipt < ptbins.size() - 1; ipt++) {
        txtFile3D << std::setw(15) << veta[ieta] << std::setw(15) << veta[ieta + 1] << std::setw(15) << vrho[irho]
                  << std::setw(15);
        if (vrho[irho + 1] == 50)
          txtFile3D << "200" << std::setw(15);
        else
          txtFile3D << vrho[irho + 1] << std::setw(15);
        txtFile3D << ptbins[ipt] << std::setw(15);
        if (ipt + 1 == ptbins.size() - 1)
          txtFile3D << "6500" << std::setw(15);
        else
          txtFile3D << ptbins[ipt + 1] << std::setw(15);
        txtFile3D << "2" << std::setw(15) << ptbins[ipt] << std::setw(15) << ptbins[ipt + 1] << std::setw(15) << endl;
      }
    }
  }
  txtFile3D.close();
}

void testJetCorrectorParameters::benchmarkBinIndex(bool is3D) {
  float oldCPU = 0, newCPU = 0, oldReal = 0, newReal = 0;
  unsigned int ntests = (is3D) ? 100000 : 1000000;
  if (is3D)
    fX = {5.0, 50.0, 100.0};
  else
    fX = {5.0};
  setupCorrector(is3D);

  cout << endl << "testJetCorrectorParameters::benchmarkBinIndex NTests = " << ntests << endl;
  cout << "testJetCorrectorParameters::benchmarkBinIndex benchmarking binIndex with file " << filename << " ... "
       << flush;
  m_benchmark->Reset();
  m_benchmark->Start("event");
  for (unsigned int i = 0; i < ntests; i++) {
    L1JetPar->binIndex(fX);
  }
  m_benchmark->Stop("event");
  cout << "DONE" << endl
       << "testJetCorrectorParameters::benchmarkBinIndex" << endl
       << "\tCPU time = " << m_benchmark->GetCpuTime("event") / double(ntests) << " s" << endl
       << "\tReal time = " << m_benchmark->GetRealTime("event") / double(ntests) << " s" << endl;
  oldCPU = m_benchmark->GetCpuTime("event") / double(ntests);
  oldReal = m_benchmark->GetRealTime("event") / double(ntests);

  cout << "testJetCorrectorParameters::benchmarkBinIndex benchmarking binIndexN with file " << filename << " ... "
       << flush;
  m_benchmark->Reset();
  m_benchmark->Start("event");
  for (unsigned int i = 0; i < ntests; i++) {
    L1JetPar->binIndexN(fX);
  }
  m_benchmark->Stop("event");
  cout << "DONE" << endl
       << "testJetCorrectorParameters::benchmarkBinIndex" << endl
       << "\tCPU time = " << m_benchmark->GetCpuTime("event") / double(ntests) << " s" << endl
       << "\tReal time = " << m_benchmark->GetRealTime("event") / double(ntests) << " s" << endl;
  newCPU = m_benchmark->GetCpuTime("event") / double(ntests);
  newReal = m_benchmark->GetRealTime("event") / double(ntests);

  cout << "testJetCorrectorParameters::benchmarkBinIndex" << endl
       << "\tCPU speedup = " << oldCPU / newCPU << endl
       << "\tReal speedup = " << oldReal / newReal << endl;

  CPPUNIT_ASSERT(oldCPU / newCPU >= 1.0);
  //CPPUNIT_ASSERT(oldReal/newReal >= 1.0); //this might fail not due to longer L1JetPar->binIndexN(fX) execution
  if (oldReal / newReal >= 1.0)
    cout << "newReal value increased oldReal, which might be due to system load" << endl;

  destroyCorrector();
}

void testJetCorrectorParameters::compareBinIndex1D() {
  fX = {0.0};
  float eta = -9999;
  int oldBin = -1, newBin = -1;
  setupCorrector(false);

  cout << endl << "testJetCorrectorParameters::compareBinIndex1D" << endl;
  for (unsigned int ieta = 0; ieta < veta.size() - 1; ieta++) {
    loadbar3(ieta + 1, veta.size() - 1, 50, 10, "\tProgress:");
    eta = (veta[ieta] + veta[ieta + 1]) / 2.0;
    fX = {eta};
    oldBin = L1JetPar->binIndex(fX);
    newBin = L1JetPar->binIndexN(fX);
    if ((oldBin < 0 && newBin >= 0) || (oldBin >= 0 && newBin < 0)) {
      cout << "ERROR::testJetCorrectorParameters::compareBinIndex1D Unable to find the right bin for (eta)=(" << eta
           << ")" << endl
           << "\t(oldBin,newBin)=(" << oldBin << "," << newBin << ")" << endl;
      CPPUNIT_ASSERT(oldBin >= 0 && newBin >= 0);
    } else if (oldBin != newBin) {
      cout << "ERROR::testJetCorrectorParameters::compareBinIndex1D oldBin!=newBin (" << oldBin << "!=" << newBin
           << ") for (eta)=(" << eta << ")" << endl;
      CPPUNIT_ASSERT(oldBin == newBin);
    }

    jetCorrector->setJetEta(eta);
    jetCorrector->setRho(50.0);
    jetCorrector->setJetA(0.5);
    jetCorrector->setJetPt(100.0);
    CPPUNIT_ASSERT(jetCorrector->getCorrection() >= 0);
  }
  destroyCorrector();
  cout << endl
       << "testJetCorrectorParameters::compareBinIndex1D All bins match between the linear and non-linear search "
          "algorithms for 1D files."
       << endl;
}

void testJetCorrectorParameters::compareBinIndex3D() {
  fX = {0.0, 0.0, 0.0};
  float eta = -9999, rho = -9999, pt = -9999;
  int oldBin = -1, newBin = -1;
  setupCorrector(true);

  cout << endl << "testJetCorrectorParameters::compareBinIndex3D" << endl;
  for (unsigned int ieta = 0; ieta < veta.size() - 1; ieta++) {
    for (unsigned int irho = 0; irho < vrho.size() - 1; irho++) {
      for (unsigned int ipt = 0; ipt < vpt.size() - 1; ipt++) {
        loadbar3(ieta * ((vrho.size() - 1) * (vpt.size() - 1)) + irho * (vpt.size() - 1) + ipt + 1,
                 (veta.size() - 1) * (vrho.size() - 1) * (vpt.size() - 1),
                 50,
                 100,
                 "\tProgress");
        eta = (veta[ieta] + veta[ieta + 1]) / 2.0;
        rho = (vrho[irho] + vrho[irho + 1]) / 2.0;
        pt = (vpt[ipt] + vpt[ipt + 1]) / 2.0;
        fX = {eta, rho, pt};
        oldBin = L1JetPar->binIndex(fX);
        newBin = L1JetPar->binIndexN(fX);
        if ((oldBin < 0 && newBin >= 0) || (oldBin >= 0 && newBin < 0)) {
          cout << "ERROR::testJetCorrectorParameters::compareBinIndex3D Unable to find the right bin for (eta,rho,pt)=("
               << eta << "," << rho << "," << pt << ")" << endl
               << "\t(oldBin,newBin)=(" << oldBin << "," << newBin << ")" << endl;
          CPPUNIT_ASSERT(oldBin >= 0 && newBin >= 0);
        } else if (oldBin != newBin) {
          cout << "ERROR::testJetCorrectorParameters::compareBinIndex3D oldBin!=newBin (" << oldBin << "!=" << newBin
               << ") for (eta,rho,pt)=(" << eta << "," << rho << "," << pt << ")" << endl;
          CPPUNIT_ASSERT(oldBin == newBin);
        }

        jetCorrector->setJetEta(eta);
        jetCorrector->setRho(rho);
        jetCorrector->setJetA(0.5);
        jetCorrector->setJetPt(pt);
        CPPUNIT_ASSERT(jetCorrector->getCorrection() >= 0);
      }
    }
  }
  destroyCorrector();
  cout << endl
       << "testJetCorrectorParameters::compareBinIndex3D All bins match between the linear and non-linear search "
          "algorithms for 3D files."
       << endl;
}
