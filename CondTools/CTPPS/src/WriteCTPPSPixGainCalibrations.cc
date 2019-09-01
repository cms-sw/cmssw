#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//CTPPS Gain Calibration Conditions Object
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibration.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"
//CTPPS tracker DetId
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "TFile.h"
#include "TH2F.h"
#include <iostream>
#include <vector>
#include <string>

//
// class declaration
//

class WriteCTPPSPixGainCalibrations : public edm::one::EDAnalyzer<> {
public:
  explicit WriteCTPPSPixGainCalibrations(const edm::ParameterSet&);
  ~WriteCTPPSPixGainCalibrations() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void getHistos();
  void fillDB();
  void getGainsPedsFromHistos(uint32_t detid,
                              int rocId,
                              int index,
                              std::vector<double>& peds,
                              std::vector<double>& gains,
                              std::map<int, int>& myindxmap,
                              int nrocs);
  void setDummyFullPlane(std::vector<float>& peds, std::vector<float>& gains, int npixplane);
  // ----------Member data ---------------------------
  std::string m_record;
  std::string m_inputHistosFileName;
  bool m_usedummy;
  int npfitmin;
  double gainlow, gainhigh;
  TFile* m_inputRootFile;
  std::map<uint32_t, std::vector<std::string> > detidHistoNameMap;
  //  std::map<uint32_t, std::vector<std::string> > detidSlopeNameMap;
  //  std::map<uint32_t, std::vector<std::string> > detidInterceptNameMap;
  //  std::map<uint32_t, std::vector<std::string> > detidChi2NameMap;
  //  std::map<uint32_t, std::vector<std::string> > detidNpfitNameMap;
  std::map<uint32_t, std::vector<int> > detidROCsPresent;
};

//
// constructors and destructor
//
WriteCTPPSPixGainCalibrations::WriteCTPPSPixGainCalibrations(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getUntrackedParameter<std::string>("record", "CTPPSPixelGainCalibrationsRcd")),
      m_inputHistosFileName(iConfig.getUntrackedParameter<std::string>("inputrootfile", "inputfile.root")),
      m_usedummy(iConfig.getUntrackedParameter<bool>("useDummyValues", true)),
      npfitmin(iConfig.getUntrackedParameter<int>("minimumNpfit", 3)),
      gainlow(iConfig.getUntrackedParameter<double>("gainLowLimit", 0.0)),
      gainhigh(iConfig.getUntrackedParameter<double>("gainHighLimit", 100.0)) {}

WriteCTPPSPixGainCalibrations::~WriteCTPPSPixGainCalibrations() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void WriteCTPPSPixGainCalibrations::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //   using namespace edm;
}

// ------------ method called once each job just before starting event loop  ------------
void WriteCTPPSPixGainCalibrations::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void WriteCTPPSPixGainCalibrations::endJob() {
  getHistos();
  fillDB();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void WriteCTPPSPixGainCalibrations::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void WriteCTPPSPixGainCalibrations::getHistos() {
  //  std::cout <<"Parsing file " <<m_inputHistosFileName << std::endl;
  m_inputRootFile = new TFile(m_inputHistosFileName.c_str());
  m_inputRootFile->cd();

  int sector[2] = {45, 56};  // arm 0 is sector 45 and arm 1 is sector 56
  int nsec = 2;
  int station[2] = {0, 2};  // for each arm
  int nst = 2;
  //int pot[6]={3}; // index of the pot within the 6 pot configuration (vertical top or bottom and horizontal)
  int npt = 6;

  for (int i = 0; i < nsec; i++)
    for (int st = 0; st < nst; st++)
      for (int pot = 0; pot < npt; pot++) {
        int arm = i;

        //Check which pots present
        char temppathrp[100];
        sprintf(temppathrp, "CTPPS/CTPPS_SEC%d/CTPPS_SEC%d_RP%d%d%d", sector[i], sector[i], arm, station[st], pot);
        if (!m_inputRootFile->Get(temppathrp))
          continue;

        for (int plane = 0; plane < 6; plane++) {
          //Check which planes present
          char temppathplane[100];
          sprintf(temppathplane,
                  "CTPPS/CTPPS_SEC%d/CTPPS_SEC%d_RP%d%d%d/CTPPS_SEC%d_RP%d%d%d_PLN%d",
                  sector[i],
                  sector[i],
                  arm,
                  station[st],
                  pot,
                  sector[i],
                  arm,
                  station[st],
                  pot,
                  plane);

          // do not skip the missing plane, instead put dummy values
          //	  if(!m_inputRootFile->Get(temppathplane)) continue;

          CTPPSPixelDetId mytempid(arm, station[st], pot, plane);
          std::vector<std::string> histnamevec;
          std::vector<int> listrocs;
          for (int roc = 0; roc < 6; roc++) {
            char temppathhistos[200];

            sprintf(
                temppathhistos,
                "CTPPS/CTPPS_SEC%d/CTPPS_SEC%d_RP%d%d%d/CTPPS_SEC%d_RP%d%d%d_PLN%d/CTPPS_SEC%d_RP%d%d%d_PLN%d_ROC%d",
                sector[i],
                sector[i],
                arm,
                station[st],
                pot,
                sector[i],
                arm,
                station[st],
                pot,
                plane,
                sector[i],
                arm,
                station[st],
                pot,
                plane,
                roc);

            std::string pathhistos(temppathhistos);
            std::string pathslope = pathhistos + "_Slope2D";
            std::string pathintercept = pathhistos + "_Intercept2D";
            if (m_inputRootFile->Get(pathslope.c_str()) && m_inputRootFile->Get(pathintercept.c_str())) {
              histnamevec.push_back(pathhistos);
              listrocs.push_back(roc);
            }
          }
          detidHistoNameMap[mytempid.rawId()] = histnamevec;
          detidROCsPresent[mytempid.rawId()] = listrocs;
          edm::LogInfo("CTPPSPixGainsCalibrationWriter")
              << "Raw DetId = " << mytempid.rawId() << " Arm = " << arm << " Sector = " << sector[arm]
              << " Station = " << station[st] << " Pot = " << pot << " Plane = " << plane;
        }
      }
}

void WriteCTPPSPixGainCalibrations::fillDB() {
  CTPPSPixelGainCalibrations* gainCalibsTest = new CTPPSPixelGainCalibrations();
  CTPPSPixelGainCalibrations* gainCalibsTest1 = new CTPPSPixelGainCalibrations();

  //  std::cout<<"Here! "<<std::endl;

  for (std::map<uint32_t, std::vector<int> >::iterator it = detidROCsPresent.begin(); it != detidROCsPresent.end();
       ++it) {
    uint32_t tempdetid = it->first;
    std::vector<int> rocs = it->second;
    unsigned int nrocs = rocs.size();
    std::map<int, int> mapIPixIndx;

    std::vector<double> gainsFromHistos;
    std::vector<double> pedsFromHistos;

    CTPPSPixelGainCalibration tempPGCalib;

    for (unsigned int i = 0; i < nrocs; i++) {
      getGainsPedsFromHistos(tempdetid, i, rocs[i], pedsFromHistos, gainsFromHistos, mapIPixIndx, nrocs);
    }

    std::vector<float> orderedGains;
    std::vector<float> orderedPeds;
    for (unsigned int k = 0; k < nrocs * 52 * 80; k++) {
      int indx = mapIPixIndx[k];
      float tmpped = pedsFromHistos[indx];
      float tmpgain = gainsFromHistos[indx];
      orderedGains.push_back(tmpgain);
      orderedPeds.push_back(tmpped);
      tempPGCalib.putData(k, tmpped, tmpgain);
    }

    if (nrocs == 0) {
      edm::LogWarning("CTPPSPixGainsCalibrationWriter") << " plane with detID =" << tempdetid << " is empty";
      setDummyFullPlane(orderedPeds, orderedGains, 6 * 52 * 80);
    }

    gainCalibsTest->setGainCalibration(tempdetid, orderedPeds, orderedGains);
    //	 std::cout << "Here detid = "<<tempdetid <<std::endl;
    gainCalibsTest1->setGainCalibration(tempdetid, tempPGCalib);
    //	 std::cout << "Here again"<<std::endl;
  }
  //  std::cout<<" Here 3!"<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("CTPPSPixGainsCalibrationWriter") << "Db Service Unavailable";
    return;
  }
  mydbservice->writeOne(gainCalibsTest, mydbservice->currentTime(), m_record);
}

void WriteCTPPSPixGainCalibrations::setDummyFullPlane(std::vector<float>& peds,
                                                      std::vector<float>& gains,
                                                      int npixplane) {
  for (int i = 0; i < npixplane; ++i) {
    peds.push_back(20.);
    gains.push_back(0.5);
  }
}

void WriteCTPPSPixGainCalibrations::getGainsPedsFromHistos(uint32_t detid,
                                                           int ROCindex,
                                                           int rocId,
                                                           std::vector<double>& peds,
                                                           std::vector<double>& gains,
                                                           std::map<int, int>& mymap,
                                                           int nrocs) {
  CTPPSPixelIndices modulepixels(52 * nrocs / 2, 160);

  std::string tmpslopename = detidHistoNameMap[detid][ROCindex] + "_Slope2D";
  std::string tmpitcpname = detidHistoNameMap[detid][ROCindex] + "_Intercept2D";
  std::string tmpchi2name = detidHistoNameMap[detid][ROCindex] + "_Chisquare2D";
  std::string tmpnpfitsname = detidHistoNameMap[detid][ROCindex] + "_Npfits2D";
  TH2D* tempslope = (TH2D*)m_inputRootFile->Get(tmpslopename.c_str());
  TH2D* tempintrcpt = (TH2D*)m_inputRootFile->Get(tmpitcpname.c_str());
  // TH2D * tempchi2    = (TH2D*) m_inputRootFile->Get(tmpchi2name.c_str());
  TH2D* tempnpfit = (TH2D*)m_inputRootFile->Get(tmpnpfitsname.c_str());
  int ncols = tempslope->GetNbinsX();
  int nrows = tempslope->GetNbinsY();
  if (nrows != 80 || ncols != 52)
    edm::LogWarning("CTPPSPixGainsCalibrationWriter")
        << "Something wrong ncols = " << ncols << " and nrows = " << nrows;

  for (int jrow = 0; jrow < nrows;
       ++jrow)  // when scanning through the 2d histo make sure to avoid underflow bin i or j ==0
    for (int icol = 0; icol < ncols; ++icol) {
      double tmpslp = tempslope->GetBinContent(icol + 1, jrow + 1);
      double tmpgain = (tmpslp == 0.0) ? 0.0 : 1.0 / tmpslp;
      double tmpped = tempintrcpt->GetBinContent(icol + 1, jrow + 1);
      // check for noisy/badly calibrated pixels
      int tmpnpfit = tempnpfit->GetBinContent(icol + 1, jrow + 1);
      //double tmpchi2  = tempchi2 -> GetBinContent(icol+1,jrow+1);
      if (tmpnpfit < npfitmin || tmpgain < gainlow || tmpgain > gainhigh) {
        //	std::cout << " *** Badly calibrated pixel, NPoints = "<<tmpnpfit << " setting dummy values gain = 0.5 and  ped =20.0 ***" <<std::endl;
        //	std::cout << " **** bad Pixel column icol = "<<icol <<" and jrow = "<<jrow <<" Name= "<< tmpslopename <<std::endl;
        if (m_usedummy) {
          tmpgain = 1.0 / 2.0;
          tmpped = 20.0;
        }
        // else  leave as is and set noisy in mask?
      }

      gains.push_back(tmpgain);
      peds.push_back(tmpped);
      int modCol = -1;
      int modRow = -1;
      modulepixels.transformToModule(icol, jrow, rocId, modCol, modRow);
      int indx = gains.size() - 1;
      int pixIndx = modCol + modRow * (52 * nrocs / 2);
      mymap[pixIndx] = indx;
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WriteCTPPSPixGainCalibrations);
