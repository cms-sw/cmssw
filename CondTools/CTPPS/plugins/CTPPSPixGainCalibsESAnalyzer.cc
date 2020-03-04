#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelGainCalibrationsRcd.h"
#include "TH2D.h"
#include "TFile.h"

class CTPPSPixGainCalibsESAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSPixGainCalibsESAnalyzer(edm::ParameterSet const& p)
      : m_outfilename(p.getUntrackedParameter<std::string>("outputrootfile", "output.root")) {
    //    std::cout<<"CTPPSPixGainCalibsESAnalyzer"<<std::endl;
    setReadablePlaneNames();
  }
  explicit CTPPSPixGainCalibsESAnalyzer(int i) {
    //std::cout<<"CTPPSPixGainCalibsESAnalyzer "<<i<<std::endl;
    setReadablePlaneNames();
  }
  ~CTPPSPixGainCalibsESAnalyzer() override {
    //std::cout<<"~CTPPSPixGainCalibsESAnalyzer "<<std::endl;
  }
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void setReadablePlaneNames();

private:
  std::map<uint32_t, std::string> detId_readable;
  std::string m_outfilename;
};

void CTPPSPixGainCalibsESAnalyzer::setReadablePlaneNames() {
  detId_readable[2014838784] = "Arm_0_Sec_45_St_0_Pot_3_Plane_0";
  detId_readable[2014904320] = "Arm_0_Sec_45_St_0_Pot_3_Plane_1";
  detId_readable[2014969856] = "Arm_0_Sec_45_St_0_Pot_3_Plane_2";
  detId_readable[2015035392] = "Arm_0_Sec_45_St_0_Pot_3_Plane_3";
  detId_readable[2015100928] = "Arm_0_Sec_45_St_0_Pot_3_Plane_4";
  detId_readable[2015166464] = "Arm_0_Sec_45_St_0_Pot_3_Plane_5";
  detId_readable[2023227392] = "Arm_0_Sec_45_St_2_Pot_3_Plane_0";
  detId_readable[2023292928] = "Arm_0_Sec_45_St_2_Pot_3_Plane_1";
  detId_readable[2023358464] = "Arm_0_Sec_45_St_2_Pot_3_Plane_2";
  detId_readable[2023424000] = "Arm_0_Sec_45_St_2_Pot_3_Plane_3";
  detId_readable[2023489536] = "Arm_0_Sec_45_St_2_Pot_3_Plane_4";
  detId_readable[2023555072] = "Arm_0_Sec_45_St_2_Pot_3_Plane_5";
  detId_readable[2031616000] = "Arm_1_Sec_56_St_0_Pot_3_Plane_0";
  detId_readable[2031681536] = "Arm_1_Sec_56_St_0_Pot_3_Plane_1";
  detId_readable[2031747072] = "Arm_1_Sec_56_St_0_Pot_3_Plane_2";
  detId_readable[2031812608] = "Arm_1_Sec_56_St_0_Pot_3_Plane_3";
  detId_readable[2031878144] = "Arm_1_Sec_56_St_0_Pot_3_Plane_4";
  detId_readable[2031943680] = "Arm_1_Sec_56_St_0_Pot_3_Plane_5";
  detId_readable[2040004608] = "Arm_1_Sec_56_St_2_Pot_3_Plane_0";
  detId_readable[2040070144] = "Arm_1_Sec_56_St_2_Pot_3_Plane_1";
  detId_readable[2040135680] = "Arm_1_Sec_56_St_2_Pot_3_Plane_2";
  detId_readable[2040201216] = "Arm_1_Sec_56_St_2_Pot_3_Plane_3";
  detId_readable[2040266752] = "Arm_1_Sec_56_St_2_Pot_3_Plane_4";
  detId_readable[2040332288] = "Arm_1_Sec_56_St_2_Pot_3_Plane_5";
}

void CTPPSPixGainCalibsESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
  edm::LogPrint("CTPPSPixGainCalibsReader") << "###CTPPSPixGainCalibsESAnalyzer::analyze";
  edm::LogPrint("CTPPSPixGainCalibsReader") << " I AM IN RUN NUMBER " << e.id().run();
  edm::LogPrint("CTPPSPixGainCalibsReader") << " ---EVENT NUMBER " << e.id().event();
  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("CTPPSPixelGainCalibrationsRcd"));
  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogPrint("CTPPSPixGainCalibsReader") << "Record \"CTPPSPixelGainCalibrationsRcd"
                                              << "\" does not exist ";
  }
  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<CTPPSPixelGainCalibrations> calhandle;
  edm::LogPrint("CTPPSPixGainCalibsReader") << "got eshandle";
  context.get<CTPPSPixelGainCalibrationsRcd>().get(calhandle);
  edm::LogPrint("CTPPSPixGainCalibsReader") << "got context";
  const CTPPSPixelGainCalibrations* pPixelGainCalibrations = calhandle.product();
  edm::LogPrint("CTPPSPixGainCalibsReader") << "got CTPPSPixelGainCalibrations* ";
  edm::LogPrint("CTPPSPixGainCalibsReader") << "print  pointer address : ";
  edm::LogPrint("CTPPSPixGainCalibsReader") << pPixelGainCalibrations;

  TFile myfile(m_outfilename.c_str(), "RECREATE");
  myfile.cd();

  // the pPixelGainCalibrations object contains the map of detIds  to pixel gains and pedestals for current run
  // we get the map just to loop over the contents, but from here on it should be  as the code (reconstruction etc) needs.
  // Probably best to check that the key (detid) is in the list before calling the data

  edm::LogPrint("CTPPSPixGainCalibsReader") << "Size " << pPixelGainCalibrations->size();
  const CTPPSPixelGainCalibrations::CalibMap& mymap = pPixelGainCalibrations->getCalibMap();  //just to get the keys?

  for (CTPPSPixelGainCalibrations::CalibMap::const_iterator it = mymap.begin(); it != mymap.end(); ++it) {
    uint32_t detId = it->first;

    edm::LogPrint("CTPPSPixGainCalibsReader")
        << "Address of  detId = " << (&detId) << " and of it = " << (&it) << " and of it->first = " << (&(it->first));

    edm::LogPrint("CTPPSPixGainCalibsReader") << "Content  of pPixelGainCalibrations for key: detId= " << detId;
    CTPPSPixelGainCalibration mycalibs0 = pPixelGainCalibrations->getGainCalibration(detId);
    const CTPPSPixelGainCalibration& mycalibs = it->second;

    edm::LogPrint("CTPPSPixGainCalibsReader")
        << "Address of  mycalibs0 = " << (&mycalibs0) << " and of mycalibs = " << (&mycalibs) << " and of it->second "
        << (&(it->second));

    std::string namep("pedsFromDB_" + detId_readable[detId]);
    std::string nameg("gainsFromDB_" + detId_readable[detId]);
    std::string tlp("Pedestals for " + detId_readable[detId] + "; column; row");
    std::string tlg("Gains for " + detId_readable[detId] + "; column; row");
    TH2D mypeds(namep.c_str(), tlp.c_str(), 156, 0., 156., 160, 0., 160.);
    TH2D mygains(nameg.c_str(), tlg.c_str(), 156, 0., 156., 160, 0., 160.);

    int ncols = mycalibs.getNCols();
    int npix = mycalibs.getIEnd();
    int nrows = mycalibs.getNRows();  //should be == 160
    edm::LogPrint("CTPPSPixGainCalibsReader") << "Here ncols = " << ncols << " nrows =" << nrows << " npix=" << npix;
    for (int jrow = 0; jrow < nrows; ++jrow)
      for (int icol = 0; icol < ncols; ++icol) {
        if (mycalibs.isDead(icol + jrow * ncols)) {
          edm::LogPrint("CTPPSPixGainCalibsReader") << "Dead Pixel icol =" << icol << " jrow =" << jrow;
          continue;
        }
        if (mycalibs.isNoisy(icol + jrow * ncols)) {
          edm::LogPrint("CTPPSPixGainCalibsReader") << "Noisy Pixel icol =" << icol << " jrow =" << jrow;
          continue;
        }
        mygains.Fill(icol, jrow, mycalibs.getGain(icol, jrow));
        mypeds.Fill(icol, jrow, mycalibs.getPed(icol, jrow));
      }

    mypeds.Write();
    mygains.Write();
  }
  myfile.Write();
  myfile.Close();
}
DEFINE_FWK_MODULE(CTPPSPixGainCalibsESAnalyzer);
