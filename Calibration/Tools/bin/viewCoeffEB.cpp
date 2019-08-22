#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"

#include "TH2.h"
#include "TH1.h"
#include "TFile.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"

int main(int argc, char* argv[]) {
  std::string fileName = argv[1];
  std::cerr << "parsing coeff file: " << fileName << std::endl;

  CaloMiscalibMapEcal map;
  map.prefillMap();
  MiscalibReaderFromXMLEcalBarrel barrelreader(map);
  if (!fileName.empty())
    barrelreader.parseXMLMiscalibFile(fileName);
  EcalIntercalibConstants* constants = new EcalIntercalibConstants(map.get());
  const EcalIntercalibConstantMap& imap = constants->getMap();

  TH1F coeffDistr("coeffDistrEB", "coeffDistrEB", 500, 0, 2);
  TH2F coeffMap("coeffMapEB", "coeffMapEB", 171, -85, 86, 360, 1, 361);
  coeffMap.SetStats(false);

  // ECAL barrel
  for (int ieta = -85; ieta <= 85; ++ieta)
    for (int iphi = 1; iphi <= 360; ++iphi) {
      if (!EBDetId::validDetId(ieta, iphi))
        continue;
      EBDetId det = EBDetId(ieta, iphi, EBDetId::ETAPHIMODE);
      double coeff = (*(imap.find(det.rawId())));
      coeffDistr.Fill(coeff);
      coeffMap.Fill(ieta, iphi, coeff);
    }  // ECAL barrel

  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  TCanvas c1;
  c1.SetGrid();

  coeffMap.GetZaxis()->SetRangeUser(0, 2);
  coeffMap.GetXaxis()->SetTitle("eta");
  coeffMap.GetYaxis()->SetTitle("phi");
  coeffMap.Draw("COLZ");
  c1.Print("coeffMapEB.gif", "gif");
  c1.SetLogy();
  coeffDistr.GetXaxis()->SetTitle("calib coeff");
  coeffDistr.SetFillColor(8);
  coeffDistr.Draw();
  c1.Print("coeffDistrEB.gif", "gif");

  TFile out("coeffEB.root", "recreate");
  coeffDistr.Write();
  coeffMap.Write();
  out.Close();
}
