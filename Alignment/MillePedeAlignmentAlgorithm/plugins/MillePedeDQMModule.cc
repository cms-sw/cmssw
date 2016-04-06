/**
 * @package   Alignment/MillePedeAlignmentAlgorithm
 * @file      MillePedeDQMModule.cc
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      Feb 19, 2016
 */


/*** header-file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeDQMModule.h"

/*** ROOT objects ***/
#include "TCanvas.h"
#include "TH1F.h"



MillePedeDQMModule
::MillePedeDQMModule(const edm::ParameterSet& config) :
  mpReaderConfig_(
    config.getParameter<edm::ParameterSet>("MillePedeFileReader")
  ),
  mpReader(mpReaderConfig_),

  sigCut_     (mpReaderConfig_.getParameter<double>("sigCut")),
  Xcut_       (mpReaderConfig_.getParameter<double>("Xcut")),
  tXcut_      (mpReaderConfig_.getParameter<double>("tXcut")),
  Ycut_       (mpReaderConfig_.getParameter<double>("Ycut")),
  tYcut_      (mpReaderConfig_.getParameter<double>("tYcut")),
  Zcut_       (mpReaderConfig_.getParameter<double>("Zcut")),
  tZcut_      (mpReaderConfig_.getParameter<double>("tZcut")),
  maxMoveCut_ (mpReaderConfig_.getParameter<double>("maxMoveCut"))
{
}

MillePedeDQMModule
::~MillePedeDQMModule()
{
}

//=============================================================================
//===   INTERFACE IMPLEMENTATION                                            ===
//=============================================================================

void MillePedeDQMModule
::bookHistograms(DQMStore::IBooker& booker,
                 edm::Run const& /* run */,
                 edm::EventSetup const& /* setup */)
{
  edm::LogInfo("MillePedeDQMModule") << "Booking histograms";

  booker.cd();
  booker.setCurrentFolder("AlCaReco/SiPixelAli/");

  h_xPos[0] = booker.book1D("Xpos",   "#Delta X;;#mu m", 6, 0, 6.);
  h_xPos[1] = booker.book1D("Xpos_1", "Xpos_1",          6, 0, 6.);
  h_xPos[2] = booker.book1D("Xpos_2", "Xpos_2",          6, 0, 6.);
  h_xPos[3] = booker.book1D("Xpos_3", "Xpos_3",          6, 0, 6.);

  h_xRot[0] = booker.book1D("Xrot",   "#Delta #theta_{X};;#mu rad", 6, 0, 6.);
  h_xRot[1] = booker.book1D("Xrot_1", "Xrot_1",                     6, 0, 6.);
  h_xRot[2] = booker.book1D("Xrot_2", "Xrot_2",                     6, 0, 6.);
  h_xRot[3] = booker.book1D("Xrot_3", "Xrot_3",                     6, 0, 6.);

  h_yPos[0] = booker.book1D("Ypos",   "#Delta Y;;#mu m", 6, 0., 6.);
  h_yPos[1] = booker.book1D("Ypos_1", "Ypos_1",          6, 0., 6.);
  h_yPos[2] = booker.book1D("Ypos_2", "Ypos_2",          6, 0., 6.);
  h_yPos[3] = booker.book1D("Ypos_3", "Ypos_3",          6, 0., 6.);

  h_yRot[0] = booker.book1D("Yrot",   "#Delta #theta_{Y};;#mu rad", 6, 0, 6.);
  h_yRot[1] = booker.book1D("Yrot_1", "Yrot_1",                     6, 0, 6.);
  h_yRot[2] = booker.book1D("Yrot_2", "Yrot_2",                     6, 0, 6.);
  h_yRot[3] = booker.book1D("Yrot_3", "Yrot_3",                     6, 0, 6.);

  h_zPos[0] = booker.book1D("Zpos",   "#Delta Z;;#mu m", 6, 0., 6.);
  h_zPos[1] = booker.book1D("Zpos_1", "Zpos_1",          6, 0., 6.);
  h_zPos[2] = booker.book1D("Zpos_2", "Zpos_2",          6, 0., 6.);
  h_zPos[3] = booker.book1D("Zpos_3", "Zpos_3",          6, 0., 6.);

  h_zRot[0] = booker.book1D("Zrot",   "#Delta #theta_{Z};;#mu rad", 6, 0, 6.);
  h_zRot[1] = booker.book1D("Zrot_1", "Zrot_1",                     6, 0, 6.);
  h_zRot[2] = booker.book1D("Zrot_2", "Zrot_2",                     6, 0, 6.);
  h_zRot[3] = booker.book1D("Zrot_3", "Zrot_3",                     6, 0, 6.);

  booker.cd();
}

void MillePedeDQMModule
::endRun(edm::Run const& /* run */, edm::EventSetup const& /* setup */)
{
  mpReader.read();
  fillExpertHistos();
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeDQMModule
::fillExpertHistos()
{
  fillExpertHisto(h_xPos,  Xcut_, mpReader.getXobs(),  mpReader.getXobsErr());
  fillExpertHisto(h_xRot, tXcut_, mpReader.getTXobs(), mpReader.getTXobsErr());

  fillExpertHisto(h_yPos,  Ycut_, mpReader.getYobs(),  mpReader.getYobsErr());
  fillExpertHisto(h_yRot, tYcut_, mpReader.getTYobs(), mpReader.getTYobsErr());

  fillExpertHisto(h_zPos,  Zcut_, mpReader.getZobs(),  mpReader.getZobsErr());
  fillExpertHisto(h_zRot, tZcut_, mpReader.getTZobs(), mpReader.getTZobsErr());

}

void MillePedeDQMModule
::fillExpertHisto(MonitorElement* histos[], const double cut,
                  std::array<double, 6> obs, std::array<double, 6> obsErr)
{
  TH1F* histo_0 = histos[0]->getTH1F();
  TH1F* histo_1 = histos[1]->getTH1F();
  TH1F* histo_2 = histos[2]->getTH1F();
  TH1F* histo_3 = histos[3]->getTH1F();

  histo_0->SetMinimum(-(maxMoveCut_));
  histo_0->SetMaximum(  maxMoveCut_);

  for (size_t i = 0; i < obs.size(); ++i) {
    histo_0->SetBinContent(i+1, obs[i]);
    histo_1->SetBinContent(i+1, obs[i]);
    histo_2->SetBinContent(i+1,  cut);
    histo_3->SetBinContent(i+1, -cut);

    histo_0->SetBinError(i+1, obsErr[i]);
    histo_1->SetBinError(i+1, (histo_1->GetBinContent(i+1) / sigCut_));
  }
}
