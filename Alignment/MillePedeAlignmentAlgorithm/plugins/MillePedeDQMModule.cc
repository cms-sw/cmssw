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
  maxMoveCut_ (mpReaderConfig_.getParameter<double>("maxMoveCut")),
  maxErrorCut_ (mpReaderConfig_.getParameter<double>("maxErrorCut"))  
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
::bookHistograms(DQMStore::IBooker& booker)
{
  edm::LogInfo("MillePedeDQMModule") << "Booking histograms";

  booker.cd();
  booker.setCurrentFolder("AlCaReco/SiPixelAli/");

  h_xPos = booker.book1D("Xpos",   "#Delta X;;#mu m", 10, 0, 10.);
  h_xRot = booker.book1D("Xrot",   "#Delta #theta_{X};;#mu rad", 10, 0, 10.);
  h_yPos = booker.book1D("Ypos",   "#Delta Y;;#mu m", 10, 0., 10.);
  h_yRot = booker.book1D("Yrot",   "#Delta #theta_{Y};;#mu rad", 10, 0, 10.);
  h_zPos = booker.book1D("Zpos",   "#Delta Z;;#mu m", 10, 0., 10.);
  h_zRot = booker.book1D("Zrot",   "#Delta #theta_{Z};;#mu rad", 10, 0, 10.);

  booker.cd();
}


void MillePedeDQMModule
::dqmEndJob(DQMStore::IBooker & booker, DQMStore::IGetter &)  
{

  bookHistograms(booker);
  mpReader.read();
  fillExpertHistos();
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeDQMModule
::fillExpertHistos()
{

  fillExpertHisto(h_xPos,  Xcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getXobs(),  mpReader.getXobsErr());
  fillExpertHisto(h_xRot, tXcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getTXobs(), mpReader.getTXobsErr());

  fillExpertHisto(h_yPos,  Ycut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getYobs(),  mpReader.getYobsErr());
  fillExpertHisto(h_yRot, tYcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getTYobs(), mpReader.getTYobsErr());

  fillExpertHisto(h_zPos,  Zcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getZobs(),  mpReader.getZobsErr());
  fillExpertHisto(h_zRot, tZcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader.getTZobs(), mpReader.getTZobsErr());

}

void MillePedeDQMModule
::fillExpertHisto(MonitorElement* histo, const double cut, const double sigCut, const double maxMoveCut, const double maxErrorCut,
                  std::array<double, 6> obs, std::array<double, 6> obsErr)
{
  TH1F* histo_0 = histo->getTH1F();
  
  histo_0->SetMinimum(-(maxMoveCut_));
  histo_0->SetMaximum(  maxMoveCut_);

  for (size_t i = 0; i < obs.size(); ++i) {
    histo_0->SetBinContent(i+1, obs[i]);
    histo_0->SetBinError(i+1, obsErr[i]);
  }
  histo_0->SetBinContent(8,cut);
  histo_0->SetBinContent(9,sigCut);
  histo_0->SetBinContent(10,maxMoveCut);
  histo_0->SetBinContent(11,maxErrorCut);  

}
