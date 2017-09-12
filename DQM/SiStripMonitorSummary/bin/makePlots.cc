#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <sstream>

#include "TFile.h"
#include "TH1F.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TSystem.h"

void setCanvasStyle(TCanvas* c, const bool logScale);
void setHistoStyle(TH1* h);
void setHistoStackStyle(TH1* h, const unsigned int lineColor);
void setLegendStyle(TLegend* l, const unsigned int nColumns);
void setPaveTextStyle(TPaveText* t, const bool isHorizontal=true);
void fillNormFactorMaps();
double findNormFactor(const std::string currentPlotType, const std::string currentPart, const bool stackOption);
void makePlots(std::string inputFileName, std::string outputFileName);

// Map with <name of tracker part, count of channels in the part>
// It is static so it can be read by the functions fillNormFactorMaps() and findNormFactor(...)
static std::map<std::string, unsigned int> modulesStackNormFactors;
static std::map<std::string, unsigned int> modulesNoStackNormFactors;
static std::map<std::string, unsigned int> fibersStackNormFactors;
static std::map<std::string, unsigned int> fibersNoStackNormFactors;
static std::map<std::string, unsigned int> APVsStackNormFactors;
static std::map<std::string, unsigned int> APVsNoStackNormFactors;
static std::map<std::string, unsigned int> stripsStackNormFactors;
static std::map<std::string, unsigned int> stripsNoStackNormFactors;

int main(int argc , char *argv[]) {

  if(argc==3) {
    char* inputFileName = argv[1];
    char* outputFileName = argv[2];

    std::cout << "ready to make plots from " << inputFileName << " to " << outputFileName << std::endl;

    
    int returncode = 0;
    makePlots(inputFileName,outputFileName);

    return  returncode;

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }

  return -9;

}


void makePlots(std::string inputFileName, std::string outputFileName)
{
  //
  
  
  // Open input and output file
  TFile* inputFile = new TFile(inputFileName.c_str(),"READ");
  TFile* outputFile = new TFile(outputFileName.c_str(), "RECREATE");
  
  std::ostringstream oss;
  std::string histoName;
  std::vector< std::string > plotType;
  plotType.push_back("BadModules"); plotType.push_back("BadFibers"); plotType.push_back("BadAPVs"); plotType.push_back("BadStrips"); plotType.push_back("BadStripsFromAPVs"); plotType.push_back("AllBadStrips");
  std::vector< std::string > subDetName;
  subDetName.push_back("TIB"); subDetName.push_back("TID+"); subDetName.push_back("TID-"); subDetName.push_back("TOB"); subDetName.push_back("TEC+"); subDetName.push_back("TEC-");
  
  // Standard plot options for THStack and for standalone histograms
  const bool stackHistograms = true;
  
  std::string plotStackOptions;
  if(stackHistograms)
    plotStackOptions = "";
  else
    plotStackOptions = "nostack p";
  const std::string plotHistoOptions("p");
  
  // Defer the filling of the normFactor maps to this function
  // Conceptually trivial but lengthy
  fillNormFactorMaps();

//   // Finds number of channels from above map
//   std::string completePartName = subDetName;
//   if(partName.compare("") != 0)
//     completePartName += " " + partName;
//   if(partNumber != 0)
//   {
//     oss.str("");
//     oss << partNumber;
//     completePartName += " " + oss.str();
//   }
//   
//   // Total number of channels in currently processed map
//   const unsigned int nModulesInPart = allModulesTK[completePartName.c_str()];
//   const unsigned int nFibersInPart = allFibersTK[completePartName.c_str()];
//   const unsigned int nAPVsInPart = allAPVsTK[completePartName.c_str()];
//   const unsigned int nStripsInPart = allStripsTK[completePartName.c_str()];
  
  
  TH1F* hTracker;
  TH1F* hTIB;
  TH1F* hTID;
  TH1F* hTOB;
  TH1F* hTEC;
  TH1F* histo;
  TH1F* histo2;
  THStack* histoStack;
  TCanvas* c1;
  TCanvas* c2;
  TLegend* legend;
  TLegend* legend2;
  TPaveText* textX;
  TPaveText* textY;
  std::string entryLabel;
  std::vector<TH1F*> hLayers;
//   unsigned int normFactor;
  
  for(std::vector< std::string >::iterator itPlot=plotType.begin(); itPlot!=plotType.end(); itPlot++)
  {
    // Put together the Tracker histograms with the TIB, TID, TOB and TEC ones
    
    histoName = "h" + *itPlot + "Tracker";
    hTracker = (TH1F*)inputFile->Get(histoName.c_str());
    if(hTracker) {
      hTracker->Scale(1/findNormFactor(*itPlot,"Tracker",stackHistograms));
    }
    else {std::cout << histoName << " not found" << std::endl;}

    histoStack = new THStack(histoName.c_str(), histoName.c_str());

    histoName = "h" + *itPlot + "TIB";
    hTIB = (TH1F*)inputFile->Get(histoName.c_str());
    if(hTIB) {
      hTIB->Scale(1/findNormFactor(*itPlot,"TIB",stackHistograms));
    }
    else {std::cout << histoName << " not found" << std::endl;}
    
    histoName = "h" + *itPlot + "TID";
    hTID = (TH1F*)inputFile->Get(histoName.c_str());
    if(hTID) {
      hTID->Scale(1/findNormFactor(*itPlot,"TID",stackHistograms));
    }
    else {std::cout << histoName << " not found" << std::endl;}

    histoName = "h" + *itPlot + "TOB";
    hTOB = (TH1F*)inputFile->Get(histoName.c_str());
    if(hTOB) {
      hTOB->Scale(1/findNormFactor(*itPlot,"TOB",stackHistograms));
    }
    else {std::cout << histoName << " not found" << std::endl;}

    histoName = "h" + *itPlot + "TEC";
    hTEC = (TH1F*)inputFile->Get(histoName.c_str());
    if(hTEC) {
      hTEC->Scale(1/findNormFactor(*itPlot,"TEC",stackHistograms));
    }
    else {std::cout << histoName << " not found" << std::endl;}
    
    c1 = new TCanvas(("c"+*itPlot+"Tracker").c_str(), "", 1200, 600);
    setCanvasStyle(c1, false);
    //     hTracker->Draw();
    if(hTracker) setHistoStackStyle(hTracker,1);
    //     hTIB->Draw("same");
    if(hTIB) {
      setHistoStackStyle(hTIB,2);
      histoStack->Add(hTIB);
    }
    //     hTID->Draw("same");
    if(hTID) {
      setHistoStackStyle(hTID,3);
      histoStack->Add(hTID);
    }
    //     hTOB->Draw("same");
    if(hTOB) {
      setHistoStackStyle(hTOB,4);
      histoStack->Add(hTOB);
    }
    //     hTEC->Draw("same");
    if(hTEC) {
      setHistoStackStyle(hTEC,6);
      histoStack->Add(hTEC);
    }
    // Bug in ROOT? If we plot a stack with the "stack" option and the Y axis is in log scale,
    // but there are no entries in any of the histograms of the stack, then ROOT crashes
    // Workaround: at this stage, check that at least one histogram has >0 entries,
    // otherwise, switch back to linear Y scale
    // Curiously, there is no crash if the "nostack" option is chosen...
    double histoStackMaximum = histoStack->GetMaximum();
    if(histoStackMaximum==0)
    {
      c1->SetLogy(0);
    }
    histoStack->Draw(plotStackOptions.c_str());
    if(histoStack->GetYaxis()->GetXmax() > 0.9)
      histoStack->GetYaxis()->SetRangeUser(0.,0.1);
    legend = new TLegend(0.4,0.9,0.9,1);
    legend->AddEntry(hTIB, "TIB");
    legend->AddEntry(hTID, "TID");
    legend->AddEntry(hTOB, "TOB");
    legend->AddEntry(hTEC, "TEC");
    setLegendStyle(legend, 2);
    legend->Draw();
    textX = new TPaveText();
    textY = new TPaveText();
    setPaveTextStyle(textX);
    setPaveTextStyle(textY, false);
    textX->Draw();
    textY->Draw();
    gSystem->ProcessEvents();
    c1->Update();
    outputFile->cd();
    c1->Write();
    c1->SaveAs((*itPlot+"Tracker.png").c_str());
    
    delete histoStack;
    delete textX;
    delete textY;
    delete legend;
    delete c1;

   
    
    // Put together the histograms for the different layers of the detectors
    for(std::vector< std::string >::iterator itSub=subDetName.begin(); itSub!=subDetName.end(); itSub++)
    {
      unsigned int nLayers = 0;
      std::string layerName;
//       std::cout << "itSub = " << (*itSub).c_str() << std::endl;
      if((*itSub)=="TIB")
      {
        nLayers=4;
        layerName="Layer";
        legend = new TLegend(0.4,0.9,0.9,1);
        setLegendStyle(legend,2);
      }
      else if((*itSub)=="TID+" || (*itSub)=="TID-")
      {
        nLayers=3;
        layerName="Disk";
        legend = new TLegend(0.35,0.9,0.9,1);
        setLegendStyle(legend,3);
      }
      else if((*itSub)=="TOB")
      {
        nLayers=6;
        layerName="Layer";
        legend = new TLegend(0.35,0.9,0.9,1);
        setLegendStyle(legend,3);
      }
      else if((*itSub)=="TEC+" || (*itSub)=="TEC-")
      {
        nLayers=9;
        layerName="Disk";
        legend = new TLegend(0.35,0.9,1,1);
        setLegendStyle(legend,5);
      }
      
      c1 = new TCanvas(("c" + *itPlot + *itSub).c_str(),"", 1200, 600);
      setCanvasStyle(c1, false);
//       if((*itSub).compare("TEC+")==0 || (*itSub).compare("TEC-")==0)
//       {
//         histoName = "h" + *itPlot + "TEC";
//       }
//       else
//       {
      histoName = "h" + *itPlot + *itSub;
//       }
//       hSubDet = (TH1F*)inputFile->Get(histoName.c_str());
//       setHistoStackStyle(hSubDet,1);
      //       hSubDet->Draw();
      histoStack = new THStack(histoName.c_str(), histoName.c_str());
      
      for(unsigned int iLayer = 1; iLayer<=nLayers; iLayer++)
      {
        oss.str("");
        oss << iLayer;
//         // TIB and TOB have no plus/minus side division
//         // While TEC has it but I plot them separately
//         // TID has plus/minus side division but I plot everything in a single plot
//         if((*itSub).compare("TID")==0)
//         {
//           histoName = "h" + *itPlot + *itSub + "+" + layerName + oss.str();
// //           std::cout << "histoName = " << histoName.c_str() << std::endl;
//           histo = (TH1F*)inputFile->Get(histoName.c_str());
//           
//           // First: plot histogram separately
//           setHistoStyle(histo);
//           c2 = new TCanvas(("c" + *itPlot + *itSub + "+" + layerName + oss.str()).c_str(), "", 1200, 600);
//           setCanvasStyle(c2, true);
//           histo->Draw(plotHistoOptions.c_str());
//           legend2 = new TLegend(0.6,0.92,0.9,0.97);
//           legend2->AddEntry(histo,(*itSub + "+" + layerName + oss.str()).c_str());
//           setLegendStyle(legend2, 1);
//           legend2->Draw();
//           gSystem->ProcessEvents();
//           c2->Update();
//           outputFile->cd();
//           c2->Write();
//           c2->SaveAs((*itPlot + *itSub + "+" + layerName + oss.str() + ".png").c_str());
//           delete legend2;
//           delete c2;
//           
//           // Second: add histogram to THStack
//           hLayers.push_back(histo);
//           setHistoStackStyle(hLayers.back(), iLayer);
//           histoStack->Add(hLayers.back());
//           entryLabel = *itSub + "+ " + layerName + " " + oss.str();
//           legend->AddEntry(hLayers.back(), entryLabel.c_str());
//           
//           
//           histoName = "h" + *itPlot + *itSub + "-" + layerName + oss.str();
// //           std::cout << "histoName = " << histoName.c_str() << std::endl;
//           histo = (TH1F*)inputFile->Get(histoName.c_str());
//           
//           // First: plot histogram separately
//           setHistoStyle(histo);
//           c2 = new TCanvas(("c" + *itPlot + *itSub + "-" + layerName + oss.str()).c_str(), "", 1200, 600);
//           setCanvasStyle(c2, true);
//           histo->Draw(plotHistoOptions.c_str());
//           legend2 = new TLegend(0.6,0.92,0.9,0.97);
//           legend2->AddEntry(histo,(*itSub + "-" + layerName + oss.str()).c_str());
//           setLegendStyle(legend2, 1);
//           legend2->Draw();
//           gSystem->ProcessEvents();
//           c2->Update();
//           outputFile->cd();
//           c2->Write();
//           c2->SaveAs((*itPlot + *itSub + "-" + layerName + oss.str() + ".png").c_str());
//           delete legend2;
//           delete c2;
//           
//           // Second: add histogram to THStack
//           hLayers.push_back(histo);
//           setHistoStackStyle(hLayers.back(), iLayer+nLayers);
//           histoStack->Add(hLayers.back());
//           entryLabel = *itSub + "- " + layerName + " " + oss.str();
//           legend->AddEntry(hLayers.back(), entryLabel.c_str());
// //           hLayers.back()->Draw("same");
//         }
//         else
//         {
          histoName = "h" + *itPlot + *itSub + layerName + oss.str();
//           std::cout << "histoName = " << histoName.c_str() << std::endl;
          histo = (TH1F*)inputFile->Get(histoName.c_str());
	  if(histo) {
	    histo2 = new TH1F(*histo);
	    histo->Scale(1/findNormFactor(*itPlot, *itSub + " " + layerName + " " + oss.str(), false));
	    histo2->Scale(1/findNormFactor(*itPlot, *itSub + " " + layerName + " " + oss.str(), stackHistograms));
	    // First: plot histogram separately
	    setHistoStyle(histo);
	    c2 = new TCanvas(("c" + *itPlot + *itSub +  layerName + oss.str()).c_str(), "", 1200, 600);
	    setCanvasStyle(c2, true);
	    histo->Draw(plotHistoOptions.c_str());
	    double histoMaximum = histo->GetMaximum();
	    // Otherwise it does not draw the pad
	    if(histoMaximum==0)
	      {
		c2->SetLogy(0);
	      }
	    legend2 = new TLegend(0.6,0.92,0.9,0.97);
	    legend2->AddEntry(histo,(*itSub + layerName + oss.str()).c_str());
	    setLegendStyle(legend2, 1);
	    legend2->Draw();
	    textX = new TPaveText();
	    textY = new TPaveText();
	    setPaveTextStyle(textX);
	    setPaveTextStyle(textY,false);
	    textX->Draw();
	    textY->Draw();
	    gSystem->ProcessEvents();
	    c2->Update();
	    outputFile->cd();
	    c2->Write();
	    c2->SaveAs((*itPlot + *itSub + layerName + oss.str() + ".png").c_str());
	    delete textX;
	    delete textY;
	    delete legend2;
	    delete c2;
	    
	    // Second: add histogram to THStack
	    setHistoStyle(histo2);
	    hLayers.push_back(histo2);
	    setHistoStackStyle(hLayers.back(), iLayer);
	    histoStack->Add(hLayers.back());
	    entryLabel = *itSub + " " + layerName + " " + oss.str();
	    legend->AddEntry(hLayers.back(), entryLabel.c_str());
	    //           hLayers.back()->Draw("same");
	    //         }
	    //           delete histo2;
	  }
	  else {std::cout << histoName << " not found" << std::endl;}
      }
      histoStack->Draw(plotStackOptions.c_str());

      // Bug in ROOT? If we plot a stack with the "stack" option and the Y axis is in log scale,
      // but there are no entries in any of the histograms of the stack, then ROOT crashes
      // Workaround: at this stage, check that at least one histogram has >0 entries,
      // otherwise, switch back to linear Y scale
      // Curiously, there is no crash if the "nostack" option is chosen...
      double histoStackMaximum = histoStack->GetMaximum();
      if(histoStackMaximum==0)
      {
        c1->SetLogy(0);
      }
      if(histoStackMaximum > 0.01)
        histoStack->SetMaximum(0.01);
      textX = new TPaveText();
      textY = new TPaveText();
      setPaveTextStyle(textX);
      setPaveTextStyle(textY,false);
      textX->Draw();
      textY->Draw();
      legend->Draw();
      gSystem->ProcessEvents();
      c1->Update();
      outputFile->cd();
      c1->Write();
      c1->SaveAs((*itPlot + *itSub + ".png").c_str());
      delete histoStack;
      delete textX;
      delete textY;
      delete legend;
      delete c1;
    }
  }
  
  inputFile->Close();
  outputFile->Close();
  
}



void setCanvasStyle(TCanvas* c, const bool logScale)
{
  c->SetFillColor(0);
  c->SetFrameBorderMode(0);
  if(logScale)
    c->SetLogy(1);
  else
    c->SetLogy(0);
  c->SetCanvasSize(1200,600);
  c->SetWindowSize(1200,600);
}



void setHistoStyle(TH1* h)
{
  h->SetLineStyle(0);
  h->SetMarkerStyle(3);
  h->SetMarkerSize(1);
  h->SetMarkerColor(1);
  h->SetStats(kFALSE);
//   h->GetYaxis()->SetTitle("Fraction of total");
//   h->GetXaxis()->SetTitle("IOV");
//   h->GetXaxis()->SetTitleOffset(-0.3);
  // Avoid having too many bins with labels
  if(h->GetNbinsX() > 25)
    for(int i = 1; i < h->GetNbinsX()-1; i++)
      if((i%(h->GetNbinsX()/25+1)))
        h->GetXaxis()->SetBinLabel(i+1,"");
}



void setHistoStackStyle(TH1* h, const unsigned int lineColor)
{
  h->SetLineStyle(0);
  //   h->SetDrawOption("e1p");
  // Best marker types are 20-23 - use them with different colors
  h->SetMarkerStyle(lineColor%4+20);
  h->SetMarkerSize(1);
  h->SetMarkerColor(lineColor);
  h->SetLineColor(lineColor);
  h->SetFillColor(lineColor);
  h->SetLineWidth(2);
  h->SetStats(kFALSE);
  h->GetYaxis()->SetTitle("Fraction of total");
//   h->GetXaxis()->SetTitle("IOV");
//   h->GetXaxis()->SetTitleOffset(1.2);
  // Avoid having too many bins with labels
  if(h->GetNbinsX() > 25)
    for(int i = 1; i < h->GetNbinsX()-1; i++)
      if(i%(h->GetNbinsX()/25+1))
        h->GetXaxis()->SetBinLabel(i+1,"");
}



void setLegendStyle(TLegend* l, const unsigned int nColumns)
{
  l->SetNColumns(nColumns);
  l->SetFillColor(0);
}



void setPaveTextStyle(TPaveText* t, const bool isHorizontal)
{
  t->SetLineStyle(0);
  t->SetFillColor(0);
  t->SetFillStyle(0);
  t->SetBorderSize(0);
  if(isHorizontal)
  {
    t->SetX1NDC(0.905);
    t->SetX2NDC(0.975);
    t->SetY1NDC(0.062);
    t->SetY2NDC(0.095);
    t->AddText("IOV");
  }
  else
  {
    t->SetX1NDC(0.03);
    t->SetX2NDC(0.05);
    t->SetY1NDC(0.33);
    t->SetY2NDC(0.68);
    TText* t1 = t->AddText("Fraction of total");
    t1->SetTextAngle(90.);
  }
}



void fillNormFactorMaps()
{
  // Number of modules, fibers, APVs, strips for each tracker part
  std::vector< std::string > tkParts;
  tkParts.push_back("Tracker");
  tkParts.push_back("TIB");
  tkParts.push_back("TID");
  tkParts.push_back("TOB");
  tkParts.push_back("TEC");
  tkParts.push_back("TIB Layer 1");
  tkParts.push_back("TIB Layer 2");
  tkParts.push_back("TIB Layer 3");
  tkParts.push_back("TIB Layer 4");
  tkParts.push_back("TID+ Disk 1");
  tkParts.push_back("TID+ Disk 2");
  tkParts.push_back("TID+ Disk 3");
  tkParts.push_back("TID- Disk 1");
  tkParts.push_back("TID- Disk 2");
  tkParts.push_back("TID- Disk 3");
  tkParts.push_back("TOB Layer 1");
  tkParts.push_back("TOB Layer 2");
  tkParts.push_back("TOB Layer 3");
  tkParts.push_back("TOB Layer 4");
  tkParts.push_back("TOB Layer 5");
  tkParts.push_back("TOB Layer 6");
  tkParts.push_back("TEC+ Disk 1");
  tkParts.push_back("TEC+ Disk 2");
  tkParts.push_back("TEC+ Disk 3");
  tkParts.push_back("TEC+ Disk 4");
  tkParts.push_back("TEC+ Disk 5");
  tkParts.push_back("TEC+ Disk 6");
  tkParts.push_back("TEC+ Disk 7");
  tkParts.push_back("TEC+ Disk 8");
  tkParts.push_back("TEC+ Disk 9");
  tkParts.push_back("TEC- Disk 1");
  tkParts.push_back("TEC- Disk 2");
  tkParts.push_back("TEC- Disk 3");
  tkParts.push_back("TEC- Disk 4");
  tkParts.push_back("TEC- Disk 5");
  tkParts.push_back("TEC- Disk 6");
  tkParts.push_back("TEC- Disk 7");
  tkParts.push_back("TEC- Disk 8");
  tkParts.push_back("TEC- Disk 9");
  
  std::vector<unsigned int> nModulesStack;
  nModulesStack.push_back(15148);
  nModulesStack.push_back(15148);
  nModulesStack.push_back(15148);
  nModulesStack.push_back(15148);
  nModulesStack.push_back(15148);
  nModulesStack.push_back(2724);
  nModulesStack.push_back(2724);
  nModulesStack.push_back(2724);
  nModulesStack.push_back(2724);
  nModulesStack.push_back(408);
  nModulesStack.push_back(408);
  nModulesStack.push_back(408);
  nModulesStack.push_back(408);
  nModulesStack.push_back(408);
  nModulesStack.push_back(408);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(5208);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  nModulesStack.push_back(3200);
  
  std::vector<unsigned int> nModulesNoStack;
  nModulesNoStack.push_back(15148);
  nModulesNoStack.push_back(2724);
  nModulesNoStack.push_back(816);
  nModulesNoStack.push_back(5208);
  nModulesNoStack.push_back(6400);
  nModulesNoStack.push_back(672);
  nModulesNoStack.push_back(864);
  nModulesNoStack.push_back(540);
  nModulesNoStack.push_back(648);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(136);
  nModulesNoStack.push_back(1008);
  nModulesNoStack.push_back(1152);
  nModulesNoStack.push_back(648);
  nModulesNoStack.push_back(720);
  nModulesNoStack.push_back(792);
  nModulesNoStack.push_back(888);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(312);
  nModulesNoStack.push_back(312);
  nModulesNoStack.push_back(272);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(408);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(360);
  nModulesNoStack.push_back(312);
  nModulesNoStack.push_back(312);
  nModulesNoStack.push_back(272);

  //
  std::vector<unsigned int> nFibersStack;
  nFibersStack.push_back(36392);
  nFibersStack.push_back(36392);
  nFibersStack.push_back(36392);
  nFibersStack.push_back(36392);
  nFibersStack.push_back(36392);
  nFibersStack.push_back(6984);
  nFibersStack.push_back(6984);
  nFibersStack.push_back(6984);
  nFibersStack.push_back(6984);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(1104);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(12096);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  nFibersStack.push_back(7552);
  
  std::vector<unsigned int> nFibersNoStack;
  nFibersNoStack.push_back(36392);
  nFibersNoStack.push_back(6984);
  nFibersNoStack.push_back(2208);
  nFibersNoStack.push_back(12096);
  nFibersNoStack.push_back(15104);
  nFibersNoStack.push_back(2016);
  nFibersNoStack.push_back(2592);
  nFibersNoStack.push_back(1080);
  nFibersNoStack.push_back(1296);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(368);
  nFibersNoStack.push_back(2016);
  nFibersNoStack.push_back(2304);
  nFibersNoStack.push_back(1296);
  nFibersNoStack.push_back(1440);
  nFibersNoStack.push_back(2376);
  nFibersNoStack.push_back(2664);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(704);
  nFibersNoStack.push_back(704);
  nFibersNoStack.push_back(624);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(992);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(848);
  nFibersNoStack.push_back(704);
  nFibersNoStack.push_back(704);
  nFibersNoStack.push_back(624);
  
  //
  std::vector<unsigned int> nAPVsStack;
  nAPVsStack.push_back(72784);
  nAPVsStack.push_back(72784);
  nAPVsStack.push_back(72784);
  nAPVsStack.push_back(72784);
  nAPVsStack.push_back(72784);
  nAPVsStack.push_back(13968);
  nAPVsStack.push_back(13968);
  nAPVsStack.push_back(13968);
  nAPVsStack.push_back(13968);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(2208);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(24192);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  nAPVsStack.push_back(15104);
  
  std::vector<unsigned int> nAPVsNoStack;
  nAPVsNoStack.push_back(72784);
  nAPVsNoStack.push_back(13968);
  nAPVsNoStack.push_back(4416);
  nAPVsNoStack.push_back(24192);
  nAPVsNoStack.push_back(30208);
  nAPVsNoStack.push_back(4032);
  nAPVsNoStack.push_back(5184);
  nAPVsNoStack.push_back(2160);
  nAPVsNoStack.push_back(2592);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(736);
  nAPVsNoStack.push_back(4032);
  nAPVsNoStack.push_back(4608);
  nAPVsNoStack.push_back(2592);
  nAPVsNoStack.push_back(2880);
  nAPVsNoStack.push_back(4752);
  nAPVsNoStack.push_back(5328);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1408);
  nAPVsNoStack.push_back(1408);
  nAPVsNoStack.push_back(1248);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1984);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1696);
  nAPVsNoStack.push_back(1408);
  nAPVsNoStack.push_back(1408);
  nAPVsNoStack.push_back(1248);
  
  //
  std::vector<unsigned int> nStripsStack;
  nStripsStack.push_back(9316352);
  nStripsStack.push_back(9316352);
  nStripsStack.push_back(9316352);
  nStripsStack.push_back(9316352);
  nStripsStack.push_back(9316352);
  nStripsStack.push_back(1787904);
  nStripsStack.push_back(1787904);
  nStripsStack.push_back(1787904);
  nStripsStack.push_back(1787904);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(282624);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(3096576);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  nStripsStack.push_back(1933312);
  
  std::vector<unsigned int> nStripsNoStack;
  nStripsNoStack.push_back(9316352);
  nStripsNoStack.push_back(1787904);
  nStripsNoStack.push_back(565248);
  nStripsNoStack.push_back(3096576);
  nStripsNoStack.push_back(3866624);
  nStripsNoStack.push_back(516096);
  nStripsNoStack.push_back(663552);
  nStripsNoStack.push_back(276480);
  nStripsNoStack.push_back(331776);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(94208);
  nStripsNoStack.push_back(516096);
  nStripsNoStack.push_back(589824);
  nStripsNoStack.push_back(331776);
  nStripsNoStack.push_back(368640);
  nStripsNoStack.push_back(608256);
  nStripsNoStack.push_back(681984);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(180224);
  nStripsNoStack.push_back(180224);
  nStripsNoStack.push_back(159744);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(253952);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(217088);
  nStripsNoStack.push_back(180224);
  nStripsNoStack.push_back(180224);
  nStripsNoStack.push_back(159744);
  
  for(unsigned int i = 0; i < tkParts.size(); i++)
  {
    modulesStackNormFactors[tkParts[i].c_str()] = nModulesStack[i];
    modulesNoStackNormFactors[tkParts[i].c_str()] = nModulesNoStack[i];
    fibersStackNormFactors[tkParts[i].c_str()] = nFibersStack[i];
    fibersNoStackNormFactors[tkParts[i].c_str()] = nFibersNoStack[i];
    APVsStackNormFactors[tkParts[i].c_str()] = nAPVsStack[i];
    APVsNoStackNormFactors[tkParts[i].c_str()] = nAPVsNoStack[i];
    stripsStackNormFactors[tkParts[i].c_str()] = nStripsStack[i];
    stripsNoStackNormFactors[tkParts[i].c_str()] = nStripsNoStack[i];
  }
  
  //   for(std::map< std::string, unsigned int>::iterator it = allStripsTK.begin(); it != allStripsTK.end(); it++)
  //   {
    //     std::cout << it->first.c_str() << " " << it->second << std::endl;
    //   }
    
  
}



double findNormFactor(const std::string currentPlotType, const std::string currentPart, const bool stackOption)
{

//   std::cout << "findNormFactor(): Finding normalization factor for this part: \"" << currentPart.c_str() << "\".\n";
//   std::cout << "                  Plot type is: \"" << currentPlotType.c_str() << "\".\n";
//   std::cout << "                  stack option is: " << stackOption << std::endl;
  if(stackOption)
  {
    if(currentPlotType == "BadModules")
    {
      return modulesStackNormFactors[currentPart.c_str()];
    }
    else if(currentPlotType == "BadFibers")
    {
      return fibersStackNormFactors[currentPart];
    }
    else if(currentPlotType == "BadAPVs")
    {
      return APVsStackNormFactors[currentPart];
    }
    else if(currentPlotType == "AllBadStrips" || 
            currentPlotType == "BadStripsFromAPVs" || 
            currentPlotType == "BadStrips")
    {
      return stripsStackNormFactors[currentPart];
    }
    else
    {
      std::cout << "findNormFactor(): ERROR! Requested a non supported plot type: " << currentPlotType.c_str() << std::endl;
      std::cout << "                  Add this to the function body or correct the error\n";
      return 0; // This should trigger a divByZero error...
    }
  }
  else
  {
    if(currentPlotType == "BadModules")
    {
      return modulesNoStackNormFactors[currentPart.c_str()];
    }
    else if(currentPlotType == "BadFibers")
    {
      return fibersNoStackNormFactors[currentPart];
    }
    else if(currentPlotType == "BadAPVs")
    {
      return APVsNoStackNormFactors[currentPart];
    }
    else if(currentPlotType == "BadStrips" || 
            currentPlotType == "BadStripsFromAPVs" || 
            currentPlotType == "AllBadStrips")
    {
      return stripsNoStackNormFactors[currentPart];
    }
    else
    {
      std::cout << "findNormFactor(): ERROR! Requested a non supported plot type: \"" << currentPlotType.c_str() << "\"\n";
      std::cout << "                  Add this to the function body or correct the error otherwise.\n";
      return 0; // This should trigger a divByZero error...
    }
  }
}
