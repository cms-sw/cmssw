#include "TFile.h"
#include "TChain.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TRegexp.h"
#include "TGraphErrors.h"
#include "iostream"
#include "vector"
#include "map"
#include "TTreeFormula.h"
#include "GetOptimization.h"

struct filter {
  TString name;
  Int_t pathNum;
  Int_t direction; // -1: <, 0: bool, 1: >
  Double_t hltBarrelCut;
  Double_t hltEndcapCut;
  Double_t maxCut;
};

struct path {
  std::vector<TString> names;
  Int_t nCandsCut;
  std::vector<filter> filters;
};

void GetOptimization() {
  struct path thisPath;
  std::vector<path> paths;
  std::vector<TString> pathNames;
  struct filter thisFilter;
  std::vector<filter> filters;
  std::vector<std::pair<std::vector<TString>,Double_t> > xSections;
  std::vector<TString> filenamesBkg;
  std::vector<TString> filenamesSig;
  /* Parameters */
  Int_t nCuts = 120;
  Bool_t doBandE = true; // if true, do seperate for cuts in barrel and endcap

  Double_t luminosity = 2.0E33; // in cm^-2 s^-1

  // Cross-sections in mb
  filenamesBkg.push_back("../test/QCD-HLTVars-1.root");
  // filenamesBkg.push_back("sameXSection");
  xSections.push_back(make_pair(filenamesBkg, 2.16E-2));
  filenamesBkg.clear();
  // filenamesBkg.push_back("newXSection.root");
  // ...
  // xSections.push_back(make_pair(filenamesBkg, xSection));
  // filenamesBkg.clear();
  filenamesSig.push_back("../test/ZEE-HLTVars.root");
  /* ********** */
  // Filters
  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltBarrelCut = 0.;
  thisFilter.hltEndcapCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 15.;
  thisFilter.hltEndcapCut = 15.;
  thisFilter.maxCut = 60.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 3.;
  thisFilter.hltEndcapCut = 3.;
  thisFilter.maxCut = 6.;
  filters.push_back(thisFilter);
  thisFilter.name = "pixMatch";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 0;
  thisFilter.hltEndcapCut = 0;
  thisFilter.maxCut = 0;
  filters.push_back(thisFilter);
  thisFilter.name = "Eoverp";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 1.5;
  thisFilter.hltEndcapCut = 2.45;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 0.06;
  thisFilter.hltEndcapCut = 0.06;
  thisFilter.maxCut = 0.24;
  filters.push_back(thisFilter);
  pathNames.push_back("SingleElecsPT.");
  pathNames.push_back("SingleElecs.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("RelaxedSingleElecsPT.");
  pathNames.push_back("RelaxedSingleElecs.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltBarrelCut = 0.;
  thisFilter.hltEndcapCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 10.;
  thisFilter.hltEndcapCut = 10.;
  thisFilter.maxCut = 40.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 9.;
  thisFilter.hltEndcapCut = 9.;
  thisFilter.maxCut = 18.;
  filters.push_back(thisFilter);
  thisFilter.name = "pixMatch";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 0;
  thisFilter.hltEndcapCut = 0;
  thisFilter.maxCut = 0;
  filters.push_back(thisFilter);
  thisFilter.name = "Eoverp";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 15000;
  thisFilter.hltEndcapCut = 24500;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 0.4;
  thisFilter.hltEndcapCut = 0.4;
  thisFilter.maxCut = 0.12;
  filters.push_back(thisFilter);
  pathNames.push_back("DoubleElecsPT.");
  pathNames.push_back("DoubleElecs.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("RelaxedDoubleElecsPT.");
  pathNames.push_back("RelaxedDoubleElecs.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltBarrelCut = 0.;
  thisFilter.hltEndcapCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 30.;
  thisFilter.hltEndcapCut = 30.;
  thisFilter.maxCut = 60.;
  filters.push_back(thisFilter);
  thisFilter.name = "IEcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 1.5;
  thisFilter.hltEndcapCut = 1.5;
  thisFilter.maxCut = 6.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 4.;
  thisFilter.hltEndcapCut = 6.;
  thisFilter.maxCut = 12.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 1;
  thisFilter.hltEndcapCut = 1;
  thisFilter.maxCut = 5;
  filters.push_back(thisFilter);
  pathNames.push_back("SinglePhots.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("RelaxedSinglePhots.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltBarrelCut = 0.;
  thisFilter.hltEndcapCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 20.;
  thisFilter.hltEndcapCut = 20.;
  thisFilter.maxCut = 40.;
  filters.push_back(thisFilter);
  thisFilter.name = "IEcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 2.5;
  thisFilter.hltEndcapCut = 2.5;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 6.;
  thisFilter.hltEndcapCut = 8.;
  thisFilter.maxCut = 24.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 3;
  thisFilter.hltEndcapCut = 3;
  thisFilter.maxCut = 6;
  filters.push_back(thisFilter);
  pathNames.push_back("DoublePhots.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("RelaxedDoublePhots.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  /* *********** */

  Int_t cutNum = 0, pathNum = 0, filterNum = 0, oldFilterNum = 0, fileNum = 0, xSecNum = 0;
  Double_t cut = 0.;
  Long64_t passSig = 0, totalSig = 0;
  Long64_t passBkg = 0, totalBkg = 0;
  Double_t effSig = 0., errSig = 0.;
  Double_t effBkg = 0., errBkg = 0., rateTotBkg = 0., errTotBkg = 0.;
  Double_t conversion = 1.0E-27;
  std::vector<std::pair<Double_t,Double_t> > sigPass;
  std::vector<std::pair<Double_t,Double_t> > bkgPass;

  TString cutText;
  TString cutTextEcap;
  TString baseCutText;
  TString baseCutTextEcap;
  TString cutBasePT1;
  TString cutBasePT2;
  TString cutBase1;
  TString cutBase2;
  //  cutBasePT1 = "ElecHLTCutVarsPreTracks_hltCutVars_";
  //  cutBasePT2 = "ElecsPT_EGAMMAHLT.obj.";
  //  cutBase1 = "";
  //  cutBase2 = "Elecs_EGAMMAHLT.obj.";

  std::vector<std::vector<TGraphErrors> > EffVBkg;
  std::vector<std::vector<TGraphErrors> > EffVBkgEcap;
  std::vector<TGraphErrors> pathEffVBkgs;
  TGraphErrors filterEffVBkgs(nCuts);
  for (pathNum = 0; pathNum < paths.size(); pathNum++) {
    for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
      //      filterEffVBkgs = new TGraphErrors(nCuts);
      pathEffVBkgs.push_back(filterEffVBkgs);
    }
    EffVBkg.push_back(pathEffVBkgs);
    if (doBandE) {
      EffVBkgEcap.push_back(pathEffVBkgs);
    }
  }

  std::vector<std::pair<TChain*,Double_t> > bkgEvents;
  for (xSecNum = 0; xSecNum < xSections.size(); xSecNum++) {
    TChain *bkgProc = new TChain("Events");
    for (fileNum = 0; fileNum < (xSections[xSecNum].first).size(); fileNum++) {
      bkgProc->Add((xSections[xSecNum].first)[fileNum]);
    }
    bkgEvents.push_back(make_pair(bkgProc,xSections[xSecNum].second));
  }

  TChain *sigEvents = new TChain("Events");
  for (fileNum = 0; fileNum < filenamesSig.size(); fileNum++) {
    sigEvents->Add(filenamesSig[fileNum]);
  }

  Double_t testX, testY, testdX, testdY;
  Double_t thisBCut = 0., thisECut = 0.;
  TString pathName, filterName;
  for (cutNum = 0; cutNum < nCuts; cutNum++) {
    cout<<"Cut "<<cutNum<<endl;
    for (pathNum = 0; pathNum < paths.size(); pathNum++) {
      for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
	if ((paths[pathNum].filters)[filterNum].maxCut != 0.) {
	  cutText = "(Sum$(";
	  for (oldFilterNum = 0; oldFilterNum < filterNum; oldFilterNum++) { 
	    pathName = (paths[pathNum].names)[(paths[pathNum].filters)[filterNum].pathNum];
	    filterName = (paths[pathNum].filters)[oldFilterNum].name;
	    thisBCut = (paths[pathNum].filters)[oldFilterNum].hltBarrelCut;
	    thisECut = (paths[pathNum].filters)[oldFilterNum].hltEndcapCut;
	    if (thisBCut == thisECut) {
  	      cutText += pathName;
	      cutText += filterName;
  	      switch ((paths[pathNum].filters)[oldFilterNum].direction) {
	      case -1:
	        cutText += " < ";
	        cutText += (paths[pathNum].filters)[oldFilterNum].hltBarrelCut;
  	        break;
	      case 0:
	        break;
	      case 1:
	        cutText += " > ";
	        cutText += (paths[pathNum].filters)[oldFilterNum].hltBarrelCut;
	        break;
	      default:
	        cout<<"Invalid value of direction in "<<pathName<<filterName<<endl;
	        break;
	      }
	    }
	    else {
	      cutText += "((";
  	      cutText += pathName;
	      cutText += filterName;
              switch ((paths[pathNum].filters)[oldFilterNum].direction) {
	      case -1:
	        cutText += " < ";
	        cutText += thisBCut;
  	        break;
	      case 0:
	        break;
	      case 1:
	        cutText += " > ";
	        cutText += thisBCut;
	        break;
	      default:
	        cout<<"Invalid value of direction in "<<pathName<<filterName<<endl;
	        break;
	      }
	      cutText += " && abs(";
	      cutText += pathName;
	      cutText += "eta) < 1.5) || (";
	      cutText += pathName;
	      cutText += filterName;
	      switch ((paths[pathNum].filters)[oldFilterNum].direction) {
              case -1:
                cutText += " < ";
                cutText += thisECut;
                break;
              case 0:
                break;
              case 1:
                cutText += " > ";
                cutText += thisECut;
                break;
              default:
                cout<<"Invalid value of direction in "<<pathName<<filterName<<endl;
                break;
              }
	      cutText += " && abs(";
              cutText += pathName;
              cutText += "eta) > 1.5 && abs(";
	      cutText += pathName;
	      cutText += "eta) < 2.5))";
	    }
	    if (oldFilterNum != filterNum - 1) cutText += " && ";
	  }
	  baseCutText = cutText;
	  pathName = paths[pathNum].names[(paths[pathNum].filters)[filterNum].pathNum];
	  filterName = (paths[pathNum].filters)[filterNum].name;
	  cutText += " && ";
	  cutText += pathName;
	  cutText += filterName;
	  cut = (Double_t)cutNum / (Double_t)nCuts * (paths[pathNum].filters)[oldFilterNum].maxCut;
	  switch ((paths[pathNum].filters)[filterNum].direction) {
	  case -1:
	    cutText += " < ";
	    cutText += cut;
	    break;
	  case 0:
	    break;
	  case 1:
	    cutText += " > ";
	    cutText += cut;
	    break;
	  default:
	    cout<<"Invalid value of direction in "<<pathName<<filterName<<endl;
	    break;
	  }
	  if (doBandE) {
	    cutTextEcap = cutText;
	    cutText += " && abs(";
	    cutText += pathName;
	    cutText += "eta) < 1.5";
	    cutTextEcap += " && abs(";
	    cutTextEcap += pathName;
	    cutTextEcap += "eta) > 1.5 && abs(";
	    cutTextEcap += pathName;
	    cutTextEcap += ") < 2.5";
	    baseCutText += " && abs(";
	    baseCutText += pathName;
	    baseCutTextEcap = baseCutText;
	    baseCutText += "eta) < 1.5";
	    baseCutTextEcap += "eta) > 1.5 && abs(";
	    baseCutTextEcap += pathName;
	    baseCutTextEcap += "eta) < 2.5";
	  }
	  cutText += ") >= ";
	  cutText += paths[pathNum].nCandsCut;
	  cutText += ")";
	  cutTextEcap += ") >= ";
	  cutTextEcap += paths[pathNum].nCandsCut;
	  cutTextEcap += ")";
	  baseCutText += ") >= ";
	  baseCutText += paths[pathNum].nCandsCut;
	  baseCutText += ")";
	  baseCutTextEcap += ") >= ";
	  baseCutTextEcap += paths[pathNum].nCandsCut;
	  baseCutTextEcap += ")";
   
	  cout<<cutText<<endl;
	  cout<<cutTextEcap<<endl;
	  //	  cout<<cutText<<endl;
	  //	  cout<<baseCutText<<endl;
	  passSig = sigEvents->Draw("",cutText);
	  totalSig = sigEvents->Draw("",baseCutText);
	  if (totalSig != 0) {
	    effSig = (Double_t)passSig / (Double_t)totalSig;
	    errSig = sqrt(effSig * (1. - effSig) / (Double_t)totalSig);
	  }
	  else {
	    effSig = 0.;
	    errSig = 0.;
	  }
	  rateTotBkg = 0.;
	  errTotBkg = 0.;
	  for (xSecNum = 0; xSecNum < bkgEvents.size(); xSecNum++) {
	    passBkg = bkgEvents[xSecNum].first->Draw("",cutText);
	    totalBkg = bkgEvents[xSecNum].first->Draw("","");
	    if (totalBkg != 0) {
	      effBkg = (Double_t)passBkg / (Double_t)totalBkg;
	      errBkg = sqrt(effBkg * (1. - effBkg) / (Double_t)totalBkg);
	    }
	    else {
	      effBkg = 0.;
	      errBkg = 0.;
	    }
	    rateTotBkg += effBkg * bkgEvents[xSecNum].second * luminosity * conversion;
	    errTotBkg = sqrt(errTotBkg * errTotBkg + errBkg * errBkg * bkgEvents[xSecNum].second * luminosity * conversion * bkgEvents[xSecNum].second * luminosity * conversion);
	  }
	    
	  if (cutNum == 6) {
	    cout<<cutText<<endl;
	    cout<<rateTotBkg<<" +- "<<errTotBkg<<", "<<effSig<<" +- "<<errSig<<endl;;
	  }
	  EffVBkg[pathNum][filterNum].SetPoint(cutNum, rateTotBkg, effSig);
	  EffVBkg[pathNum][filterNum].SetPointError(cutNum, errTotBkg, errSig);
	  if (cutNum == 6) {
	    EffVBkg[pathNum][filterNum].GetPoint(cutNum, testX, testY);
	    cout<<testX<<", "<<testY<<endl;
	  }

	  if (doBandE) {
	    passSig = sigEvents->Draw("",cutTextEcap);
	    totalSig = sigEvents->Draw("",baseCutText);
	    if (totalSig != 0) {
	      effSig = (Double_t)passSig / (Double_t)totalSig;
	      errSig = sqrt(effSig * (1. - effSig) / (Double_t)totalSig);
	    }
	    else {
	      effSig = 0.;
	      errSig = 0.;
	    }
	    rateTotBkg = 0.;
	    errTotBkg = 0.;
	    for (xSecNum = 0; xSecNum < bkgEvents.size(); xSecNum++) {
	      passBkg = bkgEvents[xSecNum].first->Draw("",cutText);
	      totalBkg = bkgEvents[xSecNum].first->Draw("","");
	      if (totalBkg != 0) {
		effBkg = (Double_t)passBkg / (Double_t)totalBkg;
		errBkg = sqrt(effBkg * (1. - effBkg) / (Double_t)totalBkg);
	      }
	      else {
		effBkg = 0.;
		errBkg = 0.;
	      }
	      rateTotBkg += effBkg * bkgEvents[xSecNum].second * luminosity * conversion;
	      errTotBkg = sqrt(errTotBkg * errTotBkg + errBkg * errBkg * bkgEvents[xSecNum].second * luminosity * conversion * bkgEvents[xSecNum].second * luminosity * conversion);
	    }
	    
	    if (cutNum == 6) {
	      cout<<cutText<<endl;
	      cout<<rateTotBkg<<" +- "<<errTotBkg<<", "<<effSig<<" +- "<<errSig<<endl;;
	    }
	    EffVBkgEcap[pathNum][filterNum].SetPoint(cutNum, rateTotBkg, effSig);
	    EffVBkgEcap[pathNum][filterNum].SetPointError(cutNum, errTotBkg, errSig);
	    if (cutNum == 6) {
	      EffVBkg[pathNum][filterNum].GetPoint(cutNum, testX, testY);
	      cout<<testX<<", "<<testY<<endl;
	    }
	  }
	}
      }
    }
  }
  TCanvas *myCanvas;
  TString tempPathName, canvasTitle, graphTitle, outFilename;
  Int_t n;
  Int_t nGraphs, curGraph;
  for (pathNum = 0; pathNum < paths.size(); pathNum++) {
    canvasTitle = "Efficiency vs. Background for ";
    tempPathName = paths[pathNum].names[paths[pathNum].filters[(paths[pathNum].filters).size()-1].pathNum];
    tempPathName.Resize(tempPathName.Index(".", 1, 0, TString::kExact));
    outFilename = "./images/";
    outFilename += tempPathName;
    outFilename += "EffVBkg.gif";
    n = 0;
    while (tempPathName.Contains(TRegexp("[a-z][A-Z]")) && n < 10) {
      tempPathName.Insert(tempPathName.Index(TRegexp("[a-z][A-Z]"))+1, " ");
      n++;
    }
    canvasTitle += tempPathName;
    nGraphs = 0;
    for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
      if ((paths[pathNum].filters)[filterNum].maxCut != 0) nGraphs++;
    }
    myCanvas = new TCanvas("myCanvas", canvasTitle, 0, 0, 1000, 500*(nGraphs / 2 + 1));
    myCanvas->Divide(2,nGraphs / 2 + 1);
    curGraph = 0;
    for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
      if ((paths[pathNum].filters)[filterNum].maxCut != 0.) {
        myCanvas->cd(curGraph+1);
	curGraph++;
        graphTitle = "Efficiency vs. Background for ";
	graphTitle += (paths[pathNum].filters)[filterNum].name;
	graphTitle += " Filter;Background Rate (Hz);Signal Eff.";
	EffVBkg[pathNum][filterNum].SetTitle(graphTitle);
	EffVBkg[pathNum][filterNum].Draw("AP");
      }
    }
    myCanvas->Print(outFilename);
    if (doBandE) {
      canvasTitle = "Efficiency vs. Background for ";
      tempPathName = paths[pathNum].names[paths[pathNum].filters[(paths[pathNum].filters).size()-1].pathNum];
      tempPathName.Resize(tempPathName.Index(".", 1, 0, TString::kExact));
      tempPathName += "Endcap";
      outFilename = "./images/";
      outFilename += tempPathName;
      outFilename += "EffVBkg.gif";
      n = 0;
      while (tempPathName.Contains(TRegexp("[a-z][A-Z]")) && n < 10) {
	tempPathName.Insert(tempPathName.Index(TRegexp("[a-z][A-Z]"))+1, " ");
	n++;
      }
      canvasTitle += tempPathName;
      nGraphs = 0;
      for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
	if ((paths[pathNum].filters)[filterNum].maxCut != 0) nGraphs++;
      }
      myCanvas = new TCanvas("myCanvas", canvasTitle, 0, 0, 1000, 500*(nGraphs / 2 + 1));
      myCanvas->Divide(2,nGraphs / 2 + 1);
      curGraph = 0;
      for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
	if ((paths[pathNum].filters)[filterNum].maxCut != 0.) {
	  myCanvas->cd(curGraph+1);
	  curGraph++;
	  graphTitle = "Efficiency vs. Background for ";
	  graphTitle += (paths[pathNum].filters)[filterNum].name;
	  graphTitle += " Filter in Endcap;Background Rate (Hz);Signal Eff.";
	  EffVBkgEcap[pathNum][filterNum].SetTitle(graphTitle);
	  EffVBkgEcap[pathNum][filterNum].Draw("AP");
	}
      }
      myCanvas->Print(outFilename);
    }
  }

  TH1F *timingSig = new TH1F("timingSig", "Timing of Single Electron Filters in Signal Events", 6, 0, 6);
  timingSig->SetCanExtend(TH1::kAllAxes);
  timingSig->SetStats(0);
  TTreeFormula *l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",sigEvents);
  TTreeFormula *EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",sigEvents);
  TTreeFormula *IHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecIHcal",sigEvents);
  TTreeFormula *pixMatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.pixMatch",sigEvents);
  TTreeFormula *EoverpTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Eoverp",sigEvents);
  TTreeFormula *ItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecItrack",sigEvents);
  Long64_t event = 0;
  Double_t avgL1Match = 0.;
  Double_t avgEt = 0.;
  Double_t avgIHcal = 0.;
  Double_t avgPixMatch = 0.;
  Double_t avgEoverp = 0.;
  Double_t avgItrack = 0.;
  for (event = 0; event < sigEvents->GetEntries(); event++) {
    sigEvents->LoadTree(event);
    avgL1Match = (event*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgEt = (event*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgIHcal = (event*avgIHcal + IHcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgPixMatch = (event*avgPixMatch + pixMatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgEoverp = (event*avgEoverp + EoverpTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgItrack = (event*avgItrack + ItrackTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
  }
  timingSig->Fill("L1 Match", avgL1Match);
  timingSig->Fill("Et", avgEt); 
  timingSig->Fill("IHcal", avgIHcal); 
  timingSig->Fill("Pix Match", avgPixMatch); 
  timingSig->Fill("E/p", avgEoverp); 
  timingSig->Fill("Itrack", avgItrack); 
  timingSig->LabelsDeflate("X");
  timingSig->LabelsOption("v");

  TH1F *timingBkg = new TH1F("timingBkg", "Timing of Single Electron Filters in Background Events", 6, 0, 6);
  timingBkg->SetCanExtend(TH1::kAllAxes);
  timingBkg->SetStats(0);
  avgL1Match = 0.;
  avgEt = 0.;
  avgIHcal = 0.;
  avgPixMatch = 0.;
  avgEoverp = 0.;
  avgItrack = 0.;
  for (xSecNum = 0; xSecNum < bkgEvents.size(); xSecNum++) {
    delete l1MatchTiming; l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",bkgEvents[xSecNum].first);
    delete EtTiming; EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",bkgEvents[xSecNum].first);
    delete IHcalTiming; IHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecIHcal",bkgEvents[xSecNum].first);
    delete pixMatchTiming; pixMatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.pixMatch",bkgEvents[xSecNum].first);
    delete EoverpTiming; EoverpTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Eoverp",bkgEvents[xSecNum].first);
    delete ItrackTiming; ItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecItrack",bkgEvents[xSecNum].first);
    event = 0;
    for (event = 0; event <  bkgEvents[xSecNum].first->GetEntries(); event++) {
      bkgEvents[xSecNum].first->LoadTree(event);
      avgL1Match = (event*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgEt = (event*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgIHcal = (event*avgIHcal + IHcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgPixMatch = (event*avgPixMatch + pixMatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgEoverp = (event*avgEoverp + EoverpTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgItrack = (event*avgItrack + ItrackTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    }
  }
  timingBkg->Fill("L1 Match", avgL1Match);
  timingBkg->Fill("Et", avgEt); 
  timingBkg->Fill("IHcal", avgIHcal); 
  timingBkg->Fill("Pix Match", avgPixMatch); 
  timingBkg->Fill("E/p", avgEoverp); 
  timingBkg->Fill("Itrack", avgItrack); 
  timingBkg->LabelsDeflate("X");
  timingBkg->LabelsOption("v");

  myCanvas = new TCanvas("myCanvas", "Timing vs. Filter for Isolated Electron Filters", 1000, 500);
  myCanvas->Divide(2,1);
  myCanvas->cd(1);
  timingSig->Draw();
  myCanvas->cd(2);
  timingBkg->Draw();
  myCanvas->Print("images/TimingIso.gif");
  delete myCanvas;
  delete timingSig;
  delete timingBkg;

  timingSig = new TH1F("timingSig", "Timing of Single Photon Filters in Signal Events", 6, 0, 6);
  timingSig->SetCanExtend(TH1::kAllAxes);
  timingSig->SetStats(0);
  delete l1MatchTiming; l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",sigEvents);
  delete EtTiming; EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",sigEvents);
  TTreeFormula *IEcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.IEcal",sigEvents);
  TTreeFormula *PhotIHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotIHcal",sigEvents);
  TTreeFormula *PhotItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotItrack",sigEvents);
  event = 0;
  avgL1Match = 0.;
  avgEt = 0.;
  Double_t avgIEcal = 0.;
  Double_t avgPhotIHcal = 0.;
  Double_t avgPhotItrack = 0.;
  for (event = 0; event < sigEvents->GetEntries(); event++) {
    sigEvents->LoadTree(event);
    avgL1Match = (event*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgEt = (event*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgIEcal = (event*avgIEcal + IEcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgPhotIHcal = (event*avgPhotIHcal + PhotIHcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    avgPhotItrack = (event*avgPhotItrack + PhotItrackTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
  }
  timingSig->Fill("L1 Match", avgL1Match);
  timingSig->Fill("Et", avgEt); 
  timingSig->Fill("IEcal", avgIEcal); 
  timingSig->Fill("IHcal", avgPhotIHcal); 
  timingSig->Fill("Itrack", avgPhotItrack); 
  timingSig->LabelsDeflate("X");
  timingSig->LabelsOption("v");

  timingBkg = new TH1F("timingBkg", "Timing of Single Photon Filters in Background Events", 6, 0, 6);
  timingBkg->SetCanExtend(TH1::kAllAxes);
  timingBkg->SetStats(0);
  avgL1Match = 0.;
  avgEt = 0.;
  avgIEcal = 0.;
  avgPhotIHcal = 0.;
  avgPhotItrack = 0.;
  for (xSecNum = 0; xSecNum < bkgEvents.size(); xSecNum++) {
    delete l1MatchTiming; l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",bkgEvents[xSecNum].first);
    delete EtTiming; EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",bkgEvents[xSecNum].first);
    delete IEcalTiming; IEcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.IEcal",bkgEvents[xSecNum].first);
    delete PhotIHcalTiming; PhotIHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotIHcal",bkgEvents[xSecNum].first);
    delete PhotItrackTiming; PhotItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotItrack",bkgEvents[xSecNum].first);
    event = 0;
    for (event = 0; event < bkgEvents[xSecNum].first->GetEntries(); event++) {
      bkgEvents[xSecNum].first->LoadTree(event);
      avgL1Match = (event*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgEt = (event*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgIEcal = (event*avgIEcal + IEcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgPhotIHcal = (event*avgPhotIHcal + PhotIHcalTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
      avgPhotItrack = (event*avgPhotItrack + PhotItrackTiming->EvalInstance(0))/ ((Double_t) (event+1)); 
    }
  }
  timingBkg->Fill("L1 Match", avgL1Match);
  timingBkg->Fill("Et", avgEt); 
  timingBkg->Fill("IEcal", avgIEcal); 
  timingBkg->Fill("IHcal", avgPhotIHcal); 
  timingBkg->Fill("Itrack", avgPhotItrack); 
  timingBkg->LabelsDeflate("X");
  timingBkg->LabelsOption("v");
    
  myCanvas = new TCanvas("myCanvas", "Timing vs. Filter for Isolated Photon Filters", 1000, 500);
  myCanvas->Divide(2,1);
  myCanvas->cd(1);
  timingSig->Draw();
  myCanvas->cd(2);
  timingBkg->Draw();
  myCanvas->Print("images/TimingIsoPhot.gif");
}
