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
  Double_t hltCut;
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

  // Filters
  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 15.;
  thisFilter.maxCut = 60.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 3.;
  thisFilter.maxCut = 6.;
  filters.push_back(thisFilter);
  thisFilter.name = "pixMatch";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 0;
  thisFilter.maxCut = 0;
  filters.push_back(thisFilter);
  thisFilter.name = "Eoverp";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltCut = 2.45;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltCut = 0.06;
  thisFilter.maxCut = 0.24;
  filters.push_back(thisFilter);
  pathNames.push_back("ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.");
  pathNames.push_back("ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.");
  pathNames.push_back("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 10.;
  thisFilter.maxCut = 40.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 9.;
  thisFilter.maxCut = 18.;
  filters.push_back(thisFilter);
  thisFilter.name = "pixMatch";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 0;
  thisFilter.maxCut = 0;
  filters.push_back(thisFilter);
  thisFilter.name = "Eoverp";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltCut = 24500;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltCut = 0.4;
  thisFilter.maxCut = 0.12;
  filters.push_back(thisFilter);
  pathNames.push_back("ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.");
  pathNames.push_back("ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.");
  pathNames.push_back("ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 30.;
  thisFilter.maxCut = 60.;
  filters.push_back(thisFilter);
  thisFilter.name = "IEcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 1.5;
  thisFilter.maxCut = 6.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 6.;
  thisFilter.maxCut = 12.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 1;
  thisFilter.maxCut = 5;
  filters.push_back(thisFilter);
  pathNames.push_back("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 1;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear();
  filters.clear();

  thisFilter.name = "l1Match";
  thisFilter.pathNum = 0;
  thisFilter.direction =  0;
  thisFilter.hltCut = 0.;
  thisFilter.maxCut = 0.;
  filters.push_back(thisFilter);
  thisFilter.name = "Et";
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltCut = 20.;
  thisFilter.maxCut = 40.;
  filters.push_back(thisFilter);
  thisFilter.name = "IEcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 2.5;
  thisFilter.maxCut = 5.;
  filters.push_back(thisFilter);
  thisFilter.name = "IHcal";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 8.;
  thisFilter.maxCut = 24.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltCut = 3;
  thisFilter.maxCut = 6;
  filters.push_back(thisFilter);
  pathNames.push_back("PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.");
  thisPath.names = pathNames;
  thisPath.nCandsCut = 2;
  thisPath.filters = filters;
  paths.push_back(thisPath);
  pathNames.clear(); 
  pathNames.push_back("PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.");
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
  TString baseCutText;
  TString cutBasePT1;
  TString cutBasePT2;
  TString cutBase1;
  TString cutBase2;
  cutBasePT1 = "ElecHLTCutVarsPreTracks_hltCutVars_";
  cutBasePT2 = "ElecsPT_EGAMMAHLT.obj.";
  cutBase1 = "ElecHLTCutVarss_hltCutVars_";
  cutBase2 = "Elecs_EGAMMAHLT.obj.";

  std::vector<std::vector<TGraphErrors> > EffVBkg;
  std::vector<TGraphErrors> pathEffVBkgs;
  TGraphErrors filterEffVBkgs(nCuts);
  for (pathNum = 0; pathNum < paths.size(); pathNum++) {
    for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
      //      filterEffVBkgs = new TGraphErrors(nCuts);
      pathEffVBkgs.push_back(filterEffVBkgs);
    }
    EffVBkg.push_back(pathEffVBkgs);
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
  for (cutNum = 0; cutNum < nCuts; cutNum++) {
    cout<<"Cut "<<cutNum<<endl;
  
    for (pathNum = 0; pathNum < paths.size(); pathNum++) {
      for (filterNum = 0; filterNum < (paths[pathNum].filters).size(); filterNum++) {
	if ((paths[pathNum].filters)[filterNum].maxCut != 0.) {
	  cutText = "(Sum$(";
	  for (oldFilterNum = 0; oldFilterNum < filterNum; oldFilterNum++) { 
	    cutText += (paths[pathNum].names)[(paths[pathNum].filters)[filterNum].pathNum];
	    cutText += (paths[pathNum].filters)[oldFilterNum].name;
	    switch ((paths[pathNum].filters)[oldFilterNum].direction) {
	    case -1:
	      cutText += " < ";
	      cutText += (paths[pathNum].filters)[oldFilterNum].hltCut;
	      break;
	    case 0:
	      break;
	    case 1:
	      cutText += " > ";
	      cutText += (paths[pathNum].filters)[oldFilterNum].hltCut;
	      break;
	    default:
	      cout<<"Invalid value of direction in "<<paths[pathNum].names[(paths[pathNum].filters)[filterNum].pathNum]<<(paths[pathNum].filters)[oldFilterNum].name<<endl;
	      break;
	    }
	    if (oldFilterNum != filterNum - 1) cutText += " && ";
	  }
	  baseCutText = cutText;
	  baseCutText += ") >= ";
	  baseCutText += paths[pathNum].nCandsCut;
	  baseCutText += ")";
	  cutText += " && ";
	  cutText += paths[pathNum].names[(paths[pathNum].filters)[filterNum].pathNum];
	  cutText += (paths[pathNum].filters)[filterNum].name;
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
	    cout<<"Invalid value of direction in "<<paths[pathNum].names[(paths[pathNum].filters)[filterNum].pathNum]<<(paths[pathNum].filters)[oldFilterNum].name<<endl;
	    break;
	  }
	  cutText += ") >= ";
	  cutText += paths[pathNum].nCandsCut;
	  cutText += ")";
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
    tempPathName.Replace(0, tempPathName.Index("_", 1, 0, TString::kExact), "", 0);
    tempPathName.Remove(TString::kLeading, '_');
    tempPathName.Replace(0, tempPathName.Index("_", 1, 0, TString::kExact), "", 0);
    tempPathName.Remove(TString::kLeading, '_');
    tempPathName.Resize(tempPathName.Index("_", 1, 0, TString::kExact));
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
  }
}
    /*
    cut = (Double_t)i / 120. * 60.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    cout<<rate<<", "<<err<<", ";
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.l1Match) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));
    cout<<eff<<", "<<err<<endl;

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.Et > 15.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 6.;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Eoverp < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Et > 15. &&  ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.pixMatch >= 1) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 0.24;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Eoverp < 2.45 && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Et > 15. &&  ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_SingleElecs_EGAMMAHLT.obj.Eoverp < 2.45) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 90.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 4.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < 1.5) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    if (i == 0) {
      for (j = 0; j < sigPass.size(); j++) {
        sSigMin.push_back(sigPass[j].first);
        sSigMax.push_back(sigPass[j].first);
        sBkgMin.push_back(bkgPass[j].first);
        sBkgMax.push_back(bkgPass[j].first);
      }
    }
    else {
      for (j = 0; j < sigPass.size(); j++) {
        if (sSigMin[j] > sigPass[j].first) sSigMin[j] = sigPass[j].first;
        if (sSigMax[j] < sigPass[j].first) sSigMax[j] = sigPass[j].first;
        if (sBkgMin[j] > bkgPass[j].first) sBkgMin[j] = bkgPass[j].first;
        if (sBkgMax[j] < bkgPass[j].first) sBkgMax[j] = bkgPass[j].first;
      }
    }
    sBkgRate.push_back(bkgPass);
    sSigEff.push_back(sigPass);

    bkgPass.clear();
    sigPass.clear();

    cut = (Double_t)i / 120. * 60.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));
    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
   if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
   }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > 15.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));
    cut = (Double_t)i / 120. * 6.;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Eoverp < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("", "(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.pixMatch >= 1) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));
    cut = (Double_t)i / 120. * 0.24;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Et > 15. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Eoverp < 2.45 && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Et > 15. &&  ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.IHcal < 3. && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Eoverp < 2.45) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 90.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 4.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 20.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < 1.5) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    if (i == 0) {
      for (j = 0; j < sigPass.size(); j++) {
        rsSigMin.push_back(sigPass[j].first);
        rsSigMax.push_back(sigPass[j].first);
        rsBkgMin.push_back(bkgPass[j].first);
        rsBkgMax.push_back(bkgPass[j].first);
      }
    }
    else {
      for (j = 0; j < sigPass.size(); j++) {
        if (rsSigMin[j] > sigPass[j].first) rsSigMin[j] = sigPass[j].first;
        if (rsSigMax[j] < sigPass[j].first) rsSigMax[j] = sigPass[j].first;
        if (rsBkgMin[j] > bkgPass[j].first) rsBkgMin[j] = bkgPass[j].first;
        if (rsBkgMax[j] < bkgPass[j].first) rsBkgMax[j] = bkgPass[j].first;
      }
    }

    rsBkgRate.push_back(bkgPass);
    rsSigEff.push_back(sigPass);

    bkgPass.clear();
    sigPass.clear();

    cut = (Double_t)i / 120. * 60.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.l1Match) >= 2)");
    if (total != 0 ) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.Et > 10.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 6.;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Eoverp < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.pixMatch >= 1) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 0.24;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Eoverp < 24500. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_DoubleElecs_EGAMMAHLT.obj.Eoverp < 24500.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 90.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 4.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < 2.5) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    if (i == 0) {
      for (j = 0; j < sigPass.size(); j++) {
        dSigMin.push_back(sigPass[j].first);
        dSigMax.push_back(sigPass[j].first);
        dBkgMin.push_back(bkgPass[j].first);
        dBkgMax.push_back(bkgPass[j].first);
      }
    }
    else {
      for (j = 0; j < sigPass.size(); j++) {
        if (dSigMin[j] > sigPass[j].first) dSigMin[j] = sigPass[j].first;
        if (dSigMax[j] < sigPass[j].first) dSigMax[j] = sigPass[j].first;
        if (dBkgMin[j] > bkgPass[j].first) dBkgMin[j] = bkgPass[j].first;
        if (dBkgMax[j] < bkgPass[j].first) dBkgMax[j] = bkgPass[j].first;
      }
    }

    dBkgRate.push_back(bkgPass);
    dSigEff.push_back(sigPass);

    bkgPass.clear();
    sigPass.clear();

    cut = (Double_t)i / 120. * 60.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.l1Match) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.Et > 10.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 6.;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Eoverp < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.pixMatch >= 1) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 0.24;
    cutText = "(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Eoverp < 24500. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText);
    total = bkgEvents->Draw("","");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.l1Match && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Et > 10. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.IHcal < 9. && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.pixMatch >= 1 && ElecHLTCutVarss_hltCutVars_RelaxedDoubleElecs_EGAMMAHLT.obj.Eoverp < 24500.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 90.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 4.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 30.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    cut = (Double_t)i / 120. * 12.;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IHcal < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    bkgPass.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < 2.5) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sigPass.push_back(make_pair(eff, err));

    if (i == 0) {
      for (j = 0; j < sigPass.size(); j++) {
        rdSigMin.push_back(sigPass[j].first);
        rdSigMax.push_back(sigPass[j].first);
        rdBkgMin.push_back(bkgPass[j].first);
        rdBkgMax.push_back(bkgPass[j].first);
      }
    }
    else {
      for (j = 0; j < sigPass.size(); j++) {
        if (rdSigMin[j] > sigPass[j].first) rdSigMin[j] = sigPass[j].first;
        if (rdSigMax[j] < sigPass[j].first) rdSigMax[j] = sigPass[j].first;
        if (rdBkgMin[j] > bkgPass[j].first) rdBkgMin[j] = bkgPass[j].first;
        if (rdBkgMax[j] < bkgPass[j].first) rdBkgMax[j] = bkgPass[j].first;
      }
    }
    
    rdBkgRate.push_back(bkgPass);
    rdSigEff.push_back(sigPass);
    
  }
  for (i = 0; i < 120; i++) {
    sEffVBkgEt->SetPoint(i, sBkgRate[i][0].first, sSigEff[i][0].first);
    sEffVBkgEt->SetPointError(i, sBkgRate[i][0].second, sSigEff[i][0].second);
  }
  TH2F *sEffVBkgEt = new TH2F("sEffVBkgEt", "Efficiency vs. Background in Single Electron Stream", 1000, sBkgMin[0], sBkgMax[0], 1000, sSigMin[0], sSigMax[0]);
  TH2F *rsEffVBkgEt = new TH2F("rsEffVBkgEt", "Efficiency vs. Background in Relaxed Single Electron Stream", 1000, rsBkgMin[0], rsBkgMax[0], 1000, rsSigMin[0], rsSigMax[0]);
  TH2F *dEffVBkgEt = new TH2F("dEffVBkgEt", "Efficiency vs. Background in Double Electron Stream", 1000, dBkgMin[0], dBkgMax[0], 1000, dSigMin[0], dSigMax[0]);
  TH2F *rdEffVBkgEt = new TH2F("rdEffVBkgEt", "Efficiency vs. Background in Relaxed Double Electron Stream", 1000, rdBkgMin[0], rdBkgMax[0], 1000, rdSigMin[0], rdSigMax[0]);
  for (i = 0; i < 120; i++) {
    sEffVBkgEt->Fill(sBkgRate[i][0].first, sSigEff[i][0].first);
    rsEffVBkgEt->Fill(rsBkgRate[i][0].first, rsSigEff[i][0].first);
    dEffVBkgEt->Fill(dBkgRate[i][0].first, dSigEff[i][0].first);
    rdEffVBkgEt->Fill(rdBkgRate[i][0].first, rdSigEff[i][0].first);
  }

  TH2F *sEffVBkgIHcal = new TH2F("sEffVBkgIHcal", "Efficiency vs. Background in Single Electron Stream", 1000, sBkgMin[1], sBkgMax[1], 1000, sSigMin[1], sSigMax[1]);
  TH2F *rsEffVBkgIHcal = new TH2F("rsEffVBkgIHcal", "Efficiency vs. Background in Relaxed Single Electron Stream", 1000, rsBkgMin[1], rsBkgMax[1], 1000, rsSigMin[1], rsSigMax[1]);
  TH2F *dEffVBkgIHcal = new TH2F("dEffVBkgIHcal", "Efficiency vs. Background in Double Electron Stream", 1000, dBkgMin[1], dBkgMax[1], 1000, dSigMin[1], dSigMax[1]);
  TH2F *rdEffVBkgIHcal = new TH2F("rdEffVBkgIHcal", "Efficiency vs. Background in Relaxed Double Electron Stream", 1000, rdBkgMin[1], rdBkgMax[1], 1000, rdSigMin[1], rdSigMax[1]);
  for (i = 0; i < 120; i++) {
    sEffVBkgIHcal->Fill(sBkgRate[i][1].first, sSigEff[i][1].first);
    rsEffVBkgIHcal->Fill(rsBkgRate[i][1].first, rsSigEff[i][1].first);
    dEffVBkgIHcal->Fill(dBkgRate[i][1].first, dSigEff[i][1].first);
    rdEffVBkgIHcal->Fill(rdBkgRate[i][1].first, rdSigEff[i][1].first);
  }

  TH2F *sEffVBkgEoverp = new TH2F("sEffVBkgEoverp", "Efficiency vs. Background in Single Electron Stream", 1000, sBkgMin[2], sBkgMax[2], 1000, sSigMin[2], sSigMax[2]);
  TH2F *rsEffVBkgEoverp = new TH2F("rsEffVBkgEoverp", "Efficiency vs. Background in Relaxed Single Electron Stream", 1000, rsBkgMin[2], rsBkgMax[2], 1000, rsSigMin[2], rsSigMax[2]);
  TH2F *dEffVBkgEoverp = new TH2F("dEffVBkgEoverp", "Efficiency vs. Background in Double Electron Stream", 1000, dBkgMin[2], dBkgMax[2], 1000, dSigMin[2], dSigMax[2]);
  TH2F *rdEffVBkgEoverp = new TH2F("rdEffVBkgEoverp", "Efficiency vs. Background in Relaxed Double Electron Stream", 1000, rdBkgMin[2], rdBkgMax[2], 1000, rdSigMin[2], rdSigMax[2]);
  for (i = 0; i < 120; i++) {
    sEffVBkgEoverp->Fill(sBkgRate[i][2].first, sSigEff[i][2].first);
    rsEffVBkgEoverp->Fill(rsBkgRate[i][2].first, rsSigEff[i][2].first);
    dEffVBkgEoverp->Fill(dBkgRate[i][2].first, dSigEff[i][2].first);
    rdEffVBkgEoverp->Fill(rdBkgRate[i][2].first, rdSigEff[i][2].first);
  }

  TH2F *sEffVBkgItrack = new TH2F("sEffVBkgItrack", "Efficiency vs. Background in Single Electron Stream", 1000, sBkgMin[3], sBkgMax[3], 1000, sSigMin[3], sSigMax[3]);
  TH2F *rsEffVBkgItrack = new TH2F("rsEffVBkgItrack", "Efficiency vs. Background in Relaxed Single Electron Stream", 1000, rsBkgMin[3], rsBkgMax[3], 1000, rsSigMin[3], rsSigMax[3]);
  TH2F *dEffVBkgItrack = new TH2F("dEffVBkgItrack", "Efficiency vs. Background in Double Electron Stream", 1000, dBkgMin[3], dBkgMax[3], 1000, dSigMin[3], dSigMax[3]);
  TH2F *rdEffVBkgItrack = new TH2F("rdEffVBkgItrack", "Efficiency vs. Background in Relaxed Double Electron Stream", 1000, rdBkgMin[3], rdBkgMax[3], 1000, rdSigMin[3], rdSigMax[3]);
  for (i = 0; i < 120; i++) {
    sEffVBkgItrack->Fill(sBkgRate[i][3].first, sSigEff[i][3].first);
    rsEffVBkgItrack->Fill(rsBkgRate[i][3].first, rsSigEff[i][3].first);
    dEffVBkgItrack->Fill(dBkgRate[i][3].first, dSigEff[i][3].first);
    rdEffVBkgItrack->Fill(rdBkgRate[i][3].first, rdSigEff[i][3].first);
  }

  TH2F *sEffVBkgEtPhot = new TH2F("sEffVBkgEtPhot", "Efficiency vs. Background in Single Photon Stream", 1000, sBkgMin[4], sBkgMax[4], 1000, sSigMin[4], sSigMax[4]);
  TH2F *rsEffVBkgEtPhot = new TH2F("rsEffVBkgEtPhot", "Efficiency vs. Background in Relaxed Single Photon Stream", 1000, rsBkgMin[4], rsBkgMax[4], 1000, rsSigMin[4], rsSigMax[4]);
  TH2F *dEffVBkgEtPhot = new TH2F("dEffVBkgEtPhot", "Efficiency vs. Background in Double Photon Stream", 1000, dBkgMin[4], dBkgMax[4], 1000, dSigMin[4], dSigMax[4]);
  TH2F *rdEffVBkgEtPhot = new TH2F("rdEffVBkgEtPhot", "Efficiency vs. Background in Relaxed Double Photon Stream", 1000, rdBkgMin[4], rdBkgMax[4], 1000, rdSigMin[4], rdSigMax[4]);
  for (i = 0; i < 120; i++) {
    sEffVBkgEtPhot->Fill(sBkgRate[i][4].first, sSigEff[i][4].first);
    rsEffVBkgEtPhot->Fill(rsBkgRate[i][4].first, rsSigEff[i][4].first);
    dEffVBkgEtPhot->Fill(dBkgRate[i][4].first, dSigEff[i][4].first);
    rdEffVBkgEtPhot->Fill(rdBkgRate[i][4].first, rdSigEff[i][4].first);
  }

  TH2F *sEffVBkgIEcalPhot = new TH2F("sEffVBkgIEcalPhot", "Efficiency vs. Background in Single Photon Stream", 1000, sBkgMin[5], sBkgMax[5], 1000, sSigMin[5], sSigMax[5]);
  TH2F *rsEffVBkgIEcalPhot = new TH2F("rsEffVBkgIEcalPhot", "Efficiency vs. Background in Relaxed Single Photon Stream", 1000, rsBkgMin[5], rsBkgMax[5], 1000, rsSigMin[5], rsSigMax[5]);
  TH2F *dEffVBkgIEcalPhot = new TH2F("dEffVBkgIEcalPhot", "Efficiency vs. Background in Double Photon Stream", 1000, dBkgMin[5], dBkgMax[5], 1000, dSigMin[5], dSigMax[5]);
  TH2F *rdEffVBkgIEcalPhot = new TH2F("rdEffVBkgIEcalPhot", "Efficiency vs. Background in Relaxed Double Photon Stream", 1000, rdBkgMin[5], rdBkgMax[5], 1000, rdSigMin[5], rdSigMax[5]);
  for (i = 0; i < 120; i++) {
    sEffVBkgIEcalPhot->Fill(sBkgRate[i][5].first, sSigEff[i][5].first);
    rsEffVBkgIEcalPhot->Fill(rsBkgRate[i][5].first, rsSigEff[i][5].first);
    dEffVBkgIEcalPhot->Fill(dBkgRate[i][5].first, dSigEff[i][5].first);
    rdEffVBkgIEcalPhot->Fill(rdBkgRate[i][5].first, rdSigEff[i][5].first);
  }

  TH2F *sEffVBkgIHcalPhot = new TH2F("sEffVBkgIHcalPhot", "Efficiency vs. Background in Single Photon Stream", 1000, sBkgMin[6], sBkgMax[6], 1000, sSigMin[6], sSigMax[6]);
  TH2F *rsEffVBkgIHcalPhot = new TH2F("rsEffVBkgIHcalPhot", "Efficiency vs. Background in Relaxed Single Photon Stream", 1000, rsBkgMin[6], rsBkgMax[6], 1000, rsSigMin[6], rsSigMax[6]);
  TH2F *dEffVBkgIHcalPhot = new TH2F("dEffVBkgIHcalPhot", "Efficiency vs. Background in Double Photon Stream", 1000, dBkgMin[6], dBkgMax[6], 1000, dSigMin[6], dSigMax[6]);
  TH2F *rdEffVBkgIHcalPhot = new TH2F("rdEffVBkgIHcalPhot", "Efficiency vs. Background in Relaxed Double Photon Stream", 1000, rdBkgMin[6], rdBkgMax[6], 1000, rdSigMin[6], rdSigMax[6]);

  for (i = 0; i < 120; i++) {
    sEffVBkgIHcalPhot->Fill(sBkgRate[i][6].first, sSigEff[i][6].first);
    rsEffVBkgIHcalPhot->Fill(rsBkgRate[i][6].first, rsSigEff[i][6].first);
    dEffVBkgIHcalPhot->Fill(dBkgRate[i][6].first, dSigEff[i][6].first);
    rdEffVBkgIHcalPhot->Fill(rdBkgRate[i][6].first, rdSigEff[i][6].first);
  }

  std::vector<std::pair<Double_t,Double_t> > sSigEffP;
  std::vector<std::pair<Double_t,Double_t> > rsSigEffP;
  std::vector<std::pair<Double_t,Double_t> > dSigEffP;
  std::vector<std::pair<Double_t,Double_t> > rdSigEffP;
  std::vector<std::pair<Double_t,Double_t> > sBkgRateP;
  std::vector<std::pair<Double_t,Double_t> > rsBkgRateP;
  std::vector<std::pair<Double_t,Double_t> > dBkgRateP;
  std::vector<std::pair<Double_t,Double_t> > rdBkgRateP;
  Double_t sSigMinP = 0.;
  Double_t rsSigMinP = 0.;
  Double_t dSigMinP = 0.;
  Double_t rdSigMinP = 0.;
  Double_t sSigMaxP = 0.;
  Double_t rsSigMaxP = 0.;
  Double_t dSigMaxP = 0.;
  Double_t rdSigMaxP = 0.;
  Double_t sBkgMinP = 0.;
  Double_t rsBkgMinP = 0.;
  Double_t dBkgMinP = 0.;
  Double_t rdBkgMinP = 0.;
  Double_t sBkgMaxP = 0.;
  Double_t rsBkgMaxP = 0.;
  Double_t dBkgMaxP = 0.;
  Double_t rdBkgMaxP = 0.;

  for (i = 0; i < 6; i++) {
    cut = i + 1;
    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < 6. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < 6.) >= 1)");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    sBkgRateP.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < 6.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    sSigEffP.push_back(make_pair(eff, err));

    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IHcal < 6. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 1)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IHcal < 6.) >= 1)");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    rsBkgRateP.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.Et > 30. && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IEcal < 1.5 && PhotHLTCutVarss_hltCutVars_RelaxedSinglePhots_EGAMMAHLT.obj.IHcal < 6.) >= 1)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    rsSigEffP.push_back(make_pair(eff, err));

    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IHcal < 8. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IHcal < 8.) >= 2)");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    dBkgRateP.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_DoublePhots_EGAMMAHLT.obj.IHcal < 8.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    dSigEffP.push_back(make_pair(eff, err));

    cutText = "(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IHcal < 8. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Itrack < ";
    cutText += cut;
    cutText += ") >= 2)";
    pass = bkgEvents->Draw("",cutText); 
    total = bkgEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IHcal < 8.) >= 2)");
    if (total != 0) { 
      eff = (Double_t)pass / (Double_t)total;
      rate = eff * xSection * lumi * conversion;
      err = sqrt(eff*(1.-eff) / (Double_t)total) * xSection * lumi * conversion;
    }
    else {
      rate = 0.;
      err = 0.;
    }
    rdBkgRateP.push_back(make_pair(rate, err));
    pass = sigEvents->Draw("",cutText);
    total = sigEvents->Draw("","(Sum$(PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.l1Match && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.Et > 20. && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IEcal < 2.5 && PhotHLTCutVarss_hltCutVars_RelaxedDoublePhots_EGAMMAHLT.obj.IHcal < 8.) >= 2)");
    if (total != 0) {
      eff = (Double_t)pass / (Double_t)total;
      err = sqrt(eff*(1.-eff) / (Double_t)total);
    }
    else {
      eff = 0.;
      err = 0.;
    }
    rdSigEffP.push_back(make_pair(eff, err));
    if (i == 0) {
      sSigMinP = sSigEffP[i].first;
      rsSigMinP = rsSigEffP[i].first;
      dSigMinP = dSigEffP[i].first;
      rdSigMinP = rdSigEffP[i].first;
      sSigMaxP = sSigEffP[i].first;
      rsSigMaxP = rsSigEffP[i].first;
      dSigMaxP = dSigEffP[i].first;
      rdSigMaxP = rdSigEffP[i].first;
      sBkgMinP = sBkgRateP[i].first;
      rsBkgMinP = rsBkgRateP[i].first;
      dBkgMinP = dBkgRateP[i].first;
      rdBkgMinP = rdBkgRateP[i].first;
      sBkgMaxP = sBkgRateP[i].first;
      rsBkgMaxP = rsBkgRateP[i].first;
      dBkgMaxP = dBkgRateP[i].first;
      rdBkgMaxP = rdBkgRateP[i].first;
    }
    else {
      if (sSigMinP > sSigEffP[i].first) sSigMinP = sSigEffP[i].first;
      if (rsSigMinP > rsSigEffP[i].first) rsSigMinP = rsSigEffP[i].first;
      if (dSigMinP > dSigEffP[i].first) dSigMinP = dSigEffP[i].first;
      if (rdSigMinP > rdSigEffP[i].first) rdSigMinP = rdSigEffP[i].first;
      if (sSigMaxP < sSigEffP[i].first) sSigMaxP = sSigEffP[i].first;
      if (rsSigMaxP < rsSigEffP[i].first) rsSigMaxP = rsSigEffP[i].first;
      if (dSigMaxP < dSigEffP[i].first) dSigMaxP = dSigEffP[i].first;
      if (rdSigMaxP < rdSigEffP[i].first) rdSigMaxP = rdSigEffP[i].first;
      if (sBkgMinP > sBkgRateP[i].first) sBkgMinP = sBkgRateP[i].first;
      if (rsBkgMinP > rsBkgRateP[i].first) rsBkgMinP = rsBkgRateP[i].first;
      if (dBkgMinP > dBkgRateP[i].first) dBkgMinP = dBkgRateP[i].first;
      if (rdBkgMinP > rdBkgRateP[i].first) rdBkgMinP = rdBkgRateP[i].first;
      if (sBkgMaxP < sBkgRateP[i].first) sBkgMaxP = sBkgRateP[i].first;
      if (rsBkgMaxP < rsBkgRateP[i].first) rsBkgMaxP = rsBkgRateP[i].first;
      if (dBkgMaxP < dBkgRateP[i].first) dBkgMaxP = dBkgRateP[i].first;
      if (rdBkgMaxP < rdBkgRateP[i].first) rdBkgMaxP = rdBkgRateP[i].first;
    }

  TH2F *sEffVBkgItrackPhot = new TH2F("sEffVBkgItrackPhot", "Efficiency vs. Background in Single Photon Stream", 1000, sBkgMinP, sBkgMaxP, 1000, sSigMinP, sSigMaxP);
  TH2F *rsEffVBkgItrackPhot = new TH2F("rsEffVBkgItrackPhot", "Efficiency vs. Background in Relaxed Single Photon Stream", 1000, rsBkgMinP, rsBkgMaxP, 1000, rsSigMinP, rsSigMaxP);
  TH2F *dEffVBkgItrackPhot = new TH2F("dEffVBkgItrackPhot", "Efficiency vs. Background in Double Photon Stream", 1000, dBkgMinP, dBkgMaxP, 1000, dSigMinP, dSigMaxP);
  TH2F *rdEffVBkgItrackPhot = new TH2F("rdEffVBkgItrackPhot", "Efficiency vs. Background in Relaxed Double Photon Stream", 1000, rdBkgMinP, rdBkgMaxP, 1000, rdSigMinP, rdSigMaxP);

  for (i = 0; i < 6; i++) {
    sEffVBkgItrackPhot->Fill(sBkgRateP[i].first, sSigEffP[i].first);
    rsEffVBkgItrackPhot->Fill(rsBkgRateP[i].first, rsSigEffP[i].first);
    dEffVBkgItrackPhot->Fill(dBkgRateP[i].first, dSigEffP[i].first);
    rdEffVBkgItrackPhot->Fill(rdBkgRateP[i].first, rdSigEffP[i].first);
  }

  gStyle->SetOptStat(0000000);                                
  TCanvas *myCanvas = new TCanvas("myCanvas", "Efficiency vs. Background", 1000, 1000);
  myCanvas->Divide(2,2);
  myCanvas->cd(1);
  sEffVBkgEt->Draw();
  myCanvas->cd(2);
  rsEffVBkgEt->Draw();
  myCanvas->cd(3);
  dEffVBkgEt->Draw();
  myCanvas->cd(4);
  rdEffVBkgEt->Draw();
  myCanvas->Print("images/EffvBkgEt.gif");
  myCanvas->cd(1);
  sEffVBkgIHcal->Draw();
  myCanvas->cd(2);
  rsEffVBkgIHcal->Draw();
  myCanvas->cd(3);
  dEffVBkgIHcal->Draw();
  myCanvas->cd(4);
  rdEffVBkgIHcal->Draw();
  myCanvas->Print("images/EffvBkgIHcal.gif");
  myCanvas->cd(1);
  sEffVBkgEoverp->Draw();
  myCanvas->cd(2);
  rsEffVBkgEoverp->Draw();
  myCanvas->cd(3);
  dEffVBkgEoverp->Draw();
  myCanvas->cd(4);
  rdEffVBkgEoverp->Draw();
  myCanvas->Print("images/EffvBkgEoverp.gif");
  myCanvas->cd(1);
  sEffVBkgItrack->Draw();
  myCanvas->cd(2);
  rsEffVBkgItrack->Draw();
  myCanvas->cd(3);
  dEffVBkgItrack->Draw();
  myCanvas->cd(4);
  rdEffVBkgItrack->Draw();
  myCanvas->Print("images/EffvBkgItrack.gif");
  myCanvas->cd(1);
  sEffVBkgEtPhot->Draw();
  myCanvas->cd(2);
  rsEffVBkgEtPhot->Draw();
  myCanvas->cd(3);
  dEffVBkgEtPhot->Draw();
  myCanvas->cd(4);
  rdEffVBkgEtPhot->Draw();
  myCanvas->Print("images/EffvBkgEtPhot.gif");
  myCanvas->cd(1);
  sEffVBkgIEcalPhot->Draw();
  myCanvas->cd(2);
  rsEffVBkgIEcalPhot->Draw();
  myCanvas->cd(3);
  dEffVBkgIEcalPhot->Draw();
  myCanvas->cd(4);
  rdEffVBkgIEcalPhot->Draw();
  myCanvas->Print("images/EffvBkgIEcalPhot.gif");
  myCanvas->cd(1);
  sEffVBkgIHcalPhot->Draw();
  myCanvas->cd(2);
  rsEffVBkgIHcalPhot->Draw();
  myCanvas->cd(3);
  dEffVBkgIHcalPhot->Draw();
  myCanvas->cd(4);
  rdEffVBkgIHcalPhot->Draw();
  myCanvas->Print("images/EffvBkgIHcalPhot.gif");
  myCanvas->cd(1);
  sEffVBkgItrackPhot->Draw();
  myCanvas->cd(2);
  rsEffVBkgItrackPhot->Draw();
  myCanvas->cd(3);
  dEffVBkgItrackPhot->Draw();
  myCanvas->cd(4);
  rdEffVBkgItrackPhot->Draw();
  myCanvas->Print("images/EffvBkgItrackPhot.gif");
  delete myCanvas;

  TH1F *timingSig = new TH1F("timingSig", "Timing of Single Electron Filters in Signal Events", 6, 0, 6);
  timingSig->SetBit(TH1::kCanRebin);
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
    avgL1Match = (i*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEt = (i*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgIHcal = (i*avgIHcal + IHcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPixMatch = (i*avgPixMatch + pixMatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEoverp = (i*avgEoverp + EoverpTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgItrack = (i*avgItrack + ItrackTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
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
  timingBkg->SetBit(TH1::kCanRebin);
  timingBkg->SetStats(0);
  delete l1MatchTiming; l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",bkgEvents);
  delete EtTiming; EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",bkgEvents);
  delete IHcalTiming; IHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecIHcal",bkgEvents);
  delete pixMatchTiming; pixMatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.pixMatch",bkgEvents);
  delete EoverpTiming; EoverpTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Eoverp",bkgEvents);
  delete ItrackTiming; ItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.ElecItrack",bkgEvents);
  event = 0;
  avgL1Match = 0.;
  avgEt = 0.;
  avgIHcal = 0.;
  avgPixMatch = 0.;
  avgEoverp = 0.;
  avgItrack = 0.;
  for (event = 0; event <  bkgEvents->GetEntries(); event++) {
    bkgEvents->LoadTree(event);
    avgL1Match = (i*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEt = (i*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgIHcal = (i*avgIHcal + IHcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPixMatch = (i*avgPixMatch + pixMatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEoverp = (i*avgEoverp + EoverpTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgItrack = (i*avgItrack + ItrackTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
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
  timingSig->SetBit(TH1::kCanRebin);
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
    avgL1Match = (i*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEt = (i*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgIEcal = (i*avgIEcal + IEcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPhotIHcal = (i*avgPhotIHcal + PhotIHcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPhotItrack = (i*avgPhotItrack + PhotItrackTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
  }
  timingSig->Fill("L1 Match", avgL1Match);
  timingSig->Fill("Et", avgEt); 
  timingSig->Fill("IEcal", avgIEcal); 
  timingSig->Fill("IHcal", avgPhotIHcal); 
  timingSig->Fill("Itrack", avgPhotItrack); 
  timingSig->LabelsDeflate("X");
  timingSig->LabelsOption("v");

  timingBkg = new TH1F("timingBkg", "Timing of Single Photon Filters in Background Events", 6, 0, 6);
  timingBkg->SetBit(TH1::kCanRebin);
  timingBkg->SetStats(0);
  delete l1MatchTiming; l1MatchTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.l1Match",bkgEvents);
  delete EtTiming; EtTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.Et",bkgEvents);
  delete IEcalTiming; IEcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.IEcal",bkgEvents);
  delete PhotIHcalTiming; PhotIHcalTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotIHcal",bkgEvents);
  delete PhotItrackTiming; PhotItrackTiming = new TTreeFormula("Timing","HLTTiming_hltCutVars_IsoTiming_EGAMMAHLT.obj.PhotItrack",bkgEvents);
  event = 0;
  avgL1Match = 0.;
  avgEt = 0.;
  avgIEcal = 0.;
  avgPhotIHcal = 0.;
  avgPhotItrack = 0.;
  for (event = 0; event < bkgEvents->GetEntries(); event++) {
    bkgEvents->LoadTree(event);
    avgL1Match = (i*avgL1Match + l1MatchTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgEt = (i*avgEt + EtTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgIEcal = (i*avgIEcal + IEcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPhotIHcal = (i*avgPhotIHcal + PhotIHcalTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
    avgPhotItrack = (i*avgPhotItrack + PhotItrackTiming->EvalInstance(0))/ ((Double_t) (i+1)); 
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
    */
