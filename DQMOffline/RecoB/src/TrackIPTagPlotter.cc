#include <cstddef>

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMOffline/RecoB/interface/TrackIPTagPlotter.h"

TrackIPTagPlotter::TrackIPTagPlotter(const TString & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, bool update, bool mc, bool wf) :
  BaseTagInfoPlotter(tagName, etaPtBin), willFinalize_(wf)
{

  mcPlots_ = mc;
  nBinEffPur_  = pSet.getParameter<int>("nBinEffPur");
  startEffPur_ = pSet.getParameter<double>("startEffPur");
  endEffPur_   = pSet.getParameter<double>("endEffPur");

  finalized = false;

  trkNbr3D = new FlavourHistograms<int>
	("selTrksNbr_3D" + theExtensionString, "Number of selected tracks for 3D IPS", 31, -0.5, 30.5,
	false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)) ,mc);

  trkNbr2D = new FlavourHistograms<int>
	("selTrksNbr_2D" + theExtensionString, "Number of selected tracks for 2D IPS", 31, -0.5, 30.5,
	false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)) ,mc );

  lowerIPSBound = -35.0;
  upperIPSBound = +35.0;

  lowerIPBound = -0.1;
  upperIPBound = 0.1;

  lowerIPEBound = 0;
  upperIPEBound = 0.04;

  // IP significance
  // 3D
  tkcntHistosSig3D[4] = new FlavourHistograms<double>
       ("ips_3D" + theExtensionString, "3D IP significance",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig3D[0] = new FlavourHistograms<double>
       ("ips1_3D" + theExtensionString, "3D IP significance 1.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)),mc) ;

  tkcntHistosSig3D[1] = new FlavourHistograms<double>
       ("ips2_3D" + theExtensionString, "3D IP significance 2.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig3D[2] = new FlavourHistograms<double>
       ("ips3_3D" + theExtensionString, "3D IP significance 3.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig3D[3] = new FlavourHistograms<double>
       ("ips4_3D" + theExtensionString, "3D IP significance 4.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  //2D
  tkcntHistosSig2D[4] = new FlavourHistograms<double>
       ("ips_2D" + theExtensionString, "2D IP significance",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig2D[0] = new FlavourHistograms<double>
       ("ips1_2D" + theExtensionString, "2D IP significance 1.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig2D[1] = new FlavourHistograms<double>
       ("ips2_2D" + theExtensionString, "2D IP significance 2.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig2D[2] = new FlavourHistograms<double>
       ("ips3_2D" + theExtensionString, "2D IP significance 3.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosSig2D[3] = new FlavourHistograms<double>
       ("ips4" + theExtensionString, "2D IP significance 4.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  // IP value
  //3D
  tkcntHistosVal3D[4] = new FlavourHistograms<double>
       ("ip_3D" + theExtensionString, "3D IP value",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal3D[0] = new FlavourHistograms<double>
       ("ip1_3D" + theExtensionString, "3D IP value 1.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal3D[1] = new FlavourHistograms<double>
       ("ip2_3D" + theExtensionString, "3D IP value 2.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal3D[2] = new FlavourHistograms<double>
       ("ip3_3D" + theExtensionString, "3D IP value 3.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal3D[3] = new FlavourHistograms<double>
       ("ip4_3D" + theExtensionString, "3D IP value 4.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  //2D
  tkcntHistosVal2D[4] = new FlavourHistograms<double>
       ("ip_2D" + theExtensionString, "2D IP value",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal2D[0] = new FlavourHistograms<double>
       ("ip1_2D" + theExtensionString, "2D IP value 1.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal2D[1] = new FlavourHistograms<double>
       ("ip2_2D" + theExtensionString, "2D IP value 2.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal2D[2] = new FlavourHistograms<double>
       ("ip3_2D" + theExtensionString, "2D IP value 3.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosVal2D[3] = new FlavourHistograms<double>
       ("ip4" + theExtensionString, "2D IP value 4.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;


  // IP error
  // 3D
  tkcntHistosErr3D[4] = new FlavourHistograms<double>
       ("ipe_3D" + theExtensionString, "3D IP error",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr3D[0] = new FlavourHistograms<double>
       ("ipe1_3D" + theExtensionString, "3D IP error 1.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr3D[1] = new FlavourHistograms<double>
       ("ipe2_3D" + theExtensionString, "3D IP error 2.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr3D[2] = new FlavourHistograms<double>
       ("ipe3_3D" + theExtensionString, "3D IP error 3.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr3D[3] = new FlavourHistograms<double>
       ("ipe4_3D" + theExtensionString, "3D IP error 4.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  //2D
  tkcntHistosErr2D[4] = new FlavourHistograms<double>
       ("ipe_2D" + theExtensionString, "2D IP error",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr2D[0] = new FlavourHistograms<double>
       ("ipe1_2D" + theExtensionString, "2D IP error 1.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr2D[1] = new FlavourHistograms<double>
       ("ipe2_2D" + theExtensionString, "2D IP error 2.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr2D[2] = new FlavourHistograms<double>
       ("ipe3_2D" + theExtensionString, "2D IP error 3.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosErr2D[3] = new FlavourHistograms<double>
       ("ipe4" + theExtensionString, "2D IP error 4.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;



  // probability
  tkcntHistosProb3D[4] = new FlavourHistograms<float>
       ("prob_3D" + theExtensionString, "3D IP probability",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb3D[0] = new FlavourHistograms<float>
       ("prob1_3D" + theExtensionString, "3D IP probability 1.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb3D[1] = new FlavourHistograms<float>
       ("prob2_3D" + theExtensionString, "3D IP probability 2.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb3D[2] = new FlavourHistograms<float>
       ("prob3_3D" + theExtensionString, "3D IP probability 3.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb3D[3] = new FlavourHistograms<float>
       ("prob4_3D" + theExtensionString, "3D IP probability 4.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb2D[4] = new FlavourHistograms<float>
       ("prob_2D" + theExtensionString, "2D IP probability",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb2D[0] = new FlavourHistograms<float>
       ("prob1_2D" + theExtensionString, "2D IP probability 1.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb2D[1] = new FlavourHistograms<float>
       ("prob2_2D" + theExtensionString, "2D IP probability 2.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb2D[2] = new FlavourHistograms<float>
       ("prob3_2D" + theExtensionString, "2D IP probability 3.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  tkcntHistosProb2D[3] = new FlavourHistograms<float>
       ("prob4" + theExtensionString, "2D IP probability 4.trk",
	50, -1.1, 1.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc) ;

  decayLengthValuHisto = new FlavourHistograms<double>
       ("decLen" + theExtensionString, "Decay Length",
	50, -5.0, 5.0, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc);
  jetDistanceValuHisto = new FlavourHistograms<double>
       ("jetDist" + theExtensionString, "JetDistance",
	50, -0.1, 0.1, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc);
  jetDistanceSignHisto = new FlavourHistograms<double>
       ("jetDistSign" + theExtensionString, "JetDistance significance",
	50, -100.0, 100.0, false, true, true, "b", update,std::string((const char *)("TrackIPPlots"+theExtensionString)), mc);


  if (willFinalize_) createPlotsForFinalize();

}


TrackIPTagPlotter::~TrackIPTagPlotter ()
{

  delete trkNbr3D;
  delete trkNbr2D;
  delete decayLengthValuHisto;
  delete jetDistanceValuHisto;
  delete jetDistanceSignHisto;

  for(int n=0; n <= 4; n++) {
    delete tkcntHistosSig2D[n];
    delete tkcntHistosSig3D[n];
    delete tkcntHistosVal2D[n];
    delete tkcntHistosVal3D[n];
    delete tkcntHistosErr2D[n];
    delete tkcntHistosErr3D[n];
    delete tkcntHistosProb2D[n];
    delete tkcntHistosProb3D[n];
  }
  if (finalized) {
    for(int n=0; n < 4; n++) delete effPurFromHistos[n];
  }
}


void TrackIPTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
	const int & jetFlavour)
{

  const reco::TrackIPTagInfo * tagInfo = 
	dynamic_cast<const reco::TrackIPTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackIPTagInfo. " << endl;
  }

  vector<reco::TrackIPTagInfo::TrackIPData> ip = tagInfo->impactParameterData();

  vector<float> prob2d, prob3d;
  if (tagInfo->hasProbabilities()) {
    prob2d = tagInfo->probabilities(0);	
    prob3d = tagInfo->probabilities(1);	
  }

  trkNbr3D->fill(jetFlavour, ip.size());
  trkNbr2D->fill(jetFlavour, ip.size());

  vector<std::size_t> sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
  for(unsigned int n=0; n < sortedIndices.size() && n < 4; n++) {
    tkcntHistosSig2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.significance());
    tkcntHistosVal2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.value());
    tkcntHistosErr2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.error());
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob2D);
  for(unsigned int n=0; n < sortedIndices.size() && n < 4; n++) {
    tkcntHistosProb2D[n]->fill(jetFlavour, prob2d[sortedIndices[n]]);
  }
  for(unsigned int n=sortedIndices.size(); n < 4; n++){
    tkcntHistosSig2D[n]->fill(jetFlavour, lowerIPSBound-1.0);
    tkcntHistosVal2D[n]->fill(jetFlavour, lowerIPBound-1.0);
    tkcntHistosErr2D[n]->fill(jetFlavour, lowerIPEBound-1.0);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
  for(unsigned int n=0; n < sortedIndices.size() && n < 4; n++) {
    tkcntHistosSig3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.significance());
    tkcntHistosVal3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.value());
    tkcntHistosErr3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.error());
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob3D);
  for(unsigned int n=0; n < sortedIndices.size() && n < 4; n++) {
    tkcntHistosProb3D[n]->fill(jetFlavour, prob3d[sortedIndices[n]]);
  }
  for(unsigned int n=sortedIndices.size(); n < 4; n++){
    tkcntHistosSig3D[n]->fill(jetFlavour, lowerIPSBound-1.0);
    tkcntHistosVal3D[n]->fill(jetFlavour, lowerIPBound-1.0);
    tkcntHistosErr3D[n]->fill(jetFlavour, lowerIPEBound-1.0);
  }
  for(unsigned int n=0; n < ip.size(); n++) {
    tkcntHistosSig2D[4]->fill(jetFlavour, ip[n].ip2d.significance());
    tkcntHistosVal2D[4]->fill(jetFlavour, ip[n].ip2d.value());
    tkcntHistosErr2D[4]->fill(jetFlavour, ip[n].ip2d.error());
    tkcntHistosProb2D[4]->fill(jetFlavour, prob2d[n]);
  }
  for(unsigned int n=0; n < ip.size(); n++) {
    tkcntHistosSig3D[4]->fill(jetFlavour, ip[n].ip3d.significance());
    tkcntHistosVal3D[4]->fill(jetFlavour, ip[n].ip3d.value());
    tkcntHistosErr3D[4]->fill(jetFlavour, ip[n].ip3d.error());
    tkcntHistosProb3D[4]->fill(jetFlavour, prob3d[n]);
  }
  GlobalPoint pv(tagInfo->primaryVertex()->position().x(),
                 tagInfo->primaryVertex()->position().y(),
                 tagInfo->primaryVertex()->position().z());
  for(unsigned int n=0; n < ip.size(); n++) {
    double decayLen = (ip[n].closestToJetAxis - pv).mag();
    decayLengthValuHisto->fill(jetFlavour, decayLen);
  }
  for(unsigned int n=0; n < ip.size(); n++) {
    jetDistanceValuHisto->fill(jetFlavour, ip[n].distanceToJetAxis);
  }

}

void TrackIPTagPlotter::createPlotsForFinalize (){
  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],std::string((const char *)("TrackIPPlots"+theExtensionString)), mcPlots_, 
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],std::string((const char *)("TrackIPPlots"+theExtensionString)), mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],std::string((const char *)("TrackIPPlots"+theExtensionString)), mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],std::string((const char *)("TrackIPPlots"+theExtensionString)), mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
}


void TrackIPTagPlotter::finalize ()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  for(int n=0; n < 4; n++) effPurFromHistos[n]->compute();
  finalized = true;
}

void TrackIPTagPlotter::psPlot(const TString & name)
{
  TString cName = "TrackIPPlots"+ theExtensionString;
  setTDRStyle()->cd();
  TCanvas canvas(cName, "TrackIPPlots"+ theExtensionString, 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print(name + cName + ".ps[");

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosSig3D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosSig3D[n]->plot();
  }

  canvas.Print(name + cName + ".ps");
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosProb3D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosProb3D[n]->plot();
  }

  canvas.Print(name + cName + ".ps");
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosSig2D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosSig2D[n]->plot();
  }

  canvas.Print(name + cName + ".ps");
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosProb2D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosProb2D[n]->plot();
  }

  if (finalized) {
    for(int n=0; n < 2; n++) {
      canvas.Print(name + cName + ".ps");
      canvas.Clear();
      canvas.Divide(2,3);
      canvas.cd(1);
      effPurFromHistos[0+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(2);
      effPurFromHistos[0+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(3);
      effPurFromHistos[0+n]->plot();
      canvas.cd(4);
      effPurFromHistos[1+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(5);
      effPurFromHistos[1+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(6);
      effPurFromHistos[1+n]->plot();
    }
  }

  canvas.Print(name + cName + ".ps");
  canvas.Clear();
  canvas.Divide(1,3);
  canvas.cd(1);
  jetDistanceValuHisto->plot();
  canvas.cd(2);
  jetDistanceSignHisto->plot();
  canvas.cd(3);
  decayLengthValuHisto->plot();


  canvas.Print(name + cName + ".ps");
  canvas.Print(name + cName + ".ps]");
}


void TrackIPTagPlotter::epsPlot(const TString & name)
{
  trkNbr2D->epsPlot(name);
  trkNbr3D->epsPlot(name);
  decayLengthValuHisto->epsPlot(name);
  decayLengthSignHisto->epsPlot(name);
  jetDistanceValuHisto->epsPlot(name);
  jetDistanceSignHisto->epsPlot(name);
  for(int n=0; n <= 4; n++) {
    tkcntHistosSig2D[n]->epsPlot(name);
    tkcntHistosSig3D[n]->epsPlot(name);
    tkcntHistosVal2D[n]->epsPlot(name);
    tkcntHistosVal3D[n]->epsPlot(name);
    tkcntHistosErr2D[n]->epsPlot(name);
    tkcntHistosErr3D[n]->epsPlot(name);
    tkcntHistosProb2D[n]->epsPlot(name);
    tkcntHistosProb3D[n]->epsPlot(name);
  }
  if (finalized) {
    for(int n=0; n < 4; n++) effPurFromHistos[n]->epsPlot(name);
  }
}
