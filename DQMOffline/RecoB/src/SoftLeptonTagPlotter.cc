#include "DQMOffline/RecoB/interface/SoftLeptonTagPlotter.h"
#include "DataFormats/TrackReco/interface/Track.h"

static const char* ordinal[] = { "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th" };

SoftLeptonTagPlotter::SoftLeptonTagPlotter(const TString & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, bool mc, bool update) :
    BaseTagInfoPlotter(tagName, etaPtBin)
{
  mcPlots_ = mc;
  for (int i = 0; i < s_leptons; i++) {
    m_leptonId[i] = new FlavourHistograms<double> (
						   Form("%s lepton %s", ordinal[i], "id" ),
						   "Lepton identification discriminaint",
						   60, -0.1, 1.1, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_ );
    m_leptonPt[i] = new FlavourHistograms<double> (
						   Form( "%s lepton %s", ordinal[i], "pT" ),
						   "Lepton transverse moementum",
						   100, 0.0, 20.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_sip2d[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "sip2d" ),
        "Lepton signed 2D impact parameter significance",
        100, -20.0, 30.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_sip3d[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "sip3d" ),
        "Lepton signed 3D impact parameter significance",
        100, -20.0, 30.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_ptRel[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "pT rel" ),
        "Lepton transverse moementum relative to jet axis",
        100, 0.0, 10.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_p0Par[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "p0 par" ),
        "Lepton moementum along jet axis in the B rest frame",
        100, 0.0, 10.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_etaRel[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "eta rel" ),
        "Lepton pseudorapidity relative to jet axis",
        100, -5.0, 25.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_deltaR[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "delta R" ),
        "Lepton pseudoangular distance from jet axis",
        100, 0.0, 0.6, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_ratio[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "energy ratio" ),
        "Ratio of lepton momentum to jet energy",
        100, 0.0, 2.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
    m_ratioRel[i] = new FlavourHistograms<double> (
        Form( "%s lepton %s", ordinal[i], "parallel energy ratio" ),
        "Ratio of lepton momentum along the jet axis to jet energy",
        100, 0.0, 2.0, false, false, true, "b", update,std::string((const char *)("SoftLepton"+theExtensionString)),mcPlots_);
  }
}

SoftLeptonTagPlotter::~SoftLeptonTagPlotter ()
{
  for (int i = 0; i < s_leptons; i++) {
    delete m_leptonId[i];
    delete m_leptonPt[i];
    delete m_sip2d[i];
    delete m_sip3d[i];
    delete m_ptRel[i];
    delete m_p0Par[i];
    delete m_etaRel[i];
    delete m_deltaR[i];
    delete m_ratio[i];
    delete m_ratioRel[i];
  }
}

void SoftLeptonTagPlotter::analyzeTag( const reco::BaseTagInfo * baseTagInfo,
    const int & jetFlavour )
{

  const reco::SoftLeptonTagInfo * tagInfo = 
	dynamic_cast<const reco::SoftLeptonTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type SoftLeptonTagInfo. " << endl;
  }

  int n_leptons = tagInfo->leptons();

  for (int i = 0; i < n_leptons && i < s_leptons; i++) {
    m_leptonPt[i]->fill( jetFlavour, tagInfo->lepton(i)->pt() );
    m_leptonId[i]->fill( jetFlavour, tagInfo->properties(i).quality() );
    m_sip2d[i]->fill(    jetFlavour, tagInfo->properties(i).sip2d );
    m_sip3d[i]->fill(    jetFlavour, tagInfo->properties(i).sip3d );
    m_ptRel[i]->fill(    jetFlavour, tagInfo->properties(i).ptRel );
    m_p0Par[i]->fill(    jetFlavour, tagInfo->properties(i).p0Par );
    m_etaRel[i]->fill(   jetFlavour, tagInfo->properties(i).etaRel );
    m_deltaR[i]->fill(   jetFlavour, tagInfo->properties(i).deltaR );
    m_ratio[i]->fill(    jetFlavour, tagInfo->properties(i).ratio );
    m_ratioRel[i]->fill( jetFlavour, tagInfo->properties(i).ratioRel );
  }
}

void SoftLeptonTagPlotter::psPlot(const TString & name)
{
  TString cName = "SoftLeptonPlots" + theExtensionString;
  setTDRStyle()->cd();
  TCanvas canvas(cName, "SoftLeptonPlots" + theExtensionString, 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print(name + cName + ".ps[");
  for (int i = 0; i < s_leptons; i++) {
    canvas.cd(1)->Clear();
    m_leptonId[i]->plot();
    canvas.cd(2)->Clear();
    m_leptonPt[i]->plot();
    canvas.cd(3)->Clear();
    m_sip2d[i]->plot();
    canvas.cd(4)->Clear();
    m_sip3d[i]->plot();
    canvas.cd(5)->Clear();
    m_ptRel[i]->plot();
    canvas.cd(6)->Clear();
    m_p0Par[i]->plot();
    canvas.Print(name + cName + ".ps");

    canvas.cd(1)->Clear();
    m_etaRel[i]->plot();
    canvas.cd(2)->Clear();
    m_deltaR[i]->plot();
    canvas.cd(3)->Clear();
    m_ratio[i]->plot();
    canvas.cd(4)->Clear();
    m_ratioRel[i]->plot();
    canvas.cd(5)->Clear();
    canvas.cd(6)->Clear();
    canvas.Print(name + cName + ".ps");
  }
  canvas.Print(name + cName + ".ps]");
}


void SoftLeptonTagPlotter::epsPlot(const TString & name)
{
  for (int i=0; i < s_leptons; i++) {
    m_leptonId[i]->epsPlot( name );
    m_leptonPt[i]->epsPlot( name );
    m_sip2d[i]->epsPlot( name );
    m_sip3d[i]->epsPlot( name );
    m_ptRel[i]->epsPlot( name );
    m_p0Par[i]->epsPlot( name );
    m_etaRel[i]->epsPlot( name );
    m_deltaR[i]->epsPlot( name );
    m_ratio[i]->epsPlot( name );
    m_ratioRel[i]->epsPlot( name );
  }
}

