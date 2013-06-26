#include "DQMOffline/RecoB/interface/SoftLeptonTagPlotter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQMOffline/RecoB/interface/Tools.h"

#include <sstream>
#include <string>

using namespace std ;
using namespace RecoBTag;

static const string ordinal[9] = { "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th" };

SoftLeptonTagPlotter::SoftLeptonTagPlotter(const std::string & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, const unsigned int& mc, const bool& update) :
    BaseTagInfoPlotter(tagName, etaPtBin), mcPlots_(mc)
{
  const std::string softLepDir(theExtensionString.substr(1));

  for (int i = 0; i < s_leptons; i++) {
    std::ostringstream s("");
    s << ordinal[i] << " lepton ";
    m_leptonId[i] = new FlavourHistograms<double> (
						   s.str() + "id",
						   "Lepton identification discriminaint",
						   60, -0.1, 1.1, false, false, true, "b", update,softLepDir,mcPlots_ );
    m_leptonPt[i] = new FlavourHistograms<double> (
						   s.str() + "pT",
						   "Lepton transverse moementum",
						   100, 0.0, 20.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_sip2d[i] = new FlavourHistograms<double> (
        s.str() + "sip2d",
        "Lepton signed 2D impact parameter significance",
        100, -20.0, 30.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_sip3d[i] = new FlavourHistograms<double> (
        s.str() + "sip3d",
        "Lepton signed 3D impact parameter significance",
        100, -20.0, 30.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_ptRel[i] = new FlavourHistograms<double> (
        s.str() +  "pT rel",
        "Lepton transverse moementum relative to jet axis",
        100, 0.0, 10.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_p0Par[i] = new FlavourHistograms<double> (
        s.str() + "p0 par",
        "Lepton moementum along jet axis in the B rest frame",
        100, 0.0, 10.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_etaRel[i] = new FlavourHistograms<double> (
        s.str() + "eta rel",
        "Lepton pseudorapidity relative to jet axis",
        100, -5.0, 25.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_deltaR[i] = new FlavourHistograms<double> (
        s.str() + "delta R",
        "Lepton pseudoangular distance from jet axis",
        100, 0.0, 0.6, false, false, true, "b", update,softLepDir,mcPlots_);
    m_ratio[i] = new FlavourHistograms<double> (
        s.str() + "energy ratio",
        "Ratio of lepton momentum to jet energy",
        100, 0.0, 2.0, false, false, true, "b", update,softLepDir,mcPlots_);
    m_ratioRel[i] = new FlavourHistograms<double> (
        s.str() + "parallel energy ratio",
        "Ratio of lepton momentum along the jet axis to jet energy",
        100, 0.0, 2.0, false, false, true, "b", update,softLepDir,mcPlots_);
  }
}

SoftLeptonTagPlotter::~SoftLeptonTagPlotter ()
{
  for (int i = 0; i != s_leptons; ++i) {
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
  analyzeTag(baseTagInfo,jetFlavour,1.);
}

void SoftLeptonTagPlotter::analyzeTag( const reco::BaseTagInfo * baseTagInfo,
				       const int & jetFlavour,
				       const float & w)
{

  const reco::SoftLeptonTagInfo * tagInfo = 
	dynamic_cast<const reco::SoftLeptonTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type SoftLeptonTagInfo. " << endl;
  }

  int n_leptons = tagInfo->leptons();

  for (int i = 0; i != n_leptons && i != s_leptons; ++i) {
    const reco::SoftLeptonProperties& properties = tagInfo->properties(i);
    m_leptonPt[i]->fill( jetFlavour, tagInfo->lepton(i)->pt() ,w);
    m_leptonId[i]->fill( jetFlavour, properties.quality() ,w);
    m_sip2d[i]->fill(    jetFlavour, properties.sip2d ,w);
    m_sip3d[i]->fill(    jetFlavour, properties.sip3d ,w);
    m_ptRel[i]->fill(    jetFlavour, properties.ptRel ,w);
    m_p0Par[i]->fill(    jetFlavour, properties.p0Par ,w);
    m_etaRel[i]->fill(   jetFlavour, properties.etaRel ,w);
    m_deltaR[i]->fill(   jetFlavour, properties.deltaR ,w);
    m_ratio[i]->fill(    jetFlavour, properties.ratio ,w);
    m_ratioRel[i]->fill( jetFlavour, properties.ratioRel ,w);
  }
}

void SoftLeptonTagPlotter::psPlot(const std::string & name)
{
  const std::string cName("SoftLeptonPlots" + theExtensionString);
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());
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
    canvas.Print((name + cName + ".ps").c_str());

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
    canvas.Print((name + cName + ".ps").c_str());
  }
  canvas.Print((name + cName + ".ps]").c_str());
}


void SoftLeptonTagPlotter::epsPlot(const std::string & name)
{
  for (int i=0; i != s_leptons; ++i) {
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

