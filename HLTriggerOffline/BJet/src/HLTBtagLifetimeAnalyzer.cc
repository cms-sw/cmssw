#include <memory>
#include <iostream>
#include <string>

#include <TFile.h>
#include <TH1.h>
#include <TH3.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

/*
# configuration for HLTBtagLifetimeAnalyzer

module hltBtagLifetimeAnalyzer = HLTBtagLifetimeAnalyzer {
  InputTag  vertex = pixelVertices
  PSet      vertexConfiguration = {
    double maxZ = 20        // cm
    double maxR =  0.05     // cm
  }

  VPSet levels = { {
    string   name   = "L2_jets"
    string   title  = "L2 Jets"
    InputTag jets   = iterativeCone5CaloJets::HLT
  }, {
    string   name   = "L25_jets"
    string   title  = "Jets before L2.5"
    InputTag jets   = hltBLifetimeL25Jets::HLT
    InputTag tracks = hltBLifetimeL25Associator::HLT
  }, {
    ...
  } }

  PSet jetConfiguration = {
    double maxEnergy = 300  // GeV
    double maxEta    =   5  // pseudorapidity
  }

  string outputFile = "plots.root"
}
*/

static const unsigned int jetEnergyBins   =  100;
static const unsigned int jetGeometryBins =  100;
static const unsigned int vertex1DBins    = 1000;
static const unsigned int vertex3DBins    =  100;

// jet plots (for each step)
struct JetPlots {
  JetPlots()
  { }
  
  // let ROOT handle deleting the histograms
  ~JetPlots()
  { }
  
  void init(const std::string & name, const std::string & title, unsigned int energyBins, double minEnergy, double maxEnergy, unsigned int geometryBins, double maxEta, bool hasTracks = false)
  {
    m_name          = name;
    m_title         = title;
    m_energyBins    = energyBins;
    m_minEnergy     = minEnergy;
    m_maxEnergy     = maxEnergy;
    m_geometryBins  = geometryBins;
    m_maxEta        = maxEta;
    m_hasTracks     = hasTracks;
    
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);
    m_jetEnergy = new TH1F((m_name + "_Energy").c_str(), (m_title + " energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetET     = new TH1F((m_name + "_ET").c_str(),     (m_title + " ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetEta    = new TH1F((m_name + "_Eta").c_str(),    (m_title + " eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
    m_jetPhi    = new TH1F((m_name + "_Phi").c_str(),    (m_title + " phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    if (m_hasTracks) {
      m_tracksEnergy = new TH1F((m_name + "_Tracks_Energy").c_str(), ("Tracks in " + m_title + " vs. jet energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksET     = new TH1F((m_name + "_Tracks_ET").c_str(),     ("Tracks in " + m_title + " vs. jet ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksEta    = new TH1F((m_name + "_Tracks_Eta").c_str(),    ("Tracks in " + m_title + " vs. jet eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
      m_tracksPhi    = new TH1F((m_name + "_Tracks_Phi").c_str(),    ("Tracks in " + m_title + " vs. jet phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    }
    TH1::SetDefaultSumw2(sumw2);
  }

  void fill(const reco::Jet & jet) {
    m_jetEnergy->Fill(jet.energy());
    m_jetET->Fill(jet.et());
    m_jetEta->Fill(jet.eta());
    m_jetPhi->Fill(jet.phi());
  }

  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks) {
    fill(jet);
    if (m_hasTracks) {
      m_tracksEnergy->Fill(jet.energy(), tracks.size());
      m_tracksET->Fill(jet.et(), tracks.size());
      m_tracksEta->Fill(jet.eta(), tracks.size());
      m_tracksPhi->Fill(jet.phi(), tracks.size());
    }
  }

  void save(TDirectory & file) {
    m_jetEnergy->SetDirectory(&file);
    m_jetET->SetDirectory(&file);
    m_jetEta->SetDirectory(&file);
    m_jetPhi->SetDirectory(&file);
    if (m_hasTracks) {
      // normalize the number of tracks to the number of jets (i.e. compute average number of tracks per jet)
      m_tracksEnergy->Divide(m_jetEnergy);
      m_tracksEnergy->SetDirectory(&file);
      m_tracksET->Divide(m_jetET);
      m_tracksET->SetDirectory(&file);
      m_tracksEta->Divide(m_jetEta);
      m_tracksEta->SetDirectory(&file);
      m_tracksPhi->Divide(m_jetPhi);
      m_tracksPhi->SetDirectory(&file);
    }
  }

  JetPlots efficiency(const JetPlots & denominator)
  {
    JetPlots efficiency;
    efficiency.init(m_name + "_vs_" + denominator.m_name, m_title + " vs. " + denominator.m_title, m_energyBins, m_minEnergy, m_maxEnergy, m_geometryBins, m_maxEta, false);
    efficiency.m_jetEnergy->Divide(m_jetEnergy, denominator.m_jetEnergy, 1., 1., "B");
    efficiency.m_jetEnergy->SetMinimum(0.0);
    efficiency.m_jetEnergy->SetMaximum(1.0);
    efficiency.m_jetET->Divide(m_jetET, denominator.m_jetET, 1., 1., "B");
    efficiency.m_jetET->SetMinimum(0.0);
    efficiency.m_jetET->SetMaximum(1.0);
    efficiency.m_jetEta->Divide(m_jetEta, denominator.m_jetEta, 1., 1., "B");
    efficiency.m_jetEta->SetMinimum(0.0);
    efficiency.m_jetEta->SetMaximum(1.0);
    efficiency.m_jetPhi->Divide(m_jetPhi, denominator.m_jetPhi, 1., 1., "B");
    efficiency.m_jetPhi->SetMinimum(0.0);
    efficiency.m_jetPhi->SetMaximum(1.0);
    return efficiency;
  }
 
  TH1 * m_jetEnergy;
  TH1 * m_jetET;
  TH1 * m_jetEta;
  TH1 * m_jetPhi;
  TH1 * m_tracksEnergy;
  TH1 * m_tracksET;
  TH1 * m_tracksEta;
  TH1 * m_tracksPhi;

  std::string   m_name;
  std::string   m_title;
  unsigned int  m_geometryBins;
  unsigned int  m_energyBins;
  double        m_minEnergy;
  double        m_maxEnergy;
  double        m_maxEta;

  bool          m_hasTracks;
};


struct VertexPlots {
  VertexPlots() :
    /*
    m_3d(0),
    */
    m_r(0),
    m_z(0)
  { }

  void init(const std::string & name, const std::string & title, unsigned int bins, double zRange, double rRange)
  {
    /*
    m_3d = new TH3F((name + "_3D").c_str(), (title + " position").c_str(),   bins, -rRange, rRange, bins, -rRange, rRange, bins, -zRange, zRange);
    */
    m_r  = new TH1F((name + "_R").c_str(),  (title + " R position").c_str(), bins, -rRange, rRange);
    m_z  = new TH1F((name + "_Z").c_str(),  (title + " Z position").c_str(), bins, -zRange, zRange);
  }
    

  void fill(const reco::Vertex & vertex)
  {
    /*
    m_3d->Fill(vertex.front().x(), vertex.front().y(), vertex.front().z());
    */
    m_r->Fill(vertex.position().rho());
    m_z->Fill(vertex.z());
  }

  void save(TDirectory & file)
  {
    /*
    m_3d->SetDirectory(&file);
    */
    m_r->SetDirectory(&file);
    m_z->SetDirectory(&file);
  }
  
  /*
  TH3 * m_3d;
  */
  TH1 * m_r;
  TH1 * m_z;
};


class HLTBtagLifetimeAnalyzer : public edm::EDAnalyzer {
public:
  explicit HLTBtagLifetimeAnalyzer(const edm::ParameterSet& config);
  virtual ~HLTBtagLifetimeAnalyzer();
    
  virtual void beginJob(const edm::EventSetup & setup);
  virtual void analyze(const edm::Event & event, const edm::EventSetup & setup);
  virtual void endJob();

private:
  struct InputData {
    std::string     m_name;
    std::string     m_title;
    edm::InputTag   m_label;
    edm::InputTag   m_tracks;
  };
  
  // input collections
  edm::InputTag             m_trigger;      // HLT event
  edm::InputTag             m_vertex;       // primary vertex
  std::vector<InputData>    m_levels;

  // plot configuration
  double m_jetMinEnergy;
  double m_jetMaxEnergy;
  double m_jetMaxEta;

  double m_vertexMaxR;
  double m_vertexMaxZ;

  // plot data
  VertexPlots           m_vertexPlots;
  std::vector<JetPlots> m_jetPlots;

  // output configuration
  std::string m_outputFile;
};


HLTBtagLifetimeAnalyzer::HLTBtagLifetimeAnalyzer(const edm::ParameterSet & config) :
  m_vertex( config.getParameter<edm::InputTag>("vertex") ),
  m_levels(),
  m_jetMinEnergy(  0. ),    //   0 GeV
  m_jetMaxEnergy( 300. ),   // 300 GeV
  m_jetMaxEta( 5. ),        //  ±5 pseudorapidity units
  m_vertexMaxR( 0.1 ),      //   1 mm
  m_vertexMaxZ( 15. ),      //  15 cm
  m_vertexPlots(),
  m_jetPlots(),
  m_outputFile( config.getParameter<std::string>("outputFile") )
{
  const std::vector<edm::ParameterSet> levels = config.getParameter<std::vector<edm::ParameterSet> >("levels");
  for (unsigned int i = 0; i < levels.size(); ++i) {
    InputData level;
    level.m_label  = levels[i].getParameter<edm::InputTag>("jets");
    level.m_name   = levels[i].exists("name")   ? levels[i].getParameter<std::string>("name")  : level.m_label.encode();
    level.m_title  = levels[i].exists("title")  ? levels[i].getParameter<std::string>("title") : level.m_name;
    level.m_tracks = levels[i].exists("tracks") ? levels[i].getParameter<edm::InputTag>("tracks") : edm::InputTag("none");
    m_levels.push_back(level);
  }
    
  const edm::ParameterSet & jetConfig = config.getParameter<edm::ParameterSet>("jetConfiguration");
  m_jetMaxEnergy = jetConfig.getParameter<double>("maxEnergy");
  m_jetMaxEta    = jetConfig.getParameter<double>("maxEta");
  const edm::ParameterSet & vertexConfig = config.getParameter<edm::ParameterSet>("vertexConfiguration");
  m_vertexMaxR = vertexConfig.getParameter<double>("maxR");
  m_vertexMaxZ = vertexConfig.getParameter<double>("maxZ");
}

HLTBtagLifetimeAnalyzer::~HLTBtagLifetimeAnalyzer() 
{
}

void HLTBtagLifetimeAnalyzer::beginJob(const edm::EventSetup & setup) 
{
  m_jetPlots.resize( m_levels.size() );
  for (unsigned int i = 0; i < m_levels.size(); ++i)
    m_jetPlots[i].init( m_levels[i].m_name, m_levels[i].m_title, jetEnergyBins, m_jetMinEnergy, m_jetMaxEnergy, jetGeometryBins, m_jetMaxEta, m_levels[i].m_tracks.label() != "none" );
  
  m_vertexPlots.init( "PrimaryVertex", "Primary vertex", vertex1DBins, m_vertexMaxZ, m_vertexMaxR );
}

void HLTBtagLifetimeAnalyzer::analyze(const edm::Event & event, const edm::EventSetup & setup) 
{
  edm::Handle<reco::VertexCollection> h_vertex;
  event.getByLabel(m_vertex, h_vertex);
  if (h_vertex.isValid() and not h_vertex->empty())
    m_vertexPlots.fill(h_vertex->front());

  for (unsigned int i = 0; i < m_levels.size(); ++i) {
    edm::Handle<edm::View<reco::Jet> >                  h_jets;
    edm::Handle<reco::JetTracksAssociation::Container>  h_tracks;
    
    event.getByLabel(m_levels[i].m_label, h_jets);
    if (m_levels[i].m_tracks.label() != "none")
      event.getByLabel(m_levels[i].m_tracks, h_tracks);
    
    if (h_jets.isValid()) {
      const edm::View<reco::Jet> & jets = * h_jets;

      if (not h_tracks.isValid()) {
        // no tracks, fill only the jets
        for (unsigned int j = 0; j < jets.size(); ++j)
          m_jetPlots[i].fill( jets[j] );
      } else {
        // fill jets and tracks
        for (unsigned int j = 0; j < jets.size(); ++j)
          m_jetPlots[i].fill( jets[j], (*h_tracks)[jets.refAt(j)] );
      }
    }

  }
}

void HLTBtagLifetimeAnalyzer::endJob()
{
  TFile * file = new TFile(m_outputFile.c_str(), "RECREATE");
  
  for (unsigned int i = 0; i < m_levels.size(); ++i)
    m_jetPlots[i].save(*file);
  for (unsigned int i = 1; i < m_levels.size(); ++i)
    // make step-by-step efficiency plots
    m_jetPlots[i].efficiency( m_jetPlots[i-1] ).save(*file);
  for (unsigned int i = 2; i < m_levels.size(); ++i)
    // make overall plots
    m_jetPlots[i].efficiency( m_jetPlots[0] ).save(*file);

  m_vertexPlots.save(*file);

  file->Write();
  file->Close();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTBtagLifetimeAnalyzer);
