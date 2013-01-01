#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotRecHits.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotRecTracks.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotSimTracks.h"

#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotEcalRecHits.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/VZero/interface/VZero.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <fstream>
#include <iomanip>

using namespace std;

/*****************************************************************************/
class EventPlotter : public edm::EDAnalyzer
{
  public:
    explicit EventPlotter(const edm::ParameterSet& ps);
    ~EventPlotter();
    virtual void beginRun(edm::Run & run,       const edm::EventSetup& es);
    virtual void endJob();
    virtual void analyze (const edm::Event& ev, const edm::EventSetup& es);

  private:
    // void printVZeros  (const edm::Event& ev, ofstream& file);
    void printVertices(const edm::Event& ev, ofstream& file);
    std::string trackProducer;

    const MagneticField* theMagField;
};

/*****************************************************************************/
EventPlotter::EventPlotter(const edm::ParameterSet& ps)
{
  trackProducer = ps.getParameter<std::string>("trackProducer");
}

/*****************************************************************************/
EventPlotter::~EventPlotter()
{
}

/*****************************************************************************/
void EventPlotter::beginRun(edm::Run & run, const edm::EventSetup& es)
{
  // Get magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();
}

/*****************************************************************************/
void EventPlotter::endJob()
{
}

/*****************************************************************************/
/*
void EventPlotter::printVZeros(const edm::Event& ev, ofstream& file)
{
  edm::Handle<reco::VZeroCollection> vZeroHandle;
  ev.getByType(vZeroHandle);
  const reco::VZeroCollection* vZeros = vZeroHandle.product();

  file << ", If[rt, {RGBColor[0.0,0.4,0.0], AbsoluteThickness[3]";

  for(reco::VZeroCollection::const_iterator vZero = vZeros->begin();
                                            vZero!= vZeros->end();
                                            vZero++)
  {
    // Calculate closest approach to beam-line
    GlobalPoint crossing(vZero->vertex().position().x(),
                         vZero->vertex().position().y(),
                         vZero->vertex().position().z());

    GlobalVector momentum = vZero->momenta().first +
                            vZero->momenta().second;

    GlobalVector r_(crossing.x(),crossing.y(),0);
    GlobalVector p_(momentum.x(),momentum.y(),0);

    GlobalVector r (crossing.x(),crossing.y(),crossing.z());
    GlobalVector p (momentum.x(),momentum.y(),momentum.z());
    GlobalVector b  = r  - (r_*p_)*p  / p_.mag2();

    file << ", Line[{{" << vZero->vertex().position().x()
                 << "," << vZero->vertex().position().y()
                 << ",(" << vZero->vertex().position().z()
                 << "-zs)*mz}, {" << b.x()
                      << "," << b.y()
                      << ",(" << b.z() << "-zs)*mz}}]" << std::endl;
  }

  file << "}]";
}
*/
/*****************************************************************************/
void EventPlotter::printVertices(const edm::Event& ev, ofstream& file)
{
  // Get vertices
  edm::Handle<reco::VertexCollection> vertexHandle;
  ev.getByLabel("pixel3Vertices",     vertexHandle);
  const reco::VertexCollection* vertices = vertexHandle.product();

  file << ", RGBColor[0,0.8,0], AbsolutePointSize[7]";

  edm::LogVerbatim("MinBiasTracking")
       << " [EventPlotter] vertices : "
       << vertices->size();

  for(reco::VertexCollection::const_iterator vertex = vertices->begin();
                                             vertex!= vertices->end();
                                             vertex++)
  {
    file << ", Point[{" << vertex->position().x()
                 << "," << vertex->position().y()
                 << ",(" << vertex->position().z() << "-zs)*mz}]" << std::endl;
    file << ", Text[StyleForm[\"V\", URL->\"Vertex z="<<vertex->position().z()<<" cm | Tracks="
         << vertex->tracksSize() << "\"]"
         << ", {" << vertex->position().x()
           << "," << vertex->position().y()
           << ",(" << vertex->position().z() << "-zs)*mz}, {0,-1}]" << std::endl;
  }
}

/*****************************************************************************/
void EventPlotter::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::LogVerbatim("MinBiasTracking") << "[EventPlotter]";

  ofstream file("event.m");
  file << fixed << std::setprecision(3);

  // start graphics 
  file << "Graphics3D[";

  // start physics
  file << "{";
  PlotRecHits theRecHits(es,file);
  theRecHits.printRecHits(ev);

//  PlotRecTracks theRecTracks(es,trackCollections,file);
  PlotRecTracks theRecTracks(es,trackProducer,file);
  theRecTracks.printRecTracks(ev, es);

  PlotSimTracks theSimTracks(es,file);
  theSimTracks.printSimTracks(ev, es);

  PlotEcalRecHits theEcalRecHits(es,file);
  theEcalRecHits.printEcalRecHits(ev);

// FIXME
//  printVZeros  (ev,file);
  printVertices(ev,file);

  // region (tracker + ecal)
//  int mx = 160; int my = 160; int mz = 360;
  int mx = 120; int my = 120; int mz = 300;

  // add problems
/*
  std::string str;
  ifstream prob("../data/problem.m");
  getline(prob, str);
  while(prob)
  { file << str; getline(prob, str); }
  prob.close();
*/

  // beam line
  file << ", RGBColor[0.7,0.7,0.7]";

  for(int z = -mz; z < mz; z += mz/30)
    file << ", Line[{{0,0,("<<z<<"-zs)*mz}, {0,0,("<<z+mz/30<<"-zs)*mz}}]" << std::endl;

  // box
  file << ", RGBColor[0,0,0]";
  for(int iz = -1; iz <= 1; iz+=2)
  {
    file << ", Line[{";
    file << "{"<<-mx<<","<<-my<<",("<<iz*mz<<"-zs)*mz}, ";
    file << "{"<< mx<<","<<-my<<",("<<iz*mz<<"-zs)*mz}, ";
    file << "{"<< mx<<","<< my<<",("<<iz*mz<<"-zs)*mz}, ";
    file << "{"<<-mx<<","<< my<<",("<<iz*mz<<"-zs)*mz}, ";
    file << "{"<<-mx<<","<<-my<<",("<<iz*mz<<"-zs)*mz}";
    file << "}]";
  }

  for(int ix = -1; ix <= 1; ix+=2)
  for(int iy = -1; iy <= 1; iy+=2)
  {
    file << ", Line[{{"<<ix*mx<<","<<iy*my<<",("<<-mz<<"-zs)*mz},";
    file <<         "{"<<ix*mx<<","<<iy*my<<",("<< mz<<"-zs)*mz}}]";
  }

  // stop physics
  file << "}";

  // options
  file << ", PlotRange->{{"<<-mx<<","<<mx<<"}, {"
                           <<-my<<","<<my<<"}, {"
                           <<-mz<<"*z,"<<mz<<"*z}}";
  file << ", PlotLabel->\"" << "run:"    << ev.id().run()
                            << " event:" << ev.id().event() << "\"";
  file << ", Boxed->False, AxesLabel->{x,y,z}";
  file << "]";

  file.close();

  system("zip -q -m event.zip event.m ; mv event.zip ../data");

  edm::LogVerbatim("MinBiasTracking") << "[EventPlotter] event plotted";

  while(getchar() == 0);
} 

DEFINE_FWK_MODULE(EventPlotter);
