
/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <memory>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class GEMCSCStripMapping : public edm::one::EDAnalyzer<> {

public: 
  GEMCSCStripMapping( const edm::ParameterSet& pset);

  ~GEMCSCStripMapping();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  const std::string& myName() { return myName_;}

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

using namespace std;
GEMCSCStripMapping::GEMCSCStripMapping( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "GEMCSCStripMapping" ) 
{ 
  ofos.open("MytestOutput.out"); 
  ofos <<"======================== Opening output file"<< std::endl;
}


GEMCSCStripMapping::~GEMCSCStripMapping() 
{
  ofos.close();
  ofos <<"======================== Closing output file"<< std::endl;
}

void
GEMCSCStripMapping::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<GEMGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);     
  
  ofos << myName() << ": Analyzer..." << std::endl;
  ofos << "start " << dashedLine_ << std::endl;

  ofos << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMCSCStripMapping);
