#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <ostream>
#include <set>
#include <string>

class DDPartSelection;
class DDRotation;
namespace edm {
class Event;
class EventSetup;
class ParameterSet;
class Run;
namespace one {
struct WatchRuns;
}  // namespace one
}  // namespace edm

namespace {
/// is sv1 < sv2 
struct ddsvaluesCmp {
  bool operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 );
};
}

class OutputMagneticFieldDDToDDL : public edm::one::EDAnalyzer<edm::one::WatchRuns>
{
public:
  explicit OutputMagneticFieldDDToDDL( const edm::ParameterSet& iConfig );
  ~OutputMagneticFieldDDToDDL( void );

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void endJob() override {}
      
private:
  void addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore );
  void addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore );
  void addToSpecStore( const DDLogicalPart& lp, std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp > & specStore );

  int m_rotNumSeed;
  std::string m_fname;
  std::ostream* m_xos;
};

