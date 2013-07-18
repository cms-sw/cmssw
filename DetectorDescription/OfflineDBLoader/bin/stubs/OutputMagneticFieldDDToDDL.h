#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDTransform.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <DetectorDescription/Core/interface/DDLogicalPart.h>

#include <ostream>
#include <set>

class DDPartSelection;

/// is sv1 < sv2 
struct ddsvaluesCmp
{
  bool operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 );
};

class OutputMagneticFieldDDToDDL : public edm::EDAnalyzer
{
public:
  explicit OutputMagneticFieldDDToDDL( const edm::ParameterSet& iConfig );
  ~OutputMagneticFieldDDToDDL( void );
  
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob( void ) {}

private:
  void addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore );
  void addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore );
  void addToSpecStore( const DDLogicalPart& lp, std::map<DDsvalues_type, std::set<DDPartSelection*>, ddsvaluesCmp > & specStore );

  std::string m_fname;
  std::ostream* m_xos;
  int m_rotNumSeed;
  int m_specNameCount;
};

