#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDTransform.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>

#include <ostream>
#include <set>
class DDPartSelection;

class OutputDDToDDL : public edm::EDAnalyzer {

 public:
  explicit OutputDDToDDL( const edm::ParameterSet& iConfig );
  ~OutputDDToDDL();
  virtual void beginJob( edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  bool doPartSelectionsMatch( DDPartSelection* sp1, DDPartSelection* sp2 );
  bool doDDsvaluesMatch ( DDsvalues_type* sv1, DDsvalues_type* sv2 );
  void  addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore );
  void  addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore );

  int rotNumSeed_;
  std::string fname_;
  std::ostream* xos_;
  //  int specNameCount_;

};

  /// is sv1 < sv2 
struct ddsvaluesCmp {
  bool operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 );
};
