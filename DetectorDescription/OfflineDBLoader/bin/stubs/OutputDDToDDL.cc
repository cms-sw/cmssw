#include "OutputDDToDDL.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
// #include <CondCore/DBOutputService/interface/PoolDBOutputService.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDTransform.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDLogicalPart.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDRoot.h>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDPosData.h>
// #include <DetectorDescription/PersistentDDDObjects/interface/DDDToPersFactory.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

// for clhep stuff..
//#include <DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h>
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

OutputDDToDDL::OutputDDToDDL(const edm::ParameterSet& iConfig) : fname_()
{
  //  std::cout<<"OutputDDToDDL::OutputDDToDDL"<<std::endl;
  rotNumSeed_ = iConfig.getParameter<int>("rotNumSeed");
  fname_ = iConfig.getUntrackedParameter<std::string>("fileName");
  if ( fname_ == "" ) {
    xos_ = &std::cout;
  } else {
    xos_ = new std::ofstream(fname_.c_str());
  }
  (*xos_) << "<?xml version=\"1.0\"?>" << std::endl;
  (*xos_) << "<DDDefinition xmlns=\"http://www.cern.ch/cms/DDL\"" << std::endl;
  (*xos_) << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" << std::endl;
  (*xos_) << "xsi:schemaLocation=\"http://www.cern.ch/cms/DDL ../../../DetectorDescription/Schema/DDLSchema.xsd\">" << std::endl;
  (*xos_) << std::fixed << std::setprecision(18);
}
OutputDDToDDL::~OutputDDToDDL()
{
  (*xos_) << "</DDDefinition>" << std::endl;
  (*xos_) << std::endl;
  xos_->flush();

}

void
OutputDDToDDL::beginJob( edm::EventSetup const& es) 
{
  std::cout<<"OutputDDToDDL::beginJob"<<std::endl;

  edm::ESHandle<DDCompactView> pDD;

  es.get<IdealGeometryRecord>().get( pDD );

  DDCompactView::DDCompactView::graph_type gra = pDD->graph();

  DDCoreToDDXMLOutput out;
  
  std::string rn = fname_;
  size_t foundLastDot= rn.find_last_of('.');
  size_t foundLastSlash= rn.find_last_of('/');
  if ( foundLastSlash > foundLastDot && foundLastSlash != std::string::npos) {
    std::cout << "What? last . before last / in path for filename... this should die..." << std::endl;
  }
  if ( foundLastDot != std::string::npos && foundLastSlash != std::string::npos ) {
    out.ns_ = rn.substr(foundLastSlash,foundLastDot);
  } else if ( foundLastDot != std::string::npos ) {
    out.ns_ = rn.substr(0, foundLastDot);
  } else {
    std::cout << "What? no file name? Attempt at namespace =\"" << out.ns_ << "\" filename was " << fname_ <<  std::endl;
  }
  std::cout << "fname_=" << fname_ << " namespace = " << out.ns_ << std::endl;
  std::string ns_ = out.ns_;

  (*xos_) << std::scientific << std::setprecision(18);
  DDMaterial::iterator<DDMaterial> it(DDMaterial::begin()), ed(DDMaterial::end());
  (*xos_) << "<MaterialSection label=\"" << ns_ << "\">" << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;
    out.material(*it, *xos_);
  }
  (*xos_) << "</MaterialSection>" << std::endl;

  (*xos_) << "<RotationSection label=\"" << ns_ << "\">" << std::endl;
  (*xos_) << std::fixed << std::setprecision(18);
  DDRotationMatrix rotm;
  DDRotation::iterator<DDRotation> rit(DDRotation::begin()), red(DDRotation::end());
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    if ( rit->toString() != ":" ) {
      out.rotation(*rit, *xos_);
    }
  } 
  (*xos_) << "</RotationSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  DDSolid::iterator<DDSolid> sit(DDSolid::begin()), sed(DDSolid::end());
  (*xos_) << "<SolidSection label=\"" << ns_ << "\">" << std::endl;
  for (; sit != sed; ++sit) {
    if (! sit->isDefined().second) continue;  
    out.solid(*sit, *xos_);
  }
  (*xos_) << "</SolidSection>" << std::endl;

  // This CAN include DDLogicalParts that are NOT placed in the graph (user error :))
  DDLogicalPart::iterator<DDLogicalPart> lpit(DDLogicalPart::begin()), lped(DDLogicalPart::end());
  (*xos_) << "<LogicalPartSection label=\"" << ns_ << "\">" << std::endl;
  for (; lpit != lped; ++lpit) {
    if (! lpit->isDefined().second) continue;  
    const DDLogicalPart & lp = *lpit;
      out.logicalPart(lp, *xos_);
  }
  (*xos_) << "</LogicalPartSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;

  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();    
    
  DDCompactView::graph_type::index_type i=0;
  (*xos_) << "<PosPartSection label=\"" << ns_ << "\">" << std::endl;
  git = gra.begin();
  for (; git != gend; ++git) 
    {
      const DDLogicalPart & ddLP = gra.nodeData(git);
      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
	  for (; cit != cend; ++cit) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	      out.position(ddLP, ddcurLP, gra.edgeData(cit->second), rotNumSeed_, *xos_);
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
  (*xos_) << "</PosPartSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  std::vector<std::string> partSelections;
  std::map<std::string, std::vector<std::pair<std::string, double> > > values;
  std::map<std::string, int> isEvaluated;
  
  DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());

  // ======= For each DDSpecific...
  (*xos_) << "<SpecParSection label=\"" << ns_ << "\">" << std::endl;
  for (; spit != spend; ++spit) {
    if ( !spit->isDefined().second ) continue;
    out.specpar ( *spit, *xos_ );
  } 
  (*xos_) << "</SpecParSection>" << std::endl;

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OutputDDToDDL);
