#include "OutputDDToDDL.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/Framework/interface/ESTransientHandle.h>

#include <DetectorDescription/Core/interface/DDLogicalPart.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDRoot.h>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDPosData.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

// for clhep stuff..
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

bool ddsvaluesCmp::operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 ) {
  if ( sv1.size() < sv2.size() ) return true;
  if ( sv2.size() < sv1.size() ) return false;
  size_t ind = 0;
  for (; ind < sv1.size(); ++ind) {
    if ( sv1[ind].first < sv2[ind].first ) return true;
    if ( sv2[ind].first < sv1[ind].first ) return false;
    if ( sv1[ind].second < sv2[ind].second ) return true;
    if ( sv2[ind].second < sv1[ind].second ) return false;
  }
  return false;
}

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
OutputDDToDDL::beginRun( const edm::Run&, edm::EventSetup const& es) 
{
  std::cout<<"OutputDDToDDL::beginRun"<<std::endl;

  edm::ESTransientHandle<DDCompactView> pDD;

  es.get<IdealGeometryRecord>().get( pDD );

  DDCompactView::DDCompactView::graph_type gra = pDD->graph();
  // temporary stores:
  std::set<DDLogicalPart> lpStore;
  std::set<DDMaterial> matStore;
  std::set<DDSolid> solStore;
  // 2009-08-19: MEC: I've tried this with set<DDPartSelection> and
  // had to write operator< for DDPartSelection and DDPartSelectionLevel
  // the output from such an effort is different than this one.
  std::map<DDsvalues_type, std::set<DDPartSelection*>, ddsvaluesCmp > specStore;
  std::set<DDRotation> rotStore;

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
      if ( lpStore.find(ddLP) != lpStore.end() ) {
	addToSpecStore(ddLP, specStore);
      }
      lpStore.insert(ddLP);
      addToMatStore( ddLP.material(), matStore );
      addToSolStore( ddLP.solid(), solStore, rotStore );
      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
	  for (; cit != cend; ++cit) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	      if (lpStore.find(ddcurLP) != lpStore.end()) {
		addToSpecStore(ddcurLP, specStore);
	      }
	      lpStore.insert(ddcurLP);
	      addToMatStore(ddcurLP.material(), matStore);
	      addToSolStore(ddcurLP.solid(), solStore, rotStore);
	      rotStore.insert(gra.edgeData(cit->second)->rot_);
	      out.position(ddLP, ddcurLP, gra.edgeData(cit->second), rotNumSeed_, *xos_);
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
  //  std::cout << "specStore.size() = " << specStore.size() << std::endl;

  (*xos_) << "</PosPartSection>" << std::endl;
  
  (*xos_) << std::scientific << std::setprecision(18);
  std::set<DDMaterial>::const_iterator it(matStore.begin()), ed(matStore.end());
  (*xos_) << "<MaterialSection label=\"" << ns_ << "\">" << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;
    out.material(*it, *xos_);
  }
  (*xos_) << "</MaterialSection>" << std::endl;

  (*xos_) << "<RotationSection label=\"" << ns_ << "\">" << std::endl;
  (*xos_) << std::fixed << std::setprecision(18);
  std::set<DDRotation>::iterator rit(rotStore.begin()), red(rotStore.end());
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    if ( rit->toString() != ":" ) {
      DDRotation r(*rit);
      out.rotation(r, *xos_);
    }
  } 
  (*xos_) << "</RotationSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  std::set<DDSolid>::const_iterator sit(solStore.begin()), sed(solStore.end());
  (*xos_) << "<SolidSection label=\"" << ns_ << "\">" << std::endl;
  for (; sit != sed; ++sit) {
    if (! sit->isDefined().second) continue;  
    out.solid(*sit, *xos_);
  }
  (*xos_) << "</SolidSection>" << std::endl;

  std::set<DDLogicalPart>::iterator lpit(lpStore.begin()), lped(lpStore.end());
  (*xos_) << "<LogicalPartSection label=\"" << ns_ << "\">" << std::endl;
  for (; lpit != lped; ++lpit) {
    if (! lpit->isDefined().first) continue;  
    const DDLogicalPart & lp = *lpit;
      out.logicalPart(lp, *xos_);
  }
  (*xos_) << "</LogicalPartSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  std::map<DDsvalues_type, std::set<DDPartSelection*> >::const_iterator mit(specStore.begin()), mend (specStore.end());
  (*xos_) << "<SpecParSection label=\"" << ns_ << "\">" << std::endl;
  for (; mit != mend; ++mit) {
    out.specpar ( *mit, *xos_ );
  } 
  (*xos_) << "</SpecParSection>" << std::endl;

}

void  OutputDDToDDL::addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore) {
  matStore.insert(mat);
  if ( mat.noOfConstituents() != 0 ) {
    DDMaterial::FractionV::value_type frac;
    int findex(0);
    while ( findex < mat.noOfConstituents() ) {
      if ( matStore.find(mat.constituent(findex).first) == matStore.end() ) {
	addToMatStore( mat.constituent(findex).first, matStore );
      }
      ++findex;
    }
  }
}

void  OutputDDToDDL::addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore ) {
      solStore.insert(sol);
      if ( sol.shape() == ddunion || sol.shape() == ddsubtraction || sol.shape() == ddintersection ) {
	const DDBooleanSolid& bs (sol);
	if ( solStore.find(bs.solidA()) == solStore.end()) {
	  addToSolStore(bs.solidA(), solStore, rotStore);
	}
	if ( solStore.find(bs.solidB()) == solStore.end()) {
	  addToSolStore(bs.solidB(), solStore, rotStore);
	}
	rotStore.insert(bs.rotation());
      }
}

void OutputDDToDDL::addToSpecStore( const DDLogicalPart& lp, std::map<DDsvalues_type, std::set<DDPartSelection*>, ddsvaluesCmp > & specStore ) {
  std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::const_iterator spit(lp.attachedSpecifics().begin()), spend(lp.attachedSpecifics().end());
  for ( ; spit != spend; ++spit ) {
    specStore[*spit->second].insert(spit->first);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OutputDDToDDL);
