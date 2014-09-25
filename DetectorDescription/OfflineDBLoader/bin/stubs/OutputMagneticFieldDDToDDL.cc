#include "OutputMagneticFieldDDToDDL.h"

#include <FWCore/Framework/interface/ESTransientHandle.h>

#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDPosData.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

bool
ddsvaluesCmp::operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 )
{
  if( sv1.size() < sv2.size()) return true;
  if( sv2.size() < sv1.size()) return false;
  size_t ind = 0;
  for( ; ind < sv1.size(); ++ind )
  {
    if( sv1[ ind ].first < sv2[ ind ].first ) return true;
    if( sv2[ ind ].first < sv1[ ind ].first ) return false;
    if( sv1[ ind ].second < sv2[ ind ].second ) return true;
    if( sv2[ ind ].second < sv1[ ind ].second ) return false;
  }
  return false;
}

OutputMagneticFieldDDToDDL::OutputMagneticFieldDDToDDL( const edm::ParameterSet& iConfig )
  : m_fname()
{
  m_rotNumSeed = iConfig.getParameter<int>( "rotNumSeed" );
  m_fname = iConfig.getUntrackedParameter<std::string>( "fileName" );
  if( m_fname == "" )
  {
    m_xos = &std::cout;
  }
  else
  {
    m_xos = new std::ofstream( m_fname.c_str());
  }
  
  ( *m_xos ) << "<?xml version=\"1.0\"?>\n";
  ( *m_xos ) << "<DDDefinition xmlns=\"http://www.cern.ch/cms/DDL\"\n";
  ( *m_xos ) << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n";
  ( *m_xos ) << "xsi:schemaLocation=\"http://www.cern.ch/cms/DDL ../../../DetectorDescription/Schema/DDLSchema.xsd\">\n";
  ( *m_xos ) << std::fixed << std::setprecision( 18 );
}

OutputMagneticFieldDDToDDL::~OutputMagneticFieldDDToDDL()
{
  ( *m_xos ) << "</DDDefinition>\n";
  ( *m_xos ) << std::endl;
  m_xos->flush();
}

void
OutputMagneticFieldDDToDDL::beginRun( const edm::Run&, edm::EventSetup const& es ) 
{
  edm::LogInfo( "OutputMagneticFieldDDToDDL" ) << "OutputMagneticFieldDDToDDL::beginRun";

  edm::ESTransientHandle<DDCompactView> pDD;
  es.get<IdealMagneticFieldRecord>().get( pDD );

  DDCompactView::DDCompactView::graph_type gra = pDD->graph();
  
  // Temporary stores:
  std::set<DDLogicalPart> lpStore;
  std::set<DDMaterial> matStore;
  std::set<DDSolid> solStore;
  
  std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp > specStore;
  std::set<DDRotation> rotStore;

  DDCoreToDDXMLOutput out;
  
  std::string rn = m_fname;
  size_t foundLastDot= rn.find_last_of('.');
  size_t foundLastSlash= rn.find_last_of('/');

  if( foundLastSlash > foundLastDot && foundLastSlash != std::string::npos )
  {
    edm::LogError( "OutputMagneticFieldDDToDDL" ) << "What? last . before last / in path for filename... this should die...";
  }
  if( foundLastDot != std::string::npos && foundLastSlash != std::string::npos )
  {
    out.ns_ = rn.substr( foundLastSlash, foundLastDot );
  }
  else if ( foundLastDot != std::string::npos )
  {
    out.ns_ = rn.substr(0, foundLastDot);
  }
  else
  {
    edm::LogError( "OutputMagneticFieldDDToDDL" ) << "What? no file name? Attempt at namespace =\"" << out.ns_ << "\" filename was " << m_fname;
  }
  
  edm::LogInfo( "OutputMagneticFieldDDToDDL" ) << "m_fname=" << m_fname << " namespace = " << out.ns_;
  std::string ns_ = out.ns_;

  ( *m_xos ) << std::fixed << std::setprecision( 18 );
  
  typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;

  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();    
    
  DDCompactView::graph_type::index_type i=0;
  ( *m_xos) << "<PosPartSection label=\"" << ns_ << "\">\n";
  git = gra.begin();
  for( ; git != gend; ++git ) 
  {
    const DDLogicalPart & ddLP = gra.nodeData( git );
    if( lpStore.find(ddLP) != lpStore.end())
    {
      addToSpecStore( ddLP, specStore );
    }
    lpStore.insert( ddLP );
    addToMatStore( ddLP.material(), matStore );
    addToSolStore( ddLP.solid(), solStore, rotStore );
    ++i;
    if( git->size()) 
    {
      // ask for children of ddLP  
      DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
      DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
      for( ; cit != cend; ++cit ) 
      {
	const DDLogicalPart & ddcurLP = gra.nodeData( cit->first );
	if( lpStore.find(ddcurLP) != lpStore.end())
	{
	  addToSpecStore( ddcurLP, specStore );
	}
	lpStore.insert( ddcurLP );
	addToMatStore( ddcurLP.material(), matStore );
	addToSolStore( ddcurLP.solid(), solStore, rotStore );
	rotStore.insert( gra.edgeData( cit->second )->rot_ );
	out.position( ddLP, ddcurLP, gra.edgeData( cit->second ), m_rotNumSeed, *m_xos );
      } // iterate over children
    } // if (children)
  } // iterate over graph nodes  

  ( *m_xos ) << "</PosPartSection>\n";
  
  ( *m_xos ) << std::scientific << std::setprecision( 18 );
  std::set<DDMaterial>::const_iterator it( matStore.begin()), ed( matStore.end());
  ( *m_xos) << "<MaterialSection label=\"" << ns_ << "\">\n";
  for( ; it != ed; ++it )
  {
    if( ! it->isDefined().second ) continue;
    out.material( *it, *m_xos );
  }
  ( *m_xos ) << "</MaterialSection>\n";

  ( *m_xos ) << "<RotationSection label=\"" << ns_ << "\">\n";
  ( *m_xos ) << std::fixed << std::setprecision( 18 );
  std::set<DDRotation>::iterator rit( rotStore.begin()), red( rotStore.end());
  for( ; rit != red; ++rit )
  {
    if( ! rit->isDefined().second ) continue;
    if( rit->toString() != ":" )
    {
      DDRotation r( *rit );
      out.rotation( r, *m_xos );
    }
  }
  ( *m_xos ) << "</RotationSection>\n";

  ( *m_xos ) << std::fixed << std::setprecision( 18 );
  std::set<DDSolid>::const_iterator sit( solStore.begin()), sed( solStore.end());
  ( *m_xos ) << "<SolidSection label=\"" << ns_ << "\">\n";
  for( ; sit != sed; ++sit)
  {
    if( ! sit->isDefined().second ) continue;  
    out.solid( *sit, *m_xos );
  }
  ( *m_xos ) << "</SolidSection>\n";

  std::set<DDLogicalPart>::iterator lpit( lpStore.begin()), lped( lpStore.end());
  ( *m_xos ) << "<LogicalPartSection label=\"" << ns_ << "\">\n";
  for( ; lpit != lped; ++lpit )
  {
    if( ! lpit->isDefined().first ) continue;  
    const DDLogicalPart & lp = *lpit;
    out.logicalPart( lp, *m_xos );
  }
  ( *m_xos ) << "</LogicalPartSection>\n";

  ( *m_xos ) << std::fixed << std::setprecision( 18 );
  std::map<DDsvalues_type, std::set<const DDPartSelection*> >::const_iterator mit( specStore.begin()), mend( specStore.end());
  ( *m_xos ) << "<SpecParSection label=\"" << ns_ << "\">\n";
  for( ; mit != mend; ++mit )
  {
    out.specpar( *mit, *m_xos );
  }
  ( *m_xos ) << "</SpecParSection>\n";
}

void
OutputMagneticFieldDDToDDL::addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore )
{
  matStore.insert( mat );
  if( mat.noOfConstituents() != 0 )
  {
    DDMaterial::FractionV::value_type frac;
    int findex( 0 );
    while( findex < mat.noOfConstituents())
    {
      if( matStore.find( mat.constituent( findex ).first ) == matStore.end())
      {
	addToMatStore( mat.constituent( findex ).first, matStore );
      }
      ++findex;
    }
  }
}

void
OutputMagneticFieldDDToDDL::addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore )
{
  solStore.insert( sol );
  if( sol.shape() == ddunion || sol.shape() == ddsubtraction || sol.shape() == ddintersection )
  {
    const DDBooleanSolid& bs ( sol );
    if( solStore.find(bs.solidA()) == solStore.end())
    {
      addToSolStore( bs.solidA(), solStore, rotStore );
    }
    if( solStore.find( bs.solidB()) == solStore.end())
    {
      addToSolStore( bs.solidB(), solStore, rotStore );
    }
    rotStore.insert( bs.rotation());
  }
}

void
OutputMagneticFieldDDToDDL::addToSpecStore( const DDLogicalPart& lp, std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp > & specStore )
{
  std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> >::const_iterator spit( lp.attachedSpecifics().begin()), spend( lp.attachedSpecifics().end());
  for( ; spit != spend; ++spit )
  {
    specStore[ *spit->second ].insert( spit->first );
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( OutputMagneticFieldDDToDDL );
