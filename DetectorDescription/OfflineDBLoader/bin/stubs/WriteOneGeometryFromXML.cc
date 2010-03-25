#include "WriteOneGeometryFromXML.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <CondCore/DBOutputService/interface/PoolDBOutputService.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDTransform.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDLogicalPart.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDRoot.h>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDPosData.h>
#include <DetectorDescription/PersistentDDDObjects/interface/DDDToPersFactory.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>

WriteOneGeometryFromXML::WriteOneGeometryFromXML(const edm::ParameterSet& iConfig) : label_()
{
  std::cout<<"WriteOneGeometryFromXML::WriteOneGeometryFromXML"<<std::endl;
  rotNumSeed_ = iConfig.getParameter<int>("rotNumSeed");
}

WriteOneGeometryFromXML::~WriteOneGeometryFromXML()
{
  std::cout<<"WriteOneGeometryFromXML::~WriteOneGeometryFromXML"<<std::endl;
}

void
WriteOneGeometryFromXML::beginRun( const edm::Run&, edm::EventSetup const& es) 
{
  std::cout<<"WriteOneGeometryFromXML::beginRun"<<std::endl;
  PIdealGeometry* pgeom = new PIdealGeometry;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"PoolDBOutputService unavailable"<<std::endl;
    return;
  }

  //  std::cout << "About to do...   size_t callbackToken=mydbservice->callbackToken(\"PIdealGeometry\");" << std::endl;
  //  size_t callbackToken=mydbservice->callbackToken("PIdealGeometry");
  //  std::cout << "Got back token " << callbackToken << std::endl;
  edm::ESTransientHandle<DDCompactView> pDD;

  es.get<IdealGeometryRecord>().get(label_, pDD );
  if (pDD.isValid()) 
    std::cout << "DD is Valid" << std::endl;
  
  DDCompactView::DDCompactView::graph_type gra = pDD->graph();

  DDDToPersFactory dddFact;    
  DDMaterial::iterator<DDMaterial> it(DDMaterial::begin()), ed(DDMaterial::end());
  PMaterial* pm;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;
    pm = dddFact.material ( *it );
    pgeom->pMaterials.push_back ( *pm );
    delete pm;
  }

  DDRotation::iterator<DDRotation> rit(DDRotation::begin()), red(DDRotation::end());
  PRotation* pr;
  DDRotation rotn(DDName("IDENTITYDB","generatedForDB"));
  if ( !rotn.isDefined().second ) {
    DDRotationMatrix* rotID = new DDRotationMatrix();
    DDRotation mydr = DDrot (DDName("IDENTITYDB","generatedForDB"), rotID);
    pr = dddFact.rotation ( mydr );
    pgeom->pRotations.push_back ( *pr );
  }
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    // if it is the identity...
    if ( *(rit->matrix()) == *(rotn.matrix()) ) continue;
    pr = dddFact.rotation( *rit );
    pgeom->pRotations.push_back ( *pr );
  } 


  DDSolid::iterator<DDSolid> sit(DDSolid::begin()), sed(DDSolid::end());
  PSolid* ps;
  for (; sit != sed; ++sit) {
    if (! sit->isDefined().second) continue;  
    ps = dddFact.solid( *sit );
    pgeom->pSolids.push_back( *ps );
    delete ps;
  }

  // Discovered during validation:  If a user declares
  // a LogicalPart in the DDD XML and does not position it
  // in the graph, it will NOT be stored to the database
  // subsequently SpecPars (see below) that use wildcards
  // that may have selected the orphaned LogicalPart node
  // will be read into the DDD from the DB (not by this
  // code, but in the system) and so DDSpecifics will
  // throw a DDException.
  typedef DDCompactView::graph_type::const_adj_iterator adjl_iterator;
  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();    
    
  DDCompactView::graph_type::index_type i=0;
  PLogicalPart* plp;
  for (; git != gend; ++git) 
    {
      const DDLogicalPart & ddLP = gra.nodeData(git);
      //	std::cout << ddLP << std::endl;
      plp = dddFact.logicalPart ( ddLP  );
      pgeom->pLogicalParts.push_back( *plp );
      delete plp;
      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
	  PPosPart* ppp;
	  for (; cit != cend; ++cit) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	      ppp = dddFact.position ( ddLP, ddcurLP, gra.edgeData(cit->second), *pgeom, rotNumSeed_ );
	      // 		std::cout << "okay after the factory..." << std::endl;
	      pgeom->pPosParts.push_back( *ppp );
	      // 		std::cout << "okay after the push_back" << std::endl;
	      delete ppp;
	      // 		std::cout << "okay after the delete..." << std::endl;
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
  
  std::vector<std::string> partSelections;
  std::map<std::string, std::vector<std::pair<std::string, double> > > values;
  std::map<std::string, int> isEvaluated;
  
  PSpecPar* psp;

  DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());

  // ======= For each DDSpecific...
  for (; spit != spend; ++spit) {
    if ( !spit->isDefined().second ) continue;  
    psp = dddFact.specpar( *spit );
    pgeom->pSpecPars.push_back( *psp );
    delete psp;
  } 
  std::cout <<" did the static work? is there a size to this string storage? " << std::endl;
  std::cout << " dddFact.pstrs.pStrings.size() = " << dddFact.pstrs.pStrings.size() << std::endl;
  pgeom->pStrings = dddFact.pstrs.pStrings;
  pgeom->pStartNode = DDRootDef::instance().root().toString();

  if ( mydbservice->isNewTagRequest("IdealGeometryRecord") ) {
    //    mydbservice->newValidityForNewPayload<PIdealGeometry>(pgeom, mydbservice->endOfTime(), callbackToken);
    mydbservice->createNewIOV<PIdealGeometry>(pgeom
					      , mydbservice->beginOfTime()
					      , mydbservice->endOfTime()
					      , "IdealGeometryRecord");
  } else {
    std::cout << "Tag is already present." << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(WriteOneGeometryFromXML);
