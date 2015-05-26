#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/plugins/SurveyMisalignmentInput.h"

SurveyMisalignmentInput::SurveyMisalignmentInput(const edm::ParameterSet& cfg):
  textFileName( cfg.getParameter<std::string>("textFileName") )
{}

void SurveyMisalignmentInput::analyze(const edm::Event&, const edm::EventSetup& setup)
{
  if (theFirstEvent) {
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    setup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    edm::ESHandle<GeometricDet> geom;
    setup.get<IdealGeometryRecord>().get(geom);	 

    edm::ESHandle<PTrackerParameters> ptp;
    setup.get<PTrackerParametersRcd>().get( ptp );
    TrackerGeometry* tracker = TrackerGeomBuilderFromGeometricDet().build(&*geom, *ptp );
    
    addComponent(new AlignableTracker( tracker, tTopo ));

    edm::LogInfo("SurveyMisalignmentInput") << "Starting!";
    // Retrieve alignment[Error]s from DBase
    setup.get<TrackerAlignmentRcd>().get( alignments );
    
    //Get map from textreader
    SurveyInputTextReader dataReader;
    dataReader.readFile( textFileName );
    uIdMap = dataReader.UniqueIdMap();
    
    addSurveyInfo( detector() );

    theFirstEvent = false;
  }
}


void SurveyMisalignmentInput::addSurveyInfo(Alignable* ali)
{

  const align::Alignables& comp = ali->components();
  unsigned int nComp = comp.size();
  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);
	
  SurveyInputTextReader::MapType::const_iterator it
    = uIdMap.find(std::make_pair(ali->id(), ali->alignableObjectId()));

  align::ErrorMatrix error;

  if (it != uIdMap.end()){
    //survey error values
    const align::Scalars& parameters = (it)->second;
    //sets the errors for the hierarchy level
    double* errorData = error.Array();
    for (unsigned int i = 0; i < 21; ++i){errorData[i] = parameters[i+6];}
		
    //because record only needs global value of modules
    if (ali->alignableObjectId() == align::AlignableDetUnit){
      // fill survey values
      ali->setSurvey( new SurveyDet(getAlignableSurface(ali->id()), error) );
    }
    else{
      ali->setSurvey( new SurveyDet(ali->surface(), error) );
    }
  }
  else{
    //fill
    error = ROOT::Math::SMatrixIdentity();
    ali->setSurvey( new SurveyDet(ali->surface(), error*(1e-6)) );
  }
  //std::cout << "UniqueId: " << id.first << ", " << id.second << std::endl;
  //std::cout << error << std::endl;
	
}

AlignableSurface SurveyMisalignmentInput::getAlignableSurface(align::ID id)
{
  std::vector<AlignTransform>::const_iterator it;

  for (it = alignments->m_align.begin(); it != alignments->m_align.end(); ++it)
  {
    if (id == (*it).rawId())
    {
      align::PositionType position( (*it).translation().x(), (*it).translation().y(), (*it).translation().z() );
      CLHEP::HepRotation rot( (*it).rotation() );
      align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
				    rot.yx(), rot.yy(), rot.yz(),
				    rot.zx(), rot.zy(), rot.zz() );
      return AlignableSurface(position,rotation);
    }
  }
	
  return AlignableSurface();
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SurveyMisalignmentInput);
