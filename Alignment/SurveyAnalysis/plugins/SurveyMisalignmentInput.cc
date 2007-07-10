#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"

#include "SurveyMisalignmentInput.h"

SurveyMisalignmentInput::SurveyMisalignmentInput(const edm::ParameterSet& cfg):
  textFileName( cfg.getParameter<std::string>("textFileName") )
{
}

void SurveyMisalignmentInput::beginJob(const edm::EventSetup& setup)
{
	
	edm::ESHandle<DDCompactView> view;
  edm::ESHandle<GeometricDet>  geom;

  setup.get<IdealGeometryRecord>().get(view);
  setup.get<IdealGeometryRecord>().get(geom);

	TrackerGeometry* tracker =
    TrackerGeomBuilderFromGeometricDet().build(&*view, &*geom);
	addComponent( new AlignableTracker(&*geom, tracker) );

	edm::LogInfo("TrackerAlignment") << "Starting!";
  // Retrieve alignment[Error]s from DBase
  setup.get<TrackerAlignmentRcd>().get( alignments );

	//Get map from textreader
	SurveyInputTextReader dataReader;
  dataReader.readFile( textFileName );
	uIdMap = dataReader.UniqueIdMap();

	addSurveyInfo( detector() );
}


void SurveyMisalignmentInput::addSurveyInfo(Alignable* ali)
{

	const std::vector<Alignable*>& comp = ali->components();
  unsigned int nComp = comp.size();
  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);

	static TrackerAlignableId uid;
	TrackerAlignableId::UniqueId id = uid.alignableUniqueId(ali);
	
	SurveyInputTextReader::MapType::const_iterator it;
	it = uIdMap.find(id);
	align::ErrorMatrix error;

	if (it != uIdMap.end()){
		//survey error values
		const std::vector<float>& parameters = (it)->second;
		//sets the errors for the hierarchy level
		double* errorData = error.Array();
		for (int i = 0; i < 21; i++){errorData[i] = parameters[i+6];}
		
		//because record only needs global value of modules
		if (ali->alignableObjectId() == AlignableObjectId::AlignableDetUnit){
			//survey values
			AlignableSurface aliPosition;
			aliPosition = getAlignableSurface(ali->geomDetId().rawId());
			//fill
			ali->setSurvey( new SurveyDet(aliPosition, error) );
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

AlignableSurface SurveyMisalignmentInput::getAlignableSurface(uint32_t rawId)
{
	AlignableSurface alignValue;
	for ( std::vector<AlignTransform>::const_iterator it = alignments->m_align.begin();
				it != alignments->m_align.end(); it++ ){
		if (rawId == (*it).rawId()){
			align::PositionType position( (*it).translation().x(), (*it).translation().y(), (*it).translation().z() );
			CLHEP::HepRotation rot( (*it).rotation() );
			align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
				rot.yx(), rot.yy(), rot.yz(),
				rot.zx(), rot.zy(), rot.zz() );
			alignValue = AlignableSurface(position,rotation);
			break;
		}
	}
	
	return alignValue;

	
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(SurveyMisalignmentInput);
