#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "SurveyInputTrackerFromDB.h"

SurveyInputTrackerFromDB::SurveyInputTrackerFromDB(const edm::ParameterSet& cfg):
  textFileName( cfg.getParameter<std::string>("textFileName") )
{
}

void SurveyInputTrackerFromDB::beginJob(const edm::EventSetup& setup)
{
  edm::ESHandle<DDCompactView> view;
  edm::ESHandle<GeometricDet>  geom;

  setup.get<IdealGeometryRecord>().get(view);
  setup.get<IdealGeometryRecord>().get(geom);

  TrackerGeometry* tracker =
    TrackerGeomBuilderFromGeometricDet().build(&*view, &*geom);
	
	//Get map from textreader
	SurveyInputTextReader dataReader;
  dataReader.readFile( textFileName );
	uIdMap = dataReader.UniqueIdMap();
	
	addComponent( new AlignableTracker(&*geom, tracker) );
  addSurveyInfo( detector() );
}

void SurveyInputTrackerFromDB::addSurveyInfo(Alignable* ali)
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);

	static TrackerAlignableId uid;
	SurveyInputTextReader::MapType::const_iterator it;
	align::ErrorMatrix error;
	if (ali->alignableObjectId() != AlignableObjectId::AlignableDetUnit){
		TrackerAlignableId::UniqueId id = uid.alignableUniqueId(ali);
		std::cout << "UniqueId: " << id.first << ", " << id.second << std::endl;
		it = uIdMap.find(id);
		if (it != uIdMap.end()){
			
			const std::vector<float>& parameters = (it)->second;
			
			//move the surface
			//displacement
			LocalVector lvector (parameters[0], parameters[1], parameters[2]);
			GlobalVector gvector = ali->surface().toGlobal(lvector);
			ali->move(gvector);
			//rotation
			const Basic3DVector<float> rot_aa(parameters[3], parameters[4], parameters[5]);
			align::RotationType rotation(rot_aa, rot_aa.mag());
			ali->rotateInLocalFrame(rotation);
			
			//sets the errors for the hierarchy level
			double* errorData = error.Array();
			for (int i = 0; i < 21; i++){errorData[i] = parameters[i+6];}
			//std::cout << "error: " << std::endl << error << std::endl;
			ali->setSurvey( new SurveyDet(ali->surface(), error*(1e-3)) );
		}
		else {
			error = ROOT::Math::SMatrixIdentity();
			ali->setSurvey( new SurveyDet(ali->surface(), error * 1e-3) );
		}
	}
	else{
		error = ROOT::Math::SMatrixIdentity();
		ali->setSurvey( new SurveyDet(ali->surface(), error * 1e-3) );
	}
}
// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(SurveyInputTrackerFromDB);
