/****************************************************************************
*
* Authors:
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/GeometryTestModule.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"

#include <iostream>


//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

GeometryTestModule::GeometryTestModule(const edm::ParameterSet& iConfig)
{
}

//----------------------------------------------------------------------------------------------------

GeometryTestModule::~GeometryTestModule()
{
}

//----------------------------------------------------------------------------------------------------

void GeometryTestModule::beginJob()
{
}

//----------------------------------------------------------------------------------------------------

void GeometryTestModule::endJob()
{
}

//----------------------------------------------------------------------------------------------------

void GeometryTestModule::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	using namespace std;

	// retrieve the ideal geometrical description
	// actually, it searchech for DetGeomDesc in VeryForwardMeasuredGeometryRecord
	// and when it is not found, it calls TotemRPDetGeomDescESModule to produce it	
	ESHandle<DetGeomDesc> gD;
	iSetup.get<VeryForwardMeasuredGeometryRecord>().get(gD);

	// retrieve RP map, similarly as above, when TotemRPMap is not found,
	// TotemRPMapESModule is called (indeed, it must be specified in configuration file)
	ESHandle<TotemRPGeometry> idealRPMap;
	iSetup.get<VeryForwardMeasuredGeometryRecord>().get(idealRPMap);

	cout << "-------------------------------------------" 	<< endl
		<<  "        GeometryTestModule output: " 			<< endl;


	/*
	// print the structure on screen
	cout << "-------------------------------------------" << endl;

	deque<DetGeomDesc *> buffer;
	buffer.push_back((DetGeomDesc *)gD.product());

	//int count = 0;
	while (buffer.size() > 0) {
		DetGeomDesc *d = buffer.front();
		buffer.pop_front();
	//	// print only detectors
	//	if (!d->name().name().compare("RP_Silicon_Detector")) {
			//cout	<< "** RP #" << count++ << endl
			cout	<< "** name = " << d->name().name() << ", "
					<< "ID = " << d->geographicalID().rawId()
			//		<< endl
			//		<< "translation: " << d->translation() << endl
			//		<< "rotation: " << d->rotation()
					<< endl;
	//	}
		for (unsigned int i = 0; i < d->components().size(); i++) {
			DetGeomDesc *dd = d->components()[i];
			buffer.push_back(dd);
			cout << "	child[" << i << "] = " << dd->name().name() << endl;
		}

		cout << "==================================================" << endl;
	}
	*/

	/*
	cout << "-------------------------------------------" << endl;
	// get geometrical info for an existing detector
	cout << "shift = " << (*idealRPMap)[1208]->translation() << endl << "rotation:" << idealRPMap->GetDetector(1208)->rotation() << endl;
	
	// exception is raised when geometry of an non-existing detector is required
	//(*idealRPMap)[1288];

	cout << "-------------------------------------------" << endl;
	unsigned int id = 12;
	cout << "children of element with ID " << id << endl;
	set<unsigned int> rps = idealRPMap->RPsInStation(id);
	//set<unsigned int> rps = idealRPMap->RPsInStation(stationId);
	for (set<unsigned int>::iterator it = rps.begin(); it != rps.end(); ++it) {
		cout << (*it)
			<< idealRPMap->GetDetector((*it) * 10)->translation()
			<< endl;
	}

	cout << "-------------------------------------------" << endl;

	id = 1201;
	CLHEP::Hep3Vector v(1, 0, 0);
	cout << "vector = " << v << endl
		<< "detector id = " << id << endl
		<< "LocalToGlobal(v) = " << idealRPMap->LocalToGlobal(id, v) << endl
		<< "GlobalToLocal(v) = " << idealRPMap->GlobalToLocal(id, v) << endl
		<< "LocalToGlobalDirection(v) = " << idealRPMap->LocalToGlobalDirection(id, v) << endl
		<< "GlobalToLocalDirection(v) = " << idealRPMap->GlobalToLocalDirection(id, v) << endl
	   	<< endl;

	cout << "-------------------------------------------" << endl;
	*/

	/*
	// real geometry
	ESHandle<DetGeomDesc> realGD;
	iSetup.get<VeryForwardRealGeometryRecord>().get(realGD);
	ESHandle<TotemRPGeometry> realRPMap;
	iSetup.get<VeryForwardRealGeometryRecord>().get(realRPMap);
	*/

	for (TotemRPGeometry::mapType::const_iterator it = idealRPMap->beginDet(); it != idealRPMap->endDet(); ++it) {
		TotemRPDetId id(it->first);
		cout << "possition of detector " << id.detectorDecId() << endl
			<< "	ideal " << it->second->translation() << endl
//			<< "	real " << (*realRPMap)[id]->translation() << endl
			<< endl;
	}

	cout << "-------------------------------------------" << endl;

	/*
	Alignments al = TotemRPExtractAlignments::Extract(idealRPMap.product(), realRPMap.product());
	cout << "Extracted " << al.m_align.size() << " alignments." << endl;
	for (unsigned int i = 0; i < al.m_align.size(); i++) {
		AlignTransform &at = al.m_align[i];
		cout << "	ID           = " << at.rawId() << endl
			<<  "	shift        = " << at.translation() << endl
			<< 	"	euler angles = " << at.eulerAngles()
			<< endl;
	}

	cout << "-------------------------------------------" << endl;
	*/
}

DEFINE_FWK_MODULE(GeometryTestModule);
