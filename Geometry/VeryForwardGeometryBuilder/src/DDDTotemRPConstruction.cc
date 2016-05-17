/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPConstruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"

// this might be useful one day
//.#include "Geometry/TrackerNumberingBuilder/interface/ExtractStringFromDDD.h"
//.#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerBuilder.h"
//.#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDetIdBuilder.h"

#include <iostream>


//----------------------------------------------------------------------------------------------------

DDDTotemRPContruction::DDDTotemRPContruction()
{
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc* DDDTotemRPContruction::construct(const DDCompactView* cpv)
{
	using namespace std;

	// create DDFilteredView and apply the filter
	DDFilteredView fv(*cpv);
	//.fv.addFilter(filter);

	// conversion to DetGeomDesc structure
	// create the root node and recursively propagates through the tree
	// adds IDs
	DetGeomDesc* tracker = new DetGeomDesc(&fv);
	buildDetGeomDesc(&fv, tracker);

	// return the root of the structure
	return tracker;
}

//----------------------------------------------------------------------------------------------------

void DDDTotemRPContruction::buildDetGeomDesc(DDFilteredView *fv, DetGeomDesc *gd)
{
	using namespace std;

	// try to dive into next level
	if (! fv->firstChild()) return;

	// loop over siblings in the level
	do {
		// create new DetGeomDesc node and add it to the parent's (gd) list
		DetGeomDesc* newGD = new DetGeomDesc(fv);

		// add ID (only for detectors)
		if (! fv->logicalPart().name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME)) {
			const vector<int> &cN = fv->copyNumbers();
			// check size of copy numubers array
			if (cN.size() < 3)
				throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for RP_Silicon_Detector is " << cN.size() << ". It must be >= 3." << endl;

			// extract information
			unsigned int A = cN[cN.size() - 3];
			unsigned int arm = A / 100;
			unsigned int station = (A % 100) / 10;
			unsigned int rp = A % 10;
			unsigned int detector = cN[cN.size() - 1];
      //.std::cout<<"arm:"<<arm<<", station:"<<station<<", rp:"<<rp<<", detector:"<<detector<<std::endl;
      //.std::cout<<"TotemRPDetId(arm, station, rp, detector) "<<TotemRPDetId(arm, station, rp, detector).DetectorDecId()<<", "<<TotemRPDetId(arm, station, rp, detector).rawId()<<std::endl;
			newGD->setGeographicalID(TotemRPDetId(arm, station, rp, detector));
			//.cout << "A = " << A << "; arm = " << arm << " st = " << station << " rp = " << rp << " det = " << detector << " --> "<< gd->geographicalID().rawId() << endl;
		}

		gd->addComponent(newGD);

		// recursion
		buildDetGeomDesc(fv, newGD);
	} while (fv->nextSibling());

	// go a level up
	fv->parent();
}
