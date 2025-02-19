/** \file MaterialProperties.cc
 *  
 *
 *  $Date: 2009/05/12 07:17:48 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/MaterialProperties.h"
#include "G4LogicalVolumeStore.hh" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

	MaterialProperties::MaterialProperties(int DebugLevel, double SiAbsLengthScale) 
	: theMaterialTable(), theMPDebugLevel(0), 
	theSiAbsLengthScalingFactor(0),
	theMPT(),
	theTECWafer(), theTOBWafer(), theTIBWafer()
{
	theMPDebugLevel = DebugLevel;
	theSiAbsLengthScalingFactor = SiAbsLengthScale;
	/* *********************************************************************** */
	/* 
	define the materials for the sensitive detectors in TEC, TIB and TOB
		we need this to specify different properties for Barrel and Endcap
		detectors, which is by default no longer possible in CMSSW due to
		the fact that all modules are made out of the same G4Material 
	*/
	/* *********************************************************************** */
		G4double theDensity = 2.33*g/cm3;
	G4double theAtomicWeight = 28.09*g/mole;
	G4double theAtomicNumber = 14.0;

	theTECWafer = new G4Material("TEC_Wafer", theAtomicNumber, theAtomicWeight, theDensity);
	theTOBWafer = new G4Material("TOB_Wafer", theAtomicNumber, theAtomicWeight, theDensity);
	theTIBWafer = new G4Material("TIB_Wafer", theAtomicNumber, theAtomicWeight, theDensity);

	// set the properties of the materials
	setMaterialProperties();
}

MaterialProperties::~MaterialProperties()
{
	if ( theMPT != 0 )                  { delete theMPT; }
	if ( theTECWafer != 0 )             { delete theTECWafer; }
	if ( theTOBWafer != 0 )             { delete theTOBWafer; }
	if ( theTIBWafer != 0 )             { delete theTIBWafer; }
}


void MaterialProperties::setMaterialProperties()
{
	/* *********************************************************************** */
	/* 
	use this function to define material properties (like refraction
		index, absorptionlenght and so on) and add them to the 
		MaterialPropertiesTable. Finally set the MPT to a give Material.
	*/
	/* *********************************************************************** */

	// get the MaterialTable as it is defined in OSCAR (CMSSW now)
		theMaterialTable = G4Material::GetMaterialTable();

	/* *********************************************************************** */
	/*
	with the following code one can access the MaterialTable defined in
		OSCAR. This contains all the materials needed for the CMS detector,
		which are defined in the Geometry and DDD packages. COBRA takes care 
		of the proper conversion between a DDMaterial and a G4Material.
	*/
	/* *********************************************************************** */

//   if (theMPDebugLevel > 1)
//     {
			// print the materialtable
		LogDebug("SimLaserAlignment:MaterialProperties") << " **** here comes the material table **** "
		<< *(G4Material::GetMaterialTable());
//     }

	// define the MateriapropertiesTable for the Sensitive Regions in the Tracker
	// TOB_Wafer, TOB_Silicon, TID_Wafer, TIB_Wafer and TEC_Wafer

	const G4int nEntries = 3;

	// Photon energies
	G4double PhotonEnergy[nEntries] = { 1.10 * eV, 1.15 * eV, 1.20 * eV };

	// scintillation
	G4double Scintillation[nEntries] = { 0.1, 1.0, 0.1 };

	// Refractive Index
	G4double RefractiveIndex[nEntries] = { 3.5400, 3.5425, 3.5450 };
	// Refractive Index of the Mirrors (BK7)
	G4double RefractiveIndexMirror[nEntries] = { 1.50669, 1.50669, 1.50669 };

	/* *********************************************************************** */
	/*  set the refractive index for the other materials to 1.0. This is       *
	*  needed to propagate the optical photons through the detector according *
		*  to Peter Gumplinger.                                                   */
	/* *********************************************************************** */
		G4double RefractiveIndexGeneral[nEntries] = { 1.0, 1.0, 1.0 };

	// Absorption Length
	// G4double AbsorptionLengthSi[nEntries] = { 198.8 * micrometer, 198.8 * micrometer, 198.8 * micrometer }; ///////////////////////////////////
	G4double AbsorptionLengthSi[nEntries] = { 1136 * micrometer, 1136 * micrometer, 1136 * micrometer };

	G4double AbsorptionLengthSiBarrel[nEntries] = { 0.1 * fermi, 
		0.1 * fermi, 
		0.1 * fermi };


	// Absorption length of the mirrors
	G4double AbsorptionLengthMirror[nEntries] = { 11.7 * cm, 0.5 * 11.7 * cm, 11.7 * cm };

	// Absorption Length for dead material in the tracker; set to small values
	// to kill the optical photons outside the TEC. Maybe this is later a problem
	// when implementing Ray 1 to connect both TECs which eachother and with TIB
	// and TOB!??
	G4double AbsorptionLengthDead[nEntries] = { 0.001 * micrometer, 0.001 * micrometer,
		0.001 * micrometer };

	// Absorption Length of the other Materials in the Tracker
	G4double AbsorptionLengthGeneral[nEntries] = { 75 * cm, 75 * cm, 75 * cm };

	// Absorption Length of the Air in the Tracker
	G4double AbsorptionLengthTAir[nEntries] = { 10 * m, 1.8 * m, 10 * m };

	G4double AbsorptionLengthAl[nEntries] = { 10 * mm, 10 * mm, 10 * mm};
	G4double AbsorptionLengthTOB_CF_Str[nEntries] = { 1 * cm, 10 * cm, 1 * cm };
	G4double AbsorptionLengthTOBCF[nEntries] = { 0.1 * mm, 20 * mm, 0.1 * mm };
	G4double AbsorptionLengthTIBCF[nEntries] = { 15.0 * cm, 15.0 * cm, 15.0 * cm };

	// Reflectivity of the modules
	G4double SiReflectivity[nEntries] = { 0.0, 0.0, 0.0 };

	// Efficiency of the modules
	G4double TECEfficiency[nEntries] = { 0.9, 0.9, 0.9 };
	G4double BarrelEfficiency[nEntries] = { 1.0, 1.0, 1.0 };

	// Reflectivity of the mirrors in the Alignment Tubes
	G4double Reflectivity[nEntries] = { 0.05, 0.05, 0.05 };

	/* *********************************************************************** */

	/* *********************************************************************** */
	/* 
	define the materials for the sensitive detectors in TEC, TIB and TOB
		we need this to specify different properties for Barrel and Endcap
		detectors, which is by default no longer possible in CMSSW due to
		the fact that all modules are made out of the same G4Material 
	*/
	/* *********************************************************************** */

	// set the options for the materials 
	{
		for(G4MaterialTable::const_iterator theMTEntry = theMaterialTable->begin();
		theMTEntry != theMaterialTable->end(); theMTEntry++) 
		{
			if(*theMTEntry)
			{
				G4Material * theMaterial = const_cast<G4Material*>(*theMTEntry);

				if (theMaterial->GetMaterialPropertiesTable())
				{ 
					theMPT = theMaterial->GetMaterialPropertiesTable(); 
				}
				else
				{ 
					theMPT = new G4MaterialPropertiesTable; 
				}

			// properties of the TEC_Wafer
				if ( theMaterial->GetName() == "TEC_Wafer" )
				{
					theMPT->AddProperty("FASTCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("SLOWCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndex, nEntries);
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthSi, nEntries);
					theMPT->AddProperty("EFFICIENCY", PhotonEnergy, TECEfficiency, nEntries);

					theMPT->AddConstProperty("SCINTILLATIONYIELD", 12000.0/MeV);
					theMPT->AddConstProperty("RESOLTUIONSCALE", 1.0);
					theMPT->AddConstProperty("FASTTIMECONSTANT", 20.0 * ns);
					theMPT->AddConstProperty("SLOWTIMECONSTANT", 45.0 * ns);
					theMPT->AddConstProperty("YIELDRATIO", 1.0);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of Silicon (used as Module Material in CMSSW)
				else if ( theMaterial->GetName() == "Silicon" )
				{
					theMPT->AddProperty("FASTCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("SLOWCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndex, nEntries);
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthSi, nEntries);
					theMPT->AddProperty("EFFICIENCY", PhotonEnergy, TECEfficiency, nEntries);

					theMPT->AddConstProperty("SCINTILLATIONYIELD", 12000.0/MeV);
					theMPT->AddConstProperty("RESOLTUIONSCALE", 1.0);
					theMPT->AddConstProperty("FASTTIMECONSTANT", 20.0 * ns);
					theMPT->AddConstProperty("SLOWTIMECONSTANT", 45.0 * ns);
					theMPT->AddConstProperty("YIELDRATIO", 1.0);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of the TOB_Wafer, TOB_Silicon, TIB_Wafer
				else if ( ( theMaterial->GetName() == "TOB_Wafer" ) || 
					( theMaterial->GetName() == "TIB_Wafer" ) )
				{
					theMPT->AddProperty("FASTCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("SLOWCOMPONENT", PhotonEnergy, Scintillation, nEntries);
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndex, nEntries);
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthSiBarrel, nEntries);
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
					theMPT->AddProperty("EFFICIENCY", PhotonEnergy, BarrelEfficiency, nEntries);

					theMPT->AddConstProperty("SCINTILLATIONYIELD", 12000.0/MeV);
					theMPT->AddConstProperty("RESOLTUIONSCALE", 1.0);
					theMPT->AddConstProperty("FASTTIMECONSTANT", 20.0 * ns);
					theMPT->AddConstProperty("SLOWTIMECONSTANT", 45.0 * ns);
					theMPT->AddConstProperty("YIELDRATIO", 1.0);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of the TIB_ledge_side
				else if ( theMaterial->GetName() == "TIB_ledge_side" )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthGeneral, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of air
				else if ( ( theMaterial->GetName() == "T_Air" ) ||
					( theMaterial->GetName() == "Air" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthTAir, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of some materials in the Barrel
			// used to absorb photons to avoid hits in other TEC
				else if ( ( theMaterial->GetName() == "TIB_connector" ) ||  
					( theMaterial->GetName() == "TIB_cylinder" ) || 
					( theMaterial->GetName() == "TID_Connector" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthDead, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of SiO2; used for the mirrors of the Alignment Tubes
				else if ( theMaterial->GetName() == "Si O_2" )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy,RefractiveIndexMirror, nEntries);
		//set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthMirror, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY",PhotonEnergy,Reflectivity, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of Aluminium
				else if ( ( theMaterial->GetName() == "TOB_Aluminium" ) || 
					( theMaterial->GetName() == "Aluminium" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthAl, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of TOB_CF_Str
				else if ( ( theMaterial->GetName() == "TOB_CF_Str" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthTOB_CF_Str, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// some other Tracker materials
				else if ( ( theMaterial->GetName() == "TID_CF" ) ||
					( theMaterial->GetName() == "Nomex" ) || 
					( theMaterial->GetName() == "TOB_Nomex" ) ||
					( theMaterial->GetName() == "TID_Nomex" ) ||
					( theMaterial->GetName() == "TOB_plate_C" ) ||
					( theMaterial->GetName() == "TOB_rod" ) || 
					( theMaterial->GetName() == "TOB_cool_DS" ) ||
					( theMaterial->GetName() == "TOB_cool_SS" ) ||
					( theMaterial->GetName() == "TID_in_cable" ) ||
					( theMaterial->GetName() == "TOB_PA_rphi" ) ||
					( theMaterial->GetName() == "TOB_frame_ele" ) ||
					( theMaterial->GetName() == "TOB_PA_ster" ) ||
					( theMaterial->GetName() == "TOB_ICB" ) ||
					( theMaterial->GetName() == "TOB_CONN1" ) || 
					( theMaterial->GetName() == "TOB_CONN2" ) || 
					( theMaterial->GetName() == "TOB_CONN3" ) || 
					( theMaterial->GetName() == "TOB_rail" ) ||
					( theMaterial->GetName() == "TOB_sid_rail1" ) ||
					( theMaterial->GetName() == "TOB_sid_rail2" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthTOBCF, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of some TIB materials
				else if ( ( theMaterial->GetName() == "TIB_CF" ) || 
					( theMaterial->GetName() == "TIB_cables_ax_out" ) ||
					( theMaterial->GetName() == "TIB_outer_supp" ) ||
					( theMaterial->GetName() == "TIB_PA_rphi" ) ||
					( theMaterial->GetName() == "TIB_rail" ) ||
					( theMaterial->GetName() == "TIB_sid_rail1" ) ||
					( theMaterial->GetName() == "TIB_sid_rail2" ) ||
					( theMaterial->GetName() == "TIB_mod_cool" ) )
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the reflectivity
					theMPT->AddProperty("REFLECTIVITY", PhotonEnergy, SiReflectivity, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthTIBCF, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			// properties of all other materials in the detector
				else
				{
		// set the refractive index
					theMPT->AddProperty("RINDEX", PhotonEnergy, RefractiveIndexGeneral, nEntries);
		// set the absorptionlength
					theMPT->AddProperty("ABSLENGTH", PhotonEnergy, AbsorptionLengthGeneral, nEntries);

		// set the MaterialPropertiesTable
					theMaterial->SetMaterialPropertiesTable(theMPT);
				}

			}
		}
	}

	// loop over the logical volumes and set the material for the sensitive detectors
		const G4LogicalVolumeStore * theLogicalVolumeStore = G4LogicalVolumeStore::GetInstance();
	std::vector<G4LogicalVolume*>::const_iterator theLogicalVolume;

	for ( theLogicalVolume = theLogicalVolumeStore->begin(); theLogicalVolume != theLogicalVolumeStore->end(); theLogicalVolume++ )
	{
		if ( ( (*theLogicalVolume)->GetName() == "TECModule0StereoActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule0RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule1StereoActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule1RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule2RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule3RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule4StereoActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule4RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule5RphiActive" ) ||
			( (*theLogicalVolume)->GetName() == "TECModule6RphiActive" ) )
		{
		// set the material
			(*theLogicalVolume)->SetMaterial(theTECWafer);

			if (theMPDebugLevel > 2)
			{
				std::cout << "  AC1CMS: found a logical volume: " << (*theLogicalVolume)->GetName() << std::endl;
				std::cout << "  AC1CMS: the logical volume material = " << (*theLogicalVolume)->GetMaterial()->GetName() << std::endl;
				std::cout << "  AC1CMS: the MaterialPropertiesTable = " << std::endl;
				(*theLogicalVolume)->GetMaterial()->GetMaterialPropertiesTable()->DumpTable();
			}
		}
		else if ( ( (*theLogicalVolume)->GetName() == "TOBActiveSter0" ) ||
			( (*theLogicalVolume)->GetName() == "TOBActiveRphi0" ) ||
			( (*theLogicalVolume)->GetName() == "TOBActiveRphi2" ) ||
			( (*theLogicalVolume)->GetName() == "TOBActiveRphi4" ) )
		{
		// set the material
			(*theLogicalVolume)->SetMaterial(theTOBWafer);

			if (theMPDebugLevel > 2)
			{
				std::cout << "  AC1CMS: found a logical volume: " << (*theLogicalVolume)->GetName() << std::endl;
				std::cout << "  AC1CMS: the logical volume material = " << (*theLogicalVolume)->GetMaterial()->GetName() << std::endl;
				std::cout << "  AC1CMS: the MaterialPropertiesTable = " << std::endl;
				(*theLogicalVolume)->GetMaterial()->GetMaterialPropertiesTable()->DumpTable();
			}
		}
		else if ( ( (*theLogicalVolume)->GetName() == "TIBActiveSter0" ) ||
			( (*theLogicalVolume)->GetName() == "TIBActiveRphi0" ) ||
			( (*theLogicalVolume)->GetName() == "TIBActiveRphi2" ) )
		{
		// set the material
			(*theLogicalVolume)->SetMaterial(theTIBWafer);

			if (theMPDebugLevel > 2)
			{
				std::cout << "  AC1CMS: found a logical volume: " << (*theLogicalVolume)->GetName() << std::endl;
				std::cout << "  AC1CMS: the logical volume material = " << (*theLogicalVolume)->GetMaterial()->GetName() << std::endl;
				std::cout << "  AC1CMS: the MaterialPropertiesTable = " << std::endl;
				(*theLogicalVolume)->GetMaterial()->GetMaterialPropertiesTable()->DumpTable();
			}
		}
	}
}
