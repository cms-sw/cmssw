#include "CondTools/SiPixel/test/SiPixel2DTemplateDBObjectUploader.h"
#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <fstream>
#include <stdio.h>
#include <iostream>

using namespace std;

SiPixel2DTemplateDBObjectUploader::SiPixel2DTemplateDBObjectUploader(const edm::ParameterSet& iConfig):
	theTemplateCalibrations( iConfig.getParameter<vstring>("siPixelTemplateCalibrations") ),
	theTemplateBaseString( iConfig.getParameter<std::string>("theTemplateBaseString") ),
	theVersion( iConfig.getParameter<double>("Version") ),
	theMagField( iConfig.getParameter<double>("MagField") ),
	theTemplIds( iConfig.getParameter<std::vector<uint32_t> >("templateIds") )
{
}


SiPixel2DTemplateDBObjectUploader::~SiPixel2DTemplateDBObjectUploader()
{
}

void 
SiPixel2DTemplateDBObjectUploader::beginJob()
{
}

void
SiPixel2DTemplateDBObjectUploader::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
	//--- Make the POOL-ORA object to store the database object
	SiPixel2DTemplateDBObject* obj = new SiPixel2DTemplateDBObject;

	// Local variables 
	const char *tempfile;
	int m;
	
	// Set the number of templates to be passed to the dbobject
	obj->setNumOfTempl(theTemplateCalibrations.size());
	cout << obj->numOfTempl() << endl;
	cout << theVersion << endl;
	// Set the version of the template dbobject - this is an external parameter
	obj->setVersion(theVersion);

	// Open the template file(s) 
	for(m=0; m< obj->numOfTempl(); ++m){

		edm::FileInPath file( theTemplateCalibrations[m].c_str() );
		tempfile = (file.fullPath()).c_str();

		std::ifstream in_file(tempfile, std::ios::in);
			
		if(in_file.is_open()){
			//edm::LogInfo("Template Info") << "Opened Template File: " << file.fullPath().c_str();
			cout << "Opened Template File: " << file.fullPath().c_str() << "\n";

			// Local variables 
			char title_char[80], c;
			SiPixel2DTemplateDBObject::char2float temp;
			float tempstore;
			int iter,j;
			
			// Templates contain a header char - we must be clever about storing this
			for (iter = 0; (c=in_file.get()) != '\n'; ++iter) {
				if(iter < 79) {title_char[iter] = c;}
			}
			if(iter > 78) {iter=78;}
			title_char[iter+1] ='\n';
			
			for(j=0; j<80; j+=4) {
				temp.c[0] = title_char[j];
				temp.c[1] = title_char[j+1];
				temp.c[2] = title_char[j+2];
				temp.c[3] = title_char[j+3];
				obj->push_back(temp.f);
				obj->setMaxIndex(obj->maxIndex()+1);
			}
			
			// Fill the dbobject
			in_file >> tempstore;
			while(!in_file.eof()) {
				obj->setMaxIndex(obj->maxIndex()+1);
				obj->push_back(tempstore);
				in_file >> tempstore;
			}
			
			in_file.close();
		}
		else {
			// If file didn't open, report this
			//edm::LogError("SiPixel2DTemplateDBObjectUploader") << "Error opening File" << tempfile;
			cout << "Error opening File " << tempfile << "\n";
		}
	}

        //Retrieve tracker topology from geometry
        edm::ESHandle<TrackerTopology> tTopoHandle;
        es.get<TrackerTopologyRcd>().get(tTopoHandle);
        const TrackerTopology* const tTopo = tTopoHandle.product();
	
	edm::ESHandle<TrackerGeometry> pDD;
	es.get<TrackerDigiGeometryRecord>().get( pDD );

	short templids[52];
	for(int k = 0; k < 52; k++){
	templids[k] = (short) theTemplIds[k];
	}

	for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){	
		if( (*it)!=0){
			// Here is the actual looping step over all DetIds:				
			DetId detid=(*it)->geographicalId();
                        const DetId detidc = (*it)->geographicalId();

			unsigned int layer=0, disk=0, side=0, blade=0, panel=0, module=0;
					
			// Now we sort them into the Barrel and Endcap:
			if(detid.subdetId() == 1) {

                                layer=tTopo->pxbLayer(detidc.rawId());
                                module=tTopo->pxbModule(detidc.rawId());

				if(detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)){
					if (layer == 1) {
						if (module == 1) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[0] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";						
						}
						if (module == 2) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[1] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 3) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[2] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 4) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[3] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 5) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[4] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 6) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[5] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 7) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[6] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 8) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[7] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
					
					}
					if (layer == 2) {
						if (module == 1) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[8] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";		
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 2) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[9] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 3) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[10] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 4) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[11] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 5) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[12] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 6) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[13] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 7) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[14] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";											
						}
						if (module == 8) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[15] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
					
					}
					if (layer == 3) {
						if (module == 1) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[16] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 2) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[17] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 3) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[18] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 4) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[19] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 5) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[20] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 6) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[21] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";	
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 7) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[22] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";										
						}
						if (module == 8) {
							if ( ! (*obj).putTemplateID( detid.rawId(),templids[23] ) )
							//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
							cout << " Could not fill barrel layer "<<layer<<", module "<<module<<"\n";									
						}
					}
				// ----- debug:
				//cout<<"This is a barrel element with: layer "<<layer<<", ladder "<<ladder<<" and module "<<module<<".\n"; //Uncomment to read out exact position of each element.
				// -----
				}
			}
			if(detid.subdetId() == 2) {

                                disk=tTopo->pxfDisk(detidc.rawId()); //1,2,3
                                blade=tTopo->pxfBlade(detidc.rawId()); //1-24
                                side=tTopo->pxfSide(detidc.rawId()); //size=1 for -z, 2 for +z
                                panel=tTopo->pxfPanel(detidc.rawId()); //panel=1,2
                                module=tTopo->pxfModule(detidc.rawId()); // plaquette

				//short temp123abc = (short) theTemplIds[1];
				if(detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)){
					if (side ==1 ){
						if (disk == 1){
							if(panel == 1){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[24] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[25] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[26] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 4){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[27] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
							if(panel == 2){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[40] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[41] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[42] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
						}
						if (disk == 2){
							if(panel == 1){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[28] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[29] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[30] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 4){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[31] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}	
							if(panel == 2){
								if(module == 1){
										if ( ! (*obj).putTemplateID( detid.rawId(),templids[43] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[44] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[45] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
						}
					}						
					if (side ==2 ){
						if (disk == 1){
							if(panel == 1){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[32] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[33] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[34] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 4){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[35] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
							if(panel == 2){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[46] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[47] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[48] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
						}
						if (disk == 2){
							if(panel == 1){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[36] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[37] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[38] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 4){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[39] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}	
							if(panel == 2){
								if(module == 1){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[49] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 2){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[50] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
								if(module == 3){
									if ( ! (*obj).putTemplateID( detid.rawId(),templids[51] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									cout << " Could not fill barrel det unit"<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n";
								}
							}
						}
					}
						
				// ----- debug:
				//cout<<"This is an endcap element with: side "<<side<<", disk "<<disk<<", blade "<<blade<<", panel "<<panel<<" and module "<<module<<".\n"; //Uncomment to read out exact position of each element.
				// -----
				}
			}


			//cout<<"The DetID: "<<detid.rawId()<<" is mapped to the template: "<<mapnum<<".\n\n";

			//else 
				//if ( ! (*obj).putTemplateID( detid.rawId(),templids[1] ) )
									//edm::LogInfo("Template Info") << " Could not fill barrel det unit";
									//cout << "ERROR! OH NO!\n";
		}
	}

	//--- Create a new IOV
	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	if( !poolDbService.isAvailable() ) // Die if not available
		throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
	if(poolDbService->isNewTagRequest("SiPixel2DTemplateDBObjectRcd"))
		poolDbService->writeOne( obj, poolDbService->beginOfTime(), "SiPixel2DTemplateDBObjectRcd");
	else
		poolDbService->writeOne( obj, poolDbService->currentTime(), "SiPixel2DTemplateDBObjectRcd");
}

void SiPixel2DTemplateDBObjectUploader::endJob()
{
}

