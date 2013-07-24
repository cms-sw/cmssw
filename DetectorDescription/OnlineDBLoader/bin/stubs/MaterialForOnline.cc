// Original Author:  Jie Chen
//         Created:  Thu Apr  5 10:36:22 CDT 2007
// $Id: MaterialForOnline.cc,v 1.8 2010/03/25 21:55:36 case Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include <Geometry/Records/interface/IdealGeometryRecord.h>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <istream>
#include <fstream>
#include <string>



//
// class decleration
//

class MaterialForOnline : public edm::EDAnalyzer {
   public:
      explicit MaterialForOnline(const edm::ParameterSet&);
      ~MaterialForOnline();
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

   private:

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MaterialForOnline::MaterialForOnline(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


MaterialForOnline::~MaterialForOnline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MaterialForOnline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "analyze does nothing" << std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
MaterialForOnline::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{
  std::string materialFileName("MATERIALS.dat");
  std::string elementaryMaterialFileName("ELEMENTARYMATERIALS.dat");
  std::string compositeMaterialFileName("COMPOSITEMATERIALS.dat");
  std::string materialFractionFileName("MATERIALFRACTIONS.dat");

  std::ofstream materialOS(materialFileName.c_str());
  std::ofstream elementaryMaterialOS(elementaryMaterialFileName.c_str());
  std::ofstream compositeMaterialOS(compositeMaterialFileName.c_str());
  std::ofstream materialFractionOS(materialFractionFileName.c_str());


  std::cout << "MaterialForOnline Analyzer..." << std::endl;
  edm::ESTransientHandle<DDCompactView> pDD;

  iSetup.get<IdealGeometryRecord>().get( "", pDD );

  //  const DDCompactView & cpv = *pDD;
  //DDCompactView::graph_type gra = cpv.graph();
  DDMaterial::iterator<DDMaterial> mit(DDMaterial::begin()), 
    med(DDMaterial::end());
  // PMaterial* pm;
  for (; mit != med; ++mit) {
    if (! mit->isDefined().second) continue;
    const DDMaterial& material = *mit;
    materialOS<<material.name()<<","<<material.density()/g*cm3
	      << std::endl;
    
    if(material.noOfConstituents()==0){//0 for elementary materials 
     elementaryMaterialOS<<material.name()<<","<<material.z()
			 <<","<<material.a()/g*mole
			 << std::endl;
    }
    else{//compound materials.  
      compositeMaterialOS<<material.name()
			 << std::endl;
      for (int i=0; i<material.noOfConstituents(); ++i) {
        DDMaterial::FractionV::value_type f = material.constituent(i);
	materialFractionOS<<material.name()<<","<<f.first.name()
			  <<","<< f.second
			  << std::endl;
      }                

    }

  }
  elementaryMaterialOS.close();
  compositeMaterialOS.close();
  materialFractionOS.close();
  materialOS.close();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MaterialForOnline::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MaterialForOnline);
