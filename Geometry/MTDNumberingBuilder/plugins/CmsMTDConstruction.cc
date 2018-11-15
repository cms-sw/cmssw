#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

void CmsMTDConstruction::buildComponent(DDFilteredView& fv, 
					GeometricTimingDet *mother, 
					std::string attribute){
  
  //
  // at this level I check whether it is a merged detector or not
  //

  GeometricTimingDet * det  = new GeometricTimingDet(&fv,theCmsMTDStringToEnum.type(fv.logicalPart().name().fullname()));

  const std::string part_name = fv.logicalPart().name().fullname().substr(0,11);
  
  if ( theCmsMTDStringToEnum.type(part_name) ==  GeometricTimingDet::BTLModule ) {
    bool dodets = fv.firstChild(); 
    while (dodets) {
      buildBTLModule(fv,det,attribute);
      dodets = fv.nextSibling(); 
    }
    fv.parent();
  } else if ( theCmsMTDStringToEnum.type(part_name) ==  GeometricTimingDet::ETLModule ) {  
    bool dodets = fv.firstChild(); 
    while (dodets) {
      buildETLModule(fv,det,attribute);
      dodets = fv.nextSibling(); 
    }
    fv.parent();
  } else {
    throw cms::Exception("MTDConstruction") << "woops got: " << part_name << std::endl;
  }
  
  mother->addComponent(det);
}

void CmsMTDConstruction::buildBTLModule(DDFilteredView& fv,
					GeometricTimingDet *mother,
					const std::string& attribute){

  GeometricTimingDet * det  = new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ));
  
  const auto& copyNumbers = fv.copyNumbers();
  auto module_number = copyNumbers[copyNumbers.size()-2];
  
  constexpr char positive[] = "PositiveZ";
  constexpr char negative[] = "NegativeZ";

  const std::string modname = fv.logicalPart().name().fullname();
  size_t delim1 = modname.find("BModule");
  size_t delim2 = modname.find("Layer");
  module_number += atoi(modname.substr(delim1+7,delim2).c_str())-1;

  if( modname.find(positive) != std::string::npos ) {
    det->setGeographicalID(BTLDetId(1,copyNumbers[copyNumbers.size()-3],module_number,0,1));
  } else if ( modname.find(negative) != std::string::npos ) {
    det->setGeographicalID(BTLDetId(0,copyNumbers[copyNumbers.size()-3],module_number,0,1));
  } else {
    throw cms::Exception("CmsMTDConstruction::buildBTLModule") 
      << "BTL Module " << module_number << " is neither positive nor negative in Z!";
  }
  
  mother->addComponent(det);
}

void CmsMTDConstruction::buildETLModule(DDFilteredView& fv,
					GeometricTimingDet *mother,
					const std::string& attribute){

  GeometricTimingDet * det  = new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)));
  
  const auto& copyNumbers = fv.copyNumbers();
  auto module_number = copyNumbers[copyNumbers.size()-2];
  
  size_t delim_ring = det->name().fullname().find("EModule");
  size_t delim_disc = det->name().fullname().find("Disc");
  
  std::string ringN = det->name().fullname().substr(delim_ring+7,delim_disc);

  const uint32_t side = det->translation().z() > 0 ? 1 : 0;
  
  // label geographic detid is front or back (even though it is one module per entry here)  
  det->setGeographicalID(ETLDetId(side,atoi(ringN.c_str()),module_number,0)); 
  
  mother->addComponent(det);
}
