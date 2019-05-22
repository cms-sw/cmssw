#include "Geometry/MTDNumberingBuilder/plugins/DDDCmsMTDConstruction.h"

#include <utility>
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDDetIdBuilder.h"

class DDNameFilter : public DDFilter {
public:
  void add(const std::string& add) { allowed_.emplace_back(add); } 
  void veto(const std::string& veto) { veto_.emplace_back(veto); }

  bool accept(const DDExpandedView & ev) const final {
    for( const auto& test : veto_ ) {
      if( ev.logicalPart().name().fullname().find(test) != std::string::npos )
        return false;
    }
    for( const auto& test : allowed_ ) {
      if( ev.logicalPart().name().fullname().find(test) != std::string::npos ) 
        return true;
    }
    return false;
  }
private:
  std::vector<std::string> allowed_;
  std::vector<std::string> veto_;
};

using namespace cms;

std::unique_ptr<GeometricTimingDet>
DDDCmsMTDConstruction::construct( const DDCompactView& cpv, std::vector<int> detidShifts)
{
  std::string attribute{"CMSCutsRegion"};
  DDNameFilter filter;
  filter.add("mtd:");
  filter.add("btl:");
  filter.add("etl:");
  filter.veto("service");
  filter.veto("support");
  filter.veto("FSide");
  filter.veto("BSide");
  filter.veto("LSide");
  filter.veto("RSide");
  filter.veto("Between");
  filter.veto("SupportPlate");
  filter.veto("Shield");
  
  DDFilteredView fv( cpv, filter ); 

  CmsMTDStringToEnum theCmsMTDStringToEnum;
  auto check_root = theCmsMTDStringToEnum.type( ExtractStringFromDDD::getString(attribute,&fv));
  if( check_root != GeometricTimingDet::MTD )
  {
    fv.firstChild();
    auto check_child = theCmsMTDStringToEnum.type( ExtractStringFromDDD::getString(attribute,&fv));
    if( check_child != GeometricTimingDet::MTD )
    {  
      throw cms::Exception( "Configuration" ) << " The first child of the DDFilteredView is not what is expected \n"
					      << ExtractStringFromDDD::getString( attribute, &fv ) << "\n";
    }
    fv.parent();
  }
  
  auto  mtd = std::make_unique<GeometricTimingDet>( &fv, GeometricTimingDet::MTD );
  CmsMTDBuilder theCmsMTDBuilder;
  theCmsMTDBuilder.build( fv, mtd.get(), attribute );
  
  return mtd;
}

