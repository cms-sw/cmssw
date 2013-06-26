#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"


//#define LOCAL_DEBUG

MuonDDDConstants::MuonDDDConstants( const DDCompactView& cpv ) {
#ifdef LOCAL_DEBUG
  std::cout << "MuonDDDConstants;:MuonDDDConstants ( const DDCompactView& cpv ) constructor " << std::endl;
#endif
  std::string attribute = "OnlyForMuonNumbering"; 
  std::string value     = "any";
  DDValue val(attribute, value, 0.0);
  
  DDSpecificsFilter filter;
  filter.setCriteria(val,
		     DDSpecificsFilter::not_equals,
		     DDSpecificsFilter::AND, 
		     true, // compare strings otherwise doubles
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(cpv);
  fview.addFilter(filter);
  
  DDValue val2("level");
  const DDsvalues_type params(fview.mergedSpecifics());
  
  fview.firstChild();
  
  const DDsvalues_type mySpecs (fview.mergedSpecifics());
#ifdef LOCAL_DEBUG
  std::cout << "mySpecs.size() = " << mySpecs.size() << std::endl;
#endif
  if ( mySpecs.size() < 25 ) {
    edm::LogError("MuonDDDConstants") << " MuonDDDConstants: Missing SpecPars from DetectorDescription." << std::endl;
    std::string msg = "MuonDDDConstants does not have the appropriate number of SpecPars associated";
    msg+= " with the part //MUON.";
    throw cms::Exception("GeometryBuildFailure", msg);
  }

  DDsvalues_type::const_iterator bit = mySpecs.begin();
  DDsvalues_type::const_iterator eit = mySpecs.end();
  for ( ; bit != eit; ++bit ) {
    if ( bit->second.isEvaluated() ) {
      this->addValue( bit->second.name(), int(bit->second.doubles()[0]) );
#ifdef LOCAL_DEBUG
      std::cout << "adding DDConstant of " << bit->second.name() << " = " << int(bit->second.doubles()[0]) << std::endl;
#endif
    }
    //    std::cout << "DDConstant of " << bit->second.name() << " = " << bit->second.strings()[0] << std::endl;
  }
  
}

MuonDDDConstants::~MuonDDDConstants() { 
  //  std::cout << "destructed!!!" << std::endl;
}

int MuonDDDConstants::getValue( const std::string& name ) const {
#ifdef LOCAL_DEBUG
  std::cout << "about to look for ... " << name << std::endl;
#endif
  if ( namesAndValues_.size() == 0 ) {
    std::cout << "MuonDDDConstants::getValue HAS NO VALUES!" << std::endl;
    throw cms::Exception("GeometryBuildFailure", "MuonDDDConstants does not have requested value for " + name);
  }

  std::map<std::string, int>::const_iterator findIt = namesAndValues_.find(name);

  if ( findIt == namesAndValues_.end() ) {
    std::cout << "MuonDDDConstants::getValue was asked for " << name << " and had NO clue!" << std::endl;
    throw cms::Exception("GeometryBuildFailure", "MuonDDDConstants does not have requested value for " + name);
  }

  return findIt->second;
}

void MuonDDDConstants::addValue(const std::string& name, const int& value) {
  namesAndValues_[name] = value;
}

