#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <string>
#include <iostream>
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"


//#define LOCAL_DEBUG

MuonDDDConstants::MuonDDDConstants(){ }

MuonDDDConstants::MuonDDDConstants( const DDCompactView& cpv ) {
  //  std::cout << "MuonDDDConstants;:MuonDDDConstants ( const DDCompactView& cpv ) constructor " << std::endl;
  try {
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
    DDsvalues_type::const_iterator bit = mySpecs.begin();
    DDsvalues_type::const_iterator eit = mySpecs.end();
    for ( ; bit != eit; ++bit ) {
      if ( bit->second.isEvaluated() ) {
	this->addValue( bit->second.name(), int(bit->second.doubles()[0]) );
	//	std::cout << "adding DDConstant of " << bit->second.name() << " = " << int(bit->second.doubles()[0]) << std::endl;
      }
    }

  }
  catch (const DDException & e ) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants caught a DDD Exception: " << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const std::exception & e) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants : an std::exception occured: " << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants : An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}


MuonDDDConstants::~MuonDDDConstants() { 
  //  std::cout << "destructed!!!" << std::endl;
};

int MuonDDDConstants::getValue( const std::string& name ) const {
  //  std::cout << "about to look for ... " << name << std::endl;

  if ( namesAndValues_.size() == 0 ) {
    std::cout << "MuonDDDConstants::getValue HAS NO VALUES!" << std::endl;
    throw;
  }

  std::map<std::string, int>::const_iterator findIt = namesAndValues_.find(name);

  if ( findIt == namesAndValues_.end() ) {
    std::cout << "MuonDDDConstants::getValue was asked for " << name << " and had NO clue!" << std::endl;
    throw;
  }

  return findIt->second;
}

void MuonDDDConstants::addValue(const std::string& name, const int& value) {
  namesAndValues_[name] = value;
}

