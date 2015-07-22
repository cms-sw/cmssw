#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"
#include "Geometry/HcalCommonData/interface/HcalGeomParameters.h"
#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

//#define DebugLog

namespace {
  double getTopologyMode( const char* s, const DDsvalues_type & sv ) {
    DDValue val( s );
    if( DDfetch( &sv, val )) {
      const std::vector<std::string> & fvec = val.strings();
      if( fvec.size() == 0 ) {
	throw cms::Exception( "HcalParametersFromDD" ) << "Failed to get " << s << " tag.";
      }

      StringToEnumParser<HcalTopologyMode::Mode> eparser;
      HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode) eparser.parseString( fvec[0] );

      return double( mode );
    } else {
      throw cms::Exception( "HcalParametersFromDD" ) << "Failed to get "<< s << " tag.";
    }
  }
}

bool HcalParametersFromDD::build(const DDCompactView* cpv,
				 HcalParameters& php) {

  //Special parameters at simulation level
  std::string attribute = "OnlyForHcalSimNumbering"; 
  std::string value     = "any";
  DDValue val1(attribute, value, 0.0);
  DDSpecificsFilter filter1;
  filter1.setCriteria(val1, DDCompOp::not_equals,
		      DDLogOp::AND, true, // compare strings otherwise doubles
		      true  // use merged-specifics or simple-specifics
		      );
  DDFilteredView fv1(*cpv);
  fv1.addFilter(filter1);
  bool ok = fv1.firstChild();

  const int nEtaMax=100;

  if (ok) {
    HcalGeomParameters *geom = new HcalGeomParameters();
    geom->loadGeometry( fv1, php );
    php.modHB  = geom->getModHalfHBHE(0);
    php.modHE  = geom->getModHalfHBHE(1);
    php.dzVcal = geom->getConstDzHF();
    geom->getConstRHO(php.rHO);

    php.phioff   = DDVectorGetter::get( "phioff" );
    php.etaTable = DDVectorGetter::get( "etaTable" );
    php.rTable   = DDVectorGetter::get( "rTable" );
    php.phibin   = DDVectorGetter::get( "phibin" );
    php.phitable = DDVectorGetter::get( "phitable" );  
    for (unsigned int i = 1; i<=nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupSimEta" << i;
      std::string tempName = sstm.str();
      if (DDVectorGetter::check(tempName)) {
	HcalParameters::LayerItem layerGroupEta;
	layerGroupEta.layer = i;
	layerGroupEta.layerGroup = dbl_to_int(DDVectorGetter::get(tempName));
	php.layerGroupEtaSim.push_back(layerGroupEta);
      }
    }
    php.etaMin   = dbl_to_int( DDVectorGetter::get( "etaMin" ));
    php.etaMax   = dbl_to_int( DDVectorGetter::get( "etaMax" ));
    php.etaMin[0] = 1;
    php.etaMax[1] = (int)(php.etaTable.size())-1;
    php.etaMax[2] = php.etaMin[2]+(int)(php.rTable.size())-2;
    php.etaRange = DDVectorGetter::get( "etaRange" );
    php.gparHF   = DDVectorGetter::get( "gparHF" );
    php.noff     = dbl_to_int( DDVectorGetter::get( "noff" ));
    php.Layer0Wt = DDVectorGetter::get( "Layer0Wt" );  
    php.HBGains  = DDVectorGetter::get( "HBGains" );
    php.HBShift  = dbl_to_int( DDVectorGetter::get( "HBShift" ));
    php.HEGains  = DDVectorGetter::get( "HEGains" );
    php.HEShift  = dbl_to_int( DDVectorGetter::get( "HEShift" ));
    php.HFGains  = DDVectorGetter::get( "HFGains" );
    php.HFShift  = dbl_to_int( DDVectorGetter::get( "HFShift" ));
    php.maxDepth = dbl_to_int( DDVectorGetter::get( "MaxDepth" ));
  } else {
    throw cms::Exception("HcalParametersFromDD") << "Not found "<< attribute.c_str() << " but needed.";
  }
  for( unsigned int i = 0; i < php.rTable.size(); ++i ) {
    unsigned int k = php.rTable.size() - i - 1;
    php.etaTableHF.push_back( -log( tan( 0.5 * atan( php.rTable[k] / php.gparHF[4] ))));
  }
  //Special parameters at reconstruction level
  attribute = "OnlyForHcalRecNumbering"; 
  DDValue val2( attribute, value, 0.0 );
  DDSpecificsFilter filter2;
  filter2.setCriteria(val2,
		      DDCompOp::not_equals,
		      DDLogOp::AND, true, // compare strings 
		      true  // use merged-specifics or simple-specifics
		      );
  DDFilteredView fv2(*cpv);
  fv2.addFilter(filter2);
  ok = fv2.firstChild();
  if (ok) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    php.topologyMode = getTopologyMode("TopologyMode", sv);
    php.etagroup = dbl_to_int( DDVectorGetter::get( "etagroup" ));
    php.phigroup = dbl_to_int( DDVectorGetter::get( "phigroup" ));
    for (unsigned int i = 1; i<=nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupRecEta" << i;
      std::string tempName = sstm.str();
      if (DDVectorGetter::check(tempName)) {
	HcalParameters::LayerItem layerGroupEta;
	layerGroupEta.layer = i;
	layerGroupEta.layerGroup = dbl_to_int(DDVectorGetter::get(tempName));
	php.layerGroupEtaRec.push_back(layerGroupEta);
      }
    }
  } else {			      
    throw cms::Exception( "HcalParametersFromDD" ) << "Not found "<< attribute.c_str() << " but needed.";
  }

#ifdef DebugLog
  int i(0);
  std::cout << "HcalParametersFromDD: MaxDepth: ";
  for (const auto& it : php.maxDepth) std::cout << it << ", ";
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: ModHB [" << php.modHB.size() << "]: ";
  for (const auto& it : php.modHB) std::cout << it << ", ";
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: ModHE [" << php.modHE.size() << "]: ";
  for (const auto& it : php.modHE) std::cout << it << ", ";
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.phioff.size() << " phioff values";
  std::vector<double>::const_iterator it;
  for (it=php.phioff.begin(), i=0; it!=php.phioff.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it)/CLHEP::deg;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etaTable.size() << " entries for etaTable";
  for (it=php.etaTable.begin(), i=0; it!=php.etaTable.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.rTable.size() << " entries for rTable";
  for (it=php.rTable.begin(), i=0; it!=php.rTable.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it)/CLHEP::cm;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.phibin.size() << " phibin values";
  for (it=php.phibin.begin(), i=0; it!=php.phibin.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it)/CLHEP::deg;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.phitable.size() << " phitable values";
  for (it=php.phitable.begin(), i=0; it!=php.phitable.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it)/CLHEP::deg;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.layerGroupEtaSim.size() << " layerGroupEtaSim blocks" << std::endl;
  std::vector<HcalParameters::LayerItem>::const_iterator jt;
  std::vector<int>::const_iterator kt;
  for (unsigned int k=0; k < php.layerGroupEtaSim.size(); ++k) {
    std::cout << "layerGroupEtaSim[" << k << "] Layer " << php.layerGroupEtaSim[k].layer;
    for (kt=php.layerGroupEtaSim[k].layerGroup.begin(), i=0; kt!=php.layerGroupEtaSim[k].layerGroup.end(); ++kt)
      std::cout << " " << ++i << ":" << (*kt);
    std::cout << std::endl;
  }
  std::cout << "HcalParametersFromDD: " << php.etaMin.size() << " etaMin values";
  for (kt=php.etaMin.begin(), i=0; kt!=php.etaMin.end(); ++kt) 
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etaMax.size() << " etaMax values";
  for (kt=php.etaMax.begin(), i=0; kt!=php.etaMax.end(); ++kt) 
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etaRange.size() << " etaRange values";
  for (it=php.etaRange.begin(), i=0; it!=php.etaRange.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.gparHF.size() << " gparHF values";
  for (it=php.gparHF.begin(), i=0; it!=php.gparHF.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it)/CLHEP::cm;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.noff.size() << " noff values";
  for (kt=php.noff.begin(), i=0; kt!=php.noff.end(); ++kt) 
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.Layer0Wt.size() << " Layer0Wt values";
  for (it=php.Layer0Wt.begin(), i=0; it!=php.Layer0Wt.end(); ++it) 
    std::cout << " [" << ++i << "] = " << (*it);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HBGains.size() << " Shift/Gains values for HB";
  for (unsigned k=0; k<php.HBGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HBShift[k] << ":" << php.HBGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HEGains.size() << " Shift/Gains values for HE";
  for (unsigned k=0; k<php.HEGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HEShift[k] << ":" << php.HEGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HFGains.size() << " Shift/Gains values for HF";
  for (unsigned k=0; k<php.HFGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HFShift[k] << ":" << php.HFGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etagroup.size() << " etagroup values";
  for (kt=php.etagroup.begin(), i=0; kt!=php.etagroup.end(); ++kt) 
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.phigroup.size() << " phigroup values";
  for (kt=php.phigroup.begin(), i=0; kt!=php.phigroup.end(); ++kt) 
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.layerGroupEtaRec.size() << " layerGroupEtaRec blocks" << std::endl;
  for (unsigned int k=0; k < php.layerGroupEtaRec.size(); ++k) {
    std::cout << "layerGroupEtaRec[" << k << "] Layer " << php.layerGroupEtaRec[k].layer;
    for (kt=php.layerGroupEtaRec[k].layerGroup.begin(), i=0; kt!=php.layerGroupEtaRec[k].layerGroup.end(); ++kt)
      std::cout << " " << ++i << ":" << (*kt);
    std::cout << std::endl;
  }
  std::cout << "HcalParametersFromDD: topologyMode " << php.topologyMode << std::endl;
#endif

  return true;
}
