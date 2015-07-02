#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "../interface/CocoaAnalyzer.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h" 
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h" 
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/Fit.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaFit/interface/CocoaDBMgr.h"

//----------------------------------------------------------------------
CocoaAnalyzer::CocoaAnalyzer(edm::ParameterSet const& pset) 
{
  theCocoaDaqRootFileName = pset.getParameter< std::string >("cocoaDaqRootFile");

  int maxEvents = pset.getParameter< int32_t >("maxEvents");
  GlobalOptionMgr::getInstance()->setDefaultGlobalOptions();
  GlobalOptionMgr::getInstance()->setGlobalOption("maxEvents",maxEvents);
  GlobalOptionMgr::getInstance()->setGlobalOption("writeDBAlign",1);
  GlobalOptionMgr::getInstance()->setGlobalOption("writeDBOptAlign",1);
  usesResource("CocoaAnalyzer");
}

//----------------------------------------------------------------------
void CocoaAnalyzer::beginJob()
{
}


//------------------------------------------------------------------------
void CocoaAnalyzer::RunCocoa()
{
  if(ALIUtils::debug >= 3) {
    std::cout << std::endl << "$$$ CocoaAnalyzer::RunCocoa: " << std::endl;
  }
  //-ALIFileIn fin;
  //-  GlobalOptionMgr::getInstance()->setGlobalOption("debug_verbose",5, fin );

  //---------- Build the Model out of the system description text file
  Model& model = Model::getInstance();

  model.BuildSystemDescriptionFromOA( oaList_ );

  if(ALIUtils::debug >= 3) {
    std::cout << "$$ CocoaAnalyzer::RunCocoa: geometry built " << std::endl;
  }

  model.BuildMeasurementsFromOA( measList_ );

  if(ALIUtils::debug >= 3) {
    std::cout << "$$ CocoaAnalyzer::RunCocoa: measurements built " << std::endl;
  }

  Fit::getInstance();

  Fit::startFit();

  if(ALIUtils::debug >= 0) std::cout << "............ program ended OK" << std::endl;
  if( ALIUtils::report >=1 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << "............ program ended OK" << std::endl;
  }

} // end of ::beginJob


//-----------------------------------------------------------------------
void CocoaAnalyzer::ReadXMLFile( const edm::EventSetup& evts ) 
{

  // STEP ONE:  Initial COCOA objects will be built from a DDL geometry
  // description.  
  
  edm::ESTransientHandle<DDCompactView> cpv;
  evts.get<IdealGeometryRecord>().get(cpv);

  if(ALIUtils::debug >= 3) {
    std::cout << std::endl << "$$$ CocoaAnalyzer::ReadXML: root object= " << cpv->root() << std::endl;
  }
  
  //Build OpticalAlignInfo "system"
  const DDLogicalPart lv = cpv->root();
  
  OpticalAlignInfo oaInfo;
  oaInfo.ID_ = 0;
  //--- substract file name to object name
  oaInfo.name_ = lv.name().name();
  oaInfo.parentName_ = "";
  oaInfo.x_.quality_  = 0;    
  oaInfo.x_.value_ = 0.;
  oaInfo.x_.error_ = 0.;
  oaInfo.x_.quality_  = 0;
  oaInfo.y_.value_ = 0.;
  oaInfo.y_.error_ = 0.;
  oaInfo.y_.quality_  = 0;
  oaInfo.z_.value_ = 0.;
  oaInfo.z_.error_ = 0.;
  oaInfo.z_.quality_  = 0;
  oaInfo.angx_.value_ = 0.;
  oaInfo.angx_.error_ = 0.;
  oaInfo.angx_.quality_  = 0;
  oaInfo.angy_.value_ = 0.;
  oaInfo.angy_.error_ = 0.;
  oaInfo.angy_.quality_  = 0;
  oaInfo.angz_.value_ = 0.;
  oaInfo.angz_.error_ = 0.;
  oaInfo.angz_.quality_  = 0;
    
  oaInfo.type_ = "system";

  oaList_.opticalAlignments_.push_back(oaInfo);
  oaInfo.clear();

  // Example of traversing the whole optical alignment geometry.
  // At each node we get specpars as variables and use them in 
  // constructing COCOA objects. 
  //  It stores these objects in a private data member, opt
  std::string attribute = "COCOA"; 
  std::string value     = "COCOA";
  DDValue val(attribute, value, 0.0);
  
	  // get all parts labelled with COCOA using a SpecPar
  DDSpecificsFilter filter;
  filter.setCriteria(val,  // name & value of a variable 
		     DDCompOp::matches,
		     DDLogOp::AND, 
		     true, // compare strings otherwise doubles
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fv(*cpv);
  fv.addFilter(filter);
  bool doCOCOA = fv.firstChild();
  
  // Loop on parts
  int nObjects=0;
  OpticalAlignParam oaParam;
  OpticalAlignMeasurementInfo oaMeas;

  while ( doCOCOA ){
    ++nObjects;
    //    oaInfo.ID_ = nObjects;
    const DDsvalues_type params(fv.mergedSpecifics());
    
    const DDLogicalPart lv = fv.logicalPart();
    if(ALIUtils::debug >= 4) {
      std::cout << " CocoaAnalyzer::ReadXML reading object " << lv.name() << std::endl;
    }

    std::vector<DDExpandedNode> history = fv.geoHistory();
    oaInfo.parentName_ = "";
    size_t ii;
    for(ii = 0; ii < history.size()-1;ii++ ) {
      if( ii != 0 ) oaInfo.parentName_ += "/";
      std::string name = history[ii].logicalPart().name().name();
      oaInfo.parentName_ += name;
 //    oaInfo.parentName_ = (fv.geoHistory()[fv.geoHistory().size()-2]).logicalPart().name();
//    icol = oaInfo.parentName_.find(":");
 //   oaInfo.parentName_ = oaInfo.parentName_.substr(icol+1,oaInfo.parentName_.length());
    }

    //--- build object name (= parent name + object name)
    std::string name = history[ii].logicalPart().name().name();
    //--- substract file name to object name
    oaInfo.name_ = oaInfo.parentName_ + "/" + name;
    if(ALIUtils::debug >= 5) {
      std::cout << " @@ Name built= " << oaInfo.name_ << " short_name= " << name << " parent= " << oaInfo.parentName_ << std::endl; 
    }
    //----- Read centre and angles
    oaInfo.x_.quality_  = int (myFetchDbl(params, "centre_X_quality", 0));
    DDTranslation transl = (fv.translation());
    DDRotationMatrix rot = (fv.rotation());
    DDExpandedNode parent = fv.geoHistory()[ fv.geoHistory().size()-2 ];
    DDTranslation parentTransl = parent.absTranslation();
    DDRotationMatrix parentRot = parent.absRotation();
    transl = parentRot.Inverse()*(transl - parentTransl );
    rot = parentRot.Inverse()*rot;
    rot = rot.Inverse(); //DDL uses opposite convention than COCOA
    /*    if(ALIUtils::debug >= 4) {
      ALIUtils::dumprm( rot, "local rotation ");
      ALIUtils::dump3v( transl, "local translation");
      } */

    oaInfo.x_.name_ = "X";
    oaInfo.x_.dim_type_ = "centre";
    oaInfo.x_.value_ = transl.x()*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.x_.error_ = myFetchDbl(params, "centre_X_sigma", 0)*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.x_.quality_  = int (myFetchDbl(params, "centre_X_quality", 0));
    
    oaInfo.y_.name_ = "Y";
    oaInfo.y_.dim_type_ = "centre";
    oaInfo.y_.value_ = transl.y()*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.y_.error_ = myFetchDbl(params, "centre_Y_sigma", 0)*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.y_.quality_  = int (myFetchDbl(params, "centre_Y_quality", 0));

    oaInfo.z_.name_ = "Z";
    oaInfo.z_.dim_type_ = "centre";
    oaInfo.z_.value_ = transl.z()*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.z_.error_ = myFetchDbl(params, "centre_Z_sigma", 0)*0.001; // CLHEP units are mm, COCOA are m 
    oaInfo.z_.quality_  = int (myFetchDbl(params, "centre_Z_quality", 0));

  //---- DDD convention is to use the inverse matrix, COCOA is the direct one!!!
    //---- convert it to CLHEP::Matrix
    double xx,xy,xz,yx,yy,yz,zx,zy,zz;
    rot.GetComponents (xx, xy, xz,
                 yx, yy, yz,
                 zx, zy, zz);
    CLHEP::Hep3Vector colX(xx,xy,xz);
    CLHEP::Hep3Vector colY(yx,yy,yz);
    CLHEP::Hep3Vector colZ(zx,zy,zz);
    CLHEP::HepRotation rotclhep( colX, colY, colZ );
    std::vector<double> angles = ALIUtils::getRotationAnglesFromMatrix( rotclhep,0., 0., 0. );

    oaInfo.angx_.name_ = "X";
    oaInfo.angx_.dim_type_ = "angles";
    //-    oaInfo.angx_.value_ = angles[0];
    oaInfo.angx_.value_ = myFetchDbl(params, "angles_X_value", 0);
    oaInfo.angx_.error_ = myFetchDbl(params, "angles_X_sigma", 0);
    oaInfo.angx_.quality_  = int (myFetchDbl(params, "angles_X_quality", 0));

    oaInfo.angy_.name_ = "Y";
    oaInfo.angy_.dim_type_ = "angles";
    //-    oaInfo.angy_.value_ = angles[1];
    oaInfo.angy_.value_ = myFetchDbl(params, "angles_Y_value", 0);
    oaInfo.angy_.error_ = myFetchDbl(params, "angles_Y_sigma", 0);
    oaInfo.angy_.quality_  = int (myFetchDbl(params, "angles_Y_quality", 0));

    oaInfo.angz_.name_ = "Z";
    oaInfo.angz_.dim_type_ = "angles";
    //    oaInfo.angz_.value_ = angles[2];
    oaInfo.angz_.value_ = myFetchDbl(params, "angles_Z_value", 0);
    oaInfo.angz_.error_ = myFetchDbl(params, "angles_Z_sigma", 0);
    oaInfo.angz_.quality_  = int (myFetchDbl(params, "angles_Z_quality", 0));

    oaInfo.type_ = myFetchString(params, "cocoa_type", 0);

    oaInfo.ID_ = int(myFetchDbl(params, "cmssw_ID", 0));

    if(ALIUtils::debug >= 4) {
      std::cout << "CocoaAnalyzer::ReadXML OBJECT " << oaInfo.name_ << " pos/angles read " << std::endl;
    }

    if( fabs( oaInfo.angx_.value_ - angles[0] ) > 1.E-9 || 
	fabs( oaInfo.angy_.value_ - angles[1] ) > 1.E-9 || 
	fabs( oaInfo.angz_.value_ - angles[2] ) > 1.E-9 ) {
      std::cerr << " WRONG ANGLE IN OBJECT " << oaInfo.name_<< 
	oaInfo.angx_.value_ << " =? " << angles[0] <<
	oaInfo.angy_.value_ << " =? " << angles[1] <<
	oaInfo.angz_.value_ << " =? " << angles[2] << std::endl;
    }

    //----- Read extra entries and measurements
    const std::vector<const DDsvalues_type *> params2(fv.specifics());
    std::vector<const DDsvalues_type *>::const_iterator spit = params2.begin();
    std::vector<const DDsvalues_type *>::const_iterator endspit = params2.end();
    //--- extra entries variables
    std::vector<std::string> names, dims;
    std::vector<double> values, errors, quality;
    //--- measurements variables
    std::vector<std::string> measNames;
    std::vector<std::string> measTypes;
    std::map<std::string, std::vector<std::string> > measObjectNames;
    std::map<std::string, std::vector<std::string> > measParamNames;
    std::map<std::string, std::vector<double> > measParamValues;
    std::map<std::string, std::vector<double> > measParamSigmas;
    std::map<std::string, std::vector<double> > measIsSimulatedValue;

    for ( ; spit != endspit; ++spit ) {
      DDsvalues_type::const_iterator sit = (**spit).begin();
      DDsvalues_type::const_iterator endsit = (**spit).end();
      for ( ; sit != endsit; ++sit ) {
 	if (sit->second.name() == "extra_entry") {
	  names = sit->second.strings();
	} else if (sit->second.name() == "dimType") {
	  dims = sit->second.strings();
	} else if (sit->second.name() == "value") {
	  values = sit->second.doubles();
	} else if (sit->second.name() == "sigma") {
	  errors = sit->second.doubles();
	} else if (sit->second.name() == "quality") {
	  quality = sit->second.doubles();

	} else if (sit->second.name() == "meas_name") {
	  //-	  std::cout << " meas_name found " << std::endl;
	  measNames = sit->second.strings();
	} else if (sit->second.name() == "meas_type") {
	  //- std::cout << " meas_type found " << std::endl;
	  measTypes = sit->second.strings();
	}
	
      }
    }

    //---- loop again to look for the measurement object names, that have the meas name in the SpecPar title 
    //    <Parameter name="meas_object_name_SENSOR2D:OCMS/sens2" value="OCMS/laser1"  eval="false" /> 
    //   <Parameter name="meas_object_name_SENSOR2D:OCMS/sens2" value="OCMS/sens2"  eval="false" /> 

    std::vector<std::string>::iterator vsite;
    for ( spit = params2.begin(); spit != params2.end(); ++spit ) {
      //-  std::cout << "loop vector DDsvalues " << std::endl;
      DDsvalues_type::const_iterator sit = (**spit).begin();
      DDsvalues_type::const_iterator endsit = (**spit).end();
      for ( ; sit != endsit; ++sit ) {
	for( vsite = measNames.begin(); vsite != measNames.end(); vsite++ ){
	  //- std::cout << "looping measObjectNames " << *vsite << std::endl;
	  if (sit->second.name() == "meas_object_name_"+(*vsite)) {
	    measObjectNames[*vsite] = sit->second.strings();
	  }else if (sit->second.name() == "meas_value_name_"+(*vsite)) {
	    measParamNames[*vsite] = sit->second.strings();
	  }else if (sit->second.name() == "meas_value_"+(*vsite)) {
	    measParamValues[*vsite] = sit->second.doubles();
	  }else if (sit->second.name() == "meas_sigma_"+(*vsite)) {
	    measParamSigmas[*vsite] = sit->second.doubles();
	  }else if (sit->second.name() == "meas_is_simulated_value_"+(*vsite)) {
	    measIsSimulatedValue[*vsite] = sit->second.doubles(); // this is not in OptAlignParam info
	    if(ALIUtils::debug >= 5) {
	      std::cout << *vsite << " setting issimu " << measIsSimulatedValue[*vsite][0] << std::endl;
	    }
	  }
	  if(ALIUtils::debug >= 5) {
	    std::cout << "CocoaAnalyser: looped measObjectNames " << "meas_object_name_"+(*vsite) << " n obj " << measObjectNames[*vsite].size() << std::endl;
	  }

	}
	
      }
    }
    
    if(ALIUtils::debug >= 4) {
      std::cout << " CocoaAnalyzer::ReadXML:  Fill extra entries with read parameters " << std::endl;
    }
    //--- Fill extra entries with read parameters    
    if ( names.size() == dims.size() && dims.size() == values.size() 
	 && values.size() == errors.size() && errors.size() == quality.size() ) {
      for ( size_t ind = 0; ind < names.size(); ++ind ) {
	double dimFactor = 1.;
	std::string type = oaParam.dimType();
	if( type == "centre" || type == "length" ) {
	  dimFactor = 0.001; // in XML it is in mm 
	}else if ( type == "angles" || type == "angle" || type == "nodim" ){
	  dimFactor = 1.;
	} 
	oaParam.value_ = values[ind]*dimFactor;
	oaParam.error_ = errors[ind]*dimFactor;
	oaParam.quality_ = int (quality[ind]);
	oaParam.name_ = names[ind];
	oaParam.dim_type_ = dims[ind];
	oaInfo.extraEntries_.push_back (oaParam);
	oaParam.clear();
      }

      //t      std::cout << names.size() << " OBJECT " << oaInfo.name_ << " extra entries read " << oaInfo << std::endl;

      oaList_.opticalAlignments_.push_back(oaInfo);
    } else {
      std::cout << "WARNING FOR NOW: sizes of extra parameters (names, dimType, value, quality) do"
		<< " not match!  Did not add " << nObjects << " item to OpticalAlignments." 
		<< std::endl;
    }

    if(ALIUtils::debug >= 4) {
      std::cout << " CocoaAnalyzer::ReadXML:  Fill measurements with read parameters " << std::endl;
    }
    //--- Fill measurements with read parameters    
    if ( measNames.size() == measTypes.size() ) {
      for ( size_t ind = 0; ind < measNames.size(); ++ind ) {
	oaMeas.ID_ = ind;
	oaMeas.name_ = measNames[ind];
	oaMeas.type_ = measTypes[ind];
	oaMeas.measObjectNames_ = measObjectNames[oaMeas.name_];
	if( measParamNames.size() == measParamValues.size() && measParamValues.size() == measParamSigmas.size() ) { 
	  for( size_t ind2 = 0; ind2 < measParamNames[oaMeas.name_].size(); ind2++ ){
	    oaParam.name_ = measParamNames[oaMeas.name_][ind2];
	    oaParam.value_ = measParamValues[oaMeas.name_][ind2];
	    oaParam.error_ = measParamSigmas[oaMeas.name_][ind2];
	    if( oaMeas.type_ == "SENSOR2D" || oaMeas.type_ == "COPS" || oaMeas.type_ == "DISTANCEMETER" || oaMeas.type_ == "DISTANCEMETER!DIM" || oaMeas.type_ == "DISTANCEMETER3DIM" ) {
	      oaParam.dim_type_ = "length";
	    } else if( oaMeas.type_ == "TILTMETER" ) {
	      oaParam.dim_type_ = "angle";
	    } else {
	      std::cerr << "CocoaAnalyzer::ReadXMLFile. Invalid measurement type: " <<  oaMeas.type_ << std::endl;
	      std::exception();
	    }
	    
	    oaMeas.values_.push_back( oaParam );
	    oaMeas.isSimulatedValue_.push_back( measIsSimulatedValue[oaMeas.name_][ind2] );
	    if(ALIUtils::debug >= 5) {
	      std::cout << oaMeas.name_ << " copying issimu " << oaMeas.isSimulatedValue_[oaMeas.isSimulatedValue_.size()-1]  << " = " << measIsSimulatedValue[oaMeas.name_][ind2] << std::endl;
	    //-           std::cout << ind2 << " adding meas value " << oaParam << std::endl;
	    }
            oaParam.clear();	
	  }
	} else {
	  if(ALIUtils::debug >= 2) {
	    std::cout << "WARNING FOR NOW: sizes of measurement parameters (name, value, sigma) do"
		      << " not match! for measurement " << oaMeas.name_ << " !Did not fill parameters for this measurement " << std::endl;
	  }
	}
	measList_.oaMeasurements_.push_back (oaMeas);
	if(ALIUtils::debug >= 5) {
	  std::cout << "CocoaAnalyser: MEASUREMENT " << oaMeas.name_ << " extra entries read " << oaMeas << std::endl;
	}
	oaMeas.clear();
      }
      
    } else {
      if(ALIUtils::debug >= 2) {
	std::cout << "WARNING FOR NOW: sizes of measurements (names, types do"
		  << " not match!  Did not add " << nObjects << " item to XXXMeasurements" 
		  << std::endl;
      }
    }

//       std::cout << "sizes are values=" << values.size();
//       std::cout << "  sigma(errors)=" << errors.size();
//       std::cout << "  quality=" << quality.size();
//       std::cout << "  names=" << names.size();
//       std::cout << "  dimType=" << dims.size() << std::endl;
    oaInfo.clear();
    doCOCOA = fv.next(); // go to next part
  } // while (doCOCOA)
  if(ALIUtils::debug >= 3) {
    std::cout << "CocoaAnalyzer::ReadXML: Finished building " << nObjects+1 << " OpticalAlignInfo objects" << " and " <<  measList_.oaMeasurements_.size() << " OpticalAlignMeasurementInfo objects " << std::endl;
  }
  if(ALIUtils::debug >= 5) {
    std::cout << " @@@@@@ OpticalAlignments " << oaList_ << std::endl;
    std::cout << " @@@@@@ OpticalMeasurements " << measList_ << std::endl;
  }

}

//------------------------------------------------------------------------
std::vector<OpticalAlignInfo> CocoaAnalyzer::ReadCalibrationDB( const edm::EventSetup& evts ) 
{
  if(ALIUtils::debug >= 3) {
    std::cout<< std::endl <<"$$$ CocoaAnalyzer::ReadCalibrationDB: " << std::endl;
  }
  
  using namespace edm::eventsetup;
  edm::ESHandle<OpticalAlignments> pObjs;
  evts.get<OpticalAlignmentsRcd>().get(pObjs);
  const OpticalAlignments* dbObj = pObjs.product();
  
  if(ALIUtils::debug >= 5) {
    std::vector<OpticalAlignInfo>::const_iterator it;
    for( it=dbObj->opticalAlignments_.begin();it!=dbObj->opticalAlignments_.end(); ++it ){
      std::cout<<"CocoaAnalyzer::ReadCalibrationDB:  OpticalAlignInfo READ "<< *it << std::endl;
    }
  }
  
  if(ALIUtils::debug >= 4) {
    std::cout<<"CocoaAnalyzer::ReadCalibrationDB:  Number of OpticalAlignInfo READ "<< dbObj->opticalAlignments_.size() << std::endl;
  }

  return dbObj->opticalAlignments_;
}


//------------------------------------------------------------------------
void CocoaAnalyzer::CorrectOptAlignments( std::vector<OpticalAlignInfo>& oaListCalib )
{ 
  if(ALIUtils::debug >= 3) {
    std::cout<< std::endl<< "$$$ CocoaAnalyzer::CorrectOptAlignments: " << std::endl;
  }

  std::vector<OpticalAlignInfo>::const_iterator it;
  for( it=oaListCalib.begin();it!=oaListCalib.end(); ++it ){
    OpticalAlignInfo oaInfoDB = *it;
    OpticalAlignInfo* oaInfoXML = FindOpticalAlignInfoXML( oaInfoDB );
    std::cerr << "error " << (*it).name_ << std::endl;
    if( oaInfoXML == 0 ) {
      if(ALIUtils::debug >= 2) {
	std::cerr << "@@@@@ WARNING CocoaAnalyzer::CorrectOptAlignments:  OpticalAlignInfo read from DB is not present in XML "<< *it << std::endl;
      }
    } else {
      //------ Correct info       
      if(ALIUtils::debug >= 5) {
	std::cout << "CocoaAnalyzer::CorrectOptAlignments: correcting data from DB info " << std::endl;
      }
      CorrectOaParam( &oaInfoXML->x_, oaInfoDB.x_ );
      CorrectOaParam( &oaInfoXML->y_, oaInfoDB.y_ );
      CorrectOaParam( &oaInfoXML->z_, oaInfoDB.z_ );
      CorrectOaParam( &oaInfoXML->angx_, oaInfoDB.angx_ );
      CorrectOaParam( &oaInfoXML->angy_, oaInfoDB.angy_ );
      CorrectOaParam( &oaInfoXML->angz_, oaInfoDB.angz_ );
      std::vector<OpticalAlignParam>::iterator itoap1, itoap2;
      std::vector<OpticalAlignParam> extraEntDB = oaInfoDB.extraEntries_;
      std::vector<OpticalAlignParam>* extraEntXML = &(oaInfoXML->extraEntries_);
      for( itoap1 = extraEntDB.begin(); itoap1 != extraEntDB.end(); itoap1++ ){
	bool pFound = false;
	//----- Look for the extra parameter in XML oaInfo that has the same name
	std::string oaName = (*itoap1).name_.substr( 1, (*itoap1).name_.size()-2 );
	for( itoap2 = extraEntXML->begin(); itoap2 != extraEntXML->end(); itoap2++ ){
	  if( oaName == (*itoap2).name_ ) {
	    CorrectOaParam( &(*itoap2), *itoap1 ); 
	    pFound = true;
	    break;
	  }
	}
	if( !pFound && oaName != "None" ) {
	  if(ALIUtils::debug >= 2) {
	    std::cerr << "@@@@@ WARNING CocoaAnalyzer::CorrectOptAlignments:  extra entry read from DB is not present in XML "<< *itoap1 << " in object " << *it << std::endl;
	  }
	}

      }
      if(ALIUtils::debug >= 5) {
	std::cout << "CocoaAnalyzer::CorrectOptAlignments: corrected OpticalAlingInfo " << oaList_ << std::endl;
      }

    }
  }

}


//------------------------------------------------------------------------
OpticalAlignInfo* CocoaAnalyzer::FindOpticalAlignInfoXML( const OpticalAlignInfo& oaInfo )
{
  OpticalAlignInfo* oaInfoXML = 0;
  std::vector<OpticalAlignInfo>::iterator it;
  for( it=oaList_.opticalAlignments_.begin();it!=oaList_.opticalAlignments_.end(); ++it ){
    std::string oaName = oaInfo.name_.substr( 1, oaInfo.name_.size()-2 );

    if(ALIUtils::debug >= 5) {
      std::cout << "CocoaAnalyzer::FindOpticalAlignInfoXML:  looking for OAI " << (*it).name_ << " =? " << oaName << std::endl;
    }
    if( (*it).name_ == oaName ) {
      oaInfoXML = &(*it);
      if(ALIUtils::debug >= 4) {
	std::cout << "CocoaAnalyzer::FindOpticalAlignInfoXML:  OAI found " << oaInfoXML->name_ << std::endl;
      }
      break;
    }
  }

  return oaInfoXML;
}


//------------------------------------------------------------------------
bool CocoaAnalyzer::CorrectOaParam( OpticalAlignParam* oaParamXML, const OpticalAlignParam& oaParamDB )
{
  if(ALIUtils::debug >= 4) {
    std::cout << "CocoaAnalyzer::CorrectOaParam  old value= " << oaParamXML->value_  << " new value= " << oaParamDB.value_ << std::endl;
  }
  if( oaParamDB.value_ == -9.999E9 ) return false;
 
  double dimFactor = 1.; 
  //loop for an Entry with equal type to entries to know which is the 
  std::string type = oaParamDB.dimType();
  if( type == "centre" || type == "length" ) {
    dimFactor = 0.01; // in DB it is in cm 
  }else if ( type == "angles" || type == "angle" || type == "nodim" ){
    dimFactor = 1.;
  }else {
    std::cerr << "!!! COCOA programming error: inform responsible: incorrect OpticalAlignParam type = " << type << std::endl;
    std::exception();
  }

  oaParamXML->value_ = oaParamDB.value_*dimFactor;

  return true;

}


//-#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
//-#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"

//-----------------------------------------------------------------------
void CocoaAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evts)
{
  ALIUtils::setDebugVerbosity( 5 );

  ReadXMLFile( evts );

  std::vector<OpticalAlignInfo> oaListCalib = ReadCalibrationDB( evts );

  CorrectOptAlignments( oaListCalib );

  new CocoaDaqReaderRoot( theCocoaDaqRootFileName );
 
  /*-
  int nEvents = daqReader->GetNEvents();
  for( int ii = 0; ii < nEvents; ii++) {
    if( ! daqReader->ReadEvent( ii ) ) break;
  }
  */ 

  RunCocoa();

  //  std::cout << "!!!! NOT  DumpCocoaResults() " << std::endl;  

  //  CocoaDBMgr::getInstance()->DumpCocoaResults();  

  return;


  //  using namespace edm::eventsetup;
  //  std::cout <<" I AM IN RUN NUMBER "<<evt.id().run() <<std::endl;
  //  std::cout <<" ---EVENT NUMBER "<<evt.id().event() <<std::endl;
 
  // just a quick dump of the private OpticalAlignments object
  //  std::cout << oaList_ << std::endl;
 
  // STEP 2:
  // Get calibrated OpticalAlignments.  In this case I'm using
  // some sqlite database examples that are generated using
  // testOptAlignWriter.cc
  // from CondFormats/OptAlignObjects/test/

//   edm::ESHandle<OpticalAlignments> oaESHandle;
//   context.get<OpticalAlignmentsRcd>().get(oaESHandle);

//   // This assumes they ALL come in together.  This may not be
//   // the "real" case.  One could envision different objects coming
//   // in and we would get by label each one (type).

//   std::cout << "========== eventSetup data changes with IOV =========" << std::endl;
//   std::cout << *oaESHandle << std::endl;
//   //============== COCOA WORK!
//   //  calibrated values should be used to "correct" the ones read in during beginJob
//   //==============
  
//   // 
//   // to see how to iterate over the OpticalAlignments, please
//   // refer to the << operator of OpticalAlignments, OpticalAlignInfo
//   // and OpticalAlignParam.
//   //     const OpticalAlignments* myoa=oa.product();
  
//   // STEP 3:
//   // This retrieves the Measurements
//   // for each event, a new set of measurements is available.
//  edm::Handle<OpticalAlignMeasurements> measHandle;
//  evt.getByLabel("OptAlignGeneratedSource", measHandle); 
    

//  std::cout << "========== event data product changes with every event =========" << std::endl;
//  std::cout << *measHandle << std::endl;

  //============== COCOA WORK!
  //  Each set of optical alignment measurements can be used
  //  in whatever type of analysis COCOA does. 
  //==============

} //end of ::analyze()

// STEP 4:  one could use ::endJob() to write out the OpticalAlignments
// generated by the analysis. Example code of writing is in
// CondFormats/Alignment/test/testOptAlignWriter.cc


//-----------------------------------------------------------------------
double CocoaAnalyzer::myFetchDbl(const DDsvalues_type& dvst, 
				      const std::string& spName,
				      const size_t& vecInd ) {
  DDValue val(spName, 0.0);
  if (DDfetch(&dvst,val)) {
    if ( val.doubles().size() > vecInd ) {
      //	  std::cout << "about to return: " << val.doubles()[vecInd] << std::endl;
      return val.doubles()[vecInd];
    } else {
      std::cout << "WARNING: OUT OF BOUNDS RETURNING 0 for index " << vecInd << " of SpecPar " << spName << std::endl;
    }
  }
  return 0.0;
}

//-----------------------------------------------------------------------
std::string CocoaAnalyzer::myFetchString(const DDsvalues_type& dvst, 
				      const std::string& spName,
				      const size_t& vecInd ) {
  DDValue val(spName, 0.0);
  if (DDfetch(&dvst,val)) {
    if ( val.strings().size() > vecInd ) {
      //	  std::cout << "about to return: " << val.doubles()[vecInd] << std::endl;
      return val.strings()[vecInd];
    } else {
      std::cout << "WARNING: OUT OF BOUNDS RETURNING 0 for index " << vecInd << " of SpecPar " << spName << std::endl;
    }
  }
  return "";
}

