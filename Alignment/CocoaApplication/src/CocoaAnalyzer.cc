#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "../interface/CocoaAnalyzer.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h" 
#include "DetectorDescription/Base/interface/Ptr.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "CondFormats/OptAlignObjects/interface/OAQuality.h" 
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h" 
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h" 
#include "Geometry/Records/interface/IdealGeometryRecord.h" 

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValuePair.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <list>
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <assert.h>
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/Fit.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

//----------------------------------------------------------------------
CocoaAnalyzer::CocoaAnalyzer(edm::ParameterSet const& pset) 
{
  theCocoaDaqRootFileName = pset.getParameter< std::string >("cocoaDaqRootFile");
}

//----------------------------------------------------------------------
void CocoaAnalyzer::beginJob( const edm::EventSetup& evts ) 
{
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

  DumpCocoaResults();  
}

//-----------------------------------------------------------------------
void CocoaAnalyzer::ReadXMLFile( const edm::EventSetup& evts ) 
{

  ALIUtils::setDebugVerbosity( 5 );

  // STEP ONE:  Initial COCOA objects will be built from a DDL geometry
  // description.  
  
  edm::ESHandle<DDCompactView> cpv;
  evts.get<IdealGeometryRecord>().get(cpv);
  
  if(ALIUtils::debug >= 3) {
    std::cout << " CocoaAnalyzer ROOT " << cpv->root() << std::endl;
  }

  //Build OpticalAlignInfo "system"
  const DDLogicalPart lv = cpv->root();
  
  OpticalAlignInfo oaInfo;
  oaInfo.ID_ = 0;
  //--- substract file name to object name
  oaInfo.name_ = lv.name();
  int icol = oaInfo.name_.find(":");
  oaInfo.name_ = oaInfo.name_.substr(icol+1,oaInfo.name_.length());
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
		     DDSpecificsFilter::matches,
		     DDSpecificsFilter::AND, 
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
    oaInfo.ID_ = nObjects;
    const DDsvalues_type params(fv.mergedSpecifics());
    
    const DDLogicalPart lv = fv.logicalPart();
    if(ALIUtils::debug >= 4) {
      std::cout << " CocoaAnalyzer: reading object " << lv.name() << std::endl;
    }

    std::vector<DDExpandedNode> history = fv.geoHistory();
    oaInfo.parentName_ = "";
    size_t ii;
    for(ii = 0; ii < history.size()-1;ii++ ) {
      if( ii != 0 ) oaInfo.parentName_ += "/";
      std::string name = history[ii].logicalPart().name();
      icol = name.find(":");
      name = name.substr(icol+1,name.length());
      oaInfo.parentName_ += name;
 //    oaInfo.parentName_ = (fv.geoHistory()[fv.geoHistory().size()-2]).logicalPart().name();
//    icol = oaInfo.parentName_.find(":");
 //   oaInfo.parentName_ = oaInfo.parentName_.substr(icol+1,oaInfo.parentName_.length());
    }

    //--- build object name (= parent name + object name)
    std::string name = history[ii].logicalPart().name();
    //--- substract file name to object name
    int icol = name.find(":");
    name = name.substr(icol+1,name.length());
    oaInfo.name_ = oaInfo.parentName_ + "/" + name;
    std::cout << " @@@@@@@@@@@@@@@@ NAme built " << oaInfo.name_ << " parent " << oaInfo.parentName_ << " short " << name << std::endl;

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
    oaInfo.x_.value_ = transl.x(); 
    oaInfo.x_.error_ = myFetchDbl(params, "centre_X_sigma", 0);
    oaInfo.x_.quality_  = int (myFetchDbl(params, "centre_X_quality", 0));
    
    oaInfo.y_.name_ = "Y";
    oaInfo.y_.dim_type_ = "centre";
    oaInfo.y_.value_ = transl.y();
    oaInfo.y_.error_ = myFetchDbl(params, "centre_Y_sigma", 0);
    oaInfo.y_.quality_  = int (myFetchDbl(params, "centre_Y_quality", 0));

    oaInfo.z_.name_ = "Z";
    oaInfo.z_.dim_type_ = "centre";
    oaInfo.z_.value_ = transl.z();
    oaInfo.z_.error_ = myFetchDbl(params, "centre_Z_sigma", 0);
    oaInfo.z_.quality_  = int (myFetchDbl(params, "centre_Z_quality", 0));

  //---- DDD convention is to use the inverse matrix, COCOA is the direct one!!!
    //---- convert it to CLHEP::Matrix
    double xx,xy,xz,yx,yy,yz,zx,zy,zz;
    rot.GetComponents (xx, xy, xz,
                 yx, yy, yz,
                 zx, zy, zz);
    Hep3Vector colX(xx,xy,xz);
    Hep3Vector colY(yx,yy,yz);
    Hep3Vector colZ(zx,zy,zz);
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

    if(ALIUtils::debug >= 4) {
      std::cout << "CocoaAnalyzer OBJECT " << oaInfo.name_ << " pos/angle read " << std::endl;
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
	    std::cout << *vsite << " setting issimu " << measIsSimulatedValue[*vsite][0] << std::endl;
	  }
	  if(ALIUtils::debug >= 4) {
	    std::cout << "CocoaAnalyser: looped measObjectNames " << "meas_object_name_"+(*vsite) << " n obj " << measObjectNames[*vsite].size() << std::endl;
	  }

	}
	
      }
    }
    
    if(ALIUtils::debug >= 4) {
      std::cout << " CocoaAnalyzer:  Fill extra entries with read parameters " << std::endl;
    }
    //--- Fill extra entries with read parameters    
    if ( names.size() == dims.size() && dims.size() == values.size() 
	 && values.size() == errors.size() && errors.size() == quality.size() ) {
      for ( size_t ind = 0; ind < names.size(); ++ind ) {
	oaParam.value_ = values[ind];
	oaParam.error_ = errors[ind];
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
      std::cout << " CocoaAnalyzer:  Fill measurements with read parameters " << std::endl;
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
	    std::cout << oaMeas.name_ << " copying issimu " << oaMeas.isSimulatedValue_[oaMeas.isSimulatedValue_.size()-1]  << " = " << measIsSimulatedValue[oaMeas.name_][ind2] << std::endl;
	    //-           std::cout << ind2 << " adding meas value " << oaParam << std::endl;
            oaParam.clear();	
	  }
	} else {
	  std::cout << "WARNING FOR NOW: sizes of measurement parameters (name, value, sigma) do"
		    << " not match! for measurement " << oaMeas.name_ << " !Did not fill parameters for this measurement " << std::endl;
	}
	measList_.oaMeasurements_.push_back (oaMeas);
	if(ALIUtils::debug >= 5) {
	  std::cout << "CocoaAnalyser: MEASUREMENT " << oaMeas.name_ << " extra entries read " << oaMeas << std::endl;
	}
	oaMeas.clear();
      }
      
    } else {
      std::cout << "WARNING FOR NOW: sizes of measurements (names, types do"
		<< " not match!  Did not add " << nObjects << " item to XXXMeasurements" 
		<< std::endl;
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
    std::cout << "Finished making " << nObjects+1 << " OpticalAlignInfo objects" << " and " <<  measList_.oaMeasurements_.size() << " OpticalAlignMeasurementInfo objects " << std::endl;
  }
  if(ALIUtils::debug >= 5) {
    std::cout << " @@@@@@ OpticalAlignments " << oaList_ << std::endl;
    std::cout << " @@@@@@ OpticalMeasurements " << measList_ << std::endl;
  }

}

//------------------------------------------------------------------------
std::vector<OpticalAlignInfo> CocoaAnalyzer::ReadCalibrationDB( const edm::EventSetup& evts ) 
{
  using namespace edm::eventsetup;
  edm::ESHandle<OpticalAlignments> pObjs;
  evts.get<OpticalAlignmentsRcd>().get(pObjs);
  const OpticalAlignments* dbObj = pObjs.product();
  std::vector<OpticalAlignInfo>::const_iterator it;
  for( it=dbObj->opticalAlignments_.begin();it!=dbObj->opticalAlignments_.end(); ++it ){
    std::cout<<"@@@@@ OpticalAlignInfo READ "<< *it << std::endl;
  }

  return dbObj->opticalAlignments_;
}


//------------------------------------------------------------------------
void CocoaAnalyzer::CorrectOptAlignments( std::vector<OpticalAlignInfo>& oaListCalib )
{
  std::vector<OpticalAlignInfo>::const_iterator it;
  for( it=oaListCalib.begin();it!=oaListCalib.end(); ++it ){
    OpticalAlignInfo oaInfoDB = *it;
    OpticalAlignInfo* oaInfoXML = FindOpticalAlignInfoXML( oaInfoDB );
    std::cout << " oaInfoXML found " << *oaInfoXML << std::endl;
    if( oaInfoXML == 0 ) {
      std::cerr << "@@@@@ WARNING CocoaAnalyzer::CorrectOptAlignments:  OpticalAlignInfo read from DB is not present in XML "<< *it << std::endl;
    } else {
      //------ Correct info       
      std::cout << " correcting info " << std::endl;
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
	//----- Look fot the extra parameter in XML oaInfo that has the same name
	std::string oaName = (*itoap1).name_.substr( 1, (*itoap1).name_.size()-2 );
	for( itoap2 = extraEntXML->begin(); itoap2 != extraEntXML->end(); itoap2++ ){
	  if( oaName == (*itoap2).name_ ) {
	    CorrectOaParam( &(*itoap2), *itoap1 ); 
	    pFound = true;
	    break;
	  }
	}
	if( !pFound ) {
	  std::cerr << "@@@@@ WARNING CocoaAnalyzer::CorrectOptAlignments:  extra enty read from DB is not present in XML "<< *itoap1 << " in object " << *it << std::endl;
	}

      }
      std::cout << " oaInfoXML corrected " << *oaInfoXML << std::endl;
      std::cout << " all oaInfo " << oaList_ << std::endl;

    }
  }

}


//------------------------------------------------------------------------
OpticalAlignInfo* CocoaAnalyzer::FindOpticalAlignInfoXML( OpticalAlignInfo oaInfo )
{
  OpticalAlignInfo* oaInfoXML = 0;
  std::vector<OpticalAlignInfo>::iterator it;
  for( it=oaList_.opticalAlignments_.begin();it!=oaList_.opticalAlignments_.end(); ++it ){
    std::string oaName = oaInfo.name_.substr( 1, oaInfo.name_.size()-2 );
    std::cout << "findoaixml " << (*it).name_ << " =? " << oaName << std::endl;
    if( (*it).name_ == oaName ) {
      oaInfoXML = &(*it);
      break;
    }
  }

  return oaInfoXML;
}


//------------------------------------------------------------------------
bool CocoaAnalyzer::CorrectOaParam( OpticalAlignParam* oaParamXML, OpticalAlignParam oaParamDB )
{
  std::cout << " CorrectOaParam " << std::endl;
  std::cout << " CorrectOaParam  al " << oaParamDB.value_ << std::endl;
  if( oaParamDB.value_ == -9.999E9 ) return false;
 
  oaParamXML->value_ = oaParamDB.value_;

  return true;

}


//-#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
//-#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"

//------------------------------------------------------------------------
void CocoaAnalyzer::RunCocoa()
{
  //-ALIFileIn fin;
  //-  GlobalOptionMgr::getInstance()->setGlobalOption("debug_verbose",5, fin );

  //---------- Build the Model out of the system description text file
  Model& model = Model::getInstance();

  model.BuildSystemDescriptionFromOA( oaList_ );

  if(ALIUtils::debug >= 3) {
    std::cout << "RunCocoa: geometry built " << std::endl;
  }

  model.BuildMeasurementsFromOA( measList_ );

  if(ALIUtils::debug >= 3) {
    std::cout << "RunCocoa: measurements built " << std::endl;
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
void CocoaAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
{
  return;


  using namespace edm::eventsetup;
  std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
  std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
 
  // just a quick dump of the private OpticalAlignments object
  std::cout << oaList_ << std::endl;
 
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
  edm::Handle<OpticalAlignMeasurements> measHandle;
  e.getByLabel("OptAlignGeneratedSource", measHandle); 
    

  std::cout << "========== event data product changes with every event =========" << std::endl;
  std::cout << *measHandle << std::endl;

  //============== COCOA WORK!
  //  Each set of optical alignment measurements can be used
  //  in whatever type of analysis COCOA does. 
  //==============

} //end of ::analyze()

// STEP 4:  one could use ::endJob() to write out the OpticalAlignments
// generated by the analysis. Example code of writing is in
// CondFormats/Alignment/test/testOptAlignWriter.cc


//-----------------------------------------------------------------------
double CocoaAnalyzer:: myFetchDbl(const DDsvalues_type& dvst, 
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
std::string CocoaAnalyzer:: myFetchString(const DDsvalues_type& dvst, 
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



//-----------------------------------------------------------------------
bool CocoaAnalyzer::DumpCocoaResults()
{

  OpticalAlignments* myobj=new OpticalAlignments;

  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;    
    OpticalAlignInfo data = GetOptAlignInfoFromOptO( *ite );
    myobj->opticalAlignments_.push_back(data);
  std::cout << "@@@@ OPTALIGNINFO WRITTEN TO DB " 
	    << data 
	    << std::endl;  }

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    mydbservice->createNewIOV<OpticalAlignments>(myobj,
                            mydbservice->endOfTime(),
                            "OpticalAlignmentsRcd");
       /*? compilation error??
    }catch(const cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(const std::exception& er){
      std::cout<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      std::cout<<"Funny error"<<std::endl;
    }
       */
  }

  //? gives unreadable error???  std::cout << "@@@@ OPTICALALIGNMENTS WRITTEN TO DB " << *myobj << std::endl;

  return TRUE;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalAlignInfo CocoaAnalyzer::GetOptAlignInfoFromOptO( OpticalObject* opto )
{
  std::cout << " CocoaAnalyzer::GetOptAlignInfoFromOptO " << opto->name() << std::endl;
  OpticalAlignInfo data;
  data.ID_=opto->cmsSwID();
  data.type_=opto->type();
  data.name_=opto->name();

  //----- Centre in local coordinates
  Hep3Vector centreLocal = opto->centreGlob() - opto->parent()->centreGlob();
  CLHEP::HepRotation parentRmGlobInv = inverseOf( opto->parent()->rmGlob() );
  centreLocal = parentRmGlobInv * centreLocal;

  const std::vector< Entry* > theCoordinateEntryVector = opto->CoordinateEntryList();
  std::cout << " CocoaAnalyzer::GetOptAlignInfoFromOptO starting coord " <<std::endl;

  data.x_.value_= centreLocal.x() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix() << std::endl;
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << " " << theCoordinateEntryVector[0]->fitPos() << std::endl;
  data.x_.error_= GetEntryError( theCoordinateEntryVector[0] ) / 100.; // in cm

  data.y_.value_= centreLocal.y() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat()  << " " << theCoordinateEntryVector[1]->fitPos() << std::endl;
  data.y_.error_= GetEntryError( theCoordinateEntryVector[1] ) / 100.; // in cm

  data.z_.value_= centreLocal.z() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat()  << " " << theCoordinateEntryVector[2]->fitPos() << std::endl;
  data.z_.error_= GetEntryError( theCoordinateEntryVector[2] ) / 100.; // in cm

  //----- angles in local coordinates
  std::vector<double> anglocal = opto->getLocalRotationAngles( theCoordinateEntryVector );

  data.angx_.value_= anglocal[0] *180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[3]->fitPos() << std::endl;
  data.angx_.error_= GetEntryError( theCoordinateEntryVector[3] ) * 180./M_PI; // in deg;

  data.angy_.value_= anglocal[1] * 180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[4]->fitPos() << std::endl;
  data.angy_.error_= GetEntryError( theCoordinateEntryVector[4] ) * 180./M_PI; // in deg;;

  data.angz_.value_= anglocal[2] *180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[5]->fitPos() << std::endl;
  data.angz_.error_= GetEntryError( theCoordinateEntryVector[5] ) * 180./M_PI; // in deg;

  
  const std::vector< Entry* > theExtraEntryVector = opto->ExtraEntryList();  std::cout << " CocoaAnalyzer::GetOptAlignInfoFromOptO starting entry " << std::endl;

  std::vector< Entry* >::const_iterator ite;
  for( ite = theExtraEntryVector.begin(); ite != theExtraEntryVector.end(); ite++ ) {
    OpticalAlignParam extraEntry; 
    extraEntry.name_ = (*ite)->name();
    extraEntry.dim_type_ = (*ite)->type();
    extraEntry.value_ = (*ite)->value();
    extraEntry.error_ = (*ite)->sigma();
    extraEntry.quality_ = (*ite)->quality();
    data.extraEntries_.push_back( extraEntry );
  std::cout << " CocoaAnalyzer::GetOptAlignInfoFromOptO done extra entry " << extraEntry.name_ << std::endl;

  }

  return data;
}


double CocoaAnalyzer::GetEntryError( const Entry* entry )
{
  if( entry->quality() > 0 ) {
    return sqrt(Fit::GetAtWAMatrix()->Mat()->me[entry->fitPos()][entry->fitPos()]) / 100.;
  } else { //entry not fitted, return original error
    return entry->sigma();
  }
}

