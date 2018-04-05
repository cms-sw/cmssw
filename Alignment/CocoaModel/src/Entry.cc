//   COCOA class implementation file
//Id:  Entry.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/ParameterMgr.h"
#include "Alignment/CocoaModel/interface/EntryMgr.h"
#include "Alignment/CocoaModel/interface/EntryData.h"
#include <cstdlib>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Constructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Entry::Entry( const ALIstring& type ) : type_(type), fitPos_(-1)
{
  //  std::cout << "entry" << std::endl;
  //---------- Set displacement by fitting to zero
  valueDisplacementByFitting_ = 0.; 
  if( ALIUtils::debug >= 5 ) std::cout << this << " theValueDisplacementByFitting set " << valueDisplacementByFitting_ << std::endl;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fill( const std::vector<ALIstring>& wordlist )
{

  ALIdouble byshort;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("reportOutEntriesByShortName", byshort ); 

  //----- Check format of input file
  if (ALIUtils::debug >=4) std::cout << "@@@ Filling entry: " << name() << std::endl;
  //--- Check there are 4 attributes
  if ( wordlist.size() != 4 ) {
    ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    ALIUtils::dumpVS( wordlist, " !!! Incorrect format for Entry:", std::cerr );
    std::cerr << std::endl << " There should be four words: name value sigma quality " << std::endl;
    exit(2);
  }

  EntryData* entryData;
  if( byshort == 0 ) {
    entryData = EntryMgr::getInstance()->findEntryByLongName( OptOCurrent()->longName(), name() );
  } else {
    entryData = EntryMgr::getInstance()->findEntryByShortName( OptOCurrent()->longName(), name() );
  }
  if(ALIUtils::debug >= 5) std::cout << " entryData " << entryData << " " <<  OptOCurrent()->longName() << " " << name() << std::endl;

  /*t
  if( name_ == "centre_R" || name_ == "centre_PHI" || name_ == "centre_THE" ){
    if( EntryMgr::getInstance()->numberOfEntries() > 0 ) {
      std::cerr << "!!!!FATAL ERROR:  Filling entry from 'report.out' while entry is in cylindrical or spherical coordinates is not supported yet. " << OptOCurrent()->name() << " " << name_ << std::endl;
      abort();
    }
  }
  */

  ALIdouble fre;
  gomgr->getGlobalOptionValue("reportOutReadValue", fre );
  if( entryData != nullptr && fre == 1) {
    //    std::cout << OptOCurrent()->name() << " " << name_ << "call fillFromReportOutFileValue " << type_ <<  std::endl;
    fillFromReportOutFileValue( entryData );
  } else {
    //  std::cout << OptOCurrent()->name() << " " << name_ << "call fillFromInputFileValue " << type_ <<  std::endl;
    fillFromInputFileValue( wordlist );
  }
  gomgr->getGlobalOptionValue("reportOutReadSigma", fre );
  if( entryData != nullptr && fre == 1) {
    fillFromReportOutFileSigma( entryData );
  } else {
    fillFromInputFileSigma( wordlist );
  }
  gomgr->getGlobalOptionValue("reportOutReadQuality", fre );
  if( entryData != nullptr && fre == 1) {
    fillFromReportOutFileQuality( entryData );
  } else {
    fillFromInputFileQuality( wordlist );
  }
}

 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillFromInputFileValue( const std::vector<ALIstring>& wordlist )
{

  //-  ALIUtils::dumpVS( wordlist, " fillFromInputFileValue " ); //-
  ParameterMgr* parmgr = ParameterMgr::getInstance();
  //---------- Translate parameter used for value_
  ALIdouble val = 0.;
  if ( !ALIUtils::IsNumber( wordlist[1] ) ) {
    if ( parmgr->getParameterValue( wordlist[1], val ) == 0 ) {
      ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
      std::cerr << "!!! parameter for value not found: " << wordlist[1].c_str() << std::endl;
      exit(2);
    }
    //d val *= ValueDimensionFactor();
  } else { 
    //d val = DimensionMgr()::getInstance()->extractValue( wordlist[1], ValueDimensionFactor() );
    val = atof( wordlist[1].c_str() );
  }
  val *= ValueDimensionFactor();
  if ( ALIUtils::debug >= 4 ) { 
    std::cout << "VALUE = " << val << " (ValueDimensionFactor= " << ValueDimensionFactor() <<std::endl;
  }

  value_ = val;
  valueOriginalOriginal_ = value_;


}


void Entry::fillFromInputFileSigma( const std::vector<ALIstring>& wordlist )
{

  ParameterMgr* parmgr = ParameterMgr::getInstance();
  //---------- translate parameter used for sigma_
  /*  ALIdouble sig;
  char** endptr;
  sig = strtod( wordlist[2].c_str(), endptr );
  //  ALIint isNumber =  sscanf(wordlist[2].c_str(),"%f",sig);
  if ( *endptr == wordlist[2].c_str() ) {
  // if ( !isNumber ) { */
  ALIdouble sig = 0.;
  if ( !ALIUtils::IsNumber( wordlist[2] ) ) {
    if ( parmgr->getParameterValue( wordlist[2], sig ) == 0 ) {
      ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
      //      std::cerr << "!!! parameter for sigma not found: " << wordlist[2].c_str() << std::endl;
      std::cerr << "!!! parameter for sigma not found: " << wordlist[0] << " " << wordlist[1] << " " <<  wordlist[2] << std::endl;
      exit(2);
    }
    //d    sig *= SigmaDimensionFactor();
    //-    std::cout << sig<< " valueparam " << wordlist[2] << std::endl;
  } else { 
    //d sig = DimensionMgr()::getInstance()->extractValue( wordlist[2], ValueDimensionFactor() );
    sig = atof( wordlist[2].c_str() );
    // for range studies, make all 'cal' entries 'fix'
    ALIdouble rs;
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    gomgr->getGlobalOptionValue("range_studies", rs );
    if(rs == 1) sig *= 1.E-6;

    //-    std::cout << sig << " valuem " << wordlist[2] << std::endl;
  }
  sig *= SigmaDimensionFactor();
  if (ALIUtils::debug >= 4) {
    std::cout << "SIGMA = " << sig << " (SigmaDimensionFactor= " << SigmaDimensionFactor() << std::endl;
  }
  sigma_ = sig;
  sigmaOriginalOriginal_ = sigma_;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillFromInputFileQuality( const std::vector<ALIstring>& wordlist )
{
  //---------- set _quality
  if( wordlist[3] == ALIstring("unk") ) {
    quality_ = 2;
  } else if( wordlist[3] == ALIstring("cal") ) {
    quality_ = 1;
    //t  // for range studies, make all 'cal' entries 'fix'
    //t ALIdouble rs;
    //t Model::getGlobalOptionValue("range_studies", rs );
    //t if(rs == 1) quality_ = 0;
  } else if( wordlist[3] == ALIstring("fix") ) { 
    quality_ = 0;
  } else {
    ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    std::cerr << " quality should be 'unk' or 'cal' or 'fix', instead of " << wordlist[3] << std::endl;
    exit(3);
  }
  //------ If sigma_ = 0 make quality_ 'fix'
  if( sigma_ == 0) {
    //      std::cout << "SIG=0" << std::endl;
    quality_ = 0;
  }
  if ( ALIUtils::debug >= 4 ) std::cout << OptOCurrent()->name() << " " << name() << " " << sigma_ << "QUALITY:" << quality_  << std::endl;
    
  sigmaOriginalOriginal_ = sigma_;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Fill the attributes with values read from a 'report.out' file
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillFromReportOutFileValue( EntryData* entryData )
{
  value_ = entryData->valueOriginal();
  //---- For extra entries, value is not in proper units, as the 'report.out' file does not have the type (length/angle/nodim)
  EntryMgr* entryMgr = EntryMgr::getInstance();
  //-  std::cout << OptOCurrent()->name() << " " << name_ << " fillFromReportOutFileValue " << type_ << std::endl;
  if( type_ == "centre" || type_ == "length" ) {
   value_ *= entryMgr->getDimOutLengthVal();
   //set valueDisp as it will be used to displace entries
   entryData->setValueDisplacement( entryData->valueDisplacement() * entryMgr->getDimOutLengthVal());
   if(ALIUtils::debug >= 5) std::cout << " fillFromReportOut " << OptOCurrent()->name() << " " << name() << "" <<  value_ << " disp " <<  entryData->valueDisplacement() * entryMgr->getDimOutLengthVal() << std::endl;
  }else if( type_ == "angles" || type_ == "angle" ) {
    value_ *= entryMgr->getDimOutAngleVal();
    entryData->setValueDisplacement( entryData->valueDisplacement() * entryMgr->getDimOutAngleVal());
    if(ALIUtils::debug >= 5) std::cout << " fillFromReportOut " << OptOCurrent()->name() << " " << name() << "" <<  value_ << " disp " <<  entryData->valueDisplacement() * entryMgr->getDimOutAngleVal() << std::endl;
  }

  valueOriginalOriginal_ = value_;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillFromReportOutFileSigma( const EntryData* entryData )
{
  sigma_ = entryData->sigma();
  //---- For extra entries, value is not in proper units, as the 'report.out' file does not have the type (length/angle/nodim)
  EntryMgr* entryMgr = EntryMgr::getInstance();
  if( type_ == "centre" || type_ == "length" ) {
   sigma_ *= entryMgr->getDimOutLengthSig();
   //-   std::cout << " fillFromReportOut " << value_ << " +- " << sigma_ << std::endl;
  }else if( type_ == "angles" || type_ == "angle" ) {
   sigma_ *= entryMgr->getDimOutAngleSig();
  }

  sigmaOriginalOriginal_ = sigma_;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillFromReportOutFileQuality( const EntryData* entryData )
{
  quality_ = entryData->quality();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill the name (in derived classes is not simply calling setName)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillName( const ALIstring& name )
{
  setName( name );
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Fill the attributes  
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::fillNull()
{
  //-  fillName( name );
  value_ = 0.;
  valueOriginalOriginal_ = value_;
  sigma_ = 0.;
  sigmaOriginalOriginal_ = sigma_;
  quality_ = 0;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Displace an extra entry (coordinate entries have their own classes) 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::displace( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "ExtraEntry::Displace" <<  disp <<std::endl;
  ALIuint entryNo = OptOCurrent()->extraEntryNo( name() );

  OptOCurrent()->displaceExtraEntry( entryNo, disp );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Displace an extra entry Original value for iteratin in non linear fit (coordinate entries have their own classes) 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::displaceOriginal( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "ExtraEntry::DisplaceOriginal" <<  disp <<std::endl;
  ALIuint entryNo = OptOCurrent()->extraEntryNo( name() );

  OptOCurrent()->displaceExtraEntryOriginal( entryNo, disp );
 
} 


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Displace an extra entry OriginalOriginal value for iteratin in non linear fit (coordinate entries have their own classes) 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::displaceOriginalOriginal( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "ExtraEntry::DisplaceOriginalOriginal" <<  disp <<std::endl;
  ALIuint entryNo = OptOCurrent()->extraEntryNo( name() );

  OptOCurrent()->displaceExtraEntryOriginalOriginal( entryNo, disp );
 
} 


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Destructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Entry::~Entry()
{
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Add fitted displacement to value: save it as valueDisplacementByFitting_, as when the value is asked for, it will get the original value + this displacement
//@@ Then update the rmGlob, centreGlob 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::addFittedDisplacementToValue( const ALIdouble val )
{
  valueDisplacementByFitting_ += val;
  lastAdditionToValueDisplacementByFitting_ = val;
  if( ALIUtils::debug >= 3 ) std::cout << OptOCurrent()->name() << " " << name() << " Entry::addFittedDisplacementToValue " << val << " total= " << valueDisplacementByFitting_ << std::endl;
  
  //---------- Displace original centre, rotation matrix, ...
  displaceOriginal( val );
  OptOCurrent()->resetGlobalCoordinates();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Add fitted displacement to value: save it as theValueDisplacementByFitting, as when the value is asked for, it will get the origianl value + this displacement
//@@ Then update the rmGlob, centreGlob 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::substractToHalfFittedDisplacementToValue()
{
  addFittedDisplacementToValue( -lastAdditionToValueDisplacementByFitting_/2. );
  // addFittedDisplacementToValue( -1.01*theLastAdditionToValueDisplacementByFitting );
  //addFittedDisplacementToValue( -theLastAdditionToValueDisplacementByFitting );
  lastAdditionToValueDisplacementByFitting_ *= -1;
  //  addFittedDisplacementToValue( 0. );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble Entry::valueDisplaced() const
{
  ALIuint entryNo = OptOCurrent()->extraEntryNo( name() );
  if(ALIUtils::debug >= 5) std::cout << entryNo << " Entry::valueDisplaced " << name() << " in " << OptOCurrent()->name() 
       << " orig " <<  OptOCurrent()->ExtraEntryValueOriginalList()[entryNo] << " new " <<  OptOCurrent()->ExtraEntryValueList()[entryNo] << std::endl;
  return OptOCurrent()->ExtraEntryValueList()[entryNo] - OptOCurrent()->ExtraEntryValueOriginalList()[entryNo];
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Entry::resetValueDisplacementByFitting()
{
  valueDisplacementByFitting_ = 0.;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::ostream& operator << (std::ostream& os, const Entry& c) 
{

  os << "ENTRY: " << c.name() << " of type: " << c.type() << std::endl
     << " value " << c.value_ << " original " << c.valueOriginalOriginal_ << std::endl
     << " sigma " << c.sigma_ << " original " << c.sigmaOriginalOriginal_ << std::endl
     << " quality " << c.quality_ << " opto " << (c.OptOCurrent_)->name() << std::endl
     << " fitpos " << c.fitPos_ << " valueDisplacementByFitting " << c.valueDisplacementByFitting_ << " lastAdditionToValueDisplacementByFitting " << c.lastAdditionToValueDisplacementByFitting_ << std::endl;

  return os;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
const ALIstring Entry::longName() const
{
  return OptOCurrent_->name()+"/"+name_;
}


