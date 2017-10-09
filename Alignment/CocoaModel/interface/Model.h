//   COCOA class header file
//Id:  Model.h
//CAT: Model
//
//   Utility class that steers the reading of the system description file
//              and contains the static data 
// 
//   History: v1.0 
//   Pedro Arce

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <map>
//#include <multimap.h>


#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 
class Entry;
//#include "Alignment/CocoaModel/interface/Entry.h"  //temporal
class OpticalObject;
//#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class ALIFileIn;
class FittedEntriesReader;

class OpticalAlignments;
class OpticalAlignMeasurements;

#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"


enum sectionType{ sectGlobalOptions, sectParameters, sectSystemTreeDescription, sectSystemTreeData, sectMeasurements, sectReportOut };

enum cocoaStatus{ COCOA_Init, COCOA_ReadingModel, COCOA_InitFit, COCOA_FirstIterationInEvent, COCOA_NextIterationInEvent, COCOA_FitOK, COCOA_FitImproving, COCOA_FitCannotImprove, COCOA_FitChi2Worsened, COCOA_FitMatrixNonInversable };

class Model 
{
public:
  //---------- Constructor / destructor 
  Model();
  ~Model(){};

  ///---------- Gets the only instance of this class
  static Model& getInstance();  

  static cocoaStatus getCocoaStatus(){ return theCocoaStatus;}
  static void setCocoaStatus(const cocoaStatus cs ){ theCocoaStatus = cs; }
  static std::string printCocoaStatus(const cocoaStatus cs);

  ///---------- Read the different sections of the SDF and act accordingly
  static void readSystemDescription();

  //----------- Build OpticalObjects's from info in XML file
  void BuildSystemDescriptionFromOA( OpticalAlignments& optAlig );
  OpticalAlignInfo FindOptAlignInfoByType( const ALIstring& type );
  //----------- Build Measurements's from info in XML file
  void BuildMeasurementsFromOA( OpticalAlignMeasurements& measList );

/// ACCESS STATIC DATA MEMBERS

  //  static std::map< ALIstring, ALIdouble, std::less<ALIstring> >& Parameters() {
  //     return theParameters;
  //  }

  static std::vector< std::vector<ALIstring> >& OptODictionary() {
      return theOptODictionary;
  }

  static std::vector< OpticalObject* >& OptOList(){
      return theOptOList;
  }

  static std::vector< Entry* >& EntryList() {
    return theEntryVector;
  }

  static std::vector< Measurement* >& MeasurementList() {
     return theMeasurementVector;
  }

  static Measurement* getMeasurementByName( const ALIstring& name, ALIbool exists = 1);

  /// the name of the System Description File
  static ALIstring& SDFName(){
    return theSDFName;
  }

  /// the name of the Measurements File
  static ALIstring& MeasFName(){
    return theMeasFName;
  }

  /// the name of the report File
  static ALIstring& ReportFName(){
    return theReportFName;
  }

  /// the name of the File for storing the matrices
  static ALIstring& MatricesFName(){
    return theMatricesFName;
  }

///************ ACCESS INFO FROM STATIC DATA 


  ///----- Search a string in theParameters and return 1 if found
  static int getParameterValue( const ALIstring& sstr, ALIdouble& val );

  ///-----  Find an OptO name in theOptOList and return a pointer to it
  static OpticalObject* getOptOByName( const ALIstring& opto_name );

  ///-----  Find the first OptO of type 'opto_type' in theOptOList and return a pointer to it
  static OpticalObject* getOptOByType( const ALIstring& type );

  ///-----  Search an Entry name in the Entry* list and return a pointer to it
  static Entry* getEntryByName( const ALIstring& opto_name, const ALIstring& entry_name );

  ///-----  Search an Entry from the full entry path 
  /// (first substract the name of the OptO and then look in the Entry* list)
  static Entry* getEntryByName( const ALIstring& opto_entry_name) {
    ALIint slash_pos = opto_entry_name.rfind('/');
    ALIint length = opto_entry_name.length();
    ALIstring opto_name = opto_entry_name.substr(0, slash_pos);
    ALIstring entry_name = opto_entry_name.substr(slash_pos+1, length);
    Entry* entry = getEntryByName( opto_name, entry_name);
    return entry;
  }

  ///----- Get from theOptODictionary the list of component OptO types
  static ALIbool getComponentOptOTypes( const ALIstring& opto_type, std::vector<ALIstring>& vcomponents );

  ///----- Get from theOptOList the list of pointers to component OptOs 
  static ALIbool getComponentOptOs( const ALIstring& opto_name, std::vector<OpticalObject*>& vcomponents);

  static struct tm& MeasurementsTime() {
    return theMeasurementsTime;
  }

  static std::vector<OpticalAlignInfo> getOpticalAlignments() { return theOpticalAlignments; }


///*****************  SET DATA MEMBERS
  static void addEntryToList( Entry* entry ) {
     theEntryVector.push_back( entry );
     //-     std::cout << entry << entry->OptOCurrent()->name() << "ADDENTRY " << entry->name() << " " << EntryList().size() << std::endl;
  }

  static void addMeasurementToList( Measurement* measadd ) {
      theMeasurementVector.push_back( measadd);  
      //   std::cout << "ADD MEASUREMENT" << theMeasurementVector.size() << std::endl ;  
  }

  //----- Set the name of the System Description File
  static void setSDFName( const ALIstring& name ) {
    theSDFName = name;
  }
  //----- Set the name of the report File
  static void setReportFName( const ALIstring& name ) {
    theReportFName = name;
  }
  //----- Set the name of the matrices File
  static void setMatricesFName( const ALIstring& name ) {
    theMatricesFName = name;
  }

  static void setMeasurementsTime( struct tm& tim ) {
    theMeasurementsTime = tim;
  }


  static ALIbool readMeasurementsFromFile( ALIstring only1Date = ALIstring(""), ALIstring only1Time = ALIstring("") );
 

///********** private METHODS
private:
  /// Reorder the list of OptOs in a hierarchical structure (tree-like)
  static void reorderOptODictionary( const ALIstring& ssearch, std::vector< std::vector<ALIstring> >& OptODictionary2); 

  /// Read Measurements (to be implemented for reading from an external file the DATA of the measurements)
  //  static void readMeasurements( ALIFileIn& filein );

  /// Build for each measuremnt its link to the OptO that take part in it
  static void buildMeasurementsLinksToOptOs();

  static void SetValueDisplacementsFromReportOut();

///********** private DATA MEMBERS 
  /// Only instance of Model
  static Model* theInstance;

  /// parameters
  //-  static std::map< ALIstring, ALIdouble, std::less<ALIstring> > theParameters;
 
  /// std::vector of OptOs with components (in tree structure)
  static std::vector< std::vector<ALIstring> > theOptODictionary;

  /// map of OptO*/type of parent OptO, for navigation down the tree structure 
    //-  static multimap< ALIstring, OpticalObject*, std::less<ALIstring> > theOptOtree;
  /// map of OptO*/name of OptO for quick search based on name
    //  static map< ALIstring, OpticalObject*, std::less<ALIstring> > theOptOList;
  static std::vector< OpticalObject* > theOptOList;

  /// std::vector of all Entries
  static std::vector< Entry* > theEntryVector;

  /// std::vector of all Measurements
  static std::vector< Measurement* > theMeasurementVector;

  /// the name of the System Description File
  static ALIstring theSDFName;
  /// the name of the Measurements File
  static ALIstring theMeasFName;
  /// the name of the report File
  static ALIstring theReportFName;
  /// the name of the File for storing the matrices
  static ALIstring theMatricesFName;

  ///**************** FOR COPYING AN OPTO
 public:
 //----- Steers the storing of the components of OptO named 'optoname'
  static ALIbool createCopyComponentList( const ALIstring& optoname );
 //----- Get next object to copy from the stored list of components and copy it
  static OpticalObject* nextOptOToCopy();
 private:
 //----- Stores the components of opto 
  static ALIbool fillCopyComponentList( const OpticalObject* opto );
  //----- List of components of an OptO to copy
  static std::vector<OpticalObject*> theOptOsToCopyList;
  //----- Iterator of the list of components of an OptO to copy
  static std::vector<OpticalObject*>::const_iterator theOptOsToCopyListIterator;



   ///*************** FOR RANGE STUDIES
 public:
   static ALIint Ncmslinkrange;
   static std::vector<ALIdouble> CMSLinkRangeDetValue;


  ///*************** FOR CMS LINK SYSTEM (to fit it part by part)
public:
  void CMSLinkFit( ALIint cmslink);
private:
  void CMSLinkCleanModel();
  static void CMSLinkDeleteOptOs();
  static void CMSLinkSaveParamFittedSigma( ALIint cmslink );
  static void CMSLinkSaveParamFittedValueDisplacement( ALIint cmslink );
  static void CMSLinkRecoverParamFittedSigma(ALIint cmslink);
  static void CMSLinkRecoverParamFittedValueDisplacement(ALIint cmslink);

  static ALIint CMSLinkIteration;
 
  //----- METHODS FOR FITTING IN SEVERAL STEPS
  static void deleteOptO( const ALIstring& opto_name );
  static void deleteOptO( OpticalObject* opto );

  static void saveParamFittedSigma( const ALIstring& opto_name, const ALIstring& entry_name);

  static void saveParamFittedCorrelation( const ALIstring& opto_name1, const ALIstring& entry_name1, const ALIstring& opto_name2, const ALIstring& entry_name2);

  static void recoverParamFittedSigma( const ALIstring& opto_name, const ALIstring& entry_name, const ALIuint position );

 public:
  static ALIdouble getParamFittedSigmaVectorItem( const ALIuint position );
  static FittedEntriesReader* getFittedEntriesReader(){
    return theFittedEntriesReader; }

 private:
  static void cleanParamFittedSigmaVector() {
    ALIuint pfsv_size = theParamFittedSigmaVector.size();    
    for( ALIuint ii = 0; ii < pfsv_size; ii++) {
      theParamFittedSigmaVector.pop_back();
    }
  }

  static void cleanParamFittedValueDisplacementMap() {
    theParamFittedValueDisplacementMap.erase( theParamFittedValueDisplacementMap.begin(), theParamFittedValueDisplacementMap.end() );   
  }

  static void copyMeasurements( const std::vector<ALIstring>& wl );
 
 private:

  static cocoaStatus theCocoaStatus;

  static std::vector<ALIdouble> theParamFittedSigmaVector;

  static std::map<ALIstring, ALIdouble, std::less<ALIstring> > theParamFittedValueDisplacementMap;

  static struct tm theMeasurementsTime;

  static FittedEntriesReader* theFittedEntriesReader;

  static std::vector<OpticalAlignInfo> theOpticalAlignments;
};

#endif 
