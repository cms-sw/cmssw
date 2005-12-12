//  COCOA class header file
//Id:  Entry.h
//CAT: Model
//
//   Base class for entries 
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ENTRY_HH
#define _ENTRY_HH
#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
class OpticalObject;
class EntryData;

class Entry
{
public:
  //----- Constructor / destructor 
  //-  Entry(){ };
  Entry( const ALIstring& type );
  virtual ~Entry();

  //----- Fill the attributes
  void fill( const std::vector<ALIstring>& wordlist);
  //----- Fill the name (in derived classes is not simply calling setName)
  virtual void fillName( const ALIstring& name );
  //----- Fill the attributes setting them to 0.
  void fillNull( );
 
  //----- Return value and sigma dimension factor (implemented in derived classes 
  virtual ALIdouble ValueDimensionFactor() const{ return 1.; }
  virtual ALIdouble SigmaDimensionFactor() const{ return 1.; }
  virtual ALIdouble OutputValueDimensionFactor() const{ return 1.; }
  virtual ALIdouble OutputSigmaDimensionFactor() const{ return 1.; }

  //----- Displace the value by 'disp' (affine frame entries do it theirselves)
  virtual void displace( ALIdouble disp );
  //----- Tell the corresponding OptO to displace the Extra Entry (affine frame entries do it theirselves)
  virtual void displaceOriginal( ALIdouble disp );
  virtual void displaceOriginalOriginal( ALIdouble disp );

  //----- return the value, that is in Global Reference Frame
  virtual ALIdouble valueInGlobalReferenceFrame() const {return value();}; 

 // Access DATA MEMBERS
  const ALIstring& name() const { return _name; }
  const ALIstring& type() const { return type_; }
  ALIdouble value() const { return _value; }
  ALIdouble valueOriginalOriginal() const { return _valueOriginalOriginal; }
  ALIdouble sigma() const { return _sigma; }
  ALIdouble sigmaOriginalOriginal() const { return _sigmaOriginalOriginal; }
  ALIint quality() const { return _quality; }
  ALIint fitPos() const { return theFitPos; }
  OpticalObject* OptOCurrent() const{ return _OptOCurrent; } // non const, Displace( ) modifies it return _OptOCurrent;
  virtual ALIdouble valueDisplaced() const;
  ALIdouble valueDisplacementByFitting() const{ 
    //-    cout << this << " " << name() << " get valueDisplacementByFitting " << theValueDisplacementByFitting << endl;
    return theValueDisplacementByFitting; }
  void resetValueDisplacementByFitting();
  virtual ALIdouble startingDisplacement(){ return 0.; };
  ALIdouble lastAdditionToValueDisplacementByFitting() const {
    return theLastAdditionToValueDisplacementByFitting; }
  void setLastAdditionToValueDisplacementByFitting( const ALIdouble val ){
    theLastAdditionToValueDisplacementByFitting = val; }

public:
 // Set DATA MEMBERS
  void setName( const ALIstring& name ) { _name = name; }
  void setType( ALIstring type ){ type_ = type; }
  void setValue( ALIdouble val ){ _value = val; }
  void setSigma( ALIdouble sig ){ _sigma = sig; }
  void setQuality( ALIuint qual ){ _quality = qual; }
  void setFitPos( const ALIint fitpos ) { theFitPos = fitpos; } 
  void setOptOCurrent( OpticalObject* opto ){ _OptOCurrent = opto; }
  void addFittedDisplacementToValue(const ALIdouble val);

  void substractToHalfFittedDisplacementToValue();


private:
  //-----  Fill the attributes with values read from a 'report.out' file
  void fillFromReportOutFileValue( EntryData* entryData );
  void fillFromReportOutFileSigma( const EntryData* entryData );
  void fillFromReportOutFileQuality( const EntryData* entryData );
  //-----  Fill the attributes with values read from the input file
  void fillFromInputFileValue( const std::vector<ALIstring>& wordlist );
  void fillFromInputFileSigma( const std::vector<ALIstring>& wordlist );
  void fillFromInputFileQuality( const std::vector<ALIstring>& wordlist );

private:
 // private DATA MEMBERS
protected:
  ALIstring _name;
  ALIstring type_;
  ALIdouble _value;
  ALIdouble _valueOriginalOriginal;
  ALIdouble _sigma;
  ALIdouble _sigmaOriginalOriginal;
  ALIuint _quality;
  OpticalObject* _OptOCurrent;
  ALIint theFitPos;

  ALIdouble theValueDisplacementByFitting;
  ALIdouble theLastAdditionToValueDisplacementByFitting;

  /*  
  virtual void DisplaceParameter( ALIint paramNo, ALIdouble displace ){ };  
  virtual ALIdouble Check_displacementDimensions( ALIdouble displace ){
    return displace; }
  */

};

#endif
