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
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
class OpticalObject;
class EntryData;
enum EntryDim { ED_length, ED_angle, ED_nodim };

class Entry {
  friend std::ostream& operator<<(std::ostream& os, const Entry& c);

public:
  //----- Constructor / destructor
  //-  Entry(){ };
  Entry(const ALIstring& type);
  virtual ~Entry();

  //----- Fill the attributes
  void fill(const std::vector<ALIstring>& wordlist);
  //----- Fill the name (in derived classes is not simply calling setName)
  virtual void fillName(const ALIstring& name);
  //----- Fill the attributes setting them to 0.
  void fillNull();

  //----- Return value and sigma dimension factor (implemented in derived classes
  virtual ALIdouble ValueDimensionFactor() const { return 1.; }
  virtual ALIdouble SigmaDimensionFactor() const { return 1.; }
  virtual ALIdouble OutputValueDimensionFactor() const { return 1.; }
  virtual ALIdouble OutputSigmaDimensionFactor() const { return 1.; }

  //----- Displace the value by 'disp' (affine frame entries do it theirselves)
  virtual void displace(ALIdouble disp);
  //----- Tell the corresponding OptO to displace the Extra Entry (affine frame entries do it theirselves)
  virtual void displaceOriginal(ALIdouble disp);
  virtual void displaceOriginalOriginal(ALIdouble disp);

  //----- return the value, that is in Global Reference Frame
  virtual ALIdouble valueInGlobalReferenceFrame() const { return value(); };

  // Access DATA MEMBERS
  const ALIstring& name() const { return name_; }
  const ALIstring longName() const;
  const ALIstring& type() const { return type_; }
  ALIdouble value() const { return value_; }
  ALIdouble valueOriginalOriginal() const { return valueOriginalOriginal_; }
  ALIdouble sigma() const { return sigma_; }
  ALIdouble sigmaOriginalOriginal() const { return sigmaOriginalOriginal_; }
  ALIint quality() const { return quality_; }
  ALIint fitPos() const { return fitPos_; }
  OpticalObject* OptOCurrent() const {
    return OptOCurrent_;
  }  // non const, Displace( ) modifies it return _OptOCurrent;
  virtual ALIdouble valueDisplaced() const;
  ALIdouble valueDisplacementByFitting() const {
    //-    cout << this << " " << name() << " get valueDisplacementByFitting " << theValueDisplacementByFitting << endl;
    return valueDisplacementByFitting_;
  }
  void resetValueDisplacementByFitting();
  virtual ALIdouble startingDisplacement() { return 0.; };
  ALIdouble lastAdditionToValueDisplacementByFitting() const { return lastAdditionToValueDisplacementByFitting_; }
  void setLastAdditionToValueDisplacementByFitting(const ALIdouble val) {
    lastAdditionToValueDisplacementByFitting_ = val;
  }

public:
  // Set DATA MEMBERS
  void setName(const ALIstring& name) { name_ = name; }
  void setType(ALIstring type) { type_ = type; }
  void setValue(ALIdouble val) { value_ = val; }
  void setSigma(ALIdouble sig) { sigma_ = sig; }
  void setQuality(ALIuint qual) { quality_ = qual; }
  void setFitPos(const ALIint fitpos) { fitPos_ = fitpos; }
  void setOptOCurrent(OpticalObject* opto) { OptOCurrent_ = opto; }
  void addFittedDisplacementToValue(const ALIdouble val);

  void substractToHalfFittedDisplacementToValue();

  EntryDim getDimType() const { return theDimType; }

private:
  //-----  Fill the attributes with values read from a 'report.out' file
  void fillFromReportOutFileValue(EntryData* entryData);
  void fillFromReportOutFileSigma(const EntryData* entryData);
  void fillFromReportOutFileQuality(const EntryData* entryData);
  //-----  Fill the attributes with values read from the input file
  void fillFromInputFileValue(const std::vector<ALIstring>& wordlist);
  void fillFromInputFileSigma(const std::vector<ALIstring>& wordlist);
  void fillFromInputFileQuality(const std::vector<ALIstring>& wordlist);

private:
  // private DATA MEMBERS
protected:
  ALIstring name_;
  ALIstring type_;
  ALIdouble value_;
  ALIdouble valueOriginalOriginal_;
  ALIdouble sigma_;
  ALIdouble sigmaOriginalOriginal_;
  ALIuint quality_;
  OpticalObject* OptOCurrent_;
  ALIint fitPos_;

  ALIdouble valueDisplacementByFitting_;
  ALIdouble lastAdditionToValueDisplacementByFitting_;

  /*  
  virtual void DisplaceParameter( ALIint paramNo, ALIdouble displace ){ };  
  virtual ALIdouble Check_displacementDimensions( ALIdouble displace ){
    return displace; }
  */
  EntryDim theDimType;
};

#endif
