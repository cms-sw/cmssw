//
// ********************************************************************
// * DISCLAIMER                                                       *
// *                                                                  *
// * The following disclaimer summarizes all the specific disclaimers *
// * of contributors to this software. The specific disclaimers,which *
// * govern, are listed with their locations in:                      *
// *   http://cern.ch/geant4/license                                  *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.                                                             *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * GEANT4 collaboration.                                            *
// * By copying,  distributing  or modifying the Program (or any work *
// * based  on  the Program)  you indicate  your  acceptance of  this *
// * statement, and all its terms.                                    *
// ********************************************************************
//
// -----------------------------------------------------------------
//
//      ------------------- class ALIUnitsTable -----------------
//
// Class description:
//
// This class maintains a table of Units.
// A Unit has a name, a symbol, a value and belong to a category (i.e. its
// dimensional definition): Length, Time, Energy, etc...
// The Units are grouped by category. The TableOfUnits is a list of categories.
// The class G4BestUnit allows to convert automaticaly a physical quantity
// from its internal value into the most appropriate Unit of the same category.
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef ALIUnitsTable_HH
#define ALIUnitsTable_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
#include <CLHEP/Vector/ThreeVector.h>

class ALIUnitsCategory;
typedef std::vector<ALIUnitsCategory*> ALIUnitsTable;


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class ALIUnitDefinition
{
public:  // with description

    ALIUnitDefinition(ALIstring name, ALIstring symbol,
                     ALIstring category, ALIdouble value);
	    
public:  // without description
	    
   ~ALIUnitDefinition();
    ALIint operator==(const ALIUnitDefinition&) const;
    ALIint operator!=(const ALIUnitDefinition&) const;
    
private:

    ALIUnitDefinition(ALIUnitDefinition&);
    ALIUnitDefinition& operator=(const ALIUnitDefinition&);
   
public:  // with description

    ALIstring      GetName()   const {return Name;}
    ALIstring      GetSymbol() const {return SymbolName;}
    ALIdouble      GetValue()  const {return Value;}
    
    void          PrintDefinition();
    
    static void BuildUnitsTable();    
    static void PrintUnitsTable();
    
    static ALIUnitsTable& GetUnitsTable() {return theUnitsTable;}

    static ALIdouble GetValueOf (ALIstring);
    static ALIstring GetCategory(ALIstring);

private:

    ALIstring Name;            // SI name
    ALIstring SymbolName;      // SI symbol
    ALIdouble Value;           // value in the internal system of units
    
    static 
    ALIUnitsTable theUnitsTable;   // table of Units
    
    
    size_t CategoryIndex;         // category index of this unit
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

typedef std::vector<ALIUnitDefinition*> ALIUnitsContainer;

class ALIUnitsCategory
{
public:  // without description

    ALIUnitsCategory(ALIstring name);
   ~ALIUnitsCategory();
    ALIint operator==(const ALIUnitsCategory&) const;
    ALIint operator!=(const ALIUnitsCategory&) const;
    
private:

    ALIUnitsCategory(ALIUnitsCategory&);
    ALIUnitsCategory& operator=(const ALIUnitsCategory&);
   
public:  // without description

    ALIstring          GetName()      const {return Name;}
    ALIUnitsContainer& GetUnitsList()       {return UnitsList;}
    ALIint             GetNameMxLen() const {return NameMxLen;}
    ALIint             GetSymbMxLen() const {return SymbMxLen;}
    void  UpdateNameMxLen(ALIint len) {if (NameMxLen<len) NameMxLen=len;}
    void  UpdateSymbMxLen(ALIint len) {if (SymbMxLen<len) SymbMxLen=len;}
    void  PrintCategory();

private:

    ALIstring          Name;        // dimensional family: Length,Volume,Energy ...
    ALIUnitsContainer  UnitsList;   // List of units in this family
    ALIint             NameMxLen;   // max length of the units name
    ALIint             SymbMxLen;   // max length of the units symbol
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class ALIBestUnit
{
public:  // with description

    ALIBestUnit(ALIdouble internalValue, ALIstring category);
    ALIBestUnit(const CLHEP::Hep3Vector& internalValue, ALIstring category);    
      // These constructors convert a physical quantity from its internalValue
      // into the most appropriate unit of the same category.
      // In practice it builds an object VU = (newValue, newUnit)

   ~ALIBestUnit();
   
public:  // without description

    ALIdouble*  GetValue()                 {return Value;}
    ALIstring   GetCategory()        const {return Category;}
    size_t     GetIndexOfCategory() const {return IndexOfCategory;}
    
public:  // with description 
   
    friend
    std::ostream&  operator<<(std::ostream&,ALIBestUnit VU);
      // Default format to print the objet VU above.

private:

    ALIdouble   Value[3];        // value in the internal system of units
    ALIint      nbOfVals;        // ALIdouble=1; CLHEP::Hep3Vector=3
    ALIstring   Category;        // dimensional family: Length,Volume,Energy ...
    size_t IndexOfCategory;     // position of Category in UnitsTable
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
