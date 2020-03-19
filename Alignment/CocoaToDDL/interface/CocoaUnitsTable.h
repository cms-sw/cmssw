//
// This class is copied from G4UnitsTable
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef CocoaUnitsTable_HH
#define CocoaUnitsTable_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
#include "CLHEP/Vector/ThreeVector.h"

class CocoaUnitsCategory;
typedef std::vector<CocoaUnitsCategory*> CocoaUnitsTable;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class CocoaUnitDefinition {
public:  // with description
  CocoaUnitDefinition(const ALIstring& name, const ALIstring& symbol, const ALIstring& category, ALIdouble value);

public:  // without description
  ~CocoaUnitDefinition();
  ALIint operator==(const CocoaUnitDefinition&) const;
  ALIint operator!=(const CocoaUnitDefinition&) const;

private:
  CocoaUnitDefinition(const CocoaUnitDefinition&);
  CocoaUnitDefinition& operator=(const CocoaUnitDefinition&);

public:  // with description
  const ALIstring& GetName() const { return Name; }
  const ALIstring& GetSymbol() const { return SymbolName; }
  ALIdouble GetValue() const { return Value; }

  void PrintDefinition();

  static void BuildUnitsTable();
  static void PrintUnitsTable();

  static CocoaUnitsTable& GetUnitsTable();

  static ALIdouble GetValueOf(const ALIstring&);
  static ALIstring GetCategory(const ALIstring&);

private:
  ALIstring Name;        // SI name
  ALIstring SymbolName;  // SI symbol
  ALIdouble Value;       // value in the internal system of units

  static CocoaUnitsTable theUnitsTable;  // table of Units

  size_t CategoryIndex;  // category index of this unit
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

typedef std::vector<CocoaUnitDefinition*> CocoaUnitsContainer;

class CocoaUnitsCategory {
public:  // without description
  CocoaUnitsCategory(const ALIstring& name);
  ~CocoaUnitsCategory();
  ALIint operator==(const CocoaUnitsCategory&) const;
  ALIint operator!=(const CocoaUnitsCategory&) const;

private:
  CocoaUnitsCategory(const CocoaUnitsCategory&);
  CocoaUnitsCategory& operator=(const CocoaUnitsCategory&);

public:  // without description
  const ALIstring& GetName() const { return Name; }
  CocoaUnitsContainer& GetUnitsList() { return UnitsList; }
  ALIint GetNameMxLen() const { return NameMxLen; }
  ALIint GetSymbMxLen() const { return SymbMxLen; }
  void UpdateNameMxLen(ALIint len) {
    if (NameMxLen < len)
      NameMxLen = len;
  }
  void UpdateSymbMxLen(ALIint len) {
    if (SymbMxLen < len)
      SymbMxLen = len;
  }
  void PrintCategory();

private:
  ALIstring Name;                 // dimensional family: Length,Volume,Energy ...
  CocoaUnitsContainer UnitsList;  // List of units in this family
  ALIint NameMxLen;               // max length of the units name
  ALIint SymbMxLen;               // max length of the units symbol
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class CocoaBestUnit {
public:  // with description
  CocoaBestUnit(ALIdouble internalValue, const ALIstring& category);
  CocoaBestUnit(const CLHEP::Hep3Vector& internalValue, const ALIstring& category);
  // These constructors convert a physical quantity from its internalValue
  // into the most appropriate unit of the same category.
  // In practice it builds an object VU = (newValue, newUnit)

  ~CocoaBestUnit();

public:  // without description
  ALIdouble* GetValue() { return Value; }
  const ALIstring& GetCategory() const { return Category; }
  size_t GetIndexOfCategory() const { return IndexOfCategory; }

public:  // with description
  friend std::ostream& operator<<(std::ostream&, CocoaBestUnit VU);
  // Default format to print the objet VU above.

private:
  ALIdouble Value[3];      // value in the internal system of units
  ALIint nbOfVals;         // ALIdouble=1; CLHEP::Hep3Vector=3
  ALIstring Category;      // dimensional family: Length,Volume,Energy ...
  size_t IndexOfCategory;  // position of Category in UnitsTable
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
