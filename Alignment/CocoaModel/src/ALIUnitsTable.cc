
#include "Alignment/CocoaModel/interface/ALIUnitsTable.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iomanip>
#include <cstdlib>
#include <cmath>  // include floating-point std::abs functions

ALIUnitsTable ALIUnitDefinition::theUnitsTable;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition::ALIUnitDefinition(ALIstring name, ALIstring symbol, ALIstring category, ALIdouble value)
    : Name(name), SymbolName(symbol), Value(value) {
  //
  //does the Category objet already exist ?
  size_t nbCat = theUnitsTable.size();
  size_t i = 0;
  while ((i < nbCat) && (theUnitsTable[i]->GetName() != category))
    i++;
  if (i == nbCat)
    theUnitsTable.push_back(new ALIUnitsCategory(category));
  CategoryIndex = i;
  //
  //insert this Unit in the Unitstable
  (theUnitsTable[CategoryIndex]->GetUnitsList()).emplace_back(shared_from_this());

  //update ALIstring max length for name and symbol
  theUnitsTable[i]->UpdateNameMxLen((ALIint)name.length());
  theUnitsTable[i]->UpdateSymbMxLen((ALIint)symbol.length());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition::~ALIUnitDefinition() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition::ALIUnitDefinition(ALIUnitDefinition& right) { *this = right; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition& ALIUnitDefinition::operator=(const ALIUnitDefinition& right) {
  if (this != &right) {
    Name = right.Name;
    SymbolName = right.SymbolName;
    Value = right.Value;
    CategoryIndex = right.CategoryIndex;
  }
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIint ALIUnitDefinition::operator==(const ALIUnitDefinition& right) const { return (this == &right); }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIint ALIUnitDefinition::operator!=(const ALIUnitDefinition& right) const { return (this != &right); }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
ALIdouble ALIUnitDefinition::GetValueOf(ALIstring stri) {
  if (theUnitsTable.empty())
    BuildUnitsTable();
  ALIstring name, symbol;
  for (size_t i = 0; i < theUnitsTable.size(); i++) {
    ALIUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
    for (size_t j = 0; j < units.size(); j++) {
      name = units[j]->GetName();
      symbol = units[j]->GetSymbol();
      if (stri == name || stri == symbol)
        return units[j]->GetValue();
    }
  }
  std::cout << "Warning from ALIUnitDefinition::GetValueOf(" << stri << ")."
            << " The unit " << stri << " does not exist in UnitsTable."
            << " Return Value = 0." << std::endl;
  return 0.;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIstring ALIUnitDefinition::GetCategory(ALIstring stri) {
  if (theUnitsTable.empty())
    BuildUnitsTable();
  ALIstring name, symbol;
  for (size_t i = 0; i < theUnitsTable.size(); i++) {
    ALIUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
    for (size_t j = 0; j < units.size(); j++) {
      name = units[j]->GetName();
      symbol = units[j]->GetSymbol();
      if (stri == name || stri == symbol)
        return theUnitsTable[i]->GetName();
    }
  }
  std::cout << "Warning from ALIUnitDefinition::GetCategory(" << stri << ")."
            << " The unit " << stri << " does not exist in UnitsTable."
            << " Return category = None" << std::endl;
  name = "None";
  return name;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void ALIUnitDefinition::PrintDefinition() {
  ALIint nameL = theUnitsTable[CategoryIndex]->GetNameMxLen();
  ALIint symbL = theUnitsTable[CategoryIndex]->GetSymbMxLen();
  std::cout << std::setw(nameL) << Name << " (" << std::setw(symbL) << SymbolName << ") = " << Value << std::endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void ALIUnitDefinition::BuildUnitsTable() {
  //Length
  std::make_shared<ALIUnitDefinition>("kilometer", "km", "Length", kilometer);
  std::make_shared<ALIUnitDefinition>("meter", "m", "Length", meter);
  std::make_shared<ALIUnitDefinition>("centimeter", "cm", "Length", centimeter);
  std::make_shared<ALIUnitDefinition>("millimeter", "mm", "Length", millimeter);
  std::make_shared<ALIUnitDefinition>("micrometer", "mum", "Length", micrometer);
  std::make_shared<ALIUnitDefinition>("nanometer", "nm", "Length", nanometer);
  std::make_shared<ALIUnitDefinition>("angstrom", "Ang", "Length", angstrom);
  std::make_shared<ALIUnitDefinition>("fermi", "fm", "Length", fermi);

  //Surface
  std::make_shared<ALIUnitDefinition>("kilometer2", "km2", "Surface", kilometer2);
  std::make_shared<ALIUnitDefinition>("meter2", "m2", "Surface", meter2);
  std::make_shared<ALIUnitDefinition>("centimeter2", "cm2", "Surface", centimeter2);
  std::make_shared<ALIUnitDefinition>("millimeter2", "mm2", "Surface", millimeter2);
  std::make_shared<ALIUnitDefinition>("barn", "barn", "Surface", barn);
  std::make_shared<ALIUnitDefinition>("millibarn", "mbarn", "Surface", millibarn);
  std::make_shared<ALIUnitDefinition>("microbarn", "mubarn", "Surface", microbarn);
  std::make_shared<ALIUnitDefinition>("nanobarn", "nbarn", "Surface", nanobarn);
  std::make_shared<ALIUnitDefinition>("picobarn", "pbarn", "Surface", picobarn);

  //Volume
  std::make_shared<ALIUnitDefinition>("kilometer3", "km3", "Volume", kilometer3);
  std::make_shared<ALIUnitDefinition>("meter3", "m3", "Volume", meter3);
  std::make_shared<ALIUnitDefinition>("centimeter3", "cm3", "Volume", centimeter3);
  std::make_shared<ALIUnitDefinition>("millimeter3", "mm3", "Volume", millimeter3);

  //Angle
  std::make_shared<ALIUnitDefinition>("radian", "rad", "Angle", radian);
  std::make_shared<ALIUnitDefinition>("milliradian", "mrad", "Angle", milliradian);
  std::make_shared<ALIUnitDefinition>("milliradian", "murad", "Angle", 0.001 * milliradian);
  std::make_shared<ALIUnitDefinition>("steradian", "sr", "Angle", steradian);
  std::make_shared<ALIUnitDefinition>("degree", "deg", "Angle", degree);

  //Time
  std::make_shared<ALIUnitDefinition>("second", "s", "Time", second);
  std::make_shared<ALIUnitDefinition>("millisecond", "ms", "Time", millisecond);
  std::make_shared<ALIUnitDefinition>("microsecond", "mus", "Time", microsecond);
  std::make_shared<ALIUnitDefinition>("nanosecond", "ns", "Time", nanosecond);
  std::make_shared<ALIUnitDefinition>("picosecond", "ps", "Time", picosecond);

  //Frequency
  std::make_shared<ALIUnitDefinition>("hertz", "Hz", "Frequency", hertz);
  std::make_shared<ALIUnitDefinition>("kilohertz", "kHz", "Frequency", kilohertz);
  std::make_shared<ALIUnitDefinition>("megahertz", "MHz", "Frequency", megahertz);

  //Electric charge
  std::make_shared<ALIUnitDefinition>("eplus", "e+", "Electric charge", eplus);
  std::make_shared<ALIUnitDefinition>("coulomb", "C", "Electric charge", coulomb);

  //Energy
  std::make_shared<ALIUnitDefinition>("electronvolt", "eV", "Energy", electronvolt);
  std::make_shared<ALIUnitDefinition>("kiloelectronvolt", "keV", "Energy", kiloelectronvolt);
  std::make_shared<ALIUnitDefinition>("megaelectronvolt", "MeV", "Energy", megaelectronvolt);
  std::make_shared<ALIUnitDefinition>("gigaelectronvolt", "GeV", "Energy", gigaelectronvolt);
  std::make_shared<ALIUnitDefinition>("teraelectronvolt", "TeV", "Energy", teraelectronvolt);
  std::make_shared<ALIUnitDefinition>("petaelectronvolt", "PeV", "Energy", petaelectronvolt);
  std::make_shared<ALIUnitDefinition>("joule", "J", "Energy", joule);

  //Mass
  std::make_shared<ALIUnitDefinition>("milligram", "mg", "Mass", milligram);
  std::make_shared<ALIUnitDefinition>("gram", "g", "Mass", gram);
  std::make_shared<ALIUnitDefinition>("kilogram", "kg", "Mass", kilogram);

  //Volumic Mass
  std::make_shared<ALIUnitDefinition>("g/cm3", "g/cm3", "Volumic Mass", g / cm3);
  std::make_shared<ALIUnitDefinition>("mg/cm3", "mg/cm3", "Volumic Mass", mg / cm3);
  std::make_shared<ALIUnitDefinition>("kg/m3", "kg/m3", "Volumic Mass", kg / m3);

  //Power
  std::make_shared<ALIUnitDefinition>("watt", "W", "Power", watt);

  //Force
  std::make_shared<ALIUnitDefinition>("newton", "N", "Force", newton);

  //Pressure
  std::make_shared<ALIUnitDefinition>("pascal", "Pa", "Pressure", pascal);
  std::make_shared<ALIUnitDefinition>("bar", "bar", "Pressure", bar);
  std::make_shared<ALIUnitDefinition>("atmosphere", "atm", "Pressure", atmosphere);

  //Electric current
  std::make_shared<ALIUnitDefinition>("ampere", "A", "Electric current", ampere);
  std::make_shared<ALIUnitDefinition>("milliampere", "mA", "Electric current", milliampere);
  std::make_shared<ALIUnitDefinition>("microampere", "muA", "Electric current", microampere);
  std::make_shared<ALIUnitDefinition>("nanoampere", "nA", "Electric current", nanoampere);

  //Electric potential
  std::make_shared<ALIUnitDefinition>("volt", "V", "Electric potential", volt);
  std::make_shared<ALIUnitDefinition>("kilovolt", "kV", "Electric potential", kilovolt);
  std::make_shared<ALIUnitDefinition>("megavolt", "MV", "Electric potential", megavolt);

  //Magnetic flux
  std::make_shared<ALIUnitDefinition>("weber", "Wb", "Magnetic flux", weber);

  //Magnetic flux density
  std::make_shared<ALIUnitDefinition>("tesla", "T", "Magnetic flux density", tesla);
  std::make_shared<ALIUnitDefinition>("kilogauss", "kG", "Magnetic flux density", kilogauss);
  std::make_shared<ALIUnitDefinition>("gauss", "G", "Magnetic flux density", gauss);

  //Temperature
  std::make_shared<ALIUnitDefinition>("kelvin", "K", "Temperature", kelvin);

  //Amount of substance
  std::make_shared<ALIUnitDefinition>("mole", "mol", "Amount of substance", mole);

  //Activity
  std::make_shared<ALIUnitDefinition>("becquerel", "Bq", "Activity", becquerel);
  std::make_shared<ALIUnitDefinition>("curie", "Ci", "Activity", curie);

  //Dose
  std::make_shared<ALIUnitDefinition>("gray", "Gy", "Dose", gray);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void ALIUnitDefinition::PrintUnitsTable() {
  std::cout << "\n          ----- The Table of Units ----- \n";
  for (size_t i = 0; i < theUnitsTable.size(); i++)
    theUnitsTable[i]->PrintCategory();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitsCategory::ALIUnitsCategory(ALIstring name) : Name(name), NameMxLen(0), SymbMxLen(0) {
  UnitsList = *(new ALIUnitsContainer);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitsCategory::~ALIUnitsCategory() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitsCategory::ALIUnitsCategory(ALIUnitsCategory& right) { *this = right; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitsCategory& ALIUnitsCategory::operator=(const ALIUnitsCategory& right) {
  if (this != &right) {
    Name = right.Name;
    UnitsList = right.UnitsList;
    NameMxLen = right.NameMxLen;
    SymbMxLen = right.SymbMxLen;
  }
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIint ALIUnitsCategory::operator==(const ALIUnitsCategory& right) const { return (this == &right); }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIint ALIUnitsCategory::operator!=(const ALIUnitsCategory& right) const { return (this != &right); }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void ALIUnitsCategory::PrintCategory() {
  std::cout << "\n  category: " << Name << std::endl;
  for (size_t i = 0; i < UnitsList.size(); i++)
    UnitsList[i]->PrintDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIBestUnit::ALIBestUnit(ALIdouble value, ALIstring category) {
  // find the category
  ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
  size_t nbCat = theUnitsTable.size();
  size_t i = 0;
  while ((i < nbCat) && (theUnitsTable[i]->GetName() != category))
    i++;
  if (i == nbCat) {
    std::cout << " ALIBestUnit: the category " << category << " does not exist !!" << std::endl;
    std::cerr << "Missing unit category !" << std::endl;
    abort();
  }
  //
  IndexOfCategory = i;
  nbOfVals = 1;
  Value[0] = value;
  Value[1] = 0.;
  Value[2] = 0.;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIBestUnit::ALIBestUnit(const CLHEP::Hep3Vector& value, ALIstring category) {
  // find the category
  ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
  size_t nbCat = theUnitsTable.size();
  size_t i = 0;
  while ((i < nbCat) && (theUnitsTable[i]->GetName() != category))
    i++;
  if (i == nbCat) {
    std::cerr << " ALIBestUnit: the category " << category << " does not exist." << std::endl;
    std::cerr << "Unit category not existing !" << std::endl;
    abort();
  }
  //
  IndexOfCategory = i;
  nbOfVals = 3;
  Value[0] = value.x();
  Value[1] = value.y();
  Value[2] = value.z();
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIBestUnit::~ALIBestUnit() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

std::ostream& operator<<(std::ostream& flux, ALIBestUnit a) {
  ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
  ALIUnitsContainer& List = theUnitsTable[a.IndexOfCategory]->GetUnitsList();
  ALIint len = theUnitsTable[a.IndexOfCategory]->GetSymbMxLen();

  ALIint ksup(-1), kinf(-1);
  ALIdouble umax(0.), umin(1.E12);
  ALIdouble rsup(1.E12), rinf(0.);

  //for a ThreeVector, choose the best unit for the biggest value
  ALIdouble value = std::max(std::max(std::abs(a.Value[0]), std::abs(a.Value[1])), std::abs(a.Value[2]));

  for (size_t k = 0; k < List.size(); k++) {
    ALIdouble unit = List[k]->GetValue();
    if (value == 1.E12) {
      if (unit > umax) {
        umax = unit;
        ksup = k;
      }
    } else if (value <= -1.E12) {
      if (unit < umin) {
        umin = unit;
        kinf = k;
      }
    }

    else {
      ALIdouble ratio = value / unit;
      if ((ratio >= 1.) && (ratio < rsup)) {
        rsup = ratio;
        ksup = k;
      }
      if ((ratio < 1.) && (ratio > rinf)) {
        rinf = ratio;
        kinf = k;
      }
    }
  }

  ALIint index = ksup;
  if (index == -1)
    index = kinf;
  if (index == -1)
    index = 0;

  for (ALIint j = 0; j < a.nbOfVals; j++) {
    flux << a.Value[j] / (List[index]->GetValue()) << " ";
  }

#ifdef ALIUSE_STD_NAMESPACE
  std::ios::fmtflags oldform = std::cout.flags();
#else
  //    ALIint oldform = std::cout.flags();
#endif

  flux.setf(std::ios::left, std::ios::adjustfield);
  flux << std::setw(len) << List[index]->GetSymbol();
  //??  flux.flags(oldform);

  return flux;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
