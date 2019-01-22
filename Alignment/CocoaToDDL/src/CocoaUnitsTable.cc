 
#include "Alignment/CocoaToDDL/interface/CocoaUnitsTable.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iomanip>
#include <cmath>		// include floating-point std::abs functions

CocoaUnitsTable      CocoaUnitDefinition::theUnitsTable;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitDefinition::CocoaUnitDefinition(const ALIstring& name, const ALIstring& symbol,
                                   const ALIstring& category, ALIdouble value)
  : Name(name),SymbolName(symbol),Value(value)				   
{
    //
    //does the Category objet already exist ?
    size_t nbCat = theUnitsTable.size();
    size_t i = 0;
    while ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
    if (i == nbCat) theUnitsTable.push_back( new CocoaUnitsCategory(category));
    CategoryIndex = i;
    //
    //insert this Unit in the Unitstable
    (theUnitsTable[CategoryIndex]->GetUnitsList()).push_back(this);
    
    //update string max length for name and symbol
    theUnitsTable[i]->UpdateNameMxLen((ALIint)name.length());
    theUnitsTable[i]->UpdateSymbMxLen((ALIint)symbol.length());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitDefinition::~CocoaUnitDefinition()
{
  for (size_t i=0;i<theUnitsTable.size();i++)
  {
    delete theUnitsTable[i];
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitDefinition::CocoaUnitDefinition(const CocoaUnitDefinition& right)
{
    *this = right;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitDefinition& CocoaUnitDefinition::operator=(const CocoaUnitDefinition& right)
{
  if (this != &right)
    {
      Name          = right.Name;
      SymbolName    = right.SymbolName;
      Value         = right.Value;
      CategoryIndex = right.CategoryIndex;
    }
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIint CocoaUnitDefinition::operator==(const CocoaUnitDefinition& right) const
{
  return (this == &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIint CocoaUnitDefinition::operator!=(const CocoaUnitDefinition &right) const
{
  return (this != &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitsTable& CocoaUnitDefinition::GetUnitsTable()
{
  return theUnitsTable;
}
 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIdouble CocoaUnitDefinition::GetValueOf(const ALIstring& str)
{
  if(theUnitsTable.empty()) BuildUnitsTable();
  ALIstring name,symbol;
  for (size_t i=0;i<theUnitsTable.size();i++)
     { CocoaUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
       for (size_t j=0;j<units.size();j++)
          { name=units[j]->GetName(); symbol=units[j]->GetSymbol();
            if(str==name||str==symbol) 
               return units[j]->GetValue();
          }
     }
  std::cout << "Warning from CocoaUnitDefinition::GetValueOf(" << str << ")."
       << " The unit " << str << " does not exist in UnitsTable."
       << " Return Value = 0." << std::endl;     
  return 0.;             
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
  
ALIstring CocoaUnitDefinition::GetCategory(const ALIstring& str)
{
  if(theUnitsTable.empty()) BuildUnitsTable();
  ALIstring name,symbol;
  for (size_t i=0;i<theUnitsTable.size();i++)
     { CocoaUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
       for (size_t j=0;j<units.size();j++)
          { name=units[j]->GetName(); symbol=units[j]->GetSymbol();
            if(str==name||str==symbol) 
               return theUnitsTable[i]->GetName();
          }
     }
  std::cout << "Warning from CocoaUnitDefinition::GetCategory(" << str << ")."
       << " The unit " << str << " does not exist in UnitsTable."
       << " Return category = None" << std::endl;
  name = "None";     
  return name;             
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void CocoaUnitDefinition::PrintDefinition()
{
  ALIint nameL = theUnitsTable[CategoryIndex]->GetNameMxLen();
  ALIint symbL = theUnitsTable[CategoryIndex]->GetSymbMxLen();
  std::cout << std::setw(nameL) << Name << " (" 
         << std::setw(symbL) << SymbolName << ") = " << Value << std::endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void CocoaUnitDefinition::BuildUnitsTable()
{
 //Length
 new CocoaUnitDefinition(     "meter","m"       ,"Length",meter);
 new CocoaUnitDefinition("centimeter","cm"      ,"Length",centimeter); 
 new CocoaUnitDefinition("millimeter","mm"      ,"Length",millimeter);
 // new CocoaUnitDefinition("micrometer","um"      ,"Length",micrometer);
 new CocoaUnitDefinition("micrometer","mum"     ,"Length",micrometer);
 new CocoaUnitDefinition( "nanometer","nm"      ,"Length",nanometer);
 new CocoaUnitDefinition(  "angstrom","Ang"     ,"Length",angstrom);    
 new CocoaUnitDefinition(     "fermi","fm"      ,"Length",fermi);
 
 //Surface
 new CocoaUnitDefinition( "kilometer2","km2"    ,"Surface",kilometer2);
 new CocoaUnitDefinition(     "meter2","m2"     ,"Surface",meter2);
 new CocoaUnitDefinition("centimeter2","cm2"    ,"Surface",centimeter2); 
 new CocoaUnitDefinition("millimeter2","mm2"    ,"Surface",millimeter2);
 new CocoaUnitDefinition(       "barn","barn"   ,"Surface",barn);
 new CocoaUnitDefinition(  "millibarn","mbarn"  ,"Surface",millibarn);   
 new CocoaUnitDefinition(  "microbarn","mubarn" ,"Surface",microbarn);
 new CocoaUnitDefinition(   "nanobarn","nbarn"  ,"Surface",nanobarn);
 new CocoaUnitDefinition(   "picobarn","pbarn"  ,"Surface",picobarn);
 
 //Volume
 new CocoaUnitDefinition( "kilometer3","km3"    ,"Volume",kilometer3);
 new CocoaUnitDefinition(     "meter3","m3"     ,"Volume",meter3);
 new CocoaUnitDefinition("centimeter3","cm3"    ,"Volume",centimeter3); 
 new CocoaUnitDefinition("millimeter3","mm3"    ,"Volume",millimeter3);

 //Angle
 new CocoaUnitDefinition(     "radian","rad"    ,"Angle",radian);
 new CocoaUnitDefinition("milliradian","mrad"   ,"Angle",milliradian); 
 new CocoaUnitDefinition(  "steradian","sr"     ,"Angle",steradian);
 new CocoaUnitDefinition(     "degree","deg"    ,"Angle",degree);
 
 //Time
 new CocoaUnitDefinition(     "second","s"      ,"Time",second);
 new CocoaUnitDefinition("millisecond","ms"     ,"Time",millisecond);
 new CocoaUnitDefinition("microsecond","mus"    ,"Time",microsecond);
 new CocoaUnitDefinition( "nanosecond","ns"     ,"Time",nanosecond);
 new CocoaUnitDefinition( "picosecond","ps"     ,"Time",picosecond);
 
 //Frequency
 new CocoaUnitDefinition(    "hertz","Hz"       ,"Frequency",hertz);
 new CocoaUnitDefinition("kilohertz","kHz"      ,"Frequency",kilohertz);
 new CocoaUnitDefinition("megahertz","MHz"      ,"Frequency",megahertz);
 
 //Electric charge
 new CocoaUnitDefinition(  "eplus","e+"         ,"Electric charge",eplus);
 new CocoaUnitDefinition("coulomb","C"          ,"Electric charge",coulomb); 
 
 //Energy
 new CocoaUnitDefinition(    "electronvolt","eV" ,"Energy",electronvolt);
 new CocoaUnitDefinition("kiloelectronvolt","keV","Energy",kiloelectronvolt);
 new CocoaUnitDefinition("megaelectronvolt","MeV","Energy",megaelectronvolt);
 new CocoaUnitDefinition("gigaelectronvolt","GeV","Energy",gigaelectronvolt);
 new CocoaUnitDefinition("teraelectronvolt","TeV","Energy",teraelectronvolt);
 new CocoaUnitDefinition("petaelectronvolt","PeV","Energy",petaelectronvolt);
 new CocoaUnitDefinition(           "joule","J"  ,"Energy",joule);
 
 //Mass
 new CocoaUnitDefinition("milligram","mg","Mass",milligram);
 new CocoaUnitDefinition(     "gram","g" ,"Mass",gram);
 new CocoaUnitDefinition( "kilogram","kg","Mass",kilogram);
 
 //Volumic Mass
 new CocoaUnitDefinition( "g/cm3", "g/cm3","Volumic Mass", g/cm3);
 new CocoaUnitDefinition("mg/cm3","mg/cm3","Volumic Mass",mg/cm3);
 new CocoaUnitDefinition("kg/m3", "kg/m3", "Volumic Mass",kg/m3);
 
 //Power
 new CocoaUnitDefinition("watt","W","Power",watt);
 
 //Force
 new CocoaUnitDefinition("newton","N","Force",newton);
 
 //Pressure
 new CocoaUnitDefinition(    "pascal","Pa" ,"Pressure",pascal);
 new CocoaUnitDefinition(       "bar","bar","Pressure",bar); 
 new CocoaUnitDefinition("atmosphere","atm","Pressure",atmosphere);
 
 //Electric current
 new CocoaUnitDefinition(     "ampere","A"  ,"Electric current",ampere);
 new CocoaUnitDefinition("milliampere","mA" ,"Electric current",milliampere);
 new CocoaUnitDefinition("microampere","muA","Electric current",microampere);
 new CocoaUnitDefinition( "nanoampere","nA" ,"Electric current",nanoampere);   
 
 //Electric potential
 new CocoaUnitDefinition(    "volt","V" ,"Electric potential",volt); 
 new CocoaUnitDefinition("kilovolt","kV","Electric potential",kilovolt);
 new CocoaUnitDefinition("megavolt","MV","Electric potential",megavolt);
 
 //Magnetic flux
 new CocoaUnitDefinition("weber","Wb","Magnetic flux",weber);
 
 //Magnetic flux density
 new CocoaUnitDefinition(    "tesla","T" ,"Magnetic flux density",tesla);
 new CocoaUnitDefinition("kilogauss","kG","Magnetic flux density",kilogauss);
 new CocoaUnitDefinition(    "gauss","G" ,"Magnetic flux density",gauss);
 
 //Temperature
 new CocoaUnitDefinition("kelvin","K","Temperature",kelvin);
 
 //Amount of substance
 new CocoaUnitDefinition("mole","mol","Amount of substance",mole);
 
 //Activity
 new CocoaUnitDefinition("becquerel","Bq","Activity",becquerel);
 new CocoaUnitDefinition(    "curie","Ci","Activity",curie);
 
 //Dose
 new CocoaUnitDefinition("gray","Gy","Dose",gray);                          
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void CocoaUnitDefinition::PrintUnitsTable()
{
  std::cout << "\n          ----- The Table of Units ----- \n";
  for(size_t i=0;i<theUnitsTable.size();i++)
  {
    theUnitsTable[i]->PrintCategory();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
   
CocoaUnitsCategory::CocoaUnitsCategory(const ALIstring& name)
  : Name(name),UnitsList(),NameMxLen(0),SymbMxLen(0)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitsCategory::~CocoaUnitsCategory()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitsCategory::CocoaUnitsCategory(const CocoaUnitsCategory& right)
{
  *this = right;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaUnitsCategory& CocoaUnitsCategory::operator=(const CocoaUnitsCategory& right)
{
  if (this != &right)
    {
      Name      = right.Name;
      UnitsList = right.UnitsList;
      NameMxLen = right.NameMxLen;
      SymbMxLen = right.SymbMxLen;
    }
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIint CocoaUnitsCategory::operator==(const CocoaUnitsCategory& right) const
{
  return (this == &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIint CocoaUnitsCategory::operator!=(const CocoaUnitsCategory &right) const
{
  return (this != &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void CocoaUnitsCategory::PrintCategory()
{
  std::cout << "\n  category: " << Name << std::endl;
  for(size_t i=0;i<UnitsList.size();i++)
      UnitsList[i]->PrintDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
       
CocoaBestUnit::CocoaBestUnit(ALIdouble value, const ALIstring& category)
{
 // find the category
    CocoaUnitsTable& theUnitsTable = CocoaUnitDefinition::GetUnitsTable();
    if( theUnitsTable.empty() ) CocoaUnitDefinition::BuildUnitsTable(); //t should be done somewhere else
    size_t nbCat = theUnitsTable.size();
    size_t i = 0;
    while
     ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
    if (i == nbCat) 
       { std::cout << " CocoaBestUnit: the category " << category 
		   << " does not exist !!" << nbCat << std::endl;
       std::exception();//"Missing unit category !");
       }  
  //
    IndexOfCategory = i;
    nbOfVals = 1;
    Value[0] = value; Value[1] = 0.; Value[2] = 0.;

    //COCOA internal units are in meters, not mm as in CLHEP
    if(category == "Length" ) {
      Value[0] *= 1000.;
      Value[1] *= 1000.;
      Value[2] *= 1000.;
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
       
CocoaBestUnit::CocoaBestUnit(const CLHEP::Hep3Vector& value, const ALIstring& category)
{
 // find the category
    CocoaUnitsTable& theUnitsTable = CocoaUnitDefinition::GetUnitsTable();
    size_t nbCat = theUnitsTable.size();
    size_t i = 0;
    while
     ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
    if (i == nbCat) 
       { std::cerr << " CocoaBestUnit: the category " << category 
                << " does not exist." << std::endl;
       std::exception();//"Unit category not existing !");
       }  
  //
    IndexOfCategory = i;
    nbOfVals = 3;
    Value[0] = value.x();
    Value[1] = value.y();
    Value[2] = value.z();
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
CocoaBestUnit::~CocoaBestUnit()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
std::ostream& operator<<(std::ostream& flux, CocoaBestUnit a)
{
  CocoaUnitsTable& theUnitsTable = CocoaUnitDefinition::GetUnitsTable();
  CocoaUnitsContainer& List = theUnitsTable[a.IndexOfCategory]
                           ->GetUnitsList();
  ALIint len = theUnitsTable[a.IndexOfCategory]->GetSymbMxLen();
                           
  ALIint    ksup(-1), kinf(-1);
  ALIdouble umax(0.), umin(ALI_DBL_MAX);
  ALIdouble rsup(ALI_DBL_MAX), rinf(0.);

  //for a ThreeVector, choose the best unit for the biggest value 
  ALIdouble value = std::max(std::max(std::abs(a.Value[0]),std::abs(a.Value[1])),
                              std::abs(a.Value[2]));

  for (size_t k=0; k<List.size(); k++)
     {
       ALIdouble unit = List[k]->GetValue();
            if (value==ALI_DBL_MAX) {if(unit>umax) {umax=unit; ksup=k;}}
       else if (value<=ALI_DBL_MIN) {if(unit<umin) {umin=unit; kinf=k;}}
       
       else { ALIdouble ratio = value/unit;
              if ((ratio>=1.)&&(ratio<rsup)) {rsup=ratio; ksup=k;}
              if ((ratio< 1.)&&(ratio>rinf)) {rinf=ratio; kinf=k;}
	    } 
     }
	 
  ALIint index=ksup; if(index==-1) index=kinf; if(index==-1) index=0;
  
  for (ALIint j=0; j<a.nbOfVals; j++) 
     {flux << a.Value[j]/(List[index]->GetValue()) << " ";}

  std::ios::fmtflags oldform = flux.flags();

  flux.setf(std::ios::left,std::ios::adjustfield);
  flux << std::setw(len) << List[index]->GetSymbol();       
  flux.flags(oldform);

  return flux;
}       

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
        
