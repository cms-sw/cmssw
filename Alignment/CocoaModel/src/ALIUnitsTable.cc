 
#include "Alignment/CocoaModel/interface/ALIUnitsTable.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iomanip>
#include <cstdlib>

ALIUnitsTable      ALIUnitDefinition::theUnitsTable;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIUnitDefinition::ALIUnitDefinition(ALIstring name, ALIstring symbol,
                                   ALIstring category, ALIdouble value)
  : Name(name),SymbolName(symbol),Value(value)				   
{
  //
  //does the Category objet already exist ?
  size_t nbCat = theUnitsTable.size();
  size_t i = 0;
  while ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
  if (i == nbCat) theUnitsTable.push_back( new ALIUnitsCategory(category));
  CategoryIndex = i;
  //
  //insert this Unit in the Unitstable
  (theUnitsTable[CategoryIndex]->GetUnitsList()).push_back(this);
  
  //update ALIstring max length for name and symbol
  theUnitsTable[i]->UpdateNameMxLen((ALIint)name.length());
  theUnitsTable[i]->UpdateSymbMxLen((ALIint)symbol.length());
  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIUnitDefinition::~ALIUnitDefinition()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition::ALIUnitDefinition(ALIUnitDefinition& right)
{
  *this = right;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIUnitDefinition& ALIUnitDefinition::operator=(const ALIUnitDefinition& right)
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

ALIint ALIUnitDefinition::operator==(const ALIUnitDefinition& right) const
{
  return (this == &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

ALIint ALIUnitDefinition::operator!=(const ALIUnitDefinition &right) const
{
  return (this != &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
ALIdouble ALIUnitDefinition::GetValueOf(ALIstring stri)
{
  if(theUnitsTable.size()==0) BuildUnitsTable();
  ALIstring name,symbol;
  for (size_t i=0;i<theUnitsTable.size();i++)
     { ALIUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
       for (size_t j=0;j<units.size();j++)
          { name=units[j]->GetName(); symbol=units[j]->GetSymbol();
            if(stri==name||stri==symbol) 
               return units[j]->GetValue();
          }
     }
  std::cout << "Warning from ALIUnitDefinition::GetValueOf(" << stri << ")."
       << " The unit " << stri << " does not exist in UnitsTable."
       << " Return Value = 0." << std::endl;     
  return 0.;             
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
  
ALIstring ALIUnitDefinition::GetCategory(ALIstring stri)
{
  if(theUnitsTable.size()==0) BuildUnitsTable();
  ALIstring name,symbol;
  for (size_t i=0;i<theUnitsTable.size();i++)
     { ALIUnitsContainer& units = theUnitsTable[i]->GetUnitsList();
       for (size_t j=0;j<units.size();j++)
          { name=units[j]->GetName(); symbol=units[j]->GetSymbol();
            if(stri==name||stri==symbol) 
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
 
void ALIUnitDefinition::PrintDefinition()
{
  ALIint nameL = theUnitsTable[CategoryIndex]->GetNameMxLen();
  ALIint symbL = theUnitsTable[CategoryIndex]->GetSymbMxLen();
  std::cout << std::setw(nameL) << Name << " (" 
         << std::setw(symbL) << SymbolName << ") = " << Value << std::endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void ALIUnitDefinition::BuildUnitsTable()
{
 //Length
 new ALIUnitDefinition( "kilometer","km"      ,"Length",kilometer);
 new ALIUnitDefinition(     "meter","m"       ,"Length",meter);
 new ALIUnitDefinition("centimeter","cm"      ,"Length",centimeter); 
 new ALIUnitDefinition("millimeter","mm"      ,"Length",millimeter);
 new ALIUnitDefinition("micrometer","mum"     ,"Length",micrometer);
 new ALIUnitDefinition( "nanometer","nm"      ,"Length",nanometer);
 new ALIUnitDefinition(  "angstrom","Ang"     ,"Length",angstrom);    
 new ALIUnitDefinition(     "fermi","fm"      ,"Length",fermi);
 
 //Surface
 new ALIUnitDefinition( "kilometer2","km2"    ,"Surface",kilometer2);
 new ALIUnitDefinition(     "meter2","m2"     ,"Surface",meter2);
 new ALIUnitDefinition("centimeter2","cm2"    ,"Surface",centimeter2); 
 new ALIUnitDefinition("millimeter2","mm2"    ,"Surface",millimeter2);
 new ALIUnitDefinition(       "barn","barn"   ,"Surface",barn);
 new ALIUnitDefinition(  "millibarn","mbarn"  ,"Surface",millibarn);   
 new ALIUnitDefinition(  "microbarn","mubarn" ,"Surface",microbarn);
 new ALIUnitDefinition(   "nanobarn","nbarn"  ,"Surface",nanobarn);
 new ALIUnitDefinition(   "picobarn","pbarn"  ,"Surface",picobarn);
 
 //Volume
 new ALIUnitDefinition( "kilometer3","km3"    ,"Volume",kilometer3);
 new ALIUnitDefinition(     "meter3","m3"     ,"Volume",meter3);
 new ALIUnitDefinition("centimeter3","cm3"    ,"Volume",centimeter3); 
 new ALIUnitDefinition("millimeter3","mm3"    ,"Volume",millimeter3);

 //Angle
 new ALIUnitDefinition(     "radian","rad"    ,"Angle",radian);
 new ALIUnitDefinition("milliradian","mrad"   ,"Angle",milliradian); 
 new ALIUnitDefinition("milliradian","murad"   ,"Angle",0.001*milliradian); 
 new ALIUnitDefinition(  "steradian","sr"     ,"Angle",steradian);
 new ALIUnitDefinition(     "degree","deg"    ,"Angle",degree);
 
 //Time
 new ALIUnitDefinition(     "second","s"      ,"Time",second);
 new ALIUnitDefinition("millisecond","ms"     ,"Time",millisecond);
 new ALIUnitDefinition("microsecond","mus"    ,"Time",microsecond);
 new ALIUnitDefinition( "nanosecond","ns"     ,"Time",nanosecond);
 new ALIUnitDefinition( "picosecond","ps"     ,"Time",picosecond);
 
 //Frequency
 new ALIUnitDefinition(    "hertz","Hz"       ,"Frequency",hertz);
 new ALIUnitDefinition("kilohertz","kHz"      ,"Frequency",kilohertz);
 new ALIUnitDefinition("megahertz","MHz"      ,"Frequency",megahertz);
 
 //Electric charge
 new ALIUnitDefinition(  "eplus","e+"         ,"Electric charge",eplus);
 new ALIUnitDefinition("coulomb","C"          ,"Electric charge",coulomb); 
 
 //Energy
 new ALIUnitDefinition(    "electronvolt","eV" ,"Energy",electronvolt);
 new ALIUnitDefinition("kiloelectronvolt","keV","Energy",kiloelectronvolt);
 new ALIUnitDefinition("megaelectronvolt","MeV","Energy",megaelectronvolt);
 new ALIUnitDefinition("gigaelectronvolt","GeV","Energy",gigaelectronvolt);
 new ALIUnitDefinition("teraelectronvolt","TeV","Energy",teraelectronvolt);
 new ALIUnitDefinition("petaelectronvolt","PeV","Energy",petaelectronvolt);
 new ALIUnitDefinition(           "joule","J"  ,"Energy",joule);
 
 //Mass
 new ALIUnitDefinition("milligram","mg","Mass",milligram);
 new ALIUnitDefinition(     "gram","g" ,"Mass",gram);
 new ALIUnitDefinition( "kilogram","kg","Mass",kilogram);
 
 //Volumic Mass
 new ALIUnitDefinition( "g/cm3", "g/cm3","Volumic Mass", g/cm3);
 new ALIUnitDefinition("mg/cm3","mg/cm3","Volumic Mass",mg/cm3);
 new ALIUnitDefinition("kg/m3", "kg/m3", "Volumic Mass",kg/m3);
 
 //Power
 new ALIUnitDefinition("watt","W","Power",watt);
 
 //Force
 new ALIUnitDefinition("newton","N","Force",newton);
 
 //Pressure
 new ALIUnitDefinition(    "pascal","Pa" ,"Pressure",pascal);
 new ALIUnitDefinition(       "bar","bar","Pressure",bar); 
 new ALIUnitDefinition("atmosphere","atm","Pressure",atmosphere);
 
 //Electric current
 new ALIUnitDefinition(     "ampere","A"  ,"Electric current",ampere);
 new ALIUnitDefinition("milliampere","mA" ,"Electric current",milliampere);
 new ALIUnitDefinition("microampere","muA","Electric current",microampere);
 new ALIUnitDefinition( "nanoampere","nA" ,"Electric current",nanoampere);   
 
 //Electric potential
 new ALIUnitDefinition(    "volt","V" ,"Electric potential",volt); 
 new ALIUnitDefinition("kilovolt","kV","Electric potential",kilovolt);
 new ALIUnitDefinition("megavolt","MV","Electric potential",megavolt);
 
 //Magnetic flux
 new ALIUnitDefinition("weber","Wb","Magnetic flux",weber);
 
 //Magnetic flux density
 new ALIUnitDefinition(    "tesla","T" ,"Magnetic flux density",tesla);
 new ALIUnitDefinition("kilogauss","kG","Magnetic flux density",kilogauss);
 new ALIUnitDefinition(    "gauss","G" ,"Magnetic flux density",gauss);
 
 //Temperature
 new ALIUnitDefinition("kelvin","K","Temperature",kelvin);
 
 //Amount of substance
 new ALIUnitDefinition("mole","mol","Amount of substance",mole);
 
 //Activity
 new ALIUnitDefinition("becquerel","Bq","Activity",becquerel);
 new ALIUnitDefinition(    "curie","Ci","Activity",curie);
 
 //Dose
 new ALIUnitDefinition("gray","Gy","Dose",gray);                          
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void ALIUnitDefinition::PrintUnitsTable()
{
  std::cout << "\n          ----- The Table of Units ----- \n";
  for(size_t i=0;i<theUnitsTable.size();i++)
      theUnitsTable[i]->PrintCategory();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
   
ALIUnitsCategory::ALIUnitsCategory(ALIstring name)
:Name(name),NameMxLen(0),SymbMxLen(0)
{
    UnitsList = *(new ALIUnitsContainer);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIUnitsCategory::~ALIUnitsCategory()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIUnitsCategory::ALIUnitsCategory(ALIUnitsCategory& right)
{
    *this = right;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIUnitsCategory& ALIUnitsCategory::operator=(const ALIUnitsCategory& right)
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
 
ALIint ALIUnitsCategory::operator==(const ALIUnitsCategory& right) const
{
  return (this == &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
ALIint ALIUnitsCategory::operator!=(const ALIUnitsCategory &right) const
{
  return (this != &right);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void ALIUnitsCategory::PrintCategory()
{
  std::cout << "\n  category: " << Name << std::endl;
  for(size_t i=0;i<UnitsList.size();i++)
      UnitsList[i]->PrintDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
       
ALIBestUnit::ALIBestUnit(ALIdouble value,ALIstring category)
{
 // find the category
    ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
    size_t nbCat = theUnitsTable.size();
    size_t i = 0;
    while
     ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
    if (i == nbCat) 
       { std::cout << " ALIBestUnit: the category " << category 
                << " does not exist !!" << std::endl;
       std::cerr << "Missing unit category !" << std::endl;
       abort();
       }  
  //
    IndexOfCategory = i;
    nbOfVals = 1;
    Value[0] = value; Value[1] = 0.; Value[2] = 0.;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
       
ALIBestUnit::ALIBestUnit(const CLHEP::Hep3Vector& value,ALIstring category)
{
 // find the category
    ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
    size_t nbCat = theUnitsTable.size();
    size_t i = 0;
    while
     ((i<nbCat)&&(theUnitsTable[i]->GetName()!=category)) i++;
    if (i == nbCat) 
       { std::cerr << " ALIBestUnit: the category " << category 
                << " does not exist." << std::endl;
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
 
ALIBestUnit::~ALIBestUnit()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
std::ostream& operator<<(std::ostream& flux, ALIBestUnit a)
{
  ALIUnitsTable& theUnitsTable = ALIUnitDefinition::GetUnitsTable();
  ALIUnitsContainer& List = theUnitsTable[a.IndexOfCategory]
                           ->GetUnitsList();
  ALIint len = theUnitsTable[a.IndexOfCategory]->GetSymbMxLen();
                           
  ALIint    ksup(-1), kinf(-1);
  ALIdouble umax(0.), umin(1.E12);
  ALIdouble rsup(1.E12), rinf(0.);

  //for a ThreeVector, choose the best unit for the biggest value 
  ALIdouble value = std::max(std::max(fabs(a.Value[0]),fabs(a.Value[1])),
                              fabs(a.Value[2]));

  for (size_t k=0; k<List.size(); k++)
     {
       ALIdouble unit = List[k]->GetValue();
            if (value==1.E12) {if(unit>umax) {umax=unit; ksup=k;}}
       else if (value<=-1.E12) {if(unit<umin) {umin=unit; kinf=k;}}
       
       else { ALIdouble ratio = value/unit;
              if ((ratio>=1.)&&(ratio<rsup)) {rsup=ratio; ksup=k;}
              if ((ratio< 1.)&&(ratio>rinf)) {rinf=ratio; kinf=k;}
	    } 
     }
	 
  ALIint index=ksup; if(index==-1) index=kinf; if(index==-1) index=0;
  
  for (ALIint j=0; j<a.nbOfVals; j++) 
     {flux << a.Value[j]/(List[index]->GetValue()) << " ";}

  #ifdef ALIUSE_STD_NAMESPACE
    std::ios::fmtflags oldform = std::cout.flags();
  #else
    //    ALIint oldform = std::cout.flags();
  #endif

  flux.setf(std::ios::left,std::ios::adjustfield);
  flux << std::setw(len) << List[index]->GetSymbol();       
  //??  flux.flags(oldform);

  return flux;
}       

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
        
