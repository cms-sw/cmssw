#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDSplit.h"

#include "boost/spirit/include/classic.hpp"

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>

namespace boost { namespace spirit { namespace classic { } } } using namespace boost::spirit::classic;

struct DDSelLevelCollector
{
  std::string namespace_;
  std::string name_;
  int copyNo_;
  bool isCopyNoValid_;
  bool isChild_;
  std::vector<DDPartSelRegExpLevel>* p_;

  std::vector<DDPartSelRegExpLevel>* path(std::vector<DDPartSelRegExpLevel>* p=0) {
    if (p) {
      p_=p; 
      namespace_="";
      name_="";
      copyNo_=0;
      isCopyNoValid_=false;
      isChild_=false;
    }
    return p_;
  }
};


void noNameSpace(char const * /*first*/, char const* /*last*/) {
  DDI::Singleton<DDSelLevelCollector>::instance().namespace_="";
}
/* Functor for the parser; it does not consume memory -
  pointers are only used to store references to memory
  managed elsewhere 
*/
struct DDSelLevelFtor
{
  DDSelLevelFtor() 
    : c_(DDI::Singleton<DDSelLevelCollector>::instance())
  { }
  
  // parser calls this whenever a selection has been parsed ( //ns:nm[cn], /nm, //ns:nm, .... ) 
  void operator() (char const* /*first*/, char const* /*last*/) const {
   if(c_.path()){
    if (c_.isCopyNoValid_ && c_.isChild_) {
      c_.path()->push_back(DDPartSelRegExpLevel(c_.namespace_,c_.name_,c_.copyNo_,ddchildposp));
      //edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddchildposp << std::endl;
    } else
    if (c_.isCopyNoValid_ && !c_.isChild_) {
      c_.path()->push_back(DDPartSelRegExpLevel(c_.namespace_,c_.name_,c_.copyNo_,ddanyposp));
      //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddanyposp << std::endl;
    } else
    if (!c_.isCopyNoValid_ && c_.isChild_) {
      c_.path()->push_back(DDPartSelRegExpLevel(c_.namespace_,c_.name_,c_.copyNo_,ddchildlogp));
      //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddchildlogp << std::endl;
    } else
    if (!c_.isCopyNoValid_ && !c_.isChild_) {
      c_.path()->push_back(DDPartSelRegExpLevel(c_.namespace_,c_.name_,c_.copyNo_,ddanylogp));
      //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddanylogp << std::endl;
    } 
    c_.namespace_="";
    c_.name_="";
    c_.isCopyNoValid_=false;   
   } 
  }
     
  DDSelLevelCollector & c_;    
};

struct DDIsChildFtor
{
  
  void operator()(char const* first, char const* last) const {
   DDSelLevelCollector & sl = DDI::Singleton<DDSelLevelCollector>::instance();
   if ( (last-first) > 1) 
     sl.isChild_=false;
   if ( (last-first) ==1 )
     sl.isChild_=true;
   //edm::LogInfo("DDPartSelection")  << "DDIsChildFtor  isChild=" << (last-first) << std::endl;
  }
 
};


struct DDNameSpaceFtor
{
  
  void operator()(char const* first, char const* last) const {
    DDSelLevelCollector & sl = DDI::Singleton<DDSelLevelCollector>::instance();  
    sl.namespace_.assign(first,last);
    // edm::LogInfo("DDPartSelection")  << "DDNameSpaceFtor singletonname=" << DDI::Singleton<DDSelLevelCollector>::instance().namespace_ << std::endl;
  }
  
  DDSelLevelFtor* selLevelFtor_;
};


struct DDNameFtor
{
 
  void operator()(char const* first, char const* last) const {
    DDSelLevelCollector & sl = DDI::Singleton<DDSelLevelCollector>::instance();
    sl.name_.assign(first,last);  
    // edm::LogInfo("DDPartSelection")  << "DDNameFtor singletonname=" << Singleton<DDSelLevelCollector>::instance().name_ << std::endl;
  }
  
};


struct DDCopyNoFtor
{
  
  void operator()(int i) const {
    DDSelLevelCollector & sl = DDI::Singleton<DDSelLevelCollector>::instance();  
    sl.copyNo_ = i;
    sl.isCopyNoValid_ = true;
    // edm::LogInfo("DDPartSelection")  << "DDCopyNoFtor ns=" << i;
  }
 
};
 
 

/** A boost::spirit parser for the <SpecPar path="xxx"> syntax */
struct SpecParParser : public grammar<SpecParParser>
{
  template <typename ScannerT>
  struct definition
  {
    definition(SpecParParser const& /*self*/) {
         
        Selection  //= FirstStep[selLevelFtor()] 
                  //>> *SelectionStep[selLevelFtor()]
		   = +SelectionStep[selLevelFtor()]
                   ;

        FirstStep  = Descendant 
                  >> Part
                   ; 

        Part       = PartNameCopyNumber 
                   | PartName
                   ;

        PartNameCopyNumber = PartName 
                  >> CopyNumber
                   ;

        SelectionStep = NavigationalElement[isChildFtor()] 
                  >> Part
                   ;

        NavigationalElement = Descendant 
                   | Child
                   ;

        CopyNumber = ch_p('[') 
                  >> int_p[copyNoFtor()] 
                  >> ch_p(']')
                   ;

        PartName   = NameSpaceName 
                   | SimpleName[nameFtor()][&noNameSpace]
                   ;

	SimpleName = +( alnum_p | ch_p('_') | ch_p('.') | ch_p('*') )
                   ;

        NameSpaceName = SimpleName[nameSpaceFtor()] 
                  >> ':' 
                  >> SimpleName[nameFtor()]
                   ;

        Descendant = ch_p('/') 
                  >> ch_p('/')
                   ;

        Child      = ch_p('/')
	            ;
  
         }
  
    rule<ScannerT> Selection, FirstStep, Part, SelectionStep, NavigationalElement,
        CopyNumber, PartName, PartNameCopyNumber, NameSpaceName, SimpleName, 
        Descendant, Child; 

    rule<ScannerT> const& start() const { return Selection; }
    
    DDSelLevelFtor & selLevelFtor() {
      return DDI::Singleton<DDSelLevelFtor>::instance();
    }
    
    DDNameFtor & nameFtor() {
     static DDNameFtor f_;
     return f_;
    }
    
    DDNameSpaceFtor & nameSpaceFtor() {
     static DDNameSpaceFtor f_;
     return f_;
    }
    
    DDIsChildFtor & isChildFtor() {
     static DDIsChildFtor f_;
     return f_;
    }

    DDCopyNoFtor & copyNoFtor() {
     static DDCopyNoFtor f_;
     return f_;
    }    
  };
  
};


/*
std::ostream & operator<<(std::ostream & os, const DDPartSelection & ps)
{
  DDPartSelection::const_iterator it = ps.begin();
  for (; it != ps.end(); ++it) {
    std::string s;
    switch (it->selectionType_) {
      case ddunknown: case ddanynode: case ddanychild:
        os << "*ERROR*";
	break;
      case ddanylogp: 
        os << "//" << it->lp_.ddname();
	break;
      case ddanyposp:
        os << "//" << it->lp_.ddname() << '[' << it->copyno_ << ']';
	break;
      case ddchildlogp:			
        os << "/" << it->lp_.ddname();
        break;
      case ddchildposp:
        os << "/" << it->lp_.ddname() << '[' << it->copyno_ << ']'; 		
	break;
    }
  
  }
  return os;
}
*/

DDPartSelectionLevel::DDPartSelectionLevel(const DDLogicalPart & lp, int c, ddselection_type t)
 : lp_(lp), copyno_(c), selectionType_(t)
{
  
}



void DDTokenize2(const std::string & sel, std::vector<DDPartSelRegExpLevel> & path)
{
  static SpecParParser parser;
  DDI::Singleton<DDSelLevelCollector>::instance().path(&path);
  bool result = parse(sel.c_str(), parser).full;
  if (!result) {
    edm::LogError("DDPartSelection") << "DDTokenize2() error in parsing of " << sel << std::endl;
  }
}

// uhhhhhhhhhhhhhhhh! Spaghetti code!!!!!!!!! (or worse?)
// FIXME: DDTokenize: if a LogicalPart is not yet defined during parsing of a SpecPar 
// FIXME: (ddunknown is then the corresponding ddselection_type of the PartSelection)
// FIXME: then set a state-variable to 'undefined' .
// FIXME: After parsing, reprocess all undefined SpecPars ...
void DDTokenize(const std::string & sel, std::vector<DDPartSelRegExpLevel> &  path)
{


  static bool isInit(false);
  static std::vector<std::string> tokens;
  if(!isInit) {
    // initialize with valid tokens
    tokens.push_back("/"); 
    tokens.push_back("//");
    tokens.push_back("[");
    tokens.push_back("]");
  }
  std::string s = sel;
  std::string::size_type st = std::string::npos;
  std::string::size_type cu = 0;
  std::vector<std::string> toksVec;
  std::vector<std::string> textVec;
  std::string tkn, txt;
  bool braceOpen(false);
  /*
    the following code should decompose a selection std::string into 2 std::vectors,
    a token-std::vector and a text-std::vector. Tokens are /,//,[,] 
    example: "//Abc[3]/Def" ->
             tokens  text
	     //      ""
	     [       Abc
	     ]       3
	     /       ""
	             "Def"
  */  
  
  while (s.size()) {
    std::vector<std::string>::iterator tkit = tokens.begin();
    std::vector<std::string>::iterator mint = tokens.end();
    st=s.size();
    std::string::size_type ts;
    for(;tkit!=tokens.end();++tkit) { // find the first match of one of the token std::strings
      ts = s.find(*tkit); 
      if (ts<=st) { 
	st=ts;
	mint = tkit;
      }  
    }
    
    if (mint!=tokens.end())	
     tkn = s.substr(st,mint->size());
    else
     tkn=""; 
    txt = s.substr(cu,st);
    toksVec.push_back(tkn);
    textVec.push_back(txt);
    if (braceOpen) {
      if (tkn!="]")
        throw DDException(std::string("PartSelector: syntaxerror in ") + sel + 
	                  std::string("\ncheck the braces!") );
      else
        braceOpen=false;
    }		
    if (tkn=="[")
      braceOpen=true;
    DCOUT_V('C',"tkn=" << tkn << " txt=" << txt);   
    if (mint!=tokens.end())
      s.erase(cu,st+mint->size()); 
    else
      s.erase();  
    DCOUT_V('C', std::endl << "s=" << s);   
    //break;	
  }
  DCOUT_V('C', "The original std::string was:" << std::endl);
  unsigned int i=0;
  DCOUT_V('C', "toks\ttext");
  for (i=0; i<toksVec.size();++i) {
    DCOUT_V('C',  toksVec[i] << "\t" << textVec[i]);
  }  
  DCOUT_V('C', std::endl); 
  
  // now some spaghetti code 
  std::string nm = "blabla" ; // std::string("PartSelector=") + ns() + std::string(":") + name();
  if (textVec[0] != "")
    throw DDException( nm 
                      +std::string(" selection must not start with a LogicalPart-name")); 
  
  if ((toksVec[0] != "//"))
    throw DDException( nm 
                      +std::string(" selection must start with '//' !")); 
  
  if (textVec.size() < 2)
    throw DDException( nm + std::string(" internal error [textVec.size()<2]!"));
    
  std::vector<std::string>::iterator tk_it = toksVec.begin();
  std::vector<std::string>::iterator tx_it = textVec.begin(); ++tx_it;
  // the BIG switch - yes, this is OO!!!
  while(tk_it != toksVec.end() && tx_it != textVec.end()) {
 
    // anynode ... token //* makes no sense (except as last entry which is forbidden ...)
    DCOUT_V('C', ">- anynode tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "//" && *tx_it=="*" ) { 
      path.push_back(DDPartSelRegExpLevel("","",0,ddanynode));
      DCOUT_V('C', "--anynode: //*" << std::endl);
      ++tk_it;
      ++tx_it;
      continue;
    }
             
    // anychild
    DCOUT_V('C', ">- anychild tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "/" && *tx_it=="*" ) { 
      path.push_back(DDPartSelRegExpLevel("","",0,ddanychild));
      DCOUT_V('C', "--anychild: /*" << std::endl);
      ++tk_it;
      ++tx_it;
      continue;
    }    

    // anylogp
    DCOUT_V('C', ">- anylogp tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "//" && tx_it->size()) {
      ++tk_it;
      if ( tk_it != toksVec.end() && *tk_it != "[" && *tk_it != "]") {
	std::pair<std::string,std::string> p(DDSplit(*tx_it));
          path.push_back(DDPartSelRegExpLevel(p.second,p.first,0,ddanylogp));
	DCOUT_V('C', "--anylogp: " << *tx_it << std::endl);
        ++tx_it;
        continue;
      }	
      --tk_it;
    }    

    // childlogp
    DCOUT_V('C', ">- childlogp tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "/" && tx_it->size()) {
      ++tk_it;
      if ( tk_it == toksVec.end() - 1 ) {
        DCOUT_V('C', "--childlogp: " << *tx_it << std::endl);
	std::pair<std::string,std::string> p(DDSplit(*tx_it));
          path.push_back(DDPartSelRegExpLevel(p.second,p.first,0,ddchildlogp));
	++tx_it;   
	continue;
      }
      if ( *tk_it == "/" || *tk_it=="//") {
        DCOUT_V('C', "--childlogp: " << *tx_it << std::endl);
	std::pair<std::string,std::string> p(DDSplit(*tx_it));
          path.push_back(DDPartSelRegExpLevel(p.second,p.first,0,ddchildlogp));
	++tx_it;
	continue;
      }
      --tk_it;
    }  	


    // anyposp
    DCOUT_V('C', ">- anyposp tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "//" && tx_it->size()) {
      ++tk_it;
      if ( tk_it != toksVec.end() && *tk_it == "[" ) {
        ++tk_it;
	if ( tk_it == toksVec.end() || (tk_it != toksVec.end() && *tk_it != "]")) {
	 DCOUT_V('C', *tk_it << " " << *tx_it );
	   break;
	}  
        ++tx_it;
	++tk_it;
	std::pair<std::string,std::string> p(DDSplit(*(tx_it-1)));
          path.push_back(DDPartSelRegExpLevel(p.second,p.first,atoi(tx_it->c_str()),ddanyposp));
	DCOUT_V('C', "--anyposp: " << *tx_it << " " << atoi(tx_it->c_str()) << std::endl);
        ++tx_it;
	++tx_it;
        continue;
      }	
    }    
        
         
    // childposp
    DCOUT_V('C', ">- childposp tkn=" << *tk_it << " d=" << tk_it-toksVec.begin() << " txt=" << *tx_it << std::endl);
    if ( *tk_it == "/" && tx_it->size()) {
      ++tk_it;
      if ( tk_it != toksVec.end() && *tk_it=="[" ) {
        DCOUT_V('C', "--childposp: " << *tx_it << " " << *tk_it << *(tx_it+1) << std::endl);
	std::pair<std::string,std::string> p(DDSplit(*tx_it));
          path.push_back(DDPartSelRegExpLevel(p.second,p.first,atoi((tx_it+1)->c_str()),ddchildposp));

	++tx_it;

	++tx_it;
	++tk_it;
	if (tk_it != toksVec.end() && *tk_it != "]") 
	  break;
	++tk_it;
	++tx_it;
	continue;  
      }
    }
    
    // any
    throw DDException( nm + std::string(" syntax error in:\n") + sel + 
                            std::string("\n  tkn=") + *tk_it + std::string("  txt=")+ *tx_it);		      	    	    
  }
  //FIXME: DDPartSelectorImpl::tokenize : prototype has restricted support for selection std::string (code below restricts)
  ddselection_type tmp = path.back().selectionType_;
  if (tmp==ddunknown || tmp==ddanynode || tmp==ddanychild ) 
        throw DDException(std::string("PartSelector: last element in selection std::string in ") + sel + 
	                  std::string("\nmust address a distinct LogicalPart or PosPart!") );
}


std::ostream & operator<<(std::ostream & o, const DDPartSelection & p)
{
  DDPartSelection::const_iterator it(p.begin()), ed(p.end());
  for (; it != ed; ++it) {
    const DDPartSelectionLevel lv  =*it;
    switch (lv.selectionType_) {
    case ddanylogp:
      o << "//" << lv.lp_.ddname();
      break;
    case ddanyposp:
      o << "//" << lv.lp_.ddname() << '[' << lv.copyno_ << ']';
      break;
    case ddchildlogp:
      o << "/" << lv.lp_.ddname();
      break;
    case ddchildposp:
      o << "/" << lv.lp_.ddname() << '[' << lv.copyno_ << ']';
      break;
    default:
      o << "{Syntax ERROR}";
    }
  }
  return o;
}

std::ostream & operator<<(std::ostream & os, const std::vector<DDPartSelection> & v)
{
  std::vector<DDPartSelection>::const_iterator it(v.begin()), ed(v.end());
  for (; it != (ed-1); ++it) {
    os << *it << std::endl;
  }
  if ( it != ed ) {
    ++it;
    os << *it;
  }
  return os;
}

// explicit template instantiation.

template class DDI::Singleton<DDSelLevelFtor>;
//template class DDI::Singleton<DDI::Store<DDName, DDSelLevelCollector> >;
template class DDI::Singleton<DDSelLevelCollector>;
#include <DetectorDescription/Base/interface/Singleton.icc>
