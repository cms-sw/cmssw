// -*- C++ -*-

#if !defined(VariableMapCont_H)
#define VariableMapCont_H

#include <map>
#include <vector>
#include <string>

namespace optutl
{

class VariableMapCont
{
   public:
      //////////////////////
      // Public Constants //
      //////////////////////

      // typedefs
      typedef std::vector< int >                   IVec;
      typedef std::vector< double >                DVec;
      typedef std::vector< std::string >           SVec;
      typedef std::map< std::string, int >         SIMap;
      typedef std::map< std::string, double >      SDMap;
      typedef std::map< std::string, bool >        SBMap;
      typedef std::map< std::string, std::string > SSMap;
      typedef std::map< std::string, IVec >        SIVecMap;
      typedef std::map< std::string, DVec >        SDVecMap;
      typedef std::map< std::string, SVec >        SSVecMap;
      // Iterators
      typedef IVec::iterator            IVecIter;
      typedef DVec::iterator            DVecIter;
      typedef SVec::iterator            SVecIter;
      typedef SIMap::iterator           SIMapIter;
      typedef SDMap::iterator           SDMapIter;
      typedef SBMap::iterator           SBMapIter;
      typedef SSMap::iterator           SSMapIter;
      typedef SIVecMap::iterator        SIVecMapIter;
      typedef SDVecMap::iterator        SDVecMapIter;
      typedef SSVecMap::iterator        SSVecMapIter;
      // constant iterators
      typedef IVec::const_iterator      IVecConstIter;
      typedef DVec::const_iterator      DVecConstIter;
      typedef SVec::const_iterator      SVecConstIter;
      typedef SIMap::const_iterator     SIMapConstIter;
      typedef SDMap::const_iterator     SDMapConstIter;
      typedef SBMap::const_iterator     SBMapConstIter;
      typedef SSMap::const_iterator     SSMapConstIter;
      typedef SIVecMap::const_iterator  SIVecMapConstIter;
      typedef SDVecMap::const_iterator  SDVecMapConstIter;
      typedef SSVecMap::const_iterator  SSVecMapConstIter;
   

      // constants
      static const int         kDefaultInteger;
      static const double      kDefaultDouble;
      static const std::string kDefaultString;
      static const bool        kDefaultBool;
      static const IVec        kEmptyIVec;
      static const DVec        kEmptyDVec;
      static const SVec        kEmptySVec;

      enum OptionType
      {
         kNone = 0,
         kInteger,
         kDouble,
         kString,
         kBool,
         kIntegerVector,
         kDoubleVector,
         kStringVector,
         kNumOptionTypes
      };

      /////////////
      // friends //
      /////////////
      // tells particle data how to print itself out
      friend std::ostream& operator<< (std::ostream& o_stream, 
                                       const VariableMapCont &rhs);

      //////////////////////////
      //            _         //
      // |\/|      |_         //
      // |  |EMBER | UNCTIONS //
      //                      //
      //////////////////////////

      /////////////////////////////////
      // Constructors and Destructor //
      /////////////////////////////////
      VariableMapCont();

      //////////////////////////////
      // Regular Member Functions //
      //////////////////////////////

      // prints out '--help' screen, then exits.
      void help();

      // returns OptionType (or kNone (0)) of a given option.  
      OptionType hasVariable (std::string key);
      OptionType hasOption (std::string key)
      { return hasVariable (key); }


      // Add variable to option maps.  'key' is passed in by copy
      // because it is modified in place.
      void addOption (std::string key, OptionType type,
                      const std::string &description = "");
      void addOption (std::string key, OptionType type,
                      const std::string &description, 
                      int defaultValue);
      void addOption (std::string key, OptionType type,
                      const std::string &description, 
                      double defaultValue);
      void addOption (std::string key, OptionType type,
                      const std::string &description, 
                      const std::string &defaultValue);
      void addOption (std::string key, OptionType type,
                      const std::string &description, 
                      const char *defaultValue);
      void addOption (std::string key, OptionType type,
                      const std::string &description, 
                      bool defaultValue);
      //   addVariable works just like addOption, but has no description.
      void addVariable (std::string key, OptionType type)
      { addOption (key, type, ""); }
      void addVariable (std::string key, OptionType type, int defaultValue)
      { addOption (key, type, "", defaultValue); }
      void addVariable (std::string key, OptionType type, double defaultValue)
      { addOption (key, type, "", defaultValue); }
      void addVariable (std::string key, OptionType type, 
                        const std::string &defaultValue)
      { addOption (key, type, "", defaultValue); }
      void addVariable (std::string key, OptionType type, 
                        const char *defaultValue)
      { addOption (key, type, "", defaultValue); }
      void addVariable (std::string key, OptionType type, bool defaultValue)
      { addOption (key, type, "", defaultValue); }


      // some of the guts of above
      void _checkKey (std::string &key, const std::string &description = "");

      int         &integerValue  (std::string key);
      double      &doubleValue   (std::string key);
      std::string &stringValue   (std::string key);
      bool        &boolValue     (std::string key);
      IVec        &integerVector (std::string key);
      DVec        &doubleVector  (std::string key);
      SVec        &stringVector  (std::string key);

      /////////////////////////////
      // Static Member Functions //
      /////////////////////////////

      // converts a string to lower case characters
      static void lowercaseString(std::string &arg); 

      // converts a single character to lower case
      static char toLower (char &ch);


   protected:

      // returns true if a variable has been modified from the command
      // line.
      bool _valueHasBeenModified (const std::string &key);

      /////////////////////////
      // Private Member Data //
      /////////////////////////

      SIMap     m_integerMap;
      SDMap     m_doubleMap;
      SSMap     m_stringMap;
      SBMap     m_boolMap;
      SIVecMap  m_integerVecMap;
      SDVecMap  m_doubleVecMap;
      SSVecMap  m_stringVecMap;
      SBMap     m_variableModifiedMap;
      SSMap     m_variableDescriptionMap;

};

}
#endif // VariableMapCont_H
