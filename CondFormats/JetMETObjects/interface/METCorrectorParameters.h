//
#ifndef METCorrectorParameters_h
#define METCorrectorParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class METCorrectorParameters 
{
  //---------------- METCorrectorParameters class ----------------
  //-- Encapsulates all the information of the parametrization ---
  public:
    //---------------- Definitions class ---------------------------
    //-- Global iformation about the parametrization is kept here --
    class Definitions 
    {
      public:
        //-------- Constructors -------------- 
        Definitions() {}
        Definitions(const std::vector<std::string>& fVar, const std::vector<std::string>& fParVar, const std::string& fFormula); 
        Definitions(const std::string& fLine); 
        //-------- Member functions ----------
        unsigned nBinVar()                  const {return mBinVar.size(); }
        unsigned nParVar()                  const {return mParVar.size(); }
        std::vector<std::string> parVar()   const {return mParVar;        }
        std::vector<std::string> binVar()   const {return mBinVar;        } 
        std::string parVar(unsigned fIndex) const {return mParVar[fIndex];}
        std::string binVar(unsigned fIndex) const {return mBinVar[fIndex];} 
        std::string formula()               const {return mFormula;       }
      private:
        //-------- Member variables ----------
	int 			ptclType;
        std::string              mFormula;
        std::vector<std::string> mParVar;
        std::vector<std::string> mBinVar;

      COND_SERIALIZABLE;
    };
    //---------------- Record class --------------------------------
    //-- Each Record holds the properties of a bin ----------------- 
    class Record 
    {
      public:
        //-------- Constructors --------------
        Record() : mNvar(0),mMin(0),mMax(0), mParameters(0) {}
        Record(unsigned fNvar, const std::vector<float>& fXMin, const std::vector<float>& fXMax, const std::vector<float>& fParameters) : mNvar(fNvar),mMin(fXMin),mMax(fXMax),mParameters(fParameters) {}
        Record(const std::string& fLine, unsigned fNvar);
        //-------- Member functions ----------
        float xMin(unsigned fVar)           const {return mMin[fVar];                 }
        float xMax(unsigned fVar)           const {return mMax[fVar];                 }
        float xMiddle(unsigned fVar)        const {return 0.5*(xMin(fVar)+xMax(fVar));}
        float parameter(unsigned fIndex)    const {return mParameters[fIndex];        }
        std::vector<float> parameters()     const {return mParameters;                }
        unsigned nParameters()              const {return mParameters.size();         }
        int operator< (const Record& other) const {return xMin(0) < other.xMin(0);    }
      private:
        //-------- Member variables ----------
        unsigned           mNvar;
        std::vector<float> mMin;
        std::vector<float> mMax;
        std::vector<float> mParameters;

      COND_SERIALIZABLE;
    };

    //-------- Constructors --------------
    METCorrectorParameters() { valid_ = false;}
    METCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
    METCorrectorParameters(const METCorrectorParameters::Definitions& fDefinitions,
			 const std::vector<METCorrectorParameters::Record>& fRecords) 
      : mDefinitions(fDefinitions),mRecords(fRecords) { valid_ = true;}
    //-------- Member functions ----------
    const Record& record(unsigned fBin)                          const {return mRecords[fBin]; }
    const Definitions& definitions()                             const {return mDefinitions;   }
    unsigned size()                                              const {return mRecords.size();}
    unsigned size(unsigned fVar)                                 const;
    int binIndex(const std::vector<float>& fX)                   const;
    int neighbourBin(unsigned fIndex, unsigned fVar, bool fNext) const;
    std::vector<float> binCenters(unsigned fVar)                 const;
    void printScreen()                                           const;
    void printFile(const std::string& fFileName)                 const;
    bool isValid() const { return valid_; }

  private:
    //-------- Member variables ----------
    METCorrectorParameters::Definitions         mDefinitions;
    std::vector<METCorrectorParameters::Record> mRecords;
    bool                                        valid_; /// is this a valid set?

  COND_SERIALIZABLE;
};


class METCorrectorParametersCollection {
 public:
  enum Level_t { MiniAod=0,
		 N_LEVELS=1
  };

  typedef int                            key_type;
  typedef std::string                    label_type;
  typedef METCorrectorParameters         value_type;
  typedef std::pair<key_type,value_type> pair_type;
  typedef std::vector<pair_type>         collection_type;

  // Constructor... initialize all three vectors to zero
  METCorrectorParametersCollection() { correctionsMiniAod_.clear();}

  // Add a METCorrectorParameter object, for each source 
  void push_back( key_type i, value_type const & j, label_type const & source = "" );

  // Access the METCorrectorParameter via the key k.
  // key_type is hashed to deal with the three collections
  METCorrectorParameters const & operator[]( key_type k ) const;

  // Access the METCorrectorParameter via a string. 
  // Will find the hashed value for the label, and call via that 
  // operator. 
  METCorrectorParameters const & operator[]( std::string const & label ) const {
    return operator[]( findKey(label) );
  }

  // Get a list of valid keys. These will contain hashed keys
  // that are aware of all three collections. 
  void validKeys(std::vector<key_type> & keys ) const;


  // Helper method to find all of the sections in a given 
  // parameters file
  static void getSections( std::string inputFile,
			   std::vector<std::string> & outputs );
  // Find the MiniAod bin for hashing
  static key_type getMiniAodBin( std::string const & source );

  static bool isMiniAod( key_type k);

  static std::string findLabel( key_type k );
  static std::string findMiniAodSource( key_type k );

 protected:

  // Find the key corresponding to each label
  key_type findKey( std::string const & label ) const;

  collection_type                        correctionsMiniAod_;

 COND_SERIALIZABLE;
};


#endif
