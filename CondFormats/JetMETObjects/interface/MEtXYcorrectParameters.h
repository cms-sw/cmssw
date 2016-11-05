//
#ifndef MEtXYcorrectParameters_h
#define MEtXYcorrectParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MEtXYcorrectParameters 
{
  //---------------- MEtParameters class ----------------
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
        int PtclType()                      const {return ptclType; }
	std::vector<unsigned> parVar()   const {return mParVar;        } // parameterized Variable
        std::vector<std::string> binVar()   const {return mBinVar;        } 
        unsigned parVar(unsigned fIndex) const {return mParVar[fIndex];}
        std::string binVar(unsigned fIndex) const {return mBinVar[fIndex];} 
        std::string formula()               const {return mFormula;       }
      private:
        //-------- Member variables ----------
	int 			ptclType;
        std::string              mFormula;
        std::vector<unsigned>	mParVar;
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
	std::string MetAxis()  const {return mMetAxis;       }
        int operator< (const Record& other) const {return xMin(0) < other.xMin(0);    }
      private:
        //-------- Member variables ----------
        unsigned           mNvar;
        std::vector<float> mMin;
        std::vector<float> mMax;
        std::vector<float> mParameters;
	std::string        mMetAxis;

      COND_SERIALIZABLE;
    };

    //-------- Constructors --------------
    MEtXYcorrectParameters() { valid_ = false;}
    MEtXYcorrectParameters(const std::string& fFile, const std::string& fSection = "");
    MEtXYcorrectParameters(const MEtXYcorrectParameters::Definitions& fDefinitions,
			 const std::vector<MEtXYcorrectParameters::Record>& fRecords) 
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
    void printScreen(const std::string& Section)                 const;
    void printFile(const std::string& fFileName)                 const;
    void printFile(const std::string& fFileName, const std::string& Section)const;
    bool isValid() const { return valid_; }

  private:
    //-------- Member variables ----------
    MEtXYcorrectParameters::Definitions         mDefinitions;
    std::vector<MEtXYcorrectParameters::Record> mRecords;
    bool                                        valid_; /// is this a valid set?

  COND_SERIALIZABLE;
};


class MEtXYcorrectParametersCollection {
 public:
  enum Level_t { shiftMC=0,
    		 shiftDY=1,
    		 shiftTTJets=2,
    		 shiftWJets=3,
    		 shiftData=4,
		 N_LEVELS=5
  };

  typedef int                            key_type;
  typedef std::string                    label_type;
  typedef MEtXYcorrectParameters         value_type;
  typedef std::pair<key_type,value_type> pair_type;
  typedef std::vector<pair_type>         collection_type;

  // Constructor... initialize all three vectors to zero
  MEtXYcorrectParametersCollection() {
    correctionsShift_.clear();
  }

  // Add a MEtXYshiftParameter object, for each source 
  void push_back( key_type i, value_type const & j, label_type const & flav = "" );

  // Access the MEtXYshiftParameter via the key k.
  // key_type is hashed to deal with the three collections
  MEtXYcorrectParameters const & operator[]( key_type k ) const;

  // Access the MEtXYshiftParameter via a string. 
  // Will find the hashed value for the label, and call via that 
  // operator. 
  MEtXYcorrectParameters const & operator[]( std::string const & label ) const {
    return operator[]( findKey(label) );
  }

  // Get a list of valid keys. These will contain hashed keys
  // that are aware of all three collections. 
  void validKeys(std::vector<key_type> & keys ) const;


  // Helper method to find all of the sections in a given 
  // parameters file
  void getSections( std::string inputFile,
			   std::vector<std::string> & outputs );

  key_type getShiftMcFlavBin( std::string const & Flav );
  key_type getShiftDyFlavBin( std::string const & Flav );
  key_type getShiftTTJetsFlavBin( std::string const & Flav );
  key_type getShiftWJetsFlavBin( std::string const & Flav );
  key_type getShiftDataFlavBin( std::string const & Flav );

  static bool isShiftMC( key_type k);
  static bool isShiftDY( key_type k);
  static bool isShiftTTJets( key_type k);
  static bool isShiftWJets( key_type k);
  static bool isShiftData( key_type k);

  static std::string findLabel( key_type k );
  static std::string levelName( key_type k );


  static std::string findShiftMCflavor( key_type k );
  static std::string findShiftDYflavor( key_type k );
  static std::string findShiftTTJetsFlavor( key_type k );
  static std::string findShiftWJetsFlavor( key_type k );
  static std::string findShiftDataFlavor( key_type k );

 protected:

  // Find the key corresponding to each label
  key_type findKey( std::string const & label ) const; // Not used

  collection_type                        correctionsShift_;

 COND_SERIALIZABLE;
};


#endif
