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
        Definitions(const std::vector<std::string>& fVar, const std::vector<int>& fParVar, const std::string& fFormula); 
        Definitions(const std::string& fLine); 
        //-------- Member functions ----------
        unsigned nBinVar()                  const {return mBinVar.size(); }
        unsigned nParVar()                  const {return mParVar.size(); }
        int ptclType()                      const {return ptclType_; }
        std::vector<int> parVar()   const {return mParVar;        }
        std::vector<std::string> binVar()   const {return mBinVar;        } 
        int parVar(unsigned fIndex) const {return mParVar[fIndex];}
        std::string binVar(unsigned fIndex) const {return mBinVar[fIndex];} 
        std::string formula()               const {return mFormula;       }
      private:
        //-------- Member variables ----------
	int 			 ptclType_;
        std::string              mFormula;
        std::vector<std::string> mBinVar;
        std::vector<int> mParVar;

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
    METCorrectorParameters() { valid_ = false;}
    METCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
    METCorrectorParameters(const METCorrectorParameters::Definitions& fDefinitions,
			 const std::vector<METCorrectorParameters::Record>& fRecords) 
      : mDefinitions(fDefinitions),mRecords(fRecords) { valid_ = true;}
    //-------- Member functions ----------
    const Record& record(unsigned fBin)                          const {return mRecords[fBin]; }
    const Definitions& definitions()                             const {return mDefinitions;   }
    unsigned size()                                             const {return mRecords.size();}
    unsigned size(unsigned fVar)                                 const;
    int binIndex(const std::vector<float>& fX)                   const;
    int neighbourBin(unsigned fIndex, unsigned fVar, bool fNext) const;
    std::vector<float> binCenters(unsigned fVar)                 const;
    void printScreen(const std::string& Section)                 const;
    void printFile(const std::string& fFileName, const std::string& Section)const;
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
  enum Level_t { XYshiftMC=0,
    		 XYshiftDY=1,
    		 XYshiftTTJets=2,
    		 XYshiftWJets=3,
    		 XYshiftData=4,
		 N_LEVELS=5
  };

  typedef int                            key_type;
  typedef std::string                    label_type;
  typedef METCorrectorParameters         value_type;
  typedef std::pair<key_type,value_type> pair_type;
  typedef std::vector<pair_type>         collection_type;

  // Constructor... initialize all three vectors to zero
  METCorrectorParametersCollection() { correctionsXYshift_.clear();}

  // Add a METCorrectorParameter object, for each source 
  void push_back( key_type i, value_type const & j, label_type const & flav = "" );

  // Access the METCorrectorParameter via the key k.
  // key_type is hashed to deal with the three collections
  METCorrectorParameters const & operator[]( key_type k ) const;


  // Get a list of valid keys. These will contain hashed keys
  // that are aware of all three collections. 
  void validKeys(std::vector<key_type> & keys ) const;


  // Helper method to find all of the sections in a given 
  // parameters file
  static void getSections( std::string inputFile,
			   std::vector<std::string> & outputs );
  // Find the XYshift bin for hashing
  static key_type getXYshiftMcFlavBin( std::string const & Flav );
  static key_type getXYshiftDyFlavBin( std::string const & Flav );
  static key_type getXYshiftTTJetsFlavBin( std::string const & Flav );
  static key_type getXYshiftWJetsFlavBin( std::string const & Flav );
  static key_type getXYshiftDataFlavBin( std::string const & Flav );

  static bool isXYshiftMC( key_type k);
  static bool isXYshiftDY( key_type k);
  static bool isXYshiftTTJets( key_type k);
  static bool isXYshiftWJets( key_type k);
  static bool isXYshiftData( key_type k);

  static std::string findLabel( key_type k );
  static std::string levelName( key_type k );

  static std::string findXYshiftMCflavor( key_type k );
  static std::string findXYshiftDYflavor( key_type k );
  static std::string findXYshiftTTJetsFlavor( key_type k );
  static std::string findXYshiftWJetsFlavor( key_type k );
  static std::string findXYshiftDataFlavor( key_type k );

 protected:

  // Find the key corresponding to each label

  collection_type                        correctionsXYshift_;

 COND_SERIALIZABLE;
};


#endif
