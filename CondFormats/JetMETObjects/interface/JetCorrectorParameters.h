//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: JetCorrectorParameters.h,v 1.15 2012/03/01 18:24:52 srappocc Exp $
//
// Generic parameters for Jet corrections
//
#ifndef JetCorrectorParameters_h
#define JetCorrectorParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class JetCorrectorParameters 
{
  //---------------- JetCorrectorParameters class ----------------
  //-- Encapsulates all the information of the parametrization ---
  public:
    //---------------- Definitions class ---------------------------
    //-- Global iformation about the parametrization is kept here --
    class Definitions 
    {
      public:
        //-------- Constructors -------------- 
        Definitions() : mIsResponse(false) {}
        Definitions(const std::vector<std::string>& fBinVar, const std::vector<std::string>& fParVar, const std::string& fFormula, bool fIsResponse); 
        Definitions(const std::string& fLine); 
        //-------- Member functions ----------
        unsigned nBinVar()                  const {return mBinVar.size(); }
        unsigned nParVar()                  const {return mParVar.size(); }
        std::vector<std::string> parVar()   const {return mParVar;        }
        std::vector<std::string> binVar()   const {return mBinVar;        } 
        std::string parVar(unsigned fIndex) const {return mParVar[fIndex];}
        std::string binVar(unsigned fIndex) const {return mBinVar[fIndex];} 
        std::string formula()               const {return mFormula;       }
        std::string level()                 const {return mLevel;         }
        bool isResponse()                   const {return mIsResponse;    }
      private:
        //-------- Member variables ----------
        bool                     mIsResponse; 
        std::string              mLevel; 
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
        Record() : mNvar(0),mMin(0),mMax(0) {}
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
    JetCorrectorParameters() { valid_ = false;}
    JetCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
    JetCorrectorParameters(const JetCorrectorParameters::Definitions& fDefinitions,
			 const std::vector<JetCorrectorParameters::Record>& fRecords) 
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
    JetCorrectorParameters::Definitions         mDefinitions;
    std::vector<JetCorrectorParameters::Record> mRecords;
    bool                                        valid_; /// is this a valid set?

  COND_SERIALIZABLE;
};



class JetCorrectorParametersCollection {
  //---------------- JetCorrectorParametersCollection class ----------------
  //-- Adds several JetCorrectorParameters together by algorithm type ---
  //--     to reduce the number of payloads in the Database ---
  //-- NB: The enum is listed in "human-logical" order, but the actual
  //       enum value is in order of appearance when people thought of them.
 public:
  enum Level_t { L1Offset=0,
		 L1JPTOffset=7,
                 L1FastJet = 10,
		 L2Relative=1,
		 L3Absolute=2,
		 L2L3Residual=8,
		 L4EMF=3,
		 L5Flavor=4,
		 L6UE=5,
		 L7Parton=6,
		 Uncertainty=9,
		 UncertaintyAbsolute=11, 
		 UncertaintyHighPtExtra=12, 
		 UncertaintySinglePionECAL=13, 
		 UncertaintySinglePionHCAL=27, 
		 UncertaintyFlavor=14, 
		 UncertaintyTime=15,
		 UncertaintyRelativeJEREC1=16, 
		 UncertaintyRelativeJEREC2=17, 
		 UncertaintyRelativeJERHF=18,
		 UncertaintyRelativePtEC1=28,
		 UncertaintyRelativePtEC2=29,
		 UncertaintyRelativePtHF=30,
		 UncertaintyRelativeStatEC2=19, 
		 UncertaintyRelativeStatHF=20, 
		 UncertaintyRelativeFSR=21,
		 UncertaintyRelativeSample=31,
		 UncertaintyPileUpDataMC=22, 
		 UncertaintyPileUpOOT=23, 
		 UncertaintyPileUpPtBB=24,
		 UncertaintyPileUpPtEC=32, 
		 UncertaintyPileUpPtHF=33, 
		 UncertaintyPileUpBias=25, 
		 UncertaintyPileUpJetRate=26,
		 L1RC=34,
		 L1Residual=35,
		 UncertaintyAux3=36,
		 UncertaintyAux4=37,
		 N_LEVELS=38
  };


  enum L5_Species_t {L5_bJ=0,L5_cJ,L5_qJ,L5_gJ,L5_bT,L5_cT,L5_qT,L5_gT,N_L5_SPECIES};
  enum L7_Species_t {L7_gJ=0,L7_qJ,L7_cJ,L7_bJ,L7_jJ,L7_qT,L7_cT,L7_bT,L7_jT,N_L7_SPECIES};
  typedef int                            key_type;
  typedef std::string                    label_type;
  typedef JetCorrectorParameters         value_type;
  typedef std::pair<key_type,value_type> pair_type;
  typedef std::vector<pair_type>         collection_type;


  // Constructor... initialize all three vectors to zero
  JetCorrectorParametersCollection() { corrections_.clear(); correctionsL5_.clear(); correctionsL7_.clear(); }

  // Add a JetCorrectorParameter object, possibly with flavor. 
  void push_back( key_type i, value_type const & j, label_type const & flav = "" );

  // Access the JetCorrectorParameter via the key k.
  // key_type is hashed to deal with the three collections
  JetCorrectorParameters const & operator[]( key_type k ) const;

  // Access the JetCorrectorParameter via a string. 
  // Will find the hashed value for the label, and call via that 
  // operator. 
  JetCorrectorParameters const & operator[]( std::string const & label ) const {
    return operator[]( findKey(label) );
  }

  // Get a list of valid keys. These will contain hashed keys
  // that are aware of all three collections. 
  void validKeys(std::vector<key_type> & keys ) const;



  // Helper method to find all of the sections in a given 
  // parameters file
  static void getSections( std::string inputFile,
			   std::vector<std::string> & outputs );

  // Find the L5 bin for hashing
  static key_type getL5Bin( std::string const & flav );
  // Find the L7 bin for hashing
  static key_type getL7Bin( std::string const & flav );
  // Check if this is an L5 hashed value
  static bool isL5( key_type k );
  // Check if this is an L7 hashed value
  static bool isL7( key_type k );

  static std::string findLabel( key_type k );
  static std::string findL5Flavor( key_type k );
  static std::string findL7Parton( key_type k );

 protected:

  // Find the key corresponding to each label
  key_type findKey( std::string const & label ) const;

  collection_type                        corrections_;
  collection_type                        correctionsL5_;
  collection_type                        correctionsL7_;

 COND_SERIALIZABLE;

};



#endif
