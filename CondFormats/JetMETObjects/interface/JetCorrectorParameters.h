//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: JetCorrectorParameters.h,v 1.5 2009/11/11 13:34:12 kkousour Exp $
//
// Generic parameters for Jet corrections
//
#ifndef JetCorrectorParameters_h
#define JetCorrectorParameters_h

#include <string>
#include <vector>

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
        Definitions() {}
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
    };
     
    //-------- Constructors --------------
    JetCorrectorParameters() {}
    JetCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
    JetCorrectorParameters(const JetCorrectorParameters::Definitions& fDefinitions,
			 const std::vector<JetCorrectorParameters::Record>& fRecords) 
    : mDefinitions(fDefinitions),mRecords(fRecords) {}
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
    
  private:
    //-------- Member variables ----------
    JetCorrectorParameters::Definitions         mDefinitions;
    std::vector<JetCorrectorParameters::Record> mRecords;
};

#endif
