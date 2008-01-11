#ifndef _COMP2REFEQUAL_ROOT_H
#define _COMP2REFEQUAL_ROOT_H

#include "DQMServices/Core/interface/QualTestBase.h"

/// template class for strings, integers, floats
template<class T>
class Comp2RefEqual : public Comp2RefBase<T>
{
 public:
    
  Comp2RefEqual(void) : Comp2RefBase<T>(){}
  virtual ~Comp2RefEqual(void){}
  /// true if test cannot run
  bool isInvalid(const T * const h) const
  {    
    if(!h || ! Comp2RefBase<T>::ref_)
      return true;
    return false;
  }
  
};

#include <string>
/// algorithm for comparing equality of strings
class Comp2RefEqualStringROOT : public Comp2RefEqual<std::string>
{
 public:
  Comp2RefEqualStringROOT(void) : Comp2RefEqual<std::string>() {}
  virtual ~Comp2RefEqualStringROOT(){}
  ///get  algorithm name
  static std::string getAlgoName(void) {return "Comp2RefEqualString";}
  float runTest(const std::string * const t);

  void setReference(const std::string * const t)
  {Comp2RefEqual<std::string>::setReference(t);}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<std::string>::setMinimumEntries(N);}
  bool isInvalid(const std::string * const t)
  {return Comp2RefEqual<std::string>::isInvalid(t);}
  
};

/// algorithm for comparing equality of integers
class Comp2RefEqualIntROOT : public Comp2RefEqual<int>
{
 public:
  Comp2RefEqualIntROOT(void) : Comp2RefEqual<int>() {}
  virtual ~Comp2RefEqualIntROOT(){}
  ///get  algorithm name
  static std::string getAlgoName(void) {return "Comp2RefEqualInt";}
  float runTest(const int * const t);
  void setReference(const int * const t)
  {Comp2RefEqual<int>::setReference(t);}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<int>::setMinimumEntries(N);}
  bool isInvalid(const int * const t)
  {return Comp2RefEqual<int>::isInvalid(t);}
  
};

/// algorithm for comparing equality of floats
class Comp2RefEqualFloatROOT : public Comp2RefEqual<float>
{
 public:
  Comp2RefEqualFloatROOT(void) : Comp2RefEqual<float>() {}
  virtual ~Comp2RefEqualFloatROOT(){}
  ///get  algorithm name
  static std::string getAlgoName(void) {return "Comp2RefEqualFloat";}
  float runTest(const float * const t);
  void setReference(const float * const t)
  {Comp2RefEqual<float>::setReference(t);}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<float>::setMinimumEntries(N);}
  bool isInvalid(const float * const t)
  {return Comp2RefEqual<float>::isInvalid(t);}
  
};


#include <TH1F.h>

/// algorithm for comparing equality of 1D histograms
class Comp2RefEqualH1ROOT : public Comp2RefEqual<TH1F>
{
 public:
  Comp2RefEqualH1ROOT(void) : Comp2RefEqual<TH1F>() {}
  virtual ~Comp2RefEqualH1ROOT(){}
    /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TH1F * const h);
  ///get  algorithm name
  static std::string getAlgoName(void) {return "Comp2RefEqualH1";}
  /// true if test cannot run
  bool isInvalid(const TH1F * const h);
 
 protected:
  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
};

#include <TH2F.h>

/// algorithm for comparing equality of 2D histograms
class Comp2RefEqualH2ROOT : public Comp2RefEqual<TH2F>
{
 public:
  Comp2RefEqualH2ROOT(void) : Comp2RefEqual<TH2F>() {}
  virtual ~Comp2RefEqualH2ROOT(){}
    /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TH2F * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefEqualH2";}
  /// true if test cannot run
  bool isInvalid(const TH2F * const h);
 
 protected:
  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  Int_t ncy1;
  Int_t ncy2;
 
};

#include <TH3F.h>

/// algorithm for comparing equality of 3D histograms
class Comp2RefEqualH3ROOT : public Comp2RefEqual<TH3F>
{
 public:
  Comp2RefEqualH3ROOT(void) : Comp2RefEqual<TH3F>() {}
  virtual ~Comp2RefEqualH3ROOT(){}
    /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TH3F * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefEqualH3";}
  /// true if test cannot run
  bool isInvalid(const TH3F * const h);
 
 protected:
  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  Int_t ncy1;
  Int_t ncy2;
  Int_t ncz1;
  Int_t ncz2;
  
};

#if 0
/* Note: I need to understand how to implement the runTest here.... */
#include <TProfile.h>

/// algorithm for comparing equality of profiles
class Comp2RefEqualProfROOT : public Comp2RefEqual<TProfile>
{
 public:
  Comp2RefEqualProfROOT(void) : Comp2RefEqual<TProfile>() {}
  virtual ~Comp2RefEqualProfROOT(){}
    /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TProfile * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefEqualProf";}
  
};

#include <TProfile2D.h>

/// algorithm for comparing equality of 2D profiles
class Comp2RefEqualProf2DROOT : public Comp2RefEqual<TProfile2D>
{
 public:
  Comp2RefEqualProf2DROOT(void) : Comp2RefEqual<TProfile2D>() {}
  virtual ~Comp2RefEqualProf2DROOT(){}
    /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TProfile2D * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefEqualProf2D";}
  
};

#endif

#endif
