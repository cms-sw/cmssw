#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

#include "DQMServices/Core/interface/Comp2RefChi2ROOT.h"
#include "DQMServices/Core/interface/Comp2RefKolmogorovROOT.h"
#include "DQMServices/Core/interface/ContentsWithinRangeROOT.h"
#include "DQMServices/Core/interface/Comp2RefEqualROOT.h"
#include "DQMServices/Core/interface/MeanWithinExpectedROOT.h"
#include "DQMServices/Core/interface/MostProbableROOT.h"

#include <sstream>
#include <string>

#ifndef _QCRITERION_ROOT_H
#define _QCRITERION_ROOT_H

template<class T>
class QCriterionRoot : public QCriterion
{
 public: 
  QCriterionRoot(std::string name) : QCriterion(name){}
  virtual ~QCriterionRoot(void){}
  /// run the test on MonitorElement <me> (result: [0, 1] or <0 for failure)
  float runTest(const MonitorElement * const me)
  {
    if(!check(me))return -1;
    prob_ = runTest(getObject(me));
    setStatusMessage();
    return prob_;
  }
  ///get  algorithm name
  virtual std::string getAlgoName(void) = 0;

  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  virtual std::vector<dqm::me_util::Channel> getBadChannels(void) const
  /// return empty vector
  {return std::vector<dqm::me_util::Channel>();}

 protected:
  virtual const T * getObject(const MonitorElement * const me) const = 0;

  /// get (ROOT) object from MonitorElement <me>
  /// (will redefine for scalars)
  const T * getROOTObject(const MonitorElement * const me) const
  {
    const T * ret = 0;
    const MonitorElementT<TNamed>* obj = 
      dynamic_cast<const MonitorElementT<TNamed>*> (me);
    if(obj)
      ret = dynamic_cast<const T *> (obj->const_ptr());      
    return ret;
  }
  /// 
  virtual float runTest(const T * const t) = 0;
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  virtual bool isInvalid(const MonitorElement * const me) = 0;
  /// true if object <t> does not have enough statistics
  virtual bool notEnoughStats(const T * const t) const = 0;
  /// true if MonitorElement <me> does not have enough statistics
  bool notEnoughStats(const MonitorElement * const me) const
  { return notEnoughStats(getObject(me)); }
  /// set message after test has run
  virtual void setMessage(void) = 0;

  /// set status & message after test has run
  void setStatusMessage(void)
  {
    if(!validProb(prob_))
      setInvalid();
    else if(prob_ < errorProb_)
      setError();
    else if(prob_ < warningProb_)
      setWarning();
    else
      setOk();
  }
  
};
  
class MEComp2RefChi2ROOT: public QCriterionRoot<TH1F>,public Comp2RefChi2ROOT
{
 public:
  MEComp2RefChi2ROOT(std::string name) : QCriterionRoot<TH1F>(name),
    Comp2RefChi2ROOT() {setAlgoName(Comp2RefChi2ROOT::getAlgoName());}
  ~MEComp2RefChi2ROOT(void){}
  
  std::string getAlgoName(void){return Comp2RefChi2ROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return Comp2RefChi2ROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return (const TH1F *) QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return Comp2RefChi2ROOT::notEnoughStats(t);}
  
  void setReference(const MonitorElement * const me)
  {Comp2RefChi2ROOT::setReference(getObject(me)); update();}
  ///
  void setMinimumEntries(unsigned N)
  {Comp2RefChi2ROOT::setMinimumEntries(N); update();}

 protected:

  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefChi2ROOT::isInvalid(getObject(me));}
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): chi2/Ndof = " << chi2_ << "/" << Ndof_ 
	    << " prob = " << prob_
            << ", minimum needed statistics = " << min_entries_ 
            << " warning threshold = " << QCriterionRoot<TH1F>::warningProb_
            << " error threshold = " << QCriterionRoot<TH1F>::errorProb_;
	    ;
    message_ = message.str();      
  }

};

class MEComp2RefKolmogorovROOT: public QCriterionRoot<TH1F>, 
  public Comp2RefKolmogorovROOT
{
 public:
  MEComp2RefKolmogorovROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), Comp2RefKolmogorovROOT() 
  {setAlgoName(Comp2RefKolmogorovROOT::getAlgoName());}
  ~MEComp2RefKolmogorovROOT(void){}
  
  std::string getAlgoName(void)
  {return Comp2RefKolmogorovROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return Comp2RefKolmogorovROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return Comp2RefKolmogorovROOT::notEnoughStats(t);}

  void setReference(const MonitorElement * const me)
  {Comp2RefKolmogorovROOT::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefKolmogorovROOT::setMinimumEntries(N); update();}
 

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefKolmogorovROOT::isInvalid(getObject(me));}
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): prob = " << prob_;
    message_ = message.str();      
  }


};

class MEContentsXRangeROOT: public QCriterionRoot<TH1F>, 
  public ContentsXRangeROOT
{
 public:
  MEContentsXRangeROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), ContentsXRangeROOT() 
  {setAlgoName(ContentsXRangeROOT::getAlgoName());}
  ~MEContentsXRangeROOT(void){}
  
  std::string getAlgoName(void)
  {return ContentsXRangeROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return ContentsXRangeROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return ContentsXRangeROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {ContentsXRangeROOT::setMinimumEntries(N); update();}
 
 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Entry fraction within X range = " << prob_;
    message_ = message.str();      
  }

};

class MEContentsYRangeROOT: public QCriterionRoot<TH1F>, 
  public ContentsYRangeROOT
{
 public:
  MEContentsYRangeROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), ContentsYRangeROOT() 
  {setAlgoName(ContentsYRangeROOT::getAlgoName());}
  ~MEContentsYRangeROOT(void){}
  
  std::string getAlgoName(void)
  {return ContentsYRangeROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return ContentsYRangeROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return ContentsYRangeROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {ContentsYRangeROOT::setMinimumEntries(N); update();}

  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return ContentsYRangeROOT::getBadChannels();}
  
 protected:

  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Bin fraction within Y range = " << prob_;
    message_ = message.str();      
  }
};

///
template<class T>
class MEComp2RefEqualTROOT: public QCriterionRoot<T>
{
 public:
  MEComp2RefEqualTROOT(std::string name) : 
  QCriterionRoot<T>(name){}
  ~MEComp2RefEqualTROOT(void){}
  
  virtual std::string getAlgoName(void) = 0;

  virtual float runTest(const T * const t) = 0;

 protected:
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << QCriterionRoot<T>::qtname_ 
	    << " (" << getAlgoName() << "): Identical contents? ";
    if(QCriterionRoot<T>::prob_) 
      message << "Yes";
    else
      message << "No";
    QCriterionRoot<T>::message_ = message.str();      
  }

};

class MEComp2RefEqualStringROOT: public MEComp2RefEqualTROOT<std::string>,
  public Comp2RefEqualStringROOT
{
 public:
  MEComp2RefEqualStringROOT(std::string name): 
  MEComp2RefEqualTROOT<std::string>(name), Comp2RefEqualStringROOT()
  {setAlgoName(Comp2RefEqualStringROOT::getAlgoName());}
  ~MEComp2RefEqualStringROOT(){}
  ///
  std::string getAlgoName(void)
  {return Comp2RefEqualStringROOT::getAlgoName();}
  ///  
  float runTest(const std::string * const t)
  {return Comp2RefEqualStringROOT::runTest(t);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqualStringROOT::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqualStringROOT::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualStringROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const std::string * const t) const
  {return false;} // statistics not an issue for strings

 protected:
  const std::string * getObject(const MonitorElement * const me) const
  {
    const std::string * ret = 0;
    const MonitorElementT<std::string>* obj = 
      dynamic_cast<const MonitorElementT<std::string> * > (me);
    if(obj) ret = dynamic_cast<const std::string *> (obj->const_ptr());
    return ret;
  }

};

class MEComp2RefEqualIntROOT: public MEComp2RefEqualTROOT<int>,
  public Comp2RefEqualIntROOT
{
 public:
  MEComp2RefEqualIntROOT(std::string name): 
  MEComp2RefEqualTROOT<int>(name), Comp2RefEqualIntROOT()
  {setAlgoName(Comp2RefEqualIntROOT::getAlgoName());}
  ~MEComp2RefEqualIntROOT(){}
  ///
  std::string getAlgoName(void)
  {return Comp2RefEqualIntROOT::getAlgoName();}
  ///  
  float runTest(const int * const t)
  {return Comp2RefEqualIntROOT::runTest(t);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqualIntROOT::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqualIntROOT::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualIntROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const int * const t) const
  {return false;} // statistics not an issue for ints

 protected:
  const int * getObject(const MonitorElement * const me) const
  {
    const int * ret = 0;
    const MonitorElementT<int>* obj = 
      dynamic_cast<const MonitorElementT<int> * > (me);
    if(obj) ret = obj->const_ptr();
    return ret;
  }

};

class MEComp2RefEqualFloatROOT: public MEComp2RefEqualTROOT<float>,
  public Comp2RefEqualFloatROOT
{
 public:
  MEComp2RefEqualFloatROOT(std::string name): 
  MEComp2RefEqualTROOT<float>(name), Comp2RefEqualFloatROOT()
  {setAlgoName(Comp2RefEqualFloatROOT::getAlgoName());}
  ~MEComp2RefEqualFloatROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualFloatROOT::getAlgoName();}
  //  
  float runTest(const float * const t)
  {return Comp2RefEqualFloatROOT::runTest(t);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqualFloatROOT::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqualFloatROOT::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualFloatROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const float * const t) const
  {return false;} // statistics not an issue for floats

 protected:
  const float * getObject(const MonitorElement * const me) const
  {
    const float * ret = 0;
    const MonitorElementT<float>* obj = 
      dynamic_cast<const MonitorElementT<float> * > (me);
    if(obj) ret = obj->const_ptr();
    return ret;
  }

};

class MEComp2RefEqualH1ROOT: public MEComp2RefEqualTROOT<TH1F>,
  public Comp2RefEqualH1ROOT
{
 public:
  MEComp2RefEqualH1ROOT(std::string name): 
  MEComp2RefEqualTROOT<TH1F>(name), Comp2RefEqualH1ROOT()
  {setAlgoName(Comp2RefEqualH1ROOT::getAlgoName());}
  ~MEComp2RefEqualH1ROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualH1ROOT::getAlgoName();}
  //  
  float runTest(const TH1F * const t)
  {return Comp2RefEqualH1ROOT::runTest(t);}
  //
  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqual<TH1F>::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<TH1F>::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualH1ROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const TH1F * const t) const
  {return Comp2RefEqual<TH1F>::notEnoughStats(t);}
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return Comp2RefEqual<TH1F>::getBadChannels();}

};

class MEComp2RefEqualH2ROOT: public MEComp2RefEqualTROOT<TH2F>,
  public Comp2RefEqualH2ROOT
{
 public:
  MEComp2RefEqualH2ROOT(std::string name): 
  MEComp2RefEqualTROOT<TH2F>(name), Comp2RefEqualH2ROOT()
  {setAlgoName(Comp2RefEqualH2ROOT::getAlgoName());}
  ~MEComp2RefEqualH2ROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualH2ROOT::getAlgoName();}
  //  
  float runTest(const TH2F * const t)
  {return Comp2RefEqualH2ROOT::runTest(t);}
  //
  const TH2F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH2F>::getROOTObject(me);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqual<TH2F>::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<TH2F>::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualH2ROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const TH2F * const t) const
  {return Comp2RefEqual<TH2F>::notEnoughStats(t);}
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return Comp2RefEqual<TH2F>::getBadChannels();}
};

class MEComp2RefEqualH3ROOT: public MEComp2RefEqualTROOT<TH3F>,
  public Comp2RefEqualH3ROOT
{
 public:
  MEComp2RefEqualH3ROOT(std::string name): 
  MEComp2RefEqualTROOT<TH3F>(name), Comp2RefEqualH3ROOT()
  {setAlgoName(Comp2RefEqualH3ROOT::getAlgoName());}
  ~MEComp2RefEqualH3ROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualH3ROOT::getAlgoName();}
  //  
  float runTest(const TH3F * const t)
  {return Comp2RefEqualH3ROOT::runTest(t);}
  const TH3F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH3F>::getROOTObject(me);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqual<TH3F>::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<TH3F>::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  {return Comp2RefEqualH3ROOT::isInvalid(getObject(me));}
  bool notEnoughStats(const TH3F * const t) const
  {return Comp2RefEqual<TH3F>::notEnoughStats(t);}
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return Comp2RefEqual<TH3F>::getBadChannels();}
};

#if 0
class MEComp2RefEqualProfROOT: public MEComp2RefEqualTROOT<TProfile>,
  public Comp2RefEqualProfROOT
{
 public:
  MEComp2RefEqualProfROOT(std::string name): 
  MEComp2RefEqualTROOT<TProfile>(name), Comp2RefEqualProfROOT()
  {setAlgoName(Comp2RefEqualProfROOT::getAlgoName());}
  ~MEComp2RefEqualProfROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualProfROOT::getAlgoName();}
  //  
  float runTest(const TProfile * const t)
  {return Comp2RefEqualProfROOT::runTest(t);}
  const TProfile * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TProfile>::getROOTObject(me);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqual<TProfile>::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<TProfile>::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  //  {return Comp2RefEqual<TProfile>::hasNullReference();}
  {return Comp2RefEqual<TProfile>::isInvalid(getObject(me));}
  bool notEnoughStats(const TProfile * const t) const
  {return Comp2RefEqual<TProfile>::notEnoughStats(t);}
};

class MEComp2RefEqualProf2DROOT: public MEComp2RefEqualTROOT<TProfile2D>,
  public Comp2RefEqualProf2DROOT
{
 public:
  MEComp2RefEqualProf2DROOT(std::string name): 
  MEComp2RefEqualTROOT<TProfile2D>(name), Comp2RefEqualProf2DROOT()
  {setAlgoName(Comp2RefEqualProf2DROOT::getAlgoName());}
  ~MEComp2RefEqualProf2DROOT(){}
  //
  std::string getAlgoName(void)
  {return Comp2RefEqualProf2DROOT::getAlgoName();}
  //  
  float runTest(const TProfile2D * const t)
  {return Comp2RefEqualProf2DROOT::runTest(t);}
  const TProfile2D * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TProfile2D>::getROOTObject(me);}
  void setReference(const MonitorElement * const me)
  {Comp2RefEqual<TProfile2D>::setReference(getObject(me)); update();}
  void setMinimumEntries(unsigned N)
  {Comp2RefEqual<TProfile2D>::setMinimumEntries(N); update();}
  bool isInvalid(const MonitorElement * const me)
  //  {return Comp2RefEqual<TProfile2D>::hasNullReference();}
  {return Comp2RefEqual<TProfile2D>::isInvalid(getObject(me));}
  bool notEnoughStats(const TProfile2D * const t) const
  {return Comp2RefEqual<TProfile2D>::notEnoughStats(t);}
};

#endif

class MEContentsTH2FWithinRangeROOT : public QCriterionRoot<TH2F>,
  public ContentsTH2FWithinRangeROOT
{
 public:
  MEContentsTH2FWithinRangeROOT(std::string name) : 
  QCriterionRoot<TH2F>(name), ContentsTH2FWithinRangeROOT()
  {setAlgoName(ContentsTH2FWithinRangeROOT::getAlgoName());}
  ~MEContentsTH2FWithinRangeROOT(void){}

  std::string getAlgoName(void)
  {return ContentsTH2FWithinRangeROOT::getAlgoName();}

  float runTest(const TH2F * const t)
  {return ContentsTH2FWithinRangeROOT::runTest(t);} 

  const TH2F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH2F>::getROOTObject(me);}

  bool notEnoughStats(const TH2F * const t) const
  {return ContentsTH2FWithinRangeROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {ContentsTH2FWithinRangeROOT::setMinimumEntries(N); update();}

  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return ContentsTH2FWithinRangeROOT::getBadChannels();}
 
 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName()
            << "): Entry fraction within range = " << prob_;
    message_ = message.str();
  }

};

class MEContentsProfWithinRangeROOT : public QCriterionRoot<TProfile>,
  public ContentsProfWithinRangeROOT
{ 
 public:
  MEContentsProfWithinRangeROOT(std::string name) :
  QCriterionRoot<TProfile>(name), ContentsProfWithinRangeROOT()
  {setAlgoName(ContentsProfWithinRangeROOT::getAlgoName());}
  ~MEContentsProfWithinRangeROOT(void){}
    
  std::string getAlgoName(void)
  {return ContentsProfWithinRangeROOT::getAlgoName();}
  
  float runTest(const TProfile * const t)
  {return ContentsProfWithinRangeROOT::runTest(t);}
  
  const TProfile * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TProfile>::getROOTObject(me);}
  
  bool notEnoughStats(const TProfile * const t) const
  {return ContentsProfWithinRangeROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {ContentsProfWithinRangeROOT::setMinimumEntries(N); update();}
 
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return ContentsProfWithinRangeROOT::getBadChannels();}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName()
            << "): Entry fraction within range = " << prob_;
    message_ = message.str();
  }

};

class MEContentsProf2DWithinRangeROOT : public QCriterionRoot<TProfile2D>,
  public ContentsProf2DWithinRangeROOT
{ 
 public:
  MEContentsProf2DWithinRangeROOT(std::string name) :
  QCriterionRoot<TProfile2D>(name), ContentsProf2DWithinRangeROOT()
  {setAlgoName(ContentsProf2DWithinRangeROOT::getAlgoName());}
  ~MEContentsProf2DWithinRangeROOT(void){}
    
  std::string getAlgoName(void)
  {return ContentsProf2DWithinRangeROOT::getAlgoName();}
  
  float runTest(const TProfile2D * const t)
  {return ContentsProf2DWithinRangeROOT::runTest(t);}
  
  const TProfile2D * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TProfile2D>::getROOTObject(me);}
  
  bool notEnoughStats(const TProfile2D * const t) const
  {return ContentsProf2DWithinRangeROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {ContentsProf2DWithinRangeROOT::setMinimumEntries(N); update();}
 
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return ContentsProf2DWithinRangeROOT::getBadChannels();}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName()
            << "): Entry fraction within range = " << prob_;
    message_ = message.str();
  }

};

class MEMeanWithinExpectedROOT: public QCriterionRoot<TH1F>, 
  public MeanWithinExpectedROOT
{
 public:
  MEMeanWithinExpectedROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), MeanWithinExpectedROOT() 
  {setAlgoName(MeanWithinExpectedROOT::getAlgoName());}
  ~MEMeanWithinExpectedROOT(void){}
  
  std::string getAlgoName(void)
  {return MeanWithinExpectedROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return MeanWithinExpectedROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return MeanWithinExpectedROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {MeanWithinExpectedROOT::setMinimumEntries(N); update();}
 
 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): "; 
    if(useRange_)
      {
	message << "Mean within allowed range? ";
	if(prob_) 
	  message << "Yes";
	else
	  message << "No";
      }
    else
      message << "prob = " << prob_;

    message_ = message.str();      
  }

};

class MEDeadChannelROOT: public QCriterionRoot<TH1F>, 
  public DeadChannelROOT
{
 public:
  MEDeadChannelROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), DeadChannelROOT() 
  {setAlgoName(DeadChannelROOT::getAlgoName());}
  ~MEDeadChannelROOT(void){}
  
  std::string getAlgoName(void)
  {return DeadChannelROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return DeadChannelROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return DeadChannelROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {DeadChannelROOT::setMinimumEntries(N); update();}

  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return DeadChannelROOT::getBadChannels();}
 
 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Alive channel fraction = " << prob_;
    message_ = message.str();
  }

};


class MENoisyChannelROOT: public QCriterionRoot<TH1F>, 
  public NoisyChannelROOT
{
 public:
  MENoisyChannelROOT(std::string name) : 
  QCriterionRoot<TH1F>(name), NoisyChannelROOT() 
  {setAlgoName(NoisyChannelROOT::getAlgoName());}
  ~MENoisyChannelROOT(void){}
  
  std::string getAlgoName(void)
  {return NoisyChannelROOT::getAlgoName();}

  float runTest(const TH1F * const t)
  {return NoisyChannelROOT::runTest(t);}

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return NoisyChannelROOT::notEnoughStats(t);}

  void setMinimumEntries(unsigned N)
  {NoisyChannelROOT::setMinimumEntries(N); update();}

  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return NoisyChannelROOT::getBadChannels();}

 protected:

  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Fraction of non-noisy channels = " << prob_;
    message_ = message.str();      
  }
};

class MEMostProbableLandauROOT: public QCriterionRoot<TH1F>,
                                public MostProbableLandauROOT {
  public:
    inline MEMostProbableLandauROOT( std::string oName):
      QCriterionRoot<TH1F>( oName),
      MostProbableLandauROOT() {
        setAlgoName( MostProbableLandauROOT::getAlgoName());
      }

    inline virtual std::string getAlgoName() { 
      return MostProbableLandauROOT::getAlgoName(); }

    inline virtual float runTest( const TH1F *const poPLOT) {
      return MostProbableLandauROOT::runTest( poPLOT); }

    inline virtual const TH1F * getObject(const MonitorElement *const poME) 
      const { return QCriterionRoot<TH1F>::getROOTObject( poME);}

    inline virtual bool notEnoughStats( const TH1F *const poPLOT) const {
      return MostProbableLandauROOT::notEnoughStats( poPLOT);}

    // Promote method to public
    inline virtual void setMinimumEntries(unsigned N)
    { MostProbableLandauROOT::setMinimumEntries(N); update();}

  protected:
    inline virtual bool isInvalid( const MonitorElement *const poME) { 
      return MostProbableLandauROOT::isInvalid(); }

    void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << getAlgoName() 
	      << "): Fraction of Most Probable value match = " << prob_;
      message_ = message.str();      
    }
};

#endif
