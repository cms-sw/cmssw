

/** \file MEtoEDMConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2013/06/05 15:22:15 $
 *  $Revision: 1.35 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <cassert>

#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

using namespace lat;

MEtoEDMConverter::MEtoEDMConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0), deleteAfterCopy(false)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_MEtoEDMConverter";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name","MEtoEDMConverter");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity",0);
  frequency = iPSet.getUntrackedParameter<int>("Frequency",50);
  path = iPSet.getUntrackedParameter<std::string>("MEPathToSave");  
  deleteAfterCopy = iPSet.getUntrackedParameter<bool>("deleteAfterCopy",false);  
  
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    Path          = " << path << "\n"
      << "===============================\n";
  }

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  std::string sName;

  // create persistent objects

  sName = fName + "Run";
  produces<MEtoEDM<TH1F>, edm::InRun>(sName);
  produces<MEtoEDM<TH1S>, edm::InRun>(sName);
  produces<MEtoEDM<TH1D>, edm::InRun>(sName);
  produces<MEtoEDM<TH2F>, edm::InRun>(sName);
  produces<MEtoEDM<TH2S>, edm::InRun>(sName);
  produces<MEtoEDM<TH2D>, edm::InRun>(sName);
  produces<MEtoEDM<TH3F>, edm::InRun>(sName);
  produces<MEtoEDM<TProfile>, edm::InRun>(sName);
  produces<MEtoEDM<TProfile2D>, edm::InRun>(sName);
  produces<MEtoEDM<double>, edm::InRun>(sName);
  produces<MEtoEDM<long long>, edm::InRun>(sName);
  produces<MEtoEDM<TString>, edm::InRun>(sName);

  sName = fName + "Lumi";
  produces<MEtoEDM<TH1F>, edm::InLumi>(sName);
  produces<MEtoEDM<TH1S>, edm::InLumi>(sName);
  produces<MEtoEDM<TH1D>, edm::InLumi>(sName);
  produces<MEtoEDM<TH2F>, edm::InLumi>(sName);
  produces<MEtoEDM<TH2S>, edm::InLumi>(sName);
  produces<MEtoEDM<TH2D>, edm::InLumi>(sName);
  produces<MEtoEDM<TH3F>, edm::InLumi>(sName);
  produces<MEtoEDM<TProfile>, edm::InLumi>(sName);
  produces<MEtoEDM<TProfile2D>, edm::InLumi>(sName);
  produces<MEtoEDM<double>, edm::InLumi>(sName);
  produces<MEtoEDM<long long>, edm::InLumi>(sName);
  produces<MEtoEDM<TString>, edm::InLumi>(sName);

  iCount.clear();

  assert(sizeof(int64_t) == sizeof(long long));

}

MEtoEDMConverter::~MEtoEDMConverter() 
{
}

void
MEtoEDMConverter::beginJob()
{
}

void
MEtoEDMConverter::endJob(void)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_endJob";

  if (verbosity > 0) {

    // keep track just of package names
    std::map<std::string,int> packages;

    // count various objects we have
    unsigned nTH1F = 0;
    unsigned nTH1S = 0;
    unsigned nTH1D = 0;
    unsigned nTH2F = 0;
    unsigned nTH2S = 0;
    unsigned nTH2D = 0;
    unsigned nTH3F = 0;
    unsigned nTProfile = 0;
    unsigned nTProfile2D = 0;
    unsigned nDouble = 0;
    unsigned nInt64 = 0;
    unsigned nString = 0;

    if (verbosity > 1) std::cout << std::endl << "Summary :" << std::endl;

    // get contents out of DQM
    std::vector<MonitorElement *>::iterator mmi, mme;
    std::vector<MonitorElement *> items(dbe->getAllContents(""));

    for (mmi = items.begin (), mme = items.end (); mmi != mme; ++mmi) {

      // keep track of leading directory (i.e. package)
      StringList dir = StringOps::split((*mmi)->getPathname(),"/");
      ++packages[dir[0]];

      // check type
      if (verbosity > 1) std::cout << "MEobject:" << std::endl;
      MonitorElement *me = *mmi;
      TObject *tobj = me->getRootObject();
      switch (me->kind())
      {
      case MonitorElement::DQM_KIND_INT:
        ++nInt64;
        if (verbosity > 1)
	  std::cout << "   scalar: " << tobj->GetName() << ": Int64\n";
        break;

      case MonitorElement::DQM_KIND_REAL:
        ++nDouble;
        if (verbosity > 1)
	  std::cout << "   scalar: " << tobj->GetName() << ": Double\n";
        break;

      case MonitorElement::DQM_KIND_STRING:
        ++nString;
        if (verbosity > 1)
	  std::cout << "   scalar: " << tobj->GetName() << ": String\n";
        break;

      case MonitorElement::DQM_KIND_TH1F:
        ++nTH1F;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH1F\n";
        break;

      case MonitorElement::DQM_KIND_TH1S:
       ++nTH1S;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH1S\n";
        break;

      case MonitorElement::DQM_KIND_TH1D:
       ++nTH1D;
         if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH1D\n";
        break;

      case MonitorElement::DQM_KIND_TH2F:
        ++nTH2F;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH2F\n";
        break;

      case MonitorElement::DQM_KIND_TH2S:
        ++nTH2S;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH2S\n";
        break;

      case MonitorElement::DQM_KIND_TH2D:
        ++nTH2D;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH2D\n";
        break;

      case MonitorElement::DQM_KIND_TH3F:
        ++nTH3F;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TH3F\n";
        break;

      case MonitorElement::DQM_KIND_TPROFILE:
        ++nTProfile;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TProfile\n";
        break;

      case MonitorElement::DQM_KIND_TPROFILE2D:
        ++nTProfile2D;
        if (verbosity > 1)
	  std::cout << "   normal: " << tobj->GetName() << ": TProfile2D\n";
        break;

      default:
        edm::LogError(MsgLoggerCat)
	  << "ERROR: The DQM object '" << me->getFullname()
	  << "' is neither a ROOT object nor a recognised "
	  << "simple object.\n";
        continue;
      }
    } // end loop through monitor elements

    // list unique packages
    std::cout << "Packages accessing DQM:" << std::endl;
    std::map<std::string,int>::iterator pkgIter;
    for (pkgIter = packages.begin(); pkgIter != packages.end(); ++pkgIter) 
      std::cout << "  " << pkgIter->first << ": " << pkgIter->second 
		<< std::endl;

    std::cout << "We have " << nTH1F << " TH1F objects" << std::endl;
    std::cout << "We have " << nTH1S << " TH1S objects" << std::endl;
    std::cout << "We have " << nTH1D << " TH1D objects" << std::endl;
    std::cout << "We have " << nTH2F << " TH2F objects" << std::endl;
    std::cout << "We have " << nTH2S << " TH2S objects" << std::endl;
    std::cout << "We have " << nTH2D << " TH2D objects" << std::endl;
    std::cout << "We have " << nTH3F << " TH3F objects" << std::endl;
    std::cout << "We have " << nTProfile << " TProfile objects" << std::endl;
    std::cout << "We have " << nTProfile2D << " TProfile2D objects" << std::endl;
    std::cout << "We have " << nDouble << " Double objects" << std::endl;
    std::cout << "We have " << nInt64 << " Int64 objects" << std::endl;
    std::cout << "We have " << nString << " String objects" << std::endl;

    if (verbosity > 1) std::cout << std::endl;

  }

  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << iCount.size() << " runs.";

}

void
MEtoEDMConverter::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_beginRun";
    
  int nrun = iRun.run();
  
  // keep track of number of runs processed
  ++iCount[nrun];

  if (verbosity > 0) {  // keep track of number of runs processed
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << iCount.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || iCount.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << iCount.size() << " runs total)";
    }
  }

  // clear contents of monitor elements
  std::vector<MonitorElement *>::iterator mmi, mme;
  std::vector<MonitorElement *> items(dbe->getAllContents(path));

  for (mmi = items.begin (), mme = items.end (); mmi != mme; ++mmi) {

    MonitorElement *me = *mmi;

    switch (me->kind())
    {
    case MonitorElement::DQM_KIND_INT:
      break;

    case MonitorElement::DQM_KIND_REAL:
      break;

    case MonitorElement::DQM_KIND_STRING:
      break;

    case MonitorElement::DQM_KIND_TH1F:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH1S:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH1D:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH2F:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH2S:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH2D:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TH3F:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TPROFILE:
      me->Reset();
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D:
      me->Reset();
      break;

    default:
      edm::LogError(MsgLoggerCat)
	<< "ERROR: The DQM object '" << me->getFullname()
	<< "' is neither a ROOT object nor a recognised "
	<< "simple object.\n";
      continue;
    }

  } // end loop through monitor elements
}
void
MEtoEDMConverter::endRun(edm::Run const& iRun, const edm::EventSetup& iSetup)
{
}

void
MEtoEDMConverter::endRunProduce(edm::Run& iRun, const edm::EventSetup& iSetup)
{
  dbe->scaleElements();
  putData(iRun, false);
}

void
MEtoEDMConverter::endLuminosityBlockProduce(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup)
{
  putData(iLumi, true);
}

template <class T>
void
MEtoEDMConverter::putData(T& iPutTo, bool iLumiOnly)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_putData";
  
  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat) << "\nStoring MEtoEDM dataformat histograms.";

  // extract ME information into vectors
  std::vector<MonitorElement *>::iterator mmi, mme;
  std::vector<MonitorElement *> items(dbe->getAllContents(path));

  unsigned int n1F=0;
  unsigned int n1S=0;
  unsigned int n1D=0;
  unsigned int n2F=0;
  unsigned int n2S=0;
  unsigned int n2D=0;
  unsigned int n3F=0;
  unsigned int nProf=0;
  unsigned int nProf2=0;
  unsigned int nDouble=0;
  unsigned int nInt64=0;
  unsigned int nString=0;

  for (mmi = items.begin (), mme = items.end (); mmi != mme; ++mmi) {

    MonitorElement *me = *mmi;

    // store only flagged ME at endLumi transition, and Run-based
    // histo at endRun transition
    if (iLumiOnly && !me->getLumiFlag()) continue;
    if (!iLumiOnly && me->getLumiFlag()) continue;

    switch (me->kind())
    {
    case MonitorElement::DQM_KIND_INT:
      ++nInt64;
      break;

    case MonitorElement::DQM_KIND_REAL:
      ++nDouble;
      break;

    case MonitorElement::DQM_KIND_STRING:
      ++nString;
      break;

    case MonitorElement::DQM_KIND_TH1F:
      ++n1F;
      break;

    case MonitorElement::DQM_KIND_TH1S:
      ++n1S;
      break;

    case MonitorElement::DQM_KIND_TH1D:
      ++n1D;
      break;

    case MonitorElement::DQM_KIND_TH2F:
      ++n2F;
      break;

    case MonitorElement::DQM_KIND_TH2S:
      ++n2S;
      break;

    case MonitorElement::DQM_KIND_TH2D:
      ++n2D;
      break;

    case MonitorElement::DQM_KIND_TH3F:
      ++n3F;
      break;

    case MonitorElement::DQM_KIND_TPROFILE:
      ++nProf;
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D:
      ++nProf2;
      break;

    default:
      edm::LogError(MsgLoggerCat)
	<< "ERROR: The DQM object '" << me->getFullname()
	<< "' is neither a ROOT object nor a recognised "
	<< "simple object.\n";
      continue;
    }
  }

  std::auto_ptr<MEtoEDM<long long> > pOutInt(new MEtoEDM<long long>(nInt64));
  std::auto_ptr<MEtoEDM<double> > pOutDouble(new MEtoEDM<double>(nDouble));
  std::auto_ptr<MEtoEDM<TString> > pOutString(new MEtoEDM<TString>(nString));
  std::auto_ptr<MEtoEDM<TH1F> > pOut1(new MEtoEDM<TH1F>(n1F));
  std::auto_ptr<MEtoEDM<TH1S> > pOut1s(new MEtoEDM<TH1S>(n1S));
  std::auto_ptr<MEtoEDM<TH1D> > pOut1d(new MEtoEDM<TH1D>(n1D));
  std::auto_ptr<MEtoEDM<TH2F> > pOut2(new MEtoEDM<TH2F>(n2F));
  std::auto_ptr<MEtoEDM<TH2S> > pOut2s(new MEtoEDM<TH2S>(n2S));
  std::auto_ptr<MEtoEDM<TH2D> > pOut2d(new MEtoEDM<TH2D>(n2D));
  std::auto_ptr<MEtoEDM<TH3F> > pOut3(new MEtoEDM<TH3F>(n3F));
  std::auto_ptr<MEtoEDM<TProfile> > pOutProf(new MEtoEDM<TProfile>(nProf));
  std::auto_ptr<MEtoEDM<TProfile2D> > pOutProf2(new MEtoEDM<TProfile2D>(nProf2));

  for (mmi = items.begin (), mme = items.end (); mmi != mme; ++mmi) {

    MonitorElement *me = *mmi;

    // store only flagged ME at endLumi transition, and Run-based
    // histo at endRun transition
    if (iLumiOnly && !me->getLumiFlag()) continue;
    if (!iLumiOnly && me->getLumiFlag()) continue;

    // get monitor elements
    switch (me->kind())
    {
    case MonitorElement::DQM_KIND_INT:
      pOutInt->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getIntValue());
      break;

    case MonitorElement::DQM_KIND_REAL:
      pOutDouble->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getFloatValue());
      break;

    case MonitorElement::DQM_KIND_STRING:
      pOutString->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getStringValue());
      break;

    case MonitorElement::DQM_KIND_TH1F:
      pOut1->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1F());
      break;

    case MonitorElement::DQM_KIND_TH1S:
      pOut1s->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1S());
      break;

    case MonitorElement::DQM_KIND_TH1D:
      pOut1d->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1D());
      break;

    case MonitorElement::DQM_KIND_TH2F:
      pOut2->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2F());
      break;

    case MonitorElement::DQM_KIND_TH2S:
      pOut2s->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2S());
      break;

    case MonitorElement::DQM_KIND_TH2D:
      pOut2d->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2D());
      break;

    case MonitorElement::DQM_KIND_TH3F:
      pOut3->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH3F());
      break;

    case MonitorElement::DQM_KIND_TPROFILE:
      pOutProf->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTProfile());
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D:
      pOutProf2->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTProfile2D());
      break;

    default:
      edm::LogError(MsgLoggerCat)
	<< "ERROR: The DQM object '" << me->getFullname()
	<< "' is neither a ROOT object nor a recognised "
	<< "simple object.\n";
      continue;
    }

    if (!iLumiOnly) {
      // remove ME after copy to EDM is done.
      if (deleteAfterCopy)
        dbe->removeElement(me->getPathname(),me->getName());
    }

  } // end loop through monitor elements

  std::string sName;

  if (iLumiOnly) {
    sName = fName + "Lumi";
  } else {
    sName = fName + "Run";
  }

  // produce objects to put in events
  iPutTo.put(pOutInt,sName);
  iPutTo.put(pOutDouble,sName);
  iPutTo.put(pOutString,sName);
  iPutTo.put(pOut1,sName);
  iPutTo.put(pOut1s,sName);
  iPutTo.put(pOut1d,sName);
  iPutTo.put(pOut2,sName);
  iPutTo.put(pOut2s,sName);
  iPutTo.put(pOut2d,sName);
  iPutTo.put(pOut3,sName);
  iPutTo.put(pOutProf,sName);
  iPutTo.put(pOutProf2,sName);

}

void
MEtoEDMConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

}
