
/** \file MEtoEDMConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2009/10/13 07:45:31 $
 *  $Revision: 1.26 $
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
    
  // create persistent objects
  produces<MEtoEDM<TH1F>, edm::InRun>(fName);
  produces<MEtoEDM<TH1S>, edm::InRun>(fName);
  produces<MEtoEDM<TH1D>, edm::InRun>(fName);
  produces<MEtoEDM<TH2F>, edm::InRun>(fName);
  produces<MEtoEDM<TH2S>, edm::InRun>(fName);
  produces<MEtoEDM<TH2D>, edm::InRun>(fName);
  produces<MEtoEDM<TH3F>, edm::InRun>(fName);
  produces<MEtoEDM<TProfile>, edm::InRun>(fName);
  produces<MEtoEDM<TProfile2D>, edm::InRun>(fName);
  produces<MEtoEDM<double>, edm::InRun>(fName);
  produces<MEtoEDM<long long>, edm::InRun>(fName);
  produces<MEtoEDM<TString>, edm::InRun>(fName);

  firstevent = true;

  assert(sizeof(int64_t) == sizeof(long long));

} // end constructor

MEtoEDMConverter::~MEtoEDMConverter() 
{
} // end destructor

void
MEtoEDMConverter::beginJob()
{
}

void
MEtoEDMConverter::endJob(void)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_endJob";

  // information flags
  std::map<std::string,int> packages; // keep track just of package names
  unsigned nTH1F = 0; // count various objects we have
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

  if (verbosity) {
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
    std::cout << "We have " << nTProfile2D << " TProfile2D objects" 
	      << std::endl;
    std::cout << "We have " << nDouble << " Double objects" << std::endl;
    std::cout << "We have " << nInt64 << " Int64 objects" << std::endl;
    std::cout << "We have " << nString << " String objects" << std::endl;
  }

  if (verbosity > 1) std::cout << std::endl;

  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count.size() << " runs.";

  return;
}

void
MEtoEDMConverter::beginRun(edm::Run& iRun, const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_beginRun";
    
  int nrun = iRun.run();
  
  // keep track of number of runs processed
  ++count[nrun];

  if (verbosity) {  // keep track of number of runs processed
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || count.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count.size() << " runs total)";
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
MEtoEDMConverter::endRun(edm::Run& iRun, const edm::EventSetup& iSetup)
{

  int run = iRun.run();
  std::string release = edm::getReleaseVersion();

  std::string MsgLoggerCat = "MEtoEDMConverter_endRun";
  
  if (verbosity)
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

    // get monitor elements
    switch (me->kind())
    {
    case MonitorElement::DQM_KIND_INT:
      pOutInt->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getIntValue(),
				release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_REAL:
      pOutDouble->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getFloatValue(),
				  release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_STRING:
      pOutString->putMEtoEdmObject(me->getFullname(),me->getTags(),me->getStringValue(),
				   release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH1F:
      pOut1->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1F(),
			      release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH1S:
      pOut1s->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1S(),
			       release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH1D:
      pOut1d->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH1D(),
			       release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH2F:
      pOut2->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2F(),
			      release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH2S:
      pOut2s->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2S(),
			       release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH2D:
      pOut2d->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH2D(),
			       release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TH3F:
      pOut3->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTH3F(),
			      release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TPROFILE:
      pOutProf->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTProfile(),
			      release,run,datatier);
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D:
      pOutProf2->putMEtoEdmObject(me->getFullname(),me->getTags(),*me->getTProfile2D(),
				  release,run,datatier);
      break;

    default:
      edm::LogError(MsgLoggerCat)
	<< "ERROR: The DQM object '" << me->getFullname()
	<< "' is neither a ROOT object nor a recognised "
	<< "simple object.\n";
      continue;
    }
    
    // remove ME after copy to EDM is done.
    if (deleteAfterCopy)
      dbe->removeElement(me->getPathname(),me->getName());
    
  } // end loop through monitor elements

  // produce objects to put in events
  iRun.put(pOutInt,fName);
  iRun.put(pOutDouble,fName);
  iRun.put(pOutString,fName);
  iRun.put(pOut1,fName);
  iRun.put(pOut1s,fName);
  iRun.put(pOut2,fName);
  iRun.put(pOut2s,fName);
  iRun.put(pOut3,fName);
  iRun.put(pOutProf,fName);
  iRun.put(pOutProf2,fName);

}

void
MEtoEDMConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (firstevent) {
    if (iEvent.isRealData()) {
      datatier = "DATA";
    } else {
      datatier = "MC";
    }
  }

}
