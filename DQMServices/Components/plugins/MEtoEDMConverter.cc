
/** \file MEtoEDMConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2008/03/13 21:15:07 $
 *  $Revision: 1.7 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

using namespace lat;

MEtoEDMConverter::MEtoEDMConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_MEtoEDMConverter";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  
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
      << "===============================\n";
  }

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (dbe) {
    if (verbosity) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }
    
  // create persistent objects
  produces<MEtoEDM<TH1F>, edm::InRun>(fName);
  produces<MEtoEDM<TH2F>, edm::InRun>(fName);
  produces<MEtoEDM<TH3F>, edm::InRun>(fName);
  produces<MEtoEDM<TProfile>, edm::InRun>(fName);
  produces<MEtoEDM<TProfile2D>, edm::InRun>(fName);
  produces<MEtoEDM<double>, edm::InRun>(fName);
  produces<MEtoEDM<int>, edm::InRun>(fName);
  produces<MEtoEDM<TString>, edm::InRun>(fName);

  firstevent = true;

} // end constructor

MEtoEDMConverter::~MEtoEDMConverter() 
{
} // end destructor

void
MEtoEDMConverter::beginJob(const edm::EventSetup& iSetup)
{
}

void
MEtoEDMConverter::endJob(void)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_endJob";

  // information flags
  std::map<std::string,int> packages; // keep track just of package names
  unsigned nTH1F = 0; // count various objects we have
  unsigned nTH2F = 0;
  unsigned nTH3F = 0;
  unsigned nTProfile = 0;
  unsigned nTProfile2D = 0;
  unsigned nFloat = 0;
  unsigned nInt = 0;
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
      ++nInt;
      if (verbosity > 1)
	std::cout << "   scalar: " << tobj->GetName() << ": Int\n";
      break;

    case MonitorElement::DQM_KIND_REAL:
      ++nFloat;
      if (verbosity > 1)
	std::cout << "   scalar: " << tobj->GetName() << ": Float\n";
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

    case MonitorElement::DQM_KIND_TH2F:
      ++nTH2F;
      if (verbosity > 1)
	std::cout << "   normal: " << tobj->GetName() << ": TH2F\n";
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
    std::cout << "We have " << nTH2F << " TH2F objects" << std::endl;
    std::cout << "We have " << nTH3F << " TH3F objects" << std::endl;
    std::cout << "We have " << nTProfile << " TProfile objects" << std::endl;
    std::cout << "We have " << nTProfile2D << " TProfile2D objects" 
	      << std::endl;
    std::cout << "We have " << nFloat << " Float objects" << std::endl;
    std::cout << "We have " << nInt << " Int objects" << std::endl;
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
  std::vector<MonitorElement *> items(dbe->getAllContents(""));
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

    case MonitorElement::DQM_KIND_TH2F:
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

  mestorage<TH1F> TH1FME;
  mestorage<TH2F> TH2FME;
  mestorage<TH3F> TH3FME;
  mestorage<TProfile> TProfileME;
  mestorage<TProfile2D> TProfile2DME;
  mestorage<double> FloatME;
  mestorage<int> IntME;
  mestorage<TString> StringME;

  std::string MsgLoggerCat = "MEtoEDMConverter_endRun";
  
  if (verbosity)
    edm::LogInfo (MsgLoggerCat) << "\nStoring MEtoEDM dataformat histograms.";

  // extract ME information into vectors
  std::vector<MonitorElement *>::iterator mmi, mme;
  std::vector<MonitorElement *> items(dbe->getAllContents(""));
  for (mmi = items.begin (), mme = items.end (); mmi != mme; ++mmi) {
    MonitorElement *me = *mmi;

    // get monitor elements
    switch (me->kind())
    {
    case MonitorElement::DQM_KIND_INT:
      IntME.object.push_back(me->getIntValue());
      IntME.name.push_back(me->getFullname());
      IntME.tags.push_back(me->getTags());
      IntME.release.push_back(release);
      IntME.run.push_back(run);
      IntME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_REAL:
      FloatME.object.push_back(me->getFloatValue());
      FloatME.name.push_back(me->getFullname());
      FloatME.tags.push_back(me->getTags());
      FloatME.release.push_back(release);
      FloatME.run.push_back(run);
      FloatME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_STRING:
      StringME.object.push_back(me->getStringValue());
      StringME.name.push_back(me->getFullname());
      StringME.tags.push_back(me->getTags());
      StringME.release.push_back(release);
      StringME.run.push_back(run);
      StringME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_TH1F:
      TH1FME.object.push_back(*me->getTH1F());
      TH1FME.name.push_back(me->getFullname());
      TH1FME.tags.push_back(me->getTags());
      TH1FME.release.push_back(release);
      TH1FME.run.push_back(run);
      TH1FME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_TH2F:
      TH2FME.object.push_back(*me->getTH2F());
      TH2FME.name.push_back(me->getFullname());
      TH2FME.tags.push_back(me->getTags());
      TH2FME.release.push_back(release);
      TH2FME.run.push_back(run);
      TH2FME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_TH3F:
      TH3FME.object.push_back(*me->getTH3F());
      TH3FME.name.push_back(me->getFullname());
      TH3FME.tags.push_back(me->getTags());
      TH3FME.release.push_back(release);
      TH3FME.run.push_back(run);
      TH3FME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_TPROFILE:
      TProfileME.object.push_back(*me->getTProfile());
      TProfileME.name.push_back(me->getFullname());
      TProfileME.tags.push_back(me->getTags());
      TProfileME.release.push_back(release);
      TProfileME.run.push_back(run);
      TProfileME.datatier.push_back(datatier);
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D:
      TProfile2DME.object.push_back(*me->getTProfile2D());
      TProfile2DME.name.push_back(me->getFullname());
      TProfile2DME.tags.push_back(me->getTags());
      TProfile2DME.release.push_back(release);
      TProfile2DME.run.push_back(run);
      TProfile2DME.datatier.push_back(datatier);
      break;

    default:
      edm::LogError(MsgLoggerCat)
	<< "ERROR: The DQM object '" << me->getFullname()
	<< "' is neither a ROOT object nor a recognised "
	<< "simple object.\n";
      continue;
    }
  } // end loop through monitor elements

  // produce objects to put in events
  if (! TH1FME.object.empty()) {
    std::auto_ptr<MEtoEDM<TH1F> > pOut1(new MEtoEDM<TH1F>);
    pOut1->putMEtoEdmObject(TH1FME.name,TH1FME.tags,TH1FME.object,
			    TH1FME.release,TH1FME.run,TH1FME.datatier);
    iRun.put(pOut1,fName);
  }
  if (! TH2FME.object.empty()) {
    std::auto_ptr<MEtoEDM<TH2F> > pOut2(new MEtoEDM<TH2F>);
    pOut2->putMEtoEdmObject(TH2FME.name,TH2FME.tags,TH2FME.object,
			    TH2FME.release,TH2FME.run,TH2FME.datatier);
    iRun.put(pOut2,fName);
  }
  if (! TH3FME.object.empty()) {
    std::auto_ptr<MEtoEDM<TH3F> > pOut3(new MEtoEDM<TH3F>);
    pOut3->putMEtoEdmObject(TH3FME.name,TH3FME.tags,TH3FME.object,
			    TH3FME.release,TH3FME.run,TH3FME.datatier);
    iRun.put(pOut3,fName);
  }
  if (! TProfileME.object.empty()) {
    std::auto_ptr<MEtoEDM<TProfile> > pOut4(new MEtoEDM<TProfile>);
    pOut4->putMEtoEdmObject(TProfileME.name,TProfileME.tags,TProfileME.object,
			    TProfileME.release,TProfileME.run,
			    TProfileME.datatier);
    iRun.put(pOut4,fName);
  }
  if (! TProfile2DME.object.empty()) {
    std::auto_ptr<MEtoEDM<TProfile2D> > pOut5(new MEtoEDM<TProfile2D>);
    pOut5->putMEtoEdmObject(TProfile2DME.name,TProfile2DME.tags, 
			    TProfile2DME.object,TProfile2DME.release,
			    TProfile2DME.run,TProfile2DME.datatier);
    iRun.put(pOut5,fName);
  }
  if (! FloatME.object.empty()) {
    std::auto_ptr<MEtoEDM<double> > pOut6(new MEtoEDM<double>);
    pOut6->putMEtoEdmObject(FloatME.name,FloatME.tags,FloatME.object,
			    FloatME.release,FloatME.run,FloatME.datatier);
    iRun.put(pOut6,fName);
  }
  if (! IntME.object.empty()) {
    std::auto_ptr<MEtoEDM<int> > pOut7(new MEtoEDM<int>);
    pOut7->putMEtoEdmObject(IntME.name,IntME.tags,IntME.object,
			    IntME.release,IntME.run,IntME.datatier);
    iRun.put(pOut7,fName);
  }
  if (! StringME.object.empty()) {
    std::auto_ptr<MEtoEDM<TString> > 
      pOut8(new MEtoEDM<TString>);
    pOut8->putMEtoEdmObject(StringME.name,StringME.tags,StringME.object,
			    StringME.release,StringME.run,StringME.datatier);
    iRun.put(pOut8,fName);
  }
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
