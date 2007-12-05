/** \file ROOTtoMEConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2007/12/02 03:49:27 $
 *  $Revision: 1.2 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/ROOTtoMEConverter.h"

ROOTtoMEConverter::ROOTtoMEConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0), count(0)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_ROOTtoMEConverter";
  
  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  outputfile = iPSet.getParameter<std::string>("Outputfile");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
 
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  if (dbe) {
    if (verbosity > 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  //if (dbe) {
  //  if (verbosity > 0 ) dbe->showDirStructure();
  //}
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Outputfile    = " << outputfile << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }

} // end constructor

ROOTtoMEConverter::~ROOTtoMEConverter() 
{
  if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
} // end destructor

void ROOTtoMEConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void ROOTtoMEConverter::endJob()
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " runs.";
  return;
}

void ROOTtoMEConverter::beginRun(const edm::Run& iRun, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_beginRun";
  
  // keep track of number of events processed
  ++count;
  
  int nrun = iRun.run();
  
  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || count == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count << " runs total)";
    }
  }
  
  return;
}

void ROOTtoMEConverter::endRun(const edm::Run& iRun, 
			       const edm::EventSetup& iSetup)
{
  
  std::string MsgLoggerCat = "ROOTtoMEConverter_endRun";
  
  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat)
      << "\nRestoring MonitorElements.";
  
  edm::Handle<MEtoROOT> metoroot;
  iRun.getByType(metoroot);

  if (!metoroot.isValid()) {
      edm::LogWarning(MsgLoggerCat)
	<< "Invalid MEtoROOT extracted from event.";
      return;
  }

  std::vector<MEtoROOT::MEROOTObject> merootobject = 
    metoroot->getMERootObject(); 

  me.resize(merootobject.size());

  TString classname;

  for (unsigned int i = 0; i < merootobject.size(); ++i) {

    me[i] = 0;

    // get full path of monitor element
    std::string pathname = merootobject[i].name;
    //std::cout << pathname << std::endl;

    std::string dir;

    // deconstruct path from fullpath
    StringList fulldir = StringOps::split(pathname,"/");
    for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
      dir += fulldir[j];
      if (j != fulldir.size() - 2) dir += "/";
    }
    //std::cout << dir << std::endl;    

    // get class of histogram object
    classname = merootobject[i].object.ClassName();   
    //std::cout << classname << std::endl;

    // define new monitor element
    if (dbe) {
      dbe->setCurrentFolder(dir);

      if (classname == "TH1F") {
	me[i] = dbe->book1D(merootobject[i].object.GetName(),
			    merootobject[i].object.GetTitle(),
			    merootobject[i].object.GetXaxis()->GetNbins(),
			    merootobject[i].object.GetXaxis()->GetXmin(),
			    merootobject[i].object.GetXaxis()->GetXmax());
	me[i]->setAxisTitle(merootobject[i].object.GetXaxis()->GetTitle(),1);
	me[i]->setAxisTitle(merootobject[i].object.GetYaxis()->GetTitle(),2);
	
	// fill new monitor element
	Int_t nbins = merootobject[i].object.GetXaxis()->GetNbins();
	for (Int_t x = 1; x <= nbins; ++x) {
	  Double_t binx = merootobject[i].object.GetBinCenter(x);
	  Double_t value = merootobject[i].object.GetBinContent(x);

	  me[i]->Fill(binx,value);
	} // end fill
      } // end TH1F monitor elements
    } // end define new monitor elements
  } // end loop thorugh merootobject

  return;
}

void ROOTtoMEConverter::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}
